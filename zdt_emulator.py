"""
Motor emulation state machine for ZDT_X42S stepper motors.
Simulates N motors on a virtual bus, each maintaining position, speed,
current, status flags, etc. Responds to host commands with realistic responses.
"""

import struct
import time
import math
from dataclasses import dataclass, field
from typing import Optional

from zdt_protocol import ChecksumType, calc_checksum


@dataclass
class MotorEmulatorState:
    """Full state of a single emulated motor."""
    addr: int
    firmware: str = "emm"  # "emm" or "x"

    # Position and motion
    position_deg: float = 0.0
    speed_rpm: float = 0.0
    target_position_deg: float = 0.0
    target_speed_rpm: float = 0.0
    accel_rpm_s: float = 0.0
    decel_rpm_s: float = 0.0

    # Motion state
    moving: bool = False
    move_direction: int = 0  # 0=CW, 1=CCW
    move_type: str = ""  # "speed", "position", "torque"

    # Electrical
    phase_current_ma: float = 200.0
    bus_voltage_mv: float = 24000.0
    bus_current_ma: float = 150.0

    # Temperature
    temperature_c: float = 32.0

    # Encoder (single-turn absolute, 0-65535)
    encoder_raw: int = 0

    # Status flags (bitfield — see manual 5.5.15)
    # bit0: Ens_TF (enabled), bit1: Prf_TF (position reached)
    # bit2: Cgi_TF (stall), bit3: Cgp_TF (stall protection)
    # bit7: Oac_TF (power-loss)
    status_flags: int = 0b10000001  # Oac_TF=1, Ens_TF=1

    # Homing flags (bitfield — see manual 5.4.4)
    # bit0: Enc_Rdy, bit1: Cal_Rdy, bit2: Org_SF, bit3: Org_CF
    homing_flags: int = 0b00000011  # Enc_Rdy=1, Cal_Rdy=1

    # IO flags (bitfield — see manual 5.5.17)
    io_flags: int = 0b00010001  # En_Pin=1, Dir_Pin=1

    # Config params (defaults)
    microstep: int = 16
    max_current_ma: int = 3000
    open_loop_current_ma: int = 1200
    pos_window_deg: float = 0.8
    control_mode: int = 1  # 0=open, 1=closed
    motor_type: int = 0x19  # 0x19=1.8deg, 0x32=0.9deg
    direction_mode: int = 0  # 0=CW, 1=CCW

    # Version info
    fw_version: int = 200  # V2.0.0
    hw_series: int = 0    # X series
    hw_type: int = 3      # 42
    hw_version: int = 14  # V2.0

    # Phase R/L
    phase_resistance: int = 1500  # mOhm
    phase_inductance: int = 2200  # uH

    # Homing params
    homing_mode: int = 0
    homing_dir: int = 0
    homing_speed: int = 30
    homing_timeout: int = 10000
    collision_detect_speed: int = 300
    collision_detect_current: int = 800
    collision_detect_time: int = 60
    auto_homing: int = 0

    # Thresholds
    otp_threshold: int = 100
    ocp_threshold: int = 6600
    otp_ocp_time: int = 1000

    # Heartbeat
    heartbeat_time: int = 0

    # Integral limit / rigidity
    limit_rigidity: int = 388 if firmware == "x" else 65535

    # Collision return angle
    return_angle: int = 0

    # Enabled state
    enabled: bool = True

    # Timestamp for motion calc
    _last_tick: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "addr": self.addr,
            "firmware": self.firmware,
            "position_deg": round(self.position_deg, 2),
            "speed_rpm": round(self.speed_rpm, 2),
            "target_position_deg": round(self.target_position_deg, 2),
            "target_speed_rpm": round(self.target_speed_rpm, 2),
            "phase_current_ma": round(self.phase_current_ma, 1),
            "bus_voltage_mv": round(self.bus_voltage_mv, 1),
            "temperature_c": round(self.temperature_c, 1),
            "moving": self.moving,
            "enabled": self.enabled,
            "status_flags": self.status_flags,
            "homing_flags": self.homing_flags,
            "status_detail": self._status_detail(),
            "homing_detail": self._homing_detail(),
        }

    def _status_detail(self) -> dict:
        return {
            "Ens_TF": bool(self.status_flags & 0x01),
            "Prf_TF": bool(self.status_flags & 0x02),
            "Cgi_TF": bool(self.status_flags & 0x04),
            "Cgp_TF": bool(self.status_flags & 0x08),
            "Esi_LF": bool(self.status_flags & 0x10),
            "Esi_RF": bool(self.status_flags & 0x20),
            "Oac_TF": bool(self.status_flags & 0x80),
        }

    def _homing_detail(self) -> dict:
        return {
            "Enc_Rdy": bool(self.homing_flags & 0x01),
            "Cal_Rdy": bool(self.homing_flags & 0x02),
            "Org_SF":  bool(self.homing_flags & 0x04),
            "Org_CF":  bool(self.homing_flags & 0x08),
            "Otp_TF":  bool(self.homing_flags & 0x10),
            "Ocp_TF":  bool(self.homing_flags & 0x20),
        }


class EmulatorManager:
    """Manages multiple emulated motors on a virtual bus."""

    def __init__(self):
        self.motors: dict[int, MotorEmulatorState] = {}
        self.running = False

    def add_motor(self, addr: int, firmware: str = "emm"):
        self.motors[addr] = MotorEmulatorState(addr=addr, firmware=firmware)

    def remove_motor(self, addr: int):
        self.motors.pop(addr, None)

    def clear(self):
        self.motors.clear()

    def start(self, motor_configs: list):
        """Initialize emulator with configured motors.
        motor_configs: [{"addr": 1, "firmware": "emm"}, ...]
        """
        self.clear()
        for cfg in motor_configs:
            self.add_motor(cfg["addr"], cfg.get("firmware", "emm"))
        self.running = True

    def stop(self):
        self.running = False
        self.clear()

    def tick(self):
        """Advance all motors by one time step."""
        now = time.time()
        for motor in self.motors.values():
            dt = now - motor._last_tick
            motor._last_tick = now
            if dt > 0.5:  # skip large gaps
                continue
            if motor.moving:
                self._update_motion(motor, dt)

    def process_command(self, raw: bytes, cs_type: ChecksumType = ChecksumType.FIXED_6B) -> Optional[bytes]:
        """Route a raw command to the appropriate motor(s).
        Returns response bytes or None (for broadcast with no response).
        """
        if len(raw) < 2:
            return None

        addr = raw[0]
        code = raw[1]

        # Multi-motor command (0xAA) with broadcast addr
        if addr == 0x00 and code == 0xAA:
            return self._handle_multi_motor(raw, cs_type)

        # Broadcast: route to all motors, no response
        if addr == 0x00:
            for motor in self.motors.values():
                self._handle(motor, code, raw[2:-1])
            return None

        # Single motor
        motor = self.motors.get(addr)
        if not motor:
            return self._error_response(addr, code, 0xEE, cs_type)

        return self._handle(motor, code, raw[2:-1], cs_type)

    def _handle(self, motor: MotorEmulatorState, code: int,
                data: bytes, cs_type: ChecksumType = ChecksumType.FIXED_6B) -> Optional[bytes]:
        """Dispatch command to the appropriate handler."""
        handlers = {
            0x06: self._h_encoder_cal,
            0x08: self._h_restart,
            0x0A: self._h_clear_position,
            0x0E: self._h_release_protection,
            0x0F: self._h_factory_reset,
            0xF3: self._h_enable,
            0xF5: self._h_torque_mode,
            0xC5: self._h_torque_mode_limit,
            0xF6: self._h_speed_mode,
            0xC6: self._h_speed_mode_limit,
            0xFB: self._h_position_direct,
            0xCB: self._h_position_direct_limit,
            0xFD: self._h_position_trapezoid,
            0xCD: self._h_position_trapezoid_limit,
            0xFE: self._h_stop,
            0xFF: self._h_sync_trigger,
            0x9A: self._h_trigger_homing,
            0x9C: self._h_stop_homing,
            0x93: self._h_set_homing_zero,

            # Reads
            0x1F: self._h_read_version,
            0x20: self._h_read_rl,
            0x24: self._h_read_bus_voltage,
            0x26: self._h_read_bus_current,
            0x27: self._h_read_phase_current,
            0x31: self._h_read_encoder,
            0x32: self._h_read_pulse_count,
            0x33: self._h_read_target_position,
            0x34: self._h_read_set_position,
            0x35: self._h_read_speed,
            0x36: self._h_read_position,
            0x37: self._h_read_position_error,
            0x39: self._h_read_temperature,
            0x3A: self._h_read_status_flags,
            0x3B: self._h_read_homing_flags,
            0x3C: self._h_read_both_flags,
            0x3D: self._h_read_io_flags,
            0x1A: self._h_read_option_flags,
            0x22: self._h_read_homing_params,
            0x41: self._h_read_pos_window,
            0x13: self._h_read_thresholds,
            0x16: self._h_read_heartbeat,
            0x23: self._h_read_rigidity,
            0x3F: self._h_read_return_angle,
            0x15: self._h_read_id_broadcast,
            0x21: self._h_read_pid,
            0x43: self._h_read_all_status,
            0x42: self._h_read_driver_config,

            # Config writes
            0xAE: self._h_modify_id,
            0x84: self._h_modify_microstep,
            0x44: self._h_modify_open_current,
            0x45: self._h_modify_closed_current,
            0x46: self._h_modify_control_mode,
            0xD5: self._h_modify_fw_type,
            0xD7: self._h_modify_motor_type,
            0xD4: self._h_modify_direction,
            0x50: self._h_modify_powerloss_flag,
            0x4F: self._h_modify_scale,
            0xD1: self._h_modify_pos_window,
        }

        handler = handlers.get(code)
        if handler:
            return handler(motor, data, cs_type)
        return self._error_response(motor.addr, code, 0xEE, cs_type)

    # ── Motion update ────────────────────────────────────────────────────

    def _update_motion(self, m: MotorEmulatorState, dt: float):
        if not m.moving or not m.enabled:
            return

        target_speed = m.target_speed_rpm
        accel = m.accel_rpm_s if m.accel_rpm_s > 0 else 1000.0

        # Accelerate toward target speed
        if m.speed_rpm < target_speed:
            m.speed_rpm = min(m.speed_rpm + accel * dt, target_speed)
        elif m.speed_rpm > target_speed:
            m.speed_rpm = max(m.speed_rpm - accel * dt, target_speed)

        # Compute position delta
        delta_deg = m.speed_rpm * dt / 60.0 * 360.0
        if m.move_direction == 1:  # CCW
            delta_deg = -delta_deg

        m.position_deg += delta_deg

        # Update encoder (single-turn)
        m.encoder_raw = int((m.position_deg % 360.0) / 360.0 * 65535) & 0xFFFF

        # Simulated current (idle + proportional to speed)
        m.phase_current_ma = 200.0 + abs(m.speed_rpm) * 1.5

        # Position mode: check if target reached
        if m.move_type == "position":
            remaining = m.target_position_deg - m.position_deg
            # Anti-overshoot: slow down near target
            if abs(remaining) < (m.speed_rpm ** 2) / (2 * max(accel, 1)):
                m.speed_rpm = max(0.0, m.speed_rpm - accel * dt)

            if abs(remaining) <= m.pos_window_deg:
                m.position_deg = m.target_position_deg
                m.speed_rpm = 0.0
                m.moving = False
                m.status_flags |= 0x02  # Prf_TF = 1
                m.status_flags &= ~0x04  # clear stall

        # Simulate temperature rise
        m.temperature_c += abs(m.speed_rpm) * 0.001 * dt
        if not m.moving:
            m.temperature_c = max(32.0, m.temperature_c - 0.1 * dt)

        # Simulate bus current proportional to phase current
        m.bus_current_ma = m.phase_current_ma * 0.7 + 50.0

    # ── Command handlers ──────────────────────────────────────────────────

    def _ok_response(self, addr: int, code: int, cs_type: ChecksumType) -> bytes:
        frame = bytes([addr, code, 0x02])
        return frame + bytes([calc_checksum(frame, cs_type)])

    def _error_response(self, addr: int, code: int, error: int, cs_type: ChecksumType) -> bytes:
        frame = bytes([addr, code, error])
        return frame + bytes([calc_checksum(frame, cs_type)])

    def _sign_byte(self, val: float) -> int:
        return 0x01 if val < 0 else 0x00

    def _pos_to_emm(self, deg: float) -> tuple:
        """Convert degrees to Emm format (0-65535 per rev, with multi-turn)."""
        sign = self._sign_byte(deg)
        abs_deg = abs(deg)
        revs = int(abs_deg // 360)
        remainder = abs_deg % 360
        raw = revs * 65536 + int(remainder / 360.0 * 65536)
        return sign, raw

    def _pos_to_x(self, deg: float) -> tuple:
        """Convert degrees to X format (raw = deg * 10)."""
        sign = self._sign_byte(deg)
        raw = int(abs(deg) * 10)
        return sign, raw

    def _parse_direction(self, data: bytes, offset: int = 0) -> tuple:
        """Parse direction byte: returns (dir, next_offset)."""
        d = data[offset] if offset < len(data) else 0
        return d, offset + 1

    def _parse_u16(self, data: bytes, offset: int) -> tuple:
        if offset + 2 > len(data):
            return 0, offset + 2
        return int.from_bytes(data[offset:offset+2], 'big'), offset + 2

    def _parse_u32(self, data: bytes, offset: int) -> tuple:
        if offset + 4 > len(data):
            return 0, offset + 4
        return int.from_bytes(data[offset:offset+4], 'big'), offset + 4

    # ── Trigger handlers ──────────────────────────────────────────────────

    def _h_encoder_cal(self, m, data, cs):
        m.homing_flags |= 0x03  # Enc_Rdy + Cal_Rdy
        return self._ok_response(m.addr, 0x06, cs)

    def _h_restart(self, m, data, cs):
        m.speed_rpm = 0
        m.moving = False
        return self._ok_response(m.addr, 0x08, cs)

    def _h_clear_position(self, m, data, cs):
        m.position_deg = 0.0
        m.target_position_deg = 0.0
        m.encoder_raw = 0
        return self._ok_response(m.addr, 0x0A, cs)

    def _h_release_protection(self, m, data, cs):
        m.status_flags &= ~0x08  # clear Cgp_TF
        m.homing_flags &= ~0x30  # clear Otp_TF, Ocp_TF
        return self._ok_response(m.addr, 0x0E, cs)

    def _h_factory_reset(self, m, data, cs):
        return self._ok_response(m.addr, 0x0F, cs)

    # ── Motion handlers ───────────────────────────────────────────────────

    def _h_enable(self, m, data, cs):
        if len(data) < 2:
            return self._error_response(m.addr, 0xF3, 0xE2, cs)
        enable = data[1] if len(data) > 1 else data[0]
        m.enabled = (enable == 0x01)
        if m.enabled:
            m.status_flags |= 0x01
        else:
            m.status_flags &= ~0x01
            m.speed_rpm = 0
            m.moving = False
        return self._ok_response(m.addr, 0xF3, cs)

    def _h_torque_mode(self, m, data, cs):
        if len(data) < 5:
            return self._error_response(m.addr, 0xF5, 0xE2, cs)
        m.move_direction = data[0]
        accel, _ = self._parse_u16(data, 1)
        current, _ = self._parse_u16(data, 3)
        m.accel_rpm_s = 500.0  # torque mode doesn't have RPM accel
        m.target_speed_rpm = min(current / 10.0, 500.0)
        m.phase_current_ma = float(current)
        m.move_type = "torque"
        m.moving = True
        m.status_flags &= ~0x02  # clear Prf_TF
        return self._ok_response(m.addr, 0xF5, cs)

    def _h_torque_mode_limit(self, m, data, cs):
        if len(data) < 7:
            return self._error_response(m.addr, 0xC5, 0xE2, cs)
        m.move_direction = data[0]
        accel, _ = self._parse_u16(data, 1)
        current, _ = self._parse_u16(data, 3)
        max_speed, _ = self._parse_u16(data, 6)
        m.accel_rpm_s = accel / 100.0
        m.target_speed_rpm = max_speed * 0.1
        m.phase_current_ma = float(current)
        m.move_type = "torque"
        m.moving = True
        m.status_flags &= ~0x02
        return self._ok_response(m.addr, 0xC5, cs)

    def _h_speed_mode(self, m, data, cs):
        if m.firmware == "x":
            return self._h_speed_mode_x(m, data, cs)
        else:
            return self._h_speed_mode_emm(m, data, cs)

    def _h_speed_mode_x(self, m, data, cs):
        if len(data) < 5:
            return self._error_response(m.addr, 0xF6, 0xE2, cs)
        m.move_direction = data[0]
        accel, _ = self._parse_u16(data, 1)
        speed, _ = self._parse_u16(data, 3)
        m.accel_rpm_s = float(accel)
        m.target_speed_rpm = speed * 0.1
        m.move_type = "speed"
        m.moving = True
        m.status_flags &= ~0x02
        return self._ok_response(m.addr, 0xF6, cs)

    def _h_speed_mode_emm(self, m, data, cs):
        if len(data) < 4:
            return self._error_response(m.addr, 0xF6, 0xE2, cs)
        m.move_direction = data[0]
        speed, _ = self._parse_u16(data, 1)
        accel_gear = data[3]
        m.target_speed_rpm = float(speed)
        m.accel_rpm_s = 500.0 if accel_gear > 0 else 3000.0
        m.move_type = "speed"
        m.moving = True
        m.status_flags &= ~0x02
        return self._ok_response(m.addr, 0xF6, cs)

    def _h_speed_mode_limit(self, m, data, cs):
        if len(data) < 7:
            return self._error_response(m.addr, 0xC6, 0xE2, cs)
        m.move_direction = data[0]
        accel, _ = self._parse_u16(data, 1)
        speed, _ = self._parse_u16(data, 3)
        max_current, _ = self._parse_u16(data, 6)
        m.accel_rpm_s = float(accel)
        m.target_speed_rpm = speed * 0.1
        m.max_current_ma = max_current
        m.move_type = "speed"
        m.moving = True
        m.status_flags &= ~0x02
        return self._ok_response(m.addr, 0xC6, cs)

    def _h_position_direct(self, m, data, cs):
        return self._h_position_direct_limit(m, data, cs, code=0xFB)

    def _h_position_direct_limit(self, m, data, cs, code=0xCB):
        if len(data) < 8:
            return self._error_response(m.addr, code, 0xE2, cs)
        m.move_direction = data[0]
        speed, _ = self._parse_u16(data, 1)
        pos_raw, _ = self._parse_u32(data, 3)
        move_mode = data[7]
        m.target_speed_rpm = speed * 0.1
        m.accel_rpm_s = 2000.0  # default fast accel for direct mode

        # Position input is in 0.1 deg units for X firmware
        target = pos_raw * 0.1
        if move_mode == 0x00:  # relative to prev target
            m.target_position_deg += target if m.move_direction == 0 else -target
        elif move_mode == 0x01:  # absolute
            m.target_position_deg = target if m.move_direction == 0 else -target
        else:  # relative to current position
            m.target_position_deg = m.position_deg + (target if m.move_direction == 0 else -target)

        m.move_type = "position"
        m.moving = True
        m.status_flags &= ~0x02
        return self._ok_response(m.addr, code, cs)

    def _h_position_trapezoid(self, m, data, cs):
        return self._h_position_trapezoid_limit(m, data, cs, code=0xFD)

    def _h_position_trapezoid_limit(self, m, data, cs, code=0xCD):
        if m.firmware == "emm" and code == 0xFD:
            return self._h_position_emm(m, data, cs)

        if len(data) < 12:
            return self._error_response(m.addr, code, 0xE2, cs)
        m.move_direction = data[0]
        accel_acc, _ = self._parse_u16(data, 1)
        decel_acc, _ = self._parse_u16(data, 3)
        max_speed, _ = self._parse_u16(data, 5)
        pos_raw, _ = self._parse_u32(data, 7)
        move_mode = data[11]

        m.accel_rpm_s = float(accel_acc)
        m.decel_rpm_s = float(decel_acc)
        m.target_speed_rpm = max_speed * 0.1

        target = pos_raw * 0.1
        if move_mode == 0x00:
            m.target_position_deg += target if m.move_direction == 0 else -target
        elif move_mode == 0x01:
            m.target_position_deg = target if m.move_direction == 0 else -target
        else:
            m.target_position_deg = m.position_deg + (target if m.move_direction == 0 else -target)

        m.move_type = "position"
        m.moving = True
        m.status_flags &= ~0x02
        return self._ok_response(m.addr, code, cs)

    def _h_position_emm(self, m, data, cs):
        if len(data) < 9:
            return self._error_response(m.addr, 0xFD, 0xE2, cs)
        m.move_direction = data[0]
        speed, _ = self._parse_u16(data, 1)
        accel_gear = data[3]
        pulses, _ = self._parse_u32(data, 4)
        move_mode = data[8]

        m.target_speed_rpm = float(speed)
        m.accel_rpm_s = 500.0 if accel_gear > 0 else 3000.0

        # Emm: 3200 pulses = 360 degrees (at default 16 microstep)
        deg = (pulses / 3200.0) * 360.0
        if move_mode == 0x00:
            m.target_position_deg += deg if m.move_direction == 0 else -deg
        elif move_mode == 0x01:
            m.target_position_deg = deg if m.move_direction == 0 else -deg
        else:
            m.target_position_deg = m.position_deg + (deg if m.move_direction == 0 else -deg)

        m.move_type = "position"
        m.moving = True
        m.status_flags &= ~0x02
        return self._ok_response(m.addr, 0xFD, cs)

    def _h_stop(self, m, data, cs):
        m.speed_rpm = 0.0
        m.moving = False
        m.move_type = ""
        return self._ok_response(m.addr, 0xFE, cs)

    def _h_sync_trigger(self, m, data, cs):
        # In emulation, cached commands are executed immediately anyway
        return self._ok_response(m.addr, 0xFF, cs)

    # ── Homing handlers ──────────────────────────────────────────────────

    def _h_set_homing_zero(self, m, data, cs):
        return self._ok_response(m.addr, 0x93, cs)

    def _h_trigger_homing(self, m, data, cs):
        if len(data) < 1:
            return self._error_response(m.addr, 0x9A, 0xE2, cs)
        mode = data[0]
        m.homing_mode = mode
        m.homing_flags |= 0x04  # Org_SF = 1 (homing in progress)
        # Simulate instant homing completion
        m.homing_flags &= ~0x0C  # clear Org_SF and Org_CF
        m.position_deg = 0.0  # go to zero
        return self._ok_response(m.addr, 0x9A, cs)

    def _h_stop_homing(self, m, data, cs):
        m.homing_flags &= ~0x0C
        return self._ok_response(m.addr, 0x9C, cs)

    # ── Read handlers ────────────────────────────────────────────────────

    def _build_response(self, m, code: int, data: bytes, cs_type) -> bytes:
        frame = bytes([m.addr, code]) + data
        return frame + bytes([calc_checksum(frame, cs_type)])

    def _h_read_version(self, m, data, cs):
        return self._build_response(m, 0x1F,
            bytes([m.fw_version, m.hw_series, m.hw_type, m.hw_version]), cs)

    def _h_read_rl(self, m, data, cs):
        return self._build_response(m, 0x20,
            struct.pack('>HH', m.phase_resistance, m.phase_inductance), cs)

    def _h_read_bus_voltage(self, m, data, cs):
        return self._build_response(m, 0x24,
            struct.pack('>H', int(m.bus_voltage_mv)), cs)

    def _h_read_bus_current(self, m, data, cs):
        return self._build_response(m, 0x26,
            struct.pack('>H', int(m.bus_current_ma)), cs)

    def _h_read_phase_current(self, m, data, cs):
        return self._build_response(m, 0x27,
            struct.pack('>H', int(m.phase_current_ma)), cs)

    def _h_read_encoder(self, m, data, cs):
        return self._build_response(m, 0x31,
            struct.pack('>H', m.encoder_raw), cs)

    def _h_read_pulse_count(self, m, data, cs):
        # Simulated: derive from position at 16 microstep
        pulses = int(abs(m.position_deg) / 360.0 * 3200)
        sign = self._sign_byte(m.position_deg)
        return self._build_response(m, 0x32,
            bytes([sign]) + struct.pack('>I', pulses), cs)

    def _h_read_target_position(self, m, data, cs):
        if m.firmware == "emm":
            sign, raw = self._pos_to_emm(m.target_position_deg)
        else:
            sign, raw = self._pos_to_x(m.target_position_deg)
        return self._build_response(m, 0x33,
            bytes([sign]) + struct.pack('>I', raw), cs)

    def _h_read_set_position(self, m, data, cs):
        if m.firmware == "emm":
            sign, raw = self._pos_to_emm(m.target_position_deg)
        else:
            sign, raw = self._pos_to_x(m.target_position_deg)
        return self._build_response(m, 0x34,
            bytes([sign]) + struct.pack('>I', raw), cs)

    def _h_read_speed(self, m, data, cs):
        sign = self._sign_byte(m.speed_rpm)
        if m.firmware == "emm":
            raw = int(abs(m.speed_rpm))
        else:
            raw = int(abs(m.speed_rpm) * 10)  # X: 0.1 RPM units
        return self._build_response(m, 0x35,
            bytes([sign]) + struct.pack('>H', raw), cs)

    def _h_read_position(self, m, data, cs):
        if m.firmware == "emm":
            sign, raw = self._pos_to_emm(m.position_deg)
        else:
            sign, raw = self._pos_to_x(m.position_deg)
        return self._build_response(m, 0x36,
            bytes([sign]) + struct.pack('>I', raw), cs)

    def _h_read_position_error(self, m, data, cs):
        error = abs(m.target_position_deg - m.position_deg)
        sign = self._sign_byte(m.target_position_deg - m.position_deg)
        if m.firmware == "emm":
            # Emm: 0-65535 per rev
            revs = int(error // 360)
            rem = error % 360
            raw = revs * 65536 + int(rem / 360.0 * 65536)
        else:
            raw = int(error * 100)  # X: 0.01 deg units
        return self._build_response(m, 0x37,
            bytes([sign]) + struct.pack('>I', raw), cs)

    def _h_read_temperature(self, m, data, cs):
        temp = int(m.temperature_c)
        return self._build_response(m, 0x39,
            bytes([0x01, temp]), cs)

    def _h_read_status_flags(self, m, data, cs):
        return self._build_response(m, 0x3A,
            bytes([m.status_flags]), cs)

    def _h_read_homing_flags(self, m, data, cs):
        return self._build_response(m, 0x3B,
            bytes([m.homing_flags]), cs)

    def _h_read_both_flags(self, m, data, cs):
        return self._build_response(m, 0x3C,
            bytes([m.homing_flags, m.status_flags]), cs)

    def _h_read_io_flags(self, m, data, cs):
        return self._build_response(m, 0x3D,
            bytes([m.io_flags]), cs)

    def _h_read_option_flags(self, m, data, cs):
        flags = 0
        if m.motor_type == 0x32:
            flags |= 0x01
        if m.firmware == "emm":
            flags |= 0x02
        if m.control_mode == 1:
            flags |= 0x04
        if m.direction_mode == 1:
            flags |= 0x10
        return self._build_response(m, 0x1A, bytes([flags]), cs)

    def _h_read_homing_params(self, m, data, cs):
        d = bytes([m.homing_mode, m.homing_dir])
        d += struct.pack('>H', m.homing_speed)
        d += struct.pack('>I', m.homing_timeout)
        d += struct.pack('>H', m.collision_detect_speed)
        d += struct.pack('>H', m.collision_detect_current)
        d += struct.pack('>H', m.collision_detect_time)
        d += bytes([m.auto_homing])
        return self._build_response(m, 0x22, d, cs)

    def _h_read_pos_window(self, m, data, cs):
        return self._build_response(m, 0x41,
            struct.pack('>H', int(m.pos_window_deg * 10)), cs)

    def _h_read_thresholds(self, m, data, cs):
        d = struct.pack('>H', m.otp_threshold)
        d += struct.pack('>H', m.ocp_threshold)
        d += struct.pack('>H', m.otp_ocp_time)
        return self._build_response(m, 0x13, d, cs)

    def _h_read_heartbeat(self, m, data, cs):
        return self._build_response(m, 0x16,
            struct.pack('>I', m.heartbeat_time), cs)

    def _h_read_rigidity(self, m, data, cs):
        return self._build_response(m, 0x23,
            struct.pack('>I', m.limit_rigidity), cs)

    def _h_read_return_angle(self, m, data, cs):
        return self._build_response(m, 0x3F,
            struct.pack('>H', m.return_angle), cs)

    def _h_read_id_broadcast(self, m, data, cs):
        return self._build_response(m, 0x15, bytes([m.addr]), cs)

    def _h_read_pid(self, m, data, cs):
        if m.firmware == "x":
            # X firmware PID: 4 x 4-byte values
            d = struct.pack('>IIII', 126640, 126640, 15600, 26)
        else:
            # Emm firmware PID: 3 x 4-byte values
            d = struct.pack('>III', 18000, 10, 18000)
        return self._build_response(m, 0x21, d, cs)

    def _h_read_all_status(self, m, data, cs):
        if m.firmware == "x":
            # X firmware: 37 bytes of data
            d = bytes([0x25, 0x0C])  # byte count=37, param count=12
            d += struct.pack('>H', int(m.bus_voltage_mv))
            d += struct.pack('>H', int(m.bus_current_ma))
            d += struct.pack('>H', int(m.phase_current_ma))
            d += struct.pack('>H', m.encoder_raw)  # raw encoder
            d += struct.pack('>H', m.encoder_raw)  # calibrated encoder
            sign_pos, pos_raw = self._pos_to_x(m.target_position_deg)
            d += bytes([sign_pos]) + struct.pack('>I', pos_raw)
            sign_spd = self._sign_byte(m.speed_rpm)
            spd_raw = int(abs(m.speed_rpm) * 10)
            d += bytes([sign_spd]) + struct.pack('>H', spd_raw)
            sign_real, real_raw = self._pos_to_x(m.position_deg)
            d += bytes([sign_real]) + struct.pack('>I', real_raw)
            sign_err = self._sign_byte(m.target_position_deg - m.position_deg)
            err_raw = int(abs(m.target_position_deg - m.position_deg) * 100)
            d += bytes([sign_err]) + struct.pack('>I', err_raw)
            d += bytes([0x00])  # temp sign (positive)
            d += bytes([int(m.temperature_c)])
            d += bytes([m.homing_flags, m.status_flags])
        else:
            # Emm firmware: 31 bytes of data
            d = bytes([0x1F, 0x09])
            d += struct.pack('>H', int(m.bus_voltage_mv))
            d += struct.pack('>H', int(m.phase_current_ma))
            d += struct.pack('>H', m.encoder_raw)
            sign_pos, pos_raw = self._pos_to_emm(m.target_position_deg)
            d += bytes([sign_pos]) + struct.pack('>I', pos_raw)
            sign_spd = self._sign_byte(m.speed_rpm)
            d += bytes([sign_spd]) + struct.pack('>H', int(abs(m.speed_rpm)))
            sign_real, real_raw = self._pos_to_emm(m.position_deg)
            d += bytes([sign_real]) + struct.pack('>I', real_raw)
            sign_err = self._sign_byte(m.target_position_deg - m.position_deg)
            err_abs = abs(m.target_position_deg - m.position_deg)
            revs = int(err_abs // 360)
            rem = err_abs % 360
            err_raw = revs * 65536 + int(rem / 360.0 * 65536)
            d += bytes([sign_err]) + struct.pack('>I', err_raw)
            d += bytes([m.homing_flags, m.status_flags])
        return self._build_response(m, 0x43, d, cs)

    def _h_read_driver_config(self, m, data, cs):
        # Simplified - return a plausible config block
        if m.firmware == "x":
            d = bytes([0x25, 0x18])  # X: 37 bytes, 24 params
            d += bytes([0x00, 0x01, 0x01, 0x02])  # lock, ctrl, pulse, serial
            d += bytes([0x02, 0x00, 0x10])  # En, Dir, microstep
            d += bytes([0x01, 0x00])  # interp, auto off
            d += bytes([0x00, 0x00])  # reserved
            d += struct.pack('>H', m.open_loop_current_ma)
            d += struct.pack('>H', m.max_current_ma)
            d += struct.pack('>H', 3000)  # max speed
            d += struct.pack('>H', 1000)  # current loop bandwidth
            d += bytes([0x05, 0x07, 0x00])  # uart baud, can rate, checksum
            d += bytes([0x01, 0x00, 0x01])  # response, scale, stall protect
            d += struct.pack('>H', 8)  # stall speed
            d += struct.pack('>H', 2200)  # stall current
            d += struct.pack('>H', 2000)  # stall time
            d += struct.pack('>H', int(m.pos_window_deg * 10))
        else:
            d = bytes([0x21, 0x15])  # Emm: 33 bytes, 21 params
            d += bytes([m.motor_type, 0x01, 0x02, 0x02, 0x00, 0x10, 0x01])
            d += bytes([0x00])
            d += struct.pack('>H', m.open_loop_current_ma)
            d += struct.pack('>H', m.max_current_ma)
            d += struct.pack('>H', 4000)  # max output voltage
            d += bytes([0x05, 0x07])
            d += bytes([m.addr, 0x00, 0x01, 0x01])
            d += struct.pack('>H', 8)
            d += struct.pack('>H', 2200)
            d += struct.pack('>H', 2000)
            d += struct.pack('>H', int(m.pos_window_deg * 10))
        return self._build_response(m, 0x42, d, cs)

    # ── Config write handlers ─────────────────────────────────────────────

    def _h_modify_id(self, m, data, cs):
        if len(data) >= 3:
            m.addr = data[2]
        return self._ok_response(m.addr if len(data) >= 3 else m.addr, 0xAE, cs)

    def _h_modify_microstep(self, m, data, cs):
        if len(data) >= 3:
            m.microstep = data[2]
        return self._ok_response(m.addr, 0x84, cs)

    def _h_modify_open_current(self, m, data, cs):
        if len(data) >= 4:
            m.open_loop_current_ma, _ = self._parse_u16(data, 2)
        return self._ok_response(m.addr, 0x44, cs)

    def _h_modify_closed_current(self, m, data, cs):
        if len(data) >= 4:
            m.max_current_ma, _ = self._parse_u16(data, 2)
        return self._ok_response(m.addr, 0x45, cs)

    def _h_modify_control_mode(self, m, data, cs):
        if len(data) >= 3:
            m.control_mode = data[2]
        return self._ok_response(m.addr, 0x46, cs)

    def _h_modify_fw_type(self, m, data, cs):
        if len(data) >= 3:
            fw = data[2]
            if fw == 0x00:
                m.firmware = "x"
            else:
                m.firmware = "emm"
        return self._ok_response(m.addr, 0xD5, cs)

    def _h_modify_motor_type(self, m, data, cs):
        if len(data) >= 3:
            m.motor_type = data[2]
        return self._ok_response(m.addr, 0xD7, cs)

    def _h_modify_direction(self, m, data, cs):
        if len(data) >= 3:
            m.direction_mode = data[2]
        return self._ok_response(m.addr, 0xD4, cs)

    def _h_modify_powerloss_flag(self, m, data, cs):
        if len(data) >= 1:
            if data[0] == 0x00:
                m.status_flags &= ~0x80
        return self._ok_response(m.addr, 0x50, cs)

    def _h_modify_scale(self, m, data, cs):
        return self._ok_response(m.addr, 0x4F, cs)

    def _h_modify_pos_window(self, m, data, cs):
        if len(data) >= 4:
            window, _ = self._parse_u16(data, 2)
            m.pos_window_deg = window * 0.1
        return self._ok_response(m.addr, 0xD1, cs)

    # ── Multi-motor command (0xAA) ─────────────────────────────────────

    def _handle_multi_motor(self, raw: bytes, cs_type) -> Optional[bytes]:
        """Parse and route multi-motor command (0xAA)."""
        if len(raw) < 5:
            return None

        # raw format: [00][AA][len_hi][len_lo][cmd1...cmd_n][checksum]
        data_len = int.from_bytes(raw[2:4], 'big')
        sub_commands = raw[4:-1]  # between length and checksum

        # Parse sub-commands: each is a complete motor command
        offset = 0
        responses = []
        while offset < len(sub_commands):
            if offset + 3 > len(sub_commands):
                break
            sub_addr = sub_commands[offset]
            sub_code = sub_commands[offset + 1]

            # Determine sub-command length
            sub_len = self._sub_command_length(sub_code, sub_commands[offset:])
            if sub_len == 0:
                break

            sub_cmd = bytes([sub_addr]) + sub_commands[offset + 1:offset + sub_len]
            resp = self.process_command(sub_cmd, cs_type)
            if resp:
                responses.append(resp)

            offset += sub_len

        # Per protocol: only addr 1 responds to avoid bus collision
        for r in responses:
            if r[0] == 0x01:
                return r
        return None

    def _sub_command_length(self, code: int, data: bytes) -> int:
        """Determine length of a sub-command within multi-motor command."""
        # Simple commands: addr + code + checksum = 3
        simple_codes = {0x1F, 0x20, 0x24, 0x26, 0x27, 0x31, 0x32, 0x33, 0x34,
                        0x35, 0x36, 0x37, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x1A,
                        0x22, 0x41, 0x13, 0x16, 0x23, 0x3F, 0x21, 0x15}
        if code in simple_codes:
            return 3  # addr(1) + code(1) + checksum(1)

        # Look up from registry
        for (fw, c), cmd in COMMAND_REGISTRY.items():
            if c == code and cmd.params:
                host_len = 2  # addr + code
                for p in cmd.params:
                    host_len = max(host_len, 2 + p.offset + p.length)
                return host_len + 1  # +checksum

        # Fallback heuristics based on known code lengths
        known = {
            0xF3: 5, 0xF5: 8, 0xC5: 10,
            0xF6: 8, 0xC6: 10,
            0xFB: 11, 0xCB: 13,
            0xFD: 15, 0xCD: 17,
            0xFE: 4, 0xFF: 4,
            0x9A: 4, 0x9C: 3, 0x93: 4,
            0xAE: 5, 0x84: 5, 0x44: 6, 0x45: 6,
            0x46: 5, 0xD5: 5, 0xD7: 5, 0xD4: 5,
            0x50: 4, 0x4F: 5, 0xD1: 6,
        }
        return known.get(code, 3)


def create_default_emulator(num_motors: int = 3) -> EmulatorManager:
    """Create an emulator with a default set of motors for testing."""
    mgr = EmulatorManager()
    configs = []
    for i in range(1, num_motors + 1):
        fw = "emm" if i % 2 == 1 else "x"
        configs.append({"addr": i, "firmware": fw})
    mgr.start(configs)
    return mgr
