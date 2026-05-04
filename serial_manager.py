"""
Serial port manager for ZDT_X42S motor communication.
Handles port enumeration, connection lifecycle, and background read/write threads.
"""

import threading
import queue
import time
from typing import Optional

try:
    import serial
    import serial.tools.list_ports
    from serial.serialutil import SerialException
    HAS_PYSERIAL = True
except ImportError:
    HAS_PYSERIAL = False
    serial = None
    SerialException = Exception

from zdt_protocol import (
    parse_response, parse_host_command, detect_direction, Direction,
    validate_checksum, COMMAND_REGISTRY, ChecksumType,
    decode_motor_status, decode_homing_status,
)

# ── Known command lengths from the protocol ─────────────────────────────────
# Heuristic: pre-compute all possible frame lengths from the registry
# to speed up frame boundary detection.

def _build_known_lengths():
    """Build a set of all possible frame lengths from the command registry."""
    lengths = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21}
    for (fw, code), cmd in COMMAND_REGISTRY.items():
        # Host command lengths
        host_len = 2  # addr + code
        for p in cmd.params:
            end = p.offset + p.length
            host_len = max(host_len, 2 + end)
        host_len += 1  # checksum
        lengths.add(host_len)

        # Response lengths
        if cmd.is_status_response:
            lengths.add(4)  # addr + code + status + checksum
        if cmd.response_data_len > 0:
            lengths.add(3 + cmd.response_data_len + 1)  # +checksum
    return sorted(lengths)

KNOWN_FRAME_LENGTHS = _build_known_lengths()


class SerialManager:
    """Manages serial port connection and background I/O threads."""

    def __init__(self, socketio=None):
        self.socketio = socketio
        self.ser: Optional[serial.Serial] = None
        self.write_queue = queue.Queue()
        self.read_thread: Optional[threading.Thread] = None
        self.write_thread: Optional[threading.Thread] = None
        self.running = False
        self.port = ""
        self.baudrate = 115200
        self.cs_type = ChecksumType.FIXED_6B
        self.firmware = "emm"
        self.emulator = None  # set by app.py when emulator is active
        self.motor_states: dict[int, dict] = {}

    # ── Port enumeration ──────────────────────────────────────────────────

    @staticmethod
    def list_ports() -> list:
        """Return list of available serial ports."""
        if not HAS_PYSERIAL:
            return []
        ports = []
        for p in serial.tools.list_ports.comports():
            ports.append({
                "port": p.device,
                "description": p.description,
                "hwid": p.hwid,
            })
        return ports

    # ── Connection management ──────────────────────────────────────────────

    def connect(self, port: str, baudrate: int) -> bool:
        """Open serial connection and start I/O threads.

        Uses a timeout thread because serial.Serial() can hang indefinitely
        on Windows (especially for Bluetooth pseudo-COM ports).
        """
        if not HAS_PYSERIAL:
            self._emit_status("pyserial 未安装")
            return False

        # Disconnect any existing connection first
        if self.ser and self.ser.is_open:
            self.disconnect()
            time.sleep(0.3)

        # Attempt connection in a background thread with a timeout.
        # On Windows, opening a Bluetooth COM port can block forever.
        result_holder = {"ser": None, "error": None}

        def _try_open():
            try:
                s = serial.Serial(
                    port=port,
                    baudrate=baudrate,
                    timeout=0.01,
                    write_timeout=1.0,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                )
                result_holder["ser"] = s
            except Exception as e:
                result_holder["error"] = e

        open_thread = threading.Thread(target=_try_open, daemon=True)
        open_thread.start()
        open_thread.join(timeout=3.0)  # 3-second timeout for port open

        if open_thread.is_alive():
            # Port open is still blocking — the port is unresponsive
            # We cannot kill the thread, but we can report failure
            self._emit_status(f"无法打开 {port}: 端口无响应 (可能是蓝牙虚拟串口)")
            return False

        if result_holder["error"]:
            e = result_holder["error"]
            err_msg = str(e)
            if isinstance(e, SerialException):
                if "Access is denied" in err_msg or "拒绝访问" in err_msg:
                    self._emit_status(f"无法打开 {port}: 端口被其他程序占用")
                elif "does not exist" in err_msg or "系统找不到" in err_msg:
                    self._emit_status(f"无法打开 {port}: 端口不存在")
                else:
                    self._emit_status(f"无法打开 {port}: {err_msg}")
            else:
                self._emit_status(f"无法打开 {port}: {err_msg}")
            return False

        self.ser = result_holder["ser"]

        # Verify port is actually open
        if not self.ser or not self.ser.is_open:
            self._emit_status(f"无法打开 {port}: 未知错误")
            self.ser = None
            return False

        self.port = port
        self.baudrate = baudrate
        self.running = True

        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.read_thread.start()

        self.write_thread = threading.Thread(target=self._write_loop, daemon=True)
        self.write_thread.start()

        self._emit_status(f"已连接 {port} @ {baudrate} baud")
        return True

    def disconnect(self):
        """Close serial connection and stop threads."""
        self.running = False
        # Wait briefly for threads to notice
        time.sleep(0.05)
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None
        self._emit_status("已断开连接")

    @property
    def is_connected(self) -> bool:
        return self.ser is not None and self.ser.is_open

    # ── Sending data ──────────────────────────────────────────────────────

    def send(self, data: bytes):
        """Enqueue data for sending over serial."""
        self.write_queue.put(data)

    def send_direct(self, data: bytes):
        """Send data directly (for emulator mode or immediate sends)."""
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(data)
            except Exception as e:
                self._emit("serial_error", {"message": str(e)})

    def _update_motor_state(self, addr: int, code: int, decoded_params: dict, firmware: str):
        """Accumulate motor state from parsed response parameters."""
        if addr == 0 or not decoded_params:
            return
        if addr not in self.motor_states:
            self.motor_states[addr] = {"addr": addr, "firmware": firmware, "last_seen": time.time()}

        state = self.motor_states[addr]
        state["last_seen"] = time.time()
        if firmware:
            state["firmware"] = firmware

        # 0x43 system status: nearly complete state
        if code == 0x43:
            for key in ("bus_voltage_mv", "bus_current_ma", "phase_current_ma",
                        "speed_rpm", "position_deg", "target_position_deg",
                        "position_error_deg", "temperature_c",
                        "encoder_raw", "calibrated_encoder",
                        "homing_flags", "motor_flags"):
                if key in decoded_params:
                    state[key] = decoded_params[key]

            # Expand flag bitfields for UI convenience
            if "motor_flags" in decoded_params:
                state["motor_flags_detail"] = decode_motor_status(decoded_params["motor_flags"])
            if "homing_flags" in decoded_params:
                state["homing_flags_detail"] = decode_homing_status(decoded_params["homing_flags"])

        # Individual read commands — accumulate single parameters
        elif code == 0x36:  # Read real-time position
            if "position" in decoded_params:
                state["position_deg"] = decoded_params["position"]
        elif code == 0x35:  # Read real-time speed
            if "speed" in decoded_params:
                state["speed_rpm"] = decoded_params["speed"]
        elif code == 0x27:  # Read phase current
            if "phase_current" in decoded_params:
                state["phase_current_ma"] = decoded_params["phase_current"]
        elif code == 0x24:  # Read bus voltage
            if "bus_voltage" in decoded_params:
                state["bus_voltage_mv"] = decoded_params["bus_voltage"]
        elif code == 0x39:  # Read temperature
            if "temperature" in decoded_params:
                state["temperature_c"] = decoded_params["temperature"]
        elif code == 0x33:  # Read target position
            if "target_position" in decoded_params:
                state["target_position_deg"] = decoded_params["target_position"]
        elif code == 0x37:  # Read position error
            if "error" in decoded_params:
                state["position_error_deg"] = decoded_params["error"]
        elif code == 0x31:  # Read calibrated encoder
            if "encoder_value" in decoded_params:
                state["calibrated_encoder"] = decoded_params["encoder_value"]
        elif code == 0x3A:  # Read motor status flags
            if "status_flags" in decoded_params:
                state["motor_flags"] = decoded_params["status_flags"]
                if "flags_detail" in decoded_params:
                    state["motor_flags_detail"] = decoded_params["flags_detail"]
        elif code == 0x3B:  # Read homing flags
            if "homing_flags" in decoded_params:
                state["homing_flags"] = decoded_params["homing_flags"]
                if "flags_detail" in decoded_params:
                    state["homing_flags_detail"] = decoded_params["flags_detail"]

    # ── Background threads ────────────────────────────────────────────────

    def _read_loop(self):
        """Continuously read from serial port, detect frames, emit parsed data."""
        buffer = bytearray()
        while self.running:
            try:
                if self.ser and self.ser.is_open:
                    available = self.ser.in_waiting
                    if available > 0:
                        chunk = self.ser.read(min(available, 256))
                        buffer.extend(chunk)
                        self._process_buffer(buffer)
                    else:
                        time.sleep(0.001)
                else:
                    time.sleep(0.01)
            except Exception as e:
                if self.running:
                    self._emit("serial_error", {"message": str(e)})
                time.sleep(0.1)

    def _process_buffer(self, buffer: bytearray):
        """Try to extract complete frames from the buffer."""
        max_attempts = 10  # avoid infinite loop on malformed data
        attempts = 0

        while len(buffer) >= 3 and attempts < max_attempts:
            frame_info = self._try_extract_frame(buffer)
            if frame_info:
                frame_bytes = frame_info["bytes"]
                buffer[:] = frame_info["remaining"]

                parsed = parse_response(frame_bytes, self.firmware, self.cs_type)
                direction = detect_direction(frame_bytes)

                # Update accumulated motor state
                self._update_motor_state(parsed.addr, parsed.code, parsed.decoded_params, parsed.firmware)

                self._emit("serial_data", {
                    "hex": frame_bytes.hex(' ').upper(),
                    "direction": direction.value,
                    "direction_label": ">>>" if direction == Direction.HOST_TO_MOTOR else "<<<",
                    "source": "received",
                    "parsed": {
                        "addr": parsed.addr,
                        "code": parsed.code,
                        "status": parsed.status,
                        "status_text": parsed.status_text,
                        "decoded_params": parsed.decoded_params,
                        "human_readable": parsed.human_readable,
                        "firmware": parsed.firmware,
                    },
                    "emulated": False,
                })
                # Push full motor state updates to sidebar
                self._emit("motor_states_update", {"states": list(self.motor_states.values())})
                attempts = 0  # reset on successful extraction
            else:
                # No valid frame found at current position
                # Try advancing one byte
                buffer[:] = buffer[1:]
                attempts += 1

        # If buffer gets too large, trim it
        if len(buffer) > 512:
            buffer[:] = buffer[-256:]

    def _try_extract_frame(self, buffer: bytearray) -> Optional[dict]:
        """Try to extract a valid frame from the beginning of the buffer.

        Returns dict with 'bytes' and 'remaining', or None if no valid frame found.
        """
        buf = bytes(buffer)

        # Try known lengths first (fast path)
        for flen in KNOWN_FRAME_LENGTHS:
            if len(buf) >= flen:
                candidate = buf[:flen]
                if validate_checksum(candidate, ChecksumType.FIXED_6B):
                    # Additional validation: addr in range, code reasonable
                    if candidate[0] <= 255 and candidate[1] <= 0xFF:
                        return {"bytes": candidate, "remaining": buf[flen:]}

        # Fallback: scan for checksum-valid frames of any reasonable length
        for flen in range(3, min(len(buf), 40)):
            candidate = buf[:flen]
            if validate_checksum(candidate, ChecksumType.FIXED_6B):
                if flen in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 33, 37]:
                    return {"bytes": candidate, "remaining": buf[flen:]}

        return None

    def _write_loop(self):
        """Dequeue and write data to serial."""
        while self.running:
            try:
                data = self.write_queue.get(timeout=0.1)
                if self.ser and self.ser.is_open:
                    self.ser.write(data)
                    self.ser.flush()
                    # Emit the sent data to monitor
                    parsed = parse_host_command(data, self.firmware)
                    self._emit("serial_data", {
                        "hex": data.hex(' ').upper(),
                        "direction": "host_to_motor",
                        "direction_label": ">>>",
                        "source": "sent",
                        "parsed": {
                            "addr": parsed.addr,
                            "code": parsed.code,
                            "status": parsed.status,
                            "status_text": parsed.status_text,
                            "decoded_params": parsed.decoded_params,
                            "human_readable": parsed.human_readable,
                            "firmware": parsed.firmware,
                        },
                        "emulated": False,
                    })
            except queue.Empty:
                pass
            except Exception as e:
                if self.running:
                    self._emit("serial_error", {"message": str(e)})

    # ── Helpers ────────────────────────────────────────────────────────────

    def _emit(self, event: str, data: dict):
        """Emit a SocketIO event if socketio is available."""
        if self.socketio:
            try:
                self.socketio.emit(event, data, namespace='/')
            except Exception:
                pass

    def _emit_status(self, status: str):
        """Emit connection status change."""
        self._emit("serial_status", {
            "connected": self.is_connected,
            "status": status,
            "port": self.port,
            "baudrate": self.baudrate,
        })
