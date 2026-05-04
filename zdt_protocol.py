"""
ZDT_X42S stepper motor communication protocol.
Data-driven command registry: build and parse all 60+ commands.
Supports both Emm and X firmware variants.
"""

import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable

# ── CRC-8 lookup table (from manual appendix 8.1) ──────────────────────────

CRC8_TABLE = [
    0x00, 0x5E, 0xBC, 0xE2, 0x61, 0x3F, 0xDD, 0x83,
    0xC2, 0x9C, 0x7E, 0x20, 0xA3, 0xFD, 0x1F, 0x41,
    0x9D, 0xC3, 0x21, 0x7F, 0xFC, 0xA2, 0x40, 0x1E,
    0x5F, 0x01, 0xE3, 0xBD, 0x3E, 0x60, 0x82, 0xDC,
    0x23, 0x7D, 0x9F, 0xC1, 0x42, 0x1C, 0xFE, 0xA0,
    0xE1, 0xBF, 0x5D, 0x03, 0x80, 0xDE, 0x3C, 0x62,
    0xBE, 0xE0, 0x02, 0x5C, 0xDF, 0x81, 0x63, 0x3D,
    0x7C, 0x22, 0xC0, 0x9E, 0x1D, 0x43, 0xA1, 0xFF,
    0x46, 0x18, 0xFA, 0xA4, 0x27, 0x79, 0x9B, 0xC5,
    0x84, 0xDA, 0x38, 0x66, 0xE5, 0xBB, 0x59, 0x07,
    0xDB, 0x85, 0x67, 0x39, 0xBA, 0xE4, 0x06, 0x58,
    0x19, 0x47, 0xA5, 0xFB, 0x78, 0x26, 0xC4, 0x9A,
    0x65, 0x3B, 0xD9, 0x87, 0x04, 0x5A, 0xB8, 0xE6,
    0xA7, 0xF9, 0x1B, 0x45, 0xC6, 0x98, 0x7A, 0x24,
    0xF8, 0xA6, 0x44, 0x1A, 0x99, 0xC7, 0x25, 0x7B,
    0x3A, 0x64, 0x86, 0xD8, 0x5B, 0x05, 0xE7, 0xB9,
    0x8C, 0xD2, 0x30, 0x6E, 0xED, 0xB3, 0x51, 0x0F,
    0x4E, 0x10, 0xF2, 0xAC, 0x2F, 0x71, 0x93, 0xCD,
    0x11, 0x4F, 0xAD, 0xF3, 0x70, 0x2E, 0xCC, 0x92,
    0xD3, 0x8D, 0x6F, 0x31, 0xB2, 0xEC, 0x0E, 0x50,
    0xAF, 0xF1, 0x13, 0x4D, 0xCE, 0x90, 0x72, 0x2C,
    0x6D, 0x33, 0xD1, 0x8F, 0x0C, 0x52, 0xB0, 0xEE,
    0x32, 0x6C, 0x8E, 0xD0, 0x53, 0x0D, 0xEF, 0xB1,
    0xF0, 0xAE, 0x4C, 0x12, 0x91, 0xCF, 0x2D, 0x73,
    0xCA, 0x94, 0x76, 0x28, 0xAB, 0xF5, 0x17, 0x49,
    0x08, 0x56, 0xB4, 0xEA, 0x69, 0x37, 0xD5, 0x8B,
    0x57, 0x09, 0xEB, 0xB5, 0x36, 0x68, 0x8A, 0xD4,
    0x95, 0xCB, 0x29, 0x77, 0xF4, 0xAA, 0x48, 0x16,
    0xE9, 0xB7, 0x55, 0x0B, 0x88, 0xD6, 0x34, 0x6A,
    0x2B, 0x75, 0x97, 0xC9, 0x4A, 0x14, 0xF6, 0xA8,
    0x74, 0x2A, 0xC8, 0x96, 0x15, 0x4B, 0xA9, 0xF7,
    0xB6, 0xE8, 0x0A, 0x54, 0xD7, 0x89, 0x6B, 0x35,
]

# ── Enums ──────────────────────────────────────────────────────────────────

class ChecksumType(Enum):
    FIXED_6B = 0
    XOR = 1
    CRC8 = 2
    MODBUS = 3

class Firmware(Enum):
    EMM = "emm"
    X = "x"
    BOTH = "both"

class Direction(Enum):
    HOST_TO_MOTOR = "host_to_motor"
    MOTOR_TO_HOST = "motor_to_host"
    UNKNOWN = "unknown"

# ── Dataclasses ────────────────────────────────────────────────────────────

@dataclass
class ParamDef:
    """Definition of a single parameter in a command."""
    name: str
    offset: int          # byte offset in data portion (0 = first data byte after code)
    length: int          # bytes (1, 2, or 4)
    scale: float = 1.0   # multiply raw value by this
    signed: bool = False # separate sign byte?
    is_sign: bool = False  # this field IS the sign byte
    unit: str = ""
    enum_map: Optional[dict] = None

@dataclass
class ResponseParam:
    """Definition of a single parameter in a response."""
    name: str
    offset: int          # byte offset in response data
    length: int
    scale: float = 1.0
    is_sign: bool = False
    unit: str = ""
    enum_map: Optional[dict] = None

@dataclass
class CommandDef:
    """Complete definition of a host→motor command and its response."""
    name: str
    code: int
    firmware: set          # {"emm", "x"}
    category: str          # "trigger", "motion", "homing", "read", "write", "config"
    params: list = field(default_factory=list)        # host send params
    response_params: list = field(default_factory=list)  # motor response params
    is_status_response: bool = False   # True if response is just addr+code+status+checksum
    has_auxiliary: bool = False        # True if command needs auxiliary byte
    auxiliary: int = 0                 # the auxiliary byte value
    response_data_len: int = 0         # expected response data length (0 = status only)
    description: str = ""

@dataclass
class ParsedResponse:
    """Result of parsing a motor response frame."""
    addr: int
    code: int
    raw_hex: str
    status: Optional[int] = None
    status_text: str = ""
    decoded_params: dict = field(default_factory=dict)
    human_readable: str = ""
    firmware: str = ""     # inferred firmware

# ── Checksum functions ─────────────────────────────────────────────────────

def calc_crc8(data: bytes) -> int:
    crc = data[0]
    for b in data[1:]:
        crc = CRC8_TABLE[crc ^ b]
    return crc

def calc_xor(data: bytes) -> int:
    result = 0
    for b in data:
        result ^= b
    return result

def calc_checksum(data: bytes, cs_type: ChecksumType = ChecksumType.FIXED_6B) -> int:
    """Calculate checksum for data bytes."""
    if cs_type == ChecksumType.FIXED_6B:
        return 0x6B
    elif cs_type == ChecksumType.XOR:
        return calc_xor(data)
    elif cs_type == ChecksumType.CRC8:
        return calc_crc8(data)
    else:
        return 0x6B  # Modbus not fully implemented; fallback

def validate_checksum(data: bytes, cs_type: ChecksumType = ChecksumType.FIXED_6B) -> bool:
    """Check if the last byte matches the calculated checksum."""
    if len(data) < 3:
        return False
    expected = calc_checksum(data[:-1], cs_type)
    return data[-1] == expected

# ── Status / Flag decoders ─────────────────────────────────────────────────

RESPONSE_STATUS_MAP = {
    0x02: "OK - command received correctly",
    0x12: "Already at zero point / homing limit already triggered",
    0xE2: "Parameter error (out of range or condition not met)",
    0xEE: "Format error (invalid command format)",
    0x9F: "Action complete (motor auto-return: position reached / homing done / grip clamped)",
}

def decode_motor_status(flags: int) -> dict:
    return {
        "Ens_TF":  bool(flags & 0x01),   # enabled
        "Prf_TF":  bool(flags & 0x02),   # position reached
        "Cgi_TF":  bool(flags & 0x04),   # stall detected
        "Cgp_TF":  bool(flags & 0x08),   # stall protection triggered
        "Esi_LF":  bool(flags & 0x10),   # left limit switch
        "Esi_RF":  bool(flags & 0x20),   # right limit switch
        "Oac_TF":  bool(flags & 0x80),   # power-loss flag
    }

def decode_homing_status(flags: int) -> dict:
    return {
        "Enc_Rdy": bool(flags & 0x01),   # encoder ready
        "Cal_Rdy": bool(flags & 0x02),   # calibration ready
        "Org_SF":  bool(flags & 0x04),   # homing in progress
        "Org_CF":  bool(flags & 0x08),   # homing failed
        "Otp_TF":  bool(flags & 0x10),   # over-temp protection
        "Ocp_TF":  bool(flags & 0x20),   # over-current protection
    }

def decode_io_status(flags: int) -> dict:
    return {
        "En_Pin":  bool(flags & 0x01),
        "Stp_Pin": bool(flags & 0x04),
        "Dir_Pin": bool(flags & 0x10),
        "Dir_OM":  bool(flags & 0x20),
    }

def decode_option_status(flags: int) -> dict:
    return {
        "MotType": "0.9deg" if (flags & 0x01) else "1.8deg",
        "FwType":  "Emm" if (flags & 0x02) else "X",
        "CtrMode": "ClosedLoop" if (flags & 0x04) else "OpenLoop",
        "MotDir":  "CCW" if (flags & 0x10) else "CW",
        "BtLock":  bool(flags & 0x20),
        "Scale":   bool(flags & 0x80),
    }

# ── Helper: decode multi-byte values ────────────────────────────────────────

def _u16(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset:offset+2], 'big')

def _u32(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset:offset+4], 'big')

# ── Command Registry ───────────────────────────────────────────────────────

COMMAND_REGISTRY: dict = {}

def _reg(code, name, firmware_set, category, params=None, resp_params=None,
         is_status=False, has_aux=False, aux=0, resp_data_len=0, desc=""):
    """Register a command definition."""
    cmd = CommandDef(
        name=name, code=code,
        firmware=firmware_set if isinstance(firmware_set, set) else {firmware_set},
        category=category,
        params=params or [],
        response_params=resp_params or [],
        is_status_response=is_status,
        has_auxiliary=has_aux,
        auxiliary=aux,
        response_data_len=resp_data_len,
        description=desc,
    )
    for fw in cmd.firmware:
        COMMAND_REGISTRY[(fw, code)] = cmd

# ── 5.2 Trigger commands

_reg(0x06, "Trigger Encoder Calibration", {"emm", "x"}, "trigger",
     has_aux=True, aux=0x45, is_status=True, desc="Calibrate encoder (motor rotates slowly)")

_reg(0x08, "Restart Motor", {"emm", "x"}, "trigger",
     has_aux=True, aux=0x97, is_status=True, desc="Reset and restart the motor")

_reg(0x0A, "Clear Current Position", {"emm", "x"}, "trigger",
     has_aux=True, aux=0x6D, is_status=True, desc="Set current position angle to zero")

_reg(0x0E, "Release Protection", {"emm", "x"}, "trigger",
     has_aux=True, aux=0x52, is_status=True, desc="Release stall/overheat/overcurrent protection")

_reg(0x0F, "Factory Reset", {"emm", "x"}, "trigger",
     has_aux=True, aux=0x5F, is_status=True, desc="Restore factory settings (requires power cycle)")

# ── 5.3 Motion control commands

_reg(0xF3, "Motor Enable", {"emm", "x"}, "motion",
     params=[
         ParamDef("aux", 0, 1, enum_map={0xAB: "EnableControl"}),
         ParamDef("enable", 1, 1, enum_map={0x00: "Disable", 0x01: "Enable"}),
         ParamDef("sync_flag", 2, 1, enum_map={0x00: "Immediate", 0x01: "Cached"}),
     ],
     is_status=True, has_aux=True, aux=0xAB,
     desc="Enable/disable motor (lock/release shaft)")

_reg(0xF5, "Torque Mode", {"x"}, "motion",
     params=[
         ParamDef("direction", 0, 1, enum_map={0x00: "CW", 0x01: "CCW"}),
         ParamDef("accel_ma_s", 1, 2, scale=1.0, unit="mA/s"),
         ParamDef("current_ma", 3, 2, scale=1.0, unit="mA"),
         ParamDef("sync_flag", 5, 1, enum_map={0x00: "Immediate", 0x01: "Cached"}),
     ],
     is_status=True,
     desc="Torque mode - motor rotates with given current")

_reg(0xC5, "Torque Mode w/ Speed Limit", {"x"}, "motion",
     params=[
         ParamDef("direction", 0, 1, enum_map={0x00: "CW", 0x01: "CCW"}),
         ParamDef("accel_ma_s", 1, 2, scale=1.0, unit="mA/s"),
         ParamDef("current_ma", 3, 2, scale=1.0, unit="mA"),
         ParamDef("sync_flag", 5, 1, enum_map={0x00: "Immediate", 0x01: "Cached"}),
         ParamDef("max_speed", 6, 2, scale=0.1, unit="RPM"),
     ],
     is_status=True,
     desc="Torque mode with max speed limit")

_reg(0xF6, "Speed Mode (X)", {"x"}, "motion",
     params=[
         ParamDef("direction", 0, 1, enum_map={0x00: "CW", 0x01: "CCW"}),
         ParamDef("accel_rpm_s", 1, 2, scale=1.0, unit="RPM/s"),
         ParamDef("speed", 3, 2, scale=0.1, unit="RPM"),
         ParamDef("sync_flag", 5, 1, enum_map={0x00: "Immediate", 0x01: "Cached"}),
     ],
     is_status=True,
     desc="Speed mode control (X firmware)")

_reg(0xF6, "Speed Mode (Emm)", {"emm"}, "motion",
     params=[
         ParamDef("direction", 0, 1, enum_map={0x00: "CW", 0x01: "CCW"}),
         ParamDef("speed", 1, 2, scale=1.0, unit="RPM"),
         ParamDef("accel", 3, 1, scale=1.0, unit="gear", enum_map={
             0: "No ramp"},  # 0-255 gear
         ),
         ParamDef("sync_flag", 4, 1, enum_map={0x00: "Immediate", 0x01: "Cached"}),
     ],
     is_status=True,
     desc="Speed mode control (Emm firmware)")

_reg(0xC6, "Speed Mode w/ Current Limit (X)", {"x"}, "motion",
     params=[
         ParamDef("direction", 0, 1, enum_map={0x00: "CW", 0x01: "CCW"}),
         ParamDef("accel_rpm_s", 1, 2, scale=1.0, unit="RPM/s"),
         ParamDef("speed", 3, 2, scale=0.1, unit="RPM"),
         ParamDef("sync_flag", 5, 1, enum_map={0x00: "Immediate", 0x01: "Cached"}),
         ParamDef("max_current_ma", 6, 2, scale=1.0, unit="mA"),
     ],
     is_status=True,
     desc="Speed mode with max current limit (X firmware)")

_reg(0xFB, "Position Mode - Direct (X)", {"x"}, "motion",
     params=[
         ParamDef("direction", 0, 1, enum_map={0x00: "CW", 0x01: "CCW"}),
         ParamDef("speed", 1, 2, scale=0.1, unit="RPM"),
         ParamDef("position", 3, 4, scale=0.1, unit="deg"),
         ParamDef("move_mode", 7, 1, enum_map={
             0x00: "Relative (prev target)", 0x01: "Absolute", 0x02: "Relative (current pos)"}),
         ParamDef("sync_flag", 8, 1, enum_map={0x00: "Immediate", 0x01: "Cached"}),
     ],
     is_status=True,
     desc="Direct speed-limit position mode (X firmware)")

_reg(0xCB, "Position Mode - Direct w/ Current Limit (X)", {"x"}, "motion",
     params=[
         ParamDef("direction", 0, 1, enum_map={0x00: "CW", 0x01: "CCW"}),
         ParamDef("speed", 1, 2, scale=0.1, unit="RPM"),
         ParamDef("position", 3, 4, scale=0.1, unit="deg"),
         ParamDef("move_mode", 7, 1, enum_map={
             0x00: "Relative (prev target)", 0x01: "Absolute", 0x02: "Relative (current pos)"}),
         ParamDef("sync_flag", 8, 1, enum_map={0x00: "Immediate", 0x01: "Cached"}),
         ParamDef("max_current_ma", 9, 2, scale=1.0, unit="mA"),
     ],
     is_status=True,
     desc="Direct speed-limit position mode with current limit (X firmware)")

_reg(0xFD, "Position Mode - Trapezoid (X)", {"x"}, "motion",
     params=[
         ParamDef("direction", 0, 1, enum_map={0x00: "CW", 0x01: "CCW"}),
         ParamDef("accel_accel", 1, 2, scale=1.0, unit="RPM/s"),
         ParamDef("decel_accel", 3, 2, scale=1.0, unit="RPM/s"),
         ParamDef("max_speed", 5, 2, scale=0.1, unit="RPM"),
         ParamDef("position", 7, 4, scale=0.1, unit="deg"),
         ParamDef("move_mode", 11, 1, enum_map={
             0x00: "Relative (prev target)", 0x01: "Absolute", 0x02: "Relative (current pos)"}),
         ParamDef("sync_flag", 12, 1, enum_map={0x00: "Immediate", 0x01: "Cached"}),
     ],
     is_status=True,
     desc="Trapezoid accel/decel position mode (X firmware)")

_reg(0xCD, "Position Mode - Trapezoid w/ Current Limit (X)", {"x"}, "motion",
     params=[
         ParamDef("direction", 0, 1, enum_map={0x00: "CW", 0x01: "CCW"}),
         ParamDef("accel_accel", 1, 2, scale=1.0, unit="RPM/s"),
         ParamDef("decel_accel", 3, 2, scale=1.0, unit="RPM/s"),
         ParamDef("max_speed", 5, 2, scale=0.1, unit="RPM"),
         ParamDef("position", 7, 4, scale=0.1, unit="deg"),
         ParamDef("move_mode", 11, 1, enum_map={
             0x00: "Relative (prev target)", 0x01: "Absolute", 0x02: "Relative (current pos)"}),
         ParamDef("sync_flag", 12, 1, enum_map={0x00: "Immediate", 0x01: "Cached"}),
         ParamDef("max_current_ma", 13, 2, scale=1.0, unit="mA"),
     ],
     is_status=True,
     desc="Trapezoid position mode with current limit (X firmware)")

_reg(0xFD, "Position Mode (Emm)", {"emm"}, "motion",
     params=[
         ParamDef("direction", 0, 1, enum_map={0x00: "CW", 0x01: "CCW"}),
         ParamDef("speed", 1, 2, scale=1.0, unit="RPM"),
         ParamDef("accel", 3, 1, scale=1.0, unit="gear"),
         ParamDef("pulses", 4, 4, scale=1.0, unit="pulse"),
         ParamDef("move_mode", 8, 1, enum_map={
             0x00: "Relative (prev target)", 0x01: "Absolute", 0x02: "Relative (current pos)"}),
         ParamDef("sync_flag", 9, 1, enum_map={0x00: "Immediate", 0x01: "Cached"}),
     ],
     is_status=True,
     desc="Position mode (Emm firmware, pulse-based)")

_reg(0xFE, "Immediate Stop", {"emm", "x"}, "motion",
     params=[
         ParamDef("aux", 0, 1, enum_map={0x98: "Stop"}),
         ParamDef("sync_flag", 1, 1, enum_map={0x00: "Immediate", 0x01: "Cached"}),
     ],
     is_status=True, has_aux=True, aux=0x98,
     desc="Immediately stop motor motion")

_reg(0xFF, "Trigger Multi-Motor Sync", {"emm", "x"}, "motion",
     params=[
         ParamDef("aux", 0, 1, enum_map={0x66: "TriggerSync"}),
     ],
     is_status=True, has_aux=True, aux=0x66,
     desc="Trigger cached multi-motor sync motion (send with broadcast addr 0)")

# ── 5.4 Homing commands

_reg(0x93, "Set Homing Zero Position", {"emm", "x"}, "homing",
     params=[
         ParamDef("aux", 0, 1, enum_map={0x88: "SetZero"}),
         ParamDef("store", 1, 1, enum_map={0x00: "No store", 0x01: "Store"}),
     ],
     is_status=True, has_aux=True, aux=0x88,
     desc="Set single-turn homing zero position")

_reg(0x9A, "Trigger Homing", {"emm", "x"}, "homing",
     params=[
         ParamDef("homing_mode", 0, 1, enum_map={
             0x00: "Nearest (single-turn)", 0x01: "Directional (single-turn)",
             0x02: "Senless (multi-turn collision)", 0x03: "EndStop (limit switch)",
             0x04: "AbsZero (absolute zero)", 0x05: "LostPosition (last power-off pos)"}),
         ParamDef("sync_flag", 1, 1, enum_map={0x00: "Immediate", 0x01: "Cached"}),
     ],
     is_status=True,
     desc="Trigger homing operation")

_reg(0x9C, "Force Stop Homing", {"emm", "x"}, "homing",
     has_aux=True, aux=0x48, is_status=True,
     desc="Force interrupt and exit homing operation")

# ── 5.5 Read system parameter commands

_reg(0x1F, "Read Firmware/Hardware Version", {"emm", "x"}, "read",
     resp_params=[
         ResponseParam("fw_version", 0, 1, scale=1.0, unit="(e.g. 200 = V2.0.0)"),
         ResponseParam("hw_series", 1, 1, enum_map={0: "X series", 1: "Y series"}),
         ResponseParam("hw_type", 2, 1, enum_map={0: "20", 1: "28", 2: "35", 3: "42", 4: "57", 5: "86"}),
         ResponseParam("hw_version", 3, 1, scale=1.0, unit="(e.g. 14 = V2.0)"),
     ],
     resp_data_len=4,
     desc="Read firmware version and hardware version")

_reg(0x20, "Read Phase Resistance/Inductance", {"emm", "x"}, "read",
     resp_params=[
         ResponseParam("phase_resistance", 0, 2, scale=1.0, unit="mOhm"),
         ResponseParam("phase_inductance", 2, 2, scale=1.0, unit="uH"),
     ],
     resp_data_len=4,
     desc="Read motor phase resistance and inductance")

_reg(0x24, "Read Bus Voltage", {"emm", "x"}, "read",
     resp_params=[
         ResponseParam("bus_voltage", 0, 2, scale=1.0, unit="mV"),
     ],
     resp_data_len=2,
     desc="Read power supply bus voltage (after diode drop)")

_reg(0x26, "Read Bus Current", {"emm", "x"}, "read",
     resp_params=[
         ResponseParam("bus_current", 0, 2, scale=1.0, unit="mA"),
     ],
     resp_data_len=2,
     desc="Read power supply bus current")

_reg(0x27, "Read Phase Current", {"emm", "x"}, "read",
     resp_params=[
         ResponseParam("phase_current", 0, 2, scale=1.0, unit="mA"),
     ],
     resp_data_len=2,
     desc="Read motor actual phase current")

_reg(0x31, "Read Calibrated Encoder", {"emm", "x"}, "read",
     resp_params=[
         ResponseParam("encoder_value", 0, 2, scale=1.0, unit="(0-65535 = 0-360deg)"),
     ],
     resp_data_len=2,
     desc="Read linearized encoder value (single-turn absolute)")

_reg(0x32, "Read Input Pulse Count", {"emm", "x"}, "read",
     resp_params=[
         ResponseParam("sign", 0, 1, enum_map={0x00: "Positive", 0x01: "Negative"}, is_sign=True),
         ResponseParam("pulse_count", 1, 4, scale=1.0, unit="pulses"),
     ],
     resp_data_len=5,
     desc="Read accumulated input pulse count")

_reg(0x33, "Read Target Position", {"emm", "x"}, "read",
     resp_params=[
         ResponseParam("sign", 0, 1, enum_map={0x00: "CW", 0x01: "CCW"}, is_sign=True),
         ResponseParam("target_position", 1, 4, scale=1.0, unit="raw"),
     ],
     resp_data_len=5,
     desc="Read motor target position")

_reg(0x34, "Read Real-time Set Target Position", {"emm", "x"}, "read",
     resp_params=[
         ResponseParam("sign", 0, 1, enum_map={0x00: "CW", 0x01: "CCW"}, is_sign=True),
         ResponseParam("set_position", 1, 4, scale=1.0, unit="raw"),
     ],
     resp_data_len=5,
     desc="Read motor real-time set target position")

_reg(0x35, "Read Real-time Speed", {"emm", "x"}, "read",
     resp_params=[
         ResponseParam("sign", 0, 1, enum_map={0x00: "CW", 0x01: "CCW"}, is_sign=True),
         ResponseParam("speed", 1, 2, scale=1.0, unit="raw"),
     ],
     resp_data_len=3,
     desc="Read motor real-time speed (Emm: RPM, X: 0.1RPM)")

_reg(0x36, "Read Real-time Position", {"emm", "x"}, "read",
     resp_params=[
         ResponseParam("sign", 0, 1, enum_map={0x00: "CW", 0x01: "CCW"}, is_sign=True),
         ResponseParam("position", 1, 4, scale=1.0, unit="raw"),
     ],
     resp_data_len=5,
     desc="Read motor real-time position")

_reg(0x37, "Read Position Error", {"emm", "x"}, "read",
     resp_params=[
         ResponseParam("sign", 0, 1, enum_map={0x00: "CW", 0x01: "CCW"}, is_sign=True),
         ResponseParam("error", 1, 4, scale=1.0, unit="raw"),
     ],
     resp_data_len=5,
     desc="Read motor position error")

_reg(0x39, "Read Driver Temperature", {"emm", "x"}, "read",
     resp_params=[
         ResponseParam("temp_sign", 0, 1, enum_map={0x00: "Negative", 0x01: "Positive"}),
         ResponseParam("temperature", 1, 1, scale=1.0, unit="C"),
     ],
     resp_data_len=2,
     desc="Read driver board temperature")

_reg(0x3A, "Read Motor Status Flags", {"emm", "x"}, "read",
     resp_params=[
         ResponseParam("status_flags", 0, 1, scale=1.0, unit="bitfield"),
     ],
     resp_data_len=1,
     desc="Read motor status flags (bitfield)")

_reg(0x3B, "Read Homing Status Flags", {"emm", "x"}, "read",
     resp_params=[
         ResponseParam("homing_flags", 0, 1, scale=1.0, unit="bitfield"),
     ],
     resp_data_len=1,
     desc="Read homing status flags (bitfield)")

_reg(0x3C, "Read Homing + Motor Status", {"emm", "x"}, "read",
     resp_params=[
         ResponseParam("homing_flags", 0, 1, scale=1.0, unit="bitfield"),
         ResponseParam("motor_flags", 1, 1, scale=1.0, unit="bitfield"),
     ],
     resp_data_len=2,
     desc="Read both homing status and motor status flags")

_reg(0x3D, "Read IO Pin Levels", {"emm", "x"}, "read",
     resp_params=[
         ResponseParam("io_flags", 0, 1, scale=1.0, unit="bitfield"),
     ],
     resp_data_len=1,
     desc="Read IO pin level states")

_reg(0x22, "Read Homing Parameters", {"emm", "x"}, "read",
     resp_params=[
         ResponseParam("homing_mode", 0, 1, enum_map={
             0x00: "Nearest", 0x01: "Directional", 0x02: "Senless",
             0x03: "EndStop", 0x04: "AbsZero", 0x05: "LostPosition"}),
         ResponseParam("homing_dir", 1, 1, enum_map={0x00: "CW", 0x01: "CCW"}),
         ResponseParam("homing_speed", 2, 2, scale=1.0, unit="RPM"),
         ResponseParam("homing_timeout", 4, 4, scale=1.0, unit="ms"),
         ResponseParam("collision_detect_speed", 8, 2, scale=1.0, unit="RPM"),
         ResponseParam("collision_detect_current", 10, 2, scale=1.0, unit="mA"),
         ResponseParam("collision_detect_time", 12, 2, scale=1.0, unit="ms"),
         ResponseParam("auto_homing_enable", 14, 1, enum_map={0x00: "Disable", 0x01: "Enable"}),
     ],
     resp_data_len=15,
     desc="Read all homing parameters")

_reg(0x41, "Read Position Reach Window", {"emm", "x"}, "read",
     resp_params=[
         ResponseParam("pos_window", 0, 2, scale=0.1, unit="deg"),
     ],
     resp_data_len=2,
     desc="Read position reach window angle")

_reg(0x1A, "Read Option Parameter Status", {"emm", "x"}, "read",
     resp_params=[
         ResponseParam("option_flags", 0, 1, scale=1.0, unit="bitfield"),
     ],
     resp_data_len=1,
     desc="Read option parameter status flags")

_reg(0x13, "Read Overheat/Overcurrent Thresholds", {"emm", "x"}, "read",
     resp_params=[
         ResponseParam("otp_threshold", 0, 2, scale=1.0, unit="C"),
         ResponseParam("ocp_threshold", 2, 2, scale=1.0, unit="mA"),
         ResponseParam("otp_ocp_time", 4, 2, scale=1.0, unit="ms"),
     ],
     resp_data_len=6,
     desc="Read overheat/overcurrent protection thresholds")

_reg(0x16, "Read Heartbeat Protection Time", {"emm", "x"}, "read",
     resp_params=[
         ResponseParam("heartbeat_time", 0, 4, scale=1.0, unit="ms"),
     ],
     resp_data_len=4,
     desc="Read heartbeat protection timeout")

_reg(0x23, "Read Integral Limit / Rigidity Coefficient", {"emm", "x"}, "read",
     resp_params=[
         ResponseParam("limit_rigidity", 0, 4, scale=1.0, unit="raw"),
     ],
     resp_data_len=4,
     desc="Read integral limit (Emm) or rigidity coefficient (X)")

_reg(0x3F, "Read Collision Homing Return Angle", {"emm", "x"}, "read",
     resp_params=[
         ResponseParam("return_angle", 0, 2, scale=0.1, unit="deg"),
     ],
     resp_data_len=2,
     desc="Read collision homing return angle")

_reg(0x21, "Read PID Parameters", {"emm", "x"}, "read",
     resp_data_len=0,  # variable, handled specially
     desc="Read PID parameters (format differs: Emm vs X)")

_reg(0x15, "Broadcast Read ID Address", {"emm", "x"}, "read",
     resp_params=[
         ResponseParam("motor_addr", 0, 1, scale=1.0),
     ],
     resp_data_len=1,
     desc="Broadcast read motor ID address (use addr 0)")

# ── System status read-all

_reg(0x43, "Read All System Status (X)", {"x"}, "read",
     has_aux=True, aux=0x7A,
     resp_data_len=34,  # 37 - 3(addr+code+checksum), data includes 2-byte header
     desc="Read all system status parameters at once (X firmware)")

_reg(0x43, "Read All System Status (Emm)", {"emm"}, "read",
     has_aux=True, aux=0x7A,
     resp_data_len=28,  # 31 - 3(addr+code+checksum), data includes 2-byte header
     desc="Read all system status parameters at once (Emm firmware)")

# ── Read/Write config

_reg(0x42, "Read Driver Config", {"emm", "x"}, "read",
     has_aux=True, aux=0x6C,
     resp_data_len=0,  # complex, handled specially
     desc="Read all driver configuration parameters")

# ── Write (modify) commands - core ones

_reg(0xAE, "Modify Motor ID/Address", {"emm", "x"}, "config",
     params=[
         ParamDef("aux", 0, 1, enum_map={0x4B: "ModifyID"}),
         ParamDef("store", 1, 1, enum_map={0x00: "No store", 0x01: "Store"}),
         ParamDef("new_addr", 2, 1, scale=1.0),
     ],
     is_status=True, has_aux=True, aux=0x4B,
     desc="Change motor ID/address (1-255)")

_reg(0x84, "Modify Microstep", {"emm", "x"}, "config",
     params=[
         ParamDef("aux", 0, 1, enum_map={0x8A: "ModifyStep"}),
         ParamDef("store", 1, 1, enum_map={0x00: "No store", 0x01: "Store"}),
         ParamDef("microstep", 2, 1, scale=1.0),
     ],
     is_status=True, has_aux=True, aux=0x8A,
     desc="Modify microstep value (0=256, 1-255=1-255)")

_reg(0x44, "Modify Open-loop Current", {"emm", "x"}, "config",
     params=[
         ParamDef("aux", 0, 1, enum_map={0x33: "ModifyOpenCurrent"}),
         ParamDef("store", 1, 1, enum_map={0x00: "No store", 0x01: "Store"}),
         ParamDef("current_ma", 2, 2, scale=1.0, unit="mA"),
     ],
     is_status=True, has_aux=True, aux=0x33,
     desc="Modify open-loop mode working current")

_reg(0x45, "Modify Closed-loop Max Current", {"emm", "x"}, "config",
     params=[
         ParamDef("aux", 0, 1, enum_map={0x66: "ModifyClosedCurrent"}),
         ParamDef("store", 1, 1, enum_map={0x00: "No store", 0x01: "Store"}),
         ParamDef("max_current_ma", 2, 2, scale=1.0, unit="mA"),
     ],
     is_status=True, has_aux=True, aux=0x66,
     desc="Modify closed-loop mode max current")

_reg(0x46, "Modify Control Mode", {"emm", "x"}, "config",
     params=[
         ParamDef("aux", 0, 1, enum_map={0xA6: "ModifyCtrlMode"}),
         ParamDef("store", 1, 1, enum_map={0x00: "No store", 0x01: "Store"}),
         ParamDef("mode", 2, 1, enum_map={0x00: "OpenLoop", 0x01: "ClosedLoop"}),
     ],
     is_status=True, has_aux=True, aux=0xA6,
     desc="Switch between open-loop and closed-loop control")

_reg(0xD5, "Modify Firmware Type", {"emm", "x"}, "config",
     params=[
         ParamDef("aux", 0, 1, enum_map={0x69: "ModifyFW"}),
         ParamDef("store", 1, 1, enum_map={0x00: "No store", 0x01: "Store"}),
         ParamDef("fw_type", 2, 1, enum_map={0x00: "X", 0x01: "Emm", 0x02: "Emm Turbo"}),
     ],
     is_status=True, has_aux=True, aux=0x69,
     desc="Change firmware type (X / Emm / Emm Turbo)")

_reg(0xD7, "Modify Motor Type", {"emm", "x"}, "config",
     params=[
         ParamDef("aux", 0, 1, enum_map={0x35: "ModifyMotorType"}),
         ParamDef("store", 1, 1, enum_map={0x00: "No store", 0x01: "Store"}),
         ParamDef("motor_type", 2, 1, enum_map={0x19: "1.8deg", 0x32: "0.9deg"}),
     ],
     is_status=True, has_aux=True, aux=0x35,
     desc="Change motor step angle type")

_reg(0xD4, "Modify Motion Direction", {"emm", "x"}, "config",
     params=[
         ParamDef("aux", 0, 1, enum_map={0x60: "ModifyDir"}),
         ParamDef("store", 1, 1, enum_map={0x00: "No store", 0x01: "Store"}),
         ParamDef("direction", 2, 1, enum_map={0x00: "CW", 0x01: "CCW"}),
     ],
     is_status=True, has_aux=True, aux=0x60,
     desc="Modify motor motion positive direction")

_reg(0x50, "Modify Power-loss Flag", {"emm", "x"}, "config",
     params=[
         ParamDef("flag", 0, 1, enum_map={0x00: "Clear", 0x01: "Set"}),
     ],
     is_status=True,
     desc="Write power-loss flag (reads 1 after power cycle)")

_reg(0x4F, "Modify Scale Input (Pos/Vel)", {"emm", "x"}, "config",
     params=[
         ParamDef("aux", 0, 1, enum_map={0x71: "ModifyScale"}),
         ParamDef("store", 1, 1, enum_map={0x00: "No store", 0x01: "Store"}),
         ParamDef("enable", 2, 1, enum_map={0x00: "Disable", 0x01: "Enable"}),
     ],
     is_status=True, has_aux=True, aux=0x71,
     desc="X: enable 0.01deg position input; Emm: enable 0.1RPM speed input")

_reg(0xD1, "Modify Position Reach Window", {"emm", "x"}, "config",
     params=[
         ParamDef("aux", 0, 1, enum_map={0x07: "ModifyWindow"}),
         ParamDef("store", 1, 1, enum_map={0x00: "No store", 0x01: "Store"}),
         ParamDef("window", 2, 2, scale=0.1, unit="deg"),
     ],
     is_status=True, has_aux=True, aux=0x07,
     desc="Modify position reach window angle (default 0.8 deg)")


# ── Build Command ──────────────────────────────────────────────────────────

def build_command(addr: int, code: int, firmware: str,
                  param_values: dict, cs_type: ChecksumType = ChecksumType.FIXED_6B) -> bytes:
    """Build a host→motor command frame from structured parameters.

    Args:
        addr: Motor address (0-255, 0=broadcast)
        code: Command function code
        firmware: "emm" or "x"
        param_values: Dict of param_name -> value
        cs_type: Checksum type

    Returns:
        Complete command bytes ready to send over serial
    """
    cmd = COMMAND_REGISTRY.get((firmware, code))
    if not cmd:
        cmd = COMMAND_REGISTRY.get(("both", code))
    if not cmd:
        raise ValueError(f"Unknown command: firmware={firmware}, code=0x{code:02X}")

    # Calculate total data length
    max_offset = 0
    for p in cmd.params:
        end = p.offset + p.length
        if end > max_offset:
            max_offset = end

    data = bytearray(max_offset)

    # Fill in auxiliary if needed
    if cmd.has_auxiliary:
        # auxiliary is at offset 0
        pass  # handled by param_values or default

    for p in cmd.params:
        if p.name in param_values:
            val = param_values[p.name]
        elif p.name == "aux" and cmd.has_auxiliary:
            val = cmd.auxiliary
        else:
            continue  # skip unfilled optional params (let caller handle)

        val = int(val)
        data[p.offset:p.offset + p.length] = val.to_bytes(p.length, 'big')

    frame = bytes([addr, code]) + bytes(data)
    cs = calc_checksum(frame, cs_type)
    return frame + bytes([cs])


def build_simple_command(addr: int, code: int, firmware: str = "emm",
                         cs_type: ChecksumType = ChecksumType.FIXED_6B) -> bytes:
    """Build a simple read command (addr + code + checksum only)."""
    cmd = COMMAND_REGISTRY.get((firmware, code))
    if not cmd:
        cmd = COMMAND_REGISTRY.get(("both", code))
    if not cmd:
        raise ValueError(f"Unknown command: firmware={firmware}, code=0x{code:02X}")

    frame = bytes([addr, code])
    if cmd.has_auxiliary:
        frame += bytes([cmd.auxiliary])
    cs = calc_checksum(frame, cs_type)
    return frame + bytes([cs])


# ── Parse Response ─────────────────────────────────────────────────────────

def _decode_position_emm(pos_raw: int, sign: int) -> float:
    """Emm firmware: 0-65535 = 0-360 degrees."""
    angle = (pos_raw * 360.0) / 65536.0
    return -angle if sign == 1 else angle

def _decode_position_x(pos_raw: int, sign: int) -> float:
    """X firmware: raw/10 = degrees."""
    angle = pos_raw / 10.0
    return -angle if sign == 1 else angle

def _decode_speed_emm(speed_raw: int, sign: int) -> float:
    """Emm firmware: raw = RPM."""
    return -speed_raw if sign == 1 else speed_raw

def _decode_speed_x(speed_raw: int, sign: int) -> float:
    """X firmware: raw*0.1 = RPM."""
    speed = speed_raw * 0.1
    return -speed if sign == 1 else speed


def _decode_system_status_43(raw: bytes, firmware: str, result: ParsedResponse) -> ParsedResponse:
    """Decode 0x43 Read All System Status response with per-parameter sign bytes.

    Frame format: [addr][code][byte_count][param_count][params...][checksum]
    X firmware: 34 data bytes (incl 2-byte header), 12 params, 5 sign bytes
    Emm firmware: 28 data bytes (incl 2-byte header), 9 params, 4 sign bytes
    """
    data_bytes = raw[2:-1]
    params = {}

    if firmware == "x":
        if len(data_bytes) < 32:
            result.status_text = "0x43 response too short for X firmware"
            result.human_readable = f"Motor #{result.addr}: 0x43 data={data_bytes.hex(' ').upper()}"
            return result
        d = data_bytes[2:]  # skip byte_count + param_count header

        params["bus_voltage_mv"] = _u16(d, 0)
        params["bus_current_ma"] = _u16(d, 2)
        params["phase_current_ma"] = _u16(d, 4)
        params["encoder_raw"] = _u16(d, 6)
        params["calibrated_encoder"] = _u16(d, 8)

        sign1 = d[10]
        pos_raw = _u32(d, 11)
        params["target_position_deg"] = -pos_raw * 0.1 if sign1 else pos_raw * 0.1

        sign2 = d[15]
        speed_raw = _u16(d, 16)
        params["speed_rpm"] = -speed_raw * 0.1 if sign2 else speed_raw * 0.1

        sign3 = d[18]
        pos2_raw = _u32(d, 19)
        params["position_deg"] = -pos2_raw * 0.1 if sign3 else pos2_raw * 0.1

        sign4 = d[23]
        err_raw = _u32(d, 24)
        params["position_error_deg"] = -err_raw * 0.01 if sign4 else err_raw * 0.01

        params["temperature_c"] = d[29]
        params["homing_flags"] = d[30]
        params["motor_flags"] = d[31]

    elif firmware == "emm":
        if len(data_bytes) < 26:
            result.status_text = "0x43 response too short for Emm firmware"
            result.human_readable = f"Motor #{result.addr}: 0x43 data={data_bytes.hex(' ').upper()}"
            return result
        d = data_bytes[2:]

        params["bus_voltage_mv"] = _u16(d, 0)
        params["phase_current_ma"] = _u16(d, 2)
        params["calibrated_encoder"] = _u16(d, 4)

        sign1 = d[6]
        pos_raw = _u32(d, 7)
        params["target_position_deg"] = _decode_position_emm(pos_raw, sign1)

        sign2 = d[11]
        speed_raw = _u16(d, 12)
        params["speed_rpm"] = -speed_raw if sign2 else speed_raw

        sign3 = d[14]
        pos2_raw = _u32(d, 15)
        params["position_deg"] = _decode_position_emm(pos2_raw, sign3)

        sign4 = d[19]
        err_raw = _u32(d, 20)
        params["position_error_deg"] = _decode_position_emm(err_raw, sign4)

        params["homing_flags"] = d[24]
        params["motor_flags"] = d[25]

    result.decoded_params = params
    result.status_text = "System Status (0x43)"
    parts = []
    for k, v in params.items():
        if k.endswith("_flags"):
            parts.append(f"{k}=0x{v:02X}")
        elif isinstance(v, float):
            parts.append(f"{k}={v:.2f}")
        else:
            parts.append(f"{k}={v}")
    result.human_readable = f"Motor #{result.addr}: {', '.join(parts)}"
    return result


def parse_host_command(raw: bytes, firmware: str = "emm") -> ParsedResponse:
    """Parse a host-to-motor command frame and describe its intent.

    Unlike parse_response (which targets motor→host replies), this function
    interprets the *sent* command: it looks up the command code in the registry
    and decodes the sent parameters.
    """
    if len(raw) < 3:
        return ParsedResponse(addr=0, code=0, raw_hex=raw.hex(' ').upper(),
                              status_text="Frame too short")

    addr = raw[0]
    code = raw[1]
    data_part = raw[2:-1]  # between code and checksum

    cmd = COMMAND_REGISTRY.get((firmware, code))
    if not cmd:
        cmd = COMMAND_REGISTRY.get(("both", code))

    result = ParsedResponse(addr=addr, code=code, raw_hex=raw.hex(' ').upper(), firmware=firmware)

    if cmd:
        result.status_text = cmd.name
        sent_params = {}
        for p in cmd.params:
            if p.offset + p.length <= len(data_part):
                val = int.from_bytes(data_part[p.offset:p.offset + p.length], 'big')
                if p.enum_map and val in p.enum_map:
                    sent_params[p.name] = p.enum_map[val]
                else:
                    sent_params[p.name] = val * p.scale
        result.decoded_params = sent_params
        if sent_params:
            param_strs = [f"{k}={v}" for k, v in sent_params.items()]
            result.human_readable = f"#{addr} {cmd.name}: {', '.join(param_strs)}"
        else:
            result.human_readable = f"#{addr} {cmd.name}"
    else:
        result.status_text = f"0x{code:02X}"
        result.human_readable = f"#{addr} 0x{code:02X} data={data_part.hex(' ').upper() if data_part else '(none)'}"

    return result


def get_command_name(code: int, firmware: str) -> str:
    """Return a human-readable name for a command code."""
    cmd = COMMAND_REGISTRY.get((firmware, code))
    if not cmd:
        cmd = COMMAND_REGISTRY.get(("both", code))
    return cmd.name if cmd else f"0x{code:02X}"


def parse_response(raw: bytes, firmware: str = "emm",
                   cs_type: ChecksumType = ChecksumType.FIXED_6B) -> ParsedResponse:
    """Parse a motor response frame.

    Args:
        raw: Complete response frame bytes
        firmware: "emm" or "x" for position/speed decoding
        cs_type: Checksum type for validation

    Returns:
        ParsedResponse with decoded data
    """
    if len(raw) < 3:
        return ParsedResponse(addr=0, code=0, raw_hex=raw.hex(' ').upper(),
                              status_text="Frame too short")

    addr = raw[0]
    code = raw[1]
    raw_hex = raw.hex(' ').upper()

    # Validate checksum
    cs_valid = validate_checksum(raw, cs_type)

    # Check if this is a status-only response
    # Status responses: [addr][code][status][checksum] = 4 bytes
    data_bytes = raw[2:-1]  # between code and checksum

    result = ParsedResponse(addr=addr, code=code, raw_hex=raw_hex, firmware=firmware)

    # Check for status code in first data byte
    if len(data_bytes) >= 1 and data_bytes[0] in RESPONSE_STATUS_MAP:
        result.status = data_bytes[0]
        result.status_text = RESPONSE_STATUS_MAP[data_bytes[0]]
        result.human_readable = f"Motor #{addr}: {result.status_text}"

        # If there's more data after the status byte, try to decode
        extra_data = data_bytes[1:]
        if extra_data and code == 0x9F:
            # 9F action complete can come with extra info
            pass
        return result

    # Special handling for 0x43: Read All System Status (complex multi-param with headers)
    if code == 0x43 and len(data_bytes) >= 2:
        return _decode_system_status_43(raw, firmware, result)

    # Try to find command definition for structured parsing
    cmd = COMMAND_REGISTRY.get((firmware, code))
    if not cmd:
        cmd = COMMAND_REGISTRY.get(("both", code))

    if cmd and cmd.response_params:
        result = _parse_structured_response(raw, cmd, firmware, result)
    elif cmd and cmd.response_data_len > 0:
        result.status_text = f"Response data ({len(data_bytes)} bytes)"
        result.human_readable = f"Motor #{addr}: {cmd.name} data={data_bytes.hex(' ').upper()}"
    else:
        # Unknown or unsupported response format
        result.status_text = f"Unknown response format (code=0x{code:02X})"
        result.human_readable = f"Motor #{addr}: code=0x{code:02X}, data={data_bytes.hex(' ').upper()}"

    if not cs_valid:
        result.status_text += " [CHECKSUM INVALID]"

    return result


def _parse_structured_response(raw: bytes, cmd: CommandDef, firmware: str,
                                result: ParsedResponse) -> ParsedResponse:
    """Parse a response with structured parameter layout."""
    data_bytes = raw[2:-1]  # between code and checksum
    params = {}
    sign_value = 0
    parts = []

    for rp in cmd.response_params:
        if rp.offset + rp.length > len(data_bytes):
            continue

        raw_val = int.from_bytes(
            data_bytes[rp.offset:rp.offset + rp.length], 'big')

        if rp.is_sign:
            sign_value = raw_val
            if rp.enum_map:
                params[rp.name] = rp.enum_map.get(raw_val, f"0x{raw_val:02X}")
            else:
                params[rp.name] = raw_val
            continue

        # Apply scale
        display_val = raw_val * rp.scale

        # Apply unit-specific decoding based on param name
        if rp.name == "position" or rp.name == "target_position" or rp.name == "set_position" or rp.name == "error":
            if firmware == "emm":
                display_val = _decode_position_emm(raw_val, sign_value)
            else:
                display_val = _decode_position_x(raw_val, sign_value)
            rp.unit = "deg"
        elif rp.name == "speed":
            if firmware == "emm":
                display_val = _decode_speed_emm(raw_val, sign_value)
            else:
                display_val = _decode_speed_x(raw_val, sign_value)
            rp.unit = "RPM"

        # Enum mapping
        if rp.enum_map and raw_val in rp.enum_map:
            params[rp.name] = rp.enum_map[raw_val]
        else:
            params[rp.name] = display_val

        # Build human-readable parts
        if isinstance(params[rp.name], str):
            parts.append(f"{rp.name}={params[rp.name]}")
        elif isinstance(params[rp.name], float):
            parts.append(f"{rp.name}={params[rp.name]:.2f}{rp.unit}")
        else:
            parts.append(f"{rp.name}={params[rp.name]}{rp.unit}")

    result.decoded_params = params
    result.status_text = cmd.name
    result.human_readable = f"Motor #{result.addr}: {', '.join(parts)}"

    # Special bitfield decoders
    if cmd.code == 0x3A and "status_flags" in params:
        flags = data_bytes[0] if len(data_bytes) > 0 else 0
        result.decoded_params["flags_detail"] = decode_motor_status(flags)
    elif cmd.code == 0x3B and "homing_flags" in params:
        flags = data_bytes[0] if len(data_bytes) > 0 else 0
        result.decoded_params["flags_detail"] = decode_homing_status(flags)
    elif cmd.code == 0x3D and "io_flags" in params:
        flags = data_bytes[0] if len(data_bytes) > 0 else 0
        result.decoded_params["flags_detail"] = decode_io_status(flags)
    elif cmd.code == 0x1A and "option_flags" in params:
        flags = data_bytes[0] if len(data_bytes) > 0 else 0
        result.decoded_params["flags_detail"] = decode_option_status(flags)

    return result


# ── Direction Detection ────────────────────────────────────────────────────

def detect_direction(raw: bytes) -> Direction:
    """Detect whether a serial frame is host→motor or motor→host.

    Heuristic:
    1. Check byte[2] for known response status codes → motor_to_host
    2. Match frame length + code against command registry → precise
    3. Checksum validation as fallback indicator
    """
    if len(raw) < 3:
        return Direction.UNKNOWN

    code = raw[1]
    third_byte = raw[2]

    # Tier 1: status code sniffing
    if third_byte in RESPONSE_STATUS_MAP:
        return Direction.MOTOR_TO_HOST

    # Tier 2: pattern match against registry
    frame_len = len(raw)
    for (fw_key, cmd_code), cmd in COMMAND_REGISTRY.items():
        if cmd_code != code:
            continue

        # Check if frame matches a known host command length
        if cmd.params:
            host_len = 3  # addr + code
            for p in cmd.params:
                host_len = max(host_len, 3 + p.offset + p.length)
            # +1 for checksum but not included in offset calc
            if frame_len == host_len + 1 or frame_len == host_len:
                return Direction.HOST_TO_MOTOR

        # Check if frame matches a known response
        if cmd.is_status_response:
            if frame_len == 4:  # addr + code + status + checksum
                return Direction.MOTOR_TO_HOST
        if cmd.response_data_len > 0:
            resp_len = 3 + cmd.response_data_len + 1  # +checksum
            if frame_len == resp_len:
                return Direction.MOTOR_TO_HOST

    # Tier 3: bare minimum commands (addr + code + checksum, no data)
    if frame_len == 3:
        # Could be either direction; default to host
        return Direction.HOST_TO_MOTOR

    # Tier 4: checksum-based heuristic
    if validate_checksum(raw):
        # Heuristic: if frame is short, likely a response
        if frame_len <= 5:
            return Direction.MOTOR_TO_HOST
        return Direction.HOST_TO_MOTOR

    return Direction.UNKNOWN


def describe_direction(raw: bytes) -> str:
    """Return a human-readable direction label."""
    d = detect_direction(raw)
    if d == Direction.HOST_TO_MOTOR:
        return "H→M"
    elif d == Direction.MOTOR_TO_HOST:
        return "M→H"
    return "???"


# ── Utility: decode raw hex string ─────────────────────────────────────────

def decode_hex_string(hex_str: str, firmware: str = "emm") -> ParsedResponse:
    """Parse a hex string (e.g. '01 36 6B') into a ParsedResponse."""
    hex_str = hex_str.replace(' ', '').replace('\n', '').replace('\r', '')
    try:
        raw = bytes.fromhex(hex_str)
    except ValueError:
        return ParsedResponse(addr=0, code=0, raw_hex=hex_str,
                              status_text="Invalid hex string")
    return parse_response(raw, firmware)


# ── Get all available command codes for a firmware ─────────────────────────

def get_commands_for_firmware(firmware: str) -> list:
    """Return all command definitions available for a given firmware."""
    result = []
    seen = set()
    for (fw, code), cmd in COMMAND_REGISTRY.items():
        if fw == firmware or fw == "both":
            key = (cmd.name, code)
            if key not in seen:
                seen.add(key)
                result.append(cmd)
    return sorted(result, key=lambda c: c.code)


def get_command_info(code: int, firmware: str) -> Optional[CommandDef]:
    """Get a single command definition."""
    cmd = COMMAND_REGISTRY.get((firmware, code))
    if not cmd:
        cmd = COMMAND_REGISTRY.get(("both", code))
    return cmd
