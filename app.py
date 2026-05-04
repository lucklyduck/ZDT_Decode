"""
ZDT_X42S Stepper Motor Web Debug Tool
Flask + SocketIO backend for serial communication, protocol parsing,
and motor emulation.
"""

import time
import threading
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit

from zdt_protocol import (
    build_command, build_simple_command, parse_response, parse_host_command,
    detect_direction, Direction, decode_hex_string,
    get_command_info, get_commands_for_firmware,
    ChecksumType, COMMAND_REGISTRY,
)
from serial_manager import SerialManager
from zdt_emulator import EmulatorManager, MotorEmulatorState

app = Flask(__name__)
app.config['SECRET_KEY'] = 'zdt-debug-tool'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ── Global state ────────────────────────────────────────────────────────────

serial_mgr = SerialManager(socketio)
emulator_mgr = EmulatorManager()
emulator_thread: threading.Thread | None = None
emulator_run = False

# ── Flask routes ────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/ports')
def api_ports():
    ports = SerialManager.list_ports()
    return jsonify(ports)

@app.route('/api/build_command', methods=['POST'])
def api_build():
    """Build a command from structured parameters and return the hex string."""
    data = request.json
    try:
        addr = int(data.get('addr', 1))
        code = int(data.get('code', 0x36))
        firmware = data.get('firmware', 'emm')
        params = data.get('params', {})
        cs_type_str = data.get('cs_type', '0x6B')

        cs_type = ChecksumType.FIXED_6B
        if cs_type_str == 'xor':
            cs_type = ChecksumType.XOR
        elif cs_type_str == 'crc8':
            cs_type = ChecksumType.CRC8

        cmd = get_command_info(code, firmware)
        if not cmd:
            return jsonify({"error": f"Unknown command: firmware={firmware}, code=0x{code:02X}"}), 400

        if cmd.params:
            result = build_command(addr, code, firmware, params, cs_type)
        else:
            result = build_simple_command(addr, code, firmware, cs_type)

        return jsonify({
            "hex": result.hex(' ').upper(),
            "length": len(result),
            "command_name": cmd.name,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/decode', methods=['POST'])
def api_decode():
    """Decode a hex string to human-readable format."""
    data = request.json
    hex_str = data.get('hex', '')
    firmware = data.get('firmware', 'emm')

    result = decode_hex_string(hex_str, firmware)
    return jsonify({
        "addr": result.addr,
        "code": result.code,
        "status": result.status,
        "status_text": result.status_text,
        "decoded_params": result.decoded_params,
        "human_readable": result.human_readable,
        "firmware": result.firmware,
    })

@app.route('/api/commands')
def api_commands():
    """List available commands for a firmware."""
    fw = request.args.get('firmware', 'emm')
    cmds = get_commands_for_firmware(fw)
    result = []
    for c in cmds:
        result.append({
            "name": c.name,
            "code": c.code,
            "code_hex": f"0x{c.code:02X}",
            "category": c.category,
            "firmware": list(c.firmware),
            "description": c.description,
            "params": [
                {
                    "name": p.name,
                    "length": p.length,
                    "scale": p.scale,
                    "unit": p.unit,
                    "enum_map": p.enum_map,
                } for p in c.params
            ],
            "has_auxiliary": c.has_auxiliary,
            "auxiliary": c.auxiliary,
        })
    return jsonify(result)

# ── SocketIO events ─────────────────────────────────────────────────────────

@socketio.on('connect')
def on_connect():
    emit('serial_status', {
        'connected': serial_mgr.is_connected,
        'status': 'Connected to server',
        'port': serial_mgr.port,
        'baudrate': serial_mgr.baudrate,
    })
    emit('emulator_status', {
        'running': emulator_run,
        'motors': [m.to_dict() for m in emulator_mgr.motors.values()],
    })

@socketio.on('serial_connect')
def on_serial_connect(data):
    port = data.get('port', '')
    baudrate = int(data.get('baudrate', 115200))
    checksum = data.get('checksum', '0x6B')
    firmware = data.get('firmware', 'emm')

    serial_mgr.firmware = firmware
    if checksum == 'xor':
        serial_mgr.cs_type = ChecksumType.XOR
    elif checksum == 'crc8':
        serial_mgr.cs_type = ChecksumType.CRC8
    else:
        serial_mgr.cs_type = ChecksumType.FIXED_6B

    success = serial_mgr.connect(port, baudrate)
    emit('serial_status', {
        'connected': serial_mgr.is_connected,
        'status': f'Connected to {port} @ {baudrate}' if success else f'Failed to open {port}',
        'port': serial_mgr.port,
        'baudrate': serial_mgr.baudrate,
    })

@socketio.on('serial_disconnect')
def on_serial_disconnect():
    serial_mgr.disconnect()
    emit('serial_status', {
        'connected': False,
        'status': 'Disconnected',
        'port': '',
        'baudrate': 0,
    })

@socketio.on('send_command')
def on_send_command(data):
    """Send raw hex command over serial or to emulator."""
    hex_str = data.get('hex', '')
    target = data.get('target', 'serial')  # 'serial' or 'emulator'

    try:
        hex_str = hex_str.replace(' ', '').replace('\n', '')
        raw = bytes.fromhex(hex_str)
    except ValueError:
        emit('serial_error', {'message': 'Invalid hex string'})
        return

    if target == 'emulator' and emulator_run:
        # Emit sent command first (source=sent for monitor differentiation)
        sent_parsed = parse_host_command(raw, serial_mgr.firmware)
        emit('serial_data', {
            'hex': raw.hex(' ').upper(),
            'direction': 'host_to_motor',
            'direction_label': '>>>',
            'source': 'sent',
            'parsed': {
                'addr': sent_parsed.addr,
                'code': sent_parsed.code,
                'status_text': sent_parsed.status_text,
                'decoded_params': sent_parsed.decoded_params,
                'human_readable': sent_parsed.human_readable,
                'firmware': sent_parsed.firmware,
            },
            'emulated': True,
        })

        resp = emulator_mgr.process_command(raw)
        if resp:
            parsed = parse_response(resp, serial_mgr.firmware, serial_mgr.cs_type)
            serial_mgr._update_motor_state(parsed.addr, parsed.code, parsed.decoded_params, parsed.firmware)
            emit('serial_data', {
                'hex': resp.hex(' ').upper(),
                'direction': 'motor_to_host',
                'direction_label': '<<<',
                'source': 'received',
                'parsed': {
                    'addr': parsed.addr,
                    'code': parsed.code,
                    'status': parsed.status,
                    'status_text': parsed.status_text,
                    'decoded_params': parsed.decoded_params,
                    'human_readable': parsed.human_readable,
                    'firmware': parsed.firmware,
                },
                'emulated': True,
            })
            emit('motor_states_update', {'states': list(serial_mgr.motor_states.values())})
    else:
        serial_mgr.send(raw)

@socketio.on('send_structured')
def on_send_structured(data):
    """Build and send a structured command."""
    try:
        addr = int(data.get('addr', 1))
        code = int(data.get('code', 0x36))
        firmware = data.get('firmware', 'emm')
        params = data.get('params', {})
        target = data.get('target', 'serial')

        cmd = get_command_info(code, firmware)
        if not cmd:
            emit('serial_error', {'message': f'Unknown command: 0x{code:02X}'})
            return

        if cmd.params:
            raw = build_command(addr, code, firmware, params, serial_mgr.cs_type)
        else:
            raw = build_simple_command(addr, code, firmware, serial_mgr.cs_type)

        # Send via the common handler
        on_send_command({'hex': raw.hex(' '), 'target': target})
    except Exception as e:
        emit('serial_error', {'message': str(e)})

@socketio.on('emulator_start')
def on_emulator_start(data):
    global emulator_run, emulator_thread
    motors = data.get('motors', [
        {'addr': 1, 'firmware': 'emm'},
        {'addr': 2, 'firmware': 'x'},
    ])

    emulator_mgr.start(motors)
    emulator_run = True

    # Start background tick thread
    def emulator_loop():
        while emulator_run:
            emulator_mgr.tick()
            # Push state to clients at 10Hz
            motor_states = [m.to_dict() for m in emulator_mgr.motors.values()]
            socketio.emit('emulator_state', {'motors': motor_states})
            time.sleep(0.1)

    emulator_thread = threading.Thread(target=emulator_loop, daemon=True)
    emulator_thread.start()

    emit('emulator_status', {
        'running': True,
        'motors': [m.to_dict() for m in emulator_mgr.motors.values()],
    })

@socketio.on('emulator_stop')
def on_emulator_stop():
    global emulator_run
    emulator_run = False
    emulator_mgr.stop()
    emit('emulator_status', {
        'running': False,
        'motors': [],
    })

@socketio.on('emulator_send')
def on_emulator_send(data):
    """Send a command to the emulator (shorthand for send_command with target=emulator)."""
    data['target'] = 'emulator'
    on_send_command(data)

@socketio.on('set_config')
def on_set_config(data):
    """Update global configuration."""
    if 'firmware' in data:
        serial_mgr.firmware = data['firmware']
    if 'checksum' in data:
        cs = data['checksum']
        if cs == 'xor':
            serial_mgr.cs_type = ChecksumType.XOR
        elif cs == 'crc8':
            serial_mgr.cs_type = ChecksumType.CRC8
        else:
            serial_mgr.cs_type = ChecksumType.FIXED_6B

@socketio.on('emulator_update_motor')
def on_emulator_update_motor(data):
    """Update a single emulated motor's state from the frontend."""
    if not emulator_run:
        emit('serial_error', {'message': '请先启动模拟器'})
        return
    addr = int(data.get('addr', 0))
    motor = emulator_mgr.motors.get(addr)
    if not motor:
        emit('serial_error', {'message': f'未找到电机 #{addr}'})
        return

    # Update simple fields
    for field in ['position_deg', 'speed_rpm', 'target_position_deg',
                  'target_speed_rpm', 'phase_current_ma', 'bus_voltage_mv',
                  'bus_current_ma', 'temperature_c']:
        if field in data:
            setattr(motor, field, float(data[field]))

    if 'enabled' in data:
        motor.enabled = bool(data['enabled'])
        if motor.enabled:
            motor.status_flags |= 0x01
        else:
            motor.status_flags &= ~0x01

    if 'status_flags' in data:
        motor.status_flags = int(data['status_flags'])

    if 'homing_flags' in data:
        motor.homing_flags = int(data['homing_flags'])

    if 'firmware' in data and data['firmware'] in ('emm', 'x'):
        motor.firmware = data['firmware']

    if 'moving' in data:
        motor.moving = bool(data['moving'])

    # Handle addr change: update the motors dict key
    new_addr = int(data.get('new_addr', 0))
    if new_addr and new_addr != addr and 1 <= new_addr <= 255:
        if new_addr not in emulator_mgr.motors:
            emulator_mgr.motors.pop(addr, None)
            motor.addr = new_addr
            emulator_mgr.motors[new_addr] = motor

    # Push updated state
    motor_states = [m.to_dict() for m in emulator_mgr.motors.values()]
    socketio.emit('emulator_state', {'motors': motor_states})


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  ZDT_X42S 闭环步进电机调试工具")
    print("  浏览器打开 http://localhost:5000")
    print("=" * 60)
    # use_reloader=False to avoid double-process issues with serial port
    socketio.run(app, host='0.0.0.0', port=5000, debug=True,
                 use_reloader=False, allow_unsafe_werkzeug=True)
