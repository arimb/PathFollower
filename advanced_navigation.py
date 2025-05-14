import serial
import struct
from fastcrc import crc16

PORT = "/dev/ttyUSB0"
BAUD = 115200

def read_stream():
    data = ser.read(99999)
    # print(data)
    if data:
        buffer.extend(data)
        print(f"Received {len(data)} bytes. Buffer size: {len(buffer)}")
        return True
    return False

def compute_lrc(packet_id, length, crc_bytes):
    lrc_input = packet_id + length + crc_bytes[0] + crc_bytes[1]
    return ((lrc_input ^ 0xFF) + 1) & 0xFF

# def compute_crc(payload_bytes):
#     return crc16.modbus(payload_bytes)

def compute_crc(data: bytes, poly=0x1021, init=0xFFFF):
    crc = init
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = (crc << 1) ^ poly if (crc & 0x8000) else (crc << 1)
        crc &= 0xFFFF
    return crc

def find_anpp_packets(buffer: bytearray):
    packets = []
    i = 0
    while i < len(buffer) - 5:
        lrc = buffer[i]
        pid = buffer[i + 1]
        length = buffer[i + 2]
        crc_low = buffer[i + 3]
        crc_high = buffer[i + 4]
        crc_bytes = [crc_low, crc_high]

        expected_lrc = compute_lrc(pid, length, crc_bytes)
        if lrc != expected_lrc:
            i += 1
            continue

        total_length = 5 + length
        if i + total_length > len(buffer):
            break  # Wait for more data

        payload = buffer[i + 5:i + 5 + length]
        crc_expected = (crc_high << 8) | crc_low
        crc_actual = compute_crc(payload)

        if crc_expected == crc_actual:
            packets.append((pid, payload))
            print(f"[OK] Packet ID {pid} | Payload {len(payload)} bytes at offset {i}")
            i += total_length
        else:
            print(f"[CRC FAIL] Packet ID {pid} at offset {i}")
            i += 1  # Realign
    return packets

def parse_packet(packet_id, payload):
    if packet_id == 0x14:
        print("System State:", parse_system_state_packet(payload))
    elif packet_id == 0x1C:
        print("Raw Sensors:", parse_raw_sensors_packet(payload))
    else:
        print(f"Unhandled packet {packet_id}: {payload.hex()}")

def parse_system_state_packet(data):
    """Parses a System State Packet (ID=20, length=100)"""
    if len(data) != 100:
        print("Invalid length for system state packet")
        return None

    fields = struct.unpack('<HHII3d16f', data)
    return {
        'system_status': fields[0],
        'filter_status': fields[1],
        'unix_time': fields[2],
        'microseconds': fields[3],
        'latitude_rad': fields[4],
        'longitude_rad': fields[5],
        'height_m': fields[6],
        'velocity': {'N': fields[7], 'E': fields[8], 'D': fields[9]},
        'acceleration_body': {'X': fields[10], 'Y': fields[11], 'Z': fields[12]},
        'g_force': fields[13],
        'orientation': {'roll': fields[14], 'pitch': fields[15], 'heading': fields[16]},
        'angular_velocity': {'X': fields[17], 'Y': fields[18], 'Z': fields[19]},
        'position_std_dev': {'lat': fields[20], 'lon': fields[21], 'height': fields[22]},
    }

def parse_raw_sensors_packet(data):
    """Parses a Raw Sensors Packet (ID=28, length=48)"""
    if len(data) != 48:
        print("Invalid length for raw sensors packet")
        return None

    fields = struct.unpack('<12f', data)
    return {
        'accelerometer': {'X': fields[0], 'Y': fields[1], 'Z': fields[2]},
        'gyroscope': {'X': fields[3], 'Y': fields[4], 'Z': fields[5]},
        'magnetometer': {'X': fields[6], 'Y': fields[7], 'Z': fields[8]},
        'imu_temperature': fields[9],
        'pressure': fields[10],
        'pressure_temperature': fields[11],
    }


ser = serial.Serial(PORT, BAUD, timeout=1)
buffer = bytearray()
print(f"Connected to {PORT} at {BAUD} baud.")

while True:
    if read_stream():
        packets = find_anpp_packets(buffer)
        for packet_id, payload in packets:
            parse_packet(packet_id, payload)
