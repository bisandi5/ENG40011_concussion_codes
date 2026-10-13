import smbus
import time

bus = smbus.SMBus(1)
ADDRESS = 0x1C

def convert_to_g(msb, lsb):
    value = ((msb << 8) | lsb) >> 4
    if value >= 2048:
        value -= 4096
    return value / 1024.0

try:
    # Standby mode before changing settings
    bus.write_byte_data(ADDRESS, 0x2A, 0x00)

    # Measurement range: +/-2 g
    bus.write_byte_data(ADDRESS, 0x0E, 0x00)

    # Active mode, 100 sensor samples per second
    bus.write_byte_data(ADDRESS, 0x2A, 0x19)
    time.sleep(0.1)

    print("Acceleration in g. Press Ctrl+C to stop.")

    while True:
        data = bus.read_i2c_block_data(ADDRESS, 0x01, 6)

        x = convert_to_g(data[0], data[1])
        y = convert_to_g(data[2], data[3])
        z = convert_to_g(data[4], data[5])

        print(f"X: {x:+.3f} g   Y: {y:+.3f} g   Z: {z:+.3f} g")
        time.sleep(0.2)

except KeyboardInterrupt:
    print("\nStopped.")

finally:
    bus.close()