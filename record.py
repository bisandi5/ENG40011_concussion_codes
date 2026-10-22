import csv
import time
from datetime import datetime
from pathlib import Path

import max30102


def record_ppg(duration=60):
    sensor = max30102.MAX30102()
    samples = []
    overflow_detected = False

    try:
        # Pause the sensor while changing its configuration.
        sensor.bus.write_byte_data(sensor.address, 0x09, 0x80)

        # No sample averaging; FIFO rollover disabled.
        sensor.bus.write_byte_data(sensor.address, 0x08, 0x0F)

        # Keep the existing 100 Hz optical sampling configuration.
        sensor.bus.write_byte_data(sensor.address, 0x0A, 0x27)

        print("Place your finger gently on the sensor.")
        print("Support your hand and keep still.")

        for number in range(3, 0, -1):
            print(number)
            time.sleep(1)

        # Clear any previous buffered readings.
        for register in (0x04, 0x05, 0x06):
            sensor.bus.write_byte_data(sensor.address, register, 0)

        start = time.monotonic()

        # Red + infrared measurement mode.
        sensor.bus.write_byte_data(sensor.address, 0x09, 0x03)

        print("Recording for 60 seconds...")

        while time.monotonic() - start < duration:
            overflow = sensor.bus.read_byte_data(
                sensor.address, 0x05
            ) & 0x1F

            if overflow:
                overflow_detected = True
                break

            available = sensor.get_data_present()

            if available == 0:
                time.sleep(0.001)
                continue

            for _ in range(available):
                red, ir = sensor.read_fifo()
                read_elapsed = time.monotonic() - start
                samples.append((len(samples), read_elapsed, red, ir))

    finally:
        try:
            sensor.shutdown()
        finally:
            sensor.bus.close()

    if not samples:
        raise RuntimeError("No samples were collected.")

    folder = Path(__file__).resolve().parent / "recordings"
    folder.mkdir(exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = folder / f"ppg_100hz_{stamp}.csv"

    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "sample_index",
            "host_read_elapsed_s",
            "red",
            "ir",
        ])
        writer.writerows(samples)

    print(f"Saved samples: {len(samples)}")
    print(f"FIFO overflow detected: {overflow_detected}")
    print(f"Saved file: {path}")
    print("Nominal signal sampling rate: 100 Hz")

    if overflow_detected:
        print("Recording incomplete: do not use for variability.")

    return path


if __name__ == "__main__":
    try:
        record_ppg()
    except KeyboardInterrupt:
        print("\nCancelled. No completed recording produced.")
    except Exception as error:
        print(f"\nRecording failed: {error}")