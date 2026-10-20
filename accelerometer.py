import csv
import math
import time
from datetime import datetime
from pathlib import Path

import smbus


ADDRESS = 0x1C
DURATION_SECONDS = 20
READ_INTERVAL = 0.02  # Aim to read about 50 times per second.


def convert_to_g(msb, lsb):
    # Preserve the 12-bit conversion used in your existing code.
    value = ((msb << 8) | lsb) >> 4

    if value >= 2048:
        value -= 4096

    return value / 1024.0


def standard_deviation(values):
    mean = sum(values) / len(values)

    variance = sum(
        (value - mean) ** 2 for value in values
    ) / len(values)

    return math.sqrt(variance)


def record_balance(duration=DURATION_SECONDS):
    bus = smbus.SMBus(1)
    samples = []

    try:
        # Standby, +/-2 g range, then active at 100 Hz.
        bus.write_byte_data(ADDRESS, 0x2A, 0x00)
        bus.write_byte_data(ADDRESS, 0x0E, 0x00)
        bus.write_byte_data(ADDRESS, 0x2A, 0x19)
        time.sleep(0.1)

        print("Prototype balance-task recording")
        print("Keep the sensor securely fixed.")
        print("Recording starts after the countdown.")

        for number in range(3, 0, -1):
            print(number)
            time.sleep(1)

        print("Recording...")
        start = time.monotonic()
        next_read = start

        while time.monotonic() - start < duration:
            data = bus.read_i2c_block_data(ADDRESS, 0x01, 6)
            elapsed = time.monotonic() - start

            x = convert_to_g(data[0], data[1])
            y = convert_to_g(data[2], data[3])
            z = convert_to_g(data[4], data[5])

            samples.append((elapsed, x, y, z))

            next_read += READ_INTERVAL
            remaining = next_read - time.monotonic()

            if remaining > 0:
                time.sleep(remaining)
            else:
                # Avoid a burst of catch-up reads after a delay.
                next_read = time.monotonic()

        print("Recording complete.")

    finally:
        # Close the bus even if recording fails or is interrupted.
        try:
            bus.write_byte_data(ADDRESS, 0x2A, 0x00)
        finally:
            bus.close()

    if len(samples) < 2:
        raise RuntimeError("Too few samples to calculate variability.")

    # Save the original measurements.
    output_folder = Path(__file__).resolve().parent / "recordings"
    output_folder.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_file = output_folder / f"balance_{timestamp}.csv"

    with output_file.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["time_seconds", "x_g", "y_g", "z_g"])
        writer.writerows(samples)

    # Variability around each axis's own mean.
    x_sd = standard_deviation([row[1] for row in samples])
    y_sd = standard_deviation([row[2] for row in samples])
    z_sd = standard_deviation([row[3] for row in samples])

    combined = math.sqrt(x_sd ** 2 + y_sd ** 2 + z_sd ** 2)

    recorded_span = samples[-1][0] - samples[0][0]
    average_rate = (
        (len(samples) - 1) / recorded_span
        if recorded_span > 0 else 0
    )

    largest_gap = max(
        samples[index][0] - samples[index - 1][0]
        for index in range(1, len(samples))
    )

    result = {
        "status": "complete",
        "sample_count": len(samples),
        "recorded_span_seconds": recorded_span,
        "average_read_rate_hz": average_rate,
        "largest_sample_gap_seconds": largest_gap,
        "x_sd_g": x_sd,
        "y_sd_g": y_sd,
        "z_sd_g": z_sd,
        "combined_variability_g": combined,
        "assessment_score": None,
        "file": str(output_file),
    }

    return result


if __name__ == "__main__":
    try:
        result = record_balance()

        print("\nRESULTS")
        print(f"Samples: {result['sample_count']}")
        print(
            f"Average read rate: "
            f"{result['average_read_rate_hz']:.1f} Hz"
        )
        print(
            f"Largest sample gap: "
            f"{result['largest_sample_gap_seconds']:.3f} seconds"
        )
        print(f"X variability: {result['x_sd_g']:.5f} g")
        print(f"Y variability: {result['y_sd_g']:.5f} g")
        print(f"Z variability: {result['z_sd_g']:.5f} g")
        print(
            f"Combined variability: "
            f"{result['combined_variability_g']:.5f} g"
        )
        print(f"Saved to: {result['file']}")
        print("Concussion assessment score: not implemented")

    except KeyboardInterrupt:
        print("\nRecording cancelled. No result produced.")

    except Exception as error:
        print(f"\nRecording failed: {error}")