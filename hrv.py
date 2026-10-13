import time
import numpy as np
import max30102
import hrcalc


def heart_rate_monitor(duration=60):
    sensor = max30102.MAX30102()
    start_time = time.monotonic()

    print("Place your fingertip gently on the sensor.")
    print("Keep still. Press Ctrl+C to stop.")

    try:
        while time.monotonic() - start_time < duration:
            # Collect a block of red and infrared readings.
            red, ir = sensor.read_sequential(amount=100)

            if len(ir) < 100 or len(red) < 100:
                continue

            red = red[-100:]
            ir = ir[-100:]

            print(f"Raw RED: {red[-1]} | Raw IR: {ir[-1]}")

            ir_mean = np.mean(ir)
            ir_std = np.std(ir)

            # Initial finger-detection rule; needs testing.
            if ir_mean < 50000 + 0.5 * ir_std:
                print("Signal too low—check finger placement.")
                continue

            hr, hr_valid, spo2, spo2_valid = (
                hrcalc.calc_hr_and_spo2(ir, red)
            )

            if hr_valid:
                print(f"Heart rate estimate: {hr:.1f} BPM")
            else:
                print("Heart rate: waiting for a stable signal.")

            if spo2_valid:
                print(f"SpO2 estimate: {spo2:.1f}%")
            else:
                print("SpO2: waiting for a stable signal.")

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        sensor.shutdown()
        print("Sensor stopped.")


if __name__ == "__main__":
    heart_rate_monitor(duration=60)