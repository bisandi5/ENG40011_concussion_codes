"""20-second accelerometer capture, preserving the project's original conversion.
No balance diagnosis or assessment score is calculated.
"""
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

ADDRESS = 0x1C
DURATION_SECONDS = 20
READ_INTERVAL = 0.02  # Host read target 50 Hz; sensor configured for 100 Hz.


def convert_to_g(msb, lsb):
    value = ((msb << 8) | lsb) >> 4
    if value >= 2048:
        value -= 4096
    return value / 1024.0


def standard_deviation(values):
    if not values:
        raise ValueError('No samples.')
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def record_balance(duration=DURATION_SECONDS, stop_event=None, progress=None,
                   output_folder=None, run_id=None):
    # Lazy hardware import lets Flask start even if a sensor dependency is missing.
    import smbus
    samples = []
    error = None
    cancelled = False
    bus = None
    last_progress = 0.0

    def stopped():
        return stop_event is not None and stop_event.is_set()

    def wait(seconds):
        if seconds > 0:
            if stop_event is not None:
                return stop_event.wait(seconds)
            time.sleep(seconds)
        return stopped()

    def report(stage, force=False, countdown=None):
        nonlocal last_progress
        now = time.monotonic()
        if progress is None or (not force and now - last_progress < 0.1):
            return
        last_progress = now
        last = samples[-1] if samples else None
        progress(dict(stage=stage, countdown=countdown, sample_count=len(samples),
                      recording_elapsed_seconds=last[0] if last else 0.0,
                      x_g=last[1] if last else None, y_g=last[2] if last else None,
                      z_g=last[3] if last else None))

    try:
        bus = smbus.SMBus(1)
        bus.write_byte_data(ADDRESS, 0x2A, 0x00)
        bus.write_byte_data(ADDRESS, 0x0E, 0x00)
        bus.write_byte_data(ADDRESS, 0x2A, 0x19)
        if wait(0.1):
            cancelled = True
        print('Prototype movement recording. Keep sensor securely fixed.')
        for number in range(3, 0, -1):
            if cancelled or stopped():
                cancelled = True
                break
            print(number)
            report('countdown', force=True, countdown=number)
            if wait(1):
                cancelled = True
                break
        if not cancelled:
            print('Recording...')
            start = time.monotonic()
            next_read = start
            report('recording', force=True)
            while time.monotonic() - start < duration:
                if stopped():
                    cancelled = True
                    break
                data = bus.read_i2c_block_data(ADDRESS, 0x01, 6)
                if len(data) != 6:
                    raise RuntimeError('Incomplete accelerometer read.')
                elapsed = time.monotonic() - start
                x = convert_to_g(data[0], data[1])
                y = convert_to_g(data[2], data[3])
                z = convert_to_g(data[4], data[5])
                samples.append((elapsed, x, y, z))
                report('recording')
                next_read += READ_INTERVAL
                remaining = next_read - time.monotonic()
                if remaining > 0:
                    if wait(remaining):
                        cancelled = True
                        break
                else:
                    next_read = time.monotonic()  # No catch-up burst.
    except KeyboardInterrupt:
        cancelled = True
    except Exception as exc:
        error = str(exc)
    finally:
        if bus is not None:
            try:
                bus.write_byte_data(ADDRESS, 0x2A, 0x00)
            except Exception as exc:
                error = error or f'Could not return sensor to standby: {exc}'
            finally:
                try:
                    bus.close()
                except Exception as exc:
                    error = error or f'Could not close I2C bus: {exc}'
    cancelled = cancelled or stopped()
    report('stopping' if cancelled else 'saving', force=True)
    folder = Path(output_folder) if output_folder is not None else Path(__file__).resolve().parent / 'recordings'
    folder.mkdir(parents=True, exist_ok=True)
    stamp = run_id or datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output_file = folder / f'balance_{stamp}.csv'
    with output_file.open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['time_seconds', 'x_g', 'y_g', 'z_g'])
        writer.writerows(samples)
    span = samples[-1][0] - samples[0][0] if len(samples) >= 2 else 0.0
    rate = (len(samples) - 1) / span if span > 0 else 0.0
    gaps = [samples[i][0] - samples[i-1][0] for i in range(1, len(samples))]
    if not cancelled and error is None and len(samples) < 2:
        error = 'Too few samples to calculate variability.'
    result = dict(status='cancelled' if cancelled else 'error' if error else 'complete',
                  error=error, sample_count=len(samples), requested_seconds=duration,
                  recorded_span_seconds=span, average_read_rate_hz=rate,
                  largest_sample_gap_seconds=max(gaps) if gaps else None,
                  x_sd_g=None, y_sd_g=None, z_sd_g=None, combined_variability_g=None,
                  assessment_score=None, file=str(output_file),
                  device_address='0x1C', sensor_configured_rate_hz=100,
                  host_target_read_rate_hz=50, warnings=[],
                  conversion='Original signed 12-bit +/-2g conversion, 1024 counts/g; calibration unverified')
    if result['status'] == 'complete':
        x_sd, y_sd, z_sd = (standard_deviation([s[i] for s in samples]) for i in (1, 2, 3))
        result.update(x_sd_g=x_sd, y_sd_g=y_sd, z_sd_g=z_sd,
                      combined_variability_g=math.sqrt(x_sd ** 2 + y_sd ** 2 + z_sd ** 2))
        if rate < 40 or (gaps and max(gaps) > 0.1):
            result['warnings'].append('Read timing was uneven. Inspect timestamps before comparing trials.')
        if span < duration - 0.1:
            result['warnings'].append('Recorded span is shorter than requested.')
        if any(abs(v) >= 1.99 for s in samples for v in s[1:]):
            result['warnings'].append('Some readings are close to the configured range limit; clipping is possible.')
    json_file = output_file.with_suffix('.json')
    result['result_file'] = str(json_file)
    json_file.write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    try:
        result = record_balance()
        print('\nRESULT:', result['status'])
        print('Samples:', result['sample_count'])
        if result['status'] == 'complete':
            print(f"Average read rate: {result['average_read_rate_hz']:.1f} Hz")
            print(f"Largest sample gap: {result['largest_sample_gap_seconds']:.3f} seconds")
            for axis in ('x', 'y', 'z'):
                print(f"{axis.upper()} variability: {result[axis + '_sd_g']:.5f} g")
            print(f"Combined variability: {result['combined_variability_g']:.5f} g")
        elif result['error']:
            print('Error:', result['error'])
        print('Raw data:', result['file'])
        print('Details:', result['result_file'])
        print('Assessment score: not calculated.')
    except Exception as exc:
        print('Recording failed:', exc)
        raise SystemExit(1)
