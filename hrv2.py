"""Live MAX30102 pulse variability prototype; no simulated readings.
Engineering quality checks below are provisional, not clinical validation.
"""
import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks

FS = 100


def calculate_variability(intervals_ms):
    if len(intervals_ms) < 3:
        raise ValueError('At least three consecutive pulse intervals required.')
    if any(not math.isfinite(v) or v <= 0 for v in intervals_ms):
        raise ValueError('Intervals must be positive and finite.')
    mean = sum(intervals_ms) / len(intervals_ms)
    return dict(interval_count=len(intervals_ms), mean_interval_ms=mean,
                pulse_rate_bpm=60000 / mean,
                rmssd_ms=math.sqrt(np.mean(np.diff(intervals_ms) ** 2)))


def record(duration=60, settling=10):
    import max30102  # Hardware imported only when recording on the Pi.
    sensor = max30102.MAX30102()
    rows = []
    reason = None
    try:
        sensor.shutdown()
        write = lambda reg, value: sensor.bus.write_byte_data(sensor.address, reg, value)
        write(0x08, 0x0F)  # No sample averaging; no FIFO rollover.
        write(0x0A, 0x27)  # 100 Hz, 18-bit red + infrared.
        for reg in (0x04, 0x05, 0x06):
            write(reg, 0)
        print('Place your finger gently over the sensor; support your hand.')
        print(f'Sensor runs for {settling}s to settle, then records {duration}s.')
        start = time.monotonic()
        last_poll = start
        last_sample = start
        settling_count = settling * FS
        target = (duration + settling) * FS
        write(0x09, 0x03)
        announced = False
        while len(rows) < target:
            now = time.monotonic()
            if now - start > duration + settling + 10:
                reason = 'Recording timed out.'
                break
            if now - last_poll > 0.15:
                reason = 'Long read delay; sample continuity is uncertain.'
                break
            last_poll = now
            if sensor.bus.read_byte_data(sensor.address, 0x05) & 0x1F:
                reason = 'FIFO overflow: samples were lost.'
                break
            count = sensor.get_data_present()
            if count == 0:
                if now - last_sample > 0.2:
                    reason = 'No samples arriving; check sensor connection.'
                    break
                time.sleep(0.001)
                continue
            for _ in range(min(count, target - len(rows))):
                # Check before each read: reading a sample resets the overflow counter.
                before = time.monotonic()
                if before - last_poll > 0.15:
                    reason = 'Long read delay; sample continuity is uncertain.'
                    break
                last_poll = before
                if sensor.bus.read_byte_data(sensor.address, 0x05) & 0x1F:
                    reason = 'FIFO overflow: samples were lost.'
                    break
                red, ir = sensor.read_fifo()
                last_sample = time.monotonic()
                rows.append((len(rows), last_sample - start, red, ir))
                if len(rows) >= settling_count and not announced:
                    print('Settling finished. Recording now; keep still.')
                    announced = True
            if reason:
                break
        sensor.shutdown()
        if sensor.bus.read_byte_data(sensor.address, 0x05) & 0x1F:
            reason = 'FIFO overflow detected at end of capture.'
    finally:
        try:
            sensor.shutdown()
        finally:
            sensor.bus.close()
    measured = rows[settling_count:]
    rate = None
    if len(measured) > 1:
        rate = (len(measured) - 1) / (measured[-1][1] - measured[0][1])
    if not reason and (rate is None or abs(rate - FS) > 5):
        reason = 'Observed read rate differs from nominal 100 Hz by over 5%.'
    metadata = dict(nominal_sample_rate_hz=FS, host_observed_rate_hz=rate,
                    settling_seconds=settling, requested_seconds=duration,
                    captured_samples=len(rows), analysis_samples=len(measured),
                    acquisition_error=reason)
    return rows, measured, metadata


def detect(values, prominence_scale=0.5):
    signal = sosfiltfilt(butter(3, [0.5, 5], fs=FS, btype='bandpass', output='sos'), values)
    # Exclude filter edges. No intervals are stitched across rejected sections.
    signal = signal[2 * FS:-2 * FS]
    scale = np.median(np.abs(signal - np.median(signal))) * 1.4826
    if scale < 5:  # ADC counts: provisional flat-signal check.
        raise ValueError('Pulse signal is too small or flat.')
    peaks, _ = find_peaks(signal, distance=int(0.3 * FS), prominence=scale * prominence_scale)
    return peaks, signal


def analyse(red, ir):
    reasons = []
    result = {'status': 'insufficient_signal_quality', 'reasons': reasons,
              'pulse_rate_bpm': None, 'rmssd_ms': None,
              'measurement': 'PPG pulse-rate variability (PRV)',
              'quality_rules': 'provisional engineering checks; not validated',
              'score': None}
    intervals = []
    try:
        if len(ir) < 55 * FS:
            raise ValueError('Less than 55 seconds of recorded data.')
        for values in (red, ir):
            if not np.isfinite(values).all():
                raise ValueError('Non-finite raw samples.')
            if np.mean((values <= 0) | (values >= 262143)) > 0.001:
                raise ValueError('Raw signal is clipped or zero.')
        peaks, filtered = detect(ir)
        red_peaks, red_filtered = detect(red)
        if len(peaks) < 25 or len(red_peaks) < 25:
            raise ValueError('Too few candidate pulses.')
        intervals = (np.diff(peaks) * 1000 / FS).tolist()
        median_interval = float(np.median(intervals))
        # Broad detector limits, not definitions of normal/abnormal health.
        if any(v < 300 or v > 2000 for v in intervals):
            reasons.append('Candidate intervals outside the detector range.')
        if np.any(np.abs(np.asarray(intervals) - median_interval) > 0.25 * median_interval):
            reasons.append('Inconsistent pulse spacing: artifacts or missed/extra beats possible.')
        def agreement(a, b):
            nearest = np.min(np.abs(a[:, None] - b[None, :]), axis=1)
            return float(np.mean(nearest <= 8))  # 80 ms channel timing tolerance.
        match = min(agreement(peaks, red_peaks), agreement(red_peaks, peaks))
        result['red_ir_peak_agreement'] = match
        if match < 0.90:
            reasons.append('Red and infrared pulse detections disagree.')
        if np.corrcoef(filtered, red_filtered)[0, 1] < 0.8:
            reasons.append('Red and infrared pulse waveforms disagree.')
        amplitudes = [np.std(filtered[i:i + 5 * FS]) for i in range(0, len(filtered) - 5 * FS + 1, 5 * FS)]
        if min(amplitudes) <= 0 or max(amplitudes) > 5 * min(amplitudes):
            reasons.append('Large pulse-amplitude changes suggest unstable contact.')
        stats = calculate_variability(intervals)
        # A different reasonable peak threshold should not change the answer greatly.
        for factor in (0.3, 0.8):
            other, _ = detect(ir, factor)
            other_intervals = np.diff(other) * 1000 / FS
            if len(other_intervals) < 3:
                reasons.append('Pulse detection is sensitive to its threshold.')
                break
            other_stats = calculate_variability(other_intervals.tolist())
            if abs(len(other) - len(peaks)) > 1 or abs(other_stats['rmssd_ms'] - stats['rmssd_ms']) > max(10, 0.2 * stats['rmssd_ms']):
                reasons.append('Pulse detection is sensitive to its threshold.')
                break
        result['candidate_pulse_count'] = len(peaks)
        if not reasons:
            result.update(stats)
            result['status'] = 'prototype_quality_checks_passed'
    except ValueError as exc:
        reasons.append(str(exc))
    return result, intervals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, help='Analyse an existing red/ir CSV without hardware')
    args = parser.parse_args()
    output = Path(__file__).resolve().parent / 'recordings'
    output.mkdir(exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    prefix = output / f'prv_{stamp}'
    if args.csv:
        with args.csv.open(newline='') as file:
            data = list(csv.DictReader(file))
        if data and 'phase' in data[0]:
            data = [r for r in data if r['phase'] == 'recording']
        if data and 'sample_index' in data[0]:
            indices = [int(r['sample_index']) for r in data]
            if any(b - a != 1 for a, b in zip(indices, indices[1:])):
                raise ValueError('CSV sample indices are not contiguous.')
        measured = [(i, 0, float(r['red']), float(r['ir'])) for i, r in enumerate(data)]
        metadata = {'source_csv': str(args.csv), 'nominal_sample_rate_hz': FS,
                    'note': 'Assumes contiguous 100 Hz input; original acquisition must be checked.'}
    else:
        rows, measured, metadata = record()
        with prefix.with_suffix('.csv').open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['sample_index', 'host_read_elapsed_s', 'red', 'ir', 'phase'])
            for row in rows:
                writer.writerow([*row, 'settling' if row[0] < 10 * FS else 'recording'])
        print(f'Raw data saved: {prefix.with_suffix(".csv")}')
    if metadata.get('acquisition_error'):
        result = dict(status='recording_failed', reasons=[metadata['acquisition_error']], pulse_rate_bpm=None, rmssd_ms=None, score=None)
        intervals = []
    else:
        red = np.asarray([r[2] for r in measured], dtype=float)
        ir = np.asarray([r[3] for r in measured], dtype=float)
        result, intervals = analyse(red, ir)
    result['acquisition'] = metadata
    prefix.with_suffix('.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    with Path(str(prefix) + '_candidate_intervals.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['interval_index', 'candidate_interval_ms', 'accepted_recording'])
        for i, interval in enumerate(intervals):
            writer.writerow([i, interval, result['status'] == 'prototype_quality_checks_passed'])
    print('\nRESULT:', result['status'])
    if result.get('rmssd_ms') is not None:
        print(f"Pulse rate: {result['pulse_rate_bpm']:.1f} BPM")
        print(f"Pulse variability (RMSSD): {result['rmssd_ms']:.2f} ms")
        print('Prototype estimate; quality checks do not establish accuracy.')
    else:
        print('Pulse rate / RMSSD withheld:')
        for reason in result['reasons']:
            print(' -', reason)
    print('Concussion score: not calculated.')
    print('Results saved:', prefix.with_suffix('.json'))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled. No completed result generated.')
    except Exception as exc:
        print(f'Error: {exc}')
        raise SystemExit(1)
