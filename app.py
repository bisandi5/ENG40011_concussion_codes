"""Live MAX30102 pulse variability prototype; no simulated readings.
Version 2: infrared-led detection, section checks and diagnostic graph.
Engineering checks are provisional. A result is not validated HRV or a diagnosis.
"""
import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks, welch

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


def record(duration=60, settling=10, stop_event=None, progress=None):
    import max30102  # Hardware imported only when recording on the Pi.
    sensor = max30102.MAX30102()
    rows = []
    reason = None
    last_progress = 0.0
    def report(force=False):
        nonlocal last_progress
        now = time.monotonic()
        if progress is None or (not force and now - last_progress < 0.25):
            return
        last_progress = now
        count = len(rows)
        progress(dict(stage="settling" if count < settling * FS else "recording",
                      captured_samples=count, recorded_samples=max(0, count - settling * FS),
                      sample_elapsed_seconds=count / FS,
                      phase_elapsed_seconds=count / FS if count < settling * FS else (count - settling * FS) / FS,
                      latest_red=rows[-1][2] if rows else None,
                      latest_ir=rows[-1][3] if rows else None))
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
        report(force=True)
        while len(rows) < target:
            if stop_event is not None and stop_event.is_set():
                reason = "Cancelled by user."
                break
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
                if stop_event is not None and stop_event.is_set():
                    reason = "Cancelled by user."
                    break
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
                report()
                if len(rows) >= settling_count and not announced:
                    print('Settling finished. Recording now; keep still.')
                    announced = True
            if reason:
                break
        report(force=True)
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


# These are engineering thresholds for a resting fingertip prototype.
# They are NOT normal/abnormal HRV limits and have not been clinically validated.
VERSION = 'finger-prv-2'
MIN_RUN_SECONDS = 30.0
MIN_INTERVALS = 30
SECTION_SECONDS = 10


def robust_scale(x):
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def period_estimate(x):
    """Spectral estimate guides a local autocorrelation search (samples)."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    freq, power = welch(x, FS, nperseg=min(len(x), 1000))
    valid = (freq >= 0.65) & (freq <= 3.0)
    if not valid.any() or power[valid].max() <= 0:
        return None, 0.0
    spectral_period = FS / float(freq[valid][np.argmax(power[valid])])
    lags = np.arange(max(30, int(0.8 * spectral_period)),
                     min(len(x) // 2, int(1.2 * spectral_period)) + 1)
    correlations = []
    for lag in lags:
        a, b = x[:-lag], x[lag:]
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            correlations.append(0.0)
        else:
            correlations.append(float(np.corrcoef(a, b)[0, 1]))
    if not len(lags):
        return None, 0.0
    best = int(np.argmax(correlations))
    return int(lags[best]), correlations[best]


def matched(peaks, alternative, tolerance=4):
    if len(alternative) == 0:
        return np.zeros(len(peaks), dtype=bool)
    return np.min(abs(peaks[:, None] - alternative[None, :]), axis=1) <= tolerance


def contiguous_runs(mask):
    padded = np.r_[False, mask, False].astype(int)
    return list(zip(np.flatnonzero(np.diff(padded) == 1),
                    np.flatnonzero(np.diff(padded) == -1)))


def analyse(red, ir):
    """No interval correction/interpolation and no joining across rejected gaps."""
    red, ir = np.asarray(red, dtype=float), np.asarray(ir, dtype=float)
    result = dict(version=VERSION, status='insufficient_signal_quality',
                  measurement='Fingertip PPG pulse-rate variability (PRV)',
                  pulse_rate_bpm=None, rmssd_ms=None, score=None,
                  analysed_seconds=len(ir) / FS, selected_seconds=0.0,
                  reasons=[], warnings=[], sections=[],
                  quality_rules='Provisional engineering rules; accuracy not established',
                  minimum_contiguous_seconds=MIN_RUN_SECONDS,
                  minimum_intervals=MIN_INTERVALS)
    diagnostics = dict(peaks=np.array([], dtype=int), intervals=[], filtered=np.array([]),
                       raw_ir=ir, raw_red=red, selected_start=None, selected_end=None)
    if len(ir) != len(red) or len(ir) < 55 * FS:
        result['reasons'].append('Need at least 55 seconds of paired red/infrared data.')
        return result, diagnostics
    if not np.isfinite(ir).all() or not np.isfinite(red).all():
        result['reasons'].append('Missing/non-finite samples. Do not join across missing data.')
        return result, diagnostics
    # Fixed processing pipeline, used unchanged for all source recordings.
    sos = butter(3, [0.7, 3.5], fs=FS, btype='bandpass', output='sos')
    filtered = sosfiltfilt(sos, ir)
    red_filtered = sosfiltfilt(sos, red)
    diagnostics['filtered'] = filtered
    good_samples = np.zeros(len(ir), dtype=bool)
    all_peaks = []
    peak_stability = {}
    sample_period = np.zeros(len(ir))
    size = SECTION_SECONDS * FS
    for core_start in range(0, len(ir), size):
        core_end = min(len(ir), core_start + size)
        # Filter-edge exclusion; process adjacent context to avoid bin-edge peaks.
        start = max(2 * FS, core_start)
        end = min(len(ir) - 2 * FS, core_end)
        section = dict(start_s=start / FS, end_s=end / FS, accepted=False, reasons=[])
        if end <= start:
            continue
        left, right = max(2 * FS, start - FS), min(len(ir) - 2 * FS, end + FS)
        x = filtered[left:right]
        scale = robust_scale(x)
        period, periodicity = period_estimate(x)
        section.update(periodicity=round(float(periodicity), 3), pulse_scale_adc=round(scale, 2))
        raw = ir[start:end]
        if np.any((raw <= 0) | (raw >= 262143)):
            section['reasons'].append('Clipped/zero infrared samples.')
        if scale < 5:
            section['reasons'].append('Infrared waveform is too small or flat.')
        if period is None or periodicity < 0.5:
            section['reasons'].append('No sufficiently repeatable pulse pattern.')
        # Red informs the report, but weak red peaks do not veto clean IR.
        red_segment = red_filtered[start:end]
        ir_segment = filtered[start:end]
        correlation = None
        if np.std(red_segment) > 1e-8 and np.std(ir_segment) > 1e-8:
            correlation = float(np.corrcoef(red_segment, ir_segment)[0, 1])
        section['red_ir_correlation'] = correlation
        if period is not None and scale >= 5:
            distance = max(30, int(0.60 * period))
            candidates = []
            for factor in (0.3, 0.5, 0.8):
                p, _ = find_peaks(x, distance=distance, prominence=factor * scale)
                candidates.append(p + left)
            p = candidates[1]
            stable = matched(p, candidates[0]) & matched(p, candidates[2])
            core = (p >= start) & (p < end)
            for position, is_stable in zip(p[core], stable[core]):
                all_peaks.append(int(position))
                peak_stability[int(position)] = bool(is_stable)
            section['candidate_pulses'] = int(core.sum())
            section['stable_pulse_fraction'] = float(np.mean(stable[core])) if core.any() else 0.0
            if not core.any() or np.mean(stable[core]) < 0.9:
                section['reasons'].append('Pulse detections change with threshold.')
            sample_period[start:end] = period
        section['accepted'] = not section['reasons']
        if section['accepted']:
            good_samples[start:end] = True
        result['sections'].append(section)
    peaks = np.array(sorted(set(all_peaks)), dtype=int)
    diagnostics['peaks'] = peaks
    result['candidate_pulse_count'] = int(len(peaks))
    # Two peak-search contexts can disagree near a section boundary.
    # Mark conflicting neighbours as uncertain; do not silently merge them.
    conflicts = set()
    for a, b in zip(peaks[:-1], peaks[1:]):
        if b - a < 30:
            conflicts.update((int(a), int(b)))
    intervals = []
    for index, (a, b) in enumerate(zip(peaks[:-1], peaks[1:])):
        length = int(b - a)
        expected = float(np.median(sample_period[a:b])) if length else 0
        reasons = []
        if not good_samples[a:b + 1].all():
            reasons.append('Interval crosses an uncertain signal section.')
        if not peak_stability.get(int(a)) or not peak_stability.get(int(b)):
            reasons.append('A pulse endpoint is threshold-sensitive.')
        if int(a) in conflicts or int(b) in conflicts:
            reasons.append('Conflicting detections near a section boundary.')
        if expected <= 0 or not (0.60 * expected <= length <= 1.45 * expected):
            reasons.append('Interval inconsistent with local pulse period.')
        if not 30 <= length <= 180:
            reasons.append('Interval outside supported detector range.')
        intervals.append(dict(interval_index=index, start_s=float(a / FS), end_s=float(b / FS),
                              interval_ms=float(length * 1000 / FS),
                              accepted=not reasons, selected=False, reasons=reasons))
    # Flag local outliers without deleting them and stitching their neighbours.
    lengths = np.array([i['interval_ms'] for i in intervals])
    for index, item in enumerate(intervals):
        neighbourhood = lengths[max(0, index - 4):index + 5]
        typical = float(np.median(neighbourhood))
        if typical > 0 and abs(item['interval_ms'] - typical) > 0.25 * typical:
            item['accepted'] = False
            item['reasons'].append('Interval differs from neighbouring pulse spacing.')
    diagnostics['intervals'] = intervals
    runs = contiguous_runs([i['accepted'] for i in intervals])
    if runs:
        first, last = max(runs, key=lambda run: sum(i['interval_ms'] for i in intervals[run[0]:run[1]]))
        chosen = intervals[first:last]
        seconds = sum(i['interval_ms'] for i in chosen) / 1000
        result['longest_usable_run_seconds'] = round(seconds, 2)
        if seconds >= MIN_RUN_SECONDS and len(chosen) >= MIN_INTERVALS:
            for item in chosen:
                item['selected'] = True
            result.update(calculate_variability([i['interval_ms'] for i in chosen]))
            result.update(status='prototype_segment_estimate', selected_seconds=round(seconds, 2),
                          selected_start_s=chosen[0]['start_s'], selected_end_s=chosen[-1]['end_s'])
            diagnostics.update(selected_start=chosen[0]['start_s'], selected_end=chosen[-1]['end_s'])
            result['warnings'].append('Result uses only the selected continuous section, not the complete recording.')
    if result['rmssd_ms'] is None:
        result['reasons'].append(f'No continuous accepted section of at least {MIN_RUN_SECONDS:g}s and {MIN_INTERVALS} intervals.')
    if any(s['red_ir_correlation'] is not None and s['red_ir_correlation'] < 0.8 for s in result['sections']):
        result['warnings'].append('Red/infrared agreement is weak in some sections. Check the diagnostic graph.')
    result['warnings'].append('Checks cannot prove beat accuracy. Compare with reference beat intervals before relying on RMSSD.')
    return result, diagnostics


def save_plot(prefix, diagnostics, result):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return 'Graph unavailable: install python3-matplotlib.'
    ir = diagnostics['raw_ir']
    filtered = diagnostics['filtered']
    count = len(ir)
    if not count:
        return 'No samples to plot.'
    rows = 1 + math.ceil(count / (10 * FS))
    fig, axes = plt.subplots(rows, 1, figsize=(13, 2.4 * rows))
    t = np.arange(count) / FS
    axes[0].plot(t, ir, color='#176b87', label='Raw infrared')
    axes[0].set_title(f"{VERSION}: {result['status']} | no concussion score")
    axes[0].set_ylabel('ADC counts')
    axes[0].legend(loc='upper right')
    for row, start in enumerate(range(0, count, 10 * FS), 1):
        end = min(start + 10 * FS, count)
        ax = axes[row]
        if len(filtered) == count:
            ax.plot(t[start:end], filtered[start:end], color='#18334e', lw=1)
            for section in result['sections']:
                if not section['accepted']:
                    ax.axvspan(section['start_s'], section['end_s'], color='#b42336', alpha=0.1)
            for item in diagnostics['intervals']:
                if item['selected']:
                    ax.axvspan(item['start_s'], item['end_s'], color='#16834b', alpha=0.18)
            peaks = diagnostics['peaks']
            p = peaks[(peaks >= start) & (peaks < end)]
            ax.scatter(p / FS, filtered[p], s=14, color='#bd6b00', label='Candidate pulse')
        ax.set_xlim(start / FS, end / FS)
        ax.set_ylabel('Filtered ADC')
        ax.set_xlabel('Seconds after settling')
        ax.grid(alpha=0.2)
    fig.text(0.02, 0.005, 'Orange dots: candidate pulses. Pale red: rejected sections. Green: continuous section used for estimate. Unshaded does not mean validated.', fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    path = Path(str(prefix) + '_pulses.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, help='Reanalyse an existing contiguous 100 Hz red/ir CSV; no new recording')
    args = parser.parse_args()
    output = Path(__file__).resolve().parent / 'recordings'
    output.mkdir(exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    prefix = output / f'prv_v2_{stamp}'
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
        metadata = dict(source_csv=str(args.csv.resolve()), nominal_sample_rate_hz=FS)
        original_json = args.csv.with_suffix('.json')
        if original_json.exists():
            original = json.loads(original_json.read_text())
            acquisition = original.get('acquisition', {})
            if acquisition.get('nominal_sample_rate_hz', FS) != FS:
                raise ValueError('This analyser supports only nominal 100 Hz CSV data.')
            metadata.update(acquisition)
        else:
            metadata['note'] = 'Original capture metadata unavailable; assumes contiguous 100 Hz samples.'
        print('Reanalysing saved CSV. No new sensor recording is made.')
    else:
        rows, measured, metadata = record()
        raw_file = prefix.with_suffix('.csv')
        with raw_file.open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['sample_index', 'host_read_elapsed_s', 'red', 'ir', 'phase'])
            for row in rows:
                writer.writerow([*row, 'settling' if row[0] < metadata['settling_seconds'] * FS else 'recording'])
        print('Raw recording saved:', raw_file)
    red = np.asarray([r[2] for r in measured], dtype=float)
    ir = np.asarray([r[3] for r in measured], dtype=float)
    result, diagnostics = analyse(red, ir)
    if metadata.get('acquisition_error'):
        result.update(status='recording_failed', pulse_rate_bpm=None, rmssd_ms=None, selected_seconds=0)
        result['reasons'].insert(0, metadata['acquisition_error'])
        for interval in diagnostics['intervals']:
            interval['selected'] = False
    result['acquisition'] = metadata
    result['placement'] = 'finger'
    result['graph'] = save_plot(prefix, diagnostics, result)
    prefix.with_suffix('.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    intervals_file = Path(str(prefix) + '_intervals.csv')
    with intervals_file.open('w', newline='') as file:
        fields = ['interval_index', 'start_s', 'end_s', 'interval_ms', 'accepted', 'selected', 'reasons']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in diagnostics['intervals']:
            writer.writerow({**item, 'reasons': '; '.join(item['reasons'])})
    print('\nRESULT:', result['status'])
    if result['rmssd_ms'] is not None:
        print(f"Selected continuous section: {result['selected_start_s']:.2f}–{result['selected_end_s']:.2f}s ({result['selected_seconds']:.2f}s)")
        print(f"Pulse-rate estimate: {result['pulse_rate_bpm']:.1f} BPM")
        print(f"PRV RMSSD estimate: {result['rmssd_ms']:.2f} ms")
        print('Prototype segment estimate; accuracy is not established.')
    else:
        for reason in result['reasons']:
            print(' -', reason)
        print('Pulse rate and RMSSD withheld.')
        print('Longest usable run:', result.get('longest_usable_run_seconds', 0), 'seconds')
    for section in result['sections']:
        if not section['accepted']:
            print(f"Check {section['start_s']:.0f}-{section['end_s']:.0f}s: " + '; '.join(section['reasons']))
    print('Concussion score: not calculated.')
    print('JSON:', prefix.with_suffix('.json'))
    print('Intervals:', intervals_file)
    print('Graph:', result['graph'])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled. No completed result generated.')
    except Exception as exc:
        print(f'Error: {exc}')
        raise SystemExit(1)
