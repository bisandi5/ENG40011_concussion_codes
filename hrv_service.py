"""One HRV session at a time. Experimental signal-quality coverage scoring; no simulated hardware."""
import csv
import json
from pathlib import Path
import threading
import time
from uuid import uuid4
from datetime import datetime


class HRVService:
    def __init__(self, output_dir, module_loader=None):
        self.output_dir = Path(output_dir)
        self.module_loader = module_loader or self._load_module
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.files = {}
        self.state = dict(running=False, stage='idle', run_id=None, error=None,
                          result=None, downloads={}, duration_seconds=20, settling_seconds=10,
                          captured_samples=0, recorded_samples=0, latest_red=None, latest_ir=None,
                          elapsed_seconds=0, phase_elapsed_seconds=0, sample_elapsed_seconds=0)
        self.started = None

    @staticmethod
    def _load_module():
        import hrv2
        return hrv2

    def snapshot(self):
        with self.lock:
            state = dict(self.state)
            if state['running'] and self.started is not None:
                state['elapsed_seconds'] = round(time.monotonic() - self.started, 1)
            # All state is JSON-compatible; copy nested values before releasing lock.
            return json.loads(json.dumps(state, allow_nan=False))

    def start(self):
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                return False
            self.stop_event.clear()
            self.started = time.monotonic()
            run_id = datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid4().hex[:12]
            self.state.update(running=True, stage='starting', run_id=run_id, error=None,
                              result=None, downloads={}, captured_samples=0, recorded_samples=0,
                              latest_red=None, latest_ir=None, elapsed_seconds=0,
                              phase_elapsed_seconds=0, sample_elapsed_seconds=0)
            self.thread = threading.Thread(target=self._run, args=(run_id,), daemon=True)
            self.thread.start()
            return True

    def stop(self):
        with self.lock:
            if self.state['running']:
                self.stop_event.set()
                self.state['stage'] = 'stopping'

    def _progress(self, update):
        with self.lock:
            self.state.update(update)
            if self.stop_event.is_set():
                self.state['stage'] = 'stopping'

    def _publish_files(self, run_id, files):
        with self.lock:
            self.files[run_id] = dict(files)
            self.state['downloads'] = {kind: f'/api/hrv/files/{run_id}/{kind}' for kind in files}

    def file(self, run_id, kind):
        with self.lock:
            path = self.files.get(run_id, {}).get(kind)
        return path if path is not None and path.is_file() else None

    def _run(self, run_id):
        files = {}
        try:
            hrv = self.module_loader()
            self.output_dir.mkdir(parents=True, exist_ok=True)
            prefix = self.output_dir / f'prv_web_{run_id}'
            rows, measured, metadata = hrv.record(duration=self.state['duration_seconds'], settling=self.state['settling_seconds'],
                stop_event=self.stop_event, progress=self._progress)
            raw_file = prefix.with_suffix('.csv')
            with raw_file.open('w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['sample_index', 'host_read_elapsed_s', 'red', 'ir', 'phase'])
                for row in rows:
                    writer.writerow([*row, 'settling' if row[0] < metadata['settling_seconds'] * hrv.FS else 'recording'])
            files['raw'] = raw_file
            self._publish_files(run_id, files)
            failed = metadata.get('acquisition_error')
            if self.stop_event.is_set() or failed:
                result = dict(status='cancelled' if self.stop_event.is_set() else 'recording_failed',
                              pulse_rate_bpm=None, rmssd_ms=None, selected_seconds=0,
                              reasons=['Cancelled by user.'] if self.stop_event.is_set() else [failed],
                              acquisition=metadata, score=None, signal_quality_score=None,
                              score_type='signal_quality_coverage')
            else:
                self._progress(dict(stage='analysing'))
                # The existing analyser accepts array-like inputs; it converts them itself.
                result, diagnostics = hrv.analyse([r[2] for r in measured], [r[3] for r in measured])
                result['acquisition'] = metadata
                result['placement'] = 'finger'
                if not self.stop_event.is_set():
                    intervals_file = Path(str(prefix) + '_intervals.csv')
                    with intervals_file.open('w', newline='') as file:
                        fields = ['interval_index', 'start_s', 'end_s', 'interval_ms', 'accepted', 'selected', 'reasons']
                        writer = csv.DictWriter(file, fieldnames=fields)
                        writer.writeheader()
                        for item in diagnostics['intervals']:
                            writer.writerow({**item, 'reasons': '; '.join(item['reasons'])})
                    files['intervals'] = intervals_file
                    # Graph failure must not discard a completed recording/result.
                    try:
                        plot = Path(hrv.save_plot(prefix, diagnostics, result))
                        if plot.is_file():
                            files['graph'] = plot
                        else:
                            result.setdefault('warnings', []).append('Graph not generated. Check matplotlib installation.')
                    except Exception as exc:
                        result.setdefault('warnings', []).append(f'Graph not generated: {exc}')
            # Stop during analysis is cancellation, never a completed measurement.
            with self.lock:
                if self.stop_event.is_set():
                    result.update(status='cancelled', pulse_rate_bpm=None, rmssd_ms=None, selected_seconds=0,
                                  reasons=['Cancelled by user.'], signal_quality_score=None, score=None,
                                  bpm_selected_seconds=0, bpm_interval_count=0)
                    # Remove analysis artifacts for a cancelled session; raw data remains available.
                    for kind in ('graph', 'intervals'):
                        files.pop(kind, None)
                # Generic score alias always means signal quality, never concussion risk.
                result['score'] = result.get('signal_quality_score')
                json_file = prefix.with_suffix('.json')
                json_file.write_text(json.dumps(result, indent=2, allow_nan=False))
                files['result'] = json_file
                self._publish_files(run_id, files)
                self.state['result'] = result
                self.state['stage'] = ('cancelled' if result['status'] == 'cancelled' else
                                       'error' if result['status'] == 'recording_failed' else 'complete')
                if result['status'] == 'recording_failed':
                    self.state['error'] = '; '.join(result['reasons'])
                self.state['running'] = False
        except Exception as exc:
            with self.lock:
                self._publish_files(run_id, files)
                self.state.update(stage='error', error=str(exc), result=None)
        finally:
            with self.lock:
                self.state['elapsed_seconds'] = round(time.monotonic() - self.started, 1)
                self.state['running'] = False
