"""Background accelerometer sessions and safe file lookup for Flask."""
import json
from pathlib import Path
import threading
import time
from uuid import uuid4
from datetime import datetime


class AccelerometerService:
    def __init__(self, output_dir, recorder=None):
        self.output_dir = Path(output_dir)
        self.recorder = recorder
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.started = None
        self.files = {}
        self.state = self._initial_state()

    @staticmethod
    def _initial_state():
        return dict(running=False, stage='idle', error=None, result=None,
                    run_id=None, downloads={}, duration_seconds=20, countdown=None,
                    elapsed_seconds=0, recording_elapsed_seconds=0,
                    sample_count=0, x_g=None, y_g=None, z_g=None)

    def snapshot(self):
        with self.lock:
            state = dict(self.state)
            if state['running'] and self.started is not None:
                state['elapsed_seconds'] = round(time.monotonic() - self.started, 1)
            return json.loads(json.dumps(state, allow_nan=False))

    def start(self):
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                return False
            self.stop_event.clear()
            self.started = time.monotonic()
            run_id = datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid4().hex[:12]
            self.state = self._initial_state()
            self.state.update(running=True, stage='starting', run_id=run_id)
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

    def _run(self, run_id):
        try:
            recorder = self.recorder
            if recorder is None:
                from accelerometer import record_balance
                recorder = record_balance
            result = recorder(duration=20, stop_event=self.stop_event, progress=self._progress,
                              output_folder=self.output_dir, run_id=run_id)
            with self.lock:
                if self.stop_event.is_set():
                    result.update(status='cancelled', x_sd_g=None, y_sd_g=None, z_sd_g=None,
                                  combined_variability_g=None, assessment_score=None,
                                  movement_level=None, movement_label=None,
                                  movement_classification_status='unavailable',
                                  movement_note='Cancelled: no completed movement result.')
                    Path(result['result_file']).write_text(json.dumps(result, indent=2, allow_nan=False))
                files = {}
                for kind, key in [('raw', 'file'), ('result', 'result_file')]:
                    path = Path(result[key])
                    if path.is_file():
                        files[kind] = path
                self.files[run_id] = files
                self.state.update(result=result, sample_count=result['sample_count'],
                                  downloads={kind:f'/api/accelerometer/files/{run_id}/{kind}' for kind in files},
                                  stage='complete' if result['status'] == 'complete' else 'cancelled' if result['status'] == 'cancelled' else 'error',
                                  error=result.get('error'), running=False)
        except Exception as exc:
            with self.lock:
                self.state.update(stage='error', error=str(exc), result=None)
        finally:
            with self.lock:
                self.state.update(running=False, elapsed_seconds=round(time.monotonic() - self.started, 1))

    def file(self, run_id, kind):
        with self.lock:
            path = self.files.get(run_id, {}).get(kind)
        return path if path is not None and path.is_file() else None
