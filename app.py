from pathlib import Path
import queue
import threading
import time
from flask import Flask, Response, jsonify, render_template, request, send_file

from hrv_service import HRVService
from accelerometer_service import AccelerometerService

app = Flask(__name__)
hrv_service = HRVService(Path(__file__).resolve().parent / "recordings")
accelerometer_service = AccelerometerService(Path(__file__).resolve().parent / "recordings")
lock = threading.Lock()
stop_event = threading.Event()
commands = queue.Queue(maxsize=1)
worker = None
latest_frame = None
state = {'running': False, 'stage': 'Waiting', 'error': None, 'elapsed': 0, 'score': None}
started = 0

def publish(frame, metrics):
    import cv2
    global latest_frame
    ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    with lock:
        if ok:
            latest_frame = encoded.tobytes()
        state.update(metrics)
        state['elapsed'] = round(time.monotonic() - started, 1)

def run_camera():
    try:
        from camera_web import run_integrated_oculomotor_suite
        run_integrated_oculomotor_suite(stop_event, commands, publish)
    except Exception as exc:
        with lock:
            state.update(stage='Error', error=str(exc))
    finally:
        with lock:
            state['running'] = False
            if state['stage'] != 'Error':
                state['stage'] = 'Stopped'

@app.get('/')
def home():
    return render_template('index.html')

@app.get('/api/status')
def status():
    with lock:
        return jsonify(dict(state))

@app.post('/api/start')
def start():
    global worker, latest_frame, started
    with lock:
        if worker is not None and worker.is_alive():
            return jsonify(error='Camera is already running or stopping'), 409
        while not commands.empty():
            commands.get_nowait()
        stop_event.clear()
        latest_frame = None
        started = time.monotonic()
        state.clear()
        state.update(running=True, stage='Starting', error=None, elapsed=0, score=None)
        worker = threading.Thread(target=run_camera, daemon=True)
        worker.start()
    return jsonify(ok=True)

@app.post('/api/next')
def next_stage():
    with lock:
        if not state['running'] or state['stage'] not in ('PUPIL_BLINK', 'READY_JITTER', 'READY_PURSUIT'):
            return jsonify(error='Wait for a ready stage'), 409
        if state['stage'] == 'PUPIL_BLINK' and state.get('pupil_mm') is None:
            return jsonify(error='Wait for a pupil measurement before continuing'), 409
        try:
            commands.put_nowait('next')
        except queue.Full:
            pass
    return jsonify(ok=True)

@app.post('/api/stop')
def stop():
    stop_event.set()
    return jsonify(ok=True)

def frames():
    while True:
        with lock:
            frame = latest_frame
            running = state['running']
        if frame is not None:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        if not running:
            break
        time.sleep(0.08)

@app.get('/video_feed')
def video_feed():
    return Response(frames(), mimetype='multipart/x-mixed-replace; boundary=frame', headers={'Cache-Control': 'no-store'})

@app.get('/api/hrv/status')
def hrv_status():
    return jsonify(hrv_service.snapshot())

@app.post('/api/hrv/start')
def hrv_start():
    if not hrv_service.start():
        return jsonify(error='HRV recording is already running or stopping.'), 409
    return jsonify(ok=True), 202

@app.post('/api/hrv/stop')
def hrv_stop():
    hrv_service.stop()
    return jsonify(ok=True), 202

@app.get('/api/hrv/files/<run_id>/<kind>')
def hrv_file(run_id, kind):
    path = hrv_service.file(run_id, kind)
    if path is None:
        return jsonify(error='File unavailable for this run.'), 404
    return send_file(path, as_attachment=True, download_name=path.name, max_age=0)

@app.get('/api/accelerometer/status')
def accelerometer_status():
    return jsonify(accelerometer_service.snapshot())

@app.post('/api/accelerometer/start')
def accelerometer_start():
    if not accelerometer_service.start():
        return jsonify(error='Accelerometer is already running or stopping.'), 409
    return jsonify(ok=True), 202

@app.post('/api/accelerometer/stop')
def accelerometer_stop():
    accelerometer_service.stop()
    return jsonify(ok=True), 202

@app.get('/api/accelerometer/files/<run_id>/<kind>')
def accelerometer_file(run_id, kind):
    path = accelerometer_service.file(run_id, kind)
    if path is None:
        return jsonify(error='File unavailable for this run.'), 404
    return send_file(path, as_attachment=True, download_name=path.name, max_age=0)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
