from pathlib import Path
import atexit
import json
import uuid
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
session_id = None
state = {'running': False, 'stage': 'Waiting', 'error': None, 'elapsed': 0, 'score': None}
started = 0

def publish(frame, metrics):
    import cv2
    global latest_frame
    ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    with lock:
        if ok:
            latest_frame = encoded.tobytes()
        if stop_event.is_set():
            metrics = dict(metrics, score=None, stage='Stopping', can_continue=False)
        state.update(metrics)
        state['elapsed'] = round(time.monotonic() - started, 1)

def run_camera():
    try:
        from camera_web import run_integrated_oculomotor_suite
        run_integrated_oculomotor_suite(stop_event, commands, publish)
    except Exception as exc:
        with lock:
            state.update(stage='Error', error=str(exc), score=None, can_continue=False)
    finally:
        with lock:
            state['running'] = False
            if state['stage'] != 'Error':
                if stop_event.is_set() and state['stage'] != 'SUMMARY':
                    state.update(stage='Stopped', score=None)
                elif state['stage'] != 'SUMMARY':
                    state['stage'] = 'Stopped'
            state['can_continue'] = False

@app.get('/')
def home():
    return render_template('index.html')

@app.get('/api/status')
def status():
    with lock:
        return jsonify(dict(state))

@app.post('/api/start')
def start():
    global worker, latest_frame, started, session_id
    with lock:
        if worker is not None and worker.is_alive():
            return jsonify(error='Camera is already running or stopping'), 409
        while not commands.empty():
            commands.get_nowait()
        stop_event.clear()
        session_id = uuid.uuid4().hex
        latest_frame = None
        started = time.monotonic()
        state.clear()
        state.update(running=True, stage='Starting', error=None, elapsed=0, score=None, session_id=session_id, can_continue=False)
        worker = threading.Thread(target=run_camera, daemon=True)
        worker.start()
    return jsonify(ok=True)

@app.post('/api/next')
def next_stage():
    with lock:
        if not state['running'] or state['stage'] not in ('PUPIL_BLINK', 'READY_JITTER', 'READY_PURSUIT'):
            return jsonify(error='Wait for a ready stage'), 409
        if not state.get('can_continue'):
            return jsonify(error='Wait for a current pupil measurement with your eyes visible'), 409
        try:
            commands.put_nowait(('next', state['stage']))
        except queue.Full:
            pass
    return jsonify(ok=True)

@app.post('/api/stop')
def stop():
    with lock:
        if state['running']:
            stop_event.set()
            state.update(stage='Stopping', score=None, can_continue=False)
    return jsonify(ok=True)

def frames():
    with lock:
        stream_session = session_id
    while True:
        with lock:
            if stream_session != session_id:
                break
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

@app.get('/api/camera/result')
def camera_result():
    with lock:
        if state['stage'] != 'SUMMARY':
            return jsonify(error='Complete the eye test before downloading a result.'), 409
        result = dict(state)
    return Response(json.dumps(result, indent=2, allow_nan=False),
                    mimetype='application/json',
                    headers={'Content-Disposition': 'attachment; filename="camera_result.json"',
                             'Cache-Control': 'no-store'})

@atexit.register
def shutdown_camera():
    stop_event.set()
    if worker is not None and worker.is_alive():
        worker.join(timeout=3)

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
