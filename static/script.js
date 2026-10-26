'use strict';
const byId = id => document.getElementById(id);
const format = (value, digits = 1) => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
const stages = {idle:'Waiting', starting:'Starting sensor', settling:'Settling', recording:'Recording', analysing:'Analysing', stopping:'Cancelling', cancelled:'Cancelled', complete:'Complete', error:'Error'};
const instructions = {
    idle:'Start when you are ready.', starting:'Connecting to the sensor on the Pi…',
    settling:'Keep your finger in place while the sensor settles.',
    recording:'Recording real sensor readings. Keep your finger and the board still.',
    analysing:'Capture finished. You may remove your finger while the Pi analyses the recording.',
    stopping:'Waiting for the sensor to stop. Partial data will not produce a measurement.',
    cancelled:'Test cancelled. Start again when ready.',
    complete:'Capture and analysis finished. Check the result and download the saved data below.',
    error:'Check the error message and the Flask terminal on the Pi.'
};
const pending = new Set();
let hrvState = null;
let cameraState = null;
let cameraWasRunning = false;

function badge(id, text, type) {
    byId(id).textContent = text;
    byId(id).className = 'sensor-status ' + type;
}
function hrvButtons() {
    const available = hrvState !== null && !pending.has('hrv');
    byId('start-button').disabled = !available || hrvState.running;
    byId('stop-button').disabled = !available || !hrvState.running || hrvState.stage === 'stopping';
}
function cameraButtons() {
    const available = cameraState !== null && !pending.has('camera');
    byId('camera-start').disabled = !available || cameraState.running;
    byId('camera-stop').disabled = !available || !cameraState.running;
    byId('camera-next').disabled = !available || !cameraState.running ||
        !['PUPIL_BLINK','READY_JITTER','READY_PURSUIT'].includes(cameraState.stage) ||
        (cameraState.stage === 'PUPIL_BLINK' && cameraState.pupil_mm == null);
}
async function command(path, group) {
    if (pending.has(group)) return;
    pending.add(group);
    hrvButtons(); cameraButtons();
    const errorId = group === 'hrv' ? 'hrv-command-error' : 'camera-command-error';
    try {
        const response = await fetch(path, {method:'POST'});
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Request failed');
        byId(errorId).textContent = '';
        await (group === 'hrv' ? pollHRV() : pollCamera());
    } catch (error) { byId(errorId).textContent = error.message; }
    finally { pending.delete(group); hrvButtons(); cameraButtons(); }
}
byId('start-button').onclick = () => command('/api/hrv/start', 'hrv');
byId('stop-button').onclick = () => command('/api/hrv/stop', 'hrv');
byId('camera-start').onclick = () => command('/api/start', 'camera');
byId('camera-next').onclick = () => command('/api/next', 'camera');
byId('camera-stop').onclick = () => command('/api/stop', 'camera');

function renderHRV(s) {
    hrvState = s;
    byId('recording-status').textContent = stages[s.stage] || s.stage;
    byId('elapsed-time').textContent = format(s.elapsed_seconds);
    byId('hrv-progress').value = s.captured_samples;
    byId('hrv-instruction').textContent = instructions[s.stage] || '';
    byId('captured-samples').textContent = s.captured_samples;
    byId('recorded-samples').textContent = s.recorded_samples;
    byId('raw-ir').textContent = format(s.latest_ir, 0);
    byId('raw-red').textContent = format(s.latest_red, 0);
    byId('hrv-error').textContent = s.error || '';
    if (s.running) badge('heart-status', s.latest_ir == null ? 'Connecting' : 'Data arriving', s.latest_ir == null ? 'waiting' : 'reading');
    else if (s.stage === 'error') badge('heart-status', 'Capture error', 'disconnected');
    else if (s.captured_samples > 0) badge('heart-status', 'Capture stopped', 'waiting');
    else badge('heart-status', 'Not checked', 'disconnected');
    const r = s.result;
    const usable = r && r.status === 'prototype_segment_estimate';
    byId('pulse-rate').textContent = usable ? format(r.pulse_rate_bpm) : '—';
    byId('rmssd').textContent = usable ? format(r.rmssd_ms, 2) : '—';
    byId('selected-duration').textContent = r ? format(r.selected_seconds, 2) : '—';
    byId('longest-duration').textContent = r ? format(r.longest_usable_run_seconds, 2) : '—';
    let message = s.running ? 'Waiting for this recording’s result.' : 'No test completed yet.';
    if (r) {
        if (usable) message = 'Experimental estimate from the selected continuous section.';
        else if (r.status === 'insufficient_signal_quality') message = 'Data captured. Insufficient signal quality: pulse rate and RMSSD unavailable.';
        else if (r.status === 'cancelled') message = 'Cancelled: no completed measurement.';
        else message = 'Recording failed: no measurement available.';
    }
    byId('hrv-result-message').textContent = message;
    byId('hrv-reasons').replaceChildren();
    for (const reason of r?.reasons || []) {
        const li = document.createElement('li'); li.textContent = reason; byId('hrv-reasons').append(li);
    }
    byId('hrv-downloads').replaceChildren();
    const labels = {raw:'Raw sensor data (CSV)', result:'Result and capture details (JSON)', intervals:'Detected intervals (CSV)', graph:'Pulse graph (PNG)'};
    for (const [kind, url] of Object.entries(s.downloads)) {
        const li = document.createElement('li'); const a = document.createElement('a');
        a.textContent = labels[kind] || kind; a.href = url; a.download = ''; li.append(a); byId('hrv-downloads').append(li);
    }
    byId('save-status').textContent = s.downloads.raw ? 'Raw data saved on the Pi. Download a copy here.' : (s.running ? 'Capture in progress; files are saved when it stops.' : 'No saved files for this session yet.');
    byId('run-id').textContent = s.run_id ? 'Recording: ' + s.run_id : '';
    hrvButtons();
}
async function pollHRV() {
    try {
        const response = await fetch('/api/hrv/status', {cache:'no-store'});
        if (!response.ok) throw new Error('HRV status request failed');
        renderHRV(await response.json());
    } catch (error) {
        hrvState = null; hrvButtons();
        byId('hrv-error').textContent = 'Cannot reach HRV controls. Check the Pi connection and Flask terminal.';
        badge('heart-status', 'Connection unavailable', 'disconnected');
    }
}
const cameraInstructions = {
    PUPIL_BLINK:'Wait for a pupil measurement, then press Continue eye test.',
    READY_JITTER:'Look at the centre dot. Press Continue eye test to start fixation.',
    TESTING_JITTER:'Keep looking at the centre dot.',
    READY_PURSUIT:'Press Continue eye test, then follow the moving dot.',
    TESTING_PURSUIT:'Follow the moving target and keep your head still.',
    SUMMARY:'Eye test finished. Stop the camera when ready.'
};
async function pollCamera() {
    try {
        const response = await fetch('/api/status', {cache:'no-store'});
        if (!response.ok) throw new Error('Camera status unavailable');
        const s = await response.json(); cameraState = s;
        badge('camera-status', s.stage.replaceAll('_',' '), s.running ? 'reading' : 'disconnected');
        byId('camera-instruction').textContent = cameraInstructions[s.stage] || 'Camera controls are separate from the HRV test.';
        byId('camera-error').textContent = s.error || '';
        byId('camera-pupil').textContent = format(s.pupil_mm, 2);
        byId('camera-blink').textContent = format(s.blink_rate);
        if (s.running && !cameraWasRunning) {
            byId('camera-preview').hidden = false;
            byId('camera-preview').src = '/video_feed?t=' + Date.now();
        }
        cameraWasRunning = s.running;
        cameraButtons();
    } catch (error) {
        cameraState = null; cameraButtons();
        byId('camera-error').textContent = 'Camera controls unavailable. HRV can still be used if connected.';
    }
}
async function poll() {
    await Promise.allSettled([pollHRV(), pollCamera()]);
    setTimeout(poll, 500);
}
poll();
