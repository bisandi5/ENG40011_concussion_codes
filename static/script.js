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
let accelState = null;
let cameraWasRunning = false;
let cameraSession = null;

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
    byId('camera-stop').disabled = !available || !cameraState.running || cameraState.stage === 'Stopping';
    byId('camera-next').disabled = !available || !cameraState.running ||
        !['PUPIL_BLINK','READY_JITTER','READY_PURSUIT'].includes(cameraState.stage) ||
        !cameraState.can_continue;
}
async function command(path, group) {
    if (pending.has(group)) return;
    pending.add(group);
    hrvButtons(); cameraButtons(); accelButtons();
    const errorId = group === 'hrv' ? 'hrv-command-error' : group === 'accel' ? 'accel-command-error' : 'camera-command-error';
    try {
        const response = await fetch(path, {method:'POST'});
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Request failed');
        byId(errorId).textContent = '';
        await (group === 'hrv' ? pollHRV() : group === 'accel' ? pollAccelerometer() : pollCamera());
    } catch (error) { byId(errorId).textContent = error.message; }
    finally { pending.delete(group); hrvButtons(); cameraButtons(); accelButtons(); }
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
    // Follow backend timing so labels, counts and progress cannot disagree.
    const positive = (value, fallback) => typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : fallback;
    const duration = positive(s.duration_seconds, 20);
    const settling = typeof s.settling_seconds === 'number' && Number.isFinite(s.settling_seconds) && s.settling_seconds >= 0 ? s.settling_seconds : 10;
    const sampleRate = positive(s.sample_rate_hz ?? s.result?.acquisition?.nominal_sample_rate_hz, 100);
    const target = Math.round(duration * sampleRate);
    const total = Math.round((duration + settling) * sampleRate);
    byId('hrv-settling-seconds').textContent = format(settling, 0);
    byId('hrv-recording-seconds').textContent = format(duration, 0);
    byId('hrv-target-samples').textContent = target;
    byId('hrv-progress').max = total;
    byId('hrv-progress').value = s.stage === 'complete' ? total : Math.max(0, Math.min(total, s.captured_samples || 0));
    byId('hrv-instruction').textContent = instructions[s.stage] || '';
    byId('captured-samples').textContent = s.captured_samples;
    byId('recorded-samples').textContent = s.recorded_samples;
    byId('raw-ir').textContent = format(s.latest_ir, 0);
    byId('raw-red').textContent = format(s.latest_red, 0);
    byId('hrv-error').textContent = s.error || '';
    if (s.running) badge('heart-status', s.latest_ir == null ? 'Connecting' : 'Data arriving', s.latest_ir == null ? 'waiting' : 'reading');
    else if (s.stage === 'error') badge('heart-status', 'Capture error', 'disconnected');
    else if (s.stage === 'complete') badge('heart-status', 'Capture complete', 'waiting');
    else if (s.captured_samples > 0) badge('heart-status', s.stage === 'cancelled' ? 'Capture cancelled' : 'Capture stopped', 'waiting');
    else badge('heart-status', 'Not checked', 'disconnected');
    const r = s.result;
    const usable = r && r.status === 'prototype_segment_estimate';
    const bpmUsable = usable || r?.status === 'prototype_pulse_rate_only';
    const finished = !s.running && s.stage === 'complete';
    byId('pulse-rate').textContent = finished && bpmUsable ? format(r.pulse_rate_bpm) : '—';
    byId('rmssd').textContent = finished && usable ? format(r.rmssd_ms, 2) : '—';
    const quality = r?.signal_quality_score;
    byId('hrv-quality-score').textContent = finished && typeof quality === 'number' && Number.isFinite(quality) && quality >= 0 && quality <= 100 ? format(quality) : '—';
    byId('selected-duration').textContent = r ? format(r.selected_seconds, 2) : '—';
    byId('longest-duration').textContent = r ? format(r.longest_usable_run_seconds, 2) : '—';
    let message = s.running ? 'Waiting for this recording’s result.' : 'No test completed yet.';
    if (r) {
        if (usable) message = 'Experimental estimate from the selected continuous section.';
        else if (r.status === 'prototype_pulse_rate_only') message = 'Pulse-rate estimate available. The usable continuous section did not meet the PRV RMSSD requirements.';
        else if (r.status === 'insufficient_signal_quality') message = 'Data captured. Insufficient signal quality: pulse rate and RMSSD unavailable.';
        else if (r.status === 'cancelled') message = 'Cancelled: no completed measurement.';
        else message = 'Recording failed: no measurement available.';
    }
    byId('hrv-result-message').textContent = message;
    byId('hrv-reasons').replaceChildren();
    for (const reason of r?.reasons || []) {
        const li = document.createElement('li'); li.textContent = reason; byId('hrv-reasons').append(li);
    }
    byId('hrv-warnings').replaceChildren();
    for (const warning of r?.warnings || []) {
        const li = document.createElement('li'); li.textContent = warning; byId('hrv-warnings').append(li);
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
    READY_PURSUIT:'Press Continue eye test, then follow the physical LEDs left and right.',
    TESTING_PURSUIT:'Follow the physical LEDs with your eyes and keep your head still.',
    SUMMARY:'Eye test finished. Download your result before stopping or starting again.',
    Stopping:'Stopping the camera and switching off the LEDs…'
};
function renderCamera(s) {
        cameraState = s;
        badge('camera-status', s.stage.replaceAll('_',' '), s.running ? 'reading' : 'disconnected');
        byId('camera-instruction').textContent = cameraInstructions[s.stage] || 'Camera controls are separate from the HRV test.';
        byId('camera-error').textContent = s.error || '';
        byId('camera-pupil').textContent = format(s.pupil_mm, 2);
        byId('camera-blink').textContent = format(s.blink_rate);
        byId('camera-tracking').textContent = s.face_detected ? (s.is_blinking ? 'Blink detected' : 'Face detected') : 'No face detected';
        byId('camera-elapsed').textContent = format(s.elapsed);
        byId('camera-fixation').textContent = s.fixation_grade || '—';
        byId('camera-jitter').textContent = format(s.rms_jitter, 2);
        byId('camera-pursuit').textContent = s.pursuit_grade || '—';
        byId('camera-gain').textContent = format(s.pursuit_gain, 2);
        byId('camera-pursuit-error').textContent = format(s.pursuit_error, 2);
        const hasScore = s.stage === 'SUMMARY' && typeof s.score === 'number' && Number.isFinite(s.score) && s.score >= 0 && s.score <= 100;
        byId('camera-score').textContent = hasScore ? format(s.score) + '%' : 'Unavailable';
        byId('camera-score-note').textContent = s.score_note || 'Complete the eye test to obtain a model output.';
        byId('camera-download').hidden = s.stage !== 'SUMMARY';
        byId('camera-warnings').replaceChildren();
        for (const warning of s.warnings || []) {
            const li = document.createElement('li'); li.textContent = warning; byId('camera-warnings').append(li);
        }
        if (s.running && (!cameraWasRunning || cameraSession !== s.session_id)) {
            byId('camera-preview').hidden = false;
            byId('camera-preview').src = '/video_feed?t=' + Date.now();
        }
        cameraWasRunning = s.running;
        cameraSession = s.session_id;
        cameraButtons();
}
async function pollCamera() {
    try {
        const response = await fetch('/api/status', {cache:'no-store'});
        if (!response.ok) throw new Error('Camera status unavailable');
        renderCamera(await response.json());
    } catch (error) {
        cameraState = null; cameraButtons();
        byId('camera-score').textContent = 'Unavailable';
        byId('camera-download').hidden = true;
        byId('camera-error').textContent = 'Camera controls unavailable. HRV can still be used if connected.';
    }
}
function accelButtons() {
    const available = accelState !== null && !pending.has('accel');
    byId('accel-start').disabled = !available || accelState.running;
    byId('accel-stop').disabled = !available || !accelState.running || accelState.stage === 'stopping';
}
byId('accel-start').onclick = () => command('/api/accelerometer/start', 'accel');
byId('accel-stop').onclick = () => command('/api/accelerometer/stop', 'accel');
function renderAccelerometer(s) {
    accelState = s;
    byId('accel-stage').textContent = s.stage === 'countdown' ? 'Countdown: ' + s.countdown : stages[s.stage] || s.stage;
    byId('accel-elapsed').textContent = format(s.elapsed_seconds);
    byId('accel-progress').value = s.stage === 'complete' ? 20 : Math.min(20, s.recording_elapsed_seconds);
    byId('accel-samples').textContent = s.sample_count;
    byId('accel-x').textContent = format(s.x_g, 3);
    byId('accel-y').textContent = format(s.y_g, 3);
    byId('accel-z').textContent = format(s.z_g, 3);
    byId('accel-error').textContent = s.error || '';
    const tips = {idle:'Start when the sensor is ready.', starting:'Connecting to the accelerometer on the Pi…',
        countdown:'Get ready: ' + s.countdown, recording:'Recording live X, Y and Z acceleration.',
        saving:'Capture finished. Saving the recording…', stopping:'Waiting for capture to stop.',
        cancelled:'Cancelled. No completed variability result.', complete:'Capture finished. Results and saved data are below.',
        error:'Check the error message and Pi terminal.'};
    byId('accel-instruction').textContent = tips[s.stage] || '';
    badge('accelerometer-status', s.stage === 'error' ? 'Capture error' : s.stage === 'complete' ? 'Capture complete' :
        s.stage === 'recording' ? 'Data arriving' : s.stage === 'idle' ? 'Not checked' : 'Waiting',
        s.stage === 'recording' || s.stage === 'complete' ? 'reading' : s.stage === 'error' || s.stage === 'idle' ? 'disconnected' : 'waiting');
    const r = s.result;
    const complete = r && r.status === 'complete';
    for (const [id, key] of Object.entries({'accel-x-sd':'x_sd_g','accel-y-sd':'y_sd_g','accel-z-sd':'z_sd_g','accel-combined':'combined_variability_g'})) {
        byId(id).textContent = complete ? format(r[key], 5) : '—';
    }
    byId('accel-movement-level').textContent = complete ? (r.movement_classification_status === 'available' ? r.movement_label : 'Unavailable') : '—';
    byId('accel-movement-note').textContent = complete ? (r.movement_note || 'Movement category unavailable. Check that the updated accelerometer backend is installed.') : 'Experimental movement categories based on team testing; no concussion probability is calculated.';
    byId('accel-rate').textContent = r ? format(r.average_read_rate_hz) : '—';
    byId('accel-gap').textContent = r ? format(r.largest_sample_gap_seconds, 3) : '—';
    byId('accel-result-message').textContent = complete ? 'Movement variability calculated from this recording.' :
        r?.status === 'cancelled' ? 'Cancelled: partial data is saved, with no variability result.' :
        r?.status === 'error' ? 'Capture failed: partial data may be available below.' :
        s.running ? 'Waiting for this recording’s result.' : 'No accelerometer test completed yet.';
    byId('accel-warnings').replaceChildren();
    for (const message of r?.warnings || []) {
        const li = document.createElement('li'); li.textContent = message; byId('accel-warnings').append(li);
    }
    byId('accel-downloads').replaceChildren();
    for (const [kind, url] of Object.entries(s.downloads)) {
        const li = document.createElement('li'); const a = document.createElement('a');
        a.href = url; a.download = ''; a.textContent = kind === 'raw' ? 'Acceleration readings (CSV)' : 'Result and timing details (JSON)';
        li.append(a); byId('accel-downloads').append(li);
    }
    byId('accel-save-status').textContent = s.downloads.raw ? 'Data saved on the Pi. Download a copy here.' : 'Recording files will appear after capture.';
    accelButtons();
}
async function pollAccelerometer() {
    try {
        const response = await fetch('/api/accelerometer/status', {cache:'no-store'});
        if (!response.ok) throw new Error('Accelerometer status unavailable');
        renderAccelerometer(await response.json());
    } catch (error) {
        accelState = null; accelButtons();
        byId('accel-error').textContent = 'Cannot reach accelerometer controls. Check the Pi connection and Flask terminal.';
        badge('accelerometer-status', 'Connection unavailable', 'disconnected');
    }
}
async function poll() {
    await Promise.allSettled([pollHRV(), pollCamera(), pollAccelerometer()]);
    setTimeout(poll, 500);
}
poll();
