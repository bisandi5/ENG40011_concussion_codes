"""Transparent, team-defined prototype index. Not a diagnostic probability."""
import math

WEIGHTS = {'accelerometer': 0.20, 'hrv': 0.15, 'camera': 0.65}
MOVEMENT_SCORES = {'Low': 0, 'Medium': 50, 'High': 100}
HR_DERIVATIVE_THRESHOLD = 25.0
NOTE = ('Experimental project score using team-defined mappings and weights; '
        'not a validated concussion probability or measurement confidence. '
        'Higher values indicate more elevated inputs under these rules.')


def number(value, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0 or (maximum is not None and value > maximum):
        return None
    return float(value)


def component(name, value=None, category=None, raw=None, unit=None, reason=None, run_id=None):
    return dict(score=value, weight=WEIGHTS[name], category=category,
                contribution=None if value is None else round(value * WEIGHTS[name], 4),
                raw_value=raw, unit=unit, reason=reason, run_id=run_id)


def completed(snapshot, previous_id, camera=False):
    key = 'session_id' if camera else 'run_id'
    identifier = snapshot.get(key)
    stage = 'SUMMARY' if camera else 'complete'
    return (bool(identifier) and identifier != previous_id and
            snapshot.get('running') is False and snapshot.get('stage') == stage)


def calculate_assessment(hrv, accel, camera, baseline=None):
    """Require completed, usable, fresh results; never renormalize missing sensors."""
    baseline = baseline or {}
    parts = {}
    r = accel.get('result') or {}
    value = number(r.get('combined_variability_g'))
    reason = None
    if not completed(accel, baseline.get('accelerometer')):
        reason = 'Complete an accelerometer test for this assessment.'
    elif (r.get('status') != 'complete' or r.get('movement_classification_status') != 'available'
          or any(r.get(flag) for flag in ('possible_clipping', 'uneven_read_timing', 'short_recording'))):
        reason = 'Accelerometer result is unavailable or failed its capture quality checks.'
    elif value is None:
        reason = 'Accelerometer combined movement is missing or invalid.'
    category = ('Low' if value < .95 else 'Medium' if value <= 1.5 else 'High') if value is not None else None
    parts['accelerometer'] = component('accelerometer', MOVEMENT_SCORES[category] if not reason else None,
                                       category if not reason else None, value, 'g', reason, accel.get('run_id'))

    r = hrv.get('result') or {}
    value = number(r.get('peak_hr_derivative'))
    reason = None
    if not completed(hrv, baseline.get('hrv')):
        reason = 'Complete an HRV test for this assessment.'
    elif r.get('status') != 'prototype_segment_estimate':
        reason = 'HR derivative requires a usable continuous pulse result.'
    elif value is None:
        reason = 'Peak HR derivative is missing or invalid.'
    category = ('Elevated' if value > HR_DERIVATIVE_THRESHOLD else 'Stable') if value is not None else None
    parts['hrv'] = component('hrv', (100 if category == 'Elevated' else 0) if not reason else None,
                            category if not reason else None, value, 'BPM/s', reason, hrv.get('run_id'))

    value = number(camera.get('score'), 100)
    reason = None
    if not completed(camera, baseline.get('camera'), camera=True):
        reason = 'Complete the eye-camera test for this assessment.'
    elif camera.get('error') or value is None:
        reason = 'Camera confidence score is unavailable or invalid.'
    parts['camera'] = component('camera', value if not reason else None,
                               'Model output' if not reason else None, value, '/100', reason,
                               camera.get('session_id'))
    ready = all(p['score'] is not None for p in parts.values())
    return dict(status='complete' if ready else 'incomplete',
                final_score=round(sum(p['score'] * p['weight'] for p in parts.values()), 2) if ready else None,
                score_label='Final confidence score (experimental project index)',
                note=NOTE, components=parts, reasons=[p['reason'] for p in parts.values() if p['reason']],
                rules=dict(weights=WEIGHTS, movement_scores=MOVEMENT_SCORES,
                           movement_thresholds_g=dict(low_below=.95, medium_through=1.5),
                           hr_derivative_elevated_above=HR_DERIVATIVE_THRESHOLD,
                           hr_derivative_scores=dict(stable=0, elevated=100)),
                hrv_signal_quality_score=(hrv.get('result') or {}).get('signal_quality_score'),
                camera_warnings=camera.get('warnings', []))
