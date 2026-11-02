import os
import time
import math
import csv
import queue
from pathlib import Path
from collections import deque
import cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp
from picamera2 import Picamera2

# GPIO Zero for 6-LED Flight Path Array
try:
    from gpiozero import LED
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("gpiozero not available. Running in mock mode.")

# Pin mapping based on your table: Right side first (indices 0, 1, 2), then Left side (indices 3, 4, 5)
LED_PINS = [17, 27, 22, 25, 24, 23]
leds = []  # Acquire GPIO only while a camera session is running.

# MediaPipe Tasks API Imports
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = str(Path(__file__).resolve().parent / "face_landmarker.task")
ML_MODEL_PATH = str(Path(__file__).resolve().parent / "concussion_detector_model.pkl")

# =============================================================================
# CLINICAL BASELINES & CONSTANTS
# =============================================================================
CLINICAL_BASELINES = {
    'FLASHLIGHT': {'target_mm': 2.00, 'min_mm': 1.60, 'max_mm': 2.30, 'description': 'Direct Flashlight (>1000 lx)'},
    'BRIGHT':     {'target_mm': 2.40, 'min_mm': 1.80, 'max_mm': 2.80, 'description': 'Bright Room (300-1000 lx)'},
    'AMBIENT':    {'target_mm': 3.20, 'min_mm': 2.60, 'max_mm': 4.00, 'description': 'Normal Ambient (50-300 lx)'},
    'DIM':        {'target_mm': 3.80, 'min_mm': 3.20, 'max_mm': 4.50, 'description': 'Dim Light (5-50 lx)'},
    'DARK':       {'target_mm': 4.40, 'min_mm': 3.60, 'max_mm': 5.20, 'description': 'Scotopic / Dark (<5 lx)'}
}

BLINK_PHYSIO_BOUNDS = {
    'BPM_LOW': 10.0,       
    'BPM_HIGH': 25.0,      
    'DUR_HIGH_MS': 300.0   
}

AVERAGE_HUMAN_IRIS_MM = 11.7  
BLINK_THRESHOLD = 0.65        


# =============================================================================
# HELPER: 2D IRIS-IN-SOCKET GAZE VECTOR CALCULATION
# =============================================================================
def compute_gaze_vector_2d(face_landmarks, img_w, img_h):
    l_outer = np.array([face_landmarks[33].x * img_w, face_landmarks[33].y * img_h])
    l_inner = np.array([face_landmarks[133].x * img_w, face_landmarks[133].y * img_h])
    l_top = np.array([face_landmarks[159].x * img_w, face_landmarks[159].y * img_h])
    l_bot = np.array([face_landmarks[145].x * img_w, face_landmarks[145].y * img_h])
    l_pupil = np.array([face_landmarks[468].x * img_w, face_landmarks[468].y * img_h])

    r_inner = np.array([face_landmarks[362].x * img_w, face_landmarks[362].y * img_h])
    r_outer = np.array([face_landmarks[263].x * img_w, face_landmarks[263].y * img_h])
    r_top = np.array([face_landmarks[386].x * img_w, face_landmarks[386].y * img_h])
    r_bot = np.array([face_landmarks[374].x * img_w, face_landmarks[374].y * img_h])
    r_pupil = np.array([face_landmarks[473].x * img_w, face_landmarks[473].y * img_h])

    l_width = np.linalg.norm(l_inner - l_outer) + 1e-6
    r_width = np.linalg.norm(r_outer - r_inner) + 1e-6
    l_rx = (l_pupil[0] - l_outer[0]) / l_width
    r_rx = (r_pupil[0] - r_inner[0]) / r_width
    avg_rx = float((l_rx + r_rx) / 2.0)

    l_height = np.linalg.norm(l_bot - l_top) + 1e-6
    r_height = np.linalg.norm(r_bot - r_top) + 1e-6
    l_ry = (l_pupil[1] - l_top[1]) / l_height
    r_ry = (r_pupil[1] - r_top[1]) / r_height
    avg_ry = float((l_ry + r_ry) / 2.0)

    return avg_rx, avg_ry


# =============================================================================
# HELPER: PUPILLOMETRY & LIGHTING UTILITIES
# =============================================================================
def check_blink_or_squint(face_blendshapes):
    if not face_blendshapes:
        return False, 0.0, 0.0

    blend_dict = {b.category_name: b.score for b in face_blendshapes[0]}
    blink_left = blend_dict.get('eyeBlinkLeft', 0.0)
    blink_right = blend_dict.get('eyeBlinkRight', 0.0)

    is_blinking = max(blink_left, blink_right) > BLINK_THRESHOLD
    return is_blinking, blink_left, blink_right


def estimate_lighting_environment(frame, face_landmarks, img_w, img_h):
    xs = [int(lm.x * img_w) for lm in face_landmarks]
    ys = [int(lm.y * img_h) for lm in face_landmarks]

    x1, x2 = max(0, min(xs)), min(img_w, max(xs))
    y1, y2 = max(0, min(ys)), min(img_h, max(ys))

    face_roi = frame[y1:y2, x1:x2]
    if face_roi.size == 0:
        return 'AMBIENT', 120.0

    luminance = 0.2126 * face_roi[:, :, 0] + 0.7152 * face_roi[:, :, 1] + 0.0722 * face_roi[:, :, 2]
    mean_lum = float(np.mean(luminance))
    p95_lum = float(np.percentile(luminance, 95))

    if p95_lum > 230 and mean_lum > 160:
        return 'FLASHLIGHT', mean_lum
    elif mean_lum >= 150:
        return 'BRIGHT', mean_lum
    elif mean_lum >= 80:
        return 'AMBIENT', mean_lum
    elif mean_lum >= 30:
        return 'DIM', mean_lum
    else:
        return 'DARK', mean_lum


def compute_iris_pixel_info(face_landmarks, img_w, img_h):
    l_left = np.array([face_landmarks[469].x * img_w, face_landmarks[469].y * img_h])
    l_right = np.array([face_landmarks[471].x * img_w, face_landmarks[471].y * img_h])
    r_left = np.array([face_landmarks[474].x * img_w, face_landmarks[474].y * img_h])
    r_right = np.array([face_landmarks[476].x * img_w, face_landmarks[476].y * img_h])

    left_iris_px = np.linalg.norm(l_left - l_right)
    right_iris_px = np.linalg.norm(r_left - r_right)

    avg_iris_px = (left_iris_px + right_iris_px) / 2.0
    if avg_iris_px < 4.0:
        return None, None

    mm_per_px = AVERAGE_HUMAN_IRIS_MM / avg_iris_px
    return mm_per_px, avg_iris_px


def suppress_local_glints(gray_crop):
    max_val = np.max(gray_crop)
    if max_val < 200:
        return gray_crop

    glint_thresh = max(210, int(max_val * 0.90))
    _, glint_mask = cv2.threshold(gray_crop, glint_thresh, 255, cv2.THRESH_BINARY)

    if cv2.countNonZero(glint_mask) == 0:
        return gray_crop

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    glint_mask = cv2.dilate(glint_mask, kernel, iterations=1)

    return cv2.inpaint(gray_crop, glint_mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)


def extract_pupil_restricted(eye_crop, eye_offset, red_anchor_global, mm_per_px, avg_iris_px):
    if eye_crop.size == 0 or eye_crop.shape[0] < 12 or eye_crop.shape[1] < 12:
        return None

    crop_h, crop_w, _ = eye_crop.shape
    offset_x, offset_y = eye_offset
    red_gx, red_gy = red_anchor_global

    red_lx = int(red_gx - offset_x)
    red_ly = int(red_gy - offset_y)
    red_local = np.array([red_lx, red_ly], dtype=np.float32)

    max_pupil_radius_px = int((avg_iris_px / 2.0) * 0.65)
    max_pupil_area = np.pi * (max_pupil_radius_px ** 2)
    min_pupil_area = np.pi * (1.2 / mm_per_px / 2.0) ** 2

    gray = cv2.cvtColor(eye_crop, cv2.COLOR_RGB2GRAY)
    gray_clean = suppress_local_glints(gray)

    roi_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    cv2.circle(roi_mask, (red_lx, red_ly), max_pupil_radius_px, 255, -1)
    masked_gray = cv2.bitwise_and(gray_clean, gray_clean, mask=roi_mask)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    equalized = clahe.apply(masked_gray)
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)

    mask_anchor = np.zeros((crop_h, crop_w), dtype=np.uint8)
    cv2.circle(mask_anchor, (red_lx, red_ly), 5, 255, -1)
    min_val, _, _, _ = cv2.minMaxLoc(blurred, mask=mask_anchor)

    best_cnt = None
    min_dist_to_anchor = float('inf')

    for delta in [10, 15, 20, 25, 30]:
        thresh_val = min(min_val + delta, 110)
        _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)

        thresh = cv2.bitwise_and(thresh, thresh, mask=roi_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_pupil_area or area > max_pupil_area:
                continue

            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
                cnt_center = np.array([cx, cy], dtype=np.float32)

                dist_to_red = np.linalg.norm(cnt_center - red_local)

                if dist_to_red < max_pupil_radius_px * 0.8:
                    if dist_to_red < min_dist_to_anchor:
                        min_dist_to_anchor = dist_to_red
                        best_cnt = cnt

        if best_cnt is not None:
            break

    if best_cnt is not None and len(best_cnt) >= 5:
        ellipse = cv2.fitEllipse(best_cnt)
        (cx, cy), (d1, d2), angle = ellipse

        max_allowed_dia_px = max_pupil_radius_px * 2.0
        d1 = min(d1, max_allowed_dia_px)
        d2 = min(d2, max_allowed_dia_px)

        avg_dia_px = (d1 + d2) / 2.0
        avg_dia_mm = avg_dia_px * mm_per_px

        full_center = (int(cx + offset_x), int(cy + offset_y))
        full_axes = (int(d1 / 2.0), int(d2 / 2.0))

        return {
            'center': full_center,
            'axes': full_axes,
            'angle': angle,
            'diameter_mm': avg_dia_mm
        }

    return None


# =============================================================================
# HELPER: LED FLIGHT PATH SYNCHRONIZATION (Slightly Slower Custom Pacing)
# =============================================================================
def update_led_flight_path(elapsed_time):
    if not GPIO_AVAILABLE:
        return

    for led in leds:
        led.off()

    # --- ADJUSTED SPEED (~5% slower to match screen dot perfectly) ---
    # Stretched cycle period to 4.85 seconds so the 2nd oscillation lags slightly behind.
    cycle_period = 4.85 
    half_cycle = cycle_period / 2.0  # ~2.42s per side (Right vs Left)
    
    oscillation_time = elapsed_time % cycle_period
    is_right_side = (oscillation_time < half_cycle)
    side_time = oscillation_time if is_right_side else (oscillation_time - half_cycle)

    # Proportional timing breakdown across the 2.42s side sweep:
    t1 = half_cycle * 0.165  # Inner
    t2 = half_cycle * 0.330  # Mid
    t3 = half_cycle * 0.670  # Outer peak hold
    t4 = half_cycle * 0.835  # Mid returning

    if side_time < t1:
        idx = 0  # Inner
    elif side_time < t2:
        idx = 1  # Mid
    elif side_time < t3:
        idx = 2  # Outer (Peak hold)
    elif side_time < t4:
        idx = 1  # Mid (returning)
    else:
        idx = 0  # Inner (finishing)

    # Map to correct hardware indices (0-2 for Right, 3-5 for Left)
    target_idx = idx if is_right_side else (idx + 3)
    leds[target_idx].on()


# =============================================================================
# CLASSES: TRACKING & METRICS MODULES
# =============================================================================
class BlinkKineticsTracker:
    def __init__(self, blink_threshold=0.60, min_blink_duration_ms=50.0):
        self.blink_threshold = blink_threshold
        self.min_blink_duration_ms = min_blink_duration_ms
        self.is_blinking_state = False
        self.blink_start_time = 0.0
        self.blink_durations_ms = []
        self.blink_timestamps = []

    def update(self, left_blink_score, right_blink_score, current_time):
        current_score = max(left_blink_score, right_blink_score)
        is_closing = current_score > self.blink_threshold

        if is_closing and not self.is_blinking_state:
            self.is_blinking_state = True
            self.blink_start_time = current_time
        elif not is_closing and self.is_blinking_state:
            self.is_blinking_state = False
            duration_ms = (current_time - self.blink_start_time) * 1000.0
            if duration_ms >= self.min_blink_duration_ms:
                self.blink_durations_ms.append(duration_ms)
                self.blink_timestamps.append(current_time)

    def get_metrics(self, window_seconds=60.0, current_time=None):
        if current_time is None:
            current_time = time.time()

        cutoff = current_time - window_seconds
        valid_indices = [i for i, t in enumerate(self.blink_timestamps) if t >= cutoff]
        recent_durations = [self.blink_durations_ms[i] for i in valid_indices]
        blink_count = len(valid_indices)

        if self.blink_timestamps:
            first_ts = self.blink_timestamps[valid_indices[0]] if valid_indices else current_time
            elapsed_window = min(window_seconds, max(1.0, current_time - first_ts))
        else:
            elapsed_window = window_seconds

        bpm = (blink_count / elapsed_window) * 60.0 if blink_count > 0 else 0.0
        avg_duration_ms = float(np.mean(recent_durations)) if recent_durations else 0.0

        bpm_status = "NORMAL"
        if bpm > 0:
            if bpm < BLINK_PHYSIO_BOUNDS['BPM_LOW']:
                bpm_status = "LOW (Hypoblink)"
            elif bpm > BLINK_PHYSIO_BOUNDS['BPM_HIGH']:
                bpm_status = "HIGH (Hyperblink)"

        dur_status = "NORMAL"
        if avg_duration_ms > BLINK_PHYSIO_BOUNDS['DUR_HIGH_MS']:
            dur_status = "PROLONGED"

        return bpm, avg_duration_ms, bpm_status, dur_status


class FixationalJitterTracker:
    def __init__(self, target_duration=3.0, user_distance_cm=50.0):
        self.target_duration = target_duration
        self.user_distance_cm = user_distance_cm
        self.accumulated_active_time = 0.0
        self.raw_gaze_ratios = []
        self.filtered_gaze_ratios = []
        self.rms_jitter_dva = 0.0
        self.peak_drift_dva = 0.0
        self.fixation_grade = "N/A"

    def reset(self):
        self.accumulated_active_time = 0.0
        self.raw_gaze_ratios.clear()
        self.filtered_gaze_ratios.clear()

    def update(self, face_landmarks, img_w, img_h, is_blinking, dt):
        if is_blinking or face_landmarks is None:
            return False

        rx, _ = compute_gaze_vector_2d(face_landmarks, img_w, img_h)
        self.raw_gaze_ratios.append(rx)

        alpha = 0.35
        if len(self.filtered_gaze_ratios) == 0:
            self.filtered_gaze_ratios.append(rx)
        else:
            f_ratio = alpha * rx + (1 - alpha) * self.filtered_gaze_ratios[-1]
            self.filtered_gaze_ratios.append(f_ratio)

        self.accumulated_active_time += dt
        if self.accumulated_active_time >= self.target_duration:
            self._compute_results()
            return True
        return False

    def _compute_results(self):
        if len(self.filtered_gaze_ratios) < 5:
            self.fixation_grade = "INSUFFICIENT DATA"
            return

        ratios = np.array(self.filtered_gaze_ratios)
        centered_ratios = ratios - np.mean(ratios)
        dva_estimates = centered_ratios * 40.0

        self.rms_jitter_dva = float(np.sqrt(np.mean(dva_estimates**2)))
        self.peak_drift_dva = float(np.max(np.abs(dva_estimates)))

        if self.rms_jitter_dva < 0.35 and self.peak_drift_dva < 0.85:
            self.fixation_grade = "STABLE (NORMAL)"
        elif self.rms_jitter_dva < 0.75:
            self.fixation_grade = "MILD UNSTABLE"
        else:
            self.fixation_grade = "UNSTABLE (ELEVATED DRIFT)"


class SmoothPursuitTracker:
    def __init__(self, duration=8.0, freq_hz=0.20, user_distance_cm=50.0):
        self.duration = duration
        self.freq_hz = freq_hz
        self.user_distance_cm = user_distance_cm
        self.start_time = 0.0
        self.elapsed_time = 0.0
        self.target_dva = []
        self.eye_dva = []
        self.target_v_dva = []
        self.eye_v_dva = []
        self.saccade_count = 0
        self.pursuit_gain = 0.0
        self.positional_error_dva = 0.0
        self.pursuit_grade = "N/A"

    def reset(self):
        self.start_time = time.time()
        self.elapsed_time = 0.0
        self.target_dva.clear()
        self.eye_dva.clear()
        self.target_v_dva.clear()
        self.eye_v_dva.clear()
        self.saccade_count = 0

    def update(self, face_landmarks, img_w, img_h, is_blinking, dt):
        self.elapsed_time = time.time() - self.start_time
        px_per_cm = img_w / 30.0

        amp_x_px = img_w * 0.30
        amp_y_px = img_h * 0.22
        center_x = img_w / 2.0
        center_y = img_h / 2.0

        target_px_x = center_x + amp_x_px * math.sin(2 * math.pi * self.freq_hz * self.elapsed_time)
        target_px_y = center_y + amp_y_px * math.cos(2 * math.pi * (self.freq_hz * 2) * self.elapsed_time)

        target_cm_x = (target_px_x - center_x) / px_per_cm
        target_cm_y = (target_px_y - center_y) / px_per_cm
        t_dva_x = math.degrees(math.atan2(target_cm_x, self.user_distance_cm))
        t_dva_y = math.degrees(math.atan2(target_cm_y, self.user_distance_cm))

        if not is_blinking and face_landmarks is not None:
            rx, ry = compute_gaze_vector_2d(face_landmarks, img_w, img_h)
            e_dva_x = (rx - 0.5) * 45.0
            e_dva_y = (ry - 0.5) * 30.0

            self.target_dva.append((t_dva_x, t_dva_y))
            self.eye_dva.append((e_dva_x, e_dva_y))

            if len(self.eye_dva) > 1 and dt > 0:
                prev_t, curr_t = self.target_dva[-2], self.target_dva[-1]
                prev_e, curr_e = self.eye_dva[-2], self.eye_dva[-1]

                vt = math.sqrt((curr_t[0] - prev_t[0])**2 + (curr_t[1] - prev_t[1])**2) / dt
                ve = math.sqrt((curr_e[0] - prev_e[0])**2 + (curr_e[1] - prev_e[1])**2) / dt

                self.target_v_dva.append(vt)
                self.eye_v_dva.append(ve)

                if ve > 45.0:
                    self.saccade_count += 1

        if self.elapsed_time >= self.duration:
            self._compute_results()
            return True
        return False

    def draw_target(self, frame, img_w, img_h):
        amp_x_px = img_w * 0.30
        amp_y_px = img_h * 0.22
        center_x = img_w / 2.0
        center_y = img_h / 2.0

        target_x = int(center_x + amp_x_px * math.sin(2 * math.pi * self.freq_hz * self.elapsed_time))
        target_y = int(center_y + amp_y_px * math.cos(2 * math.pi * (self.freq_hz * 2) * self.elapsed_time))

        radius = int(14 + 3 * math.sin(self.elapsed_time * 8))
        cv2.circle(frame, (target_x, target_y), radius, (0, 255, 0), -1)
        cv2.circle(frame, (target_x, target_y), radius + 4, (255, 255, 255), 2)

    def _compute_results(self):
        if len(self.eye_v_dva) < 10:
            self.pursuit_grade = "INSUFFICIENT DATA"
            return

        target_v = np.array(self.target_v_dva)
        eye_v = np.array(self.eye_v_dva)

        smooth_mask = eye_v < 45.0
        if np.sum(smooth_mask) > 5:
            mean_target_v = np.mean(target_v[smooth_mask])
            mean_eye_v = np.mean(eye_v[smooth_mask])
            self.pursuit_gain = float(mean_eye_v / (mean_target_v + 1e-5))
        else:
            self.pursuit_gain = 0.0

        t_arr = np.array(self.target_dva)
        e_arr = np.array(self.eye_dva)
        pos_err = np.sqrt((t_arr[:, 0] - e_arr[:, 0])**2 + (t_arr[:, 1] - e_arr[:, 1])**2)
        self.positional_error_dva = float(np.mean(pos_err))

        is_normal_gain = (0.25 <= self.pursuit_gain <= 1.35)
        is_tight_error = (self.positional_error_dva <= 8.5)
        is_low_saccades = (self.saccade_count <= 8)

        is_severe_failure = (
            (self.pursuit_gain < 0.15) or 
            (self.positional_error_dva > 12.0) or 
            (self.saccade_count > 15)
        )

        if is_normal_gain and is_tight_error and is_low_saccades and not is_severe_failure:
            self.pursuit_grade = "NORMAL PURSUIT"
        elif is_severe_failure:
            self.pursuit_grade = "SEVERELY IMPAIRED PURSUIT"
        else:
            self.pursuit_grade = "IMPAIRED (SACCADIC PURSUIT)"


# =============================================================================
# HELPER: DASHBOARD & UI RENDERING
# =============================================================================
def get_grade_color(grade_str):
    if "SEVERELY" in grade_str or "UNSTABLE (" in grade_str:
        return (0, 0, 255)       
    elif "MILD" in grade_str or "IMPAIRED" in grade_str:
        return (0, 165, 255)     
    elif "STABLE" in grade_str or "NORMAL" in grade_str:
        return (0, 255, 0)       
    return (200, 200, 200)


def draw_clinical_dashboard(frame, curr_mm, env_key, dev_mm, anomaly_flag, tracked_eyes, is_blinking, bpm, avg_blink_dur, bpm_status, dur_status, accumulated_time):
    margin = 20
    panel_w, panel_h = 390, 245

    overlay = frame.copy()
    cv2.rectangle(overlay, (margin, margin), (margin + panel_w, margin + panel_h), (18, 22, 28), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    env_info = CLINICAL_BASELINES.get(env_key, CLINICAL_BASELINES['AMBIENT'])
    baseline_target = env_info['target_mm']

    cv2.rectangle(frame, (margin, margin), (margin + panel_w, margin + 4), (0, 200, 255), -1)
    cv2.putText(frame, "CLINICAL PUPILLOMETRY STANDARDS", (margin + 12, margin + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 190, 200), 1, cv2.LINE_AA)

    if is_blinking:
        status_str = "[PAUSED] Blink"
        status_color = (0, 165, 255)
    elif tracked_eyes == 0:
        status_str = "[PAUSED] No Eyes"
        status_color = (0, 0, 255)
    else:
        status_str = f"Active ({accumulated_time:.1f}s/5.0s)"
        status_color = (0, 255, 120)

    cv2.putText(frame, status_str, (margin + 210, margin + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, status_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Lighting: {env_info['description']}", (margin + 12, margin + 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    meas_str = f"{curr_mm:.2f} mm" if curr_mm is not None else "Calibrating..."
    base_str = f"{baseline_target:.2f} mm"
    cv2.putText(frame, f"Measured (5s Med): {meas_str}", (margin + 12, margin + 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Research Standard: {base_str}", (margin + 12, margin + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.line(frame, (margin + 12, margin + 112), (margin + panel_w - 12, margin + 112), (50, 60, 70), 1)

    bpm_color = (0, 255, 120) if bpm_status == "NORMAL" else (0, 165, 255)
    dur_color = (0, 255, 120) if dur_status == "NORMAL" else (0, 0, 255)

    cv2.putText(frame, f"Blink Rate: {bpm:.1f} BPM ({bpm_status})", (margin + 12, margin + 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, bpm_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Avg Duration: {avg_blink_dur:.0f} ms ({dur_status})", (margin + 12, margin + 158),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, dur_color, 1, cv2.LINE_AA)

    cv2.line(frame, (margin + 12, margin + 172), (margin + panel_w - 12, margin + 172), (50, 60, 70), 1)

    dev_str = f"Pupil Dev: {'+' if dev_mm > 0 else ''}{dev_mm:.2f} mm" if dev_mm is not None else "Pupil Dev: -- mm"
    alert_text = "PUPIL ANOMALY!" if anomaly_flag else "PUPIL REFLEX OK"
    alert_color = (0, 0, 255) if anomaly_flag else (0, 255, 120)

    cv2.putText(frame, dev_str, (margin + 12, margin + 205),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, alert_color, 2, cv2.LINE_AA)
    cv2.putText(frame, alert_text, (margin + 200, margin + 205),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, alert_color, 2, cv2.LINE_AA)
    cv2.putText(frame, "Use Continue eye test below the preview", (margin, margin + panel_h + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)


# =============================================================================
# MAIN EXECUTION WORKFLOW
# =============================================================================
def model_output(model, pupil, deviation, anomaly, blink_rate, blink_duration,
                 jitter, pursuit, coverage, pupil_age):
    """Expose model output only with measured features and enough tracking data."""
    if model is None:
        return None, "Model unavailable. No replacement percentage is generated."
    if pupil is None or deviation is None or pupil_age is None or pupil_age > 60:
        return None, "Pupil measurement missing or stale; repeat the test."
    if len(jitter.filtered_gaze_ratios) < 5 or len(pursuit.eye_v_dva) < 10 or coverage < 0.6:
        return None, "Insufficient eye tracking; repeat with your face and eyes visible."
    values = [pupil, deviation, int(anomaly), blink_rate, blink_duration,
              jitter.rms_jitter_dva, jitter.peak_drift_dva, pursuit.pursuit_gain,
              pursuit.positional_error_dva, pursuit.saccade_count]
    if not all(math.isfinite(float(x)) for x in values):
        return None, "Non-finite eye measurements; model output withheld."
    columns = ["pupil_diameter", "pupil_deviation", "pupil_anomaly", "blink_rate",
               "blink_duration", "rms_jitter", "peak_drift", "pursuit_gain",
               "pursuit_error", "saccade_count"]
    try:
        classes = list(model.classes_)
        if 1 not in classes:
            return None, "Model does not contain the expected class 1 label."
        probability = float(model.predict_proba(pd.DataFrame([values], columns=columns))[0][classes.index(1)])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            return None, "Model returned an invalid probability."
        return probability * 100, "Unvalidated class-1 model output; not a diagnosis or overall sensor score."
    except Exception as exc:
        return None, f"Model prediction unavailable: {exc}"


def run_integrated_oculomotor_suite(stop_event, commands, publish):
    """Run original LED camera workflow without desktop windows or keyboard input."""
    global leds
    landmarker = None
    picam2 = None
    session_started = time.monotonic()
    warnings = [
        "Experimental camera measurements and model output have not been clinically validated.",
        "Physical LEDs move horizontally, but the supplied pursuit algorithm compares gaze with a 2D screen path. Its gain/error and model output are not validated for this LED task.",
    ]
    try:
        if not Path(MODEL_PATH).is_file():
            raise FileNotFoundError(f"Missing face landmark model: {MODEL_PATH}")
        ml_model = None
        if Path(ML_MODEL_PATH).is_file():
            try:
                ml_model = joblib.load(ML_MODEL_PATH)
            except Exception as exc:
                warnings.append(f"Could not load ML model: {exc}")
        else:
            warnings.append("concussion_detector_model.pkl is missing; model output is unavailable.")
        if GPIO_AVAILABLE:
            for pin in LED_PINS:
                leds.append(LED(pin))
        else:
            raise RuntimeError("gpiozero is missing. Physical LED tracking cannot start.")
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO, num_faces=1,
            min_face_detection_confidence=0.5, min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5, output_face_blendshapes=True,
            output_facial_transformation_matrixes=False)
        landmarker = vision.FaceLandmarker.create_from_options(options)
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}))
        picam2.start()
        blink_tracker = BlinkKineticsTracker(blink_threshold=BLINK_THRESHOLD)
        jitter_tracker = FixationalJitterTracker(target_duration=3.0, user_distance_cm=50.0)
        pursuit_tracker = SmoothPursuitTracker(duration=8.0, freq_hz=0.20, user_distance_cm=50.0)

        state = 'PUPIL_BLINK'
        history_5s = deque()
        accumulated_active_time = 0.0
        prev_frame_time = time.time()
        frame_timestamp_ms = 0

        current_env = 'AMBIENT'
        display_curr_mm = None
        display_dev_mm = None
        display_anomaly = False
    
        final_bpm = 0.0
        final_dur = 0.0
        concussion_confidence = None
        score_note = "Complete both eye tests to obtain a model output."
        pupil_age = None
        last_pupil_time = None
        pursuit_frames = 0
        pursuit_valid_frames = 0

        print("[SUCCESS] Browser camera pipeline active.")

        while not stop_event.is_set():
            if time.monotonic() - session_started > 180:
                raise TimeoutError("Camera session timed out after 180 seconds. Start a new test.")
            raw_frame = picam2.capture_array()
            if raw_frame is None:
                raise RuntimeError("Camera stopped delivering frames.")

            # Hardware orientation correction using OpenCV rotation
            raw_frame = cv2.rotate(raw_frame, cv2.ROTATE_180)
            raw_frame = cv2.flip(raw_frame, 1)

            gray_single = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.cvtColor(gray_single, cv2.COLOR_GRAY2RGB)

            img_h, img_w, _ = frame.shape
            now = time.time()
            dt = min(max(now - prev_frame_time, 0.001), 0.2)
            prev_frame_time = now

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            frame_timestamp_ms += int(dt * 1000)
            if frame_timestamp_ms <= 0:
                frame_timestamp_ms = 1

            results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            face_landmarks = None
            face_blendshapes = results.face_blendshapes if results else None
        
            is_blinking, left_blink_score, right_blink_score = check_blink_or_squint(face_blendshapes)
            blink_tracker.update(left_blink_score, right_blink_score, now)

            tracked_eyes_count = 0
            frame_mm_readings = []

            if results and results.face_landmarks and len(results.face_landmarks) > 0:
                face_landmarks = results.face_landmarks[0]
                if state == 'PUPIL_BLINK':
                    current_env, _ = estimate_lighting_environment(frame, face_landmarks, img_w, img_h)
                    mm_per_px, avg_iris_px = compute_iris_pixel_info(face_landmarks, img_w, img_h)

                    if mm_per_px is not None and avg_iris_px is not None:
                        for _, pupil_landmark in [('Left', face_landmarks[468]), ('Right', face_landmarks[473])]:
                            cx, cy = int(pupil_landmark.x * img_w), int(pupil_landmark.y * img_h)
                            cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)

                            if not is_blinking:
                                roi_radius = int(avg_iris_px * 0.9)
                                x1, x2 = max(0, cx - roi_radius), min(img_w, cx + roi_radius)
                                y1, y2 = max(0, cy - roi_radius), min(img_h, cy + roi_radius)
                                eye_crop = frame[y1:y2, x1:x2]
                                pupil_info = extract_pupil_restricted(eye_crop, (x1, y1), (cx, cy), mm_per_px, avg_iris_px)

                                if pupil_info:
                                    tracked_eyes_count += 1
                                    frame_mm_readings.append(pupil_info['diameter_mm'])
                                    cv2.ellipse(frame, pupil_info['center'], pupil_info['axes'], pupil_info['angle'], 0, 360, (0, 255, 0), 2, cv2.LINE_AA)

            center_pt = (int(img_w / 2), int(img_h / 2))

            if state == 'PUPIL_BLINK':
                if frame_mm_readings and not is_blinking and tracked_eyes_count > 0:
                    history_5s.append(float(np.mean(frame_mm_readings)))
                    accumulated_active_time += dt

                if accumulated_active_time >= 5.0:
                    accumulated_active_time = 0.0
                    if history_5s:
                        display_curr_mm = float(np.median(history_5s))
                        last_pupil_time = now
                        history_5s.clear()
                        baseline_target = CLINICAL_BASELINES[current_env]['target_mm']
                        display_dev_mm = display_curr_mm - baseline_target
                        display_anomaly = (display_curr_mm < CLINICAL_BASELINES[current_env]['min_mm'] or 
                                           display_curr_mm > CLINICAL_BASELINES[current_env]['max_mm'])

                bpm, avg_blink_dur, bpm_status, dur_status = blink_tracker.get_metrics(window_seconds=60.0, current_time=now)
                draw_clinical_dashboard(frame, display_curr_mm, current_env, display_dev_mm, display_anomaly, 
                                        tracked_eyes_count, is_blinking, bpm, avg_blink_dur, bpm_status, dur_status, accumulated_active_time)

            elif state == 'READY_JITTER':
                cv2.circle(frame, center_pt, 12, (0, 0, 255), -1)
                cv2.circle(frame, center_pt, 16, (255, 255, 255), 2)
                cv2.putText(frame, "TEST 1/2: FIXATIONAL JITTER ASSESSMENT", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "INSTRUCTION: Fixate your gaze steadily on the RED target.", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, "Use Continue eye test below the preview to start fixation", (30, img_h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            elif state == 'TESTING_JITTER':
                cv2.circle(frame, center_pt, 12, (0, 0, 255), -1)
                cv2.circle(frame, center_pt, 16, (255, 255, 255), 2)
                is_done = jitter_tracker.update(face_landmarks, img_w, img_h, is_blinking, dt)
                progress = jitter_tracker.accumulated_active_time / jitter_tracker.target_duration
                cv2.putText(frame, f"TESTING 1/2: HOLD GAZE STILL ({progress*100:.0f}%)", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if is_done:
                    state = 'READY_PURSUIT'

            elif state == 'READY_PURSUIT':
                cv2.circle(frame, center_pt, 14, (0, 255, 0), -1)
                cv2.circle(frame, center_pt, 18, (255, 255, 255), 2)
                cv2.putText(frame, "TEST 2/2: 2D SMOOTH PURSUIT ASSESSMENT", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "INSTRUCTION: Follow the physical LEDs with your eyes.", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, "Use Continue eye test; then follow the physical LEDs", (30, img_h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            elif state == 'TESTING_PURSUIT':
                pursuit_frames += 1
                if face_landmarks is not None and not is_blinking:
                    pursuit_valid_frames += 1
                pursuit_tracker.draw_target(frame, img_w, img_h)
            
                # --- SYNC PHYSICAL LEDS TO FLIGHT PATH (Fine-Tuned Slower Sweep) ---
                update_led_flight_path(pursuit_tracker.elapsed_time)
            
                is_done = pursuit_tracker.update(face_landmarks, img_w, img_h, is_blinking, dt)
                cv2.putText(frame, "TESTING 2/2: FOLLOW THE PHYSICAL LEDS", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if is_done:
                    if GPIO_AVAILABLE:
                        for led in leds:
                            led.off()

                    final_bpm, final_dur, _, _ = blink_tracker.get_metrics(window_seconds=120.0, current_time=now)

                    concussion_confidence, score_note = model_output(
                        ml_model, display_curr_mm, display_dev_mm, display_anomaly,
                        final_bpm, final_dur, jitter_tracker, pursuit_tracker,
                        pursuit_valid_frames / max(1, pursuit_frames),
                        now - last_pupil_time if last_pupil_time is not None else None)

                    state = 'SUMMARY'

            elif state == 'SUMMARY':
                overlay = frame.copy()
                cv2.rectangle(overlay, (30, 30), (img_w - 30, img_h - 30), (15, 15, 15), -1)
                cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

                cv2.putText(frame, "OCULOMOTOR ASSESSMENT SUMMARY", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.line(frame, (50, 85), (img_w - 50, 85), (200, 200, 200), 1)

                pupil_text = f"Diameter: {display_curr_mm:.2f} mm" if display_curr_mm else "Diameter: N/A"
            
                cv2.putText(frame, "0. PUPILOMETRY & BLINK KINETICS:", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                cv2.putText(frame, f"{pupil_text} | Blink Rate: {final_bpm:.1f} BPM | Avg Duration: {final_dur:.0f} ms",
                            (70, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                jitter_color = get_grade_color(jitter_tracker.fixation_grade)
                cv2.putText(frame, "1. FIXATIONAL JITTER:", (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                cv2.putText(frame, f"Grade: {jitter_tracker.fixation_grade}", (70, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.6, jitter_color, 2)
                cv2.putText(frame, f"RMS Jitter: {jitter_tracker.rms_jitter_dva:.2f} deg | Peak Drift: {jitter_tracker.peak_drift_dva:.2f} deg", 
                            (70, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                pursuit_color = get_grade_color(pursuit_tracker.pursuit_grade)
                cv2.putText(frame, "2. SMOOTH PURSUIT (2D):", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                cv2.putText(frame, f"Grade: {pursuit_tracker.pursuit_grade}", (70, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pursuit_color, 2)
                cv2.putText(frame, f"Gain: {pursuit_tracker.pursuit_gain:.2f} | Saccades: {pursuit_tracker.saccade_count} | Error: {pursuit_tracker.positional_error_dva:.2f} deg", 
                            (70, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                score_text = f"{concussion_confidence:.1f}%" if concussion_confidence is not None else "Unavailable"
                cv2.putText(frame, f"EXPERIMENTAL MODEL OUTPUT: {score_text}", (40, 390),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)
                cv2.putText(frame, "Results and controls are in the browser below the preview.",
                            (40, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

            pupil_age = now - last_pupil_time if last_pupil_time is not None else None
            blink_rate, blink_duration, _, _ = blink_tracker.get_metrics(current_time=now)
            can_continue = (state in ('READY_JITTER', 'READY_PURSUIT') or
                            (state == 'PUPIL_BLINK' and display_curr_mm is not None and
                             pupil_age is not None and pupil_age < 15 and tracked_eyes_count > 0))
            metrics = {
                'stage': state, 'pupil_mm': display_curr_mm,
                'blink_rate': final_bpm if state == 'SUMMARY' else blink_rate,
                'blink_duration_ms': final_dur if state == 'SUMMARY' else blink_duration,
                'face_detected': face_landmarks is not None,
                'is_blinking': is_blinking, 'can_continue': can_continue,
                'pupil_active_seconds': round(accumulated_active_time, 1),
                'fixation_grade': jitter_tracker.fixation_grade,
                'rms_jitter': jitter_tracker.rms_jitter_dva if state in ('READY_PURSUIT','TESTING_PURSUIT','SUMMARY') else None,
                'peak_drift': jitter_tracker.peak_drift_dva if state in ('READY_PURSUIT','TESTING_PURSUIT','SUMMARY') else None,
                'pursuit_grade': pursuit_tracker.pursuit_grade,
                'pursuit_gain': pursuit_tracker.pursuit_gain if state == 'SUMMARY' else None,
                'pursuit_error': pursuit_tracker.positional_error_dva if state == 'SUMMARY' else None,
                'saccade_count': pursuit_tracker.saccade_count if state == 'SUMMARY' else None,
                'pursuit_tracking_coverage': pursuit_valid_frames / max(1, pursuit_frames),
                'score': concussion_confidence if state == 'SUMMARY' else None,
                'score_note': score_note, 'warnings': list(warnings),
                'leds_available': GPIO_AVAILABLE, 'error': None,
            }
            # Frame arrays are RGB internally. JPEG encoding expects BGR.
            publish(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), metrics)
            try:
                command = commands.get_nowait()
            except queue.Empty:
                command = None
            # A command is tied to the stage seen by the browser, so a delayed
            # double-click cannot accidentally skip the following ready screen.
            if command == ('next', state) and can_continue:
                if state == 'PUPIL_BLINK':
                    state = 'READY_JITTER'
                elif state == 'READY_JITTER':
                    jitter_tracker.reset()
                    state = 'TESTING_JITTER'
                elif state == 'READY_PURSUIT':
                    pursuit_tracker.reset()
                    pursuit_frames = pursuit_valid_frames = 0
                    state = 'TESTING_PURSUIT'
    finally:
        # Always release every acquired resource, even if initialization failed.
        for led in leds:
            try:
                led.off()
                led.close()
            except Exception:
                pass
        leds = []
        if picam2 is not None:
            try:
                picam2.stop()
            except Exception:
                pass
            try:
                picam2.close()
            except Exception:
                pass
        if landmarker is not None:
            landmarker.close()
