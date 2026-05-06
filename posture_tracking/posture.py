import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import numpy as np
import time # Added for the 3-second timer

def calculate_angle(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.degrees(math.atan2(y2 - y1, x2 - x1))

def calculate_vertical_tilt(top_point, bottom_point):
    dx = top_point[0] - bottom_point[0]
    dy = bottom_point[1] - top_point[1] 
    angle = math.degrees(math.atan2(dy, dx))
    tilt_degrees = 90 - angle
    
    if tilt_degrees > 1.5:
        direction = "Right" 
    elif tilt_degrees < -1.5:
        direction = "Left"  
    else:
        direction = "Centered"
        
    return abs(tilt_degrees), direction

# 1. Setup the Model Options
model_path = 'posture_tracking/pose_landmarker_full.task' # Ensure this path is correct!

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5
)

# 2. Create the Landmarker
detector = vision.PoseLandmarker.create_from_options(options)

# Start video capture
cap = cv2.VideoCapture(0)
print("Starting Clinical Posture Scanner... Press 'q' to quit.")

# Timer setup
start_time = time.time()
warning_duration = 5.0 # 5 seconds

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    elapsed_time = current_time - start_time

    # --- Phase A: The Warning Screen (First 3 Seconds) ---
    if elapsed_time < warning_duration:
        # Create a dark semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        # Blend it with the original frame (70% black, 30% camera feed)
        frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
        
        # Calculate countdown
        seconds_left = int(warning_duration - elapsed_time) + 1
        
        # Display Warning Text
        cv2.putText(frame, "CLINICAL SCREENING PROTOCOL", (30, 80), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, "1. Please wear form-fitting clothing.", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "2. Baggy clothes will skew hip readings.", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "3. Stand perfectly straight.", (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Starting in {seconds_left}...", (30, 300), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)

    # --- Phase B: The Main Tracking Logic (After 3 Seconds) ---
    else:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = detector.detect(mp_image)
        
        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]
            h, w, _ = frame.shape
            
            l_shoulder = [int(landmarks[11].x * w), int(landmarks[11].y * h)]
            r_shoulder = [int(landmarks[12].x * w), int(landmarks[12].y * h)]
            l_hip = [int(landmarks[23].x * w), int(landmarks[23].y * h)]
            r_hip = [int(landmarks[24].x * w), int(landmarks[24].y * h)]
            
            mid_shoulder = [int((l_shoulder[0] + r_shoulder[0]) / 2), int((l_shoulder[1] + r_shoulder[1]) / 2)]
            mid_hip = [int((l_hip[0] + r_hip[0]) / 2), int((l_hip[1] + r_hip[1]) / 2)]

            shoulder_angle = abs(calculate_angle(l_shoulder, r_shoulder))
            shoulder_dev = abs(180 - shoulder_angle) if shoulder_angle > 90 else shoulder_angle
            
            hip_angle = abs(calculate_angle(l_hip, r_hip))
            hip_dev = abs(180 - hip_angle) if hip_angle > 90 else hip_angle
            
            tilt_deg, tilt_dir = calculate_vertical_tilt(mid_shoulder, mid_hip)

            # --- SENSITIVE THRESHOLDS ---
            shoulder_threshold = 3.0
            hip_threshold = 1.5 # Now highly sensitive
            tilt_threshold = 2.0

            status_color = (0, 255, 0)
            
            hud_messages = [
                f"Shoulders: {shoulder_dev:.1f} deg",
                f"Hips: {hip_dev:.1f} deg",
                f"Spine: {tilt_deg:.1f} deg ({tilt_dir})"
            ]

            if shoulder_dev > shoulder_threshold or hip_dev > hip_threshold or tilt_deg > tilt_threshold:
                status_color = (0, 0, 255)
                cv2.putText(frame, "ASYMMETRY DETECTED", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, status_color, 2)
            else:
                cv2.putText(frame, "ALIGNMENT NORMAL", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, status_color, 2)

            for i, msg in enumerate(hud_messages):
                cv2.putText(frame, msg, (20, 80 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            cv2.line(frame, tuple(l_shoulder), tuple(r_shoulder), status_color, 2)
            cv2.line(frame, tuple(l_hip), tuple(r_hip), status_color, 2)
            cv2.line(frame, tuple(mid_shoulder), tuple(mid_hip), (255, 255, 0), 3)
            
            for point in [l_shoulder, r_shoulder, l_hip, r_hip, mid_shoulder, mid_hip]:
                cv2.circle(frame, tuple(point), 6, status_color, -1)

    cv2.imshow('Scoliosis Screening Tool', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()