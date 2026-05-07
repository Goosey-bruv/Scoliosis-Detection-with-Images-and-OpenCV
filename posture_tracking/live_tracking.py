import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Initialize Modern Tasks API
print("🧠 Loading MediaPipe Tasks API...")
base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    num_poses=1) 
detector = vision.PoseLandmarker.create_from_options(options)

# 2. Open Webcam
print("📷 Starting Webcam... (Press 'q' to quit)")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 3. Format Image for Tasks API
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Run Inference
    detection_result = detector.detect(mp_image)
    image = frame.copy()

    # 4. Clinical Math & Custom Wireframe Logic
    if detection_result.pose_landmarks:
        pose_landmarks = detection_result.pose_landmarks[0]
        h, w = image.shape[:2]
        
        # Extract Shoulders and Hips (Normalized coordinates * Image dimensions)
        l_shldr_pt = (int(pose_landmarks[11].x * w), int(pose_landmarks[11].y * h))
        r_shldr_pt = (int(pose_landmarks[12].x * w), int(pose_landmarks[12].y * h))
        l_hip_pt = (int(pose_landmarks[23].x * w), int(pose_landmarks[23].y * h))
        r_hip_pt = (int(pose_landmarks[24].x * w), int(pose_landmarks[24].y * h))

        # Calculate shoulder tilt using Trigonometry
        dx = r_shldr_pt[0] - l_shldr_pt[0]
        dy = r_shldr_pt[1] - l_shldr_pt[1]
        
        # Prevent math crashes if you stand perfectly parallel to the camera pixel grid
        if dx == 0:
            angle = 90.0
        else:
            # np.arctan keeps it between -90 and 90. abs() makes it a clean 0-90 tilt.
            angle = abs(np.degrees(np.arctan(dy / dx)))

        # Diagnostic Thresholds
        if angle > 3.0:
            status, color = "IMBALANCE DETECTED", (0, 0, 255) # Red
        else:
            status, color = "GOOD ALIGNMENT", (0, 255, 0)     # Green

        # --- DRAW DASHBOARD OVERLAY ---
        cv2.rectangle(image, (0, 0), (600, 100), (30, 30, 30), -1)
        cv2.putText(image, f"POSTURE: {status}", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
        cv2.putText(image, f"SHOULDER TILT: {angle:.1f} DEG", (20, 80), cv2.FONT_HERSHEY_DUPLEX, 0.7, (240, 240, 240), 1)

        # --- DRAW CLINICAL WIREFRAME ---
        # Draw joints
        for pt in [l_shldr_pt, r_shldr_pt, l_hip_pt, r_hip_pt]:
            cv2.circle(image, pt, 6, (245, 117, 66), -1)

        # Draw Torso Box & Dynamic Shoulder Line
        cv2.line(image, l_shldr_pt, r_shldr_pt, color, 4) # Shoulder line changes color based on tilt!
        cv2.line(image, l_hip_pt, r_hip_pt, (200, 200, 200), 2)
        cv2.line(image, l_shldr_pt, l_hip_pt, (200, 200, 200), 2)
        cv2.line(image, r_shldr_pt, r_hip_pt, (200, 200, 200), 2)
        
        # Draw "Digital Spine" down the middle
        mid_shldr = ((l_shldr_pt[0] + r_shldr_pt[0]) // 2, (l_shldr_pt[1] + r_shldr_pt[1]) // 2)
        mid_hip = ((l_hip_pt[0] + r_hip_pt[0]) // 2, (l_hip_pt[1] + r_hip_pt[1]) // 2)
        cv2.line(image, mid_shldr, mid_hip, (245, 66, 230), 3) 

    # 5. Display the window
    cv2.imshow('Scoliosis AI - Live Posture Tracker', image)
    if cv2.waitKey(10) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()