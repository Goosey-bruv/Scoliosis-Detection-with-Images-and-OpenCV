import base64
import sqlite3
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. DATABASE SETUP ---
conn = sqlite3.connect('patients.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS diagnostics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT, diagnosis TEXT, cobb_angle TEXT, timestamp DATETIME
    )
''')
conn.commit()

# --- 2. AI MODEL INITIALIZATION ---
print("Initializing YOLOv8 & MediaPipe...")
yolo_model = YOLO('scoliosis_yolo.pt') 
yolo_model.model.names = {0: 'Vertebra', 1: 'Scoliosis Spine', 2: 'Normal Spine'}

base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
mp_options = vision.PoseLandmarkerOptions(base_options=base_options, num_poses=1)
pose_detector = vision.PoseLandmarker.create_from_options(mp_options)

# --- 3. FASTAPI SETUP ---
app = FastAPI(title="Scoliosis Detection Suite", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/docs")

# --- 4. X-RAY ENDPOINT ---
@app.post("/analyze-xray/")
async def analyze_xray(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = yolo_model(image)
    annotated_image = results[0].plot(line_width=1, font_size=1, conf=False)
    vertebra_boxes = [box for box in results[0].boxes if int(box.cls) == 0]
    
    if len(vertebra_boxes) < 6:
        raise HTTPException(status_code=400, detail="Could not detect a clear spine.")

    centers = sorted([[(b.xyxy[0][0] + b.xyxy[0][2])/2, (b.xyxy[0][1] + b.xyxy[0][3])/2] for b in vertebra_boxes], key=lambda x: x[1])
    centers = np.array(centers)
    mid = len(centers) // 2
    
    def get_deg(pts):
        if len(pts) < 3: return 0.0
        s, e = pts[0], pts[-1]
        devs = np.abs(np.cross(e-s, s-pts)) / np.linalg.norm(e-s)
        height = abs(e[1] - s[1])
        return 0.0 if height == 0 else np.degrees(2 * np.arctan(np.max(devs) / (height / 2)))

    u_deg, l_deg = get_deg(centers[:mid+1]), get_deg(centers[mid:])
    spine_type, display_degree = "Normal Spine", f"{max(u_deg, l_deg):.2f}"
    
    if u_deg > 10 and l_deg > 10:
        spine_type, display_degree = "S-Curve (Double Major)", f"T:{u_deg:.1f} / L:{l_deg:.1f}"
    elif max(u_deg, l_deg) > 10:
        spine_type = "Thoracic Scoliosis" if u_deg > l_deg else "Lumbar Scoliosis"

    cursor.execute('INSERT INTO diagnostics (filename, diagnosis, cobb_angle, timestamp) VALUES (?, ?, ?, ?)', 
                   (file.filename, spine_type.upper(), display_degree, datetime.now()))
    conn.commit()

    success, buffer = cv2.imencode('.jpg', annotated_image)
    return {
        "diagnosis": spine_type.upper(),
        "cobb_angle": display_degree,
        "image_data": base64.b64encode(buffer.tobytes()).decode('utf-8')
    }

# --- 5. LIVE WEBCAM STREAM GENERATOR ---
def generate_frames():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = pose_detector.detect(mp_image)
        image = frame.copy()

        if detection_result.pose_landmarks:
            pose = detection_result.pose_landmarks[0]
            h, w = image.shape[:2]
            ls, rs = (int(pose[11].x * w), int(pose[11].y * h)), (int(pose[12].x * w), int(pose[12].y * h))
            lh, rh = (int(pose[23].x * w), int(pose[23].y * h)), (int(pose[24].x * w), int(pose[24].y * h))

            dx, dy = rs[0] - ls[0], rs[1] - ls[1]
            angle = 90.0 if dx == 0 else abs(np.degrees(np.arctan(dy / dx)))
            status, color = ("IMBALANCE DETECTED", (0, 0, 255)) if angle > 3.0 else ("GOOD ALIGNMENT", (0, 255, 0))

            cv2.rectangle(image, (0, 0), (600, 100), (30, 30, 30), -1)
            cv2.putText(image, f"POSTURE: {status}", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
            cv2.putText(image, f"TILT: {angle:.1f} DEG", (20, 80), cv2.FONT_HERSHEY_DUPLEX, 0.7, (240, 240, 240), 1)

            for pt in [ls, rs, lh, rh]: cv2.circle(image, pt, 6, (245, 117, 66), -1)
            cv2.line(image, ls, rs, color, 4)
            cv2.line(image, lh, rh, (200, 200, 200), 2)
            cv2.line(image, ls, lh, (200, 200, 200), 2)
            cv2.line(image, rs, rh, (200, 200, 200), 2)
            cv2.line(image, ((ls[0]+rs[0])//2, (ls[1]+rs[1])//2), ((lh[0]+rh[0])//2, (lh[1]+rh[1])//2), (245, 66, 230), 3)

        # Encode frame as JPEG and yield it in the specific byte format browsers expect for streams
        ret, buffer = cv2.imencode('.jpg', image)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
    cap.release()

@app.get("/video-feed/")
async def video_feed():
    # Returns the continuous stream of JPEG frames
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")