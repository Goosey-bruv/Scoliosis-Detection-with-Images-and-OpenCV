import cv2
from ultralytics import YOLO
import numpy as np
import math
from scipy.stats import linregress

# 1. Load your newly trained 'brain'
model = YOLO('best.pt') 

def get_cobb_angle(image_path):
    results = model.predict(source=image_path, save=False)
    boxes = results[0].boxes
    
    if len(boxes) < 2:
        return 0, "Insufficient vertebrae detected"

    # Extract centers (x, y) of each vertebra box
    # Filter for Class 0 (Vertebra)
    centers = []
    for box in boxes:
        if int(box.cls) == 0:
            x_center, y_center, w, h = box.xywh[0].tolist()
            centers.append([x_center, y_center])

    # Sort vertebrae from top to bottom (by Y coordinate)
    centers = sorted(centers, key=lambda x: x[1])
    centers = np.array(centers)

    # Calculate angles between consecutive vertebrae
    angles = []
    for i in range(len(centers) - 1):
        dx = centers[i+1][0] - centers[i][0]
        dy = centers[i+1][1] - centers[i][1]
        angle = math.degrees(math.atan2(dx, dy)) # Vertical tilt
        angles.append(angle)

    # The Cobb Angle is the difference between the most extreme tilts
    if angles:
        max_tilt = max(angles)
        min_tilt = min(angles)
        cobb_angle = abs(max_tilt - min_tilt)
    else:
        cobb_angle = 0

    # Medical Classification
    if cobb_angle < 10:
        severity = "Normal / Non-Scoliotic"
    elif 10 <= cobb_angle <= 25:
        severity = "Mild Scoliosis"
    elif 26 <= cobb_angle <= 40:
        severity = "Moderate Scoliosis"
    else:
        severity = "Severe Scoliosis"

    return round(cobb_angle, 2), severity

# Test it on a sample from your 'test' folder
angle, level = get_cobb_angle('scoliosis2.v16i.tensorflow/test/images/your_sample.jpg')
print(f"Calculated Cobb Angle: {angle}°")
print(f"Diagnosis: {level}")