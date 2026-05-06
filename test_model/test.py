import cv2
import numpy as np
from ultralytics import YOLO

# 1. INITIALIZATION
model = YOLO('scoliosis_yolo.pt') 
model.model.names = {0: 'Vertebra', 1: 'Scoliosis Spine', 2: 'Normal Spine'}

# 2. RUN INFERENCE
image_path = 'sample_xray.jpg'
results = model(image_path)
# Standardize the output style (thin lines, no messy labels)
annotated_image = results[0].plot(line_width=1, font_size=1, conf=False)

# 3. DOUBLE APEX MATH LOGIC
vertebra_boxes = [box for box in results[0].boxes if int(box.cls) == 0]
spine_type, display_degree = "Normal Spine", "0.00"
text_color = (240, 240, 240)

if len(vertebra_boxes) >= 6:
    centers = sorted([[ (b.xyxy[0][0] + b.xyxy[0][2])/2, (b.xyxy[0][1] + b.xyxy[0][3])/2 ] 
                     for b in vertebra_boxes], key=lambda x: x[1])
    centers = np.array(centers)
    
    mid = len(centers) // 2
    def get_deg(pts):
        s, e = pts[0], pts[-1]
        devs = np.abs(np.cross(e-s, s-pts)) / np.linalg.norm(e-s)
        return (np.max(devs) / (e[1] - s[1])) * 100

    u_deg, l_deg = get_deg(centers[:mid+1]), get_deg(centers[mid:])

    if u_deg > 10 and l_deg > 10:
        spine_type, display_degree = "S-Curve (Double Major)", f"T:{u_deg:.1f} / L:{l_deg:.1f}"
        text_color = (0, 215, 255)
    elif max(u_deg, l_deg) > 10:
        text_color = (0, 215, 255)
        spine_type = "Thoracic Scoliosis" if u_deg > l_deg else "Lumbar Scoliosis"
        display_degree = f"{max(u_deg, l_deg):.2f}"

# 4. FIX IMAGE RESOLUTION (ASPECT RATIO PRESERVATION)
h, w = annotated_image.shape[:2]

# Define the maximum size you want on your screen
max_screen_w, max_screen_h = 1000, 800

# Calculate the scaling factor that fits both constraints
scale = min(max_screen_w / w, max_screen_h / h)

# Apply the scale
new_w, new_h = int(w * scale), int(h * scale)
resized_xray = cv2.resize(annotated_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

# 5. CREATE THE REPORT WINDOW (FIXED SIZE)
report_canvas = np.zeros((200, 500, 3), dtype=np.uint8)
report_canvas[:] = (35, 35, 35)

font = cv2.FONT_HERSHEY_DUPLEX
cv2.putText(report_canvas, "SCOLIOSIS ANALYSIS REPORT", (20, 40), font, 0.5, (150, 150, 150), 1)
cv2.putText(report_canvas, f"DIAGNOSIS: {spine_type.upper()}", (20, 95), font, 0.7, (240, 240, 240), 1)
cv2.putText(report_canvas, f"COBB ANGLE: {display_degree} DEG", (20, 150), font, 0.7, text_color, 1)

# 6. DISPLAY
cv2.imshow("X-Ray Analysis View", resized_xray)
cv2.imshow("Clinical Report", report_canvas)

# Optional: Position them side-by-side
cv2.moveWindow("X-Ray Analysis View", 50, 50)
cv2.moveWindow("Clinical Report", 50 + new_w + 10, 50)

cv2.waitKey(0)
cv2.destroyAllWindows()