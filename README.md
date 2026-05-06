
# Scoliosis Detection & Severity Analysis System

An end-to-end medical computer vision pipeline designed to screen for spinal asymmetry in real-time and analyze scoliosis severity from X-ray imagery.

## 🛠️ Technical Stack

***Languages:** Python 3.10+
***Computer Vision:** OpenCV, MediaPipe
***Deep Learning:** PyTorch, YOLOv8 (Ultralytics)
***Math & Science:** NumPy, SciPy (Polynomial Regression)
***Hardware Training:** Google Colab (v5e1 TPU)

## 📂 System Architecture

### Phase 1: Real-time Posture Screening

This module acts as a preliminary screening tool using a standard webcam to detect physical markers associated with scoliosis.

***Marker Detection:** Uses MediaPipe to track 33 body landmarks.
***Clinical Protocols:**
    *3-second countdown to ensure user readiness.
    *Validation of form-fitting clothing for hip accuracy.
***Real-time Math:**
    *Shoulder Tilt Threshold: $3.0^{\circ}$
    *Hip Asymmetry Threshold: $1.5^{\circ}$
    *Spine "Plumb Line" deviation calculation

### Phase 2: X-Ray Severity Analysis

A deep learning approach to automate the **Cobb Angle** calculation, which is the clinical gold standard for scoliosis diagnosis.

***Model:** YOLOv8 Nano (trained on spinal X-ray datasets).
***Classes Detected:** `Vertebra`, `Scoliosis Spine`, and `Normal Spine`.
***Algorithm:**
    1.  Detects bounding boxes for every vertebra.
    2.  Extracts centroids and applies a 3rd-degree polynomial fit.
    3.  Calculates the angle of maximum deviation between the most tilted vertebrae.

---

## ⚙️ Installation & Setup

### 1. Environment Configuration

```bash
python -m venv venv
.\venv\Scripts\activate
pip install opencv-python mediapipe ultralytics scipy
```

### 2. Running the screening tool

```bash
python posture_tracking/posture.py
```

---

## 📝 Diagnostic Scale

The system classifies results based on the following Cobb Angle ranges:

| Angle Range | Classification | Action Recommended |
| :--- | :--- | :--- |
| **< 10°** | Normal | No action / Observation |
| **10° - 25°** | Mild Scoliosis | Physical Therapy / Monitoring |
| **26° - 40°** | Moderate Scoliosis | Bracing / Specialist Consultation |
| **> 40°** | Severe Scoliosis | Surgical Consultation |
