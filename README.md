
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

### 1. Clone the Repository

```bash
   git clone [https://github.com/Goosey-bruv/Scoliosis-Detection-with-Images-and-OpenCV.git](https://github.com/Goosey-bruv/Scoliosis-Detection-with-Images-and-OpenCV.git)
   cd Scoliosis-Detection-with-Images-and-OpenCV
```

### 2. Environment Configuration

python -m venv venv

On Windows:

venv\Scripts\activate

On Mac/Linux:

source venv/bin/activate

### 3. Install the required dependencies

pip install -r requirements.txt

### 4. Run the Application

uvicorn main:app

### 5. Access the Interface

Open your browser and navigate to <http://127.0.0.1:8000> to view the dashboard!

---

## 📝 Diagnostic Scale

The system classifies results based on the following Cobb Angle ranges:

| Angle Range | Classification | Action Recommended |
| :--- | :--- | :--- |
| **< 10°** | Normal | No action / Observation |
| **10° - 25°** | Mild Scoliosis | Physical Therapy / Monitoring |
| **26° - 40°** | Moderate Scoliosis | Bracing / Specialist Consultation |
| **> 40°** | Severe Scoliosis | Surgical Consultation |
