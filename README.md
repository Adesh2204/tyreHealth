# AI-Based Tyre Health Monitoring System

A complete computer vision system for tyre detection, condition classification, puncture/flat detection, tread analysis, and lifespan estimation with a modern Streamlit dashboard.

## Features

- YOLOv8-based tyre detection
- CNN-based tyre condition classification (Good, Worn, Critical)
- OpenCV tread groove analysis and wear scoring
- Contour + Hough circle anomaly detection for puncture/flat tyres
- scikit-learn regression for remaining lifespan estimation
- Unified analysis pipeline returning structured JSON
- Streamlit dashboard with upload + webcam support, charts, and alerts
- Automatic model training when model files are missing

## Tech Stack

- Python 3.10+
- Ultralytics YOLOv8
- TensorFlow/Keras
- OpenCV
- scikit-learn
- Streamlit

## Project Structure

```text
project_root/
│
├── data/
│   ├── raw/
│   │   ├── good/
│   │   ├── worn/
│   │   ├── critical/
│   │   ├── puncture/
│   │   └── flat/
│   ├── processed/
│   └── annotations/
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       └── val/
│           ├── images/
│           └── labels/
│
├── models/
│   ├── yolo/
│   ├── classifier/
│   └── regression/
│
├── src/
│   ├── dataset_loader.py
│   ├── detection.py
│   ├── classification.py
│   ├── tread_analysis.py
│   ├── puncture_detection.py
│   ├── lifespan.py
│   └── pipeline.py
│
├── app/
│   └── streamlit_app.py
├── tests/
│   ├── test_pipeline.py
│   └── run_tests.py
├── train.py
├── requirements.txt
└── README.md
```

## Installation

Install dependencies with the required stack:

```bash
pip install ultralytics opencv-python tensorflow scikit-learn streamlit numpy matplotlib pillow
pip install -r requirements.txt
```

## Dataset Setup

### Classification dataset

Put images in:

- data/raw/good
- data/raw/worn
- data/raw/critical
- data/raw/puncture
- data/raw/flat

Images are resized to 224x224 and normalized to [0, 1] during loading.

### YOLO detection dataset

For custom tyre detector training, prepare YOLO annotations:

- data/annotations/train/images
- data/annotations/train/labels
- data/annotations/val/images
- data/annotations/val/labels

The system auto-generates data/annotations/dataset.yaml.

## Training

Train everything (classifier + regression + YOLO if annotations are available):

```bash
python train.py
```

If YOLO annotation folders are not ready:

```bash
python train.py --skip-yolo
```

## Running the Dashboard

```bash
streamlit run app/streamlit_app.py
```

UI includes:

- image upload and webcam capture
- detection overlays
- tyre condition badge (green/yellow/red)
- animated tread health bar
- tread and lifespan gauges
- training metrics (accuracy, precision, recall)

## Pipeline Usage

```python
import cv2
from src.pipeline import analyse_tyre

image = cv2.imread("sample.jpg")
result = analyse_tyre(image)
print(result)
```

Response format:

```json
{
  "condition": "Worn",
  "tread_score": 43.2,
  "remaining_km": 16200,
  "status": "monitor",
  "alerts": ["..."],
  "tyres": [...],
  "detections": [...],
  "inference_ms": 412.7
}
```

## Testing

Run edge-case tests:

```bash
python tests/run_tests.py
```

Covered edge cases:

- no tyre detected
- low quality image input
- multiple tyres in one frame

## Notes on Auto-Training and Fallbacks

- Classifier and lifespan models auto-train if model files are missing.
- Detector attempts custom YOLO model first.
- If custom YOLO is unavailable, pretrained YOLOv8n is used.
- If no YOLO detections are found, OpenCV circle fallback is applied.

## Production Tips

- Keep YOLO annotation quality high for better tyre localization.
- Use GPU-enabled TensorFlow/PyTorch environments for training speed.
- For strict <1 second inference, use smaller input images and warm model caches at app startup.
