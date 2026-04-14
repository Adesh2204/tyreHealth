<div align="center">

<h1>🛞 AI-Based Tyre Health Monitoring System</h1>

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow" />
  <img src="https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit" />
</p>

<p>
A production-grade <strong>Computer Vision</strong> pipeline that detects tyres in real time, classifies their health condition, analyses tread wear at the groove level, identifies punctures and flats, and predicts remaining safe driving distance — all served through an interactive Streamlit dashboard.
</p>

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Computer Vision Modules — In Depth](#-computer-vision-modules--in-depth)
  - [1. Object Detection — YOLOv8](#1-object-detection--yolov8)
  - [2. Condition Classification — CNN](#2-condition-classification--cnn)
  - [3. Tread Groove Analysis — OpenCV](#3-tread-groove-analysis--opencv)
  - [4. Puncture & Flat Detection — Contour + Hough Circles](#4-puncture--flat-detection--contour--hough-circles)
  - [5. Lifespan Estimation — Regression](#5-lifespan-estimation--regression)
- [Dashboard Screenshots](#-dashboard-screenshots)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Dataset Setup](#-dataset-setup)
- [Training](#-training)
- [Running the Dashboard](#-running-the-dashboard)
- [Pipeline API](#-pipeline-api)
- [Testing](#-testing)
- [Fallback & Resilience Strategy](#-fallback--resilience-strategy)
- [Production Tips](#-production-tips)
- [Authors](#-authors)

---

## 🔍 Overview

Road safety is directly tied to tyre condition. Worn treads, hidden punctures, and degraded rubber are leading causes of tyre failure — yet most vehicle owners have no objective, data-driven way to evaluate tyre health.

This project addresses that gap with a **multi-stage computer vision pipeline** that processes a single tyre image (or live webcam feed) and returns:

| Output | Description |
|---|---|
| `condition` | Overall tyre health — `Good`, `Worn`, or `Critical` |
| `tread_score` | Quantified tread depth score (0–100) |
| `remaining_km` | Predicted safe driving distance remaining |
| `status` | Action flag — `ok`, `monitor`, or `replace` |
| `alerts` | Human-readable warning messages |
| `inference_ms` | End-to-end processing latency |

---

## 🏗 System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                                 │
│            (Image Upload  /  Webcam Frame  /  API Call)            │
└───────────────────────────┬────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────────┐
│  STAGE 1 — DETECTION                                               │
│  YOLOv8 object detector localises each tyre in the frame           │
│  Fallback: OpenCV Hough Circle detection if YOLO yields no result  │
└───────────────────────────┬────────────────────────────────────────┘
                            │  Cropped tyre ROI(s)
                            ▼
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
  ┌───────────────┐ ┌──────────────┐ ┌───────────────────┐
  │  STAGE 2      │ │  STAGE 3     │ │  STAGE 4          │
  │  CNN Condition│ │  OpenCV Tread│ │  Contour + Hough  │
  │  Classifier   │ │  Groove      │ │  Puncture / Flat  │
  │  Good/Worn/   │ │  Analysis    │ │  Detection        │
  │  Critical     │ │  Wear Score  │ │                   │
  └───────┬───────┘ └──────┬───────┘ └────────┬──────────┘
          │                │                  │
          └────────────────┴──────────────────┘
                            │  Feature vector
                            ▼
┌────────────────────────────────────────────────────────────────────┐
│  STAGE 5 — REGRESSION                                              │
│  scikit-learn model predicts remaining safe driving km             │
└───────────────────────────┬────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────────┐
│  OUTPUT — Structured JSON  +  Streamlit Dashboard                  │
└────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Computer Vision Modules — In Depth

### 1. Object Detection — YOLOv8

**File:** `src/detection.py`

[You Only Look Once v8 (YOLOv8)](https://docs.ultralytics.com/) is used to localise every tyre in a frame with a single forward pass, making it suitable for both static images and real-time video.

**How it works in this project:**

- The model backbone extracts a rich feature pyramid from the input image at multiple scales (P3 / P4 / P5), enabling detection of tyres at varying distances and sizes.
- The detection head predicts bounding boxes with objectness scores and class probabilities simultaneously (anchor-free approach in YOLOv8).
- Non-Maximum Suppression (NMS) filters overlapping predictions so each tyre produces exactly one bounding box.
- Each box is cropped and forwarded to the classification, tread, and puncture modules as a **Region of Interest (ROI)**.

**Fallback chain:**

```
Custom fine-tuned YOLOv8
        │ (not found)
        ▼
Pretrained YOLOv8n (general object detector)
        │ (no tyre class detected)
        ▼
OpenCV Hough Circle Transform (geometric fallback)
```

**Key parameters:**

| Parameter | Value |
|---|---|
| Input resolution | 640 × 640 (auto-resized) |
| Confidence threshold | 0.25 |
| IoU threshold (NMS) | 0.45 |
| Base model | `yolov8n.pt` (nano — speed-optimised) |

---

### 2. Condition Classification — CNN

**File:** `src/classification.py`

A **Convolutional Neural Network** (CNN) trained from scratch — or fine-tuned on a pretrained backbone — classifies each detected tyre ROI into one of five categories:

| Class | Description |
|---|---|
| `Good` | Full tread, no visible damage |
| `Worn` | Reduced tread, uneven wear patterns visible |
| `Critical` | Dangerously low tread or structural deformation |
| `Puncture` | Foreign object penetration or sharp deformation |
| `Flat` | Tyre has fully lost air; rim contact visible |

**Network design choices:**

- **Input:** 224 × 224 × 3 (RGB), normalised to `[0, 1]`
- **Feature extraction:** Stacked Conv2D → BatchNorm → ReLU → MaxPool blocks to progressively down-sample and learn spatial texture patterns (rubber grain, tread pattern, sidewall bulge)
- **Global Average Pooling** before the dense head reduces parameters and improves generalisation
- **Output:** Softmax over 5 classes; the top-1 class drives the dashboard badge colour

**Why texture features matter for tyres:**

Tyre rubber has distinctive surface texture at different wear stages. CNNs are especially effective here because early convolutional layers learn edge detectors (tread groove edges), mid-level layers learn groove patterns, and deeper layers learn holistic wear state.

---

### 3. Tread Groove Analysis — OpenCV

**File:** `src/tread_analysis.py`

This is the core **image processing** module that quantifies tread depth without specialist hardware (no laser profilometer needed).

**Processing pipeline:**

```
ROI Image
    │
    ▼
Greyscale Conversion
    │
    ▼
CLAHE (Contrast Limited Adaptive Histogram Equalisation)
    │  Enhances groove visibility under uneven lighting
    ▼
Gaussian Blur  (σ = 1.0)
    │  Noise suppression before edge detection
    ▼
Canny Edge Detection  (low=50, high=150)
    │  Detects groove boundaries as sharp intensity transitions
    ▼
Morphological Closing  (5×5 kernel)
    │  Bridges broken groove edge segments
    ▼
Tread Region Masking  (centre 60 % of ROI width)
    │  Isolates tread contact patch, excludes sidewall
    ▼
Edge Density Computation
    │  edge_pixels / total_mask_pixels  → normalised density
    ▼
Tread Score  =  edge_density × scaling_factor  (0 – 100)
```

**Interpretation:**

- **High edge density** → deep, well-defined grooves → **high tread score** → tyre is healthy
- **Low edge density** → shallow or absent grooves → **low tread score** → tyre is worn
- The score is clamped to `[0, 100]` and displayed as an animated health bar in the dashboard

---

### 4. Puncture & Flat Detection — Contour + Hough Circles

**File:** `src/puncture_detection.py`

Puncture and flat detection relies on **geometric anomaly detection** — analysing the shape and symmetry of the tyre boundary.

**Algorithm:**

```
Greyscale → Gaussian Blur (k=5)
    │
    ▼
Binary Threshold (Otsu's method)
    │  Separates tyre from background adaptively
    ▼
Morphological Opening (5×5 kernel, 2 iterations)
    │  Removes small noise artifacts
    ▼
Contour Extraction (external contours only)
    │
    ▼
Largest Contour Selection
    │  Assumed to be the tyre boundary
    ▼
        ┌──────────────────────────────────────┐
        │  Circularity  =  4π × Area / P²      │
        │  Aspect Ratio  =  W / H of bbox       │
        └──────────────────────────────────────┘
    │
    ▼
Hough Circle Transform
    │  Detects the expected circular rim profile
    ▼
Anomaly Flags:
  • circularity < 0.7  →  deformed shape (possible flat)
  • aspect_ratio deviation > threshold  →  asymmetric bulge
  • Hough circles absent or offset  →  rim contact / flat
```

**Why this works:**

A healthy tyre mounted on a rim presents a near-perfect circular silhouette. A flat tyre deforms under vehicle weight, producing lower circularity and an off-centre rim position. Punctures cause localised boundary irregularities detectable as contour convexity defects.

---

### 5. Lifespan Estimation — Regression

**File:** `src/lifespan.py`

A **scikit-learn regression model** (e.g., Random Forest Regressor or Ridge Regression) maps extracted CV features to a predicted remaining lifespan in kilometres.

**Input feature vector:**

| Feature | Source |
|---|---|
| Tread score | `tread_analysis.py` |
| Condition class (encoded) | `classification.py` |
| Circularity score | `puncture_detection.py` |
| Edge density | `tread_analysis.py` |
| Aspect ratio | `puncture_detection.py` |

**Training target:** Labelled "remaining km" values derived from tyre wear standards (e.g., tyres are typically replaced at 1.6 mm tread depth in many regions).

**Output:** A continuous value in kilometres, shown as a gauge on the dashboard.

---

## 📸 Dashboard Screenshots

### Analysis 1 — Tyre Detection & Condition Overview
![Analysis 1](https://raw.githubusercontent.com/Adesh2204/tyreHealth/main/Analysis1.png)

---

### Analysis 2 — Tread Wear Scoring & Health Bar
![Analysis 2](https://raw.githubusercontent.com/Adesh2204/tyreHealth/main/Analysis2.png)

---

### Analysis 3 — Puncture & Anomaly Detection Output
![Analysis 3](https://raw.githubusercontent.com/Adesh2204/tyreHealth/main/Analysis3.png)

---

### Analysis 4 — Lifespan Estimation & Alert Panel
![Analysis 4](https://raw.githubusercontent.com/Adesh2204/tyreHealth/main/Analysis4.png)

---

## 🛠 Tech Stack

| Layer | Technology | Role |
|---|---|---|
| Object Detection | [Ultralytics YOLOv8](https://docs.ultralytics.com/) | Tyre localisation in frame |
| Deep Learning | TensorFlow / Keras | CNN condition classifier |
| Image Processing | OpenCV 4.x | Tread analysis, puncture geometry |
| Machine Learning | scikit-learn | Lifespan regression |
| Dashboard | Streamlit | Interactive web UI |
| Language | Python 3.10+ | Core runtime |
| Numerical | NumPy, Matplotlib, Pillow | Array ops, visualisation, image I/O |

---

## 📁 Project Structure

```
tyreHealth/
│
├── data/
│   ├── raw/
│   │   ├── good/               ← Healthy tyre images
│   │   ├── worn/               ← Moderately worn tyre images
│   │   ├── critical/           ← Critically worn tyre images
│   │   ├── puncture/           ← Punctured tyre images
│   │   └── flat/               ← Flat tyre images
│   ├── processed/              ← Augmented / preprocessed images
│   └── annotations/
│       ├── train/
│       │   ├── images/         ← YOLO training images
│       │   └── labels/         ← YOLO .txt annotation files
│       └── val/
│           ├── images/         ← YOLO validation images
│           └── labels/         ← YOLO .txt annotation files
│
├── models/
│   ├── yolo/                   ← Custom trained YOLOv8 weights
│   ├── classifier/             ← Saved CNN .h5 / SavedModel
│   └── regression/             ← Saved scikit-learn regressor
│
├── src/
│   ├── dataset_loader.py       ← Image loading, resizing, normalisation
│   ├── detection.py            ← YOLOv8 + Hough circle fallback
│   ├── classification.py       ← CNN model definition & inference
│   ├── tread_analysis.py       ← CLAHE → Canny → tread scoring
│   ├── puncture_detection.py   ← Contour + Hough anomaly detection
│   ├── lifespan.py             ← Regression feature engineering & inference
│   └── pipeline.py             ← Unified analyse_tyre() entry point
│
├── app/
│   └── streamlit_app.py        ← Streamlit dashboard UI
│
├── tests/
│   ├── test_pipeline.py        ← Unit tests for each CV module
│   └── run_tests.py            ← Test runner
│
├── train.py                    ← Full training script (YOLO + CNN + Regressor)
├── yolov8n.pt                  ← Pretrained YOLOv8 nano weights
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

**Prerequisites:** Python 3.10+, pip, (optional) CUDA-enabled GPU

```bash
# 1. Clone the repository
git clone https://github.com/Adesh2204/tyreHealth.git
cd tyreHealth

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt

# Or install manually:
pip install ultralytics opencv-python tensorflow scikit-learn \
            streamlit numpy matplotlib pillow
```

---

## 📦 Dataset Setup

### Classification Dataset

Organise your tyre images into labelled folders:

```
data/raw/
├── good/        ← Images of healthy tyres with full tread
├── worn/        ← Images showing partial tread loss or uneven wear
├── critical/    ← Images with dangerously low tread
├── puncture/    ← Images showing foreign objects or penetration
└── flat/        ← Images of fully deflated tyres
```

All images are automatically **resized to 224 × 224** and **normalised to [0, 1]** during loading via `dataset_loader.py`.

### YOLO Detection Dataset

For custom detector training, prepare bounding-box annotations in YOLO format (`.txt` files — one per image, each line: `class cx cy w h` in normalised coordinates):

```
data/annotations/
├── train/
│   ├── images/     ← Training images (.jpg / .png)
│   └── labels/     ← Corresponding .txt annotation files
└── val/
    ├── images/     ← Validation images
    └── labels/     ← Corresponding .txt annotation files
```

> `dataset.yaml` is auto-generated at `data/annotations/dataset.yaml` when training begins.

---

## 🏋️ Training

Train the full pipeline — CNN classifier, lifespan regressor, and YOLO detector:

```bash
python train.py
```

To skip YOLO training (if annotations are not yet ready):

```bash
python train.py --skip-yolo
```

**What happens during training:**

1. `dataset_loader.py` reads and preprocesses all raw images
2. CNN classifier is trained and saved to `models/classifier/`
3. Regression model is fitted and saved to `models/regression/`
4. (Optional) YOLOv8 is fine-tuned on your annotations and saved to `models/yolo/`

> If model files already exist, they are loaded directly — re-training is skipped automatically.

---

## 🚀 Running the Dashboard

```bash
streamlit run app/streamlit_app.py
```

Open your browser at `http://localhost:8501`

**Dashboard features:**

| Feature | Description |
|---|---|
| 📤 Image Upload | Upload a tyre image for instant analysis |
| 📷 Webcam Capture | Live frame capture and real-time analysis |
| 🟢🟡🔴 Condition Badge | Colour-coded tyre health indicator |
| 📊 Tread Health Bar | Animated bar showing tread score (0–100) |
| 🕹 Tread & Lifespan Gauges | Visual gauges for quick at-a-glance status |
| ⚠️ Alert Panel | Human-readable safety warnings |
| 📈 Training Metrics | Accuracy, precision, and recall from last training run |

---

## 🔌 Pipeline API

Programmatic usage — integrate into any Python application:

```python
import cv2
from src.pipeline import analyse_tyre

# Load image
image = cv2.imread("sample_tyre.jpg")

# Run the full CV pipeline
result = analyse_tyre(image)

print(result)
```

**Response schema:**

```json
{
  "condition":    "Worn",
  "tread_score":  43.2,
  "remaining_km": 16200,
  "status":       "monitor",
  "alerts": [
    "Tread depth approaching minimum safe threshold",
    "Uneven wear pattern detected on inner edge"
  ],
  "tyres":      [...],
  "detections": [...],
  "inference_ms": 412.7
}
```

**Status values:**

| Status | Meaning |
|---|---|
| `ok` | Tyre is healthy — no action needed |
| `monitor` | Tyre shows wear — inspect within 30 days |
| `replace` | Tyre is unsafe — immediate replacement recommended |

---

## 🧪 Testing

Run the full edge-case test suite:

```bash
python tests/run_tests.py
```

**Covered test scenarios:**

| Test Case | What is validated |
|---|---|
| No tyre detected | Pipeline returns graceful empty result, no crash |
| Low-quality image | Blurred / dark input handled without exception |
| Multiple tyres | All bounding boxes processed independently |
| Flat tyre input | Puncture module returns correct anomaly flags |
| Minimum image size | Small thumbnails handled without shape errors |

---

## 🔄 Fallback & Resilience Strategy

The system is designed to always return a result, even in degraded conditions:

```
Model file present?
    YES → Load and use it
    NO  → Auto-train from available data, then use it

YOLO detects tyre?
    YES → Use YOLO bounding box
    NO  → Apply OpenCV Hough Circle Transform as fallback

Circle detected?
    YES → Proceed with ROI
    NO  → Analyse full image as single tyre ROI
```

This three-tier fallback ensures the dashboard never shows a blank result.

---

## 🏭 Production Tips

- **GPU acceleration:** Use a CUDA-enabled TensorFlow build (`tensorflow-gpu`) and ensure PyTorch is installed with CUDA support for YOLOv8. Expect 5–10× speedup over CPU.
- **Model warm-up:** Call `analyse_tyre()` once with a dummy image at app startup to load all models into memory before the first user request. This eliminates first-call latency.
- **Input image size:** For strict sub-1-second inference on CPU, resize input images to 416 × 416 before passing to the pipeline.
- **Annotation quality:** YOLO detector accuracy is highly sensitive to annotation quality. Use consistent labelling conventions and include a diverse range of tyre angles, lighting conditions, and vehicle types.
- **Tread dataset balance:** Ensure roughly equal class distribution across `good`, `worn`, `critical`, `puncture`, and `flat` to prevent classifier bias.
- **Deployment:** Wrap `app/streamlit_app.py` behind a reverse proxy (e.g., Nginx) for production deployments. Use `streamlit run --server.port 8080 --server.headless true` for headless server environments.

---

<div align="center">

---

### 👨‍💻 Authors

<br/>

**Adesh Kumar Shukla**
<br/>
*Computer Vision & AI Systems Developer*

&nbsp;&nbsp;•&nbsp;&nbsp;

**Krissh Gera**
<br/>
*Computer Vision & AI Systems Developer*

<br/>

> *"Making road safety smarter — one tyre at a time."*

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-Adesh2204-black?style=flat-square&logo=github)](https://github.com/Adesh2204)

---

</div>
