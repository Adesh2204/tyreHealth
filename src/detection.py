"""Tyre detection and YOLOv8 training utilities."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
YOLO_MODEL_PATH = BASE_DIR / "models" / "yolo" / "tyre_detector.pt"
YOLO_PRETRAINED = "yolov8n.pt"
DEFAULT_ANNOTATIONS_ROOT = BASE_DIR / "data" / "annotations"
DEFAULT_YAML_PATH = DEFAULT_ANNOTATIONS_ROOT / "dataset.yaml"

VEHICLE_CLASSES = {2, 3, 5, 7}
_yolo_model = None
_auto_train_attempted = False

FAST_INFERENCE_DEFAULT = os.getenv("TYRE_FAST_MODE", "1").lower() not in {"0", "false", "no"}
ALLOW_PRETRAINED_YOLO = os.getenv("TYRE_ALLOW_PRETRAINED_YOLO", "0").lower() in {"1", "true", "yes"}


def _resolve_device() -> str | int:
    """Return GPU device index when available, else CPU."""
    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def generate_dataset_yaml(
    annotations_root: str | Path = DEFAULT_ANNOTATIONS_ROOT,
    output_path: str | Path = DEFAULT_YAML_PATH,
    class_names: List[str] | None = None,
) -> Path:
    """Generate YOLO dataset yaml file from annotation directory structure."""
    root = Path(annotations_root)
    output = Path(output_path)
    class_names = class_names or ["tyre"]

    train_images = root / "train" / "images"
    val_images = root / "val" / "images"
    train_labels = root / "train" / "labels"
    val_labels = root / "val" / "labels"

    required_paths = [train_images, val_images, train_labels, val_labels]
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "YOLO annotation dataset is incomplete. Missing paths: " + ", ".join(missing)
        )

    yaml_payload = {
        "path": str(root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {idx: name for idx, name in enumerate(class_names)},
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(yaml_payload, sort_keys=False), encoding="utf-8")
    logger.info("Generated YOLO dataset file at %s", output)
    return output


def train_yolo_detector(
    annotations_root: str | Path = DEFAULT_ANNOTATIONS_ROOT,
    dataset_yaml: str | Path | None = None,
    epochs: int = 50,
    imgsz: int = 640,
    pretrained_weights: str = YOLO_PRETRAINED,
) -> Dict[str, object]:
    """Train YOLOv8 tyre detector and persist best checkpoint."""
    from ultralytics import YOLO

    yaml_path = Path(dataset_yaml) if dataset_yaml else generate_dataset_yaml(annotations_root=annotations_root)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YOLO dataset yaml not found: {yaml_path}")

    model = YOLO(pretrained_weights)
    device = _resolve_device()
    run = model.train(data=str(yaml_path), epochs=epochs, imgsz=imgsz, device=device)

    save_dir = Path(run.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise RuntimeError("YOLO training finished without best.pt output.")

    YOLO_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, YOLO_MODEL_PATH)

    global _yolo_model
    _yolo_model = YOLO(str(YOLO_MODEL_PATH))

    return {
        "status": "trained",
        "model_path": str(YOLO_MODEL_PATH),
        "save_dir": str(save_dir),
        "device": device,
    }


def _load_yolo_model(prefer_custom: bool = True):
    """Load YOLO model for inference."""
    from ultralytics import YOLO

    if prefer_custom and YOLO_MODEL_PATH.exists():
        return YOLO(str(YOLO_MODEL_PATH))
    return YOLO(YOLO_PRETRAINED)


def ensure_detector_model(auto_train: bool | None = None, allow_pretrained: bool | None = None):
    """Return cached YOLO model, training custom model when feasible."""
    global _yolo_model, _auto_train_attempted

    if auto_train is None:
        auto_train = not FAST_INFERENCE_DEFAULT
    if allow_pretrained is None:
        allow_pretrained = ALLOW_PRETRAINED_YOLO

    if _yolo_model is not None:
        return _yolo_model

    if YOLO_MODEL_PATH.exists():
        try:
            _yolo_model = _load_yolo_model(prefer_custom=True)
            return _yolo_model
        except Exception as exc:
            logger.warning("Failed to load saved custom detector model: %s", exc)

    if auto_train and not _auto_train_attempted:
        _auto_train_attempted = True
        try:
            logger.info("Attempting automatic YOLO training because detector model is missing.")
            train_yolo_detector()
        except Exception as exc:
            logger.warning("Automatic YOLO training skipped: %s", exc)

    if _yolo_model is None and allow_pretrained:
        try:
            _yolo_model = _load_yolo_model(prefer_custom=YOLO_MODEL_PATH.exists())
        except Exception as exc:
            logger.warning("Unable to initialize YOLO runtime. Falling back to OpenCV-only detection: %s", exc)
            _yolo_model = None
    return _yolo_model


def detect_tyres(image: np.ndarray, confidence: float = 0.25) -> List[Dict[str, object]]:
    """Detect tyre regions using YOLO model with OpenCV fallback."""
    if image is None or image.size == 0:
        return []

    model = ensure_detector_model(auto_train=None, allow_pretrained=None)
    if model is None:
        return detect_tyre_opencv(image)

    detections: List[Dict[str, object]] = []

    try:
        results = model.predict(image, conf=confidence, verbose=False)
        for result in results:
            if result.boxes is None:
                continue

            names = result.names if hasattr(result, "names") else {}
            for i in range(len(result.boxes)):
                cls_id = int(result.boxes.cls[i].item())
                conf = float(result.boxes.conf[i].item())
                x1, y1, x2, y2 = [int(v) for v in result.boxes.xyxy[i].tolist()]

                cls_name = str(names.get(cls_id, "unknown")).lower()
                if "tyre" in cls_name or "tire" in cls_name or "wheel" in cls_name:
                    detections.append({"bbox": [x1, y1, x2, y2], "confidence": conf, "label": "tyre"})
                    continue

                if cls_id in VEHICLE_CLASSES:
                    h = max(1, y2 - y1)
                    w = max(1, x2 - x1)
                    tyre_h = int(h * 0.30)
                    tyre_w = int(w * 0.25)
                    detections.append(
                        {
                            "bbox": [x1, y2 - tyre_h, x1 + tyre_w, y2],
                            "confidence": conf * 0.8,
                            "label": "tyre",
                        }
                    )
                    detections.append(
                        {
                            "bbox": [x2 - tyre_w, y2 - tyre_h, x2, y2],
                            "confidence": conf * 0.8,
                            "label": "tyre",
                        }
                    )
    except Exception as exc:
        logger.warning("YOLO inference failed, using OpenCV fallback: %s", exc)
        return detect_tyre_opencv(image)

    if not detections:
        return detect_tyre_opencv(image)

    return detections


def detect_tyre_opencv(image: np.ndarray) -> List[Dict[str, object]]:
    """Fallback tyre detection using Hough circles."""
    detections: List[Dict[str, object]] = []

    orig_h, orig_w = image.shape[:2]
    max_dim = max(orig_h, orig_w)
    scale = 1.0 if max_dim <= 960 else 960.0 / float(max_dim)

    work_img = image if scale == 1.0 else cv2.resize(image, (int(orig_w * scale), int(orig_h * scale)))
    gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    h, w = gray.shape[:2]

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(30, min(h, w) // 4),
        param1=120,
        param2=38,
        minRadius=max(15, min(h, w) // 10),
        maxRadius=max(20, min(h, w) // 2),
    )

    if circles is not None:
        for circle in np.round(circles[0]).astype(int):
            cx, cy, radius = circle.tolist()
            x1, y1 = max(0, cx - radius), max(0, cy - radius)
            x2, y2 = min(w, cx + radius), min(h, cy + radius)

            if scale != 1.0:
                x1 = int(x1 / scale)
                y1 = int(y1 / scale)
                x2 = int(x2 / scale)
                y2 = int(y2 / scale)

            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(x1 + 1, min(x2, orig_w))
            y2 = max(y1 + 1, min(y2, orig_h))

            detections.append({"bbox": [x1, y1, x2, y2], "confidence": 0.55, "label": "tyre"})

    # Keep only the largest candidates to reduce downstream compute latency.
    if len(detections) > 3:
        detections = sorted(
            detections,
            key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
            reverse=True,
        )[:3]

    return detections


def crop_tyre(image: np.ndarray, bbox: List[int]) -> np.ndarray:
    """Crop one tyre patch from an image."""
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return image[y1:y2, x1:x2].copy()


def draw_detections(image: np.ndarray, detections: List[Dict[str, object]], results: List[Dict[str, object]] | None = None) -> np.ndarray:
    """Overlay detection boxes and predicted condition labels."""
    canvas = image.copy()
    colors = {"Good": (38, 188, 85), "Worn": (0, 204, 255), "Critical": (34, 34, 220)}

    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        condition = "Tyre"
        if results and idx < len(results):
            condition = str(results[idx].get("condition", "Tyre"))
        color = colors.get(condition, (230, 180, 60))

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        conf = float(det.get("confidence", 0.0))
        label = f"{condition} {conf:.2f}"
        cv2.putText(canvas, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    return canvas
