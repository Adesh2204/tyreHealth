"""Dataset utilities for tyre health monitoring."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)

IMG_SIZE: Tuple[int, int] = (224, 224)
RAW_CLASSES: Tuple[str, ...] = ("good", "worn", "critical", "puncture", "flat")
CLASSIFICATION_CLASSES: Tuple[str, ...] = ("good", "worn", "critical")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
RANDOM_SEED = 42


@dataclass
class DatasetSplit:
    """Train/validation split for numpy-ready image datasets."""

    x_train: np.ndarray
    x_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray


def ensure_dataset_folders(data_root: str | Path) -> Dict[str, Path]:
    """Create required dataset folders if missing."""
    root = Path(data_root)
    raw_root = root / "raw"
    processed_root = root / "processed"
    annotations_root = root / "annotations"

    raw_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)
    annotations_root.mkdir(parents=True, exist_ok=True)

    class_paths: Dict[str, Path] = {}
    for class_name in RAW_CLASSES:
        class_path = raw_root / class_name
        class_path.mkdir(parents=True, exist_ok=True)
        class_paths[class_name] = class_path

    return {
        "root": root,
        "raw": raw_root,
        "processed": processed_root,
        "annotations": annotations_root,
        **class_paths,
    }


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def load_images_from_folder(
    folder_path: str | Path,
    label: int,
    img_size: Tuple[int, int] = IMG_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load, resize to 224x224, and normalize image files in one folder."""
    folder = Path(folder_path)
    images: List[np.ndarray] = []
    labels: List[int] = []

    if not folder.exists():
        logger.warning("Missing folder: %s", folder)
        return np.empty((0, *img_size, 3), dtype=np.float32), np.empty((0,), dtype=np.int64)

    for file_path in sorted(folder.iterdir()):
        if not file_path.is_file() or not _is_image_file(file_path):
            continue
        try:
            with Image.open(file_path) as img:
                image = img.convert("RGB").resize(img_size, Image.Resampling.LANCZOS)
                image_array = np.asarray(image, dtype=np.float32) / 255.0
            images.append(image_array)
            labels.append(label)
        except Exception as exc:
            logger.warning("Skipping unreadable image %s: %s", file_path, exc)

    if not images:
        return np.empty((0, *img_size, 3), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.stack(images), np.asarray(labels, dtype=np.int64)


def load_classification_dataset(
    raw_root: str | Path,
    classes: Sequence[str] = CLASSIFICATION_CLASSES,
    img_size: Tuple[int, int] = IMG_SIZE,
    allow_synthetic_fallback: bool = True,
    synthetic_per_class: int = 120,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load classification images from class folders under raw_root."""
    root = Path(raw_root)
    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []

    for idx, class_name in enumerate(classes):
        class_path = root / class_name
        x_cls, y_cls = load_images_from_folder(class_path, idx, img_size=img_size)
        if x_cls.size > 0:
            x_parts.append(x_cls)
            y_parts.append(y_cls)
        else:
            logger.warning("No images found for class '%s' in %s", class_name, class_path)

    if not x_parts:
        if allow_synthetic_fallback:
            logger.warning("Raw dataset empty. Using synthetic classification samples.")
            x_syn, y_syn = generate_synthetic_samples(num_samples_per_class=synthetic_per_class, img_size=img_size)
            return x_syn, y_syn, list(classes)
        return np.empty((0, *img_size, 3), dtype=np.float32), np.empty((0,), dtype=np.int64), list(classes)

    x = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    rng = np.random.default_rng(RANDOM_SEED)
    order = rng.permutation(len(x))
    x = x[order]
    y = y[order]
    return x.astype(np.float32), y.astype(np.int64), list(classes)


def split_dataset(
    x: np.ndarray,
    y: np.ndarray,
    val_split: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> DatasetSplit:
    """Perform robust train/validation split with stratification when possible."""
    if x.size == 0 or y.size == 0:
        return DatasetSplit(
            x_train=np.empty((0, *IMG_SIZE, 3), dtype=np.float32),
            x_val=np.empty((0, *IMG_SIZE, 3), dtype=np.float32),
            y_train=np.empty((0,), dtype=np.int64),
            y_val=np.empty((0,), dtype=np.int64),
        )

    unique_classes, counts = np.unique(y, return_counts=True)
    can_stratify = len(unique_classes) > 1 and np.min(counts) > 1

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=val_split,
        random_state=random_state,
        stratify=y if can_stratify else None,
    )
    return DatasetSplit(x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val)


def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Compute balanced class weights for imbalanced datasets."""
    if y.size == 0:
        return {}

    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def count_images_by_class(raw_root: str | Path, classes: Sequence[str] = RAW_CLASSES) -> Dict[str, int]:
    """Return image counts per class folder."""
    root = Path(raw_root)
    counts: Dict[str, int] = {}
    for class_name in classes:
        class_dir = root / class_name
        if not class_dir.exists():
            counts[class_name] = 0
            continue
        counts[class_name] = sum(1 for p in class_dir.iterdir() if p.is_file() and _is_image_file(p))
    return counts


def generate_synthetic_samples(
    num_samples_per_class: int = 120,
    img_size: Tuple[int, int] = IMG_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic tyre-like samples for Good/Worn/Critical classes."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for synthetic sample generation") from exc

    h, w = img_size
    rng = np.random.default_rng(RANDOM_SEED)
    images: List[np.ndarray] = []
    labels: List[int] = []

    for class_idx, class_name in enumerate(CLASSIFICATION_CLASSES):
        for _ in range(num_samples_per_class):
            canvas = np.full((h, w, 3), 22, dtype=np.uint8)
            center = (w // 2 + int(rng.integers(-8, 9)), h // 2 + int(rng.integers(-8, 9)))
            radius = int(min(h, w) * rng.uniform(0.26, 0.34))

            cv2.circle(canvas, center, radius, (55, 55, 55), -1)
            cv2.circle(canvas, center, int(radius * 0.48), (115, 115, 120), -1)

            if class_name == "good":
                for offset in range(-radius + 10, radius - 10, 10):
                    x = center[0] + offset
                    cv2.line(canvas, (x, center[1] - radius), (x, center[1] + radius), (38, 38, 38), 2)
            elif class_name == "worn":
                for offset in range(-radius + 12, radius - 12, 18):
                    x = center[0] + offset
                    cv2.line(canvas, (x, center[1] - radius), (x, center[1] + radius), (52, 52, 52), 1)
            else:
                for _ in range(14):
                    x1 = int(rng.integers(center[0] - radius, center[0] + radius))
                    y1 = int(rng.integers(center[1] - radius, center[1] + radius))
                    x2 = x1 + int(rng.integers(-14, 15))
                    y2 = y1 + int(rng.integers(-14, 15))
                    cv2.line(canvas, (x1, y1), (x2, y2), (20, 20, 20), 1)

            noise = rng.normal(0, 9, canvas.shape).astype(np.int16)
            canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            images.append(canvas.astype(np.float32) / 255.0)
            labels.append(class_idx)

    x = np.asarray(images, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    order = rng.permutation(len(x))
    return x[order], y[order]
