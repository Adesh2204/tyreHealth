"""Tyre condition classification using TensorFlow/Keras."""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

from src.dataset_loader import (
    CLASSIFICATION_CLASSES,
    IMG_SIZE,
    compute_class_weights,
    ensure_dataset_folders,
    load_classification_dataset,
    split_dataset,
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "models" / "classifier" / "tyre_classifier.h5"
METRICS_PATH = BASE_DIR / "models" / "classifier" / "training_metrics.json"

CLASS_NAMES = ["Good", "Worn", "Critical"]
NUM_CLASSES = 3

_classifier_model = None
_classifier_auto_train_attempted = False

FAST_INFERENCE_DEFAULT = os.getenv("TYRE_FAST_MODE", "1").lower() not in {"0", "false", "no"}


def build_classifier_model(input_shape: tuple[int, int, int] = (224, 224, 3), num_classes: int = NUM_CLASSES):
    """Build the required CNN architecture for condition classification."""
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_classifier(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    epochs: int = 30,
    batch_size: int = 16,
    val_split: float = 0.2,
    model_save_path: str | Path = MODEL_PATH,
    metrics_save_path: str | Path = METRICS_PATH,
) -> Dict[str, float]:
    """Train classifier with class balancing, early stopping, and checkpointing."""
    free_bytes = shutil.disk_usage(str(MODEL_PATH.parent)).free
    min_free_bytes = 700 * 1024 * 1024
    if free_bytes < min_free_bytes:
        logger.warning("Skipping classifier training due to low disk space.")
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "error": "low_disk_space",
        }

    try:
        import tensorflow as tf
    except Exception as exc:
        logger.error("TensorFlow unavailable for classifier training: %s", exc)
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "error": "tensorflow_unavailable"}

    data_root = Path(data_root)
    ensure_dataset_folders(data_root)

    x, y, class_names = load_classification_dataset(data_root / "raw", classes=CLASSIFICATION_CLASSES)
    split = split_dataset(x, y, val_split=val_split)
    if split.x_train.size == 0:
        raise RuntimeError("No training samples available for classifier training.")

    y_train_cat = tf.keras.utils.to_categorical(split.y_train, NUM_CLASSES)
    y_val_cat = tf.keras.utils.to_categorical(split.y_val, NUM_CLASSES) if split.y_val.size > 0 else None

    model = build_classifier_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=NUM_CLASSES)

    model_path = Path(model_save_path)
    metrics_path = Path(metrics_save_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(str(model_path), monitor="val_loss", save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
    ]

    class_weights = compute_class_weights(split.y_train)
    try:
        history = model.fit(
            split.x_train,
            y_train_cat,
            validation_data=(split.x_val, y_val_cat) if y_val_cat is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
        )
    except Exception as exc:
        logger.error("Classifier training failed: %s", exc)
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "train_samples": int(split.x_train.shape[0]),
            "val_samples": int(split.x_val.shape[0]),
            "classes": [name.capitalize() for name in class_names],
            "error": str(exc),
        }

    if model_path.exists():
        model = tf.keras.models.load_model(model_path)

    eval_x = split.x_val if split.x_val.size > 0 else split.x_train
    eval_y = split.y_val if split.y_val.size > 0 else split.y_train
    probs = model.predict(eval_x, verbose=0)
    pred = np.argmax(probs, axis=1)

    metrics = {
        "accuracy": float(accuracy_score(eval_y, pred)),
        "precision": float(precision_score(eval_y, pred, average="macro", zero_division=0)),
        "recall": float(recall_score(eval_y, pred, average="macro", zero_division=0)),
        "train_samples": int(split.x_train.shape[0]),
        "val_samples": int(split.x_val.shape[0]),
        "classes": [name.capitalize() for name in class_names],
        "best_val_accuracy": float(max(history.history.get("val_accuracy", [0.0]))),
    }

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Classifier model saved at %s", model_path)
    logger.info("Classifier metrics saved at %s", metrics_path)

    global _classifier_model
    _classifier_model = model
    return metrics


def load_classifier(model_path: str | Path = MODEL_PATH):
    """Load saved classifier model if available."""
    try:
        import tensorflow as tf
    except Exception:
        return None

    model_path = Path(model_path)
    if not model_path.exists():
        return None
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as exc:
        logger.warning("Discarding unreadable classifier model at %s: %s", model_path, exc)
        try:
            model_path.unlink(missing_ok=True)
        except Exception:
            pass
        return None


def ensure_classifier_model(auto_train: bool | None = None) -> Optional[object]:
    """Return cached classifier model; train automatically if missing."""
    global _classifier_model, _classifier_auto_train_attempted

    if auto_train is None:
        auto_train = not FAST_INFERENCE_DEFAULT

    if _classifier_model is not None:
        return _classifier_model

    try:
        _classifier_model = load_classifier()
    except Exception as exc:
        logger.warning("Failed to load existing classifier model: %s", exc)
        _classifier_model = None

    if _classifier_model is None and auto_train and not _classifier_auto_train_attempted:
        _classifier_auto_train_attempted = True
        free_bytes = shutil.disk_usage(str(MODEL_PATH.parent)).free
        min_free_bytes = 700 * 1024 * 1024
        if free_bytes < min_free_bytes:
            logger.warning("Classifier auto-training skipped: low disk space.")
            return None

        logger.info("Classifier model missing. Triggering automatic training.")
        try:
            train_classifier()
        except Exception as exc:
            logger.warning("Automatic classifier training skipped due to error: %s", exc)
        _classifier_model = load_classifier()

    return _classifier_model


def get_training_metrics(metrics_path: str | Path = METRICS_PATH) -> Dict[str, float]:
    """Load latest training metrics for dashboard usage."""
    path = Path(metrics_path)
    if not path.exists():
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0}


def classify_tyre(image: np.ndarray, model=None, auto_train: bool | None = None) -> Dict[str, object]:
    """Classify tyre condition for one cropped tyre image."""
    import cv2

    if image is None or image.size == 0:
        return {
            "condition": "Worn",
            "confidence": 0.0,
            "probabilities": {"Good": 0.0, "Worn": 1.0, "Critical": 0.0},
        }

    image_resized = cv2.resize(image, IMG_SIZE)
    image_norm = image_resized.astype(np.float32) / 255.0
    input_batch = np.expand_dims(image_norm, axis=0)

    if model is None:
        model = ensure_classifier_model(auto_train=auto_train)

    if model is None:
        return _fallback_classification(image)

    probs = model.predict(input_batch, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {
        "condition": CLASS_NAMES[idx],
        "confidence": float(np.max(probs)),
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)},
    }


def _fallback_classification(image: np.ndarray) -> Dict[str, object]:
    """Heuristic fallback in case model is unavailable."""
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 140)
    edge_ratio = float(np.count_nonzero(edges)) / float(edges.size)

    if edge_ratio > 0.17:
        label = "Good"
        probs = {"Good": 0.72, "Worn": 0.22, "Critical": 0.06}
    elif edge_ratio > 0.08:
        label = "Worn"
        probs = {"Good": 0.20, "Worn": 0.64, "Critical": 0.16}
    else:
        label = "Critical"
        probs = {"Good": 0.06, "Worn": 0.28, "Critical": 0.66}

    return {"condition": label, "confidence": probs[label], "probabilities": probs}
