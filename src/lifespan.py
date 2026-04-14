"""Lifespan regression for tyre replacement planning."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "regression" / "lifespan_model.pkl"
METRICS_PATH = BASE_DIR / "models" / "regression" / "training_metrics.json"
MAX_LIFESPAN_KM = 80000
CONDITION_MAP = {"Good": 2, "Worn": 1, "Critical": 0}

_lifespan_model: Pipeline | None = None
FAST_INFERENCE_DEFAULT = os.getenv("TYRE_FAST_MODE", "1").lower() not in {"0", "false", "no"}


def build_lifespan_model() -> Pipeline:
    """Build scikit-learn regression pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=180,
                    max_depth=8,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def generate_training_data(n_samples: int = 800) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data from tyre-health mechanics."""
    rng = np.random.default_rng(42)
    tread_depth = rng.uniform(0, 100, n_samples)
    crack_count = rng.integers(0, 24, n_samples)
    condition = rng.choice([0, 1, 2], size=n_samples, p=[0.22, 0.43, 0.35])

    remaining_km = (
        tread_depth * 520.0
        + condition * 9500.0
        - crack_count * 1650.0
        + rng.normal(0, 2500, n_samples)
    )
    remaining_km = np.clip(remaining_km, 0, MAX_LIFESPAN_KM)

    features = np.column_stack([tread_depth, crack_count, condition]).astype(np.float32)
    return features, remaining_km.astype(np.float32)


def train_lifespan_model(
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    model_path: str | Path = MODEL_PATH,
    metrics_path: str | Path = METRICS_PATH,
) -> Dict[str, float]:
    """Train and persist lifespan model."""
    if x is None or y is None:
        x, y = generate_training_data()

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    model = build_lifespan_model()
    model.fit(x_train, y_train)

    pred = model.predict(x_val)
    metrics = {
        "r2": float(r2_score(y_val, pred)),
        "mae": float(mean_absolute_error(y_val, pred)),
        "train_samples": int(x_train.shape[0]),
        "val_samples": int(x_val.shape[0]),
    }

    model_path = Path(model_path)
    metrics_path = Path(metrics_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    global _lifespan_model
    _lifespan_model = model
    logger.info("Lifespan model saved at %s", model_path)
    return metrics


def load_lifespan_model(model_path: str | Path = MODEL_PATH) -> Pipeline | None:
    """Load persisted lifespan model."""
    path = Path(model_path)
    if not path.exists():
        return None
    return joblib.load(path)


def ensure_lifespan_model(auto_train: bool | None = None) -> Pipeline | None:
    """Return cached model and auto-train when missing."""
    global _lifespan_model

    if auto_train is None:
        auto_train = not FAST_INFERENCE_DEFAULT

    if _lifespan_model is not None:
        return _lifespan_model

    _lifespan_model = load_lifespan_model()
    if _lifespan_model is None and auto_train:
        train_lifespan_model()
        _lifespan_model = load_lifespan_model()
    return _lifespan_model


def predict_lifespan(
    tread_depth_score: float,
    crack_count: int,
    condition: str,
    auto_train: bool | None = None,
) -> Dict[str, object]:
    """Predict remaining life (km) and replacement urgency."""
    model = ensure_lifespan_model(auto_train=auto_train)

    cond_num = CONDITION_MAP.get(condition, 1)
    features = np.array([[float(tread_depth_score), float(crack_count), float(cond_num)]], dtype=np.float32)

    if model is None:
        remaining_km = _heuristic_lifespan(tread_depth_score, crack_count, cond_num)
        confidence = 0.5
    else:
        try:
            remaining_km = float(model.predict(features)[0])
            confidence = 0.86
        except Exception as exc:
            logger.warning("Regression prediction fallback used: %s", exc)
            remaining_km = _heuristic_lifespan(tread_depth_score, crack_count, cond_num)
            confidence = 0.5

    remaining_km = int(np.clip(remaining_km, 0, MAX_LIFESPAN_KM))
    if remaining_km < 3000:
        urgency = "Immediate"
    elif remaining_km < 10000:
        urgency = "High"
    elif remaining_km < 25000:
        urgency = "Moderate"
    else:
        urgency = "Low"

    return {
        "remaining_km": remaining_km,
        "replacement_urgency": urgency,
        "confidence": round(confidence, 2),
        "estimated_months": round(remaining_km / 1500.0, 1),
    }


def _heuristic_lifespan(tread_score: float, crack_count: int, condition_numeric: int) -> float:
    base = tread_score * 520.0 + condition_numeric * 9000.0
    penalty = crack_count * 1700.0
    return float(max(0.0, min(MAX_LIFESPAN_KM, base - penalty)))
