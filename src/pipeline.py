"""Unified tyre analysis pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from src.classification import classify_tyre
from src.detection import crop_tyre, detect_tyres, draw_detections
from src.lifespan import predict_lifespan
from src.puncture_detection import detect_puncture_and_flat
from src.tread_analysis import analyze_tread

CONDITION_RANK = {"Good": 0, "Worn": 1, "Critical": 2}
MAX_TYRES_PER_ANALYSIS = 3


def analyse_tyre(image: np.ndarray) -> Dict[str, object]:
    """Run full end-to-end tyre analysis and return structured JSON-ready dict."""
    start = time.perf_counter()

    if image is None or image.size == 0:
        return {
            "condition": "Unknown",
            "tread_score": 0.0,
            "remaining_km": 0,
            "status": "invalid_input",
            "alerts": ["Input image is empty or unreadable"],
            "tyres": [],
            "detections": [],
            "inference_ms": 0.0,
        }

    image_bgr = _ensure_bgr(image)
    detections = detect_tyres(image_bgr)
    if len(detections) > MAX_TYRES_PER_ANALYSIS:
        detections = sorted(detections, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)[:MAX_TYRES_PER_ANALYSIS]

    if not detections:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "condition": "Unknown",
            "tread_score": 0.0,
            "remaining_km": 0,
            "status": "no_tyre_detected",
            "alerts": ["No tyre detected in image"],
            "tyres": [],
            "detections": [],
            "inference_ms": round(elapsed_ms, 2),
        }

    tyre_results: List[Dict[str, object]] = []
    merged_alerts: List[str] = []

    for det in detections:
        tyre_crop = crop_tyre(image_bgr, det["bbox"])
        if tyre_crop.size == 0:
            continue

        cls = classify_tyre(tyre_crop)
        tread = analyze_tread(tyre_crop)
        puncture = detect_puncture_and_flat(tyre_crop)
        life = predict_lifespan(
            tread_depth_score=float(tread["tread_depth_score"]),
            crack_count=int(puncture["crack_count"]),
            condition=str(cls["condition"]),
        )

        result = {
            "bbox": det["bbox"],
            "condition": cls["condition"],
            "confidence": round(float(cls["confidence"]), 4),
            "probabilities": cls["probabilities"],
            "tread_score": float(tread["tread_depth_score"]),
            "wear_percentage": float(tread["wear_percentage"]),
            "remaining_km": int(life["remaining_km"]),
            "replacement_urgency": life["replacement_urgency"],
            "flat_tyre": bool(puncture["flat_tyre"]),
            "puncture_detected": bool(puncture["puncture_detected"]),
            "puncture_count": int(puncture["puncture_count"]),
            "crack_count": int(puncture["crack_count"]),
            "alerts": puncture["alerts"],
        }
        tyre_results.append(result)
        merged_alerts.extend(puncture["alerts"])

    if not tyre_results:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "condition": "Unknown",
            "tread_score": 0.0,
            "remaining_km": 0,
            "status": "analysis_failed",
            "alerts": ["Tyre crops could not be extracted"],
            "tyres": [],
            "detections": detections,
            "inference_ms": round(elapsed_ms, 2),
        }

    primary = _pick_highest_risk_tyre(tyre_results)
    merged_alerts.extend(_condition_alerts(primary))

    elapsed_ms = (time.perf_counter() - start) * 1000
    return {
        "condition": primary["condition"],
        "tread_score": round(float(primary["tread_score"]), 2),
        "remaining_km": int(primary["remaining_km"]),
        "status": _status_from_result(primary),
        "alerts": sorted(set(merged_alerts)),
        "tyres": tyre_results,
        "detections": detections,
        "inference_ms": round(elapsed_ms, 2),
    }


def analyse_tyre_with_visuals(image: np.ndarray) -> Tuple[Dict[str, object], np.ndarray]:
    """Run analysis and return JSON report plus annotated image."""
    report = analyse_tyre(image)
    image_bgr = _ensure_bgr(image)

    tyre_results = report.get("tyres", []) if isinstance(report, dict) else []
    annotations = [{"condition": tyre["condition"]} for tyre in tyre_results]
    annotated = draw_detections(image_bgr, report.get("detections", []), annotations)
    return report, annotated


def analyse_tyre_file(image_path: str | Path) -> Dict[str, object]:
    """Load image from disk and run complete tyre analysis."""
    img = cv2.imread(str(image_path))
    return analyse_tyre(img)


def save_report(report: Dict[str, object], output_path: str | Path) -> None:
    """Persist report JSON for integrations."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _pick_highest_risk_tyre(results: List[Dict[str, object]]) -> Dict[str, object]:
    return sorted(results, key=lambda r: _risk_key(r), reverse=True)[0]


def _risk_key(result: Dict[str, object]) -> Tuple[int, int, float, int]:
    condition_rank = CONDITION_RANK.get(str(result["condition"]), 1)
    urgency_rank = {"Immediate": 3, "High": 2, "Moderate": 1, "Low": 0}.get(str(result["replacement_urgency"]), 1)
    puncture_rank = 1 if result.get("puncture_detected") else 0
    tread_penalty = 100.0 - float(result.get("tread_score", 50.0))
    return condition_rank, urgency_rank, puncture_rank, int(tread_penalty)


def _condition_alerts(result: Dict[str, object]) -> List[str]:
    alerts: List[str] = []
    if result["condition"] == "Critical":
        alerts.append("Critical tyre condition")
    if float(result["tread_score"]) < 30:
        alerts.append("Severe tread wear")
    if int(result["remaining_km"]) < 5000:
        alerts.append("Replacement due very soon")
    return alerts


def _status_from_result(result: Dict[str, object]) -> str:
    if result.get("flat_tyre"):
        return "flat_tyre_alert"
    if result.get("puncture_detected"):
        return "puncture_alert"
    if result.get("condition") == "Critical":
        return "critical"
    if result.get("condition") == "Worn":
        return "monitor"
    return "healthy"


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image
