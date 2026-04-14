"""OpenCV-based tread analysis for tyre wear estimation."""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def analyze_tread(image: np.ndarray) -> dict:
    """Estimate tread depth score (0-100) from tyre crop using Canny grooves."""
    if image is None or image.size == 0:
        return _default_result("Empty tyre crop received.")

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(gray, 45, 140)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

        h, w = edges.shape
        edge_density = float(np.count_nonzero(edges)) / float(h * w)
        groove_count = _count_grooves(edges)

        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast = float(np.std(gray))

        groove_score = min(100.0, groove_count * 7.5)
        edge_score = min(100.0, edge_density * 420.0)
        texture_score = min(100.0, lap_var / 16.0)
        contrast_score = min(100.0, contrast * 2.4)

        tread_depth_score = (
            0.32 * edge_score + 0.30 * groove_score + 0.23 * texture_score + 0.15 * contrast_score
        )
        tread_depth_score = float(np.clip(tread_depth_score, 0.0, 100.0))
        wear_percentage = float(np.clip(100.0 - tread_depth_score, 0.0, 100.0))

        if tread_depth_score >= 67:
            detail = "Healthy groove visibility"
        elif tread_depth_score >= 38:
            detail = "Moderate tread wear"
        else:
            detail = "Severe wear: inspect immediately"

        return {
            "tread_depth_score": round(tread_depth_score, 2),
            "wear_percentage": round(wear_percentage, 2),
            "groove_count": int(groove_count),
            "edge_density": round(edge_density, 5),
            "texture_variance": round(lap_var, 2),
            "condition_detail": detail,
        }
    except Exception as exc:
        logger.error("Tread analysis failed: %s", exc)
        return _default_result("Analysis failed.")


def _count_grooves(edge_image: np.ndarray) -> int:
    """Approximate groove count by transition peaks in horizontal strips."""
    h, _ = edge_image.shape
    strip_rows = np.linspace(int(h * 0.2), int(h * 0.8), num=8, dtype=int)
    counts = []

    for row in strip_rows:
        line = edge_image[row, :]
        transitions = np.diff(line.astype(np.int16))
        peaks = int(np.count_nonzero(np.abs(transitions) > 120))
        counts.append(peaks // 2)

    return int(np.median(counts)) if counts else 0


def get_tread_visualization(image: np.ndarray) -> np.ndarray:
    """Overlay detected edges on tyre crop for UI visualization."""
    if image is None or image.size == 0:
        return np.zeros((224, 224, 3), dtype=np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 45, 140)

    overlay = image.copy()
    overlay[edges > 0] = (40, 220, 120)
    return cv2.addWeighted(image, 0.7, overlay, 0.6, 0)


def _default_result(reason: str) -> dict:
    return {
        "tread_depth_score": 50.0,
        "wear_percentage": 50.0,
        "groove_count": 0,
        "edge_density": 0.0,
        "texture_variance": 0.0,
        "condition_detail": reason,
    }
