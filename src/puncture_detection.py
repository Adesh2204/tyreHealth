"""Puncture and flat-tyre detection using OpenCV contours and circles."""

from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np


def detect_puncture_and_flat(image: np.ndarray) -> Dict[str, object]:
    """Detect puncture anomalies and flat-tyre deformation.

    Rules implemented:
    - Small dark anomaly contour -> puncture
    - Distorted tyre circle -> flat tyre
    """
    if image is None or image.size == 0:
        return {
            "puncture_detected": False,
            "puncture_count": 0,
            "flat_tyre": False,
            "crack_count": 0,
            "anomaly_boxes": [],
            "alerts": ["Invalid tyre crop for puncture analysis"],
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Dark anomaly segmentation
    _, dark_mask = cv2.threshold(blur, 52, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    image_area = h * w
    puncture_count = 0
    anomaly_boxes: List[List[int]] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < max(6.0, image_area * 0.00008) or area > image_area * 0.012:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.15:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        roi = gray[y : y + bh, x : x + bw]
        if roi.size == 0:
            continue

        if float(np.mean(roi)) > 95.0:
            continue

        puncture_count += 1
        anomaly_boxes.append([x, y, x + bw, y + bh])

    puncture_detected = puncture_count > 0

    # Tyre shape check with HoughCircles
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, min(h, w) // 4),
        param1=110,
        param2=32,
        minRadius=max(15, min(h, w) // 6),
        maxRadius=max(20, min(h, w) // 2),
    )

    circle_found = circles is not None and len(circles[0]) > 0
    flat_tyre = not circle_found

    # Additional distortion check from dominant contour circularity
    edge = cv2.Canny(blur, 50, 150)
    edge_contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if edge_contours:
        largest = max(edge_contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))
        perimeter = cv2.arcLength(largest, True)
        if perimeter > 0 and area > image_area * 0.03:
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.55:
                flat_tyre = True

    # Crack estimation from edge fragments
    crack_count = _estimate_crack_count(edge)

    alerts: List[str] = []
    if puncture_detected:
        alerts.append("Puncture anomaly detected")
    if flat_tyre:
        alerts.append("Flat tyre risk detected")

    return {
        "puncture_detected": puncture_detected,
        "puncture_count": puncture_count,
        "flat_tyre": flat_tyre,
        "crack_count": crack_count,
        "anomaly_boxes": anomaly_boxes,
        "alerts": alerts,
    }


def _estimate_crack_count(edge_image: np.ndarray) -> int:
    """Estimate crack-like fragments via connected components in edge map."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((edge_image > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return 0

    crack_count = 0
    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        if 10 <= area <= 200:
            crack_count += 1
    return int(crack_count)


def draw_puncture_annotations(image: np.ndarray, anomaly_boxes: List[List[int]], flat_tyre: bool) -> np.ndarray:
    """Draw puncture boxes and flat-tyre status over image."""
    canvas = image.copy()

    for box in anomaly_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 60, 255), 2)
        cv2.putText(canvas, "Puncture", (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 60, 255), 2)

    if flat_tyre:
        cv2.putText(canvas, "Flat Tyre Alert", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 230), 2)

    return canvas
