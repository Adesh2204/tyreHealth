"""Tests for end-to-end tyre analysis pipeline edge cases."""

from __future__ import annotations

import numpy as np

from src import pipeline


def test_no_tyre_detected(monkeypatch):
    image = np.zeros((256, 256, 3), dtype=np.uint8)

    monkeypatch.setattr(pipeline, "detect_tyres", lambda _img: [])

    result = pipeline.analyse_tyre(image)
    assert result["status"] == "no_tyre_detected"
    assert "No tyre detected in image" in result["alerts"]


def test_low_quality_image_still_returns_result(monkeypatch):
    image = np.zeros((224, 224, 3), dtype=np.uint8)

    monkeypatch.setattr(pipeline, "detect_tyres", lambda _img: [{"bbox": [10, 10, 200, 200], "confidence": 0.5, "label": "tyre"}])
    monkeypatch.setattr(pipeline, "classify_tyre", lambda _img: {"condition": "Worn", "confidence": 0.6, "probabilities": {"Good": 0.2, "Worn": 0.6, "Critical": 0.2}})
    monkeypatch.setattr(pipeline, "analyze_tread", lambda _img: {"tread_depth_score": 40.0, "wear_percentage": 60.0})
    monkeypatch.setattr(
        pipeline,
        "detect_puncture_and_flat",
        lambda _img: {"flat_tyre": False, "puncture_detected": False, "puncture_count": 0, "crack_count": 0, "alerts": []},
    )
    monkeypatch.setattr(
        pipeline,
        "predict_lifespan",
        lambda tread_depth_score, crack_count, condition: {
            "remaining_km": 17000,
            "replacement_urgency": "Moderate",
            "confidence": 0.8,
            "estimated_months": 11.3,
        },
    )

    result = pipeline.analyse_tyre(image)
    assert result["condition"] == "Worn"
    assert result["status"] in {"monitor", "healthy", "critical", "puncture_alert", "flat_tyre_alert"}
    assert result["remaining_km"] == 17000


def test_multiple_tyres_prioritizes_high_risk(monkeypatch):
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    boxes = [
        {"bbox": [20, 30, 170, 220], "confidence": 0.9, "label": "tyre"},
        {"bbox": [220, 40, 380, 230], "confidence": 0.88, "label": "tyre"},
    ]

    monkeypatch.setattr(pipeline, "detect_tyres", lambda _img: boxes)

    conditions = [
        {"condition": "Good", "confidence": 0.9, "probabilities": {"Good": 0.9, "Worn": 0.09, "Critical": 0.01}},
        {"condition": "Critical", "confidence": 0.92, "probabilities": {"Good": 0.02, "Worn": 0.06, "Critical": 0.92}},
    ]

    call_idx = {"i": 0}

    def fake_classify(_img):
        out = conditions[call_idx["i"]]
        call_idx["i"] += 1
        return out

    monkeypatch.setattr(pipeline, "classify_tyre", fake_classify)

    tread_values = [{"tread_depth_score": 82.0, "wear_percentage": 18.0}, {"tread_depth_score": 18.0, "wear_percentage": 82.0}]
    tread_idx = {"i": 0}

    def fake_tread(_img):
        out = tread_values[tread_idx["i"]]
        tread_idx["i"] += 1
        return out

    monkeypatch.setattr(pipeline, "analyze_tread", fake_tread)

    punct_values = [
        {"flat_tyre": False, "puncture_detected": False, "puncture_count": 0, "crack_count": 1, "alerts": []},
        {"flat_tyre": True, "puncture_detected": True, "puncture_count": 1, "crack_count": 6, "alerts": ["Flat tyre risk detected", "Puncture anomaly detected"]},
    ]
    punct_idx = {"i": 0}

    def fake_puncture(_img):
        out = punct_values[punct_idx["i"]]
        punct_idx["i"] += 1
        return out

    monkeypatch.setattr(pipeline, "detect_puncture_and_flat", fake_puncture)

    life_values = [
        {"remaining_km": 41000, "replacement_urgency": "Low", "confidence": 0.9, "estimated_months": 27.3},
        {"remaining_km": 1200, "replacement_urgency": "Immediate", "confidence": 0.9, "estimated_months": 0.8},
    ]
    life_idx = {"i": 0}

    def fake_life(*_args, **_kwargs):
        out = life_values[life_idx["i"]]
        life_idx["i"] += 1
        return out

    monkeypatch.setattr(pipeline, "predict_lifespan", fake_life)

    result = pipeline.analyse_tyre(image)
    assert result["condition"] == "Critical"
    assert result["status"] in {"flat_tyre_alert", "puncture_alert", "critical"}
    assert result["remaining_km"] == 1200
    assert len(result["tyres"]) == 2
