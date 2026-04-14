"""Immersive Streamlit dashboard for AI-Based Tyre Health Monitoring System."""

from __future__ import annotations

import io
from typing import Dict, List

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src.classification import get_training_metrics
from src.classification import ensure_classifier_model
from src.detection import ensure_detector_model
from src.lifespan import ensure_lifespan_model
from src.pipeline import analyse_tyre_with_visuals

st.set_page_config(page_title="Tyre Health AI", page_icon="AI", layout="wide")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

:root {
  --bg0: #060b12;
  --bg1: #0b1726;
  --card: rgba(13, 24, 38, 0.83);
  --line: rgba(95, 150, 203, 0.30);
  --ink: #edf3ff;
  --muted: #a5b8d4;
  --good: #2ddc87;
  --warn: #ffbf47;
  --bad: #ff5a57;
  --cyan: #59c9ff;
}

html, body, [class*="stApp"] {
  font-family: 'Plus Jakarta Sans', sans-serif;
  color: var(--ink);
  background:
    radial-gradient(circle at 12% 9%, rgba(53, 128, 191, 0.30), transparent 35%),
    radial-gradient(circle at 86% 7%, rgba(27, 182, 145, 0.20), transparent 33%),
    linear-gradient(165deg, var(--bg0), var(--bg1));
}

h1, h2, h3 {
  font-family: 'Sora', sans-serif;
  letter-spacing: 0.2px;
}

section.main > div {
  padding-top: 1rem;
  max-width: 1280px;
}

.hero-shell {
  border-radius: 22px;
  border: 1px solid var(--line);
  background: linear-gradient(145deg, rgba(11, 24, 40, 0.92), rgba(13, 34, 53, 0.70));
  box-shadow: 0 28px 55px rgba(1, 8, 16, 0.45);
  padding: 24px;
  overflow: hidden;
  position: relative;
}

.hero-shell::after {
  content: "";
  position: absolute;
  inset: -140px;
  background: conic-gradient(from 90deg, rgba(88, 193, 255, 0.16), rgba(33, 212, 157, 0.06), rgba(88, 193, 255, 0.16));
  animation: aura 10s linear infinite;
  pointer-events: none;
}

@keyframes aura {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.hero-grid {
  position: relative;
  z-index: 2;
  display: grid;
  grid-template-columns: 1.25fr 1fr;
  gap: 18px;
  align-items: center;
}

.eyebrow {
  display: inline-block;
  border-radius: 999px;
  border: 1px solid rgba(113, 181, 238, 0.55);
  background: rgba(56, 128, 185, 0.14);
  color: #9ed8ff;
  font-size: 0.74rem;
  text-transform: uppercase;
  letter-spacing: 1px;
  font-weight: 700;
  padding: 6px 12px;
}

.hero-title {
  margin: 12px 0 8px 0;
  font-size: clamp(1.7rem, 3vw, 2.7rem);
  line-height: 1.12;
}

.hero-desc {
  margin: 0;
  color: var(--muted);
  max-width: 65ch;
}

.chip-row {
  margin-top: 16px;
  display: flex;
  flex-wrap: wrap;
  gap: 9px;
}

.chip {
  border-radius: 999px;
  border: 1px solid rgba(133, 181, 231, 0.40);
  background: rgba(20, 39, 58, 0.72);
  color: #d5e7ff;
  font-size: 0.8rem;
  font-weight: 600;
  padding: 7px 12px;
}

.tyre-stage {
  position: relative;
  min-height: 205px;
  border-radius: 18px;
  border: 1px solid rgba(125, 175, 221, 0.33);
  background: linear-gradient(160deg, rgba(14, 31, 48, 0.80), rgba(8, 18, 28, 0.80));
  overflow: hidden;
}

.road {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  height: 54px;
  background: linear-gradient(to top, rgba(8, 11, 15, 0.95), rgba(25, 33, 44, 0.9));
}

.road::before {
  content: "";
  position: absolute;
  left: -30%;
  right: -30%;
  top: 22px;
  height: 4px;
  background: repeating-linear-gradient(90deg, rgba(254, 232, 157, 0.95) 0 30px, transparent 30px 58px);
  animation: roadMove 1.1s linear infinite;
}

@keyframes roadMove {
  from { transform: translateX(0); }
  to { transform: translateX(120px); }
}

.car-body {
  position: absolute;
  left: 20%;
  right: 17%;
  bottom: 72px;
  height: 54px;
  border-radius: 16px 26px 12px 12px;
  background: linear-gradient(120deg, #7f9cc0, #3f5879);
  box-shadow: 0 10px 18px rgba(0, 0, 0, 0.3);
}

.car-cabin {
  position: absolute;
  left: 34%;
  right: 36%;
  bottom: 118px;
  height: 34px;
  border-radius: 12px 12px 6px 6px;
  background: linear-gradient(130deg, #cde8ff, #7aa0c8);
}

.wheel {
  position: absolute;
  width: 84px;
  height: 84px;
  bottom: 24px;
  border-radius: 50%;
  border: 8px solid #090f16;
  background:
    radial-gradient(circle at center, #1c2a3a 0 16px, transparent 16px),
    repeating-conic-gradient(from 0deg, #8ca5c3 0deg 14deg, #2d3d53 14deg 28deg);
  box-shadow: inset 0 0 0 3px rgba(203, 225, 255, 0.24), 0 8px 18px rgba(0, 0, 0, 0.38);
  animation: spin 0.82s linear infinite;
}

.wheel::before {
  content: "";
  position: absolute;
  inset: 9px;
  border-radius: 50%;
  border: 2px dashed rgba(201, 223, 247, 0.45);
  animation: spin 2.1s linear infinite reverse;
}

.wheel-left { left: 26%; }
.wheel-right { right: 23%; }

.dust {
  position: absolute;
  border-radius: 999px;
  background: rgba(96, 168, 226, 0.36);
  filter: blur(0.5px);
  animation: drift 2.4s ease-in-out infinite;
}

.dust-a {
  width: 32px;
  height: 8px;
  bottom: 55px;
  right: 17%;
}

.dust-b {
  width: 22px;
  height: 6px;
  bottom: 45px;
  right: 14%;
  animation-delay: 0.45s;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

@keyframes drift {
  0% { transform: translateX(0); opacity: 0.18; }
  50% { transform: translateX(16px); opacity: 0.7; }
  100% { transform: translateX(32px); opacity: 0.1; }
}

.panel {
  border-radius: 18px;
  border: 1px solid var(--line);
  background: var(--card);
  backdrop-filter: blur(8px);
  box-shadow: 0 10px 30px rgba(2, 8, 18, 0.34);
  padding: 16px;
}

.panel-title {
  margin: 0 0 10px 0;
  font-size: 1.72rem;
}

.panel-subtitle {
  color: #d3e3f8;
  margin-bottom: 10px;
  font-weight: 600;
}

.status-strip {
  margin-top: 16px;
  border-radius: 14px;
  border: 1px solid color-mix(in srgb, var(--status-color) 60%, #79aeda);
  background: linear-gradient(120deg, color-mix(in srgb, var(--status-color) 18%, #0d1c2e), rgba(8, 19, 30, 0.74));
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 14px;
}

.status-wheel {
  width: 34px;
  height: 34px;
  border-radius: 50%;
  border: 3px solid #09111a;
  background: repeating-conic-gradient(#d4ebff 0deg 22deg, #446280 22deg 44deg);
  box-shadow: inset 0 0 0 2px rgba(7, 17, 27, 0.85);
  animation: spin var(--spin-speed) linear infinite;
}

.status-main {
  font-weight: 800;
  font-size: 1.03rem;
  color: #f1f7ff;
}

.status-sub {
  color: #d8e9ff;
  opacity: 0.84;
  font-weight: 600;
}

.status-latency {
  margin-left: auto;
  color: #d2e4fb;
  opacity: 0.85;
  font-size: 0.86rem;
}

.health-wrap {
  margin-top: 10px;
  width: 100%;
  height: 16px;
  border-radius: 999px;
  border: 1px solid var(--line);
  overflow: hidden;
  background: rgba(23, 38, 55, 0.9);
}

.health-bar {
  height: 100%;
  border-radius: 999px;
  animation: fill 1.25s ease-out;
  transform-origin: left;
}

@keyframes fill {
  from { transform: scaleX(0.07); }
  to { transform: scaleX(1); }
}

.kpi-card {
  border-radius: 14px;
  border: 1px solid rgba(109, 167, 221, 0.35);
  background: rgba(10, 22, 34, 0.76);
  padding: 12px;
  min-height: 90px;
}

.kpi-label {
  color: #9eb8d6;
  font-size: 0.82rem;
  text-transform: uppercase;
  letter-spacing: 0.7px;
}

.kpi-value {
  font-family: 'Sora', sans-serif;
  font-size: 1.42rem;
  margin: 2px 0;
}

.kpi-note {
  color: #8da8c5;
  font-size: 0.8rem;
}

.alert-item {
  border-left: 3px solid #ff9158;
  border-radius: 8px;
  background: rgba(255, 128, 74, 0.14);
  padding: 8px 10px;
  margin-bottom: 8px;
  color: #ffe6d9;
}

.good-item {
  border-left: 3px solid #33cf83;
  border-radius: 8px;
  background: rgba(48, 180, 118, 0.14);
  padding: 8px 10px;
  color: #dcffee;
}

.ghost-card {
  margin-top: 16px;
  border-radius: 14px;
  border: 1px dashed rgba(109, 165, 223, 0.5);
  background: rgba(10, 21, 33, 0.66);
  padding: 16px;
  color: #bdd6f3;
}

div.stButton > button {
  background: linear-gradient(120deg, #37c2ff, #2f7bff);
  border: 0;
  color: #07101a;
  font-weight: 800;
  border-radius: 12px;
  min-height: 44px;
}

div.stButton > button:hover {
  filter: brightness(1.08);
  transform: translateY(-1px);
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-testid="stFileUploaderDropzone"] {
  background: rgba(15, 30, 46, 0.65);
  border-color: rgba(116, 172, 225, 0.35);
}

@media (max-width: 920px) {
  .hero-grid {
    grid-template-columns: 1fr;
  }

  .tyre-stage {
    min-height: 180px;
  }

  .car-body {
    left: 15%;
    right: 12%;
  }
}
</style>
"""


def _render_hero() -> None:
    st.markdown(
        """
<div class="hero-shell">
  <div class="hero-grid">
    <div>
      <span class="eyebrow">Realtime Inspection Engine</span>
      <h1 class="hero-title">Tyre Intelligence Command Deck</h1>
      <p class="hero-desc">Detection, condition grading, puncture and flat anomaly checks, tread scoring, and lifespan estimation in one visual cockpit.</p>
      <div class="chip-row">
        <span class="chip">YOLOv8 Tyre Detection</span>
        <span class="chip">CNN Condition Grading</span>
        <span class="chip">OpenCV Groove Analytics</span>
        <span class="chip">Lifespan Forecasting</span>
      </div>
    </div>
    <div class="tyre-stage">
      <div class="car-cabin"></div>
      <div class="car-body"></div>
      <div class="wheel wheel-left"></div>
      <div class="wheel wheel-right"></div>
      <div class="dust dust-a"></div>
      <div class="dust dust-b"></div>
      <div class="road"></div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _load_model_metrics() -> Dict[str, float]:
    return get_training_metrics()


@st.cache_resource(show_spinner=False)
def _warmup_runtime() -> Dict[str, bool]:
  """Load heavy models once per app session to reduce click latency."""
  detector = ensure_detector_model(auto_train=False, allow_pretrained=False)
  classifier = ensure_classifier_model(auto_train=False)
  lifespan = ensure_lifespan_model(auto_train=False)
  return {
    "detector_loaded": detector is not None,
    "classifier_loaded": classifier is not None,
    "lifespan_loaded": lifespan is not None,
  }


def _decode_image(uploaded_file) -> np.ndarray:
    image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _condition_color(condition: str) -> str:
    return {"Good": "#2ddc87", "Worn": "#ffbf47", "Critical": "#ff5a57"}.get(condition, "#59c9ff")


def _status_text(status: str) -> str:
    return {
        "healthy": "Healthy",
        "monitor": "Monitor Closely",
        "critical": "Critical",
        "puncture_alert": "Puncture Alert",
        "flat_tyre_alert": "Flat Tyre Alert",
        "no_tyre_detected": "No Tyre Detected",
    }.get(status, status.replace("_", " ").title())


def _render_health_bar(score: float, color: str) -> None:
    width = int(np.clip(score, 0, 100))
    st.markdown(
        f"""
<div class="health-wrap">
  <div class="health-bar" style="width:{width}%; background:{color};"></div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_status_strip(condition: str, status: str, color: str, inference_ms: float) -> None:
    spin_speed = "0.65s" if condition == "Critical" else "0.95s" if condition == "Worn" else "1.25s"
    st.markdown(
        f"""
<div class="status-strip" style="--status-color:{color}; --spin-speed:{spin_speed};">
  <div class="status-wheel"></div>
  <div class="status-main">{condition}</div>
  <div class="status-sub">{_status_text(status)}</div>
  <div class="status-latency">Inference: {inference_ms:.1f} ms</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _kpi_card(label: str, value: str, note: str) -> str:
    return (
        "<div class=\"kpi-card\">"
        f"<div class=\"kpi-label\">{label}</div>"
        f"<div class=\"kpi-value\">{value}</div>"
        f"<div class=\"kpi-note\">{note}</div>"
        "</div>"
    )


def _chart_layout(height: int = 320) -> Dict[str, object]:
    return {
        "height": height,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#e7f1ff"},
        "margin": {"l": 16, "r": 16, "t": 44, "b": 18},
    }


def _project_tread_points(tread_score: float) -> List[float]:
    start = float(np.clip(tread_score + 14, 0, 100))
    points: List[float] = []
    for idx in range(8):
        value = start - (idx * 4.8) + (1.6 if idx % 2 == 0 else -1.8)
        points.append(round(float(np.clip(value, 0, 100)), 2))
    return points


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
_render_hero()

model_metrics = _load_model_metrics()
runtime_state = _warmup_runtime()

image_bgr = None
run = False

left, right = st.columns([1.12, 1.0], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<h3 class="panel-title">Inspection Input</h3>', unsafe_allow_html=True)
    source = st.radio("Image Source", ["Upload Image", "Use Webcam"], horizontal=True)

    if source == "Upload Image":
        uploaded = st.file_uploader("Upload tyre image", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if uploaded is not None:
            image_bgr = _decode_image(uploaded)
    else:
        camera_capture = st.camera_input("Capture tyre image")
        if camera_capture is not None:
            image_bgr = _decode_image(camera_capture)

    run = st.button("Run Full Analysis", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<h3 class="panel-title">Model Metrics</h3>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{model_metrics.get('accuracy', 0.0) * 100:.1f}%")
    m2.metric("Precision", f"{model_metrics.get('precision', 0.0) * 100:.1f}%")
    m3.metric("Recall", f"{model_metrics.get('recall', 0.0) * 100:.1f}%")
    status_bits = []
    status_bits.append("Detector: ready" if runtime_state.get("detector_loaded") else "Detector: fast fallback")
    status_bits.append("Classifier: ready" if runtime_state.get("classifier_loaded") else "Classifier: heuristic")
    status_bits.append("Lifespan: ready" if runtime_state.get("lifespan_loaded") else "Lifespan: heuristic")
    st.caption(" | ".join(status_bits))
    st.caption("Metrics are loaded from latest classifier training artifacts.")
    st.markdown("</div>", unsafe_allow_html=True)

if run and image_bgr is None:
    st.warning("Please upload or capture a tyre image before running analysis.")

if run and image_bgr is not None:
    with st.spinner("Running detection, classification, tread analytics, and lifespan forecast..."):
        report, annotated = analyse_tyre_with_visuals(image_bgr)
    st.session_state["report"] = report
    st.session_state["annotated"] = annotated
    st.session_state["input"] = image_bgr

if "report" not in st.session_state:
    st.markdown(
        """
<div class="ghost-card">
  Upload an image and click Run Full Analysis to generate a full tyre-health report with visual overlays, condition confidence, forecast charts, and alerts.
</div>
""",
        unsafe_allow_html=True,
    )
else:
    report = st.session_state["report"]
    annotated = st.session_state["annotated"]
    input_img = st.session_state["input"]

    condition = report.get("condition", "Unknown")
    status = report.get("status", "unknown")
    tread_score = float(report.get("tread_score", 0.0))
    remaining_km = int(report.get("remaining_km", 0))
    inference_ms = float(report.get("inference_ms", 0.0))
    alerts = report.get("alerts", [])
    tyres = report.get("tyres", [])

    color = _condition_color(condition)
    _render_status_strip(condition, status, color, inference_ms)

    st.markdown("### Visual Inspection")
    img_left, img_right = st.columns(2, gap="large")

    with img_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-subtitle">Input Frame</div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with img_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-subtitle">Detected Tyres and Condition Overlay</div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Tyre Health Overview")
    _render_health_bar(tread_score, color)

    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1:
        st.markdown(_kpi_card("Condition", condition, _status_text(status)), unsafe_allow_html=True)
    with k2:
        st.markdown(_kpi_card("Tread Score", f"{tread_score:.1f}/100", "Higher is healthier"), unsafe_allow_html=True)
    with k3:
        st.markdown(_kpi_card("Remaining Life", f"{remaining_km:,} km", "Estimated by regression"), unsafe_allow_html=True)
    with k4:
        st.markdown(_kpi_card("Inference", f"{inference_ms:.1f} ms", "Total pipeline latency"), unsafe_allow_html=True)

    ch1, ch2 = st.columns(2, gap="large")
    with ch1:
        fig_health = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=tread_score,
                title={"text": "Tread Health Meter"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 35], "color": "#5b2b2a"},
                        {"range": [35, 65], "color": "#5a4822"},
                        {"range": [65, 100], "color": "#1f5639"},
                    ],
                },
            )
        )
        fig_health.update_layout(**_chart_layout())
        st.plotly_chart(fig_health, use_container_width=True)

    with ch2:
        fig_life = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=remaining_km,
                number={"suffix": " km"},
                title={"text": "Lifespan Meter"},
                gauge={
                    "axis": {"range": [0, 80000]},
                    "bar": {"color": "#68c9ff"},
                    "steps": [
                        {"range": [0, 10000], "color": "#4e2424"},
                        {"range": [10000, 25000], "color": "#564924"},
                        {"range": [25000, 80000], "color": "#204862"},
                    ],
                },
            )
        )
        fig_life.update_layout(**_chart_layout())
        st.plotly_chart(fig_life, use_container_width=True)

    ch3, ch4 = st.columns(2, gap="large")
    with ch3:
        trend_points = _project_tread_points(tread_score)
        fig_trend = go.Figure()
        fig_trend.add_trace(
            go.Scatter(
                x=[f"W{i}" for i in range(1, 9)],
                y=trend_points,
                mode="lines+markers",
                line={"width": 3, "color": color},
                marker={"size": 8, "color": "#d5ebff"},
                fill="tozeroy",
                fillcolor="rgba(88, 181, 255, 0.16)",
                name="Projected tread",
            )
        )
        fig_trend.update_layout(
            title="Projected Tread Trajectory",
            xaxis_title="Weeks",
            yaxis_title="Health Index",
            yaxis={"range": [0, 100]},
            **_chart_layout(height=300),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with ch4:
        probabilities = tyres[0].get("probabilities", {}) if tyres else {}
        if probabilities:
            fig_prob = go.Figure(
                data=[
                    go.Pie(
                        labels=list(probabilities.keys()),
                        values=list(probabilities.values()),
                        hole=0.52,
                        marker={"colors": ["#2ddc87", "#ffbf47", "#ff5a57"]},
                        textinfo="label+percent",
                    )
                ]
            )
            fig_prob.update_layout(title="Condition Confidence Split", **_chart_layout(height=300))
            st.plotly_chart(fig_prob, use_container_width=True)
        else:
            st.markdown('<div class="panel">Condition confidence data unavailable for this frame.</div>', unsafe_allow_html=True)

    st.markdown("### Alerts")
    if alerts:
        for alert in alerts:
            st.markdown(f'<div class="alert-item">{alert}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="good-item">No critical alerts triggered for this sample.</div>', unsafe_allow_html=True)

    if len(tyres) > 1:
        st.markdown("### Multi-Tyre Breakdown")
        rows = [
            {
                "Tyre": idx + 1,
                "Condition": tyre["condition"],
                "Tread Score": tyre["tread_score"],
                "Remaining KM": tyre["remaining_km"],
                "Puncture": tyre["puncture_detected"],
                "Flat": tyre["flat_tyre"],
            }
            for idx, tyre in enumerate(tyres)
        ]
        st.dataframe(rows, use_container_width=True)
