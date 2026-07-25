"""
app.py — Chronoscope

A Streamlit console for the Chronoscope face + age detector API
(see detector.py). Upload an image or use your camera; Chronoscope frames
every face it finds in a viewfinder readout and estimates an age bracket
for each one.
"""

from __future__ import annotations

import io
import json
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from detector import FaceAgeDetector, ModelDownloadError, ScanResult

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Chronoscope",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
    --void: #0D1117;
    --panel: #141A21;
    --panel-alt: #1B232C;
    --border: #2A3542;
    --amber: #F2A65A;
    --amber-dim: #8A5A2B;
    --text: #E8E6E1;
    --text-dim: #7C8697;
}

.stApp { background: var(--void); color: var(--text); }
[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] * { color: var(--text); }

/* Kill Streamlit's default chrome */
#MainMenu, footer, header { visibility: hidden; }

body, .stApp, p, div, span, li { font-family: 'Inter', sans-serif; }

/* ---- Eyebrow / mono labels ---- */
.cs-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--amber);
    margin: 0 0 4px 0;
}

/* ---- Hero wordmark ---- */
.cs-hero { padding: 8px 0 20px 0; border-bottom: 1px solid var(--border); margin-bottom: 28px; }
.cs-wordmark {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 44px;
    letter-spacing: -0.01em;
    color: var(--text);
    margin: 0;
    display: flex; align-items: center; gap: 14px;
}
.cs-wordmark .ring {
    display: inline-block; width: 30px; height: 30px;
    border: 3px solid var(--amber); border-radius: 50%;
    box-shadow: inset 0 0 0 6px var(--void);
}
.cs-tagline { font-family: 'IBM Plex Mono', monospace; color: var(--text-dim); font-size: 13px; margin-top: 6px; }

/* ---- Sidebar section labels ---- */
[data-testid="stSidebar"] label p {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: var(--text-dim) !important;
}

/* ---- Viewfinder frame around the image ---- */
.cs-viewfinder {
    position: relative;
    border: 1px solid var(--border);
    background: var(--panel);
    padding: 18px;
    border-radius: 2px;
}
.cs-viewfinder::before, .cs-viewfinder::after,
.cs-corner-tl, .cs-corner-br { content: ""; }
.cs-corner {
    position: absolute; width: 22px; height: 22px;
    border-color: var(--amber); border-style: solid; border-width: 0;
    animation: cs-pulse 2.6s ease-in-out infinite;
}
.cs-corner.tl { top: -1px; left: -1px; border-top-width: 3px; border-left-width: 3px; }
.cs-corner.tr { top: -1px; right: -1px; border-top-width: 3px; border-right-width: 3px; }
.cs-corner.bl { bottom: -1px; left: -1px; border-bottom-width: 3px; border-left-width: 3px; }
.cs-corner.br { bottom: -1px; right: -1px; border-bottom-width: 3px; border-right-width: 3px; }
@keyframes cs-pulse { 0%, 100% { opacity: 0.55; } 50% { opacity: 1; } }

/* ---- Panels ---- */
.cs-panel {
    background: var(--panel); border: 1px solid var(--border);
    padding: 16px 18px; border-radius: 2px; margin-bottom: 14px;
}

/* ---- Metric readout row ---- */
.cs-metrics { display: flex; gap: 10px; margin-bottom: 14px; }
.cs-metric {
    flex: 1; background: var(--panel); border: 1px solid var(--border);
    padding: 12px 14px; border-radius: 2px;
}
.cs-metric .value {
    font-family: 'Space Grotesk', sans-serif; font-size: 26px; font-weight: 700; color: var(--amber);
}
.cs-metric .label {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--text-dim); margin-top: 2px;
}

/* ---- Readout table ---- */
table.cs-table { width: 100%; border-collapse: collapse; font-family: 'IBM Plex Mono', monospace; font-size: 12.5px; }
table.cs-table th {
    text-align: left; color: var(--text-dim); font-weight: 500; letter-spacing: 0.08em;
    text-transform: uppercase; font-size: 10px; padding: 6px 10px; border-bottom: 1px solid var(--border);
}
table.cs-table td { padding: 8px 10px; border-bottom: 1px solid var(--border); color: var(--text); }
table.cs-table tr:last-child td { border-bottom: none; }
table.cs-table .age { color: var(--amber); font-weight: 600; }

/* ---- Empty state ---- */
.cs-empty {
    text-align: center; padding: 60px 20px; color: var(--text-dim);
    font-family: 'IBM Plex Mono', monospace; font-size: 13px; border: 1px dashed var(--border);
}

/* ---- Buttons ---- */
.stButton button, .stDownloadButton button {
    background: var(--amber) !important; color: #1a1206 !important; border: none !important;
    font-family: 'IBM Plex Mono', monospace !important; font-weight: 600 !important;
    letter-spacing: 0.08em !important; text-transform: uppercase !important; font-size: 12px !important;
    border-radius: 2px !important;
}
.stButton button:hover, .stDownloadButton button:hover { background: #ffb96e !important; }

hr { border-color: var(--border) !important; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Detector (loaded once, cached across reruns)
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_detector() -> FaceAgeDetector:
    detector = FaceAgeDetector()
    detector.load()
    return detector


def load_detector_with_feedback() -> FaceAgeDetector | None:
    try:
        with st.spinner("Calibrating optics — fetching model weights on first run…"):
            return get_detector()
    except ModelDownloadError as exc:
        st.error(
            "Chronoscope couldn't download its model weights. "
            f"Details: {exc}\n\nCheck your network connection, or run "
            "`bash download_models.sh` manually and restart the app."
        )
        return None


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_png_bytes(frame_bgr: np.ndarray) -> bytes:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def render_readout_table(result: ScanResult) -> str:
    rows = "".join(
        f"<tr><td>#{d.id}</td>"
        f"<td class='age'>{d.age_bracket}</td>"
        f"<td>{d.age_confidence * 100:.1f}%</td>"
        f"<td>{d.face_confidence * 100:.1f}%</td>"
        f"<td>({d.box[0]}, {d.box[1]}) → ({d.box[2]}, {d.box[3]})</td></tr>"
        for d in result.detections
    )
    return (
        "<table class='cs-table'><thead><tr>"
        "<th>ID</th><th>Age</th><th>Age conf.</th><th>Face conf.</th><th>Bounding box</th>"
        "</tr></thead><tbody>" + rows + "</tbody></table>"
    )


# --------------------------------------------------------------------------
# Sidebar — control panel
# --------------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.markdown("<div class='cs-eyebrow'>Chronoscope / control panel</div>", unsafe_allow_html=True)
    st.markdown("### Input source")
    source = st.radio("Input source", ["Upload image", "Camera"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### Sensitivity")
    confidence_threshold = st.slider(
        "Sensitivity", min_value=0.10, max_value=0.99, value=0.70, step=0.01,
        label_visibility="collapsed",
        help="Minimum face-detector confidence required to report a face.",
    )
    st.markdown("### Crop margin")
    padding = st.slider(
        "Crop margin", min_value=0, max_value=60, value=20, step=5,
        label_visibility="collapsed",
        help="Extra pixels included around each face before age estimation.",
    )

    st.markdown("---")
    st.markdown(
        f"<div class='cs-eyebrow'>Scan log</div>"
        f"<div style='font-family:IBM Plex Mono, monospace; font-size:12px; color:var(--text-dim);'>"
        f"{len(st.session_state.history)} scan(s) this session</div>",
        unsafe_allow_html=True,
    )
    if st.session_state.history and st.button("Clear scan log"):
        st.session_state.history = []
        st.rerun()


# --------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------

st.markdown(
    """
    <div class="cs-hero">
        <div class="cs-wordmark"><span class="ring"></span>CHRONOSCOPE</div>
        <div class="cs-tagline">FACE DETECTION &amp; AGE ESTIMATION CONSOLE — OPENCV DNN</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------

image = None
if source == "Upload image":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded is not None:
        image = Image.open(uploaded)
else:
    captured = st.camera_input("Capture a frame", label_visibility="collapsed")
    if captured is not None:
        image = Image.open(captured)

left, right = st.columns([1.3, 1], gap="large")

with left:
    st.markdown("<div class='cs-eyebrow'>Viewfinder</div>", unsafe_allow_html=True)
    st.markdown("<div class='cs-viewfinder'>"
                "<div class='cs-corner tl'></div><div class='cs-corner tr'></div>"
                "<div class='cs-corner bl'></div><div class='cs-corner br'></div>",
                unsafe_allow_html=True)

    if image is None:
        st.markdown(
            "<div class='cs-empty'>NO SIGNAL — upload an image or capture a frame to begin scanning</div>",
            unsafe_allow_html=True,
        )
        result = None
        annotated_bgr = None
    else:
        detector = load_detector_with_feedback()
        if detector is not None:
            frame_bgr = pil_to_bgr(image)
            with st.spinner("Scanning frame…"):
                result = detector.analyze(
                    frame_bgr, confidence_threshold=confidence_threshold, padding=padding
                )
                annotated_bgr = detector.draw(frame_bgr, result.detections)
            st.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

            record_key = (id(image), confidence_threshold, padding)
            if not st.session_state.history or st.session_state.history[-1]["key"] != record_key:
                st.session_state.history.append(
                    {
                        "key": record_key,
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "faces": result.face_count,
                    }
                )
        else:
            result = None
            annotated_bgr = None

    st.markdown("</div>", unsafe_allow_html=True)  # close cs-viewfinder

with right:
    st.markdown("<div class='cs-eyebrow'>Readout</div>", unsafe_allow_html=True)

    if image is None or result is None:
        st.markdown(
            "<div class='cs-panel' style='color:var(--text-dim); font-family:IBM Plex Mono, monospace; font-size:12.5px;'>"
            "Awaiting input…</div>",
            unsafe_allow_html=True,
        )
    else:
        avg_age_conf = (
            sum(d.age_confidence for d in result.detections) / result.face_count
            if result.face_count else 0.0
        )
        st.markdown(
            f"""
            <div class="cs-metrics">
                <div class="cs-metric"><div class="value">{result.face_count}</div><div class="label">Faces found</div></div>
                <div class="cs-metric"><div class="value">{result.image_size[0]}×{result.image_size[1]}</div><div class="label">Frame size</div></div>
                <div class="cs-metric"><div class="value">{avg_age_conf * 100:.0f}%</div><div class="label">Avg. confidence</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if result.face_count == 0:
            st.markdown(
                "<div class='cs-empty'>NO FACES DETECTED — try lowering sensitivity in the control panel</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"<div class='cs-panel'>{render_readout_table(result)}</div>", unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.download_button(
                    "Download annotated image",
                    data=bgr_to_png_bytes(annotated_bgr),
                    file_name="chronoscope_scan.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with col_b:
                st.download_button(
                    "Download JSON",
                    data=json.dumps(result.to_records(), indent=2),
                    file_name="chronoscope_scan.json",
                    mime="application/json",
                    use_container_width=True,
                )

    if st.session_state.history:
        with st.expander(f"Scan log ({len(st.session_state.history)})"):
            log_rows = "".join(
                f"<tr><td>{h['time']}</td><td>{h['faces']} face(s)</td></tr>"
                for h in reversed(st.session_state.history)
            )
            st.markdown(
                f"<table class='cs-table'><thead><tr><th>Time</th><th>Result</th></tr></thead>"
                f"<tbody>{log_rows}</tbody></table>",
                unsafe_allow_html=True,
            )
