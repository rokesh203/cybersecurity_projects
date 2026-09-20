"""
HyperMD-Enhanced - Module 7: Real-Time Monitoring Dashboard
============================================================
Streamlit dashboard that:
  1. Collects live performance data (CPU, RAM, disk, processes)
  2. On demand: runs inference -> explainability -> response decision
  3. Shows live charts, latest prediction, SHAP bars, Grad-CAM overlay
  4. Lets the user trigger a session scan or respond to an alert

Run
---
    .venv\\Scripts\\streamlit.exe run dashboard\\dashboard.py

Layout
------
  Sidebar : controls (scan interval, thresholds, response level)
  Col 1   : Live metrics gauges + rolling time-series chart
  Col 2   : Latest prediction card + confidence gauge
  Col 3   : SHAP feature importance bar chart
  Bottom  : Grad-CAM overlay + response action panel + response log table

Safety
------
  Destructive actions (isolate, dump) show a confirmation dialog
  inside Streamlit before executing anything.
"""

import os
import sys
import time
import csv
import threading
import queue
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------
# PAGE CONFIG  (must be first Streamlit call)
# ---------------------------------------------------------
st.set_page_config(
    page_title="HyperMD Enhanced — Malware Monitor",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# CONSTANTS & PATHS
# ---------------------------------------------------------
PERFORMANCE_FILE  = os.path.join(PROJECT_ROOT, "performance_data.csv")
IMAGE_LABELS_CSV  = os.path.join(PROJECT_ROOT, "memory_images", "image_labels.csv")
EVENT_REGISTRY    = os.path.join(PROJECT_ROOT, "event_registry.csv")
RESPONSE_LOG      = os.path.join(PROJECT_ROOT, "logs", "response_log.csv")
ALERT_FILE        = os.path.join(PROJECT_ROOT, "logs", "alerts.txt")
MEMORY_IMAGES_DIR = os.path.join(PROJECT_ROOT, "memory_images")

FEATURE_LABELS = {
    "cpu_percent":      "CPU %",
    "memory_percent":   "RAM %",
    "process_count":    "Processes",
    "disk_read_bytes":  "Disk Read",
    "disk_write_bytes": "Disk Write",
}

LABEL_COLORS = {0: "#2ecc71", 1: "#e74c3c"}
LABEL_NAMES  = {0: "Normal", 1: "Suspicious"}


# ---------------------------------------------------------
# SESSION STATE INITIALISATION
# ---------------------------------------------------------
def _init_state() -> None:
    defaults = {
        "models_loaded":    False,
        "ctx":              None,
        "last_prediction":  None,
        "last_explanation": None,
        "last_event_id":    None,
        "scan_running":     False,
        "perf_history":     pd.DataFrame(),
        "response_log":     pd.DataFrame(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ---------------------------------------------------------
# LAZY MODEL LOADING
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI models (first run only)...")
def get_model_context():
    from models.inference import load_models
    return load_models()


# ---------------------------------------------------------
# DATA LOADERS
# ---------------------------------------------------------
@st.cache_data(ttl=5)   # refresh every 5 seconds
def load_perf_tail(n: int = 120) -> pd.DataFrame:
    if not os.path.isfile(PERFORMANCE_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(PERFORMANCE_FILE)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).tail(n)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=10)
def load_image_labels() -> pd.DataFrame:
    if not os.path.isfile(IMAGE_LABELS_CSV):
        return pd.DataFrame()
    try:
        return pd.read_csv(IMAGE_LABELS_CSV)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=10)
def load_response_log() -> pd.DataFrame:
    if not os.path.isfile(RESPONSE_LOG):
        return pd.DataFrame()
    try:
        return pd.read_csv(RESPONSE_LOG)
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------
# RUN SCAN (inference + explainability + response)
# ---------------------------------------------------------
def run_scan(event_id: str, image_path: str, response_level_override: str = "") -> dict:
    """
    Run the full pipeline for one session image + current perf data.
    Returns a dict with prediction, explanation, and response results.
    """
    from models.inference import predict, preprocess_perf_sequence
    from explainability.explain import explain_prediction
    from response.response_engine import respond

    ctx     = get_model_context()
    perf_df = load_perf_tail(120)
    if perf_df.empty or len(perf_df) < 10:
        st.error("Not enough performance data. Collect at least 10 rows first.")
        return {}

    with st.spinner("Running inference..."):
        pred_result = predict(perf_df, image_path, ctx)

    perf_seq = preprocess_perf_sequence(perf_df)[0]

    with st.spinner("Generating explanation (SHAP + Grad-CAM)..."):
        explanation = explain_prediction(
            perf_sequence  = perf_seq,
            image_path     = image_path,
            predict_result = pred_result,
            ctx            = ctx,
            save_gradcam   = True,
            event_id       = event_id,
        )

    with st.spinner("Evaluating response..."):
        response = respond(
            predict_result = pred_result,
            explanation    = explanation,
            event_id       = event_id,
            interactive    = False,
        )

    return {
        "prediction":  pred_result,
        "explanation": explanation,
        "response":    response,
    }


# ---------------------------------------------------------
# UI COMPONENTS
# ---------------------------------------------------------
def _render_sidebar() -> dict:
    st.sidebar.title("HyperMD Enhanced")
    st.sidebar.caption("Multi-Modal Malware Detection")
    st.sidebar.divider()

    st.sidebar.header("Scan Controls")
    auto_refresh = st.sidebar.toggle("Auto-refresh metrics", value=True)
    refresh_sec  = st.sidebar.slider("Refresh interval (s)", 5, 60, 10)

    st.sidebar.divider()
    st.sidebar.header("Response Thresholds")
    st.sidebar.caption("Current thresholds (fixed, see response_engine.py):")
    st.sidebar.markdown("""
| Level | Confidence |
|-------|-----------|
| MONITOR | any |
| ALERT | >= 60% |
| ISOLATE | >= 80% |
| DUMP | >= 95% |
""")

    st.sidebar.divider()
    st.sidebar.header("Quick Links")
    st.sidebar.markdown(f"- [Response Log]({RESPONSE_LOG})")
    st.sidebar.markdown(f"- [Alerts File]({ALERT_FILE})")
    st.sidebar.markdown(f"- [Performance Data]({PERFORMANCE_FILE})")

    return {
        "auto_refresh": auto_refresh,
        "refresh_sec":  refresh_sec,
    }


def _render_live_metrics(perf_df: pd.DataFrame) -> None:
    st.subheader("Live System Metrics")
    if perf_df.empty:
        st.info("No performance data yet. Run a collector first.")
        return

    latest = perf_df.iloc[-1]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CPU %",      f"{latest.get('cpu_percent', 0):.1f}%")
    c2.metric("RAM %",      f"{latest.get('memory_percent', 0):.1f}%")
    c3.metric("Processes",  f"{int(latest.get('process_count', 0))}")
    c4.metric("Disk Read",  f"{latest.get('disk_read_bytes', 0)/1e6:.1f} MB/s")
    c5.metric("Disk Write", f"{latest.get('disk_write_bytes', 0)/1e6:.1f} MB/s")

    # Rolling chart
    chart_cols = [c for c in ["cpu_percent", "memory_percent"] if c in perf_df.columns]
    if chart_cols and "timestamp" in perf_df.columns:
        chart_df = perf_df.set_index("timestamp")[chart_cols].tail(60)
        st.line_chart(chart_df, use_container_width=True)


def _render_prediction_card(pred: dict | None) -> None:
    st.subheader("Latest Prediction")
    if pred is None:
        st.info("No scan run yet. Select an image and click 'Run Scan'.")
        return

    label      = pred["label"]
    label_name = pred["label_name"]
    confidence = pred["confidence"]
    color      = LABEL_COLORS[label]

    st.markdown(
        f"""
        <div style="background:{color}22;border-left:6px solid {color};
                    padding:16px;border-radius:8px;margin-bottom:8px">
          <h2 style="color:{color};margin:0">{label_name.upper()}</h2>
          <p style="font-size:1.4em;margin:4px 0">
            Confidence: <strong>{confidence:.1%}</strong>
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Feature vectors
    with st.expander("Feature vector details"):
        col_a, col_b = st.columns(2)
        col_a.write("**LSTM features (16-dim)**")
        col_a.bar_chart(
            pd.DataFrame({"value": pred["lstm_features"]},
                         index=[f"f{i}" for i in range(len(pred["lstm_features"]))]),
            use_container_width=True,
        )
        col_b.write("**CNN features (128-dim)**")
        col_b.line_chart(
            pd.DataFrame({"value": pred["cnn_features"]},
                         index=[f"f{i}" for i in range(len(pred["cnn_features"]))]),
            use_container_width=True,
        )


def _render_shap_chart(explanation: dict | None) -> None:
    st.subheader("SHAP Feature Importance (LSTM Branch)")
    if explanation is None or "shap_result" not in explanation:
        st.info("Run a scan to see feature importance.")
        return

    shap_r = explanation["shap_result"]
    pairs  = shap_r.get("top_features", [])
    if not pairs:
        st.info("No SHAP values available.")
        return

    names  = [FEATURE_LABELS.get(n, n) for n, _ in pairs]
    values = [abs(v) for _, v in pairs]
    df     = pd.DataFrame({"Feature": names, "Importance": values})
    st.bar_chart(df.set_index("Feature"), use_container_width=True)
    st.caption(shap_r.get("text_summary", ""))


def _render_gradcam(explanation: dict | None) -> None:
    st.subheader("Grad-CAM Memory Heatmap (CNN Branch)")
    if explanation is None or "gradcam_result" not in explanation:
        st.info("Run a scan to see the Grad-CAM heatmap.")
        return

    gcam = explanation["gradcam_result"]
    overlay = gcam.get("overlay")
    if overlay is not None:
        col1, col2 = st.columns(2)
        col1.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)
    st.caption(gcam.get("hotspot_desc", ""))


def _render_response_panel(response: dict | None, explanation: dict | None) -> None:
    st.subheader("Response Status")
    if response is None:
        st.info("No response computed yet.")
        return

    level   = response["level"]
    actions = response["actions_taken"]
    colors  = {"MONITOR": "green", "ALERT": "orange",
                "ISOLATE": "red", "DUMP": "red"}
    color   = colors.get(level, "gray")

    st.markdown(f"**Level:** :{color}[{level}]")
    st.markdown(f"**Actions taken:** {', '.join(actions)}")

    if explanation:
        st.markdown("**Summary:**")
        st.info(explanation.get("short_summary", ""))

    # Manual escalation buttons (with confirmation)
    st.divider()
    st.markdown("**Manual Actions** (require confirmation):")
    col_a, col_b = st.columns(2)

    if col_a.button("Apply Firewall Isolation", type="secondary"):
        st.warning(
            "This will add a Windows Firewall outbound-block rule. "
            "Requires Administrator privileges. Are you sure?"
        )
        if st.button("Confirm Isolation", type="primary"):
            from response.response_engine import _action_isolate
            event_id = st.session_state.get("last_event_id", "manual")
            _action_isolate(event_id, None, interactive=False)
            st.success("Isolation rule added.")

    if col_b.button("Capture Forensic Dump", type="secondary"):
        st.warning(
            "This will dump the current process's memory (~200-500 MB). "
            "Make sure you have enough disk space. Are you sure?"
        )
        if st.button("Confirm Dump", type="primary"):
            from response.response_engine import _action_extra_dump
            event_id = st.session_state.get("last_event_id", "manual")
            _action_extra_dump(event_id, interactive=False)
            st.success("Forensic dump triggered.")


def _render_response_log() -> None:
    st.subheader("Response Log")
    df = load_response_log()
    if df.empty:
        st.info("No responses logged yet.")
        return
    st.dataframe(
        df.sort_values("timestamp", ascending=False).head(20),
        use_container_width=True,
    )


def _render_scan_panel() -> None:
    """Panel to select an image and trigger a scan."""
    st.subheader("Run a Scan")
    labels_df = load_image_labels()
    if labels_df.empty:
        st.warning(
            "No session images found. "
            "Run `event_session.py` first to collect labeled sessions."
        )
        return

    image_options = labels_df["filename"].tolist()
    selected_img  = st.selectbox("Select memory image:", image_options)

    if selected_img:
        row       = labels_df[labels_df["filename"] == selected_img].iloc[0]
        event_id  = row.get("event_id", selected_img.replace(".png", ""))
        img_path  = os.path.join(MEMORY_IMAGES_DIR, selected_img)

        col1, col2 = st.columns([1, 3])
        if os.path.isfile(img_path):
            col1.image(img_path, caption=f"label={row['label']}", width=128)
        col2.markdown(f"""
- **Event ID:** `{event_id}`
- **Label:** {LABEL_NAMES.get(int(row['label']), '?')} ({row['label']})
- **Notes:** {row.get('notes', '-')}
""")

        if st.button("Run Full Scan", type="primary"):
            results = run_scan(event_id, img_path)
            if results:
                st.session_state["last_prediction"]  = results["prediction"]
                st.session_state["last_explanation"] = results["explanation"]
                st.session_state["last_event_id"]    = event_id
                st.session_state["last_response"]    = results["response"]
                load_perf_tail.clear()
                load_response_log.clear()
                st.success("Scan complete!")
                st.rerun()


# ---------------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------------
def main() -> None:
    controls = _render_sidebar()

    st.title("HyperMD Enhanced — Real-Time Malware Monitor")
    st.caption(
        "Multi-Modal Explainable Malware Detection with Automated Response"
    )
    st.divider()

    # Top row: metrics + scan panel
    perf_df = load_perf_tail(120)
    row1_left, row1_right = st.columns([2, 1])
    with row1_left:
        _render_live_metrics(perf_df)
    with row1_right:
        _render_scan_panel()

    st.divider()

    # Middle row: prediction + SHAP + Grad-CAM
    pred  = st.session_state.get("last_prediction")
    expl  = st.session_state.get("last_explanation")
    resp  = st.session_state.get("last_response")

    col1, col2, col3 = st.columns(3)
    with col1:
        _render_prediction_card(pred)
    with col2:
        _render_shap_chart(expl)
    with col3:
        _render_gradcam(expl)

    st.divider()

    # Bottom row: response panel + log
    bot_left, bot_right = st.columns([1, 2])
    with bot_left:
        _render_response_panel(resp, expl)
    with bot_right:
        _render_response_log()

    # Auto-refresh
    if controls["auto_refresh"]:
        time.sleep(controls["refresh_sec"])
        load_perf_tail.clear()
        st.rerun()


if __name__ == "__main__":
    main()

