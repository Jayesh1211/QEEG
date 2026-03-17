# streamlit_app.py
import os
import io
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Import helper functions from phase2_1_updated (avoid main())
from phase2_1_updated import (
    load_config,
    load_preprocessors,
    preprocess,
    smooth_labels,
    detect_segments,
    compute_duration_stats,
    classify_session,
    find_longest_block,
    find_peak_seg,
    fmt_time,
)

# Import model classes so joblib can deserialise .pkl files
try:
    from API import (
        EEGClassicalSVM,
        EEGXGBoost,
        EEGLightGBM,
        EEGRandomForest,
        EEGVotingEnsemble,
        EEGStackingEnsemble,
        EEGQSVM,
        EEGVQC,
        HybridVQC_TorchModule,
    )
except ImportError as e:
    st.sidebar.warning(f"Some model classes could not be imported: {e}")

# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="🧠 EEG Stress/Calm Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------
# Cache artifact loading (config, scaler, CNN)
# -------------------------------------------------------------------
@st.cache_resource
def load_artifacts(model_dir):
    """Load config, scaler and CNN extractor from model_dir."""
    cfg = load_config(model_dir)
    scaler, cnn = load_preprocessors(model_dir, cfg)
    return cfg, scaler, cnn

# -------------------------------------------------------------------
# Sidebar: model selection & file upload
# -------------------------------------------------------------------
st.sidebar.title("🧠 EEG Analyzer")
st.sidebar.markdown("---")

# Model directory (adjust if needed)
MODEL_DIR = "trained_api_models"
if not os.path.isdir(MODEL_DIR):
    st.sidebar.error(f"Model directory '{MODEL_DIR}' not found.")
    st.stop()

# Load artifacts
with st.spinner("Loading preprocessing artifacts..."):
    cfg, scaler, cnn = load_artifacts(MODEL_DIR)

# Build list of available models from pipeline_config.json
model_files = cfg.get("model_files", {})
available_models = {
    name: os.path.join(MODEL_DIR, fname)
    for name, fname in model_files.items()
    if os.path.exists(os.path.join(MODEL_DIR, fname))
}

if not available_models:
    st.sidebar.error("No model files found in the model directory.")
    st.stop()

# Quantum model list (for tagging)
quantum_models = set(cfg.get("quantum_models", []))

# Model selector
selected_model_name = st.sidebar.selectbox(
    "Select a trained model",
    options=list(available_models.keys()),
    format_func=lambda x: f"{x} ⚛️" if x in quantum_models else x,
)
model_path = available_models[selected_model_name]

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload raw EEG CSV",
    type=["csv"],
    help=f"Required columns: {cfg['selected_channels']}",
)

# Run button
run_pressed = st.sidebar.button("🚀 Run Analysis", type="primary")

st.sidebar.markdown("---")
st.sidebar.info(
    "This tool uses pre‑trained models to detect calm/stress states "
    "from EEG signals. Upload a CSV file with the same channels used "
    "during training."
)

# -------------------------------------------------------------------
# Main area
# -------------------------------------------------------------------
st.title("🧠 EEG Stress / Calm Analysis")
st.markdown("Upload a raw EEG recording and select a model to begin.")

if not uploaded_file:
    st.info("📂 Use the sidebar to upload a CSV file.")
    st.stop()

if not run_pressed:
    st.info("👈 Press **Run Analysis** after selecting a file.")
    st.stop()

# -------------------------------------------------------------------
# Run analysis when button is pressed
# -------------------------------------------------------------------
with st.spinner("Processing..."):
    try:
        # 1. Load raw data from uploaded file (directly from memory)
        file_content = uploaded_file.getvalue().decode('utf-8')
        df = pd.read_csv(io.StringIO(file_content))

        # Check required channels
        missing = [ch for ch in cfg["selected_channels"] if ch not in df.columns]
        if missing:
            st.error(
                f"CSV is missing required channel(s): {missing}\n"
                f"Required: {cfg['selected_channels']}\n"
                f"Found: {list(df.columns)}"
            )
            st.stop()

        raw = df[cfg["selected_channels"]].values.astype(np.float32)
        st.info(
            f"CSV loaded: {raw.shape[0]} samples "
            f"({raw.shape[0] / cfg['fs']:.1f} s) × {raw.shape[1]} channels"
        )

        # 2. Preprocess: filter → windows → scale → CNN features
        windows_scaled, features_8D = preprocess(raw, cfg, scaler, cnn)

        # 3. Load the selected model
        with st.spinner("Loading model..."):
            clf = joblib.load(model_path)

        # 4. Run inference
        label_map_inv = {int(k): v for k, v in cfg["label_map_inverse"].items()}
        step_size = cfg["step_size"]
        fs = cfg["fs"]

        raw_preds = clf.predict(features_8D)
        raw_labels = [label_map_inv[int(p)] for p in raw_preds]
        labels = smooth_labels(raw_labels, window=20)  # optional smoothing

        # 5. Compute statistics
        duration = compute_duration_stats(labels, step_size, fs)
        segments = detect_segments(labels, step_size, fs)
        session_type = classify_session(duration["calm_pct"], duration["stress_pct"])
        dominant = "calm" if duration["calm_pct"] >= duration["stress_pct"] else "stress"
        transitions = len(segments) - 1

        # 6. Peak segments
        longest_calm = (find_longest_block(segments, "calm") or {}).get("duration_s", 0)
        longest_stress = (find_longest_block(segments, "stress") or {}).get("duration_s", 0)
        peak_calm = find_peak_seg(segments, "calm")
        peak_stress = find_peak_seg(segments, "stress")

        result = {
            "model_used": selected_model_name,
            "is_quantum_model": selected_model_name in quantum_models,
            "windows_processed": len(labels),
            "window_labels": labels,
            "session_type": session_type,
            "dominant_state": dominant,
            "duration": duration,
            "segments": segments,
            "transitions": transitions,
            "longest_calm_s": longest_calm,
            "longest_stress_s": longest_stress,
            "peak_calm_seg": peak_calm,
            "peak_stress_seg": peak_stress,
        }

    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.stop()

# -------------------------------------------------------------------
# Display results
# -------------------------------------------------------------------
st.success("Analysis complete!")

# ----- Top metrics -----
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Session Type", result["session_type"])
with col2:
    st.metric("Dominant State", result["dominant_state"].capitalize())
with col3:
    st.metric("Total Duration", f"{fmt_time(duration['total_s'])}")
with col4:
    st.metric("State Switches", result["transitions"])
with col5:
    st.metric("Calm / Stress", f"{duration['calm_pct']:.0f}% / {duration['stress_pct']:.0f}%")

# ----- Pie chart for distribution -----
fig_pie = px.pie(
    names=["Calm", "Stress"],
    values=[duration["calm_s"], duration["stress_s"]],
    color_discrete_sequence=["#66c2a5", "#fc8d62"],
    title="Calm / Stress Distribution",
)
st.plotly_chart(fig_pie, use_container_width=True)

# ----- Gantt chart of segments -----
if segments:
    df_seg = pd.DataFrame(segments)
    df_seg["Start"] = df_seg["start_s"].apply(fmt_time)
    df_seg["End"] = df_seg["end_s"].apply(fmt_time)
    df_seg["Duration"] = df_seg["duration_s"].apply(lambda x: f"{x:.1f}s")
    df_seg["State"] = df_seg["state"].str.upper()
    df_seg["Consistency"] = df_seg["consistency"].apply(lambda x: f"{x*100:.1f}%")

    fig_gantt = px.timeline(
        df_seg,
        x_start="start_s",
        x_end="end_s",
        y="State",
        color="State",
        color_discrete_map={"CALM": "#66c2a5", "STRESS": "#fc8d62"},
        hover_data={
            "start_s": False,
            "end_s": False,
            "Duration": True,
            "Consistency": True,
        },
        title="Detected Segments Over Time",
    )
    fig_gantt.update_yaxes(categoryorder="total ascending")
    fig_gantt.update_layout(xaxis_title="Time (s)")
    st.plotly_chart(fig_gantt, use_container_width=True)

# ----- Peak periods -----
if peak_calm or peak_stress:
    st.subheader("🌟 Peak Periods (Highest Consistency)")
    pc, ps = st.columns(2)
    if peak_calm:
        with pc:
            st.markdown(
                f"**Calm**  \n"
                f"`{fmt_time(peak_calm['start_s'])} – {fmt_time(peak_calm['end_s'])}`  \n"
                f"⏱️ {peak_calm['duration_s']:.1f} s  \n"
                f"✅ Consistency: {peak_calm['consistency']*100:.1f}%"
            )
    if peak_stress:
        with ps:
            st.markdown(
                f"**Stress**  \n"
                f"`{fmt_time(peak_stress['start_s'])} – {fmt_time(peak_stress['end_s'])}`  \n"
                f"⏱️ {peak_stress['duration_s']:.1f} s  \n"
                f"✅ Consistency: {peak_stress['consistency']*100:.1f}%"
            )

# ----- Segment table -----
st.subheader("📋 Segment Details")
if segments:
    df_display = df_seg[["Start", "End", "Duration", "State", "Consistency"]]
    st.dataframe(df_display, use_container_width=True, hide_index=True)

# ----- Download JSON -----
st.download_button(
    label="📥 Download Results as JSON",
    data=json.dumps(result, indent=2, default=str),
    file_name=f"result_{Path(uploaded_file.name).stem}_{selected_model_name}.json",
    mime="application/json",
)
