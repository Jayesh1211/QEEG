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
from datetime import timedelta

# Import helper functions from phase2_1_updated
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

# Import model classes for deserialisation
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
# Page config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="🧠 EEG Stress/Calm Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .session-banner {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        font-weight: bold;
        text-align: center;
        font-size: 1.5rem;
    }
    .calm-banner { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .stress-banner { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    .mixed-banner { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Load artifacts (cached)
# -------------------------------------------------------------------
@st.cache_resource
def load_artifacts(model_dir):
    cfg = load_config(model_dir)
    scaler, cnn = load_preprocessors(model_dir, cfg)
    return cfg, scaler, cnn

# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
st.sidebar.title("🧠 EEG Analyzer")
st.sidebar.markdown("---")

MODEL_DIR = "trained_api_models"
if not os.path.isdir(MODEL_DIR):
    st.sidebar.error(f"Model directory '{MODEL_DIR}' not found.")
    st.stop()

with st.spinner("Loading preprocessing artifacts..."):
    cfg, scaler, cnn = load_artifacts(MODEL_DIR)

model_files = cfg.get("model_files", {})
available_models = {
    name: os.path.join(MODEL_DIR, fname)
    for name, fname in model_files.items()
    if os.path.exists(os.path.join(MODEL_DIR, fname))
}
if not available_models:
    st.sidebar.error("No model files found.")
    st.stop()

quantum_models = set(cfg.get("quantum_models", []))

selected_model_name = st.sidebar.selectbox(
    "Select a trained model",
    options=list(available_models.keys()),
    format_func=lambda x: f"{x} ⚛️" if x in quantum_models else x,
)
model_path = available_models[selected_model_name]

uploaded_file = st.sidebar.file_uploader(
    "Upload raw EEG CSV",
    type=["csv"],
    help=f"Required columns: {cfg['selected_channels']}",
)

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
# Run analysis
# -------------------------------------------------------------------
with st.spinner("Processing..."):
    try:
        # Read uploaded file
        file_content = uploaded_file.getvalue().decode('utf-8')
        df = pd.read_csv(io.StringIO(file_content))

        missing = [ch for ch in cfg["selected_channels"] if ch not in df.columns]
        if missing:
            st.error(f"Missing channels: {missing}")
            st.stop()

        raw = df[cfg["selected_channels"]].values.astype(np.float32)
        st.info(f"Loaded {raw.shape[0]} samples ({raw.shape[0]/cfg['fs']:.1f} s) × {raw.shape[1]} channels")

        # Preprocess
        windows_scaled, features_8D = preprocess(raw, cfg, scaler, cnn)

        # Load model
        with st.spinner("Loading model..."):
            clf = joblib.load(model_path)

        # Inference
        label_map_inv = {int(k): v for k, v in cfg["label_map_inverse"].items()}
        step_size = cfg["step_size"]
        fs = cfg["fs"]

        raw_preds = clf.predict(features_8D)
        raw_labels = [label_map_inv[int(p)] for p in raw_preds]
        labels = smooth_labels(raw_labels, window=20)

        duration = compute_duration_stats(labels, step_size, fs)
        segments = detect_segments(labels, step_size, fs)
        session_type = classify_session(duration["calm_pct"], duration["stress_pct"])
        dominant = "calm" if duration["calm_pct"] >= duration["stress_pct"] else "stress"
        transitions = len(segments) - 1

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

# ----- Session type banner -----
banner_class = {
    "CALM": "calm-banner",
    "STRESS": "stress-banner",
    "MIXED": "mixed-banner"
}.get(session_type, "")
st.markdown(
    f'<div class="session-banner {banner_class}">SESSION TYPE: {session_type}</div>',
    unsafe_allow_html=True
)

# ----- Top metrics in columns -----
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Dominant State", dominant.capitalize())
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Total Time", fmt_time(duration['total_s']))
    st.markdown('</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("State Switches", transitions)
    st.markdown('</div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Calm", f"{duration['calm_pct']:.0f}%")
    st.markdown('</div>', unsafe_allow_html=True)
with col5:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Stress", f"{duration['stress_pct']:.0f}%")
    st.markdown('</div>', unsafe_allow_html=True)

# ----- Pie chart -----
fig_pie = px.pie(
    names=["Calm", "Stress"],
    values=[duration["calm_s"], duration["stress_s"]],
    color_discrete_sequence=["#2ecc71", "#e74c3c"],
    title="Calm / Stress Distribution",
)
fig_pie.update_traces(textposition='inside', textinfo='percent+label')
st.plotly_chart(fig_pie, use_container_width=True)

# ----- Gantt chart (fixed) -----
if segments:
    df_seg = pd.DataFrame(segments)
    # Convert seconds to datetime for proper timeline display
    base_time = pd.Timestamp('1970-01-01')
    df_seg['start_dt'] = base_time + pd.to_timedelta(df_seg['start_s'], unit='s')
    df_seg['end_dt'] = base_time + pd.to_timedelta(df_seg['end_s'], unit='s')
    df_seg['duration_str'] = df_seg['duration_s'].apply(lambda x: str(timedelta(seconds=int(x))))
    df_seg['State'] = df_seg['state'].str.upper()
    df_seg['consistency_pct'] = (df_seg['consistency'] * 100).round(1)

    fig_gantt = px.timeline(
        df_seg,
        x_start="start_dt",
        x_end="end_dt",
        y="State",
        color="State",
        color_discrete_map={"CALM": "#2ecc71", "STRESS": "#e74c3c"},
        hover_data={
            "start_dt": False,
            "end_dt": False,
            "duration_str": True,
            "consistency_pct": True,
        },
        title="Detected Segments Over Time",
        labels={"duration_str": "Duration", "consistency_pct": "Consistency (%)"}
    )
    fig_gantt.update_yaxes(categoryorder="total ascending")
    fig_gantt.update_layout(
        xaxis_title="Time (MM:SS)",
        xaxis_tickformat="%M:%S",
        hovermode="closest"
    )
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
    df_display = df_seg[["start_dt", "end_dt", "duration_str", "State", "consistency_pct"]].copy()
    df_display["Start"] = df_display["start_dt"].dt.strftime("%M:%S")
    df_display["End"] = df_display["end_dt"].dt.strftime("%M:%S")
    df_display = df_display[["Start", "End", "duration_str", "State", "consistency_pct"]]
    df_display.columns = ["Start", "End", "Duration", "State", "Consistency (%)"]
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Consistency (%)": st.column_config.ProgressColumn(
                "Consistency (%)", format="%.1f%%", min_value=0, max_value=100
            )
        }
    )

# ----- Download JSON -----
st.download_button(
    label="📥 Download Results as JSON",
    data=json.dumps(result, indent=2, default=str),
    file_name=f"result_{Path(uploaded_file.name).stem}_{selected_model_name}.json",
    mime="application/json",
)
