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
import time

# Import helper functions
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

# Import model classes
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
# PAGE CONFIG & CUSTOM CSS
# -------------------------------------------------------------------
st.set_page_config(
    page_title="🧠 EEG Stress/Calm Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for dark theme, cards, fonts, etc.
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark theme background */
    .stApp {
        background: #0E1117;
        color: #FAFAFA;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: #1E1E2E;
    }

    /* Custom card */
    .glass-card {
        background: rgba(30, 30, 46, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px 0 rgba(0,0,0,0.37);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px 0 rgba(0,255,200,0.2);
    }

    /* Metric card */
    .metric-card {
        background: linear-gradient(145deg, #2A2A3A, #1A1A28);
        border-radius: 16px;
        padding: 1rem;
        text-align: center;
        border-left: 4px solid #00D4B8;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .metric-label {
        color: #A0A0C0;
        font-size: 0.9rem;
        letter-spacing: 0.5px;
    }
    .metric-value {
        color: #FFFFFF;
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-unit {
        color: #00D4B8;
        font-size: 0.9rem;
        margin-left: 4px;
    }

    /* Session banner */
    .session-banner {
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1rem 0;
        font-weight: 700;
        text-align: center;
        font-size: 2rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        background: rgba(0,0,0,0.5);
        backdrop-filter: blur(12px);
        border: 2px solid;
        box-shadow: 0 0 30px rgba(0,255,200,0.3);
    }
    .session-calm {
        border-color: #2ECC71;
        color: #2ECC71;
        text-shadow: 0 0 10px #2ECC71;
    }
    .session-stress {
        border-color: #E74C3C;
        color: #E74C3C;
        text-shadow: 0 0 10px #E74C3C;
    }
    .session-mixed {
        border-color: #F39C12;
        color: #F39C12;
        text-shadow: 0 0 10px #F39C12;
    }

    /* Download button */
    .stDownloadButton button {
        background: linear-gradient(90deg, #00D4B8, #009B8C) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 40px !important;
        padding: 0.75rem 2rem !important;
        box-shadow: 0 4px 20px rgba(0,212,184,0.4) !important;
        transition: all 0.3s !important;
    }
    .stDownloadButton button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 8px 30px rgba(0,212,184,0.6) !important;
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00D4B8, #009B8C) !important;
    }

    /* Tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #00D4B8;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background: #2A2A3A;
        color: #fff;
        text-align: center;
        border-radius: 8px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        border: 1px solid #00D4B8;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# LOAD ARTIFACTS (CACHED)
# -------------------------------------------------------------------
@st.cache_resource
def load_artifacts(model_dir):
    cfg = load_config(model_dir)
    scaler, cnn = load_preprocessors(model_dir, cfg)
    return cfg, scaler, cnn

# -------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧠 **EEG Analyzer**")
    st.markdown("---")

    MODEL_DIR = "trained_api_models"
    if not os.path.isdir(MODEL_DIR):
        st.error(f"Model directory '{MODEL_DIR}' not found.")
        st.stop()

    with st.spinner("Loading artifacts..."):
        cfg, scaler, cnn = load_artifacts(MODEL_DIR)

    model_files = cfg.get("model_files", {})
    available_models = {
        name: os.path.join(MODEL_DIR, fname)
        for name, fname in model_files.items()
        if os.path.exists(os.path.join(MODEL_DIR, fname))
    }
    if not available_models:
        st.error("No model files found.")
        st.stop()

    quantum_models = set(cfg.get("quantum_models", []))

    selected_model_name = st.selectbox(
        "🤖 Select Model",
        options=list(available_models.keys()),
        format_func=lambda x: f"{x} ⚛️" if x in quantum_models else x,
    )
    model_path = available_models[selected_model_name]

    uploaded_file = st.file_uploader(
        "📁 Upload EEG CSV",
        type=["csv"],
        help=f"Required channels: {cfg['selected_channels']}",
    )

    run_pressed = st.button("🚀 **Run Analysis**", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption("✨ Built with Streamlit & ❤️")

# -------------------------------------------------------------------
# MAIN CONTENT
# -------------------------------------------------------------------
st.markdown("<h1 style='text-align: center; font-weight: 700;'>🧠 EEG Stress / Calm Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #A0A0C0;'>Upload a recording and let AI reveal your mental state.</p>", unsafe_allow_html=True)

if not uploaded_file:
    st.info("📂 **Start by uploading a CSV file** from the sidebar.")
    st.stop()

if not run_pressed:
    st.info("👈 **Press 'Run Analysis'** to begin.")
    st.stop()

# -------------------------------------------------------------------
# PROCESSING WITH ANIMATED PROGRESS
# -------------------------------------------------------------------
progress_bar = st.progress(0, text="Initializing...")
status_text = st.empty()

try:
    # Step 1: Read CSV
    status_text.text("📖 Reading CSV...")
    progress_bar.progress(10)
    file_content = uploaded_file.getvalue().decode('utf-8')
    df = pd.read_csv(io.StringIO(file_content))

    missing = [ch for ch in cfg["selected_channels"] if ch not in df.columns]
    if missing:
        st.error(f"Missing channels: {missing}")
        st.stop()

    raw = df[cfg["selected_channels"]].values.astype(np.float32)
    time.sleep(0.5)  # simulate work

    # Step 2: Preprocess
    status_text.text("🔧 Preprocessing (filtering, scaling, feature extraction)...")
    progress_bar.progress(30)
    windows_scaled, features_8D = preprocess(raw, cfg, scaler, cnn)
    time.sleep(0.5)

    # Step 3: Load model
    status_text.text("🤖 Loading model...")
    progress_bar.progress(50)
    clf = joblib.load(model_path)
    time.sleep(0.5)

    # Step 4: Inference
    status_text.text("⚡ Running inference...")
    progress_bar.progress(70)
    label_map_inv = {int(k): v for k, v in cfg["label_map_inverse"].items()}
    step_size = cfg["step_size"]
    fs = cfg["fs"]

    raw_preds = clf.predict(features_8D)
    raw_labels = [label_map_inv[int(p)] for p in raw_preds]
    labels = smooth_labels(raw_labels, window=20)

    # Step 5: Compute stats
    status_text.text("📊 Analyzing results...")
    progress_bar.progress(90)
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

    progress_bar.progress(100)
    status_text.text("✅ Analysis complete!")
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()

except Exception as e:
    st.error(f"Analysis failed: {e}")
    st.stop()

# -------------------------------------------------------------------
# DISPLAY RESULTS
# -------------------------------------------------------------------
st.markdown("---")

# ----- Session banner -----
banner_class = {
    "CALM": "session-calm",
    "STRESS": "session-stress",
    "MIXED": "session-mixed"
}.get(session_type, "")
st.markdown(
    f'<div class="session-banner {banner_class}">{session_type}</div>',
    unsafe_allow_html=True
)

# ----- Metric cards row -----
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">DOMINANT STATE</div>
            <div class="metric-value">{dominant.capitalize()}</div>
        </div>
        """, unsafe_allow_html=True
    )
with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">TOTAL TIME</div>
            <div class="metric-value">{fmt_time(duration['total_s'])}<span class="metric-unit">min</span></div>
        </div>
        """, unsafe_allow_html=True
    )
with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">STATE SWITCHES</div>
            <div class="metric-value">{transitions}</div>
        </div>
        """, unsafe_allow_html=True
    )
with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">CALM</div>
            <div class="metric-value">{duration['calm_pct']:.0f}%</div>
        </div>
        """, unsafe_allow_html=True
    )
with col5:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">STRESS</div>
            <div class="metric-value">{duration['stress_pct']:.0f}%</div>
        </div>
        """, unsafe_allow_html=True
    )

# ----- Two columns for charts -----
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 🥧 Distribution")
    fig_pie = px.pie(
        names=["Calm", "Stress"],
        values=[duration["calm_s"], duration["stress_s"]],
        color_discrete_sequence=["#2ECC71", "#E74C3C"],
        hole=0.4,
    )
    fig_pie.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        showlegend=False,
        margin=dict(t=30, b=0, l=0, r=0),
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label', textfont_color='white')
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.markdown("### 📊 Calm vs Stress")
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=["Calm", "Stress"],
        y=[duration["calm_s"], duration["stress_s"]],
        marker_color=["#2ECC71", "#E74C3C"],
        text=[fmt_time(duration["calm_s"]), fmt_time(duration["stress_s"])],
        textposition='outside',
    ))
    fig_bar.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        yaxis_title="Time (s)",
        margin=dict(t=30, b=0, l=0, r=0),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ----- Gantt chart -----
st.markdown("### 📈 Detected Segments Over Time")
if segments:
    df_seg = pd.DataFrame(segments)
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
        color_discrete_map={"CALM": "#2ECC71", "STRESS": "#E74C3C"},
        hover_data={
            "start_dt": False,
            "end_dt": False,
            "duration_str": True,
            "consistency_pct": True,
        },
        labels={"duration_str": "Duration", "consistency_pct": "Consistency (%)"}
    )
    fig_gantt.update_yaxes(categoryorder="total ascending")
    fig_gantt.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        xaxis_title="Time (MM:SS)",
        xaxis_tickformat="%M:%S",
        hovermode="closest",
        margin=dict(t=30, b=0, l=0, r=0),
    )
    st.plotly_chart(fig_gantt, use_container_width=True)

# ----- Peak periods -----
if peak_calm or peak_stress:
    st.markdown("### 🌟 Peak Periods (Highest Consistency)")
    pc, ps = st.columns(2)
    if peak_calm:
        with pc:
            st.markdown(
                f"""
                <div class="glass-card">
                    <h4 style="color:#2ECC71;">😌 Calm Peak</h4>
                    <p style="font-size:1.2rem;">{fmt_time(peak_calm['start_s'])} → {fmt_time(peak_calm['end_s'])}</p>
                    <p>⏱️ {peak_calm['duration_s']:.1f} s  |  ✅ {peak_calm['consistency']*100:.1f}% consistent</p>
                </div>
                """, unsafe_allow_html=True
            )
    if peak_stress:
        with ps:
            st.markdown(
                f"""
                <div class="glass-card">
                    <h4 style="color:#E74C3C;">😰 Stress Peak</h4>
                    <p style="font-size:1.2rem;">{fmt_time(peak_stress['start_s'])} → {fmt_time(peak_stress['end_s'])}</p>
                    <p>⏱️ {peak_stress['duration_s']:.1f} s  |  ✅ {peak_stress['consistency']*100:.1f}% consistent</p>
                </div>
                """, unsafe_allow_html=True
            )

# ----- Segment table -----
st.markdown("### 📋 Segment Details")
if segments:
    df_display = df_seg[["start_dt", "end_dt", "duration_str", "State", "consistency_pct"]].copy()
    df_display["Start"] = df_display["start_dt"].dt.strftime("%M:%S")
    df_display["End"] = df_display["end_dt"].dt.strftime("%M:%S")
    df_display = df_display[["Start", "End", "duration_str", "State", "consistency_pct"]]
    df_display.columns = ["Start", "End", "Duration", "State", "Consistency (%)"]

    st.dataframe(
        df_display.style.applymap(
            lambda x: 'color: #2ECC71' if x == 'CALM' else ('color: #E74C3C' if x == 'STRESS' else ''),
            subset=['State']
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Consistency (%)": st.column_config.ProgressColumn(
                "Consistency (%)", format="%.1f%%", min_value=0, max_value=100
            )
        }
    )

# ----- Download button -----
st.markdown("<br>", unsafe_allow_html=True)
st.download_button(
    label="📥 Download Full Results (JSON)",
    data=json.dumps(result, indent=2, default=str),
    file_name=f"result_{Path(uploaded_file.name).stem}_{selected_model_name}.json",
    mime="application/json",
)
