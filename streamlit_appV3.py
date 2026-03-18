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

# Import from the new phase2_1_updatedv3
from phase2_1_updatedv3 import (
    load_config,
    load_preprocessors,
    preprocess,
    run_inference,
    compute_stress_score,
    _build_intensity_series,
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

st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* Dark theme background */
    .stApp {
        background: linear-gradient(145deg, #0B0E14 0%, #1A1F2C 100%);
        color: #F0F0F0;
    }

    /* Sidebar */
    .css-1d391kg {
        background: #1E1E2E !important;
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    /* Glass card */
    .glass-card {
        background: rgba(30,30,46,0.7);
        backdrop-filter: blur(12px);
        border-radius: 24px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 30px 60px rgba(0,212,184,0.2);
        border-color: rgba(0,212,184,0.3);
    }

    /* Metric card */
    .metric-card {
        background: linear-gradient(135deg, #2A2A3A, #1E1E2E);
        border-radius: 20px;
        padding: 1.2rem 1rem;
        text-align: center;
        border-left: 6px solid #00D4B8;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        transition: all 0.3s;
    }
    .metric-card:hover {
        border-left-width: 8px;
        box-shadow: 0 12px 24px rgba(0,212,184,0.3);
    }
    .metric-label {
        color: #A0A0C0;
        font-size: 0.85rem;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        font-weight: 600;
    }
    .metric-value {
        color: #FFFFFF;
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1.2;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-unit {
        color: #00D4B8;
        font-size: 0.9rem;
        margin-left: 4px;
    }

    /* Session banner */
    .session-banner {
        padding: 1.8rem;
        border-radius: 40px;
        margin: 1.5rem 0;
        font-weight: 800;
        text-align: center;
        font-size: 2.5rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        background: rgba(0,0,0,0.3);
        backdrop-filter: blur(16px);
        border: 2px solid;
        box-shadow: 0 0 40px currentColor;
        animation: glowPulse 2s infinite alternate;
    }
    @keyframes glowPulse {
        from { box-shadow: 0 0 20px currentColor; }
        to { box-shadow: 0 0 60px currentColor; }
    }
    .session-calm {
        border-color: #2ECC71;
        color: #2ECC71;
        text-shadow: 0 0 15px #2ECC71;
    }
    .session-stress {
        border-color: #E74C3C;
        color: #E74C3C;
        text-shadow: 0 0 15px #E74C3C;
    }
    .session-mixed {
        border-color: #F39C12;
        color: #F39C12;
        text-shadow: 0 0 15px #F39C12;
    }

    /* Download button */
    .stDownloadButton button {
        background: linear-gradient(90deg, #00D4B8, #009B8C) !important;
        color: white !important;
        border: none !important;
        font-weight: 700 !important;
        border-radius: 60px !important;
        padding: 0.9rem 2.5rem !important;
        font-size: 1.1rem !important;
        box-shadow: 0 10px 20px rgba(0,212,184,0.3) !important;
        transition: all 0.3s !important;
        width: 100%;
    }
    .stDownloadButton button:hover {
        transform: scale(1.03) !important;
        box-shadow: 0 15px 30px rgba(0,212,184,0.5) !important;
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00D4B8, #2ECC71) !important;
        border-radius: 20px;
    }

    /* Dataframe styling */
    .stDataFrame {
        background: transparent !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        color: #6B6B8B;
        font-size: 0.9rem;
        border-top: 1px solid rgba(255,255,255,0.05);
    }
    .footer a {
        color: #00D4B8;
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# LOAD ARTIFACTS (CACHED)
# -------------------------------------------------------------------
@st.cache_resource
def load_artifacts(model_dir):
    cfg = load_config(model_dir)
    # Ensure slow_models key exists
    if "slow_models" not in cfg:
        cfg["slow_models"] = {}
    scaler, cnn = load_preprocessors(model_dir, cfg)
    return cfg, scaler, cnn

# -------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='color: #00D4B8; font-weight: 700;'>🧠 EEG Analyzer</h2>", unsafe_allow_html=True)
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

    st.markdown("<p style='color: #A0A0C0; margin-bottom: 0;'>🤖 Select Models</p>", unsafe_allow_html=True)
    selected_model_names = st.multiselect(
        "",
        options=list(available_models.keys()),
        format_func=lambda x: f"{x} ⚛️" if x in quantum_models else x,
        label_visibility="collapsed"
    )

    st.markdown("<p style='color: #A0A0C0; margin-bottom: 0;'>📁 Upload EEG CSV</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "",
        type=["csv"],
        help=f"Required channels: {cfg['selected_channels']}",
        label_visibility="collapsed"
    )

    run_pressed = st.button("🚀 **Run Analysis**", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption("✨ Built with Streamlit & ❤️")

# -------------------------------------------------------------------
# MAIN CONTENT
# -------------------------------------------------------------------
st.markdown("<h1 style='text-align: center; font-weight: 700; font-size: 3rem; background: linear-gradient(90deg, #00D4B8, #2ECC71); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>🧠 EEG Stress / Calm Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #A0A0C0; font-size: 1.2rem;'>Upload a recording and let AI reveal your mental state.</p>", unsafe_allow_html=True)

if not uploaded_file:
    st.info("📂 **Start by uploading a CSV file** from the sidebar.")
    st.stop()

if not run_pressed:
    st.info("👈 **Select models and press 'Run Analysis'** to begin.")
    st.stop()

if not selected_model_names:
    st.warning("Please select at least one model.")
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
    df_csv = pd.read_csv(io.StringIO(file_content))

    missing = [ch for ch in cfg["selected_channels"] if ch not in df_csv.columns]
    if missing:
        st.error(f"Missing channels: {missing}")
        st.stop()

    raw = df_csv[cfg["selected_channels"]].values.astype(np.float32)
    time.sleep(0.5)

    # Step 2: Preprocess
    status_text.text("🔧 Preprocessing (filtering, scaling, feature extraction)...")
    progress_bar.progress(30)
    windows_scaled, features_8D = preprocess(raw, cfg, scaler, cnn)
    time.sleep(0.5)

    # Step 3: Run inference for all selected models
    all_results = {}
    all_dfs = {}
    n_models = len(selected_model_names)

    for idx, model_name in enumerate(selected_model_names):
        status_text.text(f"⚡ Running inference for {model_name}...")
        progress_bar.progress(50 + int(40 * (idx / n_models)))

        model_path = available_models[model_name]

        # Load model with error handling for quantum models
        try:
            clf = joblib.load(model_path)
        except Exception as e:
            st.warning(f"Failed to load {model_name}: {e}. Skipping.")
            continue

        # Run inference using v3 function
        result, df_pred = run_inference(model_name, clf, features_8D, cfg)
        all_results[model_name] = result
        all_dfs[model_name] = df_pred

    progress_bar.progress(100)
    status_text.text("✅ Analysis complete!")
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()

    if not all_results:
        st.error("No models could be loaded. Please check your model files.")
        st.stop()

except Exception as e:
    st.error(f"Analysis failed: {e}")
    st.stop()

# -------------------------------------------------------------------
# DISPLAY RESULTS
# -------------------------------------------------------------------
st.markdown("---")

# Create tabs for each model + comparison tab
model_tabs = st.tabs([f"📊 {name}" for name in all_results.keys()] + ["🔍 Comparison"])

# For each model, display its results
for tab, (model_name, result) in zip(model_tabs[:-1], all_results.items()):
    with tab:
        duration = result["duration"]
        segments = result["segments"]
        session_type = result["session_type"]
        dominant = result["dominant_state"]
        transitions = result["transitions"]
        stress_score = result["stress_score"]

        # Session banner
        banner_class = {
            "CALM": "session-calm",
            "STRESS": "session-stress",
            "MIXED": "session-mixed"
        }.get(session_type, "")
        st.markdown(
            f'<div class="session-banner {banner_class}">{session_type}</div>',
            unsafe_allow_html=True
        )

        # Metric cards
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">DOMINANT STATE</div><div class="metric-value">{dominant.capitalize()}</div></div>', unsafe_allow_html=True)
        with col2:
            total_time_formatted = fmt_time(duration['total_s'])
            st.markdown(f'<div class="metric-card"><div class="metric-label">TOTAL TIME</div><div class="metric-value">{total_time_formatted}</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">STATE SWITCHES</div><div class="metric-value">{transitions}</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-label">CALM</div><div class="metric-value">{duration["calm_pct"]:.0f}%</div></div>', unsafe_allow_html=True)
        with col5:
            st.markdown(f'<div class="metric-card"><div class="metric-label">STRESS</div><div class="metric-value">{duration["stress_pct"]:.0f}%</div></div>', unsafe_allow_html=True)

        # Stress score display
        st.markdown("### 🧠 Stress Score")
        score = stress_score["score"]
        label = stress_score["label"]
        advice = stress_score["advice"]
        bar = "█" * int(score) + "░" * (10 - int(score))
        st.markdown(f"**{bar}**  {score}/10  —  **{label}**  \n{advice}")

        # Distribution and bar charts
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("### 🥧 Distribution")
            fig_pie = px.pie(
                names=["Calm", "Stress"],
                values=[duration["calm_s"], duration["stress_s"]],
                color_discrete_sequence=["#2ECC71", "#E74C3C"],
                hole=0.45,
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                margin=dict(t=50, b=0, l=0, r=0),
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent', textfont_color='white', marker=dict(line=dict(color='#1E1E2E', width=2)))
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
                textfont=dict(color='white', size=14),
                width=[0.6, 0.6],
            ))
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                yaxis_title="Time (MM:SS)",
                margin=dict(t=30, b=0, l=0, r=0),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Stress intensity plot for this model
        st.markdown("### 📈 Stress Intensity Over Time")
        df_plot = _build_intensity_series(all_dfs[model_name], cfg["step_size"], cfg["fs"])
        fig_intensity = go.Figure()
        fig_intensity.add_trace(go.Scatter(
            x=df_plot['Time_Minutes'],
            y=df_plot['Stress_Intensity'],
            mode='lines',
            name='Stress Intensity',
            line=dict(color='#E74C3C', width=2),
            fill='tozeroy',
            fillcolor='rgba(231, 76, 60, 0.2)'
        ))
        fig_intensity.add_hline(y=50, line_dash="dash", line_color="#F39C12", annotation_text="Threshold (50%)", annotation_position="bottom right")
        fig_intensity.update_layout(
            title=f"Continuous Cognitive State – {model_name}",
            xaxis_title="Time (Minutes)",
            yaxis_title="Stress Probability (%)",
            yaxis=dict(range=[0, 105]),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            hovermode='x unified',
            margin=dict(t=50, b=0, l=0, r=0),
        )
        st.plotly_chart(fig_intensity, use_container_width=True)

        # Gantt chart
        if segments:
            st.markdown("### 📈 Detected Segments Over Time")
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
            fig_gantt.update_traces(marker=dict(line=dict(width=0)))
            st.plotly_chart(fig_gantt, use_container_width=True)

        # Peak periods
        peak_calm = result.get("peak_calm_seg")
        peak_stress = result.get("peak_stress_seg")
        if peak_calm or peak_stress:
            st.markdown("### 🌟 Peak Periods (Longest Uninterrupted Block)")
            pc, ps = st.columns(2)
            if peak_calm:
                with pc:
                    st.markdown(
                        f"""
                        <div class="glass-card">
                            <h4 style="color:#2ECC71; margin-bottom: 0.5rem;">😌 Calm Peak</h4>
                            <p style="font-size:1.4rem; font-family: 'JetBrains Mono', monospace;">{fmt_time(peak_calm['start_s'])} → {fmt_time(peak_calm['end_s'])}</p>
                            <p>⏱️ <strong>{peak_calm['duration_s']:.1f} s</strong>  |  ✅ Consistency: <strong>{peak_calm['consistency']*100:.1f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True
                    )
            if peak_stress:
                with ps:
                    st.markdown(
                        f"""
                        <div class="glass-card">
                            <h4 style="color:#E74C3C; margin-bottom: 0.5rem;">😰 Stress Peak</h4>
                            <p style="font-size:1.4rem; font-family: 'JetBrains Mono', monospace;">{fmt_time(peak_stress['start_s'])} → {fmt_time(peak_stress['end_s'])}</p>
                            <p>⏱️ <strong>{peak_stress['duration_s']:.1f} s</strong>  |  ✅ Consistency: <strong>{peak_stress['consistency']*100:.1f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True
                    )

        # Segment table
        if segments:
            st.markdown("### 📋 Segment Details")
            df_display = df_seg[["start_dt", "end_dt", "duration_str", "State", "consistency_pct"]].copy()
            df_display["Start"] = df_display["start_dt"].dt.strftime("%M:%S")
            df_display["End"] = df_display["end_dt"].dt.strftime("%M:%S")
            df_display = df_display[["Start", "End", "duration_str", "State", "consistency_pct"]]
            df_display.columns = ["Start", "End", "Duration", "State", "Consistency (%)"]

            st.dataframe(
                df_display.style.applymap(
                    lambda x: 'color: #2ECC71; font-weight: 600;' if x == 'CALM' else ('color: #E74C3C; font-weight: 600;' if x == 'STRESS' else ''),
                    subset=['State']
                ),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Consistency (%)": st.column_config.ProgressColumn(
                        "Consistency (%)", format="%.1f%%", min_value=0, max_value=100,
                        width="medium"
                    )
                }
            )

# -------------------------------------------------------------------
# COMPARISON TAB (with model checkboxes)
# -------------------------------------------------------------------
with model_tabs[-1]:
    st.markdown("## 🔍 Model Comparison")

    # Comparison table
    st.markdown("### 📊 Key Metrics")
    comp_data = []
    for model_name, result in all_results.items():
        ss = result["stress_score"]
        d = result["duration"]
        comp_data.append({
            "Model": model_name,
            "Stress Score": f"{ss['score']}/10 ({ss['label']})",
            "Session Type": result["session_type"],
            "Stress %": f"{d['stress_pct']:.1f}%",
            "Calm %": f"{d['calm_pct']:.1f}%",
            "Switches": result["transitions"]
        })
    comp_df = pd.DataFrame(comp_data)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Overlay stress intensity plot with model checkboxes
    if len(all_dfs) > 1:
        st.markdown("### 📈 Stress Intensity Overlay")

        # Create a horizontal layout of checkboxes
        model_names = list(all_dfs.keys())
        cols = st.columns(len(model_names))
        selected_models = []
        for i, model_name in enumerate(model_names):
            with cols[i]:
                # Each checkbox returns True if checked
                if st.checkbox(model_name, value=True, key=f"chk_{model_name}"):
                    selected_models.append(model_name)

        if selected_models:
            # Define a color palette (Plotly default + extra)
            colors = px.colors.qualitative.Plotly + px.colors.qualitative.Light24
            traces = []
            for idx, model_name in enumerate(selected_models):
                df_proc = _build_intensity_series(all_dfs[model_name], cfg["step_size"], cfg["fs"])
                traces.append(go.Scatter(
                    x=df_proc["Time_Minutes"],
                    y=df_proc["Stress_Intensity"],
                    mode='lines',
                    name=model_name,
                    line=dict(color=colors[idx % len(colors)], width=2)
                ))

            # Add threshold line
            max_time = df_proc["Time_Minutes"].max()
            traces.append(go.Scatter(
                x=[0, max_time],
                y=[50, 50],
                mode='lines',
                name='Threshold (50%)',
                line=dict(color='gray', dash='dash', width=1)
            ))

            fig_overlay = go.Figure(data=traces)
            fig_overlay.update_layout(
                title="Stress Intensity – Selected Models",
                xaxis_title="Time (Minutes)",
                yaxis_title="Stress Probability (%)",
                yaxis=dict(range=[0, 105]),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                hovermode='x unified',
                margin=dict(t=50, b=0, l=0, r=0),
            )
            st.plotly_chart(fig_overlay, use_container_width=True)
        else:
            st.info("Select at least one model to display the overlay plot.")
    else:
        # If only one model, just show its intensity plot
        st.markdown("### 📈 Stress Intensity (only one model selected)")
        single_model = list(all_dfs.keys())[0]
        df_proc = _build_intensity_series(all_dfs[single_model], cfg["step_size"], cfg["fs"])
        fig_single = go.Figure()
        fig_single.add_trace(go.Scatter(
            x=df_proc["Time_Minutes"],
            y=df_proc["Stress_Intensity"],
            mode='lines',
            name=single_model,
            line=dict(color='#E74C3C', width=2),
            fill='tozeroy',
            fillcolor='rgba(231,76,60,0.2)'
        ))
        fig_single.add_hline(y=50, line_dash="dash", line_color="#F39C12")
        fig_single.update_layout(
            xaxis_title="Time (Minutes)",
            yaxis_title="Stress Probability (%)",
            yaxis=dict(range=[0,105]),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig_single, use_container_width=True)

# -------------------------------------------------------------------
# DOWNLOAD BUTTON (all results)
# -------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
combined_results = {
    "models": all_results,
    "summary": {
        "num_models": len(all_results),
        "files_processed": uploaded_file.name,
    }
}
st.download_button(
    label="📥 Download All Results (JSON)",
    data=json.dumps(combined_results, indent=2, default=str),
    file_name=f"results_{Path(uploaded_file.name).stem}.json",
    mime="application/json",
)

# -------------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------------
st.markdown("""
<div class="footer">
    🧠 EEG Stress/Calm Analyzer · Built with Streamlit · v3.0
</div>
""", unsafe_allow_html=True)
