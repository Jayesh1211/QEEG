"""
PHASE 2.1: SINGLE-MODEL OR MULTI-MODEL INFERENCE PIPELINE
=========================================================
Workflow:
  1. Accept a raw EEG CSV file path (CLI arg or prompted)
  2. Preprocess: bandpass filter → global scaler → CNN extractor
  3. Interactively list available .pkl models → select ONE or MULTIPLE
  4. Run inference per model (QSVM downsampled if configured)
  5. Display rich session analysis per model:
       - Stress score /10
       - Session type  (CALM / STRESS / MIXED)
       - Duration stats + transitions
       - Detected segments with timestamps + consistency
       - ASCII timeline strip
  6. Optionally generate:
       - Individual stress intensity plot per model  (--plot)
       - Side-by-side comparison plot of all selected models (--plot)

Usage
-----
  python phase2_1_inference.py                          # fully interactive
  python phase2_1_inference.py --csv session.csv        # CSV pre-supplied
  python phase2_1_inference.py --csv session.csv --plot # + plots
  python phase2_1_inference.py --csv session.csv \
      --model_dir trained_api_models --plot
"""

import os
import sys
import json
import joblib
import warnings
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from scipy.signal import butter, filtfilt
from concurrent.futures import ProcessPoolExecutor, as_completed

import tensorflow as tf

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Register all model classes so joblib can deserialise .pkl files
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
except ImportError:
    pass   # classes already in scope (e.g. notebook)


# ============================================================================
# 1. ARGUMENT PARSING
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2.1 — Single / Multi-Model EEG Stress/Calm Inference")
    parser.add_argument(
        "--csv", default=None,
        help="Path to raw EEG CSV file. Prompted interactively if omitted.")
    parser.add_argument(
        "--model_dir", default="trained_api_models",
        help="Folder containing Phase 1 artifacts (default: trained_api_models).")
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate individual + comparison stress intensity plots.")
    return parser.parse_args()


# ============================================================================
# 2. CONFIG
# ============================================================================
def load_config(model_dir: str) -> dict:
    config_path = os.path.join(model_dir, "pipeline_config.json")
    if not os.path.exists(config_path):
        print(f"\n[ERROR] pipeline_config.json not found in '{model_dir}'.")
        print("        Make sure Phase 1 has been run and model_dir is correct.")
        sys.exit(1)
    with open(config_path) as f:
        cfg = json.load(f)
    print(f"  Config loaded          ✓")
    return cfg


# ============================================================================
# 3. LOAD PREPROCESSING ARTIFACTS
# ============================================================================
def load_preprocessors(model_dir: str, cfg: dict):
    scaler_path = os.path.join(model_dir, cfg["global_scaler_file"])
    if not os.path.exists(scaler_path):
        print(f"\n[ERROR] Global scaler not found: {scaler_path}")
        sys.exit(1)
    scaler = joblib.load(scaler_path)
    print(f"  Global scaler loaded   ✓  ({cfg['global_scaler_file']})")

    cnn_path = os.path.join(model_dir, cfg["cnn_extractor_file"])
    if not os.path.exists(cnn_path):
        print(f"\n[ERROR] CNN extractor not found: {cnn_path}")
        sys.exit(1)
    cnn = tf.keras.models.load_model(cnn_path)
    print(f"  CNN extractor loaded   ✓  ({cfg['cnn_extractor_file']})")

    return scaler, cnn


# ============================================================================
# 4. CSV LOADING
# ============================================================================
def load_csv(csv_path: str, selected_channels: list) -> np.ndarray:
    if not os.path.exists(csv_path):
        print(f"\n[ERROR] File not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    missing = [ch for ch in selected_channels if ch not in df.columns]
    if missing:
        print(f"\n[ERROR] CSV is missing required channel(s): {missing}")
        print(f"        Required : {selected_channels}")
        print(f"        Found    : {list(df.columns)}")
        sys.exit(1)

    raw = df[selected_channels].values.astype(np.float32)
    print(f"  CSV loaded             ✓  {raw.shape[0]} samples  "
          f"({raw.shape[0] / 250:.1f} s)  ×  {raw.shape[1]} channels")
    return raw


# ============================================================================
# 5. PREPROCESSING CHAIN
# ============================================================================
def bandpass_filter(data, fs, lowcut, highcut, order):
    nyq  = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data, axis=0).astype(np.float32)


def sliding_windows(data, window_size, step_size):
    wins = []
    for i in range(0, len(data) - window_size + 1, step_size):
        wins.append(data[i: i + window_size])
    return np.array(wins, dtype=np.float32)


def preprocess(raw, cfg, scaler, cnn):
    fs, window_size = cfg["fs"], cfg["window_size"]
    step_size, n_channels = cfg["step_size"], cfg["n_channels"]

    filtered = bandpass_filter(raw, fs, cfg["bandpass_lowcut"],
                               cfg["bandpass_highcut"], cfg["bandpass_order"])
    print(f"  Bandpass filter        ✓  ({cfg['bandpass_lowcut']}–"
          f"{cfg['bandpass_highcut']} Hz, order {cfg['bandpass_order']})")

    windows = sliding_windows(filtered, window_size, step_size)
    if len(windows) == 0:
        print(f"\n[ERROR] Recording too short.")
        sys.exit(1)
    print(f"  Windowing              ✓  {len(windows)} window(s)  "
          f"(size={window_size}, step={step_size})")

    n = len(windows)
    windows_scaled = scaler.transform(
        windows.reshape(-1, n_channels)
    ).reshape(n, window_size, n_channels)
    print(f"  Global scaling         ✓  (StandardScaler per-channel)")

    features_8D = cnn.predict(windows_scaled, verbose=0)
    print(f"  CNN feature extraction ✓  {features_8D.shape}  (8-D per window)")

    return windows_scaled, features_8D


# ============================================================================
# 6. MODEL SELECTION — multi-select menu
# ============================================================================
def select_models(model_dir: str, cfg: dict) -> dict:
    """
    Print numbered menu, let user pick ONE or MULTIPLE models.
    Returns { model_name: full_path }
    """
    model_files    = cfg.get("model_files", {})
    quantum_models = set(cfg.get("quantum_models", []))

    available = {
        name: os.path.join(model_dir, fname)
        for name, fname in model_files.items()
        if os.path.exists(os.path.join(model_dir, fname))
    }

    if not available:
        print(f"\n[ERROR] No .pkl model files found in '{model_dir}'.")
        sys.exit(1)

    names = list(available.keys())

    print("\n" + "─" * 62)
    print("  AVAILABLE MODELS")
    print("─" * 62)
    for idx, name in enumerate(names, start=1):
        tag = "  [quantum]" if name in quantum_models else ""
        print(f"  [{idx:>2}]  {name}{tag}")
    print("─" * 62)
    print("  Enter one number OR multiple comma-separated (e.g. 1,3,4)")
    print("  Enter 0 to run ALL models")
    print("─" * 62)

    while True:
        raw = input(f"\n  Select model(s): ").strip()
        try:
            if raw == "0":
                chosen_indices = list(range(1, len(names) + 1))
            else:
                chosen_indices = [int(x.strip()) for x in raw.split(",")]

            if all(1 <= i <= len(names) for i in chosen_indices):
                break
            print(f"  All numbers must be between 1 and {len(names)}.")
        except ValueError:
            print("  Invalid input. Enter numbers separated by commas.")

    # Deduplicate while preserving order
    seen = set()
    chosen_names = []
    for i in chosen_indices:
        name = names[i - 1]
        if name not in seen:
            chosen_names.append(name)
            seen.add(name)

    print(f"\n  Selected : {', '.join(chosen_names)}")
    return {name: available[name] for name in chosen_names}


# ============================================================================
# 7. SMART PREDICT (with QSVM downsampling)
# ============================================================================
def smart_predict(clf, features_8D, model_name, slow_models):
    if model_name in slow_models:
        rate = slow_models[model_name]
        n    = len(features_8D)
        indices      = list(range(0, n, rate))
        sparse_preds = clf.predict(features_8D[indices])

        full_preds = np.empty(n, dtype=int)
        for i, idx in enumerate(indices):
            end = indices[i + 1] if i + 1 < len(indices) else n
            full_preds[idx:end] = sparse_preds[i]

        print(f"  [Slow model] Evaluated {len(indices)}/{n} windows "
              f"(rate={rate}), forward-filled the rest")
        return full_preds
    else:
        return clf.predict(features_8D)


# ============================================================================
# 8. ANALYSIS HELPERS
# ============================================================================
def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def smooth_labels(labels, window=30):
    half     = window // 2
    smoothed = []
    for i in range(len(labels)):
        chunk = labels[max(0, i - half): min(len(labels), i + half)]
        smoothed.append("calm" if chunk.count("calm") >= len(chunk) / 2
                        else "stress")
    return smoothed


def detect_segments(labels, step_size, fs, raw_labels=None):
    if not labels:
        return []
    sps           = step_size / fs
    source        = raw_labels if raw_labels is not None else labels
    segments      = []
    seg_start_idx = 0
    seg_label     = labels[0]

    def _close(start_idx, end_idx, label):
        seg_source  = source[start_idx: end_idx + 1]
        n_win       = len(seg_source)
        consistency = seg_source.count(label) / n_win
        start_s     = start_idx * sps
        end_s       = (end_idx + 1) * sps
        return {
            "state":       label,
            "start_s":     round(start_s, 1),
            "end_s":       round(end_s,   1),
            "duration_s":  round(end_s - start_s, 1),
            "n_windows":   n_win,
            "consistency": round(consistency, 4),
        }

    for i in range(1, len(labels)):
        if labels[i] != seg_label:
            segments.append(_close(seg_start_idx, i - 1, seg_label))
            seg_start_idx = i
            seg_label     = labels[i]
    segments.append(_close(seg_start_idx, len(labels) - 1, seg_label))
    return segments


def merge_short_segments(segments, min_duration_s=15.0):
    if not segments:
        return segments

    changed = True
    while changed:
        changed = False
        if len(segments) == 1:
            break

        # Find the shortest segment below threshold
        short_idx = None
        shortest  = float('inf')
        for i, seg in enumerate(segments):
            if seg["duration_s"] < min_duration_s and seg["duration_s"] < shortest:
                shortest  = seg["duration_s"]
                short_idx = i

        if short_idx is None:
            break

        has_prev = short_idx > 0
        has_next = short_idx < len(segments) - 1

        if has_prev and has_next:
            prev_dur    = segments[short_idx - 1]["duration_s"]
            next_dur    = segments[short_idx + 1]["duration_s"]
            absorb_into = short_idx - 1 if prev_dur >= next_dur else short_idx + 1
        elif has_prev:
            absorb_into = short_idx - 1
        else:
            absorb_into = short_idx + 1

        target = segments[absorb_into]
        short  = segments[short_idx]

        new_start            = min(target["start_s"], short["start_s"])
        new_end              = max(target["end_s"],   short["end_s"])
        target["start_s"]    = new_start
        target["end_s"]      = new_end
        target["duration_s"] = round(new_end - new_start, 1)
        target["n_windows"] += short["n_windows"]

        segments = [s for i, s in enumerate(segments) if i != short_idx]
        changed  = True

    return segments


def compute_duration_stats(labels, step_size, fs):
    sps      = step_size / fs
    total_s  = len(labels) * sps
    calm_s   = labels.count("calm")   * sps
    stress_s = labels.count("stress") * sps
    return {
        "total_s":    round(total_s,  1),
        "calm_s":     round(calm_s,   1),
        "stress_s":   round(stress_s, 1),
        "calm_pct":   round(calm_s   / total_s * 100, 1),
        "stress_pct": round(stress_s / total_s * 100, 1),
    }


def classify_session(calm_pct, stress_pct, mixed_threshold=30.0):
    if stress_pct < mixed_threshold:
        return "CALM"
    if calm_pct < mixed_threshold:
        return "STRESS"
    return "MIXED"


def find_longest_block(segments, state):
    matches = [s for s in segments if s["state"] == state]
    return max(matches, key=lambda s: s["duration_s"]) if matches else None

# ============================================================================
# 9. STRESS SCORE  (Component 1 only: stress burden)
# ============================================================================
def compute_stress_score(duration: dict) -> dict:
    """
    Score = stress_pct / 10  →  0–10
    Simple, honest, single-component for now.
    """
    score = round(duration["stress_pct"] / 10, 1)
    score = min(score, 10.0)

    if score <= 3.0:
        label  = "LOW"
        advice = "Relaxed session. No stress concerns."
    elif score <= 5.5:
        label  = "MODERATE"
        advice = "Mild stress present. Generally manageable."
    elif score <= 7.5:
        label  = "HIGH"
        advice = "Significant stress detected. Consider a break."
    else:
        label  = "SEVERE"
        advice = "Very high stress load. Immediate rest advised."

    return {"score": score, "label": label, "advice": advice}


# ============================================================================
# 10. INFERENCE (single model)
# ============================================================================
def run_inference(model_name, clf, features_8D, cfg):
    label_map_inv = {int(k): v for k, v in cfg["label_map_inverse"].items()}
    step_size     = cfg["step_size"]
    fs            = cfg["fs"]
    slow_models   = cfg.get("slow_models", {})

    raw_preds  = smart_predict(clf, features_8D, model_name, slow_models)
    raw_labels = [label_map_inv[int(p)] for p in raw_preds]

    # Skip smoothing for downsampled models (forward-fill already acts as smoothing)
    if model_name in slow_models:
        labels = raw_labels
    else:
        labels = smooth_labels(raw_labels, window=30)

    duration     = compute_duration_stats(labels, step_size, fs)
    segments     = detect_segments(labels, step_size, fs, raw_labels=raw_labels)
    segments     = merge_short_segments(segments, min_duration_s=15)
    session_type = classify_session(duration["calm_pct"], duration["stress_pct"])
    dominant     = ("calm" if duration["calm_pct"] >= duration["stress_pct"]
                    else "stress")
    stress_score = compute_stress_score(duration)

    result = {
        "model_used":        model_name,
        "is_quantum_model":  model_name in cfg.get("quantum_models", []),
        "windows_processed": len(labels),
        "window_labels":     labels,
        "session_type":      session_type,
        "dominant_state":    dominant,
        "duration":          duration,
        "segments":          segments,
        "transitions":       len(segments) - 1,
        "longest_calm_s":    (find_longest_block(segments, "calm")   or {}).get("duration_s", 0),
        "longest_stress_s":  (find_longest_block(segments, "stress") or {}).get("duration_s", 0),
        "peak_calm_seg":     find_longest_block(segments, "calm"),
        "peak_stress_seg":   find_longest_block(segments, "stress"),
        "stress_score":      stress_score,
    }

    # DataFrame for plotting — uses raw_labels for the intensity curve
    df = pd.DataFrame({
        "Window_Index":  range(1, len(labels) + 1),
        "Model_Used":    model_name,
        "Prediction":    labels,        # smoothed — used for segments
        "Raw_Prediction": raw_labels,   # raw     — used for plot curve
    })

    return result, df


# ============================================================================
# 11. DISPLAY (single model)
# ============================================================================
def display_stress_score(ss: dict):
    score  = ss["score"]
    filled = int(score)
    empty  = 10 - filled
    bar    = "█" * filled + "░" * empty
    print(f"  [{bar}]  {score}/10  —  {ss['label']}")
    print(f"  {ss['advice']}")


def display_result(r, step_size=125, fs=250):
    d      = r["duration"]
    segs   = r["segments"]
    labels = r["window_labels"]
    sps    = step_size / fs
    W      = 60

    def div(char="─"):
        print("  " + char * W)

    print()
    div("═")
    print("  SESSION ANALYSIS REPORT")
    div("═")
    print(f"  Model            :  {r['model_used']}"
          + ("  (quantum)" if r["is_quantum_model"] else ""))
    print(f"  Windows analysed :  {r['windows_processed']}")
    print(f"  Session duration :  {fmt_time(d['total_s'])}  ({d['total_s']}s)")

    # ── Stress score ──────────────────────────────────────────────────────────
    print()
    print("  STRESS SCORE")
    div()
    display_stress_score(r["stress_score"])

    # ── Session type ──────────────────────────────────────────────────────────
    stype = r["session_type"]
    print()
    div("═")
    if stype == "CALM":
        print("  SESSION TYPE  :  [ CALM ]")
    elif stype == "STRESS":
        print("  SESSION TYPE  :  [ STRESS ]")
    else:
        dom_pct = d[r["dominant_state"] + "_pct"]
        print(f"  SESSION TYPE  :  [ MIXED ]  —  both states detected")
        print(f"  Dominant state:  {r['dominant_state'].upper()}"
              f"  ({dom_pct:.1f}% of session)")
    div("═")

    # ── Duration breakdown ────────────────────────────────────────────────────
    print()
    print("  DURATION BREAKDOWN")
    div()
    calm_bar   = "·" * int(d["calm_pct"]   / 2)
    stress_bar = "█" * int(d["stress_pct"] / 2)
    print(f"  Calm    {d['calm_pct']:5.1f}%   {fmt_time(d['calm_s']):>6}   {calm_bar}")
    print(f"  Stress  {d['stress_pct']:5.1f}%   {fmt_time(d['stress_s']):>6}   {stress_bar}")
    print()
    print(f"  State switches       :  {r['transitions']}")
    if r["longest_calm_s"]:
        print(f"  Longest calm block   :  {fmt_time(r['longest_calm_s'])}"
              f"  ({r['longest_calm_s']}s)")
    if r["longest_stress_s"]:
        print(f"  Longest stress block :  {fmt_time(r['longest_stress_s'])}"
              f"  ({r['longest_stress_s']}s)")

    # ── Peak periods ──────────────────────────────────────────────────────────
    print()
    print("  PEAK PERIODS  (longest uninterrupted block per state)")
    div()
    if r["peak_calm_seg"]:
        pc = r["peak_calm_seg"]
        print(f"  Longest calm   :  {fmt_time(pc['start_s'])} → "
              f"{fmt_time(pc['end_s'])}   {pc['duration_s']}s")
    else:
        print("  Longest calm   :  —")
    if r["peak_stress_seg"]:
        ps = r["peak_stress_seg"]
        print(f"  Longest stress :  {fmt_time(ps['start_s'])} → "
              f"{fmt_time(ps['end_s'])}   {ps['duration_s']}s")
    else:
        print("  Longest stress :  —")

    # ── Segment table ─────────────────────────────────────────────────────────
    print()
    print("  DETECTED SEGMENTS")
    div()
    print(f"  {'#':<4}  {'Start':>6}  {'End':>6}  {'Duration':>8}  "
          f"{'State':<8}  Consistency")
    div()
    for i, seg in enumerate(segs, start=1):
        state_str = "CALM  " if seg["state"] == "calm" else "STRESS"
        bar       = "█" * int(seg["consistency"] * 20)
        print(f"  {i:<4}  {fmt_time(seg['start_s']):>6}"
              f"  {fmt_time(seg['end_s']):>6}"
              f"  {fmt_time(seg['duration_s']):>8}"
              f"  {state_str}"
              f"  {seg['consistency']*100:5.1f}%  {bar}")

    # ── ASCII timeline ────────────────────────────────────────────────────────
    print()
    print("  TIMELINE  ( · = calm     █ = stress )")
    div()
    n           = len(labels)
    strip       = ["·" if l == "calm" else "█" for l in labels]
    wins_per_10 = max(1, int(10 / sps))
    ruler       = [" "] * n
    for w in range(0, n, wins_per_10):
        for k, ch in enumerate(fmt_time(w * sps)):
            if w + k < n:
                ruler[w + k] = ch

    for row_start in range(0, n, W):
        row_end = min(row_start + W, n)
        print("  " + "".join(strip[row_start:row_end]))
        print("  " + "".join(ruler[row_start:row_end]))
        print()

    div("═")
    print()


# ============================================================================
# 12. PLOTTING
# ============================================================================
def _build_intensity_series(df, step_size, fs):
    """Shared helper — builds Time_Minutes and Stress_Intensity columns."""
    df = df.copy()
    df["Time_Minutes"]   = df["Window_Index"] * (step_size / fs) / 60.0
    df["Binary_Stress"]  = (df["Raw_Prediction"] == "stress").astype(int)
    df["Stress_Intensity"] = (
        df["Binary_Stress"]
        .rolling(window=30, min_periods=1, center=True)
        .mean() * 100
    )
    return df


def plot_individual(df, step_size, fs, model_name, save=True):
    """One stress intensity plot for a single model."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = _build_intensity_series(df, step_size, fs)

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.set_theme(style="whitegrid")

    ax.plot(df["Time_Minutes"], df["Stress_Intensity"],
            color="#d62728", linewidth=2, label="Stress Intensity (%)")
    ax.fill_between(df["Time_Minutes"], df["Stress_Intensity"],
                    where=(df["Stress_Intensity"] >= 50),
                    color="#d62728", alpha=0.3, label="High Stress Zone")
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_title(f"Stress Intensity — {model_name}", fontsize=14, pad=12)
    ax.set_xlabel("Time (Minutes)", fontsize=12)
    ax.set_ylabel("Stress Probability (%)", fontsize=12)
    ax.set_ylim(-5, 105)
    ax.legend(loc="upper right")
    plt.tight_layout()

    if save:
        path = f"stress_timeline_{model_name}.png"
        plt.savefig(path, dpi=300)
        print(f"  Saved: {path}")
    plt.show()
    plt.close()


def plot_comparison(all_dfs: dict, step_size, fs, save=True):
    """
    One figure with one subplot per model — all on the same time axis
    so curves are directly comparable.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    model_names = list(all_dfs.keys())
    n_models    = len(model_names)

    # Colour palette — one colour per model
    palette = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e",
               "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    fig, axes = plt.subplots(
        n_models, 1,
        figsize=(14, 3.5 * n_models),
        sharex=True)

    # If only one model selected, axes is not a list
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, df), color in zip(axes, all_dfs.items(), palette):
        df = _build_intensity_series(df, step_size, fs)

        ax.plot(df["Time_Minutes"], df["Stress_Intensity"],
                color=color, linewidth=2, label=model_name)
        ax.fill_between(df["Time_Minutes"], df["Stress_Intensity"],
                        where=(df["Stress_Intensity"] >= 50),
                        color=color, alpha=0.2)
        ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_ylim(-5, 105)
        ax.set_ylabel("Stress %", fontsize=10)
        ax.legend(loc="upper right", fontsize=10)
        ax.set_title(model_name, fontsize=11)

    axes[-1].set_xlabel("Time (Minutes)", fontsize=12)
    fig.suptitle("Model Comparison — Stress Intensity Over Time",
                 fontsize=14, y=1.01)
    plt.tight_layout()

    if save:
        path = "stress_comparison_" + "_vs_".join(model_names) + ".png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.show()
    plt.close()

def plot_overlay(all_dfs: dict, step_size, fs, save=True):
    """
    All selected models' stress intensity curves on a SINGLE set of axes.
    Good for direct visual comparison of where models agree/disagree.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    palette = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e",
               "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    fig, ax = plt.subplots(figsize=(14, 5))

    for (model_name, df), color in zip(all_dfs.items(), palette):
        df = _build_intensity_series(df, step_size, fs)
        ax.plot(df["Time_Minutes"], df["Stress_Intensity"],
                color=color, linewidth=2, label=model_name, alpha=0.85)

    # 50% threshold line
    ax.axhline(50, color="gray", linestyle="--",
               linewidth=1.0, alpha=0.6, label="50% threshold")

    # Shade the region where ALL models agree on high stress
    # (min of all curves >= 50 means every model says stress)
    all_series = []
    for df in all_dfs.values():
        df = _build_intensity_series(df, step_size, fs)
        all_series.append(df.set_index("Time_Minutes")["Stress_Intensity"])

    combined    = pd.concat(all_series, axis=1)
    combined.columns = list(all_dfs.keys())
    time_index  = combined.index
    min_curve   = combined.min(axis=1)   # all models agree on stress
    max_curve   = combined.max(axis=1)   # at least one model says stress

    # Light grey band = models disagree (spread between min and max)
    ax.fill_between(time_index, min_curve, max_curve,
                    alpha=0.10, color="gray", label="Model disagreement zone")

    # Red band = all models agree on high stress
    ax.fill_between(time_index, min_curve, 50,
                    where=(min_curve >= 50),
                    alpha=0.20, color="#d62728", label="All models: high stress")

    ax.set_title("Stress Intensity — All Models Overlay", fontsize=14, pad=12)
    ax.set_xlabel("Time (Minutes)", fontsize=12)
    ax.set_ylabel("Stress Probability (%)", fontsize=12)
    ax.set_ylim(-5, 105)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    plt.tight_layout()

    if save:
        path = "stress_overlay_all_models.png"
        plt.savefig(path, dpi=300)
        print(f"  Saved: {path}")
    plt.show()
    plt.close()


# ============================================================================
# 13. COMPARISON SUMMARY TABLE (terminal)
# ============================================================================
def display_comparison_table(all_results: dict):
    """Print a compact side-by-side summary of all selected models."""
    print("\n" + "═" * 72)
    print("  MODEL COMPARISON SUMMARY")
    print("═" * 72)
    print(f"  {'Model':<22}  {'Score':>6}  {'Label':<8}  "
          f"{'Session':<8}  {'Stress%':>7}  {'Switches':>8}")
    print("  " + "─" * 68)
    for name, r in all_results.items():
        ss   = r["stress_score"]
        d    = r["duration"]
        bar  = "█" * int(ss["score"]) + "░" * (10 - int(ss["score"]))
        print(f"  {name:<22}  {ss['score']:>5.1f}  "
              f"[{bar}]  "
              f"{ss['label']:<8}  "
              f"{r['session_type']:<8}  "
              f"{d['stress_pct']:>6.1f}%  "
              f"{r['transitions']:>8}")
    print("═" * 72 + "\n")


# ============================================================================
# 14. MAIN
# ============================================================================
def main():
    args = parse_args()

    print("\n" + "=" * 62)
    print("  PHASE 2.1 — MULTI-MODEL EEG INFERENCE")
    print("=" * 62)

    if not os.path.isdir(args.model_dir):
        print(f"\n[ERROR] model_dir not found: '{args.model_dir}'")
        sys.exit(1)
    print(f"\n  Artifact dir: {args.model_dir}\n")

    cfg         = load_config(args.model_dir)
    scaler, cnn = load_preprocessors(args.model_dir, cfg)

    csv_path = args.csv
    if not csv_path:
        print()
        csv_path = input("  Enter path to raw EEG CSV file: ").strip()

    print()
    raw = load_csv(csv_path, cfg["selected_channels"])

    print("\n  PREPROCESSING")
    print("─" * 62)
    _, features_8D = preprocess(raw, cfg, scaler, cnn)

    # ── Multi-model selection ─────────────────────────────────────────────────
    selected_models = select_models(args.model_dir, cfg)

    # ── Run inference per model ───────────────────────────────────────────────
    all_results = {}
    all_dfs     = {}

    for model_name, model_path in selected_models.items():
        print(f"\n{'─'*62}")
        print(f"  RUNNING INFERENCE  —  {model_name}")
        print(f"{'─'*62}")

        clf = joblib.load(model_path)
        print(f"  Model loaded           ✓  {model_name}")

        result, df = run_inference(model_name, clf, features_8D, cfg)
        display_result(result, cfg["step_size"], cfg["fs"])

        all_results[model_name] = result
        all_dfs[model_name]     = df

    # ── Comparison table (always shown when >1 model) ─────────────────────────
    if len(selected_models) > 1:
        display_comparison_table(all_results)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if args.plot:
        print("\n  GENERATING PLOTS")
        print("─" * 62)
        for model_name, df in all_dfs.items():
            plot_individual(df, cfg["step_size"], cfg["fs"], model_name)
        if len(all_dfs) > 1:
            print("  Generating comparison plot...")
            plot_comparison(all_dfs, cfg["step_size"], cfg["fs"])
            print("  Generating overlay plot...")        # ← ADD
            plot_overlay(all_dfs, cfg["step_size"], cfg["fs"])   # ← ADD
            
    # ── Optional JSON save ────────────────────────────────────────────────────
    save = input("\n  Save results to JSON? (y/N): ").strip().lower()
    if save == "y":
        base = os.path.splitext(os.path.basename(csv_path))[0]
        for model_name, result in all_results.items():
            out_name = f"result_{base}_{model_name}.json"
            to_save  = {k: v for k, v in result.items()
                        if k != "window_labels"}
            with open(out_name, "w") as f:
                json.dump(to_save, f, indent=2)
            print(f"  Saved -> {out_name}")


if __name__ == "__main__":
    main()