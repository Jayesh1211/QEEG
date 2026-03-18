"""
PHASE 2.1: SINGLE-MODEL INFERENCE PIPELINE
=========================================================
Workflow:
  1. Accept a raw EEG CSV file path (CLI arg or prompted)
  2. Preprocess: bandpass filter → global scaler → CNN extractor
  3. Interactively list available .pkl models and let user pick ONE
  4. Run inference (with optional QSVM downsampling) and display rich session analysis:
       - Session type  (CALM / STRESS / MIXED)
       - Duration stats
       - Detected segments with timestamps
       - Transition count
       - ASCII timeline strip
       - Per-segment consistency
  5. Optionally generate a graphical timeline (--plot)

Usage
-----
  python phase2_1_inference.py                         # fully interactive
  python phase2_1_inference.py --csv session.csv       # CSV pre-supplied
  python phase2_1_inference.py --csv session.csv --plot # + graphical timeline
  python phase2_1_inference.py --csv session.csv \
      --model_dir trained_api_models                   # custom artifact dir
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

import tensorflow as tf

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Import all model classes so joblib can deserialise .pkl files
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
    pass   # classes already in scope


# ============================================================================
# 1. ARGUMENT PARSING
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2.1 — Single-Model EEG Stress/Calm Inference")
    parser.add_argument(
        "--csv", default=None,
        help="Path to raw EEG CSV file. Prompted interactively if omitted.")
    parser.add_argument(
        "--model_dir", default="trained_api_models",
        help="Folder containing Phase 1 artifacts (default: trained_api_models).")
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate a graphical timeline (stress intensity plot).")
    return parser.parse_args()


# ============================================================================
# 2. LOAD CONFIG & VALIDATE ARTIFACT DIR
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
# 4. CSV LOADING & VALIDATION
# ============================================================================
def load_csv(csv_path: str, selected_channels: list) -> np.ndarray:
    if not os.path.exists(csv_path):
        print(f"\n[ERROR] File not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    missing = [ch for ch in selected_channels if ch not in df.columns]
    if missing:
        print(f"\n[ERROR] CSV is missing required channel(s): {missing}")
        print(f"        Required: {selected_channels}")
        print(f"        Found   : {list(df.columns)}")
        sys.exit(1)

    raw = df[selected_channels].values.astype(np.float32)
    print(f"  CSV loaded             ✓  {raw.shape[0]} samples  "
          f"({raw.shape[0] / 250:.1f} s)  ×  {raw.shape[1]} channels")
    return raw


# ============================================================================
# 5. PREPROCESSING CHAIN
# ============================================================================
def bandpass_filter(data: np.ndarray, fs: int,
                    lowcut: float, highcut: float, order: int) -> np.ndarray:
    nyq  = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data, axis=0).astype(np.float32)


def sliding_windows(data: np.ndarray,
                    window_size: int, step_size: int) -> np.ndarray:
    wins = []
    for i in range(0, len(data) - window_size + 1, step_size):
        wins.append(data[i: i + window_size])
    return np.array(wins, dtype=np.float32)


def preprocess(raw: np.ndarray, cfg: dict, scaler, cnn) -> tuple:
    """
    raw → bandpass → sliding windows → global scale → CNN → features_8D

    Returns
    -------
    windows_scaled : (n_windows, window_size, n_channels)
    features_8D    : (n_windows, 8)
    """
    fs          = cfg["fs"]
    window_size = cfg["window_size"]
    step_size   = cfg["step_size"]
    n_channels  = cfg["n_channels"]

    filtered = bandpass_filter(
        raw, fs,
        cfg["bandpass_lowcut"],
        cfg["bandpass_highcut"],
        cfg["bandpass_order"])
    print(f"  Bandpass filter        ✓  ({cfg['bandpass_lowcut']}–"
          f"{cfg['bandpass_highcut']} Hz, order {cfg['bandpass_order']})")

    windows = sliding_windows(filtered, window_size, step_size)
    if len(windows) == 0:
        print(f"\n[ERROR] Recording too short. Need >= {window_size} samples "
              f"({window_size/fs:.2f} s), got {len(raw)}.")
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
# 6. MODEL SELECTION — interactive numbered menu
# ============================================================================
def select_model(model_dir: str, cfg: dict) -> tuple:
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

    while True:
        try:
            choice = int(input(f"\n  Select a model (1-{len(names)}): ").strip())
            if 1 <= choice <= len(names):
                break
            print(f"  Please enter a number between 1 and {len(names)}.")
        except ValueError:
            print("  Invalid input — enter a number.")

    chosen_name = names[choice - 1]
    chosen_path = available[chosen_name]

    print(f"\n  Loading '{chosen_name}' ...")
    clf = joblib.load(chosen_path)
    print(f"  Model loaded           ✓  {chosen_name}")

    return chosen_name, clf


# ============================================================================
# 7. SMART PREDICTION (with optional downsampling for slow models)
# ============================================================================
def smart_predict(clf, features_8D: np.ndarray, model_name: str,
                  slow_models: dict) -> np.ndarray:
    """
    If model_name is in slow_models, run inference only every `rate` windows
    and forward-fill the predictions. Otherwise, run on all windows.
    """
    if model_name in slow_models:
        rate = slow_models[model_name]
        n = len(features_8D)
        # indices to actually evaluate
        indices = list(range(0, n, rate))
        features_subset = features_8D[indices]
        print(f"  [Slow model] Downsampling inference: evaluating "
              f"{len(indices)}/{n} windows (rate={rate})")
        sparse_preds = clf.predict(features_subset)

        # forward fill to full length
        full_preds = np.empty(n, dtype=int)
        for i, idx in enumerate(indices):
            end = indices[i+1] if i+1 < len(indices) else n
            full_preds[idx:end] = sparse_preds[i]
        return full_preds
    else:
        return clf.predict(features_8D)


# ============================================================================
# 8. ANALYSIS HELPERS (unchanged from original)
# ============================================================================
def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def detect_segments(labels: list, step_size: int, fs: int,
                    raw_labels: list = None) -> list:
    if not labels:
        return []

    sps      = step_size / fs
    segments = []
    seg_start_idx = 0
    seg_label     = labels[0]

    def _close(start_idx, end_idx, label):
        source = raw_labels if raw_labels is not None else labels
        # seg_labels  = labels[start_idx: end_idx + 1]
        seg_labels  = source[start_idx: end_idx + 1]
        n_win       = len(seg_labels)
        consistency = seg_labels.count(label) / n_win
        start_s     = start_idx * sps
        end_s       = (end_idx + 1) * sps
        return {
            "state":       label,
            "start_s":     round(start_s, 1),
            "end_s":       round(end_s, 1),
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


def compute_duration_stats(labels: list, step_size: int, fs: int) -> dict:
    sps       = step_size / fs
    total_s   = len(labels) * sps
    calm_s    = labels.count("calm")   * sps
    stress_s  = labels.count("stress") * sps
    return {
        "total_s":    round(total_s,  1),
        "calm_s":     round(calm_s,   1),
        "stress_s":   round(stress_s, 1),
        "calm_pct":   round(calm_s   / total_s * 100, 1),
        "stress_pct": round(stress_s / total_s * 100, 1),
    }


def classify_session(calm_pct: float, stress_pct: float,
                     mixed_threshold: float = 30.0) -> str:
    if stress_pct < mixed_threshold:
        return "CALM"
    if calm_pct < mixed_threshold:
        return "STRESS"
    return "MIXED"


def find_longest_block(segments: list, state: str):
    matches = [s for s in segments if s["state"] == state]
    return max(matches, key=lambda s: s["duration_s"]) if matches else None


def find_peak_seg(segments: list, state: str):
    matches = [s for s in segments if s["state"] == state]
    return max(matches, key=lambda s: s["consistency"]) if matches else None


def smooth_labels(labels: list, window: int = 20) -> list:
    half = window // 2
    smoothed = []
    for i in range(len(labels)):
        start = max(0, i - half)
        end   = min(len(labels), i + half)
        chunk = labels[start:end]
        smoothed.append("calm" if chunk.count("calm") >= len(chunk) / 2 else "stress")
    return smoothed


def merge_short_segments(segments: list, min_duration_s: float = 15.0) -> list:
    changed = True
    while changed:
        changed = False
        merged = []
        i = 0
        while i < len(segments):
            seg = segments[i]
            if seg["duration_s"] < min_duration_s and len(segments) > 1:
                # Merge with previous if same state
                if merged and merged[-1]["state"] == seg["state"]:
                    prev = merged[-1]
                    prev["end_s"]      = seg["end_s"]
                    prev["duration_s"] = round(prev["end_s"] - prev["start_s"], 1)
                    prev["n_windows"] += seg["n_windows"]
                    changed = True
                # Otherwise merge with next (by skipping)
                elif i + 1 < len(segments):
                    next_seg = segments[i + 1]
                    next_seg["start_s"]    = seg["start_s"]
                    next_seg["duration_s"] = round(next_seg["end_s"] - next_seg["start_s"], 1)
                    next_seg["n_windows"] += seg["n_windows"]
                    i += 1
                    changed = True
                else:
                    merged.append(seg)
            else:
                merged.append(seg)
            i += 1
        segments = merged
    return segments


# ============================================================================
# 9. INFERENCE (modified to use smart_predict)
# ============================================================================
def run_inference(model_name: str, clf, features_8D: np.ndarray,
                  cfg: dict) -> tuple:
    """
    Returns
    -------
    result : dict   – summary dictionary (same as before)
    df     : DataFrame – per-window predictions (for plotting)
    """
    label_map_inv  = {int(k): v for k, v in cfg["label_map_inverse"].items()}
    step_size      = cfg["step_size"]
    fs             = cfg["fs"]
    slow_models    = cfg.get("slow_models", {})   # NEW: load slow model config

    # NEW: use smart_predict to handle slow models (e.g., QSVM)
    raw_preds = smart_predict(clf, features_8D, model_name, slow_models)
    raw_labels    = [label_map_inv[int(p)] for p in raw_preds]

    # labels        = smooth_labels(raw_labels, window=20)
    # Only apply temporal smoothing on full-resolution models
    if model_name in cfg.get("slow_models", {}):
        labels = raw_labels          # forward-fill already provides smoothing
    else:
        labels = smooth_labels(raw_labels, window=30)   # also bump to 30 here

    duration     = compute_duration_stats(labels, step_size, fs)
    # segments     = detect_segments(labels, step_size, fs)
    segments     = detect_segments(labels, step_size, fs, raw_labels=raw_labels)
    segments     = merge_short_segments(segments, min_duration_s=15)
    session_type = classify_session(duration["calm_pct"], duration["stress_pct"])
    dominant     = ("calm"
                    if duration["calm_pct"] >= duration["stress_pct"]
                    else "stress")
    transitions  = len(segments) - 1

    result = {
        "model_used":        model_name,
        "is_quantum_model":  model_name in cfg.get("quantum_models", []),
        "windows_processed": len(labels),
        "window_labels":     labels,
        "session_type":      session_type,
        "dominant_state":    dominant,
        "duration":          duration,
        "segments":          segments,
        "transitions":       transitions,
        "longest_calm_s":    (find_longest_block(segments, "calm") or {}).get("duration_s", 0),
        "longest_stress_s":  (find_longest_block(segments, "stress") or {}).get("duration_s", 0),
        "peak_calm_seg":     find_longest_block(segments, "calm"),
        "peak_stress_seg":   find_longest_block(segments, "stress"),
    }

    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Window_Index': range(1, len(labels) + 1),
        'Model_Used': model_name,
        'Prediction': labels,
        'Raw_Prediction': raw_labels
    })

    display_result(result, step_size, fs)
    return result, df


# ============================================================================
# 10. DISPLAY (original, unchanged)
# ============================================================================
def display_result(r: dict, step_size: int = 125, fs: int = 250):
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

    print()
    print("  PEAK PERIODS  (longest uninterrupted block per state)")
    div()
    if r["peak_calm_seg"]:
        pc = r["peak_calm_seg"]     # now populated from find_longest_block, not find_peak_seg
        print(f"  Longest calm   :  {fmt_time(pc['start_s'])} → {fmt_time(pc['end_s'])}"
              f"   {pc['duration_s']}s")
    else:
        print("  Longest calm   :  — (no calm windows detected)")

    if r["peak_stress_seg"]:
        ps = r["peak_stress_seg"]
        print(f"  Longest stress :  {fmt_time(ps['start_s'])} → {fmt_time(ps['end_s'])}"
              f"   {ps['duration_s']}s")
    else:
        print("  Longest stress :  — (no stress windows detected)")

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

    print()
    print("  TIMELINE  ( · = calm     █ = stress )")
    div()

    n          = len(labels)
    strip      = ["·" if l == "calm" else "█" for l in labels]
    wins_per_10 = max(1, int(10 / sps))

    ruler = [" "] * n
    for w in range(0, n, wins_per_10):
        t_str = fmt_time(w * sps)
        for k, ch in enumerate(t_str):
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
# 11. PLOTTING FUNCTION (NEW)
# ============================================================================
def plot_timeline(df: pd.DataFrame, step_size: int, fs: int,
                  model_name: str, save: bool = True):
    """
    Generate a graphical timeline of stress intensity.
    """
    # Import matplotlib/seaborn only when needed
    import matplotlib.pyplot as plt
    import seaborn as sns

    time_step_sec = step_size / fs
    df['Time_Minutes'] = df['Window_Index'] * time_step_sec / 60.0
    # df['Binary_Stress'] = (df['Prediction'] == 'stress').astype(int)
    df['Binary_Stress'] = (df['Raw_Prediction'] == 'stress').astype(int)
    # Rolling average over 10 windows = 5 seconds (adjust as desired)
    # df['Stress_Intensity'] = df['Binary_Stress'].rolling(window=10, min_periods=1).mean() * 100
    df['Stress_Intensity'] = (
        df['Binary_Stress']
        .rolling(window=30, min_periods=1, center=True)   # 15s window, no lag
        .mean() * 100
    )

    plt.figure(figsize=(12, 4))
    sns.set_theme(style="whitegrid")
    plt.plot(df['Time_Minutes'], df['Stress_Intensity'],
             color='#d62728', linewidth=2, label='Stress Intensity (%)')
    plt.fill_between(df['Time_Minutes'], df['Stress_Intensity'],
                     where=(df['Stress_Intensity'] >= 50),
                     color='#d62728', alpha=0.3, label='High Stress Zone')

    plt.title(f"Continuous Cognitive State Analysis ({model_name})",
              fontsize=14, pad=15)
    plt.xlabel("Time (Minutes)", fontsize=12)
    plt.ylabel("Stress Probability (%)", fontsize=12)
    plt.ylim(-5, 105)
    plt.legend(loc='upper right')
    plt.tight_layout()

    if save:
        save_path = f"stress_timeline_{model_name}.png"
        plt.savefig(save_path, dpi=300)
        print(f"  Timeline graph saved as '{save_path}'")
    plt.show()


# ============================================================================
# 12. MAIN (updated to handle --plot)
# ============================================================================
def main():
    args = parse_args()

    print("\n" + "=" * 62)
    print("  PHASE 2.1 — SINGLE-MODEL EEG INFERENCE")
    print("=" * 62)

    if not os.path.isdir(args.model_dir):
        print(f"\n[ERROR] model_dir not found: '{args.model_dir}'")
        sys.exit(1)
    print(f"\n  Artifact dir: {args.model_dir}\n")

    cfg          = load_config(args.model_dir)
    scaler, cnn  = load_preprocessors(args.model_dir, cfg)

    csv_path = args.csv
    if not csv_path:
        print()
        csv_path = input("  Enter path to raw EEG CSV file: ").strip()

    print()
    raw = load_csv(csv_path, cfg["selected_channels"])

    print("\n  PREPROCESSING")
    print("─" * 62)
    _, features_8D = preprocess(raw, cfg, scaler, cnn)

    model_name, clf = select_model(args.model_dir, cfg)

    print("\n  RUNNING INFERENCE")
    print("─" * 62)
    result, df = run_inference(model_name, clf, features_8D, cfg)

    # NEW: generate plot if requested
    if args.plot:
        print("\n  GENERATING GRAPHICAL TIMELINE")
        print("─" * 62)
        plot_timeline(df, cfg["step_size"], cfg["fs"], model_name)

    # Optional JSON save
    save = input("\n  Save result to JSON? (y/N): ").strip().lower()
    if save == 'y':
        out_name = (
            f"result_{os.path.splitext(os.path.basename(csv_path))[0]}"
            f"_{model_name}.json")
        with open(out_name, "w") as f:
            # Remove window_labels from result before saving (optional)
            result_for_json = result.copy()
            result_for_json.pop("window_labels", None)
            json.dump(result_for_json, f, indent=2)
        print(f"  Result saved -> {out_name}\n")


if __name__ == "__main__":
    main()
