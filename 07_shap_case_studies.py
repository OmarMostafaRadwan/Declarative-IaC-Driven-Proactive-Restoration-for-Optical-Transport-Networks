"""
07_shap_case_studies.py
=======================
Phase 3, Step 1 — Move beyond generic feature importance to per-sample,
per-alarm SHAP explanations. Addresses Reviewer 1 point 6:
  "For interpretability to be meaningful in an operational context, the
   authors should demonstrate how SHAP explanations can influence
   decision-making, detect anomalies, or improve trust in deployment
   scenarios. Case studies or failure examples would significantly
   strengthen this section."

What this produces:
  1. Global feature-importance plot computed on REAL Mendeley data
     (replaces the synthetic-data SHAP bar chart in the current paper)
  2. Three case-study trajectories — one per OSNR-visible class
     (EDFA, NLI, and a stable No-failure), showing:
        - OSNR evolution over time
        - RF prediction curve (TTF estimate) over time
        - SHAP waterfall at the moment of alarm (first point with
          predicted TTF < 60 s for 3 consecutive polls)
  3. A fourth case study: a FALSE-ALARM example (if any exist) — alert
     fires but trajectory doesn't actually fail, illustrating how SHAP
     would help operators diagnose the miss.

Dependency:
    pip install shap

Run (after Phase 1 + Phase 2):
    python 07_shap_case_studies.py \
        --features_dir real_data_features \
        --rf_model real_data_results_10seeds/models/rf_seed0.pkl \
        --train_pkl real_data_processed/train_full.pkl \
        --output_dir shap_case_studies
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

try:
    import shap
except ImportError:
    raise SystemExit("shap not installed. Run:  pip install shap")


ALERT_THRESHOLD_S = 60.0   # predict TTF < 60 s to trigger
PERSISTENCE_POLLS = 3      # for 3 consecutive points (persistence filter)

LABEL_MAP = {0: "No failure", 1: "ECL", 2: "EDFA", 3: "NLI"}


# ----------------------------------------------------------------------
# Alarm detection on a predicted-TTF series
# ----------------------------------------------------------------------
def find_first_alarm(ttf_pred, threshold=ALERT_THRESHOLD_S,
                     persistence=PERSISTENCE_POLLS):
    """Return index of first alarm trigger (persistence filter), or None."""
    below = ttf_pred < threshold
    for i in range(len(ttf_pred) - persistence + 1):
        if below[i:i + persistence].all():
            return i + persistence - 1   # alarm time = 3rd consecutive below
    return None


def find_first_threshold_crossing(osnr_series, threshold=15.0):
    """Actual failure time (OSNR < 15 dB)."""
    below = osnr_series < threshold
    if below.any():
        return int(np.argmax(below))
    return None


# ----------------------------------------------------------------------
# Core single-trajectory analysis
# ----------------------------------------------------------------------
def analyze_trajectory(traj_id, df_full, rf_model, explainer,
                       feature_names, cap, window_size=10):
    """
    Returns a dict with:
      osnr, time, ttf_pred, alarm_time, fail_time,
      feature_matrix, shap_values_at_alarm, features_at_alarm,
      label, label_name
    """
    traj = df_full[df_full["trajectory_id"] == traj_id].sort_values("timestamp")
    osnr = traj["OSNR_dB"].values
    label = int(traj["failure_type"].iloc[0])

    # Rebuild the v4 feature matrix for this one trajectory
    s = pd.Series(osnr, name="SNR_dB")
    df = s.to_frame()
    for i in range(1, window_size + 1):
        df[f"SNR_Lag_{i}"] = df["SNR_dB"].shift(i)
    df = df.dropna().copy()
    df["Velocity"] = df["SNR_dB"] - df["SNR_Lag_1"]
    prev_vel = df["SNR_Lag_1"] - df["SNR_Lag_2"]
    df["Acceleration"] = df["Velocity"] - prev_vel
    lag_cols = [f"SNR_Lag_{i}" for i in range(1, 6)]
    df["Rolling_Std"]  = df[lag_cols].std(axis=1)
    df["Rolling_Mean"] = df[lag_cols].mean(axis=1)

    X = df[feature_names].values.astype(np.float32)
    time_idx = df.index.values  # original timestep indices

    ttf_pred = rf_model.predict(X)

    # Alarm and actual failure
    alarm_rel = find_first_alarm(ttf_pred)
    alarm_t = int(time_idx[alarm_rel]) if alarm_rel is not None else None
    fail_t = find_first_threshold_crossing(osnr)

    # SHAP at alarm moment
    shap_at_alarm = None
    feats_at_alarm = None
    if alarm_rel is not None:
        x_alarm = X[alarm_rel:alarm_rel + 1]
        shap_at_alarm = explainer.shap_values(x_alarm)[0]
        feats_at_alarm = x_alarm[0]

    return {
        "trajectory_id":    traj_id,
        "label":            label,
        "label_name":       LABEL_MAP[label],
        "osnr":             osnr,
        "time_full":        np.arange(len(osnr)),
        "time_feature":     time_idx,
        "ttf_pred":         ttf_pred,
        "alarm_time":       alarm_t,
        "fail_time":        fail_t,
        "lead_time":        (fail_t - alarm_t)
                              if (alarm_t is not None and fail_t is not None)
                              else None,
        "feature_matrix":   X,
        "shap_at_alarm":    shap_at_alarm,
        "feats_at_alarm":   feats_at_alarm,
    }


# ----------------------------------------------------------------------
# Pick representative trajectories
# ----------------------------------------------------------------------
def pick_cases(df_full, rf_model, feature_names, test_traj_ids,
               explainer, cap, n_per_class=1):
    """
    Score all test trajectories; pick one canonical example per class where
    an alarm AND an actual failure occur, preferring mid-range lead times.
    """
    picks = {"EDFA": None, "NLI": None, "NoFailure_stable": None,
             "FalseAlarm": None}

    # Cache analyses by trajectory
    alarms = []
    for tid in test_traj_ids:
        res = analyze_trajectory(tid, df_full, rf_model, explainer,
                                 feature_names, cap)
        alarms.append(res)

    df_s = pd.DataFrame([{
        "trajectory_id": r["trajectory_id"],
        "label":         r["label"],
        "has_alarm":     r["alarm_time"] is not None,
        "has_fail":      r["fail_time"] is not None,
        "lead_time":     r["lead_time"],
        "alarm_time":    r["alarm_time"],
        "fail_time":     r["fail_time"],
    } for r in alarms])

    # EDFA: true alarm with meaningful lead time
    edfa = df_s[(df_s["label"] == 2) & (df_s["has_alarm"]) & (df_s["has_fail"])]
    if len(edfa) > 0:
        # Prefer lead_time between 20 and 100 s (realistic operational window)
        good = edfa[(edfa["lead_time"] >= 20) & (edfa["lead_time"] <= 200)]
        chosen = good.iloc[len(good) // 2] if len(good) > 0 else edfa.iloc[0]
        picks["EDFA"] = next(r for r in alarms if r["trajectory_id"] == chosen["trajectory_id"])

    # NLI
    nli = df_s[(df_s["label"] == 3) & (df_s["has_alarm"]) & (df_s["has_fail"])]
    if len(nli) > 0:
        good = nli[(nli["lead_time"] >= 20) & (nli["lead_time"] <= 200)]
        chosen = good.iloc[len(good) // 2] if len(good) > 0 else nli.iloc[0]
        picks["NLI"] = next(r for r in alarms if r["trajectory_id"] == chosen["trajectory_id"])

    # Stable No-failure: no alarm, no failure — should be the common case
    nof = df_s[(df_s["label"] == 0) & (~df_s["has_alarm"]) & (~df_s["has_fail"])]
    if len(nof) > 0:
        tid = nof.iloc[0]["trajectory_id"]
        picks["NoFailure_stable"] = next(r for r in alarms
                                          if r["trajectory_id"] == tid)

    # False alarm: alarm fires but no actual failure
    fa = df_s[(df_s["has_alarm"]) & (~df_s["has_fail"])]
    if len(fa) > 0:
        tid = fa.iloc[0]["trajectory_id"]
        picks["FalseAlarm"] = next(r for r in alarms
                                    if r["trajectory_id"] == tid)

    summary_df = df_s
    return picks, summary_df


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_case(case, feature_names, out_path):
    """
    Produce a 3-panel figure:
      (top) OSNR over time w/ thresholds + alarm marker + failure marker
      (mid) Predicted TTF over time, showing how prediction evolves
      (bot) SHAP bar chart at alarm moment
    """
    if case is None:
        return

    fig, axes = plt.subplots(3, 1, figsize=(11, 10),
                             gridspec_kw={"height_ratios": [1, 1, 1.3]})

    # --- OSNR panel ---
    ax = axes[0]
    ax.plot(case["time_full"], case["osnr"], color="steelblue", lw=1.3,
            label="OSNR")
    ax.axhline(15, color="red", ls="--", lw=1.5,
               label="Hard threshold (15 dB)")
    ax.axhline(18, color="orange", ls="--", lw=1.5,
               label="Soft alarm (18 dB)")

    if case["alarm_time"] is not None:
        ax.axvline(case["alarm_time"], color="green", lw=2,
                   alpha=0.7, label=f"Alarm (t={case['alarm_time']})")
    if case["fail_time"] is not None:
        ax.axvline(case["fail_time"], color="red", lw=2,
                   alpha=0.7, label=f"Failure (t={case['fail_time']})")

    title_pieces = [f"{case['label_name']}", f"traj {case['trajectory_id']}"]
    if case["lead_time"] is not None:
        title_pieces.append(f"lead time = {case['lead_time']} s")
    elif case["alarm_time"] is not None and case["fail_time"] is None:
        title_pieces.append("FALSE ALARM")
    ax.set_title("  |  ".join(title_pieces))
    ax.set_ylabel("OSNR (dB)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    # --- Predicted TTF panel ---
    ax = axes[1]
    ax.plot(case["time_feature"], case["ttf_pred"],
            color="purple", lw=1.3, label="Predicted TTF")
    ax.axhline(ALERT_THRESHOLD_S, color="green", ls="--", lw=1.5,
               label=f"Alert threshold ({ALERT_THRESHOLD_S:.0f} s)")
    if case["alarm_time"] is not None:
        ax.axvline(case["alarm_time"], color="green", lw=2, alpha=0.7)
    if case["fail_time"] is not None:
        ax.axvline(case["fail_time"], color="red", lw=2, alpha=0.7)
    ax.set_ylabel("Predicted TTF (s)")
    ax.set_xlabel("Time (samples)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    # --- SHAP panel ---
    ax = axes[2]
    if case["shap_at_alarm"] is not None:
        svals = case["shap_at_alarm"]
        # Sort by absolute contribution for readability
        order = np.argsort(np.abs(svals))[::-1]
        top_k = min(10, len(svals))
        sv = svals[order[:top_k]]
        fn = [feature_names[i] for i in order[:top_k]]
        colors = ["#c0392b" if v < 0 else "#27ae60" for v in sv]

        # At alarm -> negative SHAP = pushing TTF DOWN (toward alarm)
        ax.barh(range(len(sv)), sv, color=colors, alpha=0.8)
        ax.set_yticks(range(len(sv)))
        ax.set_yticklabels(fn)
        ax.axvline(0, color="black", lw=0.5)
        ax.invert_yaxis()
        ax.set_xlabel("SHAP value (contribution to predicted TTF)")
        ax.set_title("Feature attribution at alarm moment "
                     "(red ↓ TTF, green ↑ TTF)")
        ax.grid(alpha=0.3, axis="x")
    else:
        ax.text(0.5, 0.5, "No alarm triggered on this trajectory",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_global_importance(rf_model, X_sample, feature_names,
                           out_path, sample_label="real Mendeley test data"):
    """Global SHAP summary computed on REAL data (not synthetic)."""
    print("Computing global SHAP values on real data sample ...")
    explainer = shap.TreeExplainer(rf_model)
    # Limit sample size for speed (SHAP TreeExplainer is O(nodes * samples))
    n_max = min(5000, len(X_sample))
    idx = np.random.default_rng(0).choice(len(X_sample), n_max, replace=False)
    shap_vals = explainer.shap_values(X_sample[idx])
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(mean_abs)),
            mean_abs[order],
            color=["#e74c3c" if feature_names[i] == "SNR_dB"
                   else "#f39c12" if feature_names[i] in ("Velocity", "Acceleration")
                   else "#3498db" if "Lag" in feature_names[i]
                   else "#95a5a6" for i in order])
    ax.set_yticks(range(len(mean_abs)))
    ax.set_yticklabels([feature_names[i] for i in order])
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Global feature importance on {sample_label} "
                 f"(n={n_max:,} samples)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return pd.DataFrame({
        "feature":         [feature_names[i] for i in order],
        "mean_abs_shap":   mean_abs[order],
        "relative_import": mean_abs[order] / mean_abs.sum(),
    })


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", default="real_data_features")
    parser.add_argument("--rf_model", required=True,
                        help="Path to a trained RF .pkl (e.g., "
                             "real_data_results_10seeds/models/rf_seed0.pkl)")
    parser.add_argument("--train_pkl", required=True,
                        help="Path to train_full.pkl from Phase 1 step 1")
    parser.add_argument("--output_dir", default="shap_case_studies")
    parser.add_argument("--n_candidate_trajectories", type=int, default=100,
                        help="How many test trajectories to scan for cases")
    args = parser.parse_args()

    feat_dir = Path(args.features_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    print(f"Loading RF model: {args.rf_model}")
    rf_model = joblib.load(args.rf_model)

    print(f"Loading raw data: {args.train_pkl}")
    df_full = pd.read_pickle(args.train_pkl)

    feature_names = json.loads((feat_dir / "feature_names.json").read_text())
    cap = float((feat_dir / "ttf_cap.txt").read_text().strip())

    X_te = np.load(feat_dir / "X_test.npy")
    test_traj_ids = np.load(feat_dir / "test_trajectory_ids.npy")

    # Sample subset of test trajectories for case hunting
    rng = np.random.default_rng(42)
    scan_ids = rng.choice(test_traj_ids,
                          size=min(args.n_candidate_trajectories,
                                   len(test_traj_ids)),
                          replace=False)

    # Build explainer once (expensive)
    print("Building TreeExplainer ...")
    explainer = shap.TreeExplainer(rf_model)

    print("Scanning for case-study trajectories ...")
    picks, summary_df = pick_cases(df_full, rf_model, feature_names,
                                    scan_ids, explainer, cap)
    summary_df.to_csv(out_dir / "candidate_trajectories.csv", index=False)

    for name, case in picks.items():
        if case is None:
            print(f"  no candidate found for {name}")
            continue
        path = out_dir / f"case_{name}_traj{case['trajectory_id']}.png"
        print(f"  {name}: traj={case['trajectory_id']}  "
              f"alarm={case['alarm_time']}  fail={case['fail_time']}  "
              f"lead={case['lead_time']} s  ->  {path.name}")
        plot_case(case, feature_names, path)

    # Global feature importance on REAL data (replaces synthetic-data SHAP in paper)
    fi_df = plot_global_importance(
        rf_model, X_te, feature_names,
        out_path=out_dir / "global_feature_importance_real.png",
        sample_label="real Mendeley test data")
    fi_df.to_csv(out_dir / "global_feature_importance_real.csv", index=False)

    print("\n" + "=" * 72)
    print("Global feature importance on REAL data:")
    print("=" * 72)
    print(fi_df.to_string(index=False))

    print(f"\nAll outputs in: {out_dir}/")
    print(">>> SEND BACK: the 3-4 case_*.png plots AND "
          "global_feature_importance_real.csv <<<")


if __name__ == "__main__":
    main()