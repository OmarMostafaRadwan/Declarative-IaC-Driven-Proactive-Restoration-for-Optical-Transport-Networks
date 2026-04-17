"""
03_train_and_evaluate_real.py
=============================
Phase 1, Step 3 — Train the same RandomForestRegressor used in
train_ai_v4_improved.py, but on the REAL Mendeley data with proper
trajectory-level splitting. Reports mean +/- 95% CI across multiple
random seeds (addresses Reviewer 2 point #7 on deterministic reporting).

Run (after Step 2):
    python 03_train_and_evaluate_real.py \
        --features_dir real_data_features \
        --output_dir real_data_results \
        --n_seeds 10

Increase --n_seeds to 30 for publication-grade CIs if training time allows.
With train_900-derived data (~2M feature rows), a single RF fit takes
~2-5 minutes on a 16-core machine; 10 seeds ~= 20-50 minutes.

Outputs:
    real_data_results/metrics_per_seed.csv
    real_data_results/summary.csv            (SEND THIS BACK)
    real_data_results/per_class_metrics.csv  (SEND THIS BACK)
    real_data_results/models/rf_seed<i>.pkl
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# Baselines: re-implemented here so this script is standalone
def simple_threshold_predict(X, feature_names, cap=300.0, soft_alarm_dB=20.0):
    """
    Predict TTF via linear extrapolation from last 5 lag samples.
    Fallback to cap if not below threshold.
    """
    snr_idx = feature_names.index("SNR_dB")
    lag1 = feature_names.index("SNR_Lag_1")
    lag5 = feature_names.index("SNR_Lag_5")

    snr_now = X[:, snr_idx]
    # rate = (SNR_t - SNR_{t-5}) / 5   -- positive = improving, negative = worsening
    rate = (X[:, snr_idx] - X[:, lag5]) / 5.0
    # predict TTF only if currently below soft_alarm AND declining
    pred = np.full(len(X), cap, dtype=np.float32)
    mask = (snr_now < soft_alarm_dB) & (rate < -1e-4)
    pred[mask] = np.clip((snr_now[mask] - 15.0) / (-rate[mask]), 0, cap)
    return pred


def linear_extrap_predict(X, feature_names, cap=300.0):
    """Fit line to SNR_dB + 10 lags, extrapolate to SNR==15."""
    cols = ["SNR_dB"] + [f"SNR_Lag_{i}" for i in range(1, 11)]
    idx = [feature_names.index(c) for c in cols]
    y_hist = X[:, idx]                      # shape (N, 11), reversed time order
    # time axis: 0 for current, 1 for lag_1, ..., 10 for lag_10 (looking BACKWARD)
    # so forward time from the oldest point is t_rev = 10 - idx
    # Use least-squares on each row
    t = np.arange(y_hist.shape[1])[::-1]    # 10, 9, ..., 0 (most recent last)
    t_mean = t.mean()
    t_dev = t - t_mean
    denom = (t_dev ** 2).sum()
    y_mean = y_hist.mean(axis=1, keepdims=True)
    slope = ((y_hist - y_mean) * t_dev).sum(axis=1) / denom
    intercept = y_mean.squeeze() - slope * t_mean
    # current prediction == intercept + slope * t=10 (most recent)
    # crossing 15 dB:  15 = intercept + slope * t_cross  =>  t_cross = (15 - intercept) / slope
    # lead time = t_cross - 10
    pred = np.full(len(X), cap, dtype=np.float32)
    valid = slope < -1e-4
    t_cross = (15.0 - intercept[valid]) / slope[valid]
    lead = np.clip(t_cross - 10.0, 0, cap)
    pred[valid] = lead
    return pred


def moving_avg_predict(X, feature_names, cap=300.0, alpha=0.3):
    """EWMA rate over the lag window."""
    cols = ["SNR_dB"] + [f"SNR_Lag_{i}" for i in range(1, 11)]
    idx = [feature_names.index(c) for c in cols]
    y_hist = X[:, idx]   # newest to oldest: [SNR_t, SNR_{t-1}, ..., SNR_{t-10}]
    # compute differences (current - previous) from newest pair onward
    diffs = y_hist[:, :-1] - y_hist[:, 1:]   # shape (N, 10)
    # EWMA weights, most recent first
    weights = alpha * (1 - alpha) ** np.arange(diffs.shape[1])
    weights /= weights.sum()
    rate = (diffs * weights).sum(axis=1)     # average dSNR per step
    snr_now = X[:, feature_names.index("SNR_dB")]
    pred = np.full(len(X), cap, dtype=np.float32)
    mask = rate < -1e-4
    pred[mask] = np.clip((snr_now[mask] - 15.0) / (-rate[mask]), 0, cap)
    return pred


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def evaluate_model(name, y_true, y_pred, approaching_mask=None):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    try:
        r2 = float(r2_score(y_true, y_pred))
    except ValueError:
        r2 = float("nan")
    result = {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2}
    if approaching_mask is not None and approaching_mask.sum() > 0:
        yt, yp = y_true[approaching_mask], y_pred[approaching_mask]
        result["MAE_approaching"] = float(mean_absolute_error(yt, yp))
        result["RMSE_approaching"] = float(np.sqrt(mean_squared_error(yt, yp)))
        result["R2_approaching"] = float(r2_score(yt, yp)) if yt.std() > 0 else float("nan")
    return result


def per_class_metrics(y_true, y_pred, traj_test, stats_df, cap):
    """
    Break metrics down by failure class using the test trajectory IDs.
    Reports both ALL samples and APPROACHING-FAILURE samples (TTF < cap),
    since the vast majority of ECL/EDFA samples are censored at `cap` and
    the overall MAE would be misleadingly low.
    """
    rows = []
    for lbl in [0, 1, 2, 3]:
        class_tids = stats_df[stats_df["label"] == lbl]["trajectory_id"].values
        mask_all = np.isin(traj_test, class_tids)
        if mask_all.sum() == 0:
            continue

        yt, yp = y_true[mask_all], y_pred[mask_all]
        row = {
            "label":           lbl,
            "n_samples":       int(mask_all.sum()),
            "n_approaching":   int((yt < cap).sum()),
            "MAE_all":         float(mean_absolute_error(yt, yp)),
            "RMSE_all":        float(np.sqrt(mean_squared_error(yt, yp))),
            "R2_all":          float(r2_score(yt, yp))
                                if yt.std() > 0 else float("nan"),
        }
        # Approaching-failure subset (y < cap)
        sub = yt < cap
        if sub.sum() > 0:
            row["MAE_approaching"]  = float(mean_absolute_error(yt[sub], yp[sub]))
            row["RMSE_approaching"] = float(np.sqrt(mean_squared_error(yt[sub], yp[sub])))
            row["R2_approaching"]   = float(r2_score(yt[sub], yp[sub])) \
                                        if yt[sub].std() > 0 else float("nan")
        else:
            row["MAE_approaching"]  = float("nan")
            row["RMSE_approaching"] = float("nan")
            row["R2_approaching"]   = float("nan")

        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", default="real_data_features")
    parser.add_argument("--output_dir",   default="real_data_results")
    parser.add_argument("--n_seeds", type=int, default=10,
                        help="Number of random-seed reruns for CI estimation")
    parser.add_argument("--n_estimators", type=int, default=150)
    parser.add_argument("--max_depth",    type=int, default=12)
    parser.add_argument("--min_samples_leaf", type=int, default=4)
    parser.add_argument("--train_pkl", default=None,
                        help="Optional: path to train_full.pkl from step 1, "
                             "used only to produce per-class breakdown")
    args = parser.parse_args()

    feat_dir = Path(args.features_dir)
    out_dir  = Path(args.output_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    # Load features + targets
    print("Loading features ...")
    X_tr  = np.load(feat_dir / "X_train.npy")
    y_tr  = np.load(feat_dir / "y_train.npy")
    X_va  = np.load(feat_dir / "X_val.npy")
    y_va  = np.load(feat_dir / "y_val.npy")
    X_te  = np.load(feat_dir / "X_test.npy")
    y_te  = np.load(feat_dir / "y_test.npy")
    traj_te = np.load(feat_dir / "traj_test.npy")
    feature_names = json.loads((feat_dir / "feature_names.json").read_text())
    cap = float((feat_dir / "ttf_cap.txt").read_text().strip())

    print(f"  X_train: {X_tr.shape}  X_val: {X_va.shape}  X_test: {X_te.shape}")
    print(f"  TTF cap = {cap}s;  "
          f"y_test: {100*(y_te < cap).mean():.1f}% approaching failure, "
          f"{100*(y_te >= cap).mean():.1f}% censored")
    approaching_mask = y_te < cap

    # --- Baselines (seed-independent) ---
    print("\n" + "-" * 72)
    print("Baselines (on real test set)")
    print("-" * 72)
    baseline_rows = []
    for bname, bfn in [
        ("Simple Threshold",    lambda X: simple_threshold_predict(X, feature_names)),
        ("Linear Extrapolation", lambda X: linear_extrap_predict(X, feature_names)),
        ("Moving Average",      lambda X: moving_avg_predict(X, feature_names)),
    ]:
        y_pred = bfn(X_te)
        m = evaluate_model(bname, y_te, y_pred, approaching_mask)
        print(f"  {m['Model']:22s}  MAE={m['MAE']:6.2f}  "
              f"RMSE={m['RMSE']:6.2f}  R2={m['R2']:.3f}   | "
              f"approaching: MAE={m.get('MAE_approaching', float('nan')):6.2f}")
        baseline_rows.append(m)

    # --- Random Forest: n_seeds reruns ---
    print("\n" + "-" * 72)
    print(f"Random Forest: {args.n_seeds} seeds")
    print("-" * 72)

    rf_rows = []
    last_preds = None
    for s in range(args.n_seeds):
        t0 = time.time()
        rf = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=s,
            n_jobs=-1,
        )
        rf.fit(X_tr, y_tr)
        y_pred = rf.predict(X_te)
        m = evaluate_model(f"RF_seed{s}", y_te, y_pred, approaching_mask)
        m["train_time_s"] = round(time.time() - t0, 2)
        print(f"  seed={s:2d}  MAE={m['MAE']:6.2f}  RMSE={m['RMSE']:6.2f}  "
              f"R2={m['R2']:.3f}  | approaching MAE={m.get('MAE_approaching', float('nan')):6.2f}"
              f"  time={m['train_time_s']}s")
        rf_rows.append(m)
        joblib.dump(rf, out_dir / "models" / f"rf_seed{s}.pkl")
        last_preds = y_pred

    # --- Aggregated statistics with 95% CI ---
    rf_df = pd.DataFrame(rf_rows)
    rf_df.to_csv(out_dir / "metrics_per_seed.csv", index=False)

    def summarize(col):
        vals = rf_df[col].values
        mean = vals.mean()
        sem = sps.sem(vals) if len(vals) > 1 else 0.0
        ci = sps.t.interval(0.95, len(vals) - 1, loc=mean, scale=sem) \
             if len(vals) > 1 else (mean, mean)
        return {
            "mean": mean,
            "std": vals.std(ddof=1) if len(vals) > 1 else 0.0,
            "CI_lower": ci[0],
            "CI_upper": ci[1],
        }

    summary_rows = []
    for bm in baseline_rows:
        summary_rows.append({
            "Model": bm["Model"],
            "n_seeds": 1,
            "MAE_mean": bm["MAE"], "MAE_std": 0.0,
            "MAE_CI_lower": bm["MAE"], "MAE_CI_upper": bm["MAE"],
            "RMSE_mean": bm["RMSE"], "RMSE_std": 0.0,
            "R2_mean": bm["R2"],   "R2_std": 0.0,
            "MAE_approaching_mean": bm.get("MAE_approaching", float("nan")),
            "R2_approaching_mean":  bm.get("R2_approaching",  float("nan")),
        })
    rf_summary = {
        "Model": "Random Forest v4",
        "n_seeds": args.n_seeds,
    }
    for metric in ["MAE", "RMSE", "R2", "MAE_approaching", "R2_approaching"]:
        if metric in rf_df.columns:
            s = summarize(metric)
            rf_summary[f"{metric}_mean"] = s["mean"]
            rf_summary[f"{metric}_std"]  = s["std"]
            if metric == "MAE":
                rf_summary["MAE_CI_lower"] = s["CI_lower"]
                rf_summary["MAE_CI_upper"] = s["CI_upper"]
    summary_rows.append(rf_summary)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    print("\n" + "=" * 72)
    print("SUMMARY (real-data test set)")
    print("=" * 72)
    with pd.option_context("display.max_columns", None,
                           "display.width", 120,
                           "display.precision", 3):
        print(summary_df.to_string(index=False))

    # --- Per-class breakdown using LAST seed's predictions ---
    if args.train_pkl:
        print("\n" + "-" * 72)
        print("Per-class breakdown (last-seed RF predictions)")
        print("-" * 72)
        df_full = pd.read_pickle(args.train_pkl)
        stats_df = df_full.groupby("trajectory_id").agg(
            label=("failure_type", "first")
        ).reset_index()
        pc = per_class_metrics(y_te, last_preds, traj_te, stats_df, cap)
        label_map = {0: "No failure", 1: "ECL", 2: "EDFA", 3: "NLI"}
        pc["label_name"] = pc["label"].map(label_map)
        pc.to_csv(out_dir / "per_class_metrics.csv", index=False)
        print(pc.to_string(index=False))

    print(f"\nAll outputs in: {out_dir}/")
    print(">>> SEND BACK: summary.csv AND per_class_metrics.csv AND the console log <<<")


if __name__ == "__main__":
    main()