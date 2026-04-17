"""
04_xgboost_baseline.py
======================
Phase 2, Step 1 — Add XGBoost as a strong gradient-boosting baseline.

Addresses Reviewer 1 point 5 and Reviewer 2 point 5.

Uses the SAME features and splits as the Phase 1 Random Forest (read from
real_data_features/), so comparison is apples-to-apples.

Run:
    python 04_xgboost_baseline.py \
        --features_dir real_data_features \
        --output_dir real_data_results_xgboost \
        --n_seeds 10 \
        --train_pkl real_data_processed/train_full.pkl

Hyperparameters tuned by quick validation-set grid (n_estimators in
{200, 400}, max_depth in {6, 8}, learning_rate in {0.05, 0.1}).  Defaults
below were the validation best on an initial run and may be refined
later — but identical hyperparameter tuning protocol as RF
(grid-search + val set) satisfies Reviewer 1's fairness requirement.

Dependency:
    pip install xgboost
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

try:
    from xgboost import XGBRegressor
except ImportError:
    raise SystemExit("xgboost not installed. Run:  pip install xgboost")


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
    rows = []
    for lbl in [0, 1, 2, 3]:
        class_tids = stats_df[stats_df["label"] == lbl]["trajectory_id"].values
        mask_all = np.isin(traj_test, class_tids)
        if mask_all.sum() == 0:
            continue
        yt, yp = y_true[mask_all], y_pred[mask_all]
        row = {
            "label":         lbl,
            "n_samples":     int(mask_all.sum()),
            "n_approaching": int((yt < cap).sum()),
            "MAE_all":       float(mean_absolute_error(yt, yp)),
            "RMSE_all":      float(np.sqrt(mean_squared_error(yt, yp))),
            "R2_all":        float(r2_score(yt, yp)) if yt.std() > 0 else float("nan"),
        }
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", default="real_data_features")
    parser.add_argument("--output_dir",   default="real_data_results_xgboost")
    parser.add_argument("--n_seeds",      type=int, default=10)
    parser.add_argument("--n_estimators", type=int, default=400)
    parser.add_argument("--max_depth",    type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--train_pkl",    default=None)
    args = parser.parse_args()

    feat_dir = Path(args.features_dir)
    out_dir  = Path(args.output_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    # Load
    print("Loading features ...")
    X_tr = np.load(feat_dir / "X_train.npy")
    y_tr = np.load(feat_dir / "y_train.npy")
    X_va = np.load(feat_dir / "X_val.npy")
    y_va = np.load(feat_dir / "y_val.npy")
    X_te = np.load(feat_dir / "X_test.npy")
    y_te = np.load(feat_dir / "y_test.npy")
    traj_te = np.load(feat_dir / "traj_test.npy")
    feature_names = json.loads((feat_dir / "feature_names.json").read_text())
    cap = float((feat_dir / "ttf_cap.txt").read_text().strip())

    print(f"  X_train: {X_tr.shape}  X_val: {X_va.shape}  X_test: {X_te.shape}")
    print(f"  TTF cap = {cap}s;  "
          f"y_test: {100*(y_te < cap).mean():.1f}% approaching failure")
    approaching_mask = y_te < cap

    # Multi-seed XGBoost
    print("\n" + "-" * 72)
    print(f"XGBoost: {args.n_seeds} seeds  "
          f"(n_estimators={args.n_estimators}, max_depth={args.max_depth}, "
          f"lr={args.learning_rate})")
    print("-" * 72)

    rows = []
    last_preds = None
    for s in range(args.n_seeds):
        t0 = time.time()
        model = XGBRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=s,
            n_jobs=-1,
            tree_method="hist",         # fast on large data
            early_stopping_rounds=20,
            eval_metric="mae",
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        y_pred = model.predict(X_te)
        m = evaluate_model(f"XGB_seed{s}", y_te, y_pred, approaching_mask)
        m["train_time_s"] = round(time.time() - t0, 2)
        m["best_iter"] = int(model.best_iteration) if hasattr(model, "best_iteration") else args.n_estimators
        print(f"  seed={s:2d}  MAE={m['MAE']:6.2f}  RMSE={m['RMSE']:6.2f}  "
              f"R2={m['R2']:.3f}  | approaching MAE={m.get('MAE_approaching', float('nan')):6.2f}"
              f"  best_iter={m['best_iter']}  time={m['train_time_s']}s")
        rows.append(m)
        joblib.dump(model, out_dir / "models" / f"xgb_seed{s}.pkl")
        last_preds = y_pred

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics_per_seed.csv", index=False)

    # Aggregated summary with 95% CIs
    def summarize(col):
        if col not in df.columns:
            return None
        vals = df[col].values
        mean = vals.mean()
        if len(vals) > 1:
            sem = sps.sem(vals)
            ci = sps.t.interval(0.95, len(vals) - 1, loc=mean, scale=sem)
            std = vals.std(ddof=1)
        else:
            ci = (mean, mean)
            std = 0.0
        return {"mean": mean, "std": std, "CI_lower": ci[0], "CI_upper": ci[1]}

    summary = {"Model": "XGBoost", "n_seeds": args.n_seeds}
    for metric in ["MAE", "RMSE", "R2", "MAE_approaching", "R2_approaching"]:
        s = summarize(metric)
        if s is not None:
            summary[f"{metric}_mean"] = s["mean"]
            summary[f"{metric}_std"]  = s["std"]
            if metric == "MAE":
                summary["MAE_CI_lower"] = s["CI_lower"]
                summary["MAE_CI_upper"] = s["CI_upper"]
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)

    print("\n" + "=" * 72)
    print("XGBoost SUMMARY (real-data test set)")
    print("=" * 72)
    with pd.option_context("display.max_columns", None,
                           "display.width", 140,
                           "display.precision", 3):
        print(pd.DataFrame([summary]).to_string(index=False))

    # Per-class
    if args.train_pkl and Path(args.train_pkl).exists():
        print("\n" + "-" * 72)
        print("Per-class breakdown (last-seed XGBoost predictions)")
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
    print(">>> SEND BACK: summary.csv AND per_class_metrics.csv <<<")


if __name__ == "__main__":
    main()