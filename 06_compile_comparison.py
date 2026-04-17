"""
06_compile_comparison.py
========================
Phase 2, Step 3 — Collect results from:
  - 03_train_and_evaluate_real.py       (Random Forest + heuristics)
  - 04_xgboost_baseline.py              (XGBoost)
  - 05_deep_baselines.py                (LSTM, 1D-CNN)

and produce one master comparison table with MAE mean +/- 95% CI,
overall and approaching-failure, ready to drop into the paper as
Table II (Revised).

Run:
    python 06_compile_comparison.py \
        --rf_dir    real_data_results_10seeds \
        --xgb_dir   real_data_results_xgboost \
        --deep_dir  real_data_results_deep \
        --output    real_data_results_final

Any missing directory is silently skipped, so you can run with only a
subset of baselines while others are still training.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_rf_summary(rf_dir):
    """
    Script 03 puts baselines (Simple Threshold, Linear Extrapolation,
    Moving Average) + Random Forest all into one summary.csv.  Read it
    and normalize columns.
    """
    f = Path(rf_dir) / "summary.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    return df


def load_single_summary(d, model_label_override=None):
    """XGBoost / Deep script: one row summary.csv each."""
    f = Path(d) / "summary.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    if model_label_override is not None and len(df) == 1:
        df["Model"] = model_label_override
    return df


def format_headline(df):
    """Produce the compact headline table for the paper."""
    keep_cols = [
        "Model", "n_seeds",
        "MAE_mean", "MAE_std", "MAE_CI_lower", "MAE_CI_upper",
        "RMSE_mean", "R2_mean",
        "MAE_approaching_mean", "R2_approaching_mean",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    out = df[keep_cols].copy()

    # Round for display
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].round(3)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rf_dir",   default="real_data_results_10seeds",
                   help="Output dir from 03_train_and_evaluate_real.py")
    p.add_argument("--xgb_dir",  default="real_data_results_xgboost",
                   help="Output dir from 04_xgboost_baseline.py")
    p.add_argument("--deep_dir", default="real_data_results_deep",
                   help="Output dir from 05_deep_baselines.py")
    p.add_argument("--output",   default="real_data_results_final")
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    frames = []

    df_rf = load_rf_summary(args.rf_dir)
    if df_rf is not None:
        frames.append(df_rf)
    else:
        print(f"WARN: {args.rf_dir}/summary.csv not found — skipping RF + heuristics")

    df_xgb = load_single_summary(args.xgb_dir, "XGBoost")
    if df_xgb is not None:
        frames.append(df_xgb)
    else:
        print(f"WARN: {args.xgb_dir}/summary.csv not found — skipping XGBoost")

    df_deep = load_single_summary(args.deep_dir)
    if df_deep is not None:
        frames.append(df_deep)
    else:
        print(f"WARN: {args.deep_dir}/summary.csv not found — skipping LSTM/CNN")

    if not frames:
        raise SystemExit("No result directories found.  Nothing to compile.")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(out / "all_models_summary.csv", index=False)

    # Sort by MAE_mean for the headline table
    if "MAE_mean" in combined.columns:
        combined_sorted = combined.sort_values("MAE_mean", ascending=True).reset_index(drop=True)
    else:
        combined_sorted = combined

    headline = format_headline(combined_sorted)
    headline.to_csv(out / "headline_table.csv", index=False)

    print("\n" + "=" * 80)
    print("MASTER COMPARISON — all baselines + proposed model on real Mendeley data")
    print("=" * 80)
    with pd.option_context("display.max_columns", None,
                           "display.width", 180,
                           "display.precision", 3):
        print(headline.to_string(index=False))

    # Also assemble per-class if available (last-seed predictions)
    pc_frames = []
    for name, d in [("RF",     Path(args.rf_dir)   / "per_class_metrics.csv"),
                    ("XGB",    Path(args.xgb_dir)  / "per_class_metrics.csv"),
                    ("LSTM",   Path(args.deep_dir) / "per_class_metrics_lstm.csv"),
                    ("CNN",    Path(args.deep_dir) / "per_class_metrics_cnn.csv")]:
        if d.exists():
            pc = pd.read_csv(d)
            pc["Model"] = name
            pc_frames.append(pc)

    if pc_frames:
        pc_all = pd.concat(pc_frames, ignore_index=True)
        # Pivot to a readable view: rows = (label_name), cols = (Model, metric)
        pc_all.to_csv(out / "all_models_per_class.csv", index=False)
        print("\n" + "-" * 80)
        print("PER-CLASS MAE (approaching failure) by model")
        print("-" * 80)
        piv = pc_all.pivot_table(
            index="label_name",
            columns="Model",
            values="MAE_approaching",
            aggfunc="first",
        ).round(2)
        print(piv.to_string())

    print(f"\nAll outputs in: {out}/")
    print(">>> SEND BACK: headline_table.csv AND all_models_per_class.csv <<<")


if __name__ == "__main__":
    main()