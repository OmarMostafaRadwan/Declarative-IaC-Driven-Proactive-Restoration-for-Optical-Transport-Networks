"""
02_prepare_features_and_split.py
================================
Phase 1, Step 2 — Build the v4 feature set (lags, velocity, acceleration,
rolling stats) on the Mendeley data and split STRICTLY BY TRAJECTORY
(addresses Reviewer 2 point #4 on temporal leakage).

This script uses the RAW train_900 file (not test_300) and randomly assigns
trajectories to train/val/test by lightpath-level identity. Each trajectory
is an atomic unit — no samples from any given trajectory can appear in
more than one split.

TTF definition (matches paper, Eq. 11):
    y_t = min(t_fail - t, cap)
    where t_fail = first timestep with OSNR < 15 dB,
          cap    = trajectory length (trajectories that never fail are censored to `cap`)

Run (after Step 1):
    python 02_prepare_features_and_split.py \
        --input_pkl real_data_processed/train_full.pkl \
        --output_dir real_data_features \
        --test_frac 0.20 --val_frac 0.15 --seed 42

Outputs:
    real_data_features/X_train.npy,  y_train.npy,  traj_train.npy
    real_data_features/X_val.npy,    y_val.npy,    traj_val.npy
    real_data_features/X_test.npy,   y_test.npy,   traj_test.npy
    real_data_features/feature_names.json
    real_data_features/split_report.txt   (SEND THIS BACK)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FAILURE_THRESHOLD_DB = 15.0
WINDOW_SIZE = 10       # Number of SNR lags — matches paper's v4
ROLLING_WIN = 5        # Rolling std/mean window — matches paper's v4


# ----------------------------------------------------------------------
# Feature extraction (mirrors train_ai_v4_improved.py exactly)
# ----------------------------------------------------------------------
def build_v4_features_one_trajectory(osnr: np.ndarray,
                                     window_size: int = WINDOW_SIZE,
                                     rolling_win: int = ROLLING_WIN) -> pd.DataFrame:
    """
    Produce the v4 feature DataFrame for a single trajectory.
    Columns: SNR_dB, SNR_Lag_1..Lag_10, Velocity, Acceleration,
             Rolling_Std, Rolling_Mean.
    NaNs (from lag windows) are dropped. Returns a DataFrame whose index
    is the ORIGINAL timestep index so TTF can be aligned afterwards.
    """
    s = pd.Series(osnr, name="SNR_dB")
    df = s.to_frame()

    # Lags
    for i in range(1, window_size + 1):
        df[f"SNR_Lag_{i}"] = df["SNR_dB"].shift(i)

    df = df.dropna().copy()   # drops the first `window_size` rows

    # Derivatives
    df["Velocity"] = df["SNR_dB"] - df["SNR_Lag_1"]
    prev_velocity = df["SNR_Lag_1"] - df["SNR_Lag_2"]
    df["Acceleration"] = df["Velocity"] - prev_velocity

    # Rolling stats on the first `rolling_win` lags
    lag_cols = [f"SNR_Lag_{i}" for i in range(1, rolling_win + 1)]
    df["Rolling_Std"]  = df[lag_cols].std(axis=1)
    df["Rolling_Mean"] = df[lag_cols].mean(axis=1)

    return df


def compute_ttf(osnr: np.ndarray, cap: int) -> np.ndarray:
    """
    Time-to-failure target per timestep.
    t_fail = first index with OSNR < 15 dB.
    If the trajectory never reaches threshold, TTF == cap (censored).
    """
    n = len(osnr)
    below = osnr < FAILURE_THRESHOLD_DB
    if below.any():
        t_fail = int(np.argmax(below))
        ttf = np.minimum(np.arange(n, 0, -1) - (n - t_fail), cap)
        # The formulation above reduces to max(t_fail - t, 0) clipped to cap.
        # We compute it directly for clarity:
        t = np.arange(n)
        ttf = np.clip(t_fail - t, 0, cap).astype(np.float32)
    else:
        ttf = np.full(n, cap, dtype=np.float32)
    return ttf


# ----------------------------------------------------------------------
# Build the full feature matrix
# ----------------------------------------------------------------------
FEATURE_COLS = (
    ["SNR_dB"]
    + [f"SNR_Lag_{i}" for i in range(1, WINDOW_SIZE + 1)]
    + ["Velocity", "Acceleration", "Rolling_Std", "Rolling_Mean"]
)


def build_features_for_trajectories(df_full: pd.DataFrame,
                                    trajectory_ids: np.ndarray,
                                    cap: int) -> tuple:
    """
    Given the raw DataFrame and a set of trajectory IDs to include,
    return (X, y, traj_vec) arrays.
    """
    X_list, y_list, traj_list = [], [], []
    for tid in trajectory_ids:
        sub = df_full[df_full["trajectory_id"] == tid]
        osnr = sub["OSNR_dB"].values
        if len(osnr) < WINDOW_SIZE + 2:    # need at least lag+acc history
            continue

        feats = build_v4_features_one_trajectory(osnr)
        # Align TTF to the feature rows (feats has dropped the first WINDOW_SIZE rows)
        ttf_full = compute_ttf(osnr, cap=cap)
        ttf = ttf_full[feats.index.values]

        X_list.append(feats[FEATURE_COLS].values.astype(np.float32))
        y_list.append(ttf.astype(np.float32))
        traj_list.append(np.full(len(feats), tid, dtype=np.int32))

    X = np.concatenate(X_list, axis=0) if X_list else np.empty((0, len(FEATURE_COLS)), np.float32)
    y = np.concatenate(y_list, axis=0) if y_list else np.empty((0,), np.float32)
    traj = np.concatenate(traj_list, axis=0) if traj_list else np.empty((0,), np.int32)
    return X, y, traj


# ----------------------------------------------------------------------
# Trajectory-level split, stratified by failure label
# ----------------------------------------------------------------------
def trajectory_split(stats_df: pd.DataFrame,
                     test_frac: float,
                     val_frac: float,
                     seed: int) -> tuple:
    """
    Returns (train_ids, val_ids, test_ids) as np.ndarray of trajectory_id.
    Splits are stratified by failure label and reproducible by seed.
    """
    rng = np.random.default_rng(seed)
    train_ids, val_ids, test_ids = [], [], []

    for lbl in sorted(stats_df["label"].unique()):
        ids = stats_df[stats_df["label"] == lbl]["trajectory_id"].values
        ids = rng.permutation(ids)
        n = len(ids)
        n_test = int(round(n * test_frac))
        n_val  = int(round(n * val_frac))

        test_ids.append(ids[:n_test])
        val_ids.append(ids[n_test:n_test + n_val])
        train_ids.append(ids[n_test + n_val:])

    return (
        np.concatenate(train_ids),
        np.concatenate(val_ids),
        np.concatenate(test_ids),
    )


def write_split_report(stats_df, train_ids, val_ids, test_ids,
                       X_tr, y_tr, X_va, y_va, X_te, y_te, out_path):
    lines = []
    lines.append("=" * 72)
    lines.append("TRAJECTORY-BASED SPLIT REPORT")
    lines.append("=" * 72)

    def count_by_label(ids):
        sub = stats_df[stats_df["trajectory_id"].isin(ids)]
        return {int(k): int(v) for k, v in sub["label"].value_counts().to_dict().items()}

    for name, ids, X, y in [
        ("TRAIN", train_ids, X_tr, y_tr),
        ("VAL",   val_ids,   X_va, y_va),
        ("TEST",  test_ids,  X_te, y_te),
    ]:
        lines.append(f"\n[{name}]")
        lines.append(f"  Trajectories: {len(ids)}")
        lbl_counts = count_by_label(ids)
        for lbl in [0, 1, 2, 3]:
            lines.append(f"    label={lbl}: {lbl_counts.get(lbl, 0)} trajectories")
        lines.append(f"  Samples (rows after feature extraction): {len(X):,}")
        lines.append(f"  TTF target mean={y.mean():.2f}s  std={y.std():.2f}s  "
                     f"min={y.min():.0f}s  max={y.max():.0f}s")
        lines.append(f"  TTF==cap (censored): {int((y == y.max()).sum())} "
                     f"({100*(y == y.max()).mean():.1f}%)")

    # Sanity check: no trajectory overlap
    overlap = (set(train_ids.tolist()) & set(val_ids.tolist())) \
            | (set(train_ids.tolist()) & set(test_ids.tolist())) \
            | (set(val_ids.tolist())   & set(test_ids.tolist()))
    lines.append("")
    lines.append(f"Trajectory overlap between splits: {len(overlap)} "
                 f"(should be 0)")

    lines.append("=" * 72)
    report = "\n".join(lines)
    print(report)
    Path(out_path).write_text(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", required=True,
                        help="Path to train_full.pkl from step 1")
    parser.add_argument("--output_dir", default="real_data_features")
    parser.add_argument("--test_frac", type=float, default=0.20)
    parser.add_argument("--val_frac",  type=float, default=0.15)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the raw full DataFrame
    print(f"Loading {args.input_pkl} ...", flush=True)
    df = pd.read_pickle(args.input_pkl)
    print(f"  -> {len(df):,} rows, {df['trajectory_id'].max() + 1} trajectories")

    # Build a small per-trajectory label table for stratified split
    stats_df = df.groupby("trajectory_id").agg(
        label=("failure_type", "first")
    ).reset_index()

    # Split trajectories
    train_ids, val_ids, test_ids = trajectory_split(
        stats_df, args.test_frac, args.val_frac, args.seed
    )
    print(f"\nTrajectory split: "
          f"train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")

    # Determine cap (max trajectory length in the file — 900 for train_900)
    cap = int(df.groupby("trajectory_id").size().max())
    print(f"TTF cap = {cap} (max trajectory length)")

    # Build features per split
    print("\nBuilding v4 features ...")
    print("  TRAIN ...", flush=True)
    X_tr, y_tr, traj_tr = build_features_for_trajectories(df, train_ids, cap)
    print("  VAL   ...", flush=True)
    X_va, y_va, traj_va = build_features_for_trajectories(df, val_ids,   cap)
    print("  TEST  ...", flush=True)
    X_te, y_te, traj_te = build_features_for_trajectories(df, test_ids,  cap)

    # Save
    np.save(out_dir / "X_train.npy",   X_tr)
    np.save(out_dir / "y_train.npy",   y_tr)
    np.save(out_dir / "traj_train.npy", traj_tr)
    np.save(out_dir / "X_val.npy",     X_va)
    np.save(out_dir / "y_val.npy",     y_va)
    np.save(out_dir / "traj_val.npy",  traj_va)
    np.save(out_dir / "X_test.npy",    X_te)
    np.save(out_dir / "y_test.npy",    y_te)
    np.save(out_dir / "traj_test.npy", traj_te)

    # Also save the raw ID lists for traceability
    np.save(out_dir / "train_trajectory_ids.npy", train_ids)
    np.save(out_dir / "val_trajectory_ids.npy",   val_ids)
    np.save(out_dir / "test_trajectory_ids.npy",  test_ids)

    (out_dir / "feature_names.json").write_text(json.dumps(FEATURE_COLS, indent=2))
    (out_dir / "ttf_cap.txt").write_text(str(cap))

    write_split_report(
        stats_df, train_ids, val_ids, test_ids,
        X_tr, y_tr, X_va, y_va, X_te, y_te,
        out_dir / "split_report.txt"
    )

    print(f"\nDone. All outputs in: {out_dir}/")
    print(">>> SEND BACK: split_report.txt <<<")


if __name__ == "__main__":
    main()