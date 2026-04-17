"""
01_load_and_characterize.py
===========================
Phase 1, Step 1 — Load the Mendeley (Ghosh & Adhya 2025) optical failure dataset
and characterize its structure.

Mendeley DOI: 10.17632/y3pspy7j83.1
License: CC BY 4.0

Run:
    python 01_load_and_characterize.py \
        --test Lightpath_756_label_4_QoT_dataset_test_300.txt \
        --train Lightpath_756_label_4_QoT_dataset_train_900.txt \
        --output real_data_processed

Outputs:
    real_data_processed/test_full.pkl        — full test DataFrame (pickled)
    real_data_processed/train_full.pkl       — full train DataFrame (pickled)
    real_data_processed/test_stats.csv       — per-trajectory stats (test)
    real_data_processed/train_stats.csv      — per-trajectory stats (train)
    real_data_processed/characterization_*.png — diagnostic plots
    real_data_processed/summary.txt          — text summary (SEND THIS BACK)
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


FAILURE_THRESHOLD_DB = 15.0       # Matches paper's hard-failure threshold
WARNING_THRESHOLD_DB = 18.0       # Matches paper's soft-failure alarm

FAILURE_LABELS = {
    0: "No failure",
    1: "ECL failure",
    2: "EDFA failure",
    3: "NLI failure",
}

COL_NAMES = [
    "timestamp", "lp_length_km", "laser_current_mA",
    "lp_power_dBm", "OSNR_dB", "BER_dB", "failure_type",
]


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_mendeley_file(filepath: str) -> pd.DataFrame:
    """Parse one Mendeley Ghosh-Adhya file."""
    print(f"Loading {filepath} ...", flush=True)
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        skiprows=2,               # line 1 = metadata, line 2 = header (we supply names)
        header=None,
        names=COL_NAMES,
        engine="python",
    )
    print(f"  -> {len(df):,} rows loaded", flush=True)
    return df


def identify_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    """Add trajectory_id column by detecting timestamp resets (timestamp==1)."""
    df = df.copy()
    starts = (df["timestamp"] == 1).astype(int)
    df["trajectory_id"] = starts.cumsum() - 1
    return df


# ----------------------------------------------------------------------
# Characterization
# ----------------------------------------------------------------------
def characterize_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-trajectory summary statistics."""
    rows = []
    for tid, grp in df.groupby("trajectory_id", sort=True):
        osnr = grp["OSNR_dB"].values
        ber = grp["BER_dB"].values
        label = int(grp["failure_type"].iloc[0])
        lp_len = int(grp["lp_length_km"].iloc[0])

        below_hard = osnr < FAILURE_THRESHOLD_DB
        below_soft = osnr < WARNING_THRESHOLD_DB

        if below_hard.any():
            first_hard = int(np.argmax(below_hard))
            reaches_hard = True
        else:
            first_hard = -1
            reaches_hard = False

        if below_soft.any():
            first_soft = int(np.argmax(below_soft))
            reaches_soft = True
        else:
            first_soft = -1
            reaches_soft = False

        osnr_start = float(osnr[:10].mean())
        osnr_end = float(osnr[-10:].mean())

        rows.append({
            "trajectory_id":        tid,
            "label":                label,
            "label_name":           FAILURE_LABELS[label],
            "lp_length_km":         lp_len,
            "n_samples":            len(grp),
            "osnr_start":           osnr_start,
            "osnr_end":             osnr_end,
            "osnr_min":             float(osnr.min()),
            "osnr_max":             float(osnr.max()),
            "osnr_std":             float(osnr.std()),
            "total_degradation_dB": osnr_start - osnr_end,
            "ber_mean":             float(ber.mean()),
            "reaches_15dB":         reaches_hard,
            "reaches_18dB":         reaches_soft,
            "first_15dB_crossing":  first_hard,
            "first_18dB_crossing":  first_soft,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------
def plot_sample_trajectories(df, stats_df, output_dir, prefix):
    """Plot 5 example trajectories per failure class."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    rng = np.random.default_rng(42)

    for lbl in [0, 1, 2, 3]:
        ax = axes.flat[lbl]
        candidates = stats_df[stats_df["label"] == lbl]["trajectory_id"].values
        if len(candidates) == 0:
            continue
        pick = rng.choice(candidates, size=min(5, len(candidates)), replace=False)

        for tid in pick:
            sub = df[df["trajectory_id"] == tid]
            ax.plot(sub["timestamp"].values, sub["OSNR_dB"].values,
                    alpha=0.7, linewidth=1.1)

        ax.axhline(15.0, color="red", ls="--", label="Hard threshold (15 dB)")
        ax.axhline(18.0, color="orange", ls="--", label="Soft alarm (18 dB)")
        ax.set_title(f"{FAILURE_LABELS[lbl]}  (label={lbl})")
        ax.set_xlabel("Time step")
        ax.set_ylabel("OSNR (dB)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")

    fig.suptitle(f"Mendeley Dataset ({prefix}): sample trajectories per failure class")
    fig.tight_layout()
    fig.savefig(output_dir / f"characterization_{prefix}_trajectories.png", dpi=150)
    plt.close(fig)


def plot_osnr_distribution(stats_df, output_dir, prefix):
    """Histogram of osnr_min and boxplot of total degradation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of OSNR_min per class
    ax = axes[0]
    for lbl in [0, 1, 2, 3]:
        vals = stats_df[stats_df["label"] == lbl]["osnr_min"].values
        ax.hist(vals, bins=30, alpha=0.55, label=FAILURE_LABELS[lbl])
    ax.axvline(15.0, color="red", ls="--", lw=2, label="15 dB threshold")
    ax.axvline(18.0, color="orange", ls="--", lw=2, label="18 dB alarm")
    ax.set_xlabel("Minimum OSNR per trajectory (dB)")
    ax.set_ylabel("Number of trajectories")
    ax.set_title("Distribution of OSNR minimum")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Boxplot of total degradation
    ax = axes[1]
    data_by_class = [
        stats_df[stats_df["label"] == lbl]["total_degradation_dB"].values
        for lbl in [0, 1, 2, 3]
    ]
    ax.boxplot(data_by_class, tick_labels=[FAILURE_LABELS[l] for l in [0, 1, 2, 3]])
    ax.set_ylabel("OSNR degradation (dB, start - end)")
    ax.set_title("Total OSNR degradation per trajectory")
    ax.grid(alpha=0.3, axis="y")
    ax.axhline(0, color="black", lw=0.5)

    fig.suptitle(f"Mendeley Dataset ({prefix}): OSNR statistics by failure class")
    fig.tight_layout()
    fig.savefig(output_dir / f"characterization_{prefix}_osnr_stats.png", dpi=150)
    plt.close(fig)


def plot_threshold_crossings(stats_df, output_dir, prefix):
    """Bar chart: fraction of trajectories reaching 15 dB and 18 dB by class."""
    fig, ax = plt.subplots(figsize=(10, 5))

    labels = [FAILURE_LABELS[l] for l in [0, 1, 2, 3]]
    hard_pct = [100 * stats_df[stats_df["label"] == l]["reaches_15dB"].mean()
                for l in [0, 1, 2, 3]]
    soft_pct = [100 * stats_df[stats_df["label"] == l]["reaches_18dB"].mean()
                for l in [0, 1, 2, 3]]

    x = np.arange(4)
    w = 0.35
    ax.bar(x - w/2, soft_pct, w, color="orange", alpha=0.8,
           label="Reaches 18 dB (soft alarm)")
    ax.bar(x + w/2, hard_pct, w, color="red", alpha=0.8,
           label="Reaches 15 dB (hard threshold)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("% of trajectories")
    ax.set_title(f"Threshold crossings by failure class ({prefix})")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    for i, (s, h) in enumerate(zip(soft_pct, hard_pct)):
        ax.text(i - w/2, s + 1, f"{s:.1f}%", ha="center", fontsize=9)
        ax.text(i + w/2, h + 1, f"{h:.1f}%", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / f"characterization_{prefix}_crossings.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# Summary writer
# ----------------------------------------------------------------------
def write_summary(stats_test, stats_train, output_dir):
    """Write a compact text summary. THIS IS THE FILE TO SEND BACK."""
    lines = []
    lines.append("=" * 72)
    lines.append("MENDELEY DATASET CHARACTERIZATION SUMMARY")
    lines.append("Ghosh & Adhya (2025), DOI 10.17632/y3pspy7j83.1")
    lines.append("=" * 72)
    lines.append("")

    for name, stats in [("TEST", stats_test), ("TRAIN", stats_train)]:
        if stats is None:
            continue
        lines.append(f"--- {name} SET ---")
        lines.append(f"Total trajectories: {len(stats)}")
        lines.append(f"Samples per trajectory: {int(stats['n_samples'].iloc[0])}")
        lines.append(f"Total samples: {int(stats['n_samples'].sum()):,}")
        lines.append("")
        lines.append("Trajectories per failure class:")
        for lbl in [0, 1, 2, 3]:
            n = int((stats["label"] == lbl).sum())
            lines.append(f"  {FAILURE_LABELS[lbl]:12s} : {n}")
        lines.append("")

        lines.append("Trajectories that reach 15 dB (hard-failure) threshold:")
        for lbl in [0, 1, 2, 3]:
            sub = stats[stats["label"] == lbl]
            n = int(sub["reaches_15dB"].sum())
            pct = 100 * sub["reaches_15dB"].mean() if len(sub) > 0 else 0.0
            lines.append(f"  {FAILURE_LABELS[lbl]:12s} : {n:4d} / {len(sub):4d}  ({pct:5.1f}%)")
        lines.append("")

        lines.append("Trajectories that reach 18 dB (soft-alarm) threshold:")
        for lbl in [0, 1, 2, 3]:
            sub = stats[stats["label"] == lbl]
            n = int(sub["reaches_18dB"].sum())
            pct = 100 * sub["reaches_18dB"].mean() if len(sub) > 0 else 0.0
            lines.append(f"  {FAILURE_LABELS[lbl]:12s} : {n:4d} / {len(sub):4d}  ({pct:5.1f}%)")
        lines.append("")

        lines.append("OSNR statistics by failure class:")
        lines.append(f"{'Class':12s} {'start_mean':>10s} {'end_mean':>10s} "
                     f"{'min_mean':>10s} {'degrad_mean':>12s}")
        for lbl in [0, 1, 2, 3]:
            sub = stats[stats["label"] == lbl]
            if len(sub) == 0:
                continue
            lines.append(
                f"{FAILURE_LABELS[lbl]:12s} "
                f"{sub['osnr_start'].mean():10.2f} "
                f"{sub['osnr_end'].mean():10.2f} "
                f"{sub['osnr_min'].mean():10.2f} "
                f"{sub['total_degradation_dB'].mean():12.2f}"
            )
        lines.append("")

        lines.append("LP length distribution (unique values):")
        lengths = sorted(stats["lp_length_km"].unique())
        lines.append(f"  min={min(lengths)}  max={max(lengths)}  "
                     f"n_unique={len(lengths)}")
        lines.append("")
        lines.append("")

    summary = "\n".join(lines)
    (output_dir / "summary.txt").write_text(summary)
    print("\n" + summary)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True,
                        help="Path to test_300 file")
    parser.add_argument("--train", default=None,
                        help="Path to train_900 file (optional)")
    parser.add_argument("--output", default="real_data_processed",
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PHASE 1 / STEP 1 — Mendeley dataset characterization")
    print("=" * 72)

    # -- TEST set (required) --
    if not os.path.exists(args.test):
        sys.exit(f"ERROR: test file not found: {args.test}")
    df_test = load_mendeley_file(args.test)
    df_test = identify_trajectories(df_test)
    stats_test = characterize_trajectories(df_test)
    df_test.to_pickle(output_dir / "test_full.pkl")
    stats_test.to_csv(output_dir / "test_stats.csv", index=False)
    plot_sample_trajectories(df_test, stats_test, output_dir, "test")
    plot_osnr_distribution(stats_test, output_dir, "test")
    plot_threshold_crossings(stats_test, output_dir, "test")

    # -- TRAIN set (optional) --
    stats_train = None
    if args.train and os.path.exists(args.train):
        df_train = load_mendeley_file(args.train)
        df_train = identify_trajectories(df_train)
        stats_train = characterize_trajectories(df_train)
        df_train.to_pickle(output_dir / "train_full.pkl")
        stats_train.to_csv(output_dir / "train_stats.csv", index=False)
        plot_sample_trajectories(df_train, stats_train, output_dir, "train")
        plot_osnr_distribution(stats_train, output_dir, "train")
        plot_threshold_crossings(stats_train, output_dir, "train")
    elif args.train:
        print(f"WARNING: train file not found: {args.train}")

    write_summary(stats_test, stats_train, output_dir)

    print(f"\nDone. All outputs in: {output_dir}/")
    print("\n>>> SEND BACK: summary.txt AND the 3 characterization_*.png plots <<<")


if __name__ == "__main__":
    main()