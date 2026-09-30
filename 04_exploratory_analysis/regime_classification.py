"""
regime_classification.py  (variability tiers, thesis Section 4.5.6 / 4.6.8)
────────────────────────────────────────────────────────────────────────────
Sorts every analysis window into one of four VARIABILITY TIERS using fixed
absolute thresholds on within-window standard deviations.
One stacked proportion bar chart per signal (3 total).

Tiers: near-stationary | low-dynamic | medium-dynamic | high-dynamic

These are a descriptive, single-signal device (fixed cuts on one variability
signal) — NOT the multivariate behavioural regimes recovered by clustering in
the regime-discovery stage (RQ2). The term "regime" is reserved for those; here
everything is a "tier". (The script filename and the figures/regime output
folder are kept unchanged for path stability.)

Design:
  1. Tier colors from the official TUM palette.
  2. Per-segment label color chosen for contrast (black on green/orange).
  3. Legend outside the axes, so it cannot overlap the right-most bars.
  4. THRESHOLDS are frozen: a stated round convention, not a calibrated
     quantity (lineage in Section 4.5.6, limitation in Section 6.6).

Output: figures/regime/regime_<signal_name>.png  (dpi=300)
Input:  std_data/<dataset>.parquet
────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

sys.path.append(str(Path(__file__).resolve().parents[1]))
from thesis_paths import DATA_ROOT, REPO_ROOT  # noqa: E402
STD_DATA_DIR = DATA_ROOT / "std_data"
OUT_DIR      = REPO_ROOT/"04_exploratory_analysis"/ "figures" / "regime"

# ─────────────────────────────────────────────────────────────────────────────
# DATASETS — fixed display order; NOT auto-synced to the heatmap rank-sum order.
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = [
    "4seasons", "a2d2", "hdd", "boreas_rt", "durlar", "ford", "kitti", "kitti360",
    "nuscene", "nvidia", "oxford", "waymo", "zod", "kitscenes",
]

DATASET_DISPLAY = {
    "4seasons":   "4 Seasons",
    "a2d2":      "A2D2",
    "hdd":       "HDD",
    "boreas_rt": "Boreas",
    "durlar":    "DurLAR",
    "ford":      "Ford",
    "kitti":     "KITTI",
    "kitti360":  "KITTI-360",
    "nuscene":   "nuScenes",
    "nvidia":    "NVIDIA",
    "oxford":    "Oxford",
    "waymo":     "Waymo",
    "zod":       "ZOD",
    "kitscenes": "KITScenes",
}

# ─────────────────────────────────────────────────────────────────────────────
# SIGNALS — (parquet_col, filename_slug, plot_title)
# ─────────────────────────────────────────────────────────────────────────────

SIGNALS = [
    ("std_speed_kmh", "speed_kmh",  "Within-Window Std of Speed (km/h)"),
    ("std_acc_mag",   "acc_mag",    "Within-Window Std of Accel. Magnitude (m/s²)"),
    ("std_yaw_rate",  "yaw_rate",   "Within-Window Std of Yaw Rate (°/s)"),
]

# ─────────────────────────────────────────────────────────────────────────────
# TIER THRESHOLDS — ecdf_analysis.py and time_series_analysis.py keep their own
# copies of these values (not imported); keep them identical by hand.
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLDS = {
    "std_speed_kmh": (1.0,  5.0,  15.0),
    "std_acc_mag":   (0.2,  0.5,   1.5),   # Frozen: a stated round convention, not a calibrated quantity. Lineage disclosed in Section 4.5.6, limitation in Section 6.6.
    "std_yaw_rate":  (1.0,  5.0,  15.0),
}

# ─────────────────────────────────────────────────────────────────────────────
# TIER COLORS AND LABELS
# ─────────────────────────────────────────────────────────────────────────────

from tum_colors import TUM_ACCENT, TUM_EXT

TIER_COLORS      = [TUM_ACCENT["blue"],   # Near-Stationary  #64A0C8
                    TUM_ACCENT["green"],  # Low-Dynamic      #A2AD00
                    TUM_ACCENT["orange"], # Medium-Dynamic   #F6BA00
                    TUM_EXT["red"]]       # High-Dynamic     #C4071B
TIER_TEXT_COLORS = ["black", "black", "black", "white"]
TIER_LABELS = ["Near-Stationary", "Low-Dynamic", "Medium-Dynamic", "High-Dynamic"]

# Minimum segment height (percentage points) to print a label inside the bar
MIN_LABEL_PCT = 6.0

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_std_data() -> dict:
    data = {}
    for ds in DATASETS:
        path = STD_DATA_DIR / f"{ds}.parquet"
        if not path.exists():
            print(f"  [WARN] not found: {path}")
            continue
        df = pd.read_parquet(path)
        if "std_yaw_rate" in df.columns:
            df["std_yaw_rate"] = df["std_yaw_rate"] * (180.0 / np.pi)
        data[ds] = df
        print(f"  [OK]   {ds}: {len(df):,} windows")
    return data


def tier_fractions(values: np.ndarray, t1, t2, t3) -> list:
    v = values[~np.isnan(values)]
    if len(v) == 0:
        return [float("nan")] * 4
    n = len(v)
    return [
        100.0 * np.sum(v <  t1) / n,
        100.0 * np.sum((v >= t1) & (v < t2)) / n,
        100.0 * np.sum((v >= t2) & (v < t3)) / n,
        100.0 * np.sum(v >= t3) / n,
    ]


# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_tier_signal(col, slug, title, data, out_dir):
    t1, t2, t3 = THRESHOLDS[col]

    present = [ds for ds in DATASETS if ds in data and col in data[ds].columns]
    if not present:
        print(f"  [SKIP] no data for {col}")
        return

    fracs = {ds: tier_fractions(data[ds][col].values, t1, t2, t3) for ds in present}
    x_pos   = np.arange(len(present))
    x_labels = [DATASET_DISPLAY[ds] for ds in present]

    fig, ax = plt.subplots(figsize=(12, 6))

    bottoms = np.zeros(len(present))
    for r in range(4):
        heights = np.array([fracs[ds][r] for ds in present])
        bars = ax.bar(
            x_pos, heights, bottom=bottoms,
            color=TIER_COLORS[r], edgecolor="white",
            linewidth=0.6, label=TIER_LABELS[r],
        )
        # Percentage label inside segment
        for i, (bar, h) in enumerate(zip(bars, heights)):
            if np.isnan(h) or h < MIN_LABEL_PCT:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bottoms[i] + h / 2.0,
                f"{h:.1f}%",
                ha="center", va="center",
                color=TIER_TEXT_COLORS[r],
            )
        bottoms += heights

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Proportion of Windows (%)")
    ax.set_title(f"Variability-Tier Distribution {title}")
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True)

    plt.tight_layout()
    out_path = out_dir / f"regime_{slug}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading std_data …")
    data = load_std_data()
    print(f"\nGenerating variability-tier classification plots …")
    for col, slug, title in SIGNALS:
        print(f"  {slug}")
        plot_tier_signal(col, slug, title, data, OUT_DIR)
    print(f"\nDone → {OUT_DIR}")


if __name__ == "__main__":
    main()