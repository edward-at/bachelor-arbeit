"""
ecdf_analysis.py
────────────────────────────────────────────────────────────────────────────
ECDF overlays for all 6 kinematic signals.
One figure per signal, one curve per dataset on a shared axis.
Three vertical threshold lines partition each plot into 4 dynamic regime bands.

Output: figures/ecdf/ecdf_<signal_name>.png  (dpi=300)
Input:  std_data/<dataset>.parquet
────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path("/Users/edward/Downloads")
STD_DATA_DIR = PROJECT_ROOT / "dataset" / "std_data"
OUT_DIR      = PROJECT_ROOT /"bachelor-arbeit"/"inertial_exploratory_analysis"/ "figures" / "ecdf"

# ─────────────────────────────────────────────────────────────────────────────
# DATASETS — heatmap rank-sum order (most dynamic → least dynamic)
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = [
    "zod",
    "ford",
    "waymo",
    "oxford",
    "kitti",
    "kitti360",
    "nuscene",
    "durlar",
    "boreas_rt",
    "nvidia",
]

DATASET_DISPLAY = {
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
}

# ─────────────────────────────────────────────────────────────────────────────
# DATASET COLORS — one per dataset
# ─────────────────────────────────────────────────────────────────────────────

from tum_colors import TUM_PRIMARY, TUM_SECONDARY, TUM_EXT

DATASET_COLORS = {
    "boreas_rt": TUM_EXT['maroon'],
    "ford":      TUM_EXT['orange'],
    "kitti":     TUM_EXT['yellow'],
    "kitti360":  TUM_EXT['olive'],
    "nuscene":   TUM_EXT['dark_green'],
    "nvidia":    TUM_EXT['teal'],
    "oxford":    TUM_EXT['navy'],
    "waymo":     TUM_EXT['purple'],
}

# ─────────────────────────────────────────────────────────────────────────────
# SIGNALS — (parquet_col, filename_slug, x_axis_label, x_max)
# ─────────────────────────────────────────────────────────────────────────────

SIGNALS = [
    ("std_speed_kmh", "speed_kmh",  "Within-Window Std of Speed (km/h)",                  30.0),
    ("std_a_long",    "a_long",     "Within-Window Std of Long. Acceleration (m/s²)",       4.0),
    ("std_a_lat",     "a_lat",      "Within-Window Std of Lat. Acceleration (m/s²)",        3.0),
    ("std_jerk_long", "jerk_long",  "Within-Window Std of Long. Jerk (m/s³)",               6.0),
    ("std_jerk_lat",  "jerk_lat",   "Within-Window Std of Lat. Jerk (m/s³)",                4.0),
    ("std_yaw_rate",  "yaw_rate",   "Within-Window Std of Yaw Rate (°/s)",                 40.0),
]

# ─────────────────────────────────────────────────────────────────────────────
# REGIME THRESHOLDS — shared with regime_classification.py
# (t1, t2, t3): three boundaries defining four regimes
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLDS = {
    "std_speed_kmh": (1.0,  5.0,  15.0),
    "std_a_long":    (0.2,  0.5,   1.5),
    "std_a_lat":     (0.1,  0.3,   0.8),
    "std_jerk_long": (0.3,  0.8,   2.0),
    "std_jerk_lat":  (0.2,  0.5,   1.5),
    "std_yaw_rate":  (1.0,  5.0,  15.0),
}

REGIME_BAND_COLORS = ["#DDEEFF", "#D9EFD9", "#FFF3CC", "#FFE0CC"]
REGIME_LABELS      = ["Near-Stationary", "Low-Dynamic", "Medium-Dynamic", "High-Dynamic"]

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


def compute_ecdf(values: np.ndarray):
    v = values[~np.isnan(values)]
    if len(v) == 0:
        return np.array([]), np.array([])
    v_sorted = np.sort(v)
    ecdf_y   = np.arange(1, len(v_sorted) + 1) / len(v_sorted)
    return v_sorted, ecdf_y


# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_ecdf_signal(col, slug, x_label, x_max, data, out_dir):
    t1, t2, t3 = THRESHOLDS[col]
    boundaries  = [0.0, t1, t2, t3, x_max * 2]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Shaded regime bands
    for i, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        ax.axvspan(lo, hi, color=REGIME_BAND_COLORS[i], alpha=0.3)

    # Threshold vertical lines
    ax.axvline(t1, color="black", linewidth=1.0, linestyle="--")
    ax.axvline(t2, color="black", linewidth=1.0, linestyle="-.")
    ax.axvline(t3, color="black", linewidth=1.0, linestyle=":")

    # ECDF curves
    for ds in DATASETS:
        if ds not in data or col not in data[ds].columns:
            continue
        x, y = compute_ecdf(data[ds][col].values)
        if len(x) == 0:
            continue
        ax.plot(x, y, color=TUM_COLORS[ds], linewidth=1.8, label=DATASET_DISPLAY[ds])

    ax.set_xlim(left=0.0, right=x_max)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Cumulative Fraction of Windows")
    ax.set_title(f"ECDF — {x_label}")
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")

    plt.tight_layout()
    out_path = out_dir / f"ecdf_{slug}.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  [SAVED] {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading std_data …")
    data = load_std_data()
    print(f"\nGenerating ECDF plots …")
    for col, slug, x_label, x_max in SIGNALS:
        print(f"  {slug}")
        plot_ecdf_signal(col, slug, x_label, x_max, data, OUT_DIR)
    print(f"\nDone → {OUT_DIR}")


if __name__ == "__main__":
    main()