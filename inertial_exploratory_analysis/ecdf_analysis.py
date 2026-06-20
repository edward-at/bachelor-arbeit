"""
ecdf_analysis.py  (v2)
────────────────────────────────────────────────────────────────────────────
ECDF overlays for all 6 kinematic signals.
One figure per signal, one curve per dataset on a shared axis.
Three vertical threshold lines partition each plot into 4 dynamic regime bands.

─── CHANGES vs v1 ──────────────────────────────────────────────────────────
  1. BUGFIX: plotting referenced an undefined TUM_COLORS dict → NameError on
     the first curve. Now uses DATASET_COLORS (with fallback).
  2. DATASET_COLORS completed: durlar (gold) and zod (red) were missing and
     would have collapsed onto the same fallback color in a 10-curve figure.
  3. NOTE: THRESHOLDS were calibrated on pre-v3 (noisier) std values. After
     the 10 Hz / 3 Hz-band-limited re-run, re-check them against the new
     distributions before locking figures (see regime_classification.py).

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
OUT_DIR      = PROJECT_ROOT / "bachelor-arbeit" / "inertial_exploratory_analysis" / "figures" / "ecdf"

# ─────────────────────────────────────────────────────────────────────────────
# DATASETS — heatmap rank-sum order (most dynamic → least dynamic)
# NOTE: re-derive this order from the new summary heatmap after the v3 re-run;
# the ranking was computed on pre-v3 data and may have changed.
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
# DATASET COLORS — one per dataset (must match all other thesis scripts)
# ─────────────────────────────────────────────────────────────────────────────

from tum_colors import TUM_PRIMARY, TUM_SECONDARY, TUM_ACCENT, TUM_EXT

DATASET_COLORS = {
    "4seasons":   TUM_SECONDARY['blue2'],
    "a2d2":      TUM_SECONDARY['dark_gray'],
    "hdd":       TUM_SECONDARY['gray'],
    "kitscenes": TUM_SECONDARY['light_gray'],
    "boreas_rt": TUM_EXT['maroon'],
    "durlar":    TUM_SECONDARY['blue1'],
    "ford":      TUM_EXT['orange'],
    "kitti":     TUM_EXT['yellow'],
    "kitti360":  TUM_EXT['olive'],
    "nuscene":   TUM_EXT['dark_green'],
    "nvidia":    TUM_EXT['teal'],
    "oxford":    TUM_EXT['navy'],
    "waymo":     TUM_EXT['purple'],
    "zod":       TUM_EXT['red'],
}
DATASET_COLOR_FALLBACK = TUM_SECONDARY['blue2']

# ─────────────────────────────────────────────────────────────────────────────
# SIGNALS — (parquet_col, filename_slug, x_axis_label, x_max)
# ─────────────────────────────────────────────────────────────────────────────

SIGNALS = [
    ("std_speed_kmh", "speed_kmh",  "Within-Window Std of Speed (km/h)",                  30.0),
    ("std_acc_mag",   "acc_mag",    "Within-Window Std of Accel. Magnitude (m/s²)",         4.0),
    ("std_yaw_rate",  "yaw_rate",   "Within-Window Std of Yaw Rate (°/s)",                 40.0),
]

# ─────────────────────────────────────────────────────────────────────────────
# REGIME THRESHOLDS — shared with regime_classification.py
# (t1, t2, t3): three boundaries defining four regimes
# CAUTION: calibrated on pre-v3 std values — re-check after the v3 re-run.
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLDS = {
    "std_speed_kmh": (1.0,  5.0,  15.0),
    "std_acc_mag":   (0.2,  0.5,   1.5),   # PLACEHOLDER (never calibrated) — keep in sync with regime_classification.py
    "std_yaw_rate":  (1.0,  5.0,  15.0),
}

# Light tints of the TUM regime colors (regime_classification.py)
REGIME_BAND_COLORS = [TUM_ACCENT["blue"], TUM_ACCENT["green"],
                      TUM_ACCENT["orange"], TUM_EXT["red"]]
REGIME_BAND_ALPHA  = 0.12
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
        ax.axvspan(lo, hi, color=REGIME_BAND_COLORS[i], alpha=REGIME_BAND_ALPHA)

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
        ax.plot(x,y,
                color=DATASET_COLORS.get(ds, DATASET_COLOR_FALLBACK),
                linewidth=1.8, label=DATASET_DISPLAY[ds])

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