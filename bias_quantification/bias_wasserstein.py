"""
bias_wasserstein.py  —  Chapter 10 continuous-signal distance (Chapter 8 bridge)
════════════════════════════════════════════════════════════════════════════
Three horizontal-bar panels — one per Chapter-8 ECDF channel — of the
1-Wasserstein distance from each dataset's distribution to the POOLED
distribution, in physical units. This is the quantitative version of the
Chapter 8 ECDF overlays: Chapter 8 showed the curves differ; this says by how
much. Separate axes per channel (units differ; never a shared colour scale).

Reads : ARTIFACT_DIR/bias_per_dataset.parquet
Output: figures/bias_quant/bias_wasserstein.png
════════════════════════════════════════════════════════════════════════════
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tum_colors import TUM_PRIMARY, TUM_SECONDARY, TUM_ACCENT, TUM_EXT
from bias_quant import CONFOUNDED_ACCEL

# ── canonical constants block ────────────────────────────────────────────────
PROJECT_ROOT = Path("/Users/edward/Downloads")
ARTIFACT_DIR = PROJECT_ROOT / "dataset" / "regime_data"
# Chapter 10 code + figures live under bachelor-arbeit/bias_quantification/
OUT_DIR      = PROJECT_ROOT / "bachelor-arbeit" / "bias_quantification" / "figures"

DISPLAY_NAMES = {
    "boreas_rt": "Boreas", "durlar": "DurLAR", "ford": "Ford", "kitti": "KITTI",
    "kitti360": "KITTI-360", "nuscene": "nuScenes", "nvidia": "NVIDIA",
    "oxford": "Oxford", "waymo": "Waymo", "zod": "ZOD",
    "4seasons": "4Seasons", "hdd": "HDD", "a2d2": "A2D2", "kitscenes": "KITScenes",
}
DATASET_COLORS = {
    "4seasons":  TUM_SECONDARY["blue2"],
    "a2d2":      TUM_SECONDARY["dark_gray"],
    "hdd":       TUM_SECONDARY["gray"],
    "kitscenes": TUM_PRIMARY["black"],
    "boreas_rt": TUM_EXT["maroon"],
    "durlar":    TUM_SECONDARY["blue1"],
    "ford":      TUM_EXT["orange"],
    "kitti":     TUM_EXT["yellow"],
    "kitti360":  TUM_EXT["olive"],
    "nuscene":   TUM_EXT["dark_green"],
    "nvidia":    TUM_EXT["teal"],
    "oxford":    TUM_EXT["navy"],
    "waymo":     TUM_EXT["purple"],
    "zod":       TUM_EXT["red"],
}
DATASET_COLOR_FALLBACK = TUM_SECONDARY["blue2"]
DPI = 300

# (column, panel title incl. units)
PANELS = [
    ("w_std_speed_kmh", "Speed variability\n1-Wasserstein (km/h)"),
    ("w_std_acc_mag",   "Accel-magnitude variability\n1-Wasserstein (m/s²)"),
    ("w_std_yaw_rate",  "Yaw-rate variability\n1-Wasserstein (°/s)"),
]


def plot_wasserstein(per, out_dir):
    order = per.sort_values("low_dynamic_share", ascending=False).index
    y = np.arange(len(order))[::-1]
    colors = [DATASET_COLORS.get(ds, DATASET_COLOR_FALLBACK) for ds in order]
    names = [DISPLAY_NAMES.get(ds, ds) + ("*" if ds in CONFOUNDED_ACCEL else "")
             for ds in order]

    fig, axes = plt.subplots(1, 3, figsize=(14, 7), sharey=True)
    for ax, (col, title) in zip(axes, PANELS):
        vals = [per.loc[ds, col] for ds in order]
        ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)

    fig.suptitle("Per-channel 1-Wasserstein distance to the pooled reference")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bias_wasserstein.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  [SAVED] {out_path}")


def main():
    per = pd.read_parquet(ARTIFACT_DIR / "bias_per_dataset.parquet")
    plot_wasserstein(per, OUT_DIR)


if __name__ == "__main__":
    main()