"""
bias_balance_metrics.py  —  Chapter 10 per-dataset scalar metrics
════════════════════════════════════════════════════════════════════════════
Three horizontal-bar panels, datasets on a shared y-axis ordered by low-dynamic
share:
  (1) directional low-dynamic share  — RQ1 headline number (reference-free)
  (2) normalized Shannon entropy      — internal balance (ref = uniform)
  (3) Jensen–Shannon distance to pooled — deviation from the field consensus
Bars coloured by the fixed per-dataset colour for cross-figure consistency.

Reads : ARTIFACT_DIR/bias_per_dataset.parquet
Output: figures/bias_quant/bias_balance_metrics.png
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

PANELS = [
    ("low_dynamic_share", "Low-dynamic share", (0, 1)),
    ("entropy_norm",      "Normalized entropy", (0, 1)),
    ("js_to_pooled",      "JS distance to pooled", None),
]


def plot_balance(per, profiles, out_dir):
    order = per.sort_values("low_dynamic_share", ascending=False).index
    y = np.arange(len(order))[::-1]
    colors = [DATASET_COLORS.get(ds, DATASET_COLOR_FALLBACK) for ds in order]
    names = [DISPLAY_NAMES.get(ds, ds) + ("*" if ds in CONFOUNDED_ACCEL else "")
             for ds in order]

    fig, axes = plt.subplots(1, 3, figsize=(14, 7), sharey=True)
    for ax, (col, title, xlim) in zip(axes, PANELS):
        vals = [per.loc[ds, col] for ds in order]
        ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)
        if xlim is not None:
            ax.set_xlim(*xlim)
    # descriptive pooled comparator on the directional panel (§4.7.3 wording)
    pooled_low = float(profiles.loc["pooled", "low_dynamic_share"])
    axes[0].axvline(pooled_low, color=TUM_SECONDARY["gray"], linewidth=1.4,
                    label="pooled corpus")
    axes[0].legend(loc="lower right")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)

    fig.suptitle("Per-dataset balance and distance metrics (* = accel-confounded)")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bias_balance_metrics.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  [SAVED] {out_path}")


def main():
    per = pd.read_parquet(ARTIFACT_DIR / "bias_per_dataset.parquet")
    profiles = pd.read_parquet(ARTIFACT_DIR / "bias_corpus_profiles.parquet")
    plot_balance(per, profiles, OUT_DIR)


if __name__ == "__main__":
    main()