"""
bias_share_heatmap.py  —  Chapter 10 precise companion to the headline
════════════════════════════════════════════════════════════════════════════
14×5 heatmap of regime shares (%), Blues, datasets ordered by low-dynamic share.
Exact per-cell values (stacked bars are for the story; this is for reading).

Reads : ARTIFACT_DIR/bias_per_dataset.parquet
Output: figures/bias_quant/bias_share_heatmap.png
════════════════════════════════════════════════════════════════════════════
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tum_colors import TUM_PRIMARY, TUM_SECONDARY, TUM_ACCENT, TUM_EXT
from regime_style import N_REGIMES, regime_label
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
DPI = 300

SHARE_COLS = [f"share_{i}" for i in range(N_REGIMES)]
ANNOT_FONTSIZE = 8   # documented in-cell annotation exception (thesis-plot skill)


def plot_heatmap(per, out_dir):
    order = per.sort_values("low_dynamic_share", ascending=False).index
    M = np.array([[per.loc[ds, c] * 100 for c in SHARE_COLS] for ds in order])

    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(M, cmap="Blues", aspect="auto", vmin=0, vmax=max(60, M.max()))

    ax.set_xticks(range(N_REGIMES))
    ax.set_xticklabels([regime_label(i) for i in range(N_REGIMES)], rotation=30, ha="right")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([DISPLAY_NAMES.get(d, d) + ("*" if d in CONFOUNDED_ACCEL else "")
                        for d in order])
    ax.set_title("Regime share (%) per dataset   (* = accel-confounded signal)")

    thresh = 0.55 * M.max()
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            ax.text(c, r, f"{M[r, c]:.0f}", ha="center", va="center",
                    fontsize=ANNOT_FONTSIZE,
                    color="white" if M[r, c] > thresh else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Share of 20 s windows (%)")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bias_share_heatmap.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  [SAVED] {out_path}")


def main():
    per = pd.read_parquet(ARTIFACT_DIR / "bias_per_dataset.parquet")
    plot_heatmap(per, OUT_DIR)


if __name__ == "__main__":
    main()