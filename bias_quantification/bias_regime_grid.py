"""
bias_regime_grid.py  —  Chapter 10 proposal-literal figure
════════════════════════════════════════════════════════════════════════════
The thesis proposal specifies: "the distribution of clips across dynamic
regimes is counted per dataset and visualized as bar charts." This is that
figure, verbatim (at window grain per DC3): 14 small-multiple panels, one per
dataset, five regime bars each in the canonical regime colours. The stacked
headline shows the ranking story; this grid shows each dataset's PROFILE SHAPE,
which stacked segments make hard to compare.

A thin grey line marks the uniform share (1/N = 20%) — the entropy metric's
balance reference (a methodological convention, §6.6), so over/under-filled
regimes are readable per panel.

Reads : ARTIFACT_DIR/bias_per_dataset.parquet
Output: figures/bias_regime_grid.png
════════════════════════════════════════════════════════════════════════════
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tum_colors import TUM_PRIMARY, TUM_SECONDARY, TUM_ACCENT, TUM_EXT
from regime_style import N_REGIMES, regime_label, regime_color
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
NROWS, NCOLS = 2, 7
UNIFORM_PCT = 100.0 / N_REGIMES  # entropy's balance reference (20%)


def _ylim_top(per):
    """Dynamic ceiling: largest share + 5pp headroom, rounded up to 5.
    (A hardcoded 70 clipped A2D2's real 79% Calm bar — never again.)"""
    m = float(per[SHARE_COLS].values.max()) * 100.0
    return min(100.0, np.ceil((m + 5.0) / 5.0) * 5.0)


def plot_grid(per, out_dir):
    order = per.sort_values("low_dynamic_share", ascending=False).index
    ymax = _ylim_top(per)
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(16, 6.5), sharey=True)

    for k, ds in enumerate(order):
        ax = axes[k // NCOLS][k % NCOLS]
        vals = [per.loc[ds, c] * 100 for c in SHARE_COLS]
        ax.bar(range(N_REGIMES), vals,
               color=[regime_color(i) for i in range(N_REGIMES)],
               edgecolor="black", linewidth=0.5)
        ax.axhline(UNIFORM_PCT, color=TUM_SECONDARY["gray"], linewidth=1.0, alpha=0.8)
        name = DISPLAY_NAMES.get(ds, ds) + ("*" if ds in CONFOUNDED_ACCEL else "")
        ax.set_title(name)
        ax.set_xticks(range(N_REGIMES))
        ax.set_xticklabels([str(i) for i in range(N_REGIMES)])
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel("share of windows (%)")
    axes[1][0].set_ylabel("share of windows (%)")

    key = " · ".join(f"{i} {regime_label(i)}" for i in range(N_REGIMES))
    fig.suptitle(f"Per-dataset regime distributions (* = accel-confounded)\nRegimes: {key}")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bias_regime_grid.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  [SAVED] {out_path}")


def main():
    per = pd.read_parquet(ARTIFACT_DIR / "bias_per_dataset.parquet")
    plot_grid(per, OUT_DIR)


if __name__ == "__main__":
    main()