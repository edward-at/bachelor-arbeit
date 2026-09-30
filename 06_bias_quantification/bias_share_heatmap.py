"""
bias_share_heatmap.py  —  Section 4.8 figure (the share matrix, exact values)
════════════════════════════════════════════════════════════════════════════
The per-dataset regime-share matrix printed cell by cell: rows are the
datasets, ordered by the RQ1 headline metric (low-dynamic share, descending),
columns are the five regimes, and every cell carries the exact percentage.

Role among the three Section 4.8 views of the ONE share matrix:
  * bias_stacked_bar.py  — the ranking and the pooled/equal-weight gap
  * bias_regime_grid.py  — each dataset's profile SHAPE against the uniform 20 %
  * this figure          — the exact per-cell percentages, which stacked
                           segments resist comparing across datasets

Rows sum to 100 % (dataset composition). The transposed normalisation — who
SUPPLIES each regime — is bias_regime_supply.py, and the two are never
confused: one is a row-normalised composition, the other a column-normalised
provenance.

Reads : ARTIFACT_DIR/bias_per_dataset.parquet
Output: 06_bias_quantification/figures/bias_share_heatmap.png

Strict matplotlib: pure defaults, dpi=300. Single small in-cell font size —
the documented heatmap exception.
════════════════════════════════════════════════════════════════════════════
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from regime_style import N_REGIMES, regime_label
from bias_quant import CONFOUNDED_ACCEL

# ── paths and constants ──────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parents[1]))
from thesis_paths import DATA_ROOT, REPO_ROOT  # noqa: E402
ARTIFACT_DIR = DATA_ROOT / "regime_data"
# bias-quantification code + figures live under 06_bias_quantification/
OUT_DIR      = REPO_ROOT / "06_bias_quantification" / "figures"

DISPLAY_NAMES = {
    "boreas_rt": "Boreas", "durlar": "DurLAR", "ford": "Ford", "kitti": "KITTI",
    "kitti360": "KITTI-360", "nuscene": "nuScenes", "nvidia": "NVIDIA",
    "oxford": "Oxford", "waymo": "Waymo", "zod": "ZOD",
    "4seasons": "4Seasons", "hdd": "HDD", "a2d2": "A2D2", "kitscenes": "KITScenes",
}
DPI      = 300
CELL_FS  = 8      # documented heatmap exception: in-cell annotation font

SHARE_COLS = [f"share_{i}" for i in range(N_REGIMES)]


def plot_share_heatmap(per, out_dir):
    order = per.sort_values("low_dynamic_share", ascending=False).index
    names = [DISPLAY_NAMES.get(ds, ds) + ("*" if ds in CONFOUNDED_ACCEL else "")
             for ds in order]
    M = np.array([[per.loc[ds, c] * 100.0 for c in SHARE_COLS] for ds in order])

    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(M, cmap="Blues", aspect="auto", vmin=0.0, vmax=float(M.max()))

    ax.set_xticks(range(N_REGIMES))
    ax.set_xticklabels([regime_label(i) for i in range(N_REGIMES)],
                       rotation=30, ha="right")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(names)
    ax.set_title("Regime share (%) per dataset ")

    thresh = 0.55 * M.max()
    for r in range(len(order)):
        for c in range(N_REGIMES):
            ax.text(c, r, f"{M[r, c]:.0f}", ha="center", va="center",
                    fontsize=CELL_FS,
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
    plot_share_heatmap(per, OUT_DIR)


if __name__ == "__main__":
    main()