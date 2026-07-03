"""
pca_scatter_labelled.py  —  Section 4.6.5 figure
────────────────────────────────────────────────────────────────────────────
The same windows as the pre-clustering scatter, coloured by assigned regime and
labelled with the behavioural names read from the regime profile. Same crop and
fixed-seed subsample as the pre-clustering scatter so the before/after pair is
drawn on identical points and the same view.

REGIME_LABELS are BEHAVIOURAL (cornering, braking, …), not tier-levels — the
clustering recovers behaviour types, not a low/medium/high ladder. Reword the
strings freely, but keep them behavioural. Index = regime number.

The scatter points are drawn at alpha=0.30, s=4 for density legibility; the
legend uses separate full-opacity, normal-size proxy markers so the colours are
clear (legend handles otherwise inherit the faint, tiny scatter markers).

Thin reader of dataset/regime_data/cluster_assignments.parquet — no model file.
────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tum_colors import TUM_SECONDARY, TUM_ACCENT, TUM_EXT

PROJECT_ROOT    = Path("/Users/edward/Downloads")
REGIME_DATA_DIR = PROJECT_ROOT / "dataset" / "regime_data"
OUT_DIR         = PROJECT_ROOT / "bachelor-arbeit" / "regime_discovery" / "figures" / "pca"

N_PLOT             = 30000       # same subsample size + seed as the pre-clustering scatter
SEED               = 42
CROP_PCT           = (1, 99)     # same zoom as the pre-clustering scatter
EXPLAINED_VAR_PCT  = 59.8        # PC1+PC2 variance share — from clustering.py (seed 42, fixed)
DPI                = 300

# Behavioural regime names, ordered 0..k-1 (0 = calmest). Read off the profile.
REGIME_LABELS = ["Calm/Stationary", "Highway Cruising", "Turning", "Braking", "Accelerating"]
# Ordered regime palette (calmest → most dynamic), TUM colours.
REGIME_COLORS = [TUM_SECONDARY["blue2"], TUM_EXT["maroon"], TUM_EXT["orange"],
                 TUM_EXT["dark_green"], TUM_EXT["purple"]]


def main():
    df = pd.read_parquet(REGIME_DATA_DIR / "cluster_assignments.parquet",
                         columns=["pc1", "pc2", "regime"])

    x_lo, x_hi = np.percentile(df["pc1"], CROP_PCT)
    y_lo, y_hi = np.percentile(df["pc2"], CROP_PCT)
    px, py = 0.05 * (x_hi - x_lo), 0.05 * (y_hi - y_lo)

    s = df.sample(min(N_PLOT, len(df)), random_state=SEED)

    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    for r in sorted(s["regime"].unique()):
        g = s[s["regime"] == r]
        # `s` (marker size) — same single deviation as the pre-clustering scatter.
        ax.scatter(g["pc1"], g["pc2"], color=REGIME_COLORS[int(r) % len(REGIME_COLORS)],
                   alpha=0.50, s=4, edgecolor="none")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"Dynamic regimes in PCA space\n"
                 f"PC1+PC2 capture {EXPLAINED_VAR_PCT:.1f}% of the 5-D feature variance")
    ax.set_xlim(x_lo - px, x_hi + px)
    ax.set_ylim(y_lo - py, y_hi + py)
    ax.grid(True); ax.set_axisbelow(True)

    # full-opacity, normal-size legend markers (decoupled from the faint scatter)
    handles = [Line2D([0], [0], marker="o", linestyle="none",
                      color=REGIME_COLORS[i], alpha=1.0,
                      label=REGIME_LABELS[i] if i < len(REGIME_LABELS) else f"Regime {i}")
               for i in sorted(s["regime"].astype(int).unique())]
    ax.legend(handles=handles)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "pca_scatter_labelled.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  → {out}")


if __name__ == "__main__":
    main()