"""
pca_scatter_3d.py  —  Section 4.6.3/4.6.5 figure (3-D companion view)
────────────────────────────────────────────────────────────────────────────
The SAME single PCA as the 2-D scatter, shown with three components instead of
two. This is NOT a separate pipeline stage and the 2-D plot is NOT a projection
"of" this one — both are independent views of the full 5-D feature space. PC1
and PC2 are identical whether 2 or 3 components are kept; the third axis only
adds the next slice of variance. The 2-D view is for the page (a printed 3-D
scatter is a single fixed camera angle and can mislead); this 3-D view is for
interactive inspection (set SHOW = True and rotate).

Mirrors the 2-D labelled scatter's palette and names so the figures are
consistent. The legend uses separate full-opacity, normal-size proxy markers so
the colours are clear (legend handles otherwise inherit the faint, tiny scatter
markers). A fixed-seed subsample is drawn for rendering only.

Thin reader of dataset/regime_data/cluster_assignments.parquet (needs the pc3
column — re-run clustering.py after the 3-component edit). No model file.
────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
from tum_colors import TUM_SECONDARY, TUM_ACCENT, TUM_EXT

PROJECT_ROOT    = Path("/Users/edward/Downloads")
REGIME_DATA_DIR = PROJECT_ROOT / "dataset" / "regime_data"
OUT_DIR         = PROJECT_ROOT / "bachelor-arbeit" / "regime_discovery" / "figures" / "pca"

N_PLOT                = 30000      # subsample for rendering only (clustering uses all windows)
SEED                  = 42
CROP_PCT              = (1, 99)    # zoom to this percentile range per axis
EXPLAINED_VAR_3D_PCT  = 78.7       # PC1-PC3 share — from clustering.py
VIEW_ELEV             = 18         # camera elevation for the saved PNG
VIEW_AZIM             = -60        # camera azimuth for the saved PNG
SHOW                  = False      # True → open an interactive window to rotate
DPI                   = 300

# Must match pca_scatter_labelled.py exactly so the figures stay consistent.
REGIME_LABELS = ["Calm/Stationary", "Highway Cruising", "Turning", "Braking", "Accelerating"]
REGIME_COLORS = [TUM_SECONDARY["blue2"], TUM_EXT["maroon"], TUM_EXT["orange"],
                 TUM_EXT["dark_green"], TUM_EXT["purple"]]


def main():
    df = pd.read_parquet(REGIME_DATA_DIR / "cluster_assignments.parquet",
                         columns=["pc1", "pc2", "pc3", "regime"])

    # data-driven zoom on all three axes
    lims = {c: np.percentile(df[c], CROP_PCT) for c in ("pc1", "pc2", "pc3")}

    s = df.sample(min(N_PLOT, len(df)), random_state=SEED)

    fig = plt.figure(figsize=(8.0, 7.0))
    ax = fig.add_subplot(projection="3d")
    for r in sorted(s["regime"].unique()):
        g = s[s["regime"] == r]
        # `s` (marker size) — same single deviation as the 2-D scatter.
        ax.scatter(g["pc1"], g["pc2"], g["pc3"],
                   color=REGIME_COLORS[int(r) % len(REGIME_COLORS)],
                   alpha=0.30, s=4, edgecolor="none")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    ax.set_title(f"Dynamic regimes in 3-D PCA space\n"
                 f"PC1-PC3 capture {EXPLAINED_VAR_3D_PCT:.1f}% of the 5-D feature variance")
    ax.set_xlim(*lims["pc1"]); ax.set_ylim(*lims["pc2"]); ax.set_zlim(*lims["pc3"])
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    # full-opacity, normal-size legend markers (decoupled from the faint scatter)
    handles = [Line2D([0], [0], marker="o", linestyle="none",
                      color=REGIME_COLORS[i], alpha=1.0,
                      label=REGIME_LABELS[i] if i < len(REGIME_LABELS) else f"Regime {i}")
               for i in sorted(s["regime"].astype(int).unique())]
    ax.legend(handles=handles)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "pca_scatter_3d.png"
    fig.savefig(out, dpi=DPI)
    print(f"  → {out}")
    if SHOW:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()