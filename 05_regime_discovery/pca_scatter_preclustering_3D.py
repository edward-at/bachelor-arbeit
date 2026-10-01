"""
pca_scatter_preclustering_3D.py  —  Sections 4.7.3–4.7.4 figure (3-D companion view)
────────────────────────────────────────────────────────────────────────────
The windows in the 3-D PCA space BEFORE any labels are imposed — the same
unlabelled structure as the 2-D pre-clustering scatter, shown with three
components instead of two. This is NOT a separate pipeline stage: PC1 and PC2
are identical whether two or three components are kept; the third axis only adds
the next slice of variance, in which structure that overlaps in the plane is
resolved.

Pairs with pca_scatter_labeled_3D.py (the labelled 3-D view of Section 4.7.6): same
subsample seed, same crop, same camera angle, so the before/after pair is drawn
on identical points from an identical viewpoint and only the colouring differs.
The 2-D view remains the printed figure (a printed 3-D scatter fixes a single
camera angle and can mislead); this 3-D view is for interactive inspection
(set SHOW = True and rotate).

Thin reader of dataset/regime_data/cluster_assignments.parquet (needs the pc3
column). The regime column is deliberately NOT read — no labels exist at this
point in the chapter. No model file.
────────────────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
from tum_colors import TUM_SECONDARY

sys.path.append(str(Path(__file__).resolve().parents[1]))
from thesis_paths import DATA_ROOT, REPO_ROOT  # noqa: E402
REGIME_DATA_DIR = DATA_ROOT / "regime_data"
OUT_DIR         = REPO_ROOT / "05_regime_discovery" / "figures" / "pca"

N_PLOT                = 30000      # subsample for rendering only (clustering uses all windows)
SEED                  = 42         # same subsample as the labelled 3-D view
CROP_PCT              = (1, 99)    # zoom to this percentile range per axis
VIEW_ELEV             = 18         # same camera as the labelled 3-D view
VIEW_AZIM             = -60        # same camera as the labelled 3-D view
SHOW                  = False      # True → open an interactive window to rotate
DPI                   = 300


def main():
    df = pd.read_parquet(REGIME_DATA_DIR / "cluster_assignments.parquet",
                         columns=["pc1", "pc2", "pc3"])
    # variance shares from the pipeline artefact — never hard-coded, cannot go stale
    evr = pd.read_parquet(REGIME_DATA_DIR / "pca_variance.parquet")["explained_var_ratio"]
    var3_pct = 100.0 * float(evr.iloc[:3].sum())

    # data-driven zoom on all three axes
    lims = {c: np.percentile(df[c], CROP_PCT) for c in ("pc1", "pc2", "pc3")}

    s = df.sample(min(N_PLOT, len(df)), random_state=SEED)

    fig = plt.figure(figsize=(8.0, 7.0))
    ax = fig.add_subplot(projection="3d")
    # `s` (marker size) — same single deviation as the other PCA scatters.
    ax.scatter(s["pc1"], s["pc2"], s["pc3"],
               color=TUM_SECONDARY["blue2"], alpha=0.30, s=4, edgecolor="none")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    ax.set_title(f"Windows in 3-D PCA space (before clustering)\n"
                 f"PC1-PC3 capture {var3_pct:.1f}% of the 5-D feature variance")
    ax.set_xlim(*lims["pc1"]); ax.set_ylim(*lims["pc2"]); ax.set_zlim(*lims["pc3"])
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "pca_scatter_preclustering_3d.png"
    fig.savefig(out, dpi=DPI)
    print(f"  → {out}")
    if SHOW:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()