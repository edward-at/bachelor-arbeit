"""
pca_scatter_preclustering.py  —  Section 4.6.3 figure
────────────────────────────────────────────────────────────────────────────
The windows in the 2-D PCA plane BEFORE any labels are imposed — the structure
the clustering must recover. The view is cropped to the central 98 % of windows
so the dense region is legible (sparse outliers beyond are omitted — state this
in the caption). A fixed-seed subsample is drawn for rendering only.

Thin reader of dataset/regime_data/cluster_assignments.parquet — no model file.
────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tum_colors import TUM_SECONDARY

PROJECT_ROOT    = Path("/Users/edward/Downloads")
REGIME_DATA_DIR = PROJECT_ROOT / "dataset" / "regime_data"
OUT_DIR         = PROJECT_ROOT / "bachelor-arbeit" / "regime_discovery" / "figures" / "pca"

N_PLOT   = 30000       # subsample for rendering only (clustering uses all windows)
SEED     = 42
CROP_PCT = (1, 99)     # zoom to this percentile range per axis
DPI      = 300


def main():
    df = pd.read_parquet(REGIME_DATA_DIR / "cluster_assignments.parquet", columns=["pc1", "pc2"])
    # variance shares from the pipeline artefact — never hard-coded, cannot go stale
    evr = pd.read_parquet(REGIME_DATA_DIR / "pca_variance.parquet")["explained_var_ratio"]
    var2_pct = 100.0 * float(evr.iloc[:2].sum())

    # data-driven zoom: crop to the central CROP_PCT of windows, with a little padding
    x_lo, x_hi = np.percentile(df["pc1"], CROP_PCT)
    y_lo, y_hi = np.percentile(df["pc2"], CROP_PCT)
    px, py = 0.05 * (x_hi - x_lo), 0.05 * (y_hi - y_lo)

    s = df.sample(min(N_PLOT, len(df)), random_state=SEED)

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    # `s` (marker size) is the single deviation from the strict style: a
    # default-size scatter of tens of thousands of points is unreadable. Minimal.
    ax.scatter(s["pc1"], s["pc2"], color=TUM_SECONDARY["blue2"], alpha=0.4, s=4, edgecolor="none")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"Windows in PCA space (before clustering)\n"
                 f"PC1+PC2 capture {var2_pct:.1f}% of the 5-D feature variance")
    ax.set_xlim(x_lo - px, x_hi + px)
    ax.set_ylim(y_lo - py, y_hi + py)
    ax.grid(True); ax.set_axisbelow(True)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "pca_scatter_preclustering.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  → {out}")


if __name__ == "__main__":
    main()