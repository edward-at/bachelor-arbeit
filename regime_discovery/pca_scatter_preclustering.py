"""
pca_scatter_preclustering.py  —  Section 4.6.3 figure
────────────────────────────────────────────────────────────────────────────
The windows in the 2-D PCA plane BEFORE any labels are imposed — the structure
the clustering must recover, shown independently of the algorithm that will
partition it. A fixed-seed subsample is drawn for rendering only; the partition
itself uses every window. The two displayed components explain the variance
reported in Section 4.6 — state that figure in the caption.

Thin reader of dataset/regime_data/cluster_assignments.parquet — no model JSON.
────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from tum_colors import TUM_SECONDARY

PROJECT_ROOT    = Path("/Users/edward/Downloads")
REGIME_DATA_DIR = PROJECT_ROOT / "dataset" / "regime_data"
OUT_DIR         = PROJECT_ROOT / "bachelor-arbeit" / "regime_discovery" / "figures" / "pca"
N_PLOT          = 30000      # subsample for rendering only (clustering uses all windows)
SEED            = 42
DPI             = 300


def main():
    df = pd.read_parquet(REGIME_DATA_DIR / "cluster_assignments.parquet", columns=["pc1", "pc2"])
    s  = df.sample(min(N_PLOT, len(df)), random_state=SEED)

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    # `s` (marker size) is the single deviation from the strict style: a
    # default-size scatter of tens of thousands of points is unreadable. Minimal.
    ax.scatter(s["pc1"], s["pc2"], color=TUM_SECONDARY["blue2"], alpha=0.25, s=4, edgecolor="none")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Windows in PCA space (before clustering)")
    ax.grid(True); ax.set_axisbelow(True)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "pca_scatter_preclustering.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  → {out}")


if __name__ == "__main__":
    main()