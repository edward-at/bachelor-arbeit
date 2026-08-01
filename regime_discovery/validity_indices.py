"""
validity_indices.py  —  Section 4.6.7 figure
Reads validity_indices.parquet → Davies–Bouldin / Calinski–Harabasz / gap vs k,
with the silhouette-selected k marked, so the reader can see whether the three
criteria corroborate it. Thin reader; no recomputation.

Panel order follows the methodology: the two corroborating compactness indices
first (Davies–Bouldin, Calinski–Harabasz), the gap statistic last — it is
reported with its known uniform-null caveat rather than as a co-equal selector,
so it reads as the disclosed dissenter rather than as a third vote.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from tum_colors import TUM_ACCENT, TUM_EXT

PROJECT_ROOT    = Path("/Users/edward/Downloads")
REGIME_DATA_DIR = PROJECT_ROOT / "dataset" / "regime_data"
OUT_DIR         = PROJECT_ROOT / "bachelor-arbeit" / "regime_discovery" / "figures" / "validity"
DPI             = 300

# methodology order: corroborating indices first, gap (disclosed dissenter) last
PANELS = [("davies_bouldin", "Davies–Bouldin index (lower = better)"),
          ("calinski_harabasz", "Calinski–Harabasz index (higher = better)"),
          ("gap", "Gap statistic")]


def main():
    df = pd.read_parquet(REGIME_DATA_DIR / "validity_indices.parquet")
    curves = pd.read_parquet(REGIME_DATA_DIR / "kselect_curves.parquet")
    k_sel = int(curves.loc[curves["silhouette"].idxmax(), "k"])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, (col, name) in zip(axes, PANELS):
        ax.plot(df["k"], df[col], color=TUM_ACCENT["blue"], linewidth=1.8)
        ax.axvline(k_sel, color=TUM_EXT["red"], alpha=0.7, label=f"selected k = {k_sel}")
        ax.set_xlabel("Number of regimes k")
        ax.set_ylabel(name)
        ax.grid(True); ax.set_axisbelow(True); ax.legend()
    fig.suptitle("Internal validity indices across the regime-count sweep")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "validity_indices.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  → {out}")


if __name__ == "__main__":
    main()