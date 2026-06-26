"""
pca_scatter_labelled.py  —  Section 4.6.5 figure
────────────────────────────────────────────────────────────────────────────
The same windows as the pre-clustering scatter, now coloured by their assigned
regime — the partition the clustering recovered, shown against the structure of
the previous figure. The same fixed-seed subsample is used so the before/after
pair is drawn on identical points. Colours are the ordered regime palette
(0 = calmest → most dynamic).

Thin reader of dataset/regime_data/cluster_assignments.parquet — no model JSON.
────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from tum_colors import TUM_ACCENT, TUM_EXT

PROJECT_ROOT    = Path("/Users/edward/Downloads")
REGIME_DATA_DIR = PROJECT_ROOT / "dataset" / "regime_data"
OUT_DIR         = PROJECT_ROOT / "bachelor-arbeit" / "regime_discovery" / "figures" / "pca"
N_PLOT          = 30000      # same subsample size + seed as the pre-clustering scatter
SEED            = 42
DPI             = 300

# Ordered regime palette: regime 0 (calmest) → highest index (most dynamic).
REGIME_COLORS = [TUM_ACCENT["blue"], TUM_ACCENT["green"], TUM_ACCENT["orange"],
                 TUM_EXT["purple"], TUM_EXT["red"]]


def main():
    df = pd.read_parquet(REGIME_DATA_DIR / "cluster_assignments.parquet",
                         columns=["pc1", "pc2", "regime"])
    s = df.sample(min(N_PLOT, len(df)), random_state=SEED)

    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    for r in sorted(s["regime"].unique()):
        g = s[s["regime"] == r]
        # `s` (marker size) — same single deviation as the pre-clustering scatter.
        ax.scatter(g["pc1"], g["pc2"], color=REGIME_COLORS[int(r) % len(REGIME_COLORS)],
                   alpha=0.30, s=4, edgecolor="none", label=f"Regime {int(r)}")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Dynamic regimes in PCA space")
    ax.grid(True); ax.set_axisbelow(True); ax.legend()

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "pca_scatter_labelled.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  → {out}")


if __name__ == "__main__":
    main()