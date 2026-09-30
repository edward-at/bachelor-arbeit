"""
consensus_matrix.py  —  Section 4.6.7 figure
Reads consensus_matrix.parquet + consensus_rows.parquet → the ordered consensus
matrix as a Blues heatmap with regime block separators. A genuine partition is
block-diagonal: pairs within a regime co-assigned almost always, pairs across
regimes almost never. Thin reader.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from regime_style import regime_label

PROJECT_ROOT    = Path("/Users/edward/Downloads")
REGIME_DATA_DIR = PROJECT_ROOT / "dataset" / "regime_data"
OUT_DIR         = PROJECT_ROOT / "bachelor-arbeit" / "05_regime_discovery" / "figures" / "validity"
DPI             = 300


def main():
    C   = pd.read_parquet(REGIME_DATA_DIR / "consensus_matrix.parquet").to_numpy()
    reg = pd.read_parquet(REGIME_DATA_DIR / "consensus_rows.parquet")["regime"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    im = ax.imshow(C, cmap="Blues", vmin=0.0, vmax=1.0)

    # regime block separators + ticks at block centres
    bounds = np.where(np.diff(reg) != 0)[0] + 0.5
    for b in bounds:
        ax.axvline(b, color="black", linewidth=0.6, alpha=0.5)
        ax.axhline(b, color="black", linewidth=0.6, alpha=0.5)
    uniq = sorted(np.unique(reg))
    centres = [float(np.mean(np.where(reg == r)[0])) for r in uniq]
    ax.set_xticks(centres); ax.set_xticklabels([regime_label(int(r)) for r in uniq],
                                               rotation=30, ha="right")
    ax.set_yticks(centres); ax.set_yticklabels([regime_label(int(r)) for r in uniq])
    ax.set_title("Consensus matrix (co-assignment frequency, ordered by regime)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="co-assignment frequency")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "consensus_matrix.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  → {out}")


if __name__ == "__main__":
    main()