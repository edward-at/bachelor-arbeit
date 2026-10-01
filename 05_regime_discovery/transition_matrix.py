"""
transition_matrix.py  —  Section 4.7.11 figure
Reads hmm_transitions.parquet → the regime-to-regime transition-probability
matrix as a Blues heatmap with per-cell annotations; the diagonal is the
self-transition (persistence). Rows = from regime, columns = to regime. The
in-cell font and white-on-dark text are the documented heatmap latitude. Thin
reader.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from regime_style import regime_label

sys.path.append(str(Path(__file__).resolve().parents[1]))
from thesis_paths import DATA_ROOT, REPO_ROOT  # noqa: E402
REGIME_DATA_DIR = DATA_ROOT / "regime_data"
OUT_DIR         = REPO_ROOT / "05_regime_discovery" / "figures" / "hmm"
DPI             = 300
ANNOT_FS        = 9        # documented heatmap font latitude


def main():
    T = pd.read_parquet(REGIME_DATA_DIR / "hmm_transitions.parquet").to_numpy()
    k = T.shape[0]
    labels = [regime_label(r) for r in range(k)]

    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    im = ax.imshow(T, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(k)); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(k)); ax.set_yticklabels(labels)
    ax.set_xlabel("To regime")
    ax.set_ylabel("From regime")
    ax.set_title("Regime transition probabilities (HMM)")
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{T[i, j]:.2f}", ha="center", va="center",
                    color=("white" if T[i, j] > 0.5 else "black"), fontsize=ANNOT_FS)
    fig.colorbar(im, ax=ax, fraction=0.046, label="transition probability")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "transition_matrix.png"
    # bbox_inches="tight": the equal-aspect image beside the colorbar leaves the
    # y label outside the canvas under tight_layout alone
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


if __name__ == "__main__":
    main()