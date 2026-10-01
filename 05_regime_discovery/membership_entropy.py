"""
membership_entropy.py  —  Section 4.7.10 figure
Reads gmm_assignments.parquet → the windows on the PC1/PC2 plane coloured by GMM
membership entropy, so the boundary windows where regimes interpenetrate (high
entropy) stand out from windows that belong unambiguously to one regime.

Visibility: most windows have LOW entropy (light), so the few high-entropy points
get buried. They are therefore sorted to draw LAST (on top), and alpha/size are
raised, so the overlap regions are visible without abandoning the Blues
convention.
Thin reader.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
from thesis_paths import DATA_ROOT, REPO_ROOT  # noqa: E402
REGIME_DATA_DIR = DATA_ROOT / "regime_data"
OUT_DIR         = REPO_ROOT / "05_regime_discovery" / "figures" / "gmm"

N_PLOT   = 30000
SEED     = 42
CROP_PCT = (1, 99)
DPI      = 300


def main():
    df = pd.read_parquet(REGIME_DATA_DIR / "gmm_assignments.parquet",
                         columns=["pc1", "pc2", "entropy"])

    x_lo, x_hi = np.percentile(df["pc1"], CROP_PCT)
    y_lo, y_hi = np.percentile(df["pc2"], CROP_PCT)
    px, py = 0.05 * (x_hi - x_lo), 0.05 * (y_hi - y_lo)

    # sample, then sort ascending so HIGH-entropy points are drawn last (on top)
    s = df.sample(min(N_PLOT, len(df)), random_state=SEED).sort_values("entropy")

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    # `s` (marker size) — documented scatter latitude
    sc = ax.scatter(s["pc1"], s["pc2"], c=s["entropy"],
                    s=7, alpha=0.6, edgecolor="none", vmin=0.0, vmax=1.0)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Membership entropy (regime overlap on the PCA plane)")
    ax.set_xlim(x_lo - px, x_hi + px)
    ax.set_ylim(y_lo - py, y_hi + py)
    ax.grid(True); ax.set_axisbelow(True)
    fig.colorbar(sc, ax=ax, fraction=0.046, label="normalised membership entropy")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "membership_entropy.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  → {out}")


if __name__ == "__main__":
    main()