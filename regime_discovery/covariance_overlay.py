"""
covariance_overlay.py  —  Section 4.6.8 figure
Reads gmm_components_2d.parquet (projected means + covariances) and
gmm_assignments.parquet (background windows) → each regime's 1σ covariance
ellipse on the PC1/PC2 plane, plus a mean-marker dot per regime. 1σ (not 2σ)
keeps the ellipses from overlapping into an unreadable tangle, and the mean dots
make every component visible — including the calm/stationary component, which
the mixture shrinks almost to a point and whose ellipse would otherwise vanish.
Thin reader.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from tum_colors import TUM_SECONDARY
from regime_style import regime_label, regime_color

PROJECT_ROOT    = Path("/Users/edward/Downloads")
REGIME_DATA_DIR = PROJECT_ROOT / "dataset" / "regime_data"
OUT_DIR         = PROJECT_ROOT / "bachelor-arbeit" / "regime_discovery" / "figures" / "gmm"

N_PLOT   = 30000
SEED     = 42
CROP_PCT = (1, 99)
N_SIGMA  = 1.0           # 1σ ellipse — tighter, far less overlap than 2σ
DPI      = 300


def main():
    comp = pd.read_parquet(REGIME_DATA_DIR / "gmm_components_2d.parquet").sort_values("regime")
    bg   = pd.read_parquet(REGIME_DATA_DIR / "gmm_assignments.parquet", columns=["pc1", "pc2"])

    x_lo, x_hi = np.percentile(bg["pc1"], CROP_PCT)
    y_lo, y_hi = np.percentile(bg["pc2"], CROP_PCT)
    px, py = 0.05 * (x_hi - x_lo), 0.05 * (y_hi - y_lo)
    s = bg.sample(min(N_PLOT, len(bg)), random_state=SEED)

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    # faint background — `s` (marker size) is the documented scatter latitude
    ax.scatter(s["pc1"], s["pc2"], color=TUM_SECONDARY["dark_gray"],
               s=4, alpha=0.2, edgecolor="none")

    for _, row in comp.iterrows():
        r = int(row["regime"])
        cov = np.array([[row["cov00"], row["cov01"]], [row["cov01"], row["cov11"]]])
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 0, None)
        ang = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
        w, h = 2 * N_SIGMA * np.sqrt(vals[-1]), 2 * N_SIGMA * np.sqrt(vals[0])
        ax.add_patch(Ellipse((row["mean_pc1"], row["mean_pc2"]), width=w, height=h,
                             angle=ang, edgecolor=regime_color(r), facecolor="none",
                             linewidth=2.0))

    # opaque mean dots on top — make every component (incl. the collapsed one) visible
    ax.scatter(comp["mean_pc1"], comp["mean_pc2"],
               color=[regime_color(int(r)) for r in comp["regime"]],
               s=90, edgecolor="black", linewidth=1.0)

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"Gaussian-mixture components ({N_SIGMA:.0f}σ ellipses, component means marked)")
    ax.set_xlim(x_lo - px, x_hi + px)
    ax.set_ylim(y_lo - py, y_hi + py)
    ax.grid(True); ax.set_axisbelow(True)

    handles = [Line2D([0], [0], marker="o", linestyle="none", color=regime_color(int(r)),
                      markeredgecolor="black", alpha=1.0, label=regime_label(int(r)))
               for r in comp["regime"]]
    ax.legend(handles=handles)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "covariance_overlay.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  → {out}")


if __name__ == "__main__":
    main()