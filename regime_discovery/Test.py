"""
covariance_contours.py  —  Section 4.6.8 figure (companion to covariance_overlay)
────────────────────────────────────────────────────────────────────────────
Where covariance_overlay.py draws a single 1σ ring per component, this figure
shows the density fall-off: each component is drawn at several probability-mass
levels, so the reader sees how far a component actually reaches rather than one
arbitrary cut.

CONTOUR LEVELS. For a two-dimensional Gaussian the contour enclosing probability
mass p lies at Mahalanobis radius r = sqrt(-2 ln(1 - p)):

      p = 0.10 -> r = 0.459        p = 0.75 -> r = 1.665
      p = 0.25 -> r = 0.759        p = 0.90 -> r = 2.146
      p = 0.50 -> r = 1.177

Note that the 1σ ring (r = 1) of the companion figure encloses only 39 % of the
mass in two dimensions — the components reach substantially further than that
figure suggests, which is precisely what this one is for.

RENDERING. All components share one panel and every level is drawn as an
outline. Line weight and opacity are graded by level — inner contours heavier
and solid, outer contours thinner and fainter — so that each component reads as
one nested set rather than as five unrelated rings of equal weight. The mean is
marked so a component that collapses toward a point remains visible.

CAVEAT. These are two-dimensional projections of five-dimensional Gaussians:
the ellipse is the marginal density in the PC1/PC2 plane, so components that
overlap here may still be separated along the components not shown.

Thin reader of gmm_components_2d.parquet (means + projected covariances) and
gmm_assignments.parquet (background windows). No recomputation.
────────────────────────────────────────────────────────────────────────────
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

LEVELS    = [0.10, 0.25, 0.50, 0.75, 0.90]   # probability mass enclosed
LW_INNER  = 2.2       # linewidth of the innermost contour
LW_OUTER  = 0.7       # linewidth of the outermost contour
A_INNER   = 1.00      # opacity of the innermost contour
A_OUTER   = 0.45      # opacity of the outermost contour
N_PLOT    = 30000     # background subsample, rendering only (matches the overlay figure)
PT_SIZE   = 4         # background marker size — same as covariance_overlay.py
PT_ALPHA  = 0.40      # background opacity   — same as covariance_overlay.py
SEED      = 42
CROP_PCT  = (1, 99)
DPI       = 300


def _radius(p: float) -> float:
    """Mahalanobis radius enclosing probability mass p for a 2-D Gaussian."""
    return float(np.sqrt(-2.0 * np.log(1.0 - p)))


def _ellipse(row, r, **kw) -> Ellipse:
    """Ellipse patch for one component at Mahalanobis radius r."""
    cov = np.array([[row["cov00"], row["cov01"]],
                    [row["cov01"], row["cov11"]]], dtype=float)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 0.0, None)
    ang = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
    w, h = 2 * r * np.sqrt(vals[-1]), 2 * r * np.sqrt(vals[0])
    return Ellipse((row["mean_pc1"], row["mean_pc2"]), width=w, height=h,
                   angle=ang, facecolor="none", **kw)


def main():
    comp = (pd.read_parquet(REGIME_DATA_DIR / "gmm_components_2d.parquet")
              .sort_values("regime"))
    bg = pd.read_parquet(REGIME_DATA_DIR / "gmm_assignments.parquet",
                         columns=["pc1", "pc2"])

    x_lo, x_hi = np.percentile(bg["pc1"], CROP_PCT)
    y_lo, y_hi = np.percentile(bg["pc2"], CROP_PCT)
    px, py = 0.05 * (x_hi - x_lo), 0.05 * (y_hi - y_lo)
    s = bg.sample(min(N_PLOT, len(bg)), random_state=SEED)

    levels = sorted(LEVELS)                    # innermost first
    radii  = {p: _radius(p) for p in LEVELS}
    print("  contour radii: " +
          " · ".join(f"{int(p*100)}% -> {radii[p]:.3f}" for p in LEVELS))

    # grade weight and opacity from the innermost to the outermost level
    m = max(len(levels) - 1, 1)
    lw = {p: LW_INNER + (LW_OUTER - LW_INNER) * i / m for i, p in enumerate(levels)}
    al = {p: A_INNER  + (A_OUTER  - A_INNER)  * i / m for i, p in enumerate(levels)}

    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    # `s` (marker size) — documented scatter latitude
    ax.scatter(s["pc1"], s["pc2"], color=TUM_SECONDARY["dark_gray"],
               s=PT_SIZE, alpha=PT_ALPHA, edgecolor="none")

    for _, row in comp.iterrows():
        r_id = int(row["regime"])
        for p in levels:
            ax.add_patch(_ellipse(row, radii[p], edgecolor=regime_color(r_id),
                                  linewidth=lw[p], alpha=al[p], zorder=3))
    ax.scatter(comp["mean_pc1"], comp["mean_pc2"],
               color=[regime_color(int(r)) for r in comp["regime"]],
               s=90, edgecolor="black", linewidth=1.0, zorder=10)

    ax.set_xlim(x_lo - px, x_hi + px)
    ax.set_ylim(y_lo - py, y_hi + py)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    lvl_txt = ", ".join(f"{int(p*100)} %" for p in LEVELS)
    ax.set_title("Gaussian-mixture component density contours on the projection\n"
                 f"(contours at {lvl_txt} enclosed mass; inner contours drawn heavier)")
    ax.grid(True); ax.set_axisbelow(True)

    handles = [Line2D([0], [0], marker="o", linestyle="none",
                      color=regime_color(int(r["regime"])), alpha=1.0,
                      label=regime_label(int(r["regime"])))
               for _, r in comp.iterrows()]
    ax.legend(handles=handles, loc="upper right")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "covariance_contours.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  → {out}")


if __name__ == "__main__":
    main()