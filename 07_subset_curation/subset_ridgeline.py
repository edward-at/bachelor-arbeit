#!/usr/bin/env python3
"""F4 — dynamic-score ridgeline per method (budget tier).

Mirrors ridgeline.py's grammar: overlapping filled KDE rows (ROW_OVERLAP
1.45), white ridge edges, black median ticks, x capped at a high percentile,
x-only grid. Deviations, stated: rows keep the fixed slate order (Frame at
the top) instead of a median sort — see subset_stacked_bar.py for the D4
grounds — and the variable is the composite dynamic score, the Tier-1
comparison axis, not a single sigma channel. Colours = design families.
Reads ONLY the eval cache. Output: figures/subset_eval/subset_ridgeline.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from tum_colors import TUM_SECONDARY
from subset_eval_common import (DPI, METHOD_COLORS, ROW_LABELS, TIER,
                                BUDGET_N, dist_arrays, ensure_outdir,
                                load_dist, load_master, present_methods)

X_PERCENTILE_CAP = 99.0
N_GRID = 512
ROW_OVERLAP = 1.45
MIN_POINTS = 8
DENSITY_ALPHA = 0.6
FIG_W, FIG_H_PER_ROW, FIG_H_PAD = 8.5, 0.55, 1.8
MEDIAN_TICK_HEIGHT, MEDIAN_TICK_LW, MEDIAN_TICK_ALPHA = 0.18, 1.0, 0.6


def main() -> None:
    out_dir = ensure_outdir()
    dist = load_dist(TIER)
    methods = present_methods(load_master(TIER))
    rows = methods[::-1] + ["R0_frame"]          # frame drawn as the TOP row

    arrays = dict(zip(rows, dist_arrays(dist, "score", rows)))
    # np.percentile PROPAGATES NaN, so the cap must be taken on finite
    # values only; the per-row loop below already filters the same way.
    frame_vals = arrays["R0_frame"]
    frame_vals = frame_vals[np.isfinite(frame_vals)]
    if frame_vals.size == 0:
        raise SystemExit("[ridgeline] frame score channel is entirely "
                         "non-finite — check the dist cache")
    x_cap = float(np.percentile(frame_vals, X_PERCENTILE_CAP))
    xs = np.linspace(0.0, max(x_cap, 1e-6), N_GRID)
    n = len(rows)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_PER_ROW * n + FIG_H_PAD))
    for i, r in enumerate(rows):
        v = arrays[r]
        v = v[np.isfinite(v)]
        vc = v[v <= x_cap]
        if vc.size < MIN_POINTS or np.std(vc) == 0:
            continue
        dens = gaussian_kde(vc)(xs)
        dens = dens / dens.max()
        base = float(i)
        color = (TUM_SECONDARY["dark_gray"] if r == "R0_frame"
                 else METHOD_COLORS[r])
        ax.fill_between(xs, base, base + dens * ROW_OVERLAP, color=color,
                        alpha=DENSITY_ALPHA, linewidth=0.8,
                        edgecolor="white", zorder=n - i)
        med = float(np.median(vc))
        ax.plot([med, med], [base, base + MEDIAN_TICK_HEIGHT],
                color="black", linewidth=MEDIAN_TICK_LW,
                alpha=MEDIAN_TICK_ALPHA, zorder=n - i + 0.5)

    ax.set_yticks(range(n))
    ax.set_yticklabels([ROW_LABELS[r] for r in rows])
    ax.set_ylim(-0.3, n - 1 + ROW_OVERLAP + 0.2)
    ax.set_xlim(0.0, x_cap)
    ax.set_xlabel(f"Composite dynamic score per 20 s window "
                  f"(x capped at frame p{X_PERCENTILE_CAP:.0f})")
    ax.set_title(f"Dynamic-score distribution per sampling method")
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = out_dir / "subset_ridgeline.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {out}")


if __name__ == "__main__":
    main()