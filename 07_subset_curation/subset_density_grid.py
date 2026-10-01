#!/usr/bin/env python3
"""Density grid — every method gets the density plot, one panel each.

Per-method dossier form of the density comparison: one panel per
method, the method's dynamic-score KDE in the thesis blue against the
frame's density as a grey ghost. The 3-5-line OVERLAY version remains the
Section 4.10 finalist spotlight. Change CHANNEL for per-channel grids.
Reads ONLY the eval cache. Output: figures/subset_eval/subset_density_grid.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from tum_colors import TUM_SECONDARY
from subset_eval_common import (DPI, MAIN_BLUE, ROW_LABELS, TIER, BUDGET_N,
                                dist_arrays, ensure_outdir, load_dist,
                                load_master, present_methods)

CHANNEL = "score"
CHANNEL_LABEL = "dynamic score"
X_CLIP_Q = 99.5
GRID_PTS = 300
NROWS, NCOLS = 2, 6


def main() -> None:
    out_dir = ensure_outdir()
    dist = load_dist(TIER)
    methods = present_methods(load_master(TIER))
    rows = ["R0_frame"] + methods

    arrays = dict(zip(rows, dist_arrays(dist, CHANNEL, rows)))
    frame_vals = arrays["R0_frame"]
    frame_vals = frame_vals[np.isfinite(frame_vals)]
    if frame_vals.size < 10 or np.std(frame_vals) == 0:
        raise SystemExit("[density_grid] frame channel has too few distinct "
                         "values for a KDE — check the dist cache")
    x_hi = float(np.percentile(frame_vals, X_CLIP_Q))
    xs = np.linspace(0.0, x_hi, GRID_PTS)
    frame_dens = gaussian_kde(frame_vals[frame_vals <= x_hi])(xs)

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(16, 6.5),
                             sharex=True, sharey=True)
    flat = axes.ravel()
    for k, r in enumerate(rows):
        ax = flat[k]
        ax.fill_between(xs, 0, frame_dens, color=TUM_SECONDARY["gray"],
                        alpha=0.35)
        ax.plot(xs, frame_dens, color="black", linewidth=0.9, alpha=0.6)
        if r != "R0_frame":
            v = arrays[r]
            v = v[np.isfinite(v)]
            v = v[v <= x_hi]
            if len(v) >= 10 and np.std(v) > 0:
                ax.plot(xs, gaussian_kde(v)(xs), color=MAIN_BLUE,
                        linewidth=1.5)
        ax.set_title(ROW_LABELS[r])
        ax.set_xlim(0, x_hi)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
    for k in range(len(rows), NROWS * NCOLS):
        flat[k].axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("Density")
    for ax in axes[-1, :]:
        ax.set_xlabel(CHANNEL_LABEL)

    fig.suptitle(f"Per-method {CHANNEL_LABEL} density vs the frame (grey),  "
                 f"x capped at frame p{X_CLIP_Q}")
    fig.tight_layout()
    out = out_dir / "subset_density_grid.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  [SAVED] {out}")


if __name__ == "__main__":
    main()