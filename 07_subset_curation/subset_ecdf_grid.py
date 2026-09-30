#!/usr/bin/env python3
"""ECDF grid — every method gets the ECDF, without the 11-line overlay soup.

One panel per method (Frame profile leading), each showing the method's
dynamic-score ECDF in the thesis blue against the frame's ECDF as a black
ghost — the readable all-methods form of the ECDF. The 3-5-line OVERLAY
version remains the finalist / 12.3 figure, where the proposed curation
method is the highlighted line against the strongest baselines. Change
CHANNEL to a sigma channel for per-channel grids.
Reads ONLY the eval cache. Output: figures/subset_eval/subset_ecdf_grid.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from subset_eval_common import (DPI, MAIN_BLUE, ROW_LABELS, TIER, BUDGET_N,
                                dist_arrays, ensure_outdir, load_dist,
                                load_master, present_methods)

CHANNEL = "score"
CHANNEL_LABEL = "dynamic score"
X_CLIP_Q = 99.5
NROWS, NCOLS = 2, 6


def ecdf(v):
    x = np.sort(v)
    return x, np.arange(1, len(x) + 1) / len(x)


def main() -> None:
    out_dir = ensure_outdir()
    dist = load_dist(TIER)
    methods = present_methods(load_master(TIER))
    rows = ["R0_frame"] + methods

    arrays = dict(zip(rows, dist_arrays(dist, CHANNEL, rows)))
    arrays = {k: v[np.isfinite(v)] for k, v in arrays.items()}
    fx, fy = ecdf(arrays["R0_frame"])
    x_hi = float(np.percentile(arrays["R0_frame"], X_CLIP_Q))

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(16, 6.5),
                             sharex=True, sharey=True)
    flat = axes.ravel()
    for k, r in enumerate(rows):
        ax = flat[k]
        ax.plot(fx, fy, color="black", linewidth=1.0, alpha=0.55)
        if r != "R0_frame":
            mx, my = ecdf(arrays[r])
            ax.plot(mx, my, color=MAIN_BLUE, linewidth=1.4)
        ax.set_title(ROW_LABELS[r])
        ax.set_xlim(0, x_hi)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
    for k in range(len(rows), NROWS * NCOLS):
        flat[k].axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("ECDF")
    for ax in axes[-1, :]:
        ax.set_xlabel(CHANNEL_LABEL)

    fig.suptitle(f"Per-method {CHANNEL_LABEL} ECDF vs the frame (black) "
                 f"x capped at frame p{X_CLIP_Q}")
    fig.tight_layout()
    out = out_dir / "subset_ecdf_grid.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  [SAVED] {out}")


if __name__ == "__main__":
    main()