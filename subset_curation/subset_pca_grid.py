#!/usr/bin/env python3
"""PCA occupancy grid — every method on the frozen PC basis, one panel each.

Per-method dossier form (D18): each panel shows the frame's occupied region
of the frozen PC1-PC2 plane (grey 2D histogram from the seeded frame sample)
with the method's own sampled windows overlaid in the thesis blue; the
cached pc_coverage_share is annotated per panel. The basis is Chapter 9's,
read not refit; samples come row-paired from the v12.1.2 distribution cache.
Reads ONLY the eval cache. Output: figures/subset_eval/subset_pca_grid.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from subset_eval_common import (DPI, MAIN_BLUE, ROW_LABELS, TIER, BUDGET_N,
                                dist_arrays, ensure_outdir, load_dist,
                                load_master, present_methods)

NROWS, NCOLS = 2, 6
BG_BINS = 120
POINT_S, POINT_ALPHA = 1.5, 0.20
ANNOT_FS = 9   # documented fontsize exception: per-panel coverage annotation


def main() -> None:
    out_dir = ensure_outdir()
    mt = load_master(TIER)
    dist = load_dist(TIER)
    methods = present_methods(mt)
    rows = ["R0_frame"] + methods

    p1 = dict(zip(rows, dist_arrays(dist, "pc1", rows)))
    p2 = dict(zip(rows, dist_arrays(dist, "pc2", rows)))
    if len(p1["R0_frame"]) == 0:
        raise SystemExit("pc1/pc2 not in the dist cache — re-run "
                         "subset_eval_harness.py v12.1.2+")

    fx, fy = p1["R0_frame"], p2["R0_frame"]
    H, xe, ye = np.histogram2d(fx, fy, bins=BG_BINS)
    bg = np.log1p(H).T
    extent = (xe[0], xe[-1], ye[0], ye[-1])

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(16, 6.8),
                             sharex=True, sharey=True)
    flat = axes.ravel()
    for k, r in enumerate(rows):
        ax = flat[k]
        ax.imshow(bg, extent=extent, origin="lower", cmap="Greys",
                  aspect="auto", alpha=0.9)
        if r != "R0_frame":
            ax.scatter(p1[r], p2[r], s=POINT_S, color=MAIN_BLUE,
                       alpha=POINT_ALPHA, linewidths=0)
        cov = float(mt.loc[r, "pc_coverage_share"])
        ax.text(0.03, 0.95, f"cov {cov:.3f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=ANNOT_FS)
        ax.set_title(ROW_LABELS[r])
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
    for k in range(len(rows), NROWS * NCOLS):
        flat[k].axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("PC2")
    for ax in axes[-1, :]:
        ax.set_xlabel("PC1")

    fig.suptitle(f"Per-method occupancy of the frozen PC plane — budget tier "
                 f"(N = {BUDGET_N:,}); grey = frame sample, blue = method "
                 f"sample, cov = cached PC-cell coverage share")
    fig.tight_layout()
    out = out_dir / "subset_pca_grid.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  [SAVED] {out}")


if __name__ == "__main__":
    main()