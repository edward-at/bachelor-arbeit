#!/usr/bin/env python3
"""PCA occupancy grid — every method on the frozen PC basis, one panel each.

Per-method dossier form (D18). TWO figures are produced from one data pass:

  subset_pca_grid.png         points in the thesis single-series blue
  subset_pca_grid_regime.png  the same points coloured by the frozen
                              Chapter-9 regime, canonical regime_style palette

Each panel shows the frame's occupied region of the frozen PC1-PC2 plane (grey
2D histogram from the seeded frame sample) with the method's own sampled
windows overlaid, and the cached pc_coverage_share annotated. The basis and the
labels are Chapter 9's, read not refit (the fit-and-assign firewall applied on
the evaluation side); pc1, pc2 and regime come row-paired from the same index
draw in the distribution cache.

Marker size, stated: matplotlib's `s` is an AREA in points squared, so matching
the dot-to-panel ratio of the single-panel Chapter-9 scatter means scaling by
the SQUARE of the width ratio, not the width ratio. The two Chapter-9 constants
below are the only estimates in this file; correct them there and every panel
follows.

Axis crop, stated: both axes are cropped to the central CROP_PCT per cent of
the FRAME on each axis, the same convention and the same grounds as the
Chapter-9 scatters (legibility only, never a change of population). The crop is
taken on the frame so every panel shares one window and the panels stay
directly comparable; the number of method points falling outside it is counted
and printed, so nothing is hidden. Lower CROP_PCT to zoom further.

Two further choices, stated:
 (1) Points are drawn in a seeded RANDOM ORDER rather than regime by regime.
     Plotting one regime at a time would put the last-drawn regime on top
     everywhere, so a dense regime would appear to occupy territory it merely
     overdraws. Interleaving makes the visible mixture honest.
 (2) The Frame panel carries its own points too, not only the grey background,
     so the corpus geography is established once and every method read against
     it.

Reads ONLY the eval cache.
Output: figures/subset_eval/subset_pca_grid.png, subset_pca_grid_regime.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from subset_eval_common import (DPI, MAIN_BLUE, N_REGIMES, ROW_LABELS, TIER,
                                BUDGET_N, dist_arrays, ensure_outdir,
                                load_dist, load_master, present_methods,
                                regime_color, regime_label)

NROWS, NCOLS = 2, 6
FIG_W, FIG_H = 16.0, 7.4
BG_BINS = 120

# ---- marker size, derived rather than guessed -----------------------------
# Measured from pca_scatter_labeled.py (figsize=(6.8, 6.0), s=4) — no longer
# estimates; change ONLY here if the Chapter-9 scatter changes.
CH9_FIG_W_IN = 6.8      # width of the Chapter-9 scatter figure, inches
CH9_POINT_S = 4.0       # its scatter `s` (area, points squared)

_PANEL_W_IN = FIG_W / NCOLS                       # ~2.67 in per panel here
_AREA_SCALE = (_PANEL_W_IN / CH9_FIG_W_IN) ** 2   # AREA scales as width^2
POINT_S = CH9_POINT_S * _AREA_SCALE
POINT_ALPHA = 0.45      # smaller dots overlap less, so alpha can rise

CROP_PCT = 99.95        # central % of the FRAME kept on each axis (Chapter 9
                        # convention). 99 = wider, 95 = tighter zoom.
CROP_PAD = 0.03         # fraction of the cropped span added as margin

DRAW_SEED = 42          # permutation seed for the interleaved draw order
ANNOT_FS = 9            # documented fontsize exception: coverage annotation


def _load():
    mt = load_master(TIER)
    dist = load_dist(TIER)
    rows = ["R0_frame"] + present_methods(mt)

    p1 = dict(zip(rows, dist_arrays(dist, "pc1", rows)))
    p2 = dict(zip(rows, dist_arrays(dist, "pc2", rows)))
    rg = dict(zip(rows, dist_arrays(dist, "regime", rows)))

    if len(p1["R0_frame"]) == 0:
        raise SystemExit("pc1/pc2 not in the dist cache — re-run "
                         "subset_eval_harness.py")
    have_regime = len(rg["R0_frame"]) > 0
    if not have_regime:
        print('  [WARN] regime not in the dist cache — the regime-coloured '
              'figure is SKIPPED. Add "regime" to DIST_CHANNELS in '
              "subset_eval_harness.py and re-run the harness; it must be "
              "cached on the SAME index draw as pc1/pc2.")

    # pc1, pc2 and regime are row-paired: a window must be dropped from ALL of
    # them or the pairing is lost, so the finite mask is computed jointly.
    def paired(r):
        a, b = p1[r], p2[r]
        c = rg[r] if have_regime else np.zeros_like(a)
        if not (len(a) == len(b) == len(c)):
            raise SystemExit(f"[pca_grid] {r}: pc1/pc2/regime lengths differ "
                             f"({len(a)}/{len(b)}/{len(c)}) — the row-paired "
                             f"cache contract is broken; re-run the harness")
        m = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
        return a[m], b[m], c[m].astype(int)

    dropped = {r: len(p1[r]) - paired(r)[0].size
               for r in rows if len(p1[r])}
    dropped = {r: n for r, n in dropped.items() if n}
    if dropped:
        print("  [NOTE] pc1/pc2/regime: non-finite rows excluded — "
              + ", ".join(f"{r}: {n:,}" for r, n in dropped.items()))
    return mt, rows, paired, have_regime


def render(mt, rows, paired, by_regime, out_dir):
    fx, fy, _ = paired("R0_frame")
    if fx.size == 0:
        raise SystemExit("[pca_grid] frame sample is entirely non-finite — "
                         "check the dist cache")

    # Crop taken on the FRAME so all panels share one window (see docstring).
    q = (100.0 - CROP_PCT) / 2.0
    xlo, xhi = np.percentile(fx, [q, 100.0 - q])
    ylo, yhi = np.percentile(fy, [q, 100.0 - q])
    px, py = CROP_PAD * (xhi - xlo), CROP_PAD * (yhi - ylo)
    xlim, ylim = (xlo - px, xhi + px), (ylo - py, yhi + py)

    # Bin the background inside the crop, so the resolution is spent on what
    # is actually displayed rather than on the outliers being cropped away.
    H, xe, ye = np.histogram2d(fx, fy, bins=BG_BINS,
                               range=[list(xlim), list(ylim)])
    bg = np.log1p(H).T
    extent = (xlim[0], xlim[1], ylim[0], ylim[1])
    palette = np.array([regime_color(i) for i in range(N_REGIMES)])

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(FIG_W, FIG_H),
                             sharex=True, sharey=True)
    flat = axes.ravel()
    rng = np.random.default_rng(DRAW_SEED)
    outside = {}

    for k, r in enumerate(rows):
        ax = flat[k]
        ax.imshow(bg, extent=extent, origin="lower", cmap="Greys",
                  aspect="auto", alpha=0.55)
        x, y, reg = paired(r)
        if x.size:
            order = rng.permutation(x.size)     # interleave, docstring (1)
            c = (palette[np.clip(reg, 0, N_REGIMES - 1)][order]
                 if by_regime else MAIN_BLUE)
            ax.scatter(x[order], y[order], s=POINT_S, c=c,
                       alpha=POINT_ALPHA, linewidths=0)
        cov = float(mt.loc[r, "pc_coverage_share"])
        ax.text(0.03, 0.95, f"cov {cov:.3f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=ANNOT_FS,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75,
                          pad=1.5))
        if x.size:
            off = int((~((x >= xlim[0]) & (x <= xlim[1])
                         & (y >= ylim[0]) & (y <= ylim[1]))).sum())
            if off:
                outside[r] = off
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(ROW_LABELS[r])
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)

    for k in range(len(rows), NROWS * NCOLS):
        flat[k].axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("PC2")
    for ax in axes[-1, :]:
        ax.set_xlabel("PC1")

    if by_regime:
        handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                              markerfacecolor=regime_color(i),
                              markeredgecolor="none", markersize=7,
                              label=f"{i} {regime_label(i)}")
                   for i in range(N_REGIMES)]
        fig.legend(handles=handles, loc="lower center", ncol=N_REGIMES,
                   frameon=False, bbox_to_anchor=(0.5, -0.005))
        tail = "points coloured by regime"
        name = "subset_pca_grid_regime.png"
        rect = (0, 0.045, 1, 1)
    else:
        tail = "blue = method sample"
        name = "subset_pca_grid.png"
        rect = (0, 0, 1, 1)

    fig.suptitle(f"Per-method occupancy of the frozen PC plane (grey = frame sample), ({tail}), "
                 f"(cov = cached PC-cell coverage share)")
    fig.tight_layout(rect=rect)
    if not by_regime and outside:      # report once, not per figure
        print(f"  [NOTE] outside the central {CROP_PCT:g}% frame crop "
              f"(drawn but off-axis) — "
              + ", ".join(f"{r}: {n:,}" for r, n in outside.items()))
    out = out_dir / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {out}")


def main() -> None:
    out_dir = ensure_outdir()
    mt, rows, paired, have_regime = _load()
    print(f"  point size s = {POINT_S:.2f} "
          f"(Chapter-9 s = {CH9_POINT_S} at {CH9_FIG_W_IN}\" scaled by "
          f"area factor {_AREA_SCALE:.3f} for {_PANEL_W_IN:.2f}\" panels)")
    render(mt, rows, paired, False, out_dir)
    if have_regime:
        render(mt, rows, paired, True, out_dir)


if __name__ == "__main__":
    main()