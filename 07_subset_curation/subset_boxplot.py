#!/usr/bin/env python3
"""Within-window variability per method: THREE figures, one per signal.

Mirrors boxplot_comparison.py (v2 grammar) exactly: right-side half-violin
(KDE trimmed to real support), white IQR box with coloured edges, thick
coloured median, mean dot (white fill, black edge), capped whiskers at
1.5xIQR, robust y-limit with tail disclosure ("max ~ X" annotations), proxy
legend upper-left, dashed y-grid. Two deliberate deviations, stated:
(1) fixed slate order with the Frame as the leading group behind a separator
    instead of a median sort — ordering methods by outcome before the decision rule would
    presuppose the rule;
(2) statistics are computed on the seeded 25k-per-method distribution cache
    (deterministic; quartiles at that n are indistinguishable from full-n).
Colours = design-family colours from subset_eval_common.
Outputs: figures/subset_eval/subset_boxplot_{speed,acc_mag,yaw_rate}.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import gaussian_kde

from tum_colors import TUM_PRIMARY
from subset_eval_common import (DPI, METHOD_COLORS, ROW_LABELS, TIER,
                                BUDGET_N, dist_arrays, ensure_outdir,
                                load_dist, load_master, present_methods)

FIG_H = 7.0
FIG_W_MIN = 16.0
FIG_W_PER_GROUP = 1.35   # axes width scales with the slate size
VIOLIN_WIDTH, BOX_WIDTH, WHISKER_WIDTH = 0.35, 0.12, 0.08
MEAN_DOT_SIZE, VIOLIN_ALPHA = 7, 0.40
BOX_LINEWIDTH, MEDIAN_LW, WHISKER_LW = 1.6, 2.2, 1.4
KDE_POINTS, DENSITY_FLOOR = 400, 0.005
Y_TOP_PCTL, HEADROOM = 99.0, 1.08
FRAME_COLOR = TUM_PRIMARY["black"]

SIGNALS = [
    dict(key="speed",    ch="std_speed_kmh",
         title="Standard Deviation of Speed per Sampling Method",
         ylabel="\u03c3(speed) [km/h]", clip=(0.0, 200.0)),
    dict(key="acc_mag",  ch="std_acc_mag",
         title="Standard Deviation of Acceleration Magnitude per Sampling Method (*)",
         ylabel="\u03c3(acc_mag) [m/s\u00b2] *", clip=(0.0, 21.3)),
    dict(key="yaw_rate", ch="std_yaw_deg",
         title="Standard Deviation of Yaw Rate per Sampling Method",
         ylabel="\u03c3(yaw_rate) [\u00b0/s]", clip=(0.0, 90.0)),
]


def _half_violin(ax, x, vals, clip, color):
    lo_c, hi_c = clip
    v = vals[(vals >= lo_c) & (vals <= hi_c)]
    if len(v) < 10:
        return
    lo = max(lo_c, float(v.min()))
    hi = min(hi_c, float(np.percentile(v, 99.5)))
    if hi <= lo:
        hi = min(hi_c, float(v.max()))
    if hi <= lo:
        return
    grid = np.linspace(lo, hi, KDE_POINTS)
    try:
        dens = gaussian_kde(v, bw_method="scott")(grid)
    except Exception:
        return
    if dens.max() == 0:
        return
    keep = np.where(dens > DENSITY_FLOOR * dens.max())[0]
    if len(keep) == 0:
        return
    sl = slice(keep.min(), keep.max() + 1)
    grid, dens = grid[sl], dens[sl]
    off = dens / dens.max() * VIOLIN_WIDTH
    ax.fill_betweenx(grid, np.full_like(grid, x), x + off,
                     color=color, alpha=VIOLIN_ALPHA, linewidth=0)
    ax.plot(x + off, grid, color=color, linewidth=0.8, alpha=0.7)


def _box(ax, x, vals, color, clip):
    lo_c, hi_c = clip
    v = vals[np.isfinite(vals)]
    v = v[(v >= lo_c) & (v <= hi_c)]
    if len(v) < 4:
        return
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr, mean = q3 - q1, v.mean()
    lo_w = v[v >= q1 - 1.5 * iqr].min()
    hi_w = v[v <= q3 + 1.5 * iqr].max()
    hb, hw = BOX_WIDTH / 2, WHISKER_WIDTH / 2
    ax.add_patch(mpatches.FancyBboxPatch(
        (x - hb, q1), BOX_WIDTH, iqr, boxstyle="square,pad=0",
        linewidth=BOX_LINEWIDTH, edgecolor=color, facecolor="white",
        zorder=3))
    ax.plot([x - hb, x + hb], [med, med], color=color,
            linewidth=MEDIAN_LW, zorder=4)
    for w, a, b in [(lo_w, lo_w, q1), (hi_w, q3, hi_w)]:
        ax.plot([x, x], [a, b], color=color, linewidth=WHISKER_LW, zorder=3)
        ax.plot([x - hw, x + hw], [w, w], color=color,
                linewidth=WHISKER_LW, zorder=3)
    ax.plot(x, mean, marker="o", markersize=MEAN_DOT_SIZE,
            markerfacecolor="white", markeredgecolor="black",
            markeredgewidth=1.3, zorder=5)


def _fmt(v):
    return f"{v:.0f}" if v >= 100 else (f"{v:.1f}" if v >= 10 else f"{v:.2f}")


def plot_signal(sig, dist, rows, n_real):
    arrays = dist_arrays(dist, sig["ch"], rows)
    # the dist cache is a raw row-paired sample, so NaNs
    # pass through; filter here and report, because non-finite sigma values in
    # the restored corpus are themselves a data-state signal worth seeing
    n_bad = int(sum((~np.isfinite(a)).sum() for a in arrays))
    if n_bad:
        print(f"  [NOTE] {sig['key']}: {n_bad:,} non-finite sampled values "
              f"excluded from display statistics")
    arrays = [a[np.isfinite(a)] for a in arrays]
    fig_w = max(FIG_W_MIN, FIG_W_PER_GROUP * len(rows) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, FIG_H))
    ax.set_title(sig["title"], fontsize=13, pad=10)
    ax.set_ylabel(sig["ylabel"], fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    xt, xl = [], []
    for i, (row, vals) in enumerate(zip(rows, arrays)):
        color = FRAME_COLOR if row == "R0_frame" else METHOD_COLORS[row]
        x = i + 1
        _half_violin(ax, x, vals, sig["clip"], color)
        _box(ax, x, vals, color, sig["clip"])
        xt.append(x)
        xl.append(f"{ROW_LABELS[row]}\n(n={n_real[row]:,})")
    ax.axvline(1.5 + VIOLIN_WIDTH / 2, color="black", linewidth=1.0)

    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=9)
    ax.set_xlim(0.3, len(rows) + VIOLIN_WIDTH + 0.3)

    p99 = [float(np.percentile(a, Y_TOP_PCTL)) for a in arrays if len(a)]
    y_top = HEADROOM * max(p99) if p99 else sig["clip"][1]
    ax.set_ylim(0, y_top)
    for x, a in zip(xt, arrays):
        if len(a) and float(a.max()) > y_top:
            ax.text(x, y_top * 0.985, f"\u25b2 max \u2248 {_fmt(float(a.max()))}",
                    ha="center", va="top", fontsize=7.5, color="black")

    legend = [
        mpatches.Patch(facecolor="grey", alpha=VIOLIN_ALPHA,
                       label="Density (half-violin)"),
        mpatches.Patch(facecolor="white", edgecolor="grey",
                       linewidth=BOX_LINEWIDTH,
                       label="IQR box (25th\u201375th %ile)"),
        plt.Line2D([0], [0], color="grey", linewidth=MEDIAN_LW,
                   label="Median"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                   markeredgecolor="black", markeredgewidth=1.3,
                   markersize=MEAN_DOT_SIZE, label="Mean"),
        plt.Line2D([0], [0], color="grey", linewidth=WHISKER_LW,
                   label="Whiskers (1.5 \u00d7 IQR)"),
    ]
    # legend outside the axes: with Frame leading in fixed slate order, its
    # clipped-max annotation and upper tail sit exactly where an in-axes
    # upper-left legend would (the median-sorted Section 4.6 originals never
    # collided because their leftmost group was the calmest, unclipped one)
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.005, 1.0),
              frameon=True, facecolor="white", fontsize=9)
    fig.tight_layout()
    out = ensure_outdir() / f"subset_boxplot_{sig['key']}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {out}")


def main() -> None:
    mt = load_master(TIER)
    dist = load_dist(TIER)
    rows = ["R0_frame"] + present_methods(mt)
    n_real = {r: int(mt.loc[r, "n_windows"]) for r in rows}
    for sig in SIGNALS:
        plot_signal(sig, dist, rows, n_real)


if __name__ == "__main__":
    main()