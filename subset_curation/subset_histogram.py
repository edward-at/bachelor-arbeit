#!/usr/bin/env python3
"""Per-method frame-level signal histograms (distributional_analysis grammar).

66 single figures (Frame + M01..M10, six signals each) reproducing Section
8.1's grammar exactly — density bars, y-cap with overflow markers when one
bar dwarfs the rest, zero line for signed signals, dashed mean line, the
mu/sigma/p05/p95/valid-vs-total stats box — plus six comparative grids (one
per signal, all methods, frame ghost). One documented deviation: the x-axis
window per signal is the FRAME's p01/p99 rule, shared across all methods so
the dossier pages are directly comparable (the per-dataset originals
autoscaled per dataset). Counts come from the cached fine grid of
subset_frame_pass.py; display bins regroup that grid, percentiles come from
its cumulative sum (resolution = bound-range/1000, disclosed).
Reads ONLY the frame caches. Outputs: figures/subset_eval/hist/…
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tum_colors import TUM_PRIMARY
from subset_eval_common import (DPI, EVAL_DIR, MAIN_BLUE, ROW_LABELS,
                                TUM_SECONDARY, ensure_outdir)

N_BINS_DISPLAY = 125
X_PAD_FRAC = 0.10
YLIN_CAP_FACTOR = 1.20
FIG_W, FIG_H = 10.0, 6.0
MEAN_LINE_COLOR = TUM_PRIMARY["black"]

# y-cap policy (per signal). 'auto' = the faithful distributional_analysis rule
# (clip only a true spike, tallest > 2x second). 'trim' = UNCONDITIONALLY clip
# the TRIM_TOP_BARS tallest bars so the body is always legible, capping the
# y-axis at TRIM_HEADROOM x the tallest SURVIVING bar. A small caption records
# how many bars were clipped and their density, so nothing is hidden.
# Speed is the signal whose zero bar most often hides the body -> 'trim' there.
Y_CAP_MODE = {
    "speed":     "trim",
    "lon_accel": "auto",
    "lat_accel": "auto",
    "lon_jerk":  "auto",
    "lat_jerk":  "auto",
    "yaw_rate":  "auto",
}
TRIM_TOP_BARS = 1        # how many tallest bars to clip in 'trim' mode
TRIM_HEADROOM = 1.15     # y-cap = this x the tallest surviving bar

SIGNALS = {
    "speed":     dict(label="Speed", lo=0.0, hi=200.0,
                      xlabel="Speed (km/h)"),
    "lon_accel": dict(label="Longitudinal Acceleration", lo=-15.0, hi=15.0,
                      xlabel="Longitudinal Acceleration (m/s\u00b2)"),
    "lat_accel": dict(label="Lateral Acceleration", lo=-15.0, hi=15.0,
                      xlabel="Lateral Acceleration (m/s\u00b2)"),
    "lon_jerk":  dict(label="Longitudinal Jerk", lo=-30.0, hi=30.0,
                      xlabel="Longitudinal Jerk (m/s\u00b3)"),
    "lat_jerk":  dict(label="Lateral Jerk", lo=-30.0, hi=30.0,
                      xlabel="Lateral Jerk (m/s\u00b3)"),
    "yaw_rate":  dict(label="Yaw Rate", lo=-90.0, hi=90.0,
                      xlabel="Yaw Rate (\u00b0/s)"),
}
GRID_NROWS, GRID_NCOLS = 2, 6


def load():
    h = pd.read_parquet(EVAL_DIR / "frame_hist_fine.parquet")
    s = pd.read_parquet(EVAL_DIR / "frame_stats.parquet")
    fine = {}
    for (m, sig), g in h.groupby(["method", "signal"]):
        a = np.zeros(1000, dtype=np.int64)
        a[g["bin_idx"].values] = g["count"].values
        fine[(m, sig)] = a
    stats = {(r.method, r.signal): r for r in s.itertuples(index=False)}
    methods = [m for m in ["R0_frame"] + [f"M{i:02d}" for i in range(1, 11)]
               if (m, "speed") in fine]
    return fine, stats, methods


def pctl(counts, edges, q):
    c = np.cumsum(counts).astype(float)
    if c[-1] == 0:
        return np.nan
    c /= c[-1]
    i = int(np.searchsorted(c, q / 100.0))
    return float(edges[min(i + 1, len(edges) - 1)])


def frame_xlim(fine_frame, lo, hi):
    edges = np.linspace(lo, hi, 1001)
    p01 = pctl(fine_frame, edges, 1)
    p99 = pctl(fine_frame, edges, 99)
    rng = p99 - p01
    if not np.isfinite(rng) or rng < 1e-9:
        return lo, hi
    pad = rng * X_PAD_FRAC
    return max(lo, p01 - pad), min(hi, p99 + pad)


def regroup(fine, lo, hi, xlo, xhi):
    edges = np.linspace(lo, hi, 1001)
    i0 = int(np.searchsorted(edges, xlo, "left"))
    i1 = max(i0 + 2, int(np.searchsorted(edges, xhi, "right")) - 1)
    sub = fine[i0:i1]
    g = max(1, int(round(len(sub) / N_BINS_DISPLAY)))
    nb = len(sub) // g
    sub = sub[:nb * g]
    counts = sub.reshape(nb, g).sum(axis=1)
    de = edges[i0:i0 + nb * g + 1:g]
    centers = 0.5 * (de[:-1] + de[1:])
    return counts, centers, de[1] - de[0]


def ylin_cap(density):
    if len(density) < 2:
        return None
    d = np.sort(density)[::-1]
    if d[1] < 1e-9:
        return None
    return d[1] * YLIN_CAP_FACTOR if d[0] > 2.0 * d[1] else None


def trim_cap(density, n_top):
    """Unconditional cap: y-axis at TRIM_HEADROOM x the tallest bar remaining
    after removing the n_top tallest. Returns (cap, n_clipped, clipped_max)."""
    if len(density) <= n_top + 1:
        return None, 0, 0.0
    order = np.argsort(density)[::-1]
    survivor_max = float(density[order[n_top]])
    if survivor_max <= 0:
        return None, 0, 0.0
    cap = survivor_max * TRIM_HEADROOM
    clipped = density[density > cap]
    return cap, int(clipped.size), (float(clipped.max()) if clipped.size else 0.0)


def draw_single(method, sig_key, fine, stats, xlo, xhi, out_dir):
    spec = SIGNALS[sig_key]
    counts, centers, width = regroup(fine[(method, sig_key)],
                                     spec["lo"], spec["hi"], xlo, xhi)
    density = counts / max(1, counts.sum())
    st = stats[(method, sig_key)]
    mean_v = st.sum / st.n_valid if st.n_valid else np.nan
    var = st.sumsq / st.n_valid - mean_v ** 2 if st.n_valid else np.nan
    std_v = float(np.sqrt(max(0.0, var)))
    edges = np.linspace(spec["lo"], spec["hi"], 1001)
    p05 = pctl(fine[(method, sig_key)], edges, 5)
    p95 = pctl(fine[(method, sig_key)], edges, 95)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.bar(centers, density, width=width, color=MAIN_BLUE, zorder=2)

    mode = Y_CAP_MODE.get(sig_key, "auto")
    if mode == "trim":
        cap, n_clip, clip_max = trim_cap(density, TRIM_TOP_BARS)
        if cap is not None and n_clip > 0:
            ax.set_ylim(0, cap)
            for x in centers[density > cap]:
                ax.annotate("\u25b2", xy=(x, cap), xytext=(x, cap),
                            ha="center", va="bottom", fontsize=6, color="black")
            ax.text(0.03, 0.97,
                    f"y-axis clipped: {n_clip} bar(s) exceed the axis "
                    f"(max frac \u2248 {clip_max:.3f})",
                    ha="left", va="top", transform=ax.transAxes, fontsize=8,
                    color=TUM_PRIMARY["black"],
                    bbox=dict(facecolor="white", edgecolor="#cccccc",
                              alpha=0.85))
    else:
        cap = ylin_cap(density)
        if cap is not None:
            ax.set_ylim(0, cap)
            for x in centers[density > cap]:
                ax.annotate("\u25bc", xy=(x, cap), xytext=(x, cap),
                            ha="center", va="bottom", fontsize=6, color="black")
    if spec["lo"] < 0 and xlo < 0 < xhi:
        ax.axvline(0, color=MEAN_LINE_COLOR, lw=0.5, ls="-", zorder=1)
    ax.axvline(mean_v, color=MEAN_LINE_COLOR, lw=1, ls="--", zorder=4)
    ax.text(0.97, 0.97,
            f"\u03bc = {mean_v:+.3f}   \u03c3 = {std_v:.3f}\n"
            f"p05 = {p05:+.3f}   p95 = {p95:+.3f}\n"
            f"n = {int(st.n_valid):,} valid frames  |  "
            f"{int(st.n_total_frames):,} total frames",
            ha="right", va="top", transform=ax.transAxes,
            color=TUM_PRIMARY["black"], fontsize=9, zorder=5,
            bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.90))
    ax.set_xlim(xlo, xhi)
    ax.set_title(f"{spec['label']} Distribution ({ROW_LABELS[method]})")
    ax.set_xlabel(spec["xlabel"])
    ax.set_ylabel("Fraction of frames")
    ax.grid(True)
    ax.set_axisbelow(True)
    fig.tight_layout()
    rid = "frame" if method == "R0_frame" else method
    out = out_dir / f"{rid}_{sig_key}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def draw_grid(sig_key, fine, methods, xlo, xhi, spec, out_dir):
    fc, cc, w = regroup(fine[("R0_frame", sig_key)], spec["lo"], spec["hi"],
                        xlo, xhi)
    fd = fc / max(1, fc.sum())
    # shared y-cap for the whole grid, in the per-signal policy
    grid_cap = None
    if Y_CAP_MODE.get(sig_key, "auto") == "trim":
        densities = []
        for m in methods:
            c, _, _ = regroup(fine[(m, sig_key)], spec["lo"], spec["hi"],
                              xlo, xhi)
            densities.append(c / max(1, c.sum()))
        caps = [trim_cap(d, TRIM_TOP_BARS)[0] for d in densities + [fd]]
        caps = [c for c in caps if c is not None]
        grid_cap = max(caps) if caps else None
    fig, axes = plt.subplots(GRID_NROWS, GRID_NCOLS, figsize=(16, 6.2),
                             sharex=True, sharey=True)
    flat = axes.ravel()
    for k, m in enumerate(methods):
        ax = flat[k]
        c, cen, w2 = regroup(fine[(m, sig_key)], spec["lo"], spec["hi"],
                             xlo, xhi)
        d = c / max(1, c.sum())
        if m != "R0_frame":
            ax.bar(cen, d, width=w2, color=MAIN_BLUE, zorder=2)
        ax.plot(cc, fd, color=TUM_SECONDARY["dark_gray"], lw=1.0,
                alpha=0.85, zorder=3)
        ax.set_title(ROW_LABELS[m])
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
    if grid_cap is not None:
        flat[0].set_ylim(0, grid_cap)      # shared y-axis propagates the cap
    for k in range(len(methods), GRID_NROWS * GRID_NCOLS):
        flat[k].axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel(spec["xlabel"])
    for ax in axes[:, 0]:
        ax.set_ylabel("Fraction of frames")
    cap_note = (" (y-axis clipped to reveal the body)"
                if grid_cap is not None else "")
    fig.suptitle(f"{spec['label']} distribution per sampling method — "
                 f"grey line = frame; shared frame-derived x-window{cap_note}")
    fig.tight_layout()
    out = out_dir / f"subset_hist_grid_{sig_key}.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def main():
    out_root = ensure_outdir()
    singles = out_root / "hist"
    singles.mkdir(exist_ok=True)
    fine, stats, methods = load()
    for sig_key, spec in SIGNALS.items():
        if ("R0_frame", sig_key) not in fine:
            continue
        xlo, xhi = frame_xlim(fine[("R0_frame", sig_key)],
                              spec["lo"], spec["hi"])
        for m in methods:
            draw_single(m, sig_key, fine, stats, xlo, xhi, singles)
        draw_grid(sig_key, fine, methods, xlo, xhi, spec, out_root)
        print(f"  [SAVED] {sig_key}: {len(methods)} singles + grid")


if __name__ == "__main__":
    main()