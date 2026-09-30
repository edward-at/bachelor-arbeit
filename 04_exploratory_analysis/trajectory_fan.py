"""
trajectory_fans.py   (windowed)
===============================
Two cross-dataset "fan" figures for the dynamic-bias chapter, now built from
NON-OVERLAPPING TIME WINDOWS instead of whole clips:

    1 clip  ->  sliced into back-to-back WINDOW_S-second windows  ->  1 window = 1 line

Why windows, not whole clips:
  * non-overlapping  -> the lines don't correlate with each other
  * short (e.g. 20s) -> dead-reckoning can't drift to thousands of metres
  * <= 20s           -> even short-clip datasets (nuScenes/Waymo/NVIDIA) contribute

FIGURE 1  X-Y trajectory fan   — origin at LEFT-MIDDLE, fan opens right,
                                  equal aspect, percentile extent, coloured by
                                  absolute total heading change of the window.
FIGURE 2  Speed-vs-time fan     — x = time within window [0, WINDOW_S], y = speed,
                                  slope = acceleration, coloured by peak |accel|.

Reads FRAME-LEVEL data from inertial_data/<dataset>/clip_*.parquet.
On each run it prints windows-per-dataset and path-length stats so you can judge
the WINDOW_S choice empirically (sweep 10/20/30 and compare).
"""

import sys
import random
import warnings
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

warnings.filterwarnings("ignore")

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from tqdm import tqdm
except Exception:                      # graceful no-op if tqdm isn't installed
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

# ═════════════════════════════════════════════════════════════════════════════
# PATHS
# ═════════════════════════════════════════════════════════════════════════════
PROJECT_ROOT  = Path("/Users/edward/Downloads")
INERTIAL_ROOT = PROJECT_ROOT / "dataset" / "inertial_data"
FIG_ROOT      = PROJECT_ROOT / "bachelor-arbeit" / "04_exploratory_analysis" / "figures" / "trajectory_fans"

DATASETS = None   # None -> auto-discover EVERY folder under inertial_data/ (all 14, any
                  #         spelling). Or hard-code a list, e.g. ["boreas_rt", "oxford", ...].

# ═════════════════════════════════════════════════════════════════════════════
# COLUMN NAMES  ── confirm against the [SCHEMA] print, then edit
# ═════════════════════════════════════════════════════════════════════════════
SPEED_COL = "speed_kmh"
YAW_COL   = "yaw_rate"     # rad/s
ACCEL_COL = "acc_mag"      # acceleration magnitude = sqrt(a_long²+a_lat²); derived from speed if absent
TIME_COL  = "t_rel_s"      # real relative-seconds column in your clips (uniform 10 Hz grid).
                           # Falls back to GRID_HZ only if this column is absent.

GRID_HZ = 10               # uniform grid rate of the standardized clips. ONLY a fallback if
                           # TIME_COL is missing; with t_rel_s present this is never used.

# ═════════════════════════════════════════════════════════════════════════════
# WINDOWING  (the decision you wanted to make first — sweep WINDOW_S to compare)
# ═════════════════════════════════════════════════════════════════════════════
WINDOW_S         = 20.0    # seconds per window. Try 10 / 20 / 30. Recommended: 20.
MIN_WINDOW_FRAC  = 0.5     # drop a trailing window shorter than this fraction of WINDOW_S
                           #   (non-overlapping: step == WINDOW_S, so windows never share frames)

# ═════════════════════════════════════════════════════════════════════════════
# SAMPLING / VISIBILITY
# ═════════════════════════════════════════════════════════════════════════════
N_SAMPLE   = 5000     # how many random windows (= lines) to draw per dataset:
                      #   integer (e.g. 1500, 3000, 500) -> that many RANDOM windows
                      #   None                           -> use ALL windows (every line)
                      # Must be an int or None — never a list. All-data is very dense and
                      # slow to render for huge datasets (nvidia 298k clips, waymo 70k).
SEED       = 0        # change this to draw a DIFFERENT random subset (same N_SAMPLE)
N_WORKERS  = 8        # threads for reading parquet in parallel (reading is I/O-bound)
ALPHA      = 0.16
LINEWIDTH  = 0.8
EXTENT_PCT = 99.0     # robust framing: percentile of per-window max |x| / |y|
SPEED_PCT  = 99.0     # robust y-cap for the speed fan (tames bad-speed spikes visually)

# z-order = which lines end up on top (drawn last)
XY_SORT = "curvature_straight_top"   # curved underneath, straight on top -> clean centre
ST_SORT = "scalar_extreme_top"       # hard-accel lines drawn last -> they pop

# ═════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ═════════════════════════════════════════════════════════════════════════════
NCOLS    = 5
PANEL_IN = 3.4
COLORMAP = "viridis"
DPI      = 300
SEPARATE_FILES  = True         # True  = ONE PNG per dataset (avoids tiny panels / label overlap)
GRID_FILES      = True         # 2026-09-25: ALSO draw the 14-panel comparison grid per view
NCOLS_GRID      = 4            # 14 panels on a 4 x 4 grid, two cells empty
DISPLAY_NAMES = {
    "4seasons": "4Seasons", "a2d2": "A2D2", "boreas_rt": "Boreas", "durlar": "DurLAR",
    "ford": "Ford", "hdd": "HDD", "kitscenes": "KITScenes", "kitti": "KITTI",
    "kitti360": "KITTI-360", "nuscene": "nuScenes", "nvidia": "NVIDIA",
    "oxford": "Oxford", "waymo": "Waymo", "zod": "ZOD",
}
def _panel_order(per_ds):
    return sorted(per_ds.keys(), key=lambda d: DISPLAY_NAMES.get(d, d).lower())


def _grid_figure(n, panel_w, panel_h):
    """Small-multiples canvas: n panels on NCOLS_GRID columns, shared axes so that tick
    labels appear on the outer panels only. Returns (fig, axes2d)."""
    ncols = min(NCOLS_GRID, n); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, panel_h * nrows),
                             sharex=True, sharey=True, squeeze=False)
    return fig, axes


def _finish_grid(fig, axes, n, xlabel, ylabel):
    """Hide unused cells, keep one x label per column (on its last visible panel) and one
    y label per row (first column), and tighten the spacing."""
    flat = axes.ravel()
    for ax in flat[:n]:
        ax.set_xlabel(""); ax.set_ylabel("")
    for ax in flat[n:]:
        ax.set_visible(False)
    ncols = axes.shape[1]
    for c in range(ncols):
        col = [ax for ax in axes[:, c] if ax.get_visible()]
        if col:
            col[-1].set_xlabel(xlabel); col[-1].tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        if ax.get_visible():
            ax.set_ylabel(ylabel)
    fig.subplots_adjust(hspace=0.35, wspace=0.12)
                               # False = all datasets in one combined grid PNG
SINGLE_IN_XY    = (8.0, 5.0)   # figure size (inches) for a single-dataset X-Y fan
SINGLE_IN_SPEED = (8.0, 4.5)   # figure size for a single-dataset speed-change fan
ASPECT_EQUAL = True   # True  = 1 m in x equals 1 m in y (no distortion). Forward-heavy
                      #         datasets then render as WIDE STRIPS (geometrically honest).
                      # False = each panel fills a square box (paper-like, compact) but a
                      #         turn looks distorted because x and y use different scales.

# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def _now():
    return datetime.now().strftime("%H:%M:%S")


def _grid(n):
    ncols = min(NCOLS, n)
    return (n + ncols - 1) // ncols, ncols


def list_clips(ds):
    d = INERTIAL_ROOT / ds
    return sorted(d.glob("clip_*.parquet")) if d.exists() else []


def discover_datasets():
    """Every subfolder of inertial_data/ that actually contains clip_*.parquet."""
    if not INERTIAL_ROOT.exists():
        return []
    return [d.name for d in sorted(INERTIAL_ROOT.iterdir())
            if d.is_dir() and any(d.glob("clip_*.parquet"))]


def load_frames(fp, ds):
    """Return masked full-clip arrays (v_ms, yaw, t, dt, a_long_or_None) or None."""
    df = pd.read_parquet(fp)
    if SPEED_COL not in df.columns or YAW_COL not in df.columns:
        return None

    v_ms = df[SPEED_COL].to_numpy(dtype=float) / 3.6
    yaw  = df[YAW_COL].to_numpy(dtype=float)
    if TIME_COL is not None and TIME_COL in df.columns:
        t = df[TIME_COL].to_numpy(dtype=float); t = t - t[0]; dt = np.gradient(t)
    else:
        hz = GRID_HZ
        dt = np.full(len(v_ms), 1.0 / hz); t = np.cumsum(dt) - dt[0]
    a = df[ACCEL_COL].to_numpy(dtype=float) if ACCEL_COL in df.columns else None

    m = np.isfinite(v_ms) & np.isfinite(yaw) & np.isfinite(t)
    if m.sum() < 5:
        return None
    a = a[m] if a is not None else None
    return v_ms[m], yaw[m], t[m], dt[m], a


def window_records(frames, ds):
    """Slice clip into non-overlapping WINDOW_S windows -> records.
    O(N) (immune to giant/time-corrupt clips); keeps a short clip as one window."""
    v_ms, yaw, t, dt, a = frames
    rel = t - t[0]
    total = float(rel[-1])
    n = len(rel)

    if total < WINDOW_S:
        groups = [np.arange(n)]                       # short clip: whole clip = one window
    else:
        win_id = (rel / WINDOW_S).astype(np.int64)    # which 20 s bin each frame falls in
        cut    = np.flatnonzero(np.diff(win_id)) + 1  # boundaries between bins  (single pass)
        groups = np.split(np.arange(n), cut)

    recs = []
    for idx in groups:
        if idx.size < 5:
            continue
        span = rel[idx[-1]] - rel[idx[0]]
        if total >= WINDOW_S and span < MIN_WINDOW_FRAC * WINDOW_S:
            continue                                  # drop only a too-short trailing window

        vv, ww, dd = v_ms[idx], yaw[idx], dt[idx]
        tl = rel[idx] - rel[idx[0]]
        theta = np.cumsum(ww * dd); theta -= theta[0]
        x = np.cumsum(vv * np.cos(theta) * dd); x -= x[0]
        y = np.cumsum(vv * np.sin(theta) * dd); y -= y[0]

        if a is not None and np.isfinite(a[idx]).any():
            acc = a[idx]
        else:
            acc = np.gradient(vv, tl) if len(tl) > 2 else np.zeros_like(vv)
        acc = np.nan_to_num(acc, nan=0.0, posinf=0.0, neginf=0.0)

        recs.append({
            "x": x.astype(np.float32), "y": y.astype(np.float32),
            "t": tl.astype(np.float32), "speed_kmh": (vv * 3.6).astype(np.float32),
            "heading_change": float(np.sum(np.abs(ww) * dd)),
            "peak_acc": float(np.max(np.abs(acc))) if acc.size else 0.0,
        })
    return recs


def check_schema(fp, ds):
    """Print the schema for one dataset and report whether speed+yaw are present."""
    try:
        cols = list(pd.read_parquet(fp).columns)
    except Exception as e:
        print(f"  [SCHEMA] {ds}: cannot read {fp.name}: {e}")
        return False
    miss = [c for c in (SPEED_COL, YAW_COL) if c not in cols]
    if miss:
        print(f"  [SCHEMA] {ds}: MISSING {miss}  ->  skipped   (present: {cols})")
        return False
    print(f"  [SCHEMA] {ds}: ok (acc_mag={'present' if ACCEL_COL in cols else 'derived from speed'})")
    return True


def _clip_windows(fp, ds):
    """Thread worker: read one clip, return its window records (or [])."""
    try:
        frames = load_frames(fp, ds)
    except Exception:
        return []
    return window_records(frames, ds) if frames is not None else []


def order_records(records, key, mode):
    if mode == "curvature_straight_top":
        return sorted(records, key=lambda r: r["heading_change"], reverse=True)
    if mode == "scalar_extreme_top":
        return sorted(records, key=lambda r: r[key])
    return records


def _norm(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return Normalize(0, 1)
    lo, hi = np.percentile(arr, [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    return Normalize(vmin=lo, vmax=hi)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — X-Y trajectory fan (origin at left-middle)
# ═════════════════════════════════════════════════════════════════════════════
def _draw_xy_panel(ax, recs, cmap, norm, xlim, ylim):
    for r in order_records(recs, "heading_change", XY_SORT):
        ax.plot(r["x"], r["y"], color=cmap(norm(r["heading_change"])),
                linewidth=LINEWIDTH, alpha=ALPHA)
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    if ASPECT_EQUAL:
        ax.set_aspect("equal")
    ax.grid(True); ax.set_axisbelow(True)


def plot_xy_fan(per_ds, out_dir):
    cmap = plt.get_cmap(COLORMAP)
    norm = _norm([r["heading_change"] for recs in per_ds.values() for r in recs])

    xmax = np.percentile([float(r["x"].max()) for recs in per_ds.values() for r in recs], EXTENT_PCT)
    ymax = np.percentile([np.abs(r["y"]).max() for recs in per_ds.values() for r in recs], EXTENT_PCT)
    xmax = max(xmax, 1.0); ymax = max(ymax, 1.0)
    xlim = (-0.05 * xmax, xmax * 1.05)     # origin sits at the LEFT-MIDDLE
    ylim = (-ymax * 1.05, ymax * 1.05)
    label = "Absolute total heading change (rad)"

    # shared norm + shared extent ⇒ per-dataset figures stay directly comparable
    if SEPARATE_FILES:
        for ds in per_ds:
            fig, ax = plt.subplots(figsize=SINGLE_IN_XY)
            _draw_xy_panel(ax, per_ds[ds], cmap, norm, xlim, ylim)
            sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
            fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02).set_label(label)
            ax.set_title(f"Trajectory distribution {DISPLAY_NAMES.get(ds, ds)}  (window = {WINDOW_S:g}s)")
            fig.savefig(out_dir / f"trajectory_fan_xy_{ds}_w{int(WINDOW_S)}.png",
                        dpi=DPI, bbox_inches="tight")
            plt.close(fig)
        if not GRID_FILES:
            return

    # comparison grid: all datasets as small multiples on the shared extent and colour scale
    order = _panel_order(per_ds)
    panel_w = PANEL_IN * 1.15
    ratio = (ylim[1] - ylim[0]) / max(xlim[1] - xlim[0], 1e-9)      # data aspect of the shared extent
    panel_h = max(panel_w * ratio, 1.4) if ASPECT_EQUAL else PANEL_IN * 0.75
    fig, axes = _grid_figure(len(order), panel_w, panel_h)
    for ax, ds in zip(axes.ravel(), order):
        _draw_xy_panel(ax, per_ds[ds], cmap, norm, xlim, ylim)
        ax.set_title(DISPLAY_NAMES.get(ds, ds))
    _finish_grid(fig, axes, len(order), "X position (m)", "Y position (m)")
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02).set_label(label)
    plt.savefig(out_dir / f"trajectory_fan_xy_grid_w{int(WINDOW_S)}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — speed-vs-time fan
# ═════════════════════════════════════════════════════════════════════════════
def _draw_speed_panel(ax, recs, cmap, norm, vmax):
    for r in order_records(recs, "peak_acc", ST_SORT):
        ax.plot(r["t"], r["speed_kmh"] - r["speed_kmh"][0],
                color=cmap(norm(r["peak_acc"])), linewidth=LINEWIDTH, alpha=ALPHA)
    ax.axhline(0, color="0.5", linewidth=0.8, zorder=1)
    ax.set_xlabel("Time within window (s)")
    ax.set_ylabel("Speed change from start (km/h)")
    ax.set_xlim(0, WINDOW_S); ax.set_ylim(-vmax * 1.05, vmax * 1.05)
    ax.grid(True); ax.set_axisbelow(True)


def plot_speed_time_fan(per_ds, out_dir):
    cmap = plt.get_cmap(COLORMAP)
    norm = _norm([r["peak_acc"] for recs in per_ds.values() for r in recs])
    # speed CHANGE from each window's start: isolates within-window dynamics from the
    # baseline cruising speed, so accel/decel shape is visible instead of a flat band.
    dvals = [float(np.abs(r["speed_kmh"] - r["speed_kmh"][0]).max())
             for recs in per_ds.values() for r in recs if r["speed_kmh"].size]
    vmax = max(float(np.percentile(dvals, SPEED_PCT)), 1.0)
    label = "Peak acceleration magnitude (m/s\u00b2)"

    if SEPARATE_FILES:
        for ds in per_ds:
            fig, ax = plt.subplots(figsize=SINGLE_IN_SPEED)
            _draw_speed_panel(ax, per_ds[ds], cmap, norm, vmax)
            sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
            fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02).set_label(label)
            ax.set_title(f"Speed change per {WINDOW_S:g}s window ({DISPLAY_NAMES.get(ds, ds)})")
            fig.savefig(out_dir / f"speed_change_fan_{ds}_w{int(WINDOW_S)}.png",
                        dpi=DPI, bbox_inches="tight")
            plt.close(fig)
        if not GRID_FILES:
            return

    order = _panel_order(per_ds)
    fig, axes = _grid_figure(len(order), PANEL_IN * 1.05, PANEL_IN * 0.85)
    for ax, ds in zip(axes.ravel(), order):
        _draw_speed_panel(ax, per_ds[ds], cmap, norm, vmax)
        ax.set_title(DISPLAY_NAMES.get(ds, ds))
    _finish_grid(fig, axes, len(order), "Time within window (s)", "Speed change from start (km/h)")
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02).set_label(label)
    plt.savefig(out_dir / f"speed_change_fan_grid_w{int(WINDOW_S)}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    t_start = datetime.now()

    datasets = DATASETS if DATASETS else discover_datasets()
    if not datasets:
        print(f"[ERROR] No dataset folders with clip_*.parquet under {INERTIAL_ROOT}")
        sys.exit(1)

    print(f"\n{'='*72}")
    print(f"  trajectory_fans.py   window = {WINDOW_S:g}s (non-overlapping)")
    print(f"  inertial_data → {INERTIAL_ROOT}")
    print(f"  Datasets ({len(datasets)}): {', '.join(datasets)}")
    print(f"  max windows/dataset: {N_SAMPLE}")
    print(f"  Started  → {_now()}")
    print(f"{'='*72}\n")

    per_ds = {}
    print(f"  {'dataset':<12} {'windows':>8} {'med dur(s)':>11} {'med path(m)':>12} {'95p path(m)':>12} {'med |dθ|(rad)':>14}")
    for ds in datasets:
        clips = list_clips(ds)
        if not clips:
            print(f"  [WARN] {ds}: no clips under {INERTIAL_ROOT / ds}")
            continue
        if not check_schema(clips[0], ds):
            continue
        if N_SAMPLE is not None and len(clips) > N_SAMPLE:
            clips = random.sample(clips, N_SAMPLE)     # >=1 window/clip, so this is plenty

        recs = []
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = [ex.submit(_clip_windows, fp, ds) for fp in clips]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"  reading {ds:<12}", leave=False):
                recs.extend(fut.result())
        if not recs:
            continue
        if N_SAMPLE is not None and len(recs) > N_SAMPLE:
            recs = random.sample(recs, N_SAMPLE)
        per_ds[ds] = recs

        # path-length + duration stats help you judge the WINDOW_S choice
        plen = np.array([float(np.hypot(np.diff(r["x"]), np.diff(r["y"])).sum()) for r in recs])
        dth  = np.array([r["heading_change"] for r in recs])
        wdur = np.array([float(r["t"][-1]) for r in recs])
        print(f"  {ds:<12} {len(recs):>8} {np.median(wdur):>11.1f} {np.median(plen):>12.1f} "
              f"{np.percentile(plen,95):>12.1f} {np.median(dth):>14.3f}")

    if not per_ds:
        print("\n[ERROR] No data — check INERTIAL_ROOT / column names.")
        sys.exit(1)

    print(f"\n  Plotting ... [{_now()}]")
    plot_xy_fan(per_ds, FIG_ROOT)
    plot_speed_time_fan(per_ds, FIG_ROOT)

    dt = (datetime.now() - t_start).total_seconds()
    n = len(per_ds)
    print(f"\n{'='*72}")
    print(f"  \u2713 DONE  —  {n} datasets  \u00b7  window {WINDOW_S:g}s  \u00b7  {dt:.1f}s  \u00b7  {_now()}")
    if SEPARATE_FILES:
        print(f"  \u2192 {n} files  trajectory_fan_xy_<dataset>_w{int(WINDOW_S)}.png")
        print(f"  \u2192 {n} files  speed_change_fan_<dataset>_w{int(WINDOW_S)}.png")
    else:
        print(f"  \u2192 trajectory_fan_xy_w{int(WINDOW_S)}.png")
        print(f"  \u2192 speed_change_fan_w{int(WINDOW_S)}.png")
    print(f"  \u2192 in {FIG_ROOT}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()