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
FIG_ROOT      = PROJECT_ROOT / "bachelor-arbeit" / "inertial_exploratory_analysis" / "figures" / "trajectory_fans"

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
N_SAMPLE   = 1500     # max WINDOWS drawn per dataset (None = all). Lines get dense fast.
SEED       = 0
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
def plot_xy_fan(per_ds, out_path):
    cmap = plt.get_cmap(COLORMAP)
    norm = _norm([r["heading_change"] for recs in per_ds.values() for r in recs])

    xmax = np.percentile([float(r["x"].max()) for recs in per_ds.values() for r in recs], EXTENT_PCT)
    ymax = np.percentile([np.abs(r["y"]).max() for recs in per_ds.values() for r in recs], EXTENT_PCT)
    xmax = max(xmax, 1.0); ymax = max(ymax, 1.0)
    xlim = (-0.05 * xmax, xmax * 1.05)     # origin sits at the LEFT-MIDDLE
    ylim = (-ymax * 1.05, ymax * 1.05)

    nrows, ncols = _grid(len(per_ds))
    fig, axes = plt.subplots(nrows, ncols, figsize=(PANEL_IN * ncols, PANEL_IN * nrows), squeeze=False)
    axes = axes.ravel()
    for ax, ds in zip(axes, per_ds.keys()):
        for r in order_records(per_ds[ds], "heading_change", XY_SORT):
            ax.plot(r["x"], r["y"], color=cmap(norm(r["heading_change"])),
                    linewidth=LINEWIDTH, alpha=ALPHA)
        ax.set_title(ds)
        ax.set_xlabel("X position (m)")
        ax.set_ylabel("Y position (m)")
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        if ASPECT_EQUAL:
            ax.set_aspect("equal")
        ax.grid(True); ax.set_axisbelow(True)
    for ax in axes[len(per_ds):]:
        ax.axis("off")

    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("Absolute total heading change (rad)")
    fig.suptitle(f"Trajectory distribution per dataset  (window = {WINDOW_S:g}s, non-overlapping)")
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — speed-vs-time fan
# ═════════════════════════════════════════════════════════════════════════════
def plot_speed_time_fan(per_ds, out_path):
    cmap = plt.get_cmap(COLORMAP)
    norm = _norm([r["peak_acc"] for recs in per_ds.values() for r in recs])
    vmax = np.percentile([float(r["speed_kmh"].max()) for recs in per_ds.values() for r in recs], SPEED_PCT)
    vmax = max(vmax, 1.0)

    nrows, ncols = _grid(len(per_ds))
    fig, axes = plt.subplots(nrows, ncols, figsize=(PANEL_IN * 1.15 * ncols, PANEL_IN * nrows), squeeze=False)
    axes = axes.ravel()
    for ax, ds in zip(axes, per_ds.keys()):
        for r in order_records(per_ds[ds], "peak_acc", ST_SORT):
            ax.plot(r["t"], r["speed_kmh"], color=cmap(norm(r["peak_acc"])),
                    linewidth=LINEWIDTH, alpha=ALPHA)
        ax.set_title(ds)
        ax.set_xlabel("Time within window (s)")
        ax.set_ylabel("Speed (km/h)")
        ax.set_xlim(0, WINDOW_S); ax.set_ylim(0, vmax * 1.05)
        ax.grid(True); ax.set_axisbelow(True)
    for ax in axes[len(per_ds):]:
        ax.axis("off")

    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("Peak acceleration magnitude (m/s\u00b2)")
    fig.suptitle(f"Speed profiles per dataset  (window = {WINDOW_S:g}s; slope = acceleration)")
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
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
    plot_xy_fan(per_ds, FIG_ROOT / f"trajectory_fan_xy_w{int(WINDOW_S)}.png")
    plot_speed_time_fan(per_ds, FIG_ROOT / f"speed_time_fan_w{int(WINDOW_S)}.png")

    dt = (datetime.now() - t_start).total_seconds()
    print(f"\n{'='*72}")
    print(f"  \u2713 DONE  —  {len(per_ds)} datasets  \u00b7  window {WINDOW_S:g}s  \u00b7  {dt:.1f}s  \u00b7  {_now()}")
    print(f"  \u2192 trajectory_fan_xy_w{int(WINDOW_S)}.png")
    print(f"  \u2192 speed_time_fan_w{int(WINDOW_S)}.png")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()