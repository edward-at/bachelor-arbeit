"""
density_heatmap.py   (windowed)
===============================
Per-dataset trajectory DENSITY heatmaps, built from NON-OVERLAPPING 20 s windows
(same pipeline as trajectory_fans.py): each clip is sliced into back-to-back
windows, every window's ego path is reconstructed from speed + yaw and rasterized
onto a shared grid. Each pixel's colour = how often windows occupy it.

  1 clip -> non-overlapping WINDOW_S windows -> each window rasterized into the grid

Origin sits at the LEFT-MIDDLE (x from 0, y symmetric), so the cloud opens right.
Windows are streamed straight into the histogram, so memory stays flat at any size.

DENSITY_MODE:
  "fraction" : fraction of WINDOWS that pass through each cell  [DEFAULT, Hz-fair]
  "time"     : seconds spent in each cell (points weighted by dt)  [Hz-fair]
  "count"    : raw point count per cell  [Hz-BIASED — not for cross-dataset comparison]
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
from matplotlib.colors import LogNorm, Normalize
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
FIG_ROOT      = PROJECT_ROOT / "bachelor-arbeit" / "inertial_exploratory_analysis" / "figures" / "density_heatmap"

DATASETS = None   # None -> auto-discover EVERY folder under inertial_data/ (all 14).

# ═════════════════════════════════════════════════════════════════════════════
# COLUMN NAMES  ── confirm against the [SCHEMA] print, then edit
# ═════════════════════════════════════════════════════════════════════════════
SPEED_COL = "speed_kmh"
YAW_COL   = "yaw_rate"     # rad/s
TIME_COL  = "t_rel_s"      # real relative-seconds column (uniform 10 Hz grid). Falls back to
                           # GRID_HZ only if absent.

GRID_HZ = 10               # uniform grid rate of the standardized clips. ONLY a fallback if
                           # TIME_COL is missing; with t_rel_s present this is never used.

# ═════════════════════════════════════════════════════════════════════════════
# WINDOWING  (same definition as trajectory_fans.py)
# ═════════════════════════════════════════════════════════════════════════════
WINDOW_S        = 20.0     # seconds per window (non-overlapping)
MIN_WINDOW_FRAC = 0.5      # drop a trailing window shorter than this fraction of WINDOW_S

# ═════════════════════════════════════════════════════════════════════════════
# WHAT TO PLOT
# ═════════════════════════════════════════════════════════════════════════════
N_SAMPLE     = None        # max WINDOWS per dataset (None = all; density handles volume)
SEED         = 0
N_WORKERS    = 8           # threads for reading parquet in parallel (reading is I/O-bound)
DENSITY_MODE = "fraction"  # "fraction" | "time" | "count"

# ═════════════════════════════════════════════════════════════════════════════
# GRID / COLOUR
# ═════════════════════════════════════════════════════════════════════════════
NBINS         = 300
EXTENT_PCT    = 99.0       # robust extent = this percentile of per-window max x / |y|
EXTENT_SAMPLE = 400        # clips per dataset used to estimate the extent (cheap pass)
LOG_COLOR     = True
COLORMAP      = "magma"
ASPECT_EQUAL  = True       # True = 1 m x == 1 m y (honest; forward-heavy sets look wide).
                           # False = panels fill a box (paper-like, but x/y scales differ).

# ═════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ═════════════════════════════════════════════════════════════════════════════
NCOLS    = 5
PANEL_IN = 3.4
DPI      = 300

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
    if not INERTIAL_ROOT.exists():
        return []
    return [d.name for d in sorted(INERTIAL_ROOT.iterdir())
            if d.is_dir() and any(d.glob("clip_*.parquet"))]


def load_frames(fp, ds):
    """Masked full-clip arrays (v_ms, yaw, t, dt) or None."""
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

    m = np.isfinite(v_ms) & np.isfinite(yaw) & np.isfinite(t)
    if m.sum() < 5:
        return None
    return v_ms[m], yaw[m], t[m], dt[m]


def window_paths(frames):
    """Yield (x, y, dt) per non-overlapping WINDOW_S window, anchored at origin.
    O(N) (immune to giant/time-corrupt clips); keeps a short clip as one window."""
    v_ms, yaw, t, dt = frames
    rel = t - t[0]
    total = float(rel[-1])
    n = len(rel)

    if total < WINDOW_S:
        groups = [np.arange(n)]
    else:
        win_id = (rel / WINDOW_S).astype(np.int64)
        cut    = np.flatnonzero(np.diff(win_id)) + 1
        groups = np.split(np.arange(n), cut)

    for idx in groups:
        if idx.size < 5:
            continue
        span = rel[idx[-1]] - rel[idx[0]]
        if total >= WINDOW_S and span < MIN_WINDOW_FRAC * WINDOW_S:
            continue
        vv, ww, dd = v_ms[idx], yaw[idx], dt[idx]
        theta = np.cumsum(ww * dd); theta -= theta[0]
        x = np.cumsum(vv * np.cos(theta) * dd); x -= x[0]
        y = np.cumsum(vv * np.sin(theta) * dd); y -= y[0]
        yield x.astype(np.float32), y.astype(np.float32), dd.astype(np.float32)


def check_schema(fp, ds):
    """Print schema for one dataset; report whether speed+yaw are present."""
    try:
        cols = list(pd.read_parquet(fp).columns)
    except Exception as e:
        print(f"  [SCHEMA] {ds}: cannot read {fp.name}: {e}")
        return False
    miss = [c for c in (SPEED_COL, YAW_COL) if c not in cols]
    if miss:
        print(f"  [SCHEMA] {ds}: MISSING {miss}  ->  skipped   (present: {cols})")
        return False
    print(f"  [SCHEMA] {ds}: ok")
    return True


def _clip_paths(fp, ds):
    """Thread worker: read one clip, return its window (x, y, dt) tuples (or [])."""
    try:
        frames = load_frames(fp, ds)
    except Exception:
        return []
    return list(window_paths(frames)) if frames is not None else []


def accumulate(H, n_win, x, y, dt, xe, ye, step):
    if DENSITY_MODE == "fraction":
        dx = np.diff(x); dy = np.diff(y)
        s = np.concatenate([[0.0], np.cumsum(np.sqrt(dx * dx + dy * dy))])
        if s[-1] > 0 and np.isfinite(s[-1]):
            s_new = np.arange(0.0, s[-1], step)
            xi = np.interp(s_new, s, x); yi = np.interp(s_new, s, y)
        else:
            xi, yi = x, y
        ix = np.clip(np.searchsorted(xe, xi, side="right") - 1, 0, NBINS - 1)
        iy = np.clip(np.searchsorted(ye, yi, side="right") - 1, 0, NBINS - 1)
        inside = (xi >= xe[0]) & (xi <= xe[-1]) & (yi >= ye[0]) & (yi <= ye[-1])
        if inside.any():
            cells = np.unique(np.stack([ix[inside], iy[inside]], axis=1), axis=0)
            H[cells[:, 0], cells[:, 1]] += 1.0
    elif DENSITY_MODE == "time":
        h, _, _ = np.histogram2d(x, y, bins=[xe, ye], weights=dt); H += h
    else:
        h, _, _ = np.histogram2d(x, y, bins=[xe, ye]); H += h
    return n_win + 1


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
    print(f"  density_heatmap.py   mode='{DENSITY_MODE}'   window={WINDOW_S:g}s (non-overlapping)")
    print(f"  inertial_data → {INERTIAL_ROOT}")
    print(f"  Datasets ({len(datasets)}): {', '.join(datasets)}")
    print(f"  N_sample/dataset: {N_SAMPLE}   bins: {NBINS}")
    print(f"  Started  → {_now()}")
    print(f"{'='*72}\n")

    usable = {}
    for ds in datasets:
        clips = list_clips(ds)
        if not clips:
            continue
        if not check_schema(clips[0], ds):
            continue
        usable[ds] = clips
    if not usable:
        print("[ERROR] No usable datasets (speed/yaw missing everywhere).")
        sys.exit(1)

    # ---- pass 1: robust extent from a small sample of windows (threaded) -------
    print(f"  Estimating extent from up to {EXTENT_SAMPLE} clips/dataset ... [{_now()}]")
    xmax_list, ymax_list = [], []
    for ds, clips in usable.items():
        sample = clips if len(clips) <= EXTENT_SAMPLE else random.sample(clips, EXTENT_SAMPLE)
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = [ex.submit(_clip_paths, fp, ds) for fp in sample]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"  extent {ds:<10}", leave=False):
                for x, y, _ in fut.result():
                    if x.size:
                        xmax_list.append(float(x.max()))
                        ymax_list.append(float(np.abs(y).max()))

    if not xmax_list:
        print("[ERROR] No windows reconstructed — check column names / yaw availability.")
        sys.exit(1)

    x_hi = max(float(np.percentile(xmax_list, EXTENT_PCT)), 1.0)
    y_hi = max(float(np.percentile(ymax_list, EXTENT_PCT)), 1.0)
    x_lo = -0.05 * x_hi
    xe = np.linspace(x_lo, x_hi, NBINS + 1)
    ye = np.linspace(-y_hi, y_hi, NBINS + 1)
    step = 0.7 * min((x_hi - x_lo) / NBINS, (2 * y_hi) / NBINS)
    extent = [x_lo, x_hi, -y_hi, y_hi]
    print(f"  extent: x [{x_lo:.1f}, {x_hi:.1f}]  y [+/-{y_hi:.1f}] m  (origin at left-middle)")

    # ---- pass 2: stream all windows into per-dataset histograms (threaded) -----
    grids = {}
    for ds, clips in usable.items():
        use = clips
        if N_SAMPLE is not None and len(clips) > N_SAMPLE:
            use = random.sample(clips, N_SAMPLE)        # >=1 window/clip
        H = np.zeros((NBINS, NBINS), dtype=np.float64)
        n_win = 0
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = [ex.submit(_clip_paths, fp, ds) for fp in use]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"  reading {ds:<10}", leave=False):
                for x, y, dd in fut.result():
                    n_win = accumulate(H, n_win, x, y, dd, xe, ye, step)
        if n_win == 0:
            continue
        if DENSITY_MODE == "fraction":
            H = H / n_win
        grids[ds] = H
        print(f"\u2500\u2500 {ds:<12} windows: {n_win:>6}   peak: {H.max():.4g}")

    if not grids:
        print("[ERROR] No histograms built.")
        sys.exit(1)

    # ---- colour scale ---------------------------------------------------------
    positives = np.concatenate([g[g > 0].ravel() for g in grids.values()])
    vmax = float(positives.max())
    norm = LogNorm(vmin=float(positives.min()), vmax=vmax) if LOG_COLOR else Normalize(0.0, vmax)
    cmap = plt.get_cmap(COLORMAP)
    cbar_label = {"fraction": "Fraction of windows through cell",
                  "time": "Time spent in cell (s)",
                  "count": "Point count per cell"}[DENSITY_MODE]

    # ---- plot -----------------------------------------------------------------
    print(f"\n  Plotting ... [{_now()}]")
    nrows, ncols = _grid(len(grids))
    fig, axes = plt.subplots(nrows, ncols, figsize=(PANEL_IN * ncols, PANEL_IN * nrows), squeeze=False)
    axes = axes.ravel()
    im = None
    for ax, ds in zip(axes, grids.keys()):
        H = grids[ds]
        plotH = np.where(H > 0, H, np.nan) if LOG_COLOR else H
        im = ax.imshow(plotH.T, origin="lower", extent=extent, cmap=cmap, norm=norm,
                       aspect=("equal" if ASPECT_EQUAL else "auto"))
        ax.set_title(ds)
        ax.set_xlabel("X position (m)")
        ax.set_ylabel("Y position (m)")
        ax.grid(True, alpha=0.2)
        ax.set_axisbelow(True)
    for ax in axes[len(grids):]:
        ax.axis("off")

    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02)
    cbar.set_label(cbar_label)
    fig.suptitle(f"Trajectory density per dataset  (window = {WINDOW_S:g}s, mode = {DENSITY_MODE})")
    out = FIG_ROOT / f"density_heatmap_{DENSITY_MODE}_w{int(WINDOW_S)}.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    dt = (datetime.now() - t_start).total_seconds()
    print(f"\n{'='*72}")
    print(f"  \u2713 DONE  —  {len(grids)} datasets  \u00b7  {dt:.1f}s  \u00b7  {_now()}")
    print(f"  \u2192 {out.name}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()