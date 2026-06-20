"""
density_band_analysis.py
────────────────────────────────────────────────────────────────────────────
Per-dataset stacked multi-signal density bands (C4 layout × C5 style).
For each dataset, ONE figure with three vertically stacked panels — speed,
accel-magnitude, yaw rate — sharing the within-window time axis. Each panel
shows, across all of the dataset's 20 s windows:
  • median curve
  • interquartile band (25–75 %)
  • wide band (10–90 %)

Replaces the earlier representative-clip tier grid. Group = all windows of a
dataset (capped at N_SAMPLE for I/O). Windows come from std_data; frames are
recovered through the back-reference contract (clip_id, window_start_s,
window_end_s) against the standardized clips.

NOTE on option (a): aggregating every window of a dataset mixes calm and
dynamic windows, so for the calm-skewed datasets the bands will be wide and the
median roughly flat — that is the expected signature, and band WIDTH across
datasets is still a usable read (narrow = homogeneous, wide = diverse).

Threaded parquet reads + tqdm; matplotlib rendering single-threaded (not
thread-safe). Pure default style; dpi=300; tum_colors; all tunables top-level.
"""

import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from tum_colors import TUM_PRIMARY, TUM_SECONDARY, TUM_EXT

warnings.filterwarnings("ignore")   # silence all-NaN-slice warnings from nanpercentile

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  (data and code live in separate folders)
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT      = Path("/Users/edward/Downloads")
STD_DATA_DIR      = PROJECT_ROOT / "dataset" / "std_data"
INERTIAL_DATA_DIR = PROJECT_ROOT / "dataset" / "inertial_data"
OUT_DIR           = PROJECT_ROOT / "bachelor-arbeit" / "inertial_exploratory_analysis" / "figures" / "density_band"

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS  (all tunables here — never hardcoded inside functions)
# ─────────────────────────────────────────────────────────────────────────────

WINDOW_S        = 20.0
GRID_HZ         = 10
DT_GRID         = 0.1
WINDOW_FRAMES   = int(WINDOW_S * GRID_HZ)   # 200
N_SAMPLE        = None     # max windows aggregated per dataset (None = all)
MIN_WINDOWS     = 20          # need at least this many to draw a dataset
RANDOM_SEED     = 0
N_WORKERS       = 12
DPI             = 300

BAND_OUTER      = (10.0, 90.0)
BAND_INNER      = (25.0, 75.0)
ALPHA_OUTER     = 0.20
ALPHA_INNER     = 0.40
MEDIAN_LW       = 1.8
GRID_ALPHA      = 0.3
FIG_W           = 9.0
FIG_H_PER_PANEL = 2.2

# Stacked panels: (col, label, unit, lo, hi, rad_to_deg) — physical bounds masked at viz time
SIGNALS = [
    {"col": "speed_kmh", "label": "Speed",            "unit": "km/h", "lo": 0.0,   "hi": 200.0, "rad_to_deg": False},
    {"col": "acc_mag",   "label": "Accel. magnitude", "unit": "m/s²", "lo": 0.0,   "hi": 21.3,  "rad_to_deg": False},
    {"col": "yaw_rate",  "label": "Yaw rate",         "unit": "°/s",  "lo": -90.0, "hi": 90.0,  "rad_to_deg": True},
]
SIGNAL_COLS = [s["col"] for s in SIGNALS]
NEEDED_COLS = ["t_rel_s"] + SIGNAL_COLS

DISPLAY_NAMES = {
    "boreas_rt": "Boreas", "durlar": "DurLAR", "ford": "Ford", "kitti": "KITTI",
    "kitti360": "KITTI-360", "nuscene": "nuScenes", "nvidia": "NVIDIA",
    "oxford": "Oxford", "waymo": "Waymo", "zod": "ZOD",
    "4seasons": "4Seasons", "hdd": "HDD", "a2d2": "A2D2", "kitscenes": "KITScenes"
}
DATASET_ORDER = list(DISPLAY_NAMES.keys())   # 13 datasets

DATASET_COLORS = {
    "4seasons":  TUM_SECONDARY['blue2'],
    "a2d2":      TUM_SECONDARY['dark_gray'],
    "hdd":       TUM_SECONDARY['gray'],
    "kitscenes": TUM_SECONDARY['light_gray'],
    "boreas_rt": TUM_EXT['maroon'],
    "durlar":    TUM_SECONDARY['blue1'],
    "ford":      TUM_EXT['orange'],
    "kitti":     TUM_EXT['yellow'],
    "kitti360":  TUM_EXT['olive'],
    "nuscene":   TUM_EXT['dark_green'],
    "nvidia":    TUM_EXT['teal'],
    "oxford":    TUM_EXT['navy'],
    "waymo":     TUM_EXT['purple'],
    "zod":       TUM_EXT['red'],
}
DATASET_COLOR_FALLBACK = TUM_SECONDARY['blue2']

# ─────────────────────────────────────────────────────────────────────────────
# DATA  (threaded clip reads + tqdm; rendering stays single-threaded)
# ─────────────────────────────────────────────────────────────────────────────

def _clip_path(folder, clip_id):
    return INERTIAL_DATA_DIR / folder / f"clip_{clip_id:04d}.parquet"


def _read_clip_windows(task):
    """Read one clip, slice its sampled windows, return {col: list of 200-frame buffers}."""
    folder, clip_id, windows = task
    fp = _clip_path(folder, clip_id)
    if not fp.exists():
        return None
    try:
        import pyarrow.parquet as pq
        schema_cols = set(pq.read_schema(fp).names)
        cols = [c for c in NEEDED_COLS if c in schema_cols]
        df = pd.read_parquet(fp, columns=cols)
        for c in NEEDED_COLS:
            if c not in df.columns:
                df[c] = np.nan
    except Exception:
        return None

    t = df["t_rel_s"].to_numpy(dtype=float)
    out = {c: [] for c in SIGNAL_COLS}
    for (start, end) in windows:
        m = (t >= start) & (t < end)
        if m.sum() < 2:
            continue
        for c in SIGNAL_COLS:
            seg = df[c].to_numpy(dtype=float)[m]
            buf = np.full(WINDOW_FRAMES, np.nan)
            k = min(len(seg), WINDOW_FRAMES)
            buf[:k] = seg[:k]
            out[c].append(buf)
    return out


def load_dataset_windows(folder):
    std_fp = STD_DATA_DIR / f"{folder}.parquet"
    if not std_fp.exists():
        return None
    rows = pd.read_parquet(std_fp, columns=["clip_id", "window_start_s", "window_end_s"])
    if len(rows) == 0:
        return None
    if N_SAMPLE is not None and len(rows) > N_SAMPLE:
        rows = rows.sample(N_SAMPLE, random_state=RANDOM_SEED)

    tasks = [
        (folder, int(cid), list(zip(g["window_start_s"].to_numpy(dtype=float),
                                    g["window_end_s"].to_numpy(dtype=float))))
        for cid, g in rows.groupby("clip_id")
    ]

    stacks = {c: [] for c in SIGNAL_COLS}
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        for res in tqdm(ex.map(_read_clip_windows, tasks), total=len(tasks),
                        desc=DISPLAY_NAMES.get(folder, folder), unit="clip"):
            if res is None:
                continue
            for c in SIGNAL_COLS:
                stacks[c].extend(res[c])

    arrays = {}
    n_windows = 0
    for c in SIGNAL_COLS:
        arrays[c] = np.vstack(stacks[c]) if stacks[c] else None
        if arrays[c] is not None:
            n_windows = arrays[c].shape[0]
    return arrays, n_windows

# ─────────────────────────────────────────────────────────────────────────────
# PLOT (one stacked figure per dataset)
# ─────────────────────────────────────────────────────────────────────────────

def render(folder, arrays, n_windows):
    color = DATASET_COLORS.get(folder, DATASET_COLOR_FALLBACK)
    t = np.arange(WINDOW_FRAMES) * DT_GRID

    fig, axes = plt.subplots(len(SIGNALS), 1,
                             figsize=(FIG_W, FIG_H_PER_PANEL * len(SIGNALS)),
                             sharex=True)
    axes = np.atleast_1d(axes)

    for ax, sig in zip(axes, SIGNALS):
        A = arrays.get(sig["col"])
        ax.set_ylabel(f"{sig['label']}\n[{sig['unit']}]")
        ax.grid(True, alpha=GRID_ALPHA)
        ax.set_axisbelow(True)
        if A is None:
            continue
        A = A.astype(float)
        if sig["rad_to_deg"]:
            A = A * (180.0 / np.pi)
        A = np.where((A >= sig["lo"]) & (A <= sig["hi"]), A, np.nan)

        med = np.nanmedian(A, axis=0)
        lo_o, hi_o = np.nanpercentile(A, BAND_OUTER, axis=0)
        lo_i, hi_i = np.nanpercentile(A, BAND_INNER, axis=0)

        ax.fill_between(t, lo_o, hi_o, color=color, alpha=ALPHA_OUTER, linewidth=0,
                        label=f"{int(BAND_OUTER[0])}–{int(BAND_OUTER[1])}%")
        ax.fill_between(t, lo_i, hi_i, color=color, alpha=ALPHA_INNER, linewidth=0,
                        label=f"{int(BAND_INNER[0])}–{int(BAND_INNER[1])}% (IQR)")
        ax.plot(t, med, color=color, linewidth=MEDIAN_LW, label="median")

    axes[-1].set_xlabel("time within window [s]")
    axes[0].legend(loc="upper right", framealpha=0.9)
    fig.suptitle(f"{DISPLAY_NAMES.get(folder, folder)} — multi-signal density bands "
                 f"across {n_windows} windows")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{folder}_density_bands.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    count = 0
    for folder in DATASET_ORDER:
        res = load_dataset_windows(folder)
        if res is None:
            print(f"[DB] {folder}: no std_data, skipped")
            continue
        arrays, n_windows = res
        if n_windows < MIN_WINDOWS:
            print(f"[DB] {folder}: only {n_windows} windows (< {MIN_WINDOWS}), skipped")
            continue
        out = render(folder, arrays, n_windows)
        print(f"[DB] saved {out}  ({n_windows} windows)")
        count += 1
    print(f"[DB] done — {count} figures to {OUT_DIR}")


if __name__ == "__main__":
    main()