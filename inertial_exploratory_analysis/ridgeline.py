"""
extra_plot_a2_ridgeline.py
────────────────────────────────────────────────────────────────────────────
A2 — Per-window variability ridgelines. THREE separate figures, one per signal
(speed, accel-magnitude, yaw rate). Each figure overlays all datasets as
ridgeline rows, ordered by that signal's own median (calmest at the bottom).
Supports RQ1: the SHAPE of dynamic skew across datasets.

Source: std_data (3-signal schema). Threaded parquet load + tqdm; matplotlib
rendering single-threaded (not thread-safe). Pure default style; dpi=300;
tum_colors; all tunables top-level.
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from tqdm import tqdm

from tum_colors import TUM_PRIMARY, TUM_SECONDARY, TUM_EXT

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  (data and code live in separate folders)
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path("/Users/edward/Downloads")
STD_DATA_DIR = PROJECT_ROOT / "dataset" / "std_data"
OUT_DIR      = PROJECT_ROOT / "bachelor-arbeit" / "inertial_exploratory_analysis" / "figures" / "ridgeline"

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS  (all tunables here — never hardcoded inside functions)
# ─────────────────────────────────────────────────────────────────────────────

X_PERCENTILE_CAP   = 99.0
N_GRID             = 512
ROW_OVERLAP        = 1.45
MIN_POINTS         = 8
N_WORKERS          = 12
DPI                = 300

DENSITY_ALPHA      = 0.6
DENSITY_EDGECOLOR  = "white"
DENSITY_LINEWIDTH  = 0.8
MEDIAN_TICK_HEIGHT = 0.18
MEDIAN_TICK_COLOR  = "black"
MEDIAN_TICK_LW     = 1.0
MEDIAN_TICK_ALPHA  = 0.6
GRID_ALPHA         = 0.3

FIG_W              = 8.5
FIG_H_PER_ROW      = 0.55
FIG_H_PAD          = 1.8

# Signals: (std_col, tag, label, unit, rad_to_deg) — one figure each
SIGNALS = [
    ("std_speed_kmh", "speed",   "σ(speed)",            "km/h", False),
    ("std_acc_mag",   "acc_mag", "σ(accel. magnitude)", "m/s²", False),
    ("std_yaw_rate",  "yaw",     "σ(yaw rate)",         "°/s",  True),   # rad/s -> deg/s
]

DISPLAY_NAMES = {
    "boreas_rt": "Boreas", "durlar": "DurLAR", "ford": "Ford", "kitti": "KITTI",
    "kitti360": "KITTI-360", "nuscene": "nuScenes", "nvidia": "NVIDIA",
    "oxford": "Oxford", "waymo": "Waymo", "zod": "ZOD",
    "4seasons": "4Seasons", "hdd": "HDD", "a2d2": "A2D2", "kitscenes": "KITScenes"
}
DATASET_ORDER = list(DISPLAY_NAMES.keys())   # 13 datasets

DATASET_COLORS = {
    "4seasons":   TUM_SECONDARY['blue2'],
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
# DATA  (threaded load + tqdm; rendering stays single-threaded)
# ─────────────────────────────────────────────────────────────────────────────

def _read_std(task):
    folder, fp = task
    df = pd.read_parquet(fp)
    return folder, {sc: df[sc].to_numpy(dtype=float) for sc, *_ in SIGNALS if sc in df.columns}


def load_std_data() -> dict:
    tasks = [(f, STD_DATA_DIR / f"{f}.parquet") for f in DATASET_ORDER
             if (STD_DATA_DIR / f"{f}.parquet").exists()]
    data = {}
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(_read_std, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="load std_data", unit="file"):
            folder, d = fut.result()
            data[folder] = d
    return data


def order_by_median(data, std_col, rad_to_deg) -> list:
    folders = [f for f in DATASET_ORDER if f in data and std_col in data[f]]

    def med(f):
        v = data[f][std_col]
        v = v[np.isfinite(v)]
        if rad_to_deg:
            v = v * (180.0 / np.pi)
        v = v[v >= 0.0]
        return np.median(v) if v.size else np.inf

    return sorted(folders, key=med)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT (one figure per signal)
# ─────────────────────────────────────────────────────────────────────────────

def draw_ridgeline(data, std_col, tag, label, unit, rad_to_deg):
    order = order_by_median(data, std_col, rad_to_deg)
    vals = {}
    for f in order:
        v = data[f][std_col]
        v = v[np.isfinite(v)]
        if rad_to_deg:
            v = v * (180.0 / np.pi)
        v = v[v >= 0.0]
        if v.size >= MIN_POINTS and np.std(v) > 0:
            vals[f] = v
    order = [f for f in order if f in vals]
    if not order:
        print(f"[A2] {tag}: no drawable datasets")
        return

    pooled = np.concatenate([vals[f] for f in order])
    x_cap = np.percentile(pooled, X_PERCENTILE_CAP)
    xs = np.linspace(0.0, max(x_cap, 1e-6), N_GRID)
    n = len(order)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_PER_ROW * n + FIG_H_PAD))
    for i, f in enumerate(order):
        vc = vals[f][vals[f] <= x_cap]
        if vc.size < MIN_POINTS or np.std(vc) == 0:
            continue
        dens = gaussian_kde(vc)(xs)
        dens = dens / dens.max()
        base = float(i)
        color = DATASET_COLORS.get(f, DATASET_COLOR_FALLBACK)
        ax.fill_between(xs, base, base + dens * ROW_OVERLAP, color=color,
                        alpha=DENSITY_ALPHA, linewidth=DENSITY_LINEWIDTH,
                        edgecolor=DENSITY_EDGECOLOR, zorder=n - i)
        med = float(np.median(vc))
        ax.plot([med, med], [base, base + MEDIAN_TICK_HEIGHT],
                color=MEDIAN_TICK_COLOR, linewidth=MEDIAN_TICK_LW,
                alpha=MEDIAN_TICK_ALPHA, zorder=n - i + 0.5)

    ax.set_yticks(range(n))
    ax.set_yticklabels([DISPLAY_NAMES[f] for f in order])
    ax.set_ylim(-0.3, n - 1 + ROW_OVERLAP + 0.2)
    ax.set_xlim(0.0, x_cap)
    ax.set_xlabel(f"{label} per 20 s window [{unit}]")
    ax.set_title(f"Per-window variability across datasets — {label}")
    ax.grid(True, axis="x", alpha=GRID_ALPHA)
    ax.set_axisbelow(True)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"ridgeline_{tag}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[A2] saved {out}")


def main():
    data = load_std_data()
    if not data:
        print(f"[A2] no std_data parquets found in {STD_DATA_DIR}")
        return
    for std_col, tag, label, unit, r2d in SIGNALS:
        draw_ridgeline(data, std_col, tag, label, unit, r2d)


if __name__ == "__main__":
    main()