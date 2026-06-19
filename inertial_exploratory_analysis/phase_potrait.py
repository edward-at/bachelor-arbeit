"""
extra_plot_a10_phase_portrait.py
────────────────────────────────────────────────────────────────────────────
A10 — Per-dataset kinematic state-space occupancy. For each dataset, TWO
figures: speed × longitudinal acceleration, and speed × yaw rate
(13 datasets × 2 = 26 figures). Supports RQ1 (calm-biased datasets concentrate
near low speed / near-zero accel|yaw) and RQ2 (dense regions ~ dynamic regimes).

Shared bins and (by default) a shared cross-dataset log color scale so the 26
per-dataset plots remain directly comparable; set SHARED_COLOR_SCALE=False for
per-plot autoscaling.

Source: frame-level inertial_data (speed_kmh ÷3.6 → m/s; a_long; yaw_rate →°/s).
Threaded, batched parquet reads + tqdm; matplotlib rendering single-threaded.
Pure default style; dpi=300; tum_colors; all tunables top-level.
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from tqdm import tqdm

from tum_colors import TUM_PRIMARY, TUM_SECONDARY, TUM_EXT

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  (data and code live in separate folders)
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT      = Path("/Users/edward/Downloads")
INERTIAL_DATA_DIR = PROJECT_ROOT / "dataset" / "inertial_data"
OUT_DIR           = PROJECT_ROOT / "bachelor-arbeit" / "inertial_exploratory_analysis" / "figures" / "phase_portrait"

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS  (all tunables here — never hardcoded inside functions)
# ─────────────────────────────────────────────────────────────────────────────

SPEED_MAX_MS = 40.0
A_LONG_ABS   = 8.0
YAW_ABS_DEG  = 60.0
N_BINS_SPEED = 80
N_BINS_Y     = 80
N_WORKERS    = 12
CHUNK        = 5000        # clip-read batch size (bounds in-flight futures / memory)
SHARED_COLOR_SCALE = True  # one cross-dataset log scale (False = per-plot autoscale)
VMIN_RATIO   = 1.0e4
DPI = 300

CMAP            = "viridis"
ZERO_LINE_COLOR = "white"
ZERO_LINE_LW    = 0.6
ZERO_LINE_ALPHA = 0.5
CENTROID_COLOR  = "red"
CENTROID_SIZE   = 30
CENTROID_MARKER = "x"
CENTROID_LW     = 1.5
FIG_W = 4.8
FIG_H = 4.2

# Optional K-means centroid overlay per view (array of (speed_ms, y)) or None.
CENTROIDS = {"speed_along": None, "speed_yaw": None}

SPEED_EDGES = np.linspace(0.0, SPEED_MAX_MS, N_BINS_SPEED + 1)
ALONG_EDGES = np.linspace(-A_LONG_ABS, A_LONG_ABS, N_BINS_Y + 1)
YAW_EDGES   = np.linspace(-YAW_ABS_DEG, YAW_ABS_DEG, N_BINS_Y + 1)

VIEWS = [
    {"tag": "speed_along", "y_abs": A_LONG_ABS, "y_label": "long. accel. [m/s²]",
     "title": "Speed × longitudinal acceleration"},
    {"tag": "speed_yaw",   "y_abs": YAW_ABS_DEG, "y_label": "yaw rate [°/s]",
     "title": "Speed × yaw rate"},
]

DISPLAY_NAMES = {
    "boreas_rt": "Boreas", "durlar": "DurLAR", "ford": "Ford", "kitti": "KITTI",
    "kitti360": "KITTI-360", "nuscene": "nuScenes", "nvidia": "NVIDIA",
    "oxford": "Oxford", "waymo": "Waymo", "zod": "ZOD",
    "4seasons": "4Seasons", "hdd": "HDD", "a2d2": "A2D2",
}
DATASET_ORDER = list(DISPLAY_NAMES.keys())   # 13 datasets

DATASET_COLORS = {
    "4seasons":  TUM_SECONDARY['blue2'],
    "a2d2":      TUM_SECONDARY['dark_gray'],
    "hdd":       TUM_SECONDARY['gray'],
    "kitscenes": TUM_SECONDARY['light_gray'],
    "boreas_rt": TUM_EXT['maroon'],
    "durlar":    TUM_EXT['gold'],
    "ford":      TUM_EXT['orange'],
    "kitti":     TUM_EXT['yellow'],
    "kitti360":  TUM_EXT['olive'],
    "nuscene":   TUM_EXT['dark_green'],
    "nvidia":    TUM_EXT['teal'],
    "oxford":    TUM_EXT['navy'],
    "waymo":     TUM_EXT['purple'],
    "zod":       TUM_EXT['red'],
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA  (threaded, batched parquet reads + tqdm; both views in one pass)
# ─────────────────────────────────────────────────────────────────────────────

def _clip_hists(fp):
    """Per-clip partial 2D histograms (along, yaw). Returns (H_along, H_yaw) or None."""
    try:
        df = pd.read_parquet(fp, columns=["speed_kmh", "a_long", "yaw_rate"])
    except Exception:
        return None
    sp = df["speed_kmh"].to_numpy(dtype=float) / 3.6
    al = df["a_long"].to_numpy(dtype=float)
    yw = df["yaw_rate"].to_numpy(dtype=float) * (180.0 / np.pi)
    ma = np.isfinite(sp) & np.isfinite(al)
    my = np.isfinite(sp) & np.isfinite(yw)
    Ha, _, _ = np.histogram2d(sp[ma], al[ma], bins=[SPEED_EDGES, ALONG_EDGES])
    Hy, _, _ = np.histogram2d(sp[my], yw[my], bins=[SPEED_EDGES, YAW_EDGES])
    return Ha, Hy


def load_dataset(folder) -> dict | None:
    clips = sorted((INERTIAL_DATA_DIR / folder).glob("clip_*.parquet"))
    if not clips:
        return None
    Ha = np.zeros((N_BINS_SPEED, N_BINS_Y))
    Hy = np.zeros((N_BINS_SPEED, N_BINS_Y))
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        with tqdm(total=len(clips), desc=DISPLAY_NAMES.get(folder, folder), unit="clip") as pbar:
            for i in range(0, len(clips), CHUNK):
                for r in ex.map(_clip_hists, clips[i:i + CHUNK]):
                    if r is not None:
                        Ha += r[0]
                        Hy += r[1]
                    pbar.update(1)
    out = {}
    for view in VIEWS:
        H = Ha if view["tag"] == "speed_along" else Hy
        tot = H.sum()
        out[view["tag"]] = H / tot if tot > 0 else H
    return out

# ─────────────────────────────────────────────────────────────────────────────
# PLOT (one figure per dataset per view)
# ─────────────────────────────────────────────────────────────────────────────

def render(folder, view, hist, norm):
    y_abs = view["y_abs"]
    extent = [0.0, SPEED_MAX_MS, -y_abs, y_abs]
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    masked = np.ma.masked_where(hist <= 0, hist)
    im = ax.imshow(masked.T, origin="lower", extent=extent, aspect="auto",
                   cmap=CMAP, norm=norm)
    ax.axhline(0.0, color=ZERO_LINE_COLOR, linewidth=ZERO_LINE_LW, alpha=ZERO_LINE_ALPHA)
    cen = CENTROIDS.get(view["tag"])
    if cen is not None:
        ax.scatter(cen[:, 0], cen[:, 1], c=CENTROID_COLOR, s=CENTROID_SIZE,
                   marker=CENTROID_MARKER, linewidths=CENTROID_LW)
    ax.set_xlabel("speed [m/s]")
    ax.set_ylabel(view["y_label"])
    ax.set_title(f"{DISPLAY_NAMES.get(folder, folder)} — {view['title']}")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("fraction of frames (log)")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"phase_portrait_{folder}_{view['tag']}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    # Pass 1 — load every dataset (threaded reads + per-dataset progress bar)
    hists = {}
    for folder in DATASET_ORDER:
        d = load_dataset(folder)
        if d is not None:
            hists[folder] = d
        else:
            print(f"[A10] {folder}: no clips found, skipped")
    if not hists:
        print(f"[A10] no data found under {INERTIAL_DATA_DIR}")
        return

    # Shared cross-dataset log color scale per view
    shared_norm = {}
    for view in VIEWS:
        vmax = max((hists[f][view["tag"]].max() for f in hists
                    if hists[f][view["tag"]].max() > 0), default=1.0)
        shared_norm[view["tag"]] = LogNorm(vmin=vmax / VMIN_RATIO, vmax=vmax)

    # Pass 2 — render the 26 figures (single-threaded)
    count = 0
    for folder in DATASET_ORDER:
        if folder not in hists:
            continue
        for view in VIEWS:
            H = hists[folder][view["tag"]]
            if SHARED_COLOR_SCALE:
                norm = shared_norm[view["tag"]]
            else:
                vmax = H.max() if H.max() > 0 else 1.0
                norm = LogNorm(vmin=vmax / VMIN_RATIO, vmax=vmax)
            render(folder, view, H, norm)
            count += 1
    print(f"[A10] saved {count} figures to {OUT_DIR}")


if __name__ == "__main__":
    main()