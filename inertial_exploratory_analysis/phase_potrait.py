"""
phase_potrait.py — v2
────────────────────────────────────────────────────────────────────────────
A10 — Per-dataset kinematic state-space occupancy. For each dataset, TWO
figures: speed × longitudinal acceleration, and speed × yaw rate
(14 datasets × 2 = 28 figures). Supports RQ1 (calm-biased datasets concentrate
near low speed / near-zero accel|yaw) and RQ2 (dense regions ~ dynamic regimes).

v2 changes (each fixes a diagnosed defect of v1):
  1  OCCUPANCY FLOOR: bins holding fewer than MIN_FRAMES_PER_BIN frames are
     masked. Under LogNorm's clipped vmin, a 1-frame bin (0.1 s of driving)
     rendered with the same saturated colour as a 30-frame bin, so ~half the
     drawn bins carried ~1% of the data ("confetti"). The floor is a physical
     evidence threshold — "at least 0.5 s of cumulative driving" — disclosed
     per figure in the title and printed for the caption.
     ORDER OF OPERATIONS MATTERS: this floor is cosmetic triage and was only
     applied AFTER the frame data itself was proven clean (pipeline v4.3
     validity gates); a floor applied earlier would have buried six real
     defects. Never reverse that order.
  2  HONEST DENOMINATOR: v1 normalised by in-panel frames only, silently
     inflating every colour for datasets with mass outside the axes. v2
     normalises by ALL valid frames and discloses the outside-panel share in
     the title.
  3  The white zero-line is gone (it sliced through the densest bin and read
     as a scratch; the y-tick at 0 already marks the axis).
  4  tqdm degrades to a no-op if missing.

Unchanged: shared bins, shared cross-dataset LogNorm colour scale (the 28
figures stay directly comparable), viridis (documented in §8.5 for occupancy),
threaded batched reads, single-threaded rendering, pure default style, dpi=300,
RQ2 centroid overlay wired but disabled.

Frame data is v4.3-gated upstream, so physical-validity bounds are already
enforced; the fixed bin ranges act as the display window only.
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # tqdm optional — degrade to a no-op
    def tqdm(x=None, **kwargs):
        class _Bar:
            def update(self, *a): pass
            def write(self, *a): print(*a)
            def close(self): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
        return _Bar() if x is None else x

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

# v2 — occupancy floor: a bin must hold at least this many frames to be drawn.
# At the 10 Hz grid, 5 frames = 0.5 s of cumulative driving. Masked bins and
# their frame share are disclosed in each figure title and printed per dataset.
MIN_FRAMES_PER_BIN = 5

CMAP            = "viridis"
CENTROID_COLOR  = "red"
CENTROID_SIZE   = 30
CENTROID_MARKER = "x"
CENTROID_LW     = 1.5
FIG_W = 10
FIG_H = 6
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
    "4seasons": "4Seasons", "hdd": "HDD", "a2d2": "A2D2", "kitscenes": "KITScenes"
}
DATASET_ORDER = list(DISPLAY_NAMES.keys())   # 14 datasets

DATASET_COLORS = {
    "4seasons":  TUM_SECONDARY['blue2'],
    "a2d2":      TUM_SECONDARY['dark_gray'],
    "hdd":       TUM_SECONDARY['gray'],
    "kitscenes": TUM_PRIMARY['black'],
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

# ─────────────────────────────────────────────────────────────────────────────
# DATA  (threaded, batched parquet reads + tqdm; both views in one pass)
# ─────────────────────────────────────────────────────────────────────────────

def _clip_hists(fp):
    """Per-clip partial counts. Returns (H_along, H_yaw, n_valid_along, n_valid_yaw)
    where n_valid_* counts ALL finite frame pairs, in-panel or not — the honest
    denominator."""
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
    return Ha, Hy, int(ma.sum()), int(my.sum())


def load_dataset(folder):
    """Accumulate raw COUNTS per view plus the total-valid-frame denominators."""
    clips = sorted((INERTIAL_DATA_DIR / folder).glob("clip_*.parquet"))
    if not clips:
        return None
    Ha = np.zeros((N_BINS_SPEED, N_BINS_Y))
    Hy = np.zeros((N_BINS_SPEED, N_BINS_Y))
    n_along = n_yaw = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        with tqdm(total=len(clips), desc=DISPLAY_NAMES.get(folder, folder), unit="clip") as pbar:
            for i in range(0, len(clips), CHUNK):
                for r in ex.map(_clip_hists, clips[i:i + CHUNK]):
                    if r is not None:
                        Ha += r[0]
                        Hy += r[1]
                        n_along += r[2]
                        n_yaw   += r[3]
                    pbar.update(1)
    return {"speed_along": {"counts": Ha, "n_valid": n_along},
            "speed_yaw":   {"counts": Hy, "n_valid": n_yaw}}


# ─────────────────────────────────────────────────────────────────────────────
# PLOT (one figure per dataset per view)
# ─────────────────────────────────────────────────────────────────────────────

def render(folder, view, counts, n_valid, norm, log):
    """counts = raw per-bin frame counts; n_valid = the honest denominator."""
    # Occupancy floor: mask empty AND sub-floor bins; account for the disclosure.
    keep = counts >= MIN_FRAMES_PER_BIN
    subfloor = (counts > 0) & ~keep
    n_subfloor_bins   = int(subfloor.sum())
    n_subfloor_frames = float(counts[subfloor].sum())
    in_panel = float(counts.sum())
    pct_floor   = 100.0 * n_subfloor_frames / n_valid if n_valid else 0.0
    pct_outside = 100.0 * (n_valid - in_panel) / n_valid if n_valid else 0.0

    frac = np.zeros_like(counts)
    if n_valid > 0:
        frac = counts / n_valid            # honest denominator: ALL valid frames
    masked = np.ma.masked_where(~keep, frac)

    y_abs = view["y_abs"]
    extent = [0.0, SPEED_MAX_MS, -y_abs, y_abs]
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    im = ax.imshow(masked.T, origin="lower", extent=extent, aspect="auto",
                   cmap=CMAP, norm=norm)
    cen = CENTROIDS.get(view["tag"])
    if cen is not None:
        ax.scatter(cen[:, 0], cen[:, 1], c=CENTROID_COLOR, s=CENTROID_SIZE,
                   marker=CENTROID_MARKER, linewidths=CENTROID_LW)
    ax.set_xlabel("speed [m/s]")
    ax.set_ylabel(view["y_label"])
    # Self-disclosing title: the floor and the outside-panel share travel with
    # the figure, so no reader can mistake masking for absence of data.
    ax.set_title(
        f"{view['title']} ({DISPLAY_NAMES.get(folder, folder)})\n"
        f"bins < {MIN_FRAMES_PER_BIN} frames masked "
        f"({n_subfloor_bins} bins, {pct_floor:.2f}% of frames); "
        f"{pct_outside:.2f}% of frames outside panel"
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("fraction of valid frames (log)")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"phase_portrait_{folder}_{view['tag']}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.append(f"  {DISPLAY_NAMES.get(folder, folder):<10} {view['tag']:<12} "
               f"floor-masked {n_subfloor_bins:>5} bins = {pct_floor:>6.3f}% of frames"
               f" | outside panel {pct_outside:>6.3f}%")
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

    # Shared cross-dataset log colour scale per view, computed on the FLOORED,
    # honestly-normalised fractions so the scale describes what is drawn.
    shared_norm = {}
    for view in VIEWS:
        vmax = 0.0
        for f in hists:
            h = hists[f][view["tag"]]
            if h["n_valid"] > 0:
                kept = h["counts"][h["counts"] >= MIN_FRAMES_PER_BIN]
                if kept.size:
                    vmax = max(vmax, float(kept.max()) / h["n_valid"])
        if vmax <= 0.0:
            vmax = 1.0
        shared_norm[view["tag"]] = LogNorm(vmin=vmax / VMIN_RATIO, vmax=vmax)

    # Pass 2 — render the figures (single-threaded; matplotlib is not thread-safe)
    # Purge this script's previous outputs first: a dataset skipped in THIS run
    # must not survive as a stale figure from a previous one.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stale = sorted(OUT_DIR.glob("phase_portrait_*.png"))
    for f in stale:
        f.unlink()
    if stale:
        print(f"[A10] purged {len(stale)} existing figures from {OUT_DIR}")

    count, log = 0, []
    for folder in DATASET_ORDER:
        if folder not in hists:
            continue
        for view in VIEWS:
            h = hists[folder][view["tag"]]
            # v2.1 — a measurement-limited dataset (4Seasons: speed excluded)
            # has no valid frames for a speed-based view. A fully blank figure
            # discloses nothing a prose sentence cannot; skip and log instead.
            if h["n_valid"] == 0:
                log.append(f"  {DISPLAY_NAMES.get(folder, folder):<10} "
                           f"{view['tag']:<12} SKIPPED — no valid frames "
                           f"(measurement-limited channels, Section 7.3)")
                continue
            if SHARED_COLOR_SCALE:
                norm = shared_norm[view["tag"]]
            else:
                kept = h["counts"][h["counts"] >= MIN_FRAMES_PER_BIN]
                vmax = (float(kept.max()) / h["n_valid"]) if (kept.size and h["n_valid"]) else 1.0
                norm = LogNorm(vmin=vmax / VMIN_RATIO, vmax=vmax)
            render(folder, view, h["counts"], h["n_valid"], norm, log)
            count += 1

    print("\n[A10] disclosure summary (paste-ready for captions):")
    for line in log:
        print(line)
    print(f"\n[A10] saved {count} figures to {OUT_DIR}")


if __name__ == "__main__":
    main()