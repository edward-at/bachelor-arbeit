import sys
import warnings
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from thesis_paths import DATA_ROOT, REPO_ROOT  # noqa: E402

warnings.filterwarnings("ignore")

import os as _os
_os.environ.setdefault("PYTHONWARNINGS", "ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS  (edit here only)
# ─────────────────────────────────────────────────────────────────────────────

TIER_PERCENTILES = [0.25, 0.50, 0.75]     # 3 representative lines per regime band
N_REPS           = len(TIER_PERCENTILES)

# Continuity requirement: every representative window must contain at least this
# many SECONDS of gap-free data for the plotted signal — i.e. a run of consecutive
# non-NaN samples on the uniform 10 Hz grid. 190 consecutive frames = 19 s.
# (A window can have 195 valid frames but a gap in the middle; that fails.)
MIN_WINDOW_SECONDS = 19.0
GRID_HZ            = 10.0
MIN_WINDOW_FRAMES  = int(round(MIN_WINDOW_SECONDS * GRID_HZ))   # 190
MAX_WALK           = 60                    # max σ-rank steps when searching for a continuous window

DPI   = 300
FIG_W = 20.0                               # wide: 4 columns
FIG_H = 4.5                                # short: 1 row

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

INERTIAL_ROOT = DATA_ROOT / "inertial_data"
STD_ROOT      = DATA_ROOT / "std_data"
FIG_ROOT      = REPO_ROOT / "04_exploratory_analysis" / "figures" / "time_series"

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

SIGNALS = [
    {"col": "speed_kmh", "std_col": "std_speed_kmh", "label": "Speed", "unit": "km/h", "clip_lo": 0.0, "clip_hi": 200.0},
    # Uncomment to also emit acc_mag and yaw-rate trace figures (one figure per signal).
    # yaw bounds are in °/s because yaw_rate is converted to °/s before masking;
    # acc_mag upper bound 21.3 matches compute_std and the rest of the thesis.
    # {"col": "acc_mag",  "std_col": "std_acc_mag",  "label": "Accel. magnitude", "unit": "m/s²", "clip_lo":  0.0, "clip_hi": 21.3},
    # {"col": "yaw_rate", "std_col": "std_yaw_rate", "label": "Yaw rate",         "unit": "°/s",  "clip_lo": -90.0, "clip_hi": 90.0},
]

# ─────────────────────────────────────────────────────────────────────────────
# REGIME BANDS — 4 categories on ABSOLUTE σ thresholds, the same values as in
# regime_classification.py and ecdf_analysis.py.
#   Near-Stationary : σ < t1
#   Low-Dynamic     : t1 ≤ σ < t2
#   Medium-Dynamic  : t2 ≤ σ < t3
#   High-Dynamic    : σ ≥ t3
# NOTE: each of the THREE files (here, regime_classification.py,
#       ecdf_analysis.py) keeps its own copy (not imported); keep them in sync.
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLDS = {
    "std_speed_kmh": (1.0,  5.0,  15.0),
    "std_acc_mag":   (0.2,  0.5,   1.5),   # frozen round convention, see regime_classification.py
    "std_yaw_rate":  (1.0,  5.0,  15.0),
}
REGIME_LABELS = ["Near-Stationary", "Low-Dynamic", "Medium-Dynamic", "High-Dynamic"]

# Folder name -> thesis display name (all 14 datasets), shared with every other figure.
DISPLAY_NAMES = {
    "boreas_rt": "Boreas", "durlar": "DurLAR", "ford": "Ford", "kitti": "KITTI",
    "kitti360": "KITTI-360", "nuscene": "nuScenes", "nvidia": "NVIDIA",
    "oxford": "Oxford", "waymo": "Waymo", "zod": "ZOD",
    "4seasons": "4Seasons", "hdd": "HDD", "a2d2": "A2D2", "kitscenes": "KITScenes",
}

# ─────────────────────────────────────────────────────────────────────────────
# COLORS — 25% blue, 50% orange, 75% green  (same three tones, reordered)
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from tum_colors import TUM_ACCENT

REP_COLORS = [
    TUM_ACCENT["blue"],     # 25%
    TUM_ACCENT["orange"],   # 50%
    TUM_ACCENT["green"],    # 75%
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _clip_path(ds_name: str, clip_id: int) -> Path:
    return INERTIAL_ROOT / ds_name / f"clip_{clip_id:04d}.parquet"


def _read_window_data(ds_name, clip_id, window_start_s, window_end_s):
    """Recover raw frames for a window via the back-reference contract.
    Returns a df with an added t_win_s column (time relative to window start)."""
    fp = _clip_path(ds_name, clip_id)
    if not fp.exists():
        return None
    try:
        import pyarrow.parquet as pq
        needed = ["t_rel_s"] + [s["col"] for s in SIGNALS]
        schema_cols = set(pq.read_schema(fp).names)
        cols = [c for c in needed if c in schema_cols]
        df = pd.read_parquet(fp, columns=cols)
        for c in needed:
            if c not in df.columns:
                df[c] = np.nan
    except Exception:
        return None

    # short clip window: use full clip
    if window_start_s == df["t_rel_s"].min() and window_end_s == df["t_rel_s"].max():
        w_df = df.copy()
    else:
        mask = (df["t_rel_s"] >= window_start_s) & (df["t_rel_s"] < window_end_s)
        w_df = df[mask].copy()

    if len(w_df) < 2:
        return None

    w_df["t_win_s"] = w_df["t_rel_s"] - window_start_s
    return w_df


def _longest_run_frames(valid: np.ndarray) -> int:
    """Longest run of consecutive True values in a boolean mask."""
    if valid.size == 0 or not valid.any():
        return 0
    invalid = np.flatnonzero(~valid)
    if invalid.size == 0:
        return int(valid.size)
    bounds = np.concatenate(([-1], invalid, [valid.size]))
    return int(np.max(np.diff(bounds) - 1))


# ─────────────────────────────────────────────────────────────────────────────
# CONTINUITY GATE + REGIME SELECTION  (reads std_data, confirms on raw frames)
# ─────────────────────────────────────────────────────────────────────────────

def _passes_continuity(ds_name, sig_col, clip_id, win_start, win_end, cache):
    """(ok, w_df). A window passes iff the plotted signal has a gap-free run of
    >= MIN_WINDOW_FRAMES consecutive non-NaN samples. Frames cached per window."""
    key = (clip_id, round(win_start, 3), round(win_end, 3))
    if key in cache:
        return cache[key]
    w_df = _read_window_data(ds_name, clip_id, win_start, win_end)
    if w_df is None or sig_col not in w_df.columns:
        cache[key] = (False, None)
        return cache[key]
    valid = ~np.isnan(w_df[sig_col].values.astype(float))
    ok = _longest_run_frames(valid) >= MIN_WINDOW_FRAMES
    cache[key] = (ok, w_df if ok else None)
    return cache[key]


def _pick_rep(band, target_idx, std_col, sig_col, ds_name, cache):
    """Walk outward in σ-rank from target_idx until a window passes the continuity
    gate. Returns a rep dict (with cached frames) or None. Duplicates across
    percentiles are allowed when a band is tiny."""
    n = len(band)
    if n == 0:
        return None
    order = [target_idx]
    for d in range(1, min(MAX_WALK, n) + 1):
        if target_idx - d >= 0:
            order.append(target_idx - d)
        if target_idx + d < n:
            order.append(target_idx + d)

    seen = set()
    for idx in order:
        if idx in seen:
            continue
        seen.add(idx)
        row = band.iloc[idx]
        ok, w_df = _passes_continuity(ds_name, sig_col,
                                      int(row["clip_id"]),
                                      float(row["window_start_s"]),
                                      float(row["window_end_s"]), cache)
        if ok:
            return {
                "std_val":   float(row[std_col]),
                "clip_id":   int(row["clip_id"]),
                "win_start": float(row["window_start_s"]),
                "win_end":   float(row["window_end_s"]),
                "w_df":      w_df,
            }
    return None


def _select_representatives(std_df, std_col, sig_col, ds_name):
    """Assign eligible windows to the 4 absolute regime bands, then pick 3
    representatives (25/50/75 percentile of σ within the band) per band, each
    satisfying the >= 19 s continuity gate.
    Returns {regime_label: [rep|None, rep|None, rep|None]} or None."""
    if std_col not in std_df.columns:
        return None

    valid = std_df[std_df[std_col].notna()].copy()
    if valid.empty:
        return None

    # Cheap pre-filter: a window needs >= MIN_WINDOW_FRAMES valid frames before it
    # could possibly hold that many CONSECUTIVE ones (the gate confirms consecutive).
    if "n_frames" in valid.columns:
        valid = valid[valid["n_frames"] >= MIN_WINDOW_FRAMES]
    if valid.empty:
        return None

    t1, t2, t3 = THRESHOLDS[std_col]
    band_edges = [(-np.inf, t1), (t1, t2), (t2, t3), (t3, np.inf)]

    cache = {}
    result = {}
    for label, (lo, hi) in zip(REGIME_LABELS, band_edges):
        band = (valid[(valid[std_col] >= lo) & (valid[std_col] < hi)]
                .sort_values(std_col).reset_index(drop=True))
        n = len(band)
        reps = []
        for p in TIER_PERCENTILES:
            if n == 0:
                reps.append(None)
                continue
            target = int(round(p * (n - 1)))
            reps.append(_pick_rep(band, target, std_col, sig_col, ds_name, cache))
        result[label] = reps
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def _band_range_str(std_col, band_idx, unit) -> str:
    t1, t2, t3 = THRESHOLDS[std_col]
    if band_idx == 0:
        return f"σ < {t1:g} {unit}"
    if band_idx == 1:
        return f"{t1:g} ≤ σ < {t2:g} {unit}"
    if band_idx == 2:
        return f"{t2:g} ≤ σ < {t3:g} {unit}"
    return f"σ ≥ {t3:g} {unit}"


def _plot_signal(dataset_name, sig, selection, out_path):
    """1 row × 4 columns, one column per regime band. Each subplot draws up to
    N_REPS representative windows. Title shows the fixed regime σ-range.
    Signal values outside clip_lo/clip_hi are masked as NaN (visualization only)."""
    col   = sig["col"]
    lo    = sig["clip_lo"]
    hi    = sig["clip_hi"]
    unit  = sig["unit"]
    s_col = sig["std_col"]

    fig, axes = plt.subplots(1, 4, figsize=(FIG_W, FIG_H), sharey=False)

    for band_idx, (ax, label) in enumerate(zip(axes, REGIME_LABELS)):
        reps = selection.get(label, [None] * N_REPS)
        plotted = False
        for i, rep in enumerate(reps):
            if rep is None:
                continue
            w_df = rep["w_df"]
            t = w_df["t_win_s"].values
            y = w_df[col].values.astype(float)

            # Convert yaw_rate from rad/s to °/s
            if col == "yaw_rate":
                y = y * (180.0 / np.pi)

            # Mask physically implausible values (visualization only)
            y = np.where((y >= lo) & (y <= hi), y, np.nan)

            pct = f"{int(round(TIER_PERCENTILES[i] * 100))}%"
            ax.plot(t, y, color=REP_COLORS[i], label=f"{pct} (σ={rep['std_val']:.3f})")
            plotted = True

        ax.set_title(f"{label}\n{_band_range_str(s_col, band_idx, unit)}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"{sig['label']} ({unit})")
        if plotted:
            ax.legend()
        ax.grid(True)
        ax.set_axisbelow(True)

    fig.suptitle(f"{DISPLAY_NAMES.get(dataset_name, dataset_name)} {sig['label']} Representation")
    fig.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# WORKER  (one per dataset)
# ─────────────────────────────────────────────────────────────────────────────

def _process_dataset(args):
    ds_name, fig_root = args

    std_path = STD_ROOT / f"{ds_name}.parquet"
    if not std_path.exists():
        print(f"  [WARN] {ds_name}: std_data file not found at {std_path}")
        return None

    std_df = pd.read_parquet(std_path)
    # Match regime_classification.py / ecdf_analysis.py: yaw std stored rad/s -> °/s
    if "std_yaw_rate" in std_df.columns:
        std_df["std_yaw_rate"] = std_df["std_yaw_rate"] * (180.0 / np.pi)

    for sig in SIGNALS:
        selection = _select_representatives(std_df, sig["std_col"], sig["col"], ds_name)
        if selection is None:
            print(f"  [WARN] {ds_name} / {sig['col']}: no windows with ≥ {MIN_WINDOW_SECONDS:.0f}s continuous data")
            continue

        counts = {lab: sum(r is not None for r in reps) for lab, reps in selection.items()}
        print(f"── {DISPLAY_NAMES.get(ds_name, ds_name):<12} {sig['col']:<12} [{_now()}]  "
              f"reps/band  "
              f"NS={counts['Near-Stationary']}  "
              f"L={counts['Low-Dynamic']}  "
              f"M={counts['Medium-Dynamic']}  "
              f"H={counts['High-Dynamic']}")

        out_path = fig_root / f"{ds_name}_{sig['col']}.png"
        _plot_signal(ds_name, sig, selection, out_path)

    return {"dataset": ds_name}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import os

    FIG_ROOT.mkdir(parents=True, exist_ok=True)

    std_files = sorted(STD_ROOT.glob("*.parquet"))
    if not std_files:
        print(f"[ERROR] No parquet files found in {STD_ROOT}")
        print(f"        Run compute_std.py first.")
        sys.exit(1)

    ds_names   = [f.stem for f in std_files]
    n_workers  = min(len(ds_names), os.cpu_count() or 4)
    total_pngs = len(ds_names) * len(SIGNALS)
    t_start    = datetime.now()

    print(f"\n{'='*70}")
    print(f"  time_series_analysis.py  —  regime-banded representative traces")
    print(f"  Bands:    {', '.join(REGIME_LABELS)}  (absolute σ thresholds)")
    print(f"  Reps:     {N_REPS} per band at {[int(p*100) for p in TIER_PERCENTILES]}%  (25% blue, 50% orange, 75% green)")
    print(f"  Gate:     ≥ {MIN_WINDOW_SECONDS:.0f}s continuous ({MIN_WINDOW_FRAMES} consecutive frames @ {GRID_HZ:.0f} Hz)")
    print(f"  Datasets: {len(ds_names)}   Signals: {len(SIGNALS)}   PNGs: {total_pngs}")
    print(f"  Workers:  {n_workers}  (CPU cores: {os.cpu_count()})")
    print(f"  std_data → {STD_ROOT}")
    print(f"  Output   → {FIG_ROOT}")
    print(f"  Started  → {_now()}")
    print(f"{'='*70}\n")

    args = [(ds_name, FIG_ROOT) for ds_name in ds_names]

    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_dataset, a): a[0] for a in args}
        for fut in as_completed(futures):
            ds_name = futures[fut]
            try:
                if fut.result() is not None:
                    done += 1
            except Exception as e:
                print(f"  [ERROR] {ds_name}: {e}")

    elapsed = (datetime.now() - t_start).total_seconds()
    print(f"\n{'='*70}")
    print(f"  ✓ DONE  —  {done}/{len(ds_names)} datasets  "
          f"·  {done * len(SIGNALS)} PNGs  ·  {elapsed:.1f}s  ·  {_now()}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()