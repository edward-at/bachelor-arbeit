"""
compute_std.py  (v3 — bounds-masked stds on the 10 Hz common grid)
────────────────────────────────────────────────────────────────────────────
Reads inertial_data/<dataset>/clip_XXXX.parquet (10 Hz uniform grid, v3
standardization) and computes per-window standard deviations.

Output schema (unchanged from v1 — downstream scripts keep working):
  clip_id, window_idx, window_start_s, window_end_s, n_frames,
  std_speed_kmh, std_a_long, std_a_lat, std_jerk_long, std_jerk_lat,
  std_yaw_rate

─── CHANGES vs v1 ──────────────────────────────────────────────────────────
  1. MIN_VALID_FRAC rule: a signal's std is NaN unless at least
     MIN_VALID_FRAC of the window's expected frames are non-NaN for that
     signal. Needed because the v3 standardization intentionally injects
     NaN (gap masking, edge masking) — a "std" computed from 3 surviving
     frames of a 200-frame window is noise, not a measurement.
  2. Dead batch-mode code removed (it referenced an undefined BATCH_SIZE —
     a latent NameError if LARGE_DATASET_THRESHOLD was ever lowered).
     At 10 Hz all datasets fit normal mode comfortably.
  3. GRID_HZ parameter documents the expected input rate.

─── CHANGES vs v2 ──────────────────────────────────────────────────────────
  4. APPLY_BOUNDS: physical-validity masking (same bounds as the
     visualization layer) applied per signal before windowing. Residual
     impossible values (e.g. one 5,000 km/h frame) no longer inflate window
     stds; the MIN_VALID rule then governs windows that lose too many frames.
────────────────────────────────────────────────────────────────────────────
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS  (edit here only)
# ─────────────────────────────────────────────────────────────────────────────

WINDOW_S       = 20.0   # window size in seconds
STEP_S         = 10.0   # step size in seconds (50% overlap)
GRID_HZ        = 10     # sampling rate of v3 standardized clips (uniform)
MIN_VALID_FRAC = 0.5    # per signal: min fraction of expected frames that
                        # must be non-NaN, else std = NaN for that signal
MIN_VALID_ABS  = 2      # absolute floor (std needs >= 2 points regardless)
APPLY_BOUNDS   = True   # mask physically impossible values before std

# Physical-validity bounds (identical to the visualization layer; yaw_rate in
# rad/s as stored). Applied as NaN masking at READ time, never modifying the
# source clips. Rationale: a single residual 5,000 km/h frame in a 200-frame
# window inflates that window's std by orders of magnitude and poisons every
# downstream figure; out-of-bounds fractions remain quantified by the
# boundary-coverage audit, so nothing is hidden by this masking.
BOUNDS = {
    "speed_kmh": (0.0,   200.0),
    "a_long":    (-15.0,  15.0),
    "a_lat":     (-15.0,  15.0),
    "jerk_long": (-30.0,  30.0),
    "jerk_lat":  (-30.0,  30.0),
    "yaw_rate":  (-1.57,   1.57),
}

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

INERTIAL_ROOT = Path("/Users/edward/Downloads/dataset/inertial_data")
STD_ROOT      = Path("/Users/edward/Downloads/dataset/std_data")

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL COLUMNS
# ─────────────────────────────────────────────────────────────────────────────

SIGNAL_COLS = ["speed_kmh", "a_long", "a_lat", "jerk_long", "jerk_lat", "yaw_rate"]
STD_COLS    = ["std_speed_kmh", "std_a_long", "std_a_lat",
               "std_jerk_long", "std_jerk_lat", "std_yaw_rate"]
NEEDED_COLS = ["t_rel_s"] + SIGNAL_COLS

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _clip_id_from_path(fp: Path) -> int:
    """Extract integer ID from clip_XXXX.parquet filename."""
    try:
        return int(fp.stem.split("_")[1])
    except (IndexError, ValueError):
        return -1


def _read_clip(fp: Path) -> pd.DataFrame | None:
    """Read only needed columns from a clip parquet file."""
    try:
        import pyarrow.parquet as pq
        schema_cols = set(pq.read_schema(fp).names)
        cols = [c for c in NEEDED_COLS if c in schema_cols]
        df = pd.read_parquet(fp, columns=cols)
        for c in NEEDED_COLS:
            if c not in df.columns:
                df[c] = np.nan
        if APPLY_BOUNDS:
            for c, (lo, hi) in BOUNDS.items():
                if c in df.columns:
                    a = df[c].values
                    df[c] = np.where((a >= lo) & (a <= hi), a, np.nan)
        return df
    except Exception as e:
        print(f"  [WARN] Could not read {fp.name}: {e}")
        return None


def _window_stds(w_df: pd.DataFrame, duration_s: float) -> dict:
    """
    Per-signal std with the MIN_VALID_FRAC rule.
    Expected frames for this window = duration_s × GRID_HZ (uniform v3 grid).
    A signal whose non-NaN count falls below max(MIN_VALID_ABS,
    MIN_VALID_FRAC × expected) gets NaN — gap/edge-masked windows must not
    produce stds from a handful of surviving frames.
    """
    n_expected = max(1.0, duration_s * GRID_HZ)
    min_valid  = max(MIN_VALID_ABS, int(np.ceil(MIN_VALID_FRAC * n_expected)))

    stds = {}
    for sc, rc in zip(STD_COLS, SIGNAL_COLS):
        vals = w_df[rc]
        n_valid = int(vals.notna().sum())
        stds[sc] = float(vals.std(ddof=1)) if n_valid >= min_valid else float("nan")
    return stds


# ─────────────────────────────────────────────────────────────────────────────
# WINDOW COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def _compute_windows_for_clip(fp: Path) -> list[dict]:
    """
    Extract all windows from a single clip and compute std per signal.

    Windowing rules (unchanged):
    - t_max >= WINDOW_S : sliding window with STEP_S step
    - t_max <  WINDOW_S : entire clip = single window (window_idx=0)
    """
    df = _read_clip(fp)
    if df is None or len(df) < 2:
        return []

    clip_id = _clip_id_from_path(fp)
    t       = df["t_rel_s"].values.astype(float)
    t_min   = float(t.min())
    t_max   = float(t.max())

    rows = []

    # ── short clip: 1 clip = 1 window ────────────────────────────────────────
    if t_max - t_min < WINDOW_S:
        stds = _window_stds(df, duration_s=t_max - t_min)
        rows.append({
            "clip_id":        clip_id,
            "window_idx":     0,
            "window_start_s": t_min,
            "window_end_s":   t_max,
            "n_frames":       len(df),
            **stds,
        })
        return rows

    # ── normal sliding window ─────────────────────────────────────────────────
    window_idx = 0
    t_start    = t_min
    while t_start + WINDOW_S <= t_max + 1e-6:
        t_end = t_start + WINDOW_S
        mask  = (t >= t_start) & (t < t_end)
        w_df  = df[mask]
        if len(w_df) >= 2:
            stds = _window_stds(w_df, duration_s=WINDOW_S)
            rows.append({
                "clip_id":        clip_id,
                "window_idx":     window_idx,
                "window_start_s": float(t_start),
                "window_end_s":   float(t_end),
                "n_frames":       int(mask.sum()),
                **stds,
            })
            window_idx += 1
        t_start += STEP_S

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# DATASET WORKER
# ─────────────────────────────────────────────────────────────────────────────

def _process_dataset(args: tuple) -> dict | None:
    ds_dir, ds_name, out_path = args

    clip_files = sorted(ds_dir.glob("clip_*.parquet"))
    if not clip_files:
        print(f"  [WARN] {ds_name}: no clip_*.parquet files found")
        return None

    n_clips  = len(clip_files)
    all_rows = []

    for fp in tqdm(clip_files, desc=f"  {ds_name:<12}", unit="clip", leave=True):
        all_rows.extend(_compute_windows_for_clip(fp))

    if not all_rows:
        print(f"  [WARN] {ds_name}: no valid windows found")
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(all_rows)
    result_df.to_parquet(out_path, index=False)
    n_windows = len(result_df)

    print(
        f"── {ds_name.upper():<12} [{_now()}]  "
        f"clips: {n_clips:,}  →  windows: {n_windows:,}  "
        f"→  {out_path.name}"
    )
    return {"dataset": ds_name, "n_clips": n_clips, "n_windows": n_windows}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import os

    STD_ROOT.mkdir(parents=True, exist_ok=True)

    datasets = sorted(
        d for d in INERTIAL_ROOT.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    if not datasets:
        print(f"[ERROR] No dataset folders found in {INERTIAL_ROOT}")
        sys.exit(1)

    n_workers = min(len(datasets), os.cpu_count() or 4)
    t_start   = datetime.now()

    print(f"\n{'='*65}")
    print(f"  compute_std.py  (v2 — 10 Hz common grid)")
    print(f"  Window: {WINDOW_S}s   Step: {STEP_S}s   Grid: {GRID_HZ} Hz")
    print(f"  Min valid: {MIN_VALID_FRAC:.0%} of expected frames per signal")
    print(f"  Bounds masking: {'ON' if APPLY_BOUNDS else 'OFF'}")
    print(f"  Signals: {SIGNAL_COLS}")
    print(f"  Datasets found: {len(datasets)}")
    print(f"  Workers: {n_workers}  (CPU cores: {os.cpu_count()})")
    print(f"  Input  → {INERTIAL_ROOT}")
    print(f"  Output → {STD_ROOT}")
    print(f"  Started → {_now()}")
    print(f"{'='*65}\n")

    args = [
        (ds_dir, ds_dir.name, STD_ROOT / f"{ds_dir.name}.parquet")
        for ds_dir in datasets
    ]

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_dataset, a): a[1] for a in args}
        for fut in as_completed(futures):
            ds_name = futures[fut]
            try:
                result = fut.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"  [ERROR] {ds_name}: {e}")

    elapsed = (datetime.now() - t_start).total_seconds()

    # ── summary ───────────────────────────────────────────────────────────────
    results.sort(key=lambda r: r["dataset"])
    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Dataset':<15} {'Clips':>10} {'Windows':>12}")
    print(f"  {'-'*40}")
    total_clips   = 0
    total_windows = 0
    for r in results:
        print(f"  {r['dataset']:<15} {r['n_clips']:>10,} {r['n_windows']:>12,}")
        total_clips   += r["n_clips"]
        total_windows += r["n_windows"]
    print(f"  {'-'*40}")
    print(f"  {'TOTAL':<15} {total_clips:>10,} {total_windows:>12,}")
    print(f"\n{'='*65}")
    print(f"  ✓ DONE  —  {len(results)}/{len(datasets)} datasets  "
          f"·  {total_windows:,} windows  "
          f"·  {elapsed:.1f}s  ·  {_now()}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()