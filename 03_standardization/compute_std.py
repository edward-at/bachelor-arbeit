"""
compute_std.py  (per-window variability stds and feature means on the 10 Hz grid)
────────────────────────────────────────────────────────────────────────────
Reads inertial_data/<dataset>/clip_XXXX.parquet (10 Hz uniform grid from the
standardization stage) and computes, in a SINGLE per-window pass, BOTH:
  • within-window standard deviations: the variability signature (Section 4.6)
  • within-window feature means: the regime feature vector (Section 4.7)

Output schema:
  clip_id, window_idx, window_start_s, window_end_s, n_frames,
  # variability channels (Section 4.6):
  std_speed_kmh, std_acc_mag, std_yaw_rate,
  # feature-mean channels (Section 4.7):
  mean_speed_kmh, mean_acc_pos, mean_acc_neg,
  mean_abs_a_lat, mean_abs_yaw_rate, mean_abs_jerk

  acc_mag and jerk_mag are read DIRECTLY from the standardized clips (computed
  upstream as sqrt(a_long^2 + a_lat^2) and sqrt(jerk_long^2 + jerk_lat^2)) and
  are never re-derived here. The longitudinal accel/decel split is taken as the
  positive and negative parts of a_long.

─── DESIGN NOTES ───────────────────────────────────────────────────────────
  1. MIN_VALID_FRAC rule: a signal's statistic is NaN unless at least
     MIN_VALID_FRAC of the window's expected frames are non-NaN for that
     signal. Needed because the standardization intentionally injects
     NaN (gap masking, edge masking). A "std" computed from 3 surviving
     frames of a 200-frame window is noise, not a measurement. The rule
     governs EVERY per-window statistic, std OR mean.
  2. GRID_HZ documents the expected input rate. At 10 Hz all datasets are
     processed in a single in-memory pass per dataset.
  3. APPLY_BOUNDS: physical-validity masking (same bounds as the
     visualization layer) applied per signal before windowing. Residual
     impossible values (e.g. one 5,000 km/h frame) do not inflate window
     stds. The MIN_VALID rule then governs windows that lose too many frames.
     The component bounds a_long, a_lat in [-15, 15] and the implied magnitude
     bound jerk_mag in [0, 42.43] (= sqrt(30^2 + 30^2)) are also masked
     before any statistic is taken.
  4. FEATURE MEANS: the same per-window pass emits six mean_* channels that
     form the per-window feature vector consumed by the regime clustering of
     Section 4.7. Windowing: 20 s windows advanced 20 s (non-overlapping).
  5. The variability channels collapse acceleration to a single magnitude
     (std_acc_mag). The feature channels instead RE-SEPARATE longitudinal
     acceleration (mean_acc_pos), longitudinal deceleration (mean_acc_neg)
     and lateral acceleration (mean_abs_a_lat), because the clustering needs
     the directional decomposition that a single magnitude erases.
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

sys.path.append(str(Path(__file__).resolve().parents[1]))
from thesis_paths import DATA_ROOT, REPO_ROOT  # noqa: E402

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS  (edit here only)
# ─────────────────────────────────────────────────────────────────────────────

WINDOW_S       = 20.0   # window size in seconds
STEP_S         = 20.0   # step size in seconds (== WINDOW_S: non-overlapping)
GRID_HZ        = 10     # sampling rate of the standardized clips (uniform)
MIN_VALID_FRAC = 0.5    # per signal: min fraction of expected frames that
                        # must be non-NaN, else the statistic = NaN for that signal
MIN_VALID_ABS  = 2      # absolute floor (a statistic needs >= 2 points regardless)
APPLY_BOUNDS   = True   # mask physically impossible values before any statistic

# Magnitude bound implied by the per-component jerk bound (±30 m/s^3 per axis),
# exactly parallel to acc_mag's 21.3 = sqrt(15^2 + 15^2).
JERK_MAG_BOUND = round(float(np.hypot(30.0, 30.0)), 2)   # 42.43 m/s^3

# Physical-validity bounds (identical to the visualization / frame layer; yaw_rate
# in rad/s as stored). Applied as NaN masking at READ time, never modifying the
# source clips. Rationale: a single residual 5,000 km/h frame in a 200-frame
# window inflates that window's statistic by orders of magnitude and poisons every
# downstream figure; out-of-bounds fractions remain quantified by the
# boundary-coverage audit, so nothing is hidden by this masking.
BOUNDS = {
    "speed_kmh": (0.0,   200.0),
    "acc_mag":   (0.0,    21.3),   # = sqrt(15^2 + 15^2): magnitude implied by the ±15 per-axis bound
    "yaw_rate":  (-1.57,   1.57),  # rad/s, as stored
    # ── feature-channel sources ──
    "a_long":    (-15.0,  15.0),   # per-axis longitudinal acceleration bound
    "a_lat":     (-15.0,  15.0),   # per-axis lateral acceleration bound
    "jerk_mag":  (0.0, JERK_MAG_BOUND),
}

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

INERTIAL_ROOT = DATA_ROOT / "inertial_data"
STD_ROOT      = DATA_ROOT / "std_data"

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL / OUTPUT COLUMNS
# ─────────────────────────────────────────────────────────────────────────────

# ── variability channels (std) ──
# acc_mag is read directly from the standardized clips (computed there as
# sqrt(a_long^2 + a_lat^2), already gap/edge-masked), not recomputed here.
SIGNAL_COLS = ["speed_kmh", "acc_mag", "yaw_rate"]
STD_COLS    = ["std_speed_kmh", "std_acc_mag", "std_yaw_rate"]

# ── feature-mean channels (Section 4.7) ──
# Output mean columns, plus the additional standardized-clip signals they are
# built from. speed_kmh and yaw_rate are already read for the std channels;
# acc_mag and jerk_mag are precomputed upstream and read directly.
MEAN_COLS     = ["mean_speed_kmh", "mean_acc_pos", "mean_acc_neg",
                 "mean_abs_a_lat", "mean_abs_yaw_rate", "mean_abs_jerk"]
MEAN_SRC_COLS = ["a_long", "a_lat", "jerk_mag"]

# All columns needed from each clip (deduped, order-preserving).
NEEDED_COLS = list(dict.fromkeys(["t_rel_s"] + SIGNAL_COLS + MEAN_SRC_COLS))

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
    Expected frames for this window = duration_s × GRID_HZ (uniform grid).
    A signal whose non-NaN count falls below max(MIN_VALID_ABS,
    MIN_VALID_FRAC × expected) gets NaN. Gap/edge-masked windows must not
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


def _window_means(w_df: pd.DataFrame, duration_s: float) -> dict:
    """
    Per-signal within-window MEAN with the SAME MIN_VALID_FRAC rule used for the
    stds. These six channels form the per-window feature vector consumed by the
    regime clustering of Section 4.7. A mean built from a handful of surviving
    frames is as unreliable as a std, so the identical validity gate applies.
    Longitudinal acceleration is split into its positive (accel) and negative
    (decel) parts so that the two are not cancelled by averaging over the window.
    """
    n_expected = max(1.0, duration_s * GRID_HZ)
    min_valid  = max(MIN_VALID_ABS, int(np.ceil(MIN_VALID_FRAC * n_expected)))

    out = {c: float("nan") for c in MEAN_COLS}

    # translational pace
    v = w_df["speed_kmh"]
    if int(v.notna().sum()) >= min_valid:
        out["mean_speed_kmh"] = float(v.mean())

    # longitudinal acceleration, split into accel (+) and decel (-) intensity
    al = w_df["a_long"].to_numpy(dtype=float)
    al = al[~np.isnan(al)]
    if al.size >= min_valid:
        out["mean_acc_pos"] = float(np.maximum(al,  0.0).mean())
        out["mean_acc_neg"] = float(np.maximum(-al, 0.0).mean())

    # lateral acceleration intensity
    alat = w_df["a_lat"]
    if int(alat.notna().sum()) >= min_valid:
        out["mean_abs_a_lat"] = float(alat.abs().mean())

    # rotational activity (rad/s as stored; deg conversion deferred to viz)
    yr = w_df["yaw_rate"]
    if int(yr.notna().sum()) >= min_valid:
        out["mean_abs_yaw_rate"] = float(yr.abs().mean())

    # total jerk intensity (jerk_mag is precomputed and already non-negative)
    jm = w_df["jerk_mag"]
    if int(jm.notna().sum()) >= min_valid:
        out["mean_abs_jerk"] = float(jm.mean())

    return out


# ─────────────────────────────────────────────────────────────────────────────
# WINDOW COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def _compute_windows_for_clip(fp: Path) -> list[dict]:
    """
    Extract all windows from a single clip and compute std AND mean per signal.

    Windowing rules:
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
        dur   = t_max - t_min
        stds  = _window_stds(df,  duration_s=dur)
        means = _window_means(df, duration_s=dur)
        rows.append({
            "clip_id":        clip_id,
            "window_idx":     0,
            "window_start_s": t_min,
            "window_end_s":   t_max,
            "n_frames":       len(df),
            **stds,
            **means,
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
            stds  = _window_stds(w_df,  duration_s=WINDOW_S)
            means = _window_means(w_df, duration_s=WINDOW_S)
            rows.append({
                "clip_id":        clip_id,
                "window_idx":     window_idx,
                "window_start_s": float(t_start),
                "window_end_s":   float(t_end),
                "n_frames":       int(mask.sum()),
                **stds,
                **means,
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
    # Stable column order: identifiers, then variability channels, then features.
    ordered = (["clip_id", "window_idx", "window_start_s", "window_end_s", "n_frames"]
               + STD_COLS + MEAN_COLS)
    result_df = result_df[[c for c in ordered if c in result_df.columns]]
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
    print(f"  compute_std.py  (v4 — variability stds + feature means)")
    print(f"  Window: {WINDOW_S}s   Step: {STEP_S}s   Grid: {GRID_HZ} Hz")
    print(f"  Min valid: {MIN_VALID_FRAC:.0%} of expected frames per statistic")
    print(f"  Bounds masking: {'ON' if APPLY_BOUNDS else 'OFF'}  (jerk_mag ≤ {JERK_MAG_BOUND})")
    print(f"  Variability stds: {STD_COLS}")
    print(f"  Feature means:    {MEAN_COLS}")
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