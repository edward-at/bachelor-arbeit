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

WINDOW_S               = 20.0    # window size in seconds
STEP_S                 = 10.0    # step size in seconds (50% overlap)
LARGE_DATASET_THRESHOLD = 999_999_999  # effectively disabled — all datasets use normal mode

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
        # Fill missing signal cols with NaN silently
        for c in NEEDED_COLS:
            if c not in df.columns:
                df[c] = np.nan
        return df
    except Exception as e:
        print(f"  [WARN] Could not read {fp.name}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# WINDOW COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def _compute_windows_for_clip(fp: Path) -> list[dict]:
    """
    Extract all windows from a single clip and compute std per signal.

    Windowing rules:
    - t_max >= WINDOW_S : sliding window with STEP_S step
    - t_max <  WINDOW_S : entire clip = single window (window_idx=0)

    Returns a list of row dicts matching the output schema.
    """
    df = _read_clip(fp)
    if df is None or len(df) < 2:
        return []

    clip_id  = _clip_id_from_path(fp)
    t        = df["t_rel_s"].values.astype(float)
    t_min    = float(t.min())
    t_max    = float(t.max())

    rows = []

    # ── short clip: 1 clip = 1 window ────────────────────────────────────────
    if t_max - t_min < WINDOW_S:
        mask   = np.ones(len(df), dtype=bool)
        w_df   = df
        stds   = {sc: float(w_df[rc].std(ddof=1)) if w_df[rc].notna().sum() > 1
                  else float("nan")
                  for sc, rc in zip(STD_COLS, SIGNAL_COLS)}
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
            stds = {sc: float(w_df[rc].std(ddof=1)) if w_df[rc].notna().sum() > 1
                    else float("nan")
                    for sc, rc in zip(STD_COLS, SIGNAL_COLS)}
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

    n_clips    = len(clip_files)
    is_large   = n_clips > LARGE_DATASET_THRESHOLD
    all_rows   = []
    n_windows  = 0

    if is_large:
        # ── batch mode: process BATCH_SIZE clips at a time ───────────────────
        print(f"  [INFO] {ds_name}: {n_clips:,} clips — batch mode "
              f"(batch size {BATCH_SIZE:,})")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        first_batch = True

        with tqdm(total=n_clips, desc=f"  {ds_name:<12}", unit="clip", leave=True) as pbar:
            for batch_start in range(0, n_clips, BATCH_SIZE):
                batch = clip_files[batch_start: batch_start + BATCH_SIZE]
                batch_rows = []
                for fp in batch:
                    batch_rows.extend(_compute_windows_for_clip(fp))
                    pbar.update(1)

                if batch_rows:
                    batch_df = pd.DataFrame(batch_rows)
                    if first_batch:
                        batch_df.to_parquet(out_path, index=False)
                        first_batch = False
                    else:
                        existing = pd.read_parquet(out_path)
                        pd.concat([existing, batch_df], ignore_index=True) \
                          .to_parquet(out_path, index=False)
                    n_windows += len(batch_rows)

    else:
        # ── normal mode: load all clips ───────────────────────────────────────
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

    n_workers  = min(len(datasets), os.cpu_count() or 4)
    t_start    = datetime.now()

    print(f"\n{'='*65}")
    print(f"  compute_std.py")
    print(f"  Window: {WINDOW_S}s   Step: {STEP_S}s")
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