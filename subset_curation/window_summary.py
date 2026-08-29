#!/usr/bin/env python3
"""
window_summary.py — one-time frame-layer pass (run BEFORE run_sampling if you
want M07/M08; everything else works without it).

v2 fixes after the first real run:
  * clip resolver now normalizes numeric ids, so `clip_id = 7` (int) or
    "clip_7" matches a file named clip_0007.parquet regardless of zero
    padding — this recovers the exactly-999 misses seen on nvidia, waymo,
    and kitscenes.
  * an EMPTY partial no longer counts as "done": the resume logic re-runs
    any dataset whose partial has zero rows (the old behavior permanently
    skipped the twelve failed datasets).
  * empty partials are excluded from the final concat (kills the pandas
    FutureWarning) and a per-dataset coverage report prints at the end.

For every window in std_data, reads the window's frames from inertial_data
via the back-reference contract (clip_id, window_start_s, window_end_s) and
computes per-window kinematic summaries that std_data does not store:

    mean_speed_kmh, max_speed_kmh, max_decel_ms2, max_abs_yaw_degs,
    max_acc_mag, n_frames_used

Output: SUBSET_DIR / window_summaries.parquet  (keyed dataset, clip_id,
window_idx). Per-dataset partials live in SUBSET_DIR/_summaries_partial; a
partial is reused only if its (clip_id, window_idx) key set exactly matches
the CURRENT std_data for that dataset, so a regenerated std_data invalidates
stale partials automatically instead of being skipped forever. The final
write hard-verifies full coverage against std_data and exits nonzero on any
mismatch, failing here rather than one stage later in the evaluation
harness's join guard.
"""

from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

# =========================================================================
# Canonical constants
# =========================================================================
PROJECT_ROOT = Path("/Users/edward/Downloads")
INERTIAL_DIR = PROJECT_ROOT / "dataset" / "inertial_data"
STD_DATA_DIR = PROJECT_ROOT / "dataset" / "std_data"
SUBSET_DIR = PROJECT_ROOT / "dataset" / "subset_data"
PARTIAL_DIR = SUBSET_DIR / "_summaries_partial"
OUT_PATH = SUBSET_DIR / "window_summaries.parquet"

N_THREADS = 8                    # I/O-bound clip reads per dataset
RAD2DEG = 180.0 / np.pi

# Frame-layer columns we try to read (missing ones are handled):
TIME_COL = "t_rel_s"
SPEED_COL = "speed_kmh"
YAW_COL = "yaw_rate"             # stored rad/s -> converted to deg/s here
ACCMAG_COL = "acc_mag"           # already computed upstream — never re-derive
DECEL_COL = "deceleration"       # preferred; fallback = max(-a_long, 0)
ALONG_COL = "a_long"

SUMMARY_COLS = ["mean_speed_kmh", "max_speed_kmh", "max_decel_ms2",
                "max_abs_yaw_degs", "max_acc_mag", "n_frames_used"]

_NUM_RE = re.compile(r"(?:clip[_-]?)?0*(\d+)$")


def log(msg: str) -> None:
    print(f"[summaries] {msg}", flush=True)


def norm_key(s) -> str | None:
    """Normalize 'clip_0007', 'clip_7', '0007', 7 -> '#num:7'. None if
    the value has no purely-numeric form."""
    s = str(s)
    if s.endswith(".parquet"):
        s = s[:-8]
    m = _NUM_RE.fullmatch(s)
    return f"#num:{int(m.group(1))}" if m else None


def clip_file_index(ds_dir: Path) -> dict:
    """Map raw stems AND normalized numeric keys to file paths."""
    index: dict = {}
    collisions = 0
    for p in ds_dir.glob("*.parquet"):
        index[p.stem] = p
        nk = norm_key(p.stem)
        if nk is not None:
            if nk in index and index[nk] != p:
                collisions += 1
            else:
                index.setdefault(nk, p)
    if collisions:
        log(f"  WARNING: {collisions} numeric-key collisions in {ds_dir.name} "
            f"(kept first occurrence)")
    return index


def resolve_clip_path(index: dict, clip_id) -> Path | None:
    cid = str(clip_id)
    for key in (cid, f"clip_{cid}", cid.replace(".parquet", "")):
        if key in index:
            return index[key]
    nk = norm_key(cid)
    if nk is not None and nk in index:
        return index[nk]
    return None


def partial_is_done(partial: Path) -> bool:
    """A partial counts as done only if it exists AND has rows."""
    if not partial.exists():
        return False
    try:
        return pq.ParquetFile(partial).metadata.num_rows > 0
    except Exception:
        return False


def process_clip(path: Path, windows: np.ndarray) -> list:
    """windows: array of (window_idx, start_s, end_s). Returns summary rows."""
    schema_names = set(pq.ParquetFile(path).schema_arrow.names)
    want = [c for c in (TIME_COL, SPEED_COL, YAW_COL, ACCMAG_COL,
                        DECEL_COL, ALONG_COL) if c in schema_names]
    if TIME_COL not in want:
        return [(int(w), np.nan, np.nan, np.nan, np.nan, np.nan, 0)
                for w, _, _ in windows]
    tbl = pq.read_table(path, columns=want)
    df = tbl.to_pandas()

    t = df[TIME_COL].to_numpy(dtype=float)
    if len(t) > 1 and not np.all(np.diff(t) >= 0):
        order = np.argsort(t, kind="mergesort")
        df = df.iloc[order].reset_index(drop=True)
        t = df[TIME_COL].to_numpy(dtype=float)

    speed = df[SPEED_COL].to_numpy(dtype=float) if SPEED_COL in df else None
    yaw = df[YAW_COL].to_numpy(dtype=float) if YAW_COL in df else None
    accm = df[ACCMAG_COL].to_numpy(dtype=float) if ACCMAG_COL in df else None
    if DECEL_COL in df:
        decel = np.maximum(df[DECEL_COL].to_numpy(dtype=float), 0.0)
    elif ALONG_COL in df:
        decel = np.maximum(-df[ALONG_COL].to_numpy(dtype=float), 0.0)
    else:
        decel = None

    rows = []
    for widx, start, end in windows:
        lo = np.searchsorted(t, start, side="left")
        hi = np.searchsorted(t, end, side="left")
        n = int(hi - lo)
        if n <= 0:
            rows.append((int(widx), np.nan, np.nan, np.nan, np.nan, np.nan, 0))
            continue

        def agg(arr, fn):
            if arr is None:
                return np.nan
            seg = arr[lo:hi]
            seg = seg[np.isfinite(seg)]
            return float(fn(seg)) if len(seg) else np.nan

        rows.append((
            int(widx),
            agg(speed, np.mean),
            agg(speed, np.max),
            agg(decel, np.max),
            agg(yaw, lambda s: np.max(np.abs(s)) * RAD2DEG),
            agg(accm, np.max),
            n,
        ))
    return rows


def process_dataset(ds: str) -> None:
    partial = PARTIAL_DIR / f"{ds}.parquet"
    std = pd.read_parquet(STD_DATA_DIR / f"{ds}.parquet",
                          columns=["clip_id", "window_idx",
                                   "window_start_s", "window_end_s"])
    std_keys = set(zip(std["clip_id"].astype(str),
                       std["window_idx"].astype(int)))
    if partial.exists():
        try:
            part = pd.read_parquet(partial,
                                   columns=["clip_id", "window_idx"])
            part_keys = set(zip(part["clip_id"].astype(str),
                                part["window_idx"].astype(int)))
        except Exception:
            part_keys = set()
        if part_keys and part_keys == std_keys:
            # keys matching is NOT freshness: a standardization re-run with an
            # unchanged window grid reproduces identical keys over new values,
            # and reusing the partial would silently keep the old summaries
            # (the exact shape of the 1,115-vs-1,088 confusion).
            src_mtime = (STD_DATA_DIR / f"{ds}.parquet").stat().st_mtime
            ds_dir_m = INERTIAL_DIR / ds
            if ds_dir_m.exists():
                src_mtime = max([src_mtime] +
                                [e.stat().st_mtime for e in os.scandir(ds_dir_m)
                                 if e.name.endswith(".parquet")])
            if partial.stat().st_mtime >= src_mtime:
                log(f"{ds}: partial matches current std_data "
                    f"({len(part_keys):,} windows) and is newer than every "
                    f"input, skipping")
                return
            log(f"{ds}: partial keys match but an input is NEWER "
                f"(std_data or clips rewritten on an unchanged grid) — "
                f"rebuilding to avoid stale values")
        log(f"{ds}: STALE partial ({len(part_keys):,} cached vs "
            f"{len(std_keys):,} current std windows; std_data "
            f"regenerated?) - rebuilding")
    ds_dir = INERTIAL_DIR / ds
    if not ds_dir.exists():
        log(f"{ds}: WARNING inertial dir missing ({ds_dir}) — skipped")
        return
    index = clip_file_index(ds_dir)
    grouped = {cid: g[["window_idx", "window_start_s", "window_end_s"]
                     ].to_numpy()
               for cid, g in std.groupby("clip_id", sort=False)}

    misses, results = 0, []
    miss_examples: list = []
    t0 = time.time()

    def work(cid, wins):
        path = resolve_clip_path(index, cid)
        if path is None:
            return cid, None
        return cid, process_clip(path, wins)

    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        futs = [ex.submit(work, cid, wins) for cid, wins in grouped.items()]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=ds):
            cid, rows = fut.result()
            if rows is None:
                misses += 1
                if len(miss_examples) < 3:
                    miss_examples.append(str(cid))
                continue
            for r in rows:
                results.append((cid, *r))

    out = pd.DataFrame(results, columns=["clip_id", "window_idx",
                                         *SUMMARY_COLS])
    out.insert(0, "dataset", ds)
    out.to_parquet(partial, index=False)
    miss_rate = misses / max(len(grouped), 1)
    log(f"{ds}: {len(out):,}/{len(std):,} windows summarized, "
        f"{misses} clip files not found ({miss_rate:.1%}), "
        f"{time.time() - t0:.0f}s")
    if misses:
        log(f"{ds}: unresolved clip_id examples: {miss_examples}")
    if miss_rate > 0.05:
        log(f"{ds}: WARNING >5% clip files unresolved — run "
            f"inspect_clip_mapping.py and paste its output back.")


def main():
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    datasets = sorted(p.stem for p in STD_DATA_DIR.glob("*.parquet"))
    if not datasets:
        raise SystemExit(f"[summaries] no std_data parquets in {STD_DATA_DIR}")
    log(f"datasets: {datasets}")
    for ds in datasets:                       # sequential across datasets,
        process_dataset(ds)                   # threaded within (I/O bound)

    parts = []
    for p in sorted(PARTIAL_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        if len(df):
            parts.append(df)
    if not parts:
        raise SystemExit("[summaries] no non-empty partials — nothing written")
    full = pd.concat(parts, ignore_index=True)
    full.to_parquet(OUT_PATH, index=False)
    log(f"window_summaries.parquet written: {len(full):,} rows -> {OUT_PATH}")

    # Coverage verification vs std_data. Rows are one-per-std-window by
    # construction (empty windows persist as NaN rows), so the contract is
    # exact equality; the old >=95% advisory let a stale tail sit under it.
    log("coverage verification (summarized windows / std windows):")
    got = full.groupby("dataset").size()
    bad = []
    for ds in datasets:
        n_std = pq.ParquetFile(STD_DATA_DIR / f"{ds}.parquet").metadata.num_rows
        n_got = int(got.get(ds, 0))
        flag = "" if n_got == n_std else "   <-- MISMATCH"
        log(f"  {ds:12s} {n_got:>9,} / {n_std:>9,}{flag}")
        if n_got != n_std:
            bad.append(ds)
    if bad:
        raise SystemExit(
            f"[summaries] coverage mismatch for {', '.join(bad)}: the cache "
            f"does not cover current std_data. Delete the listed partial(s) "
            f"under {PARTIAL_DIR} if the rebuild above did not already "
            f"replace them, and re-run.")


if __name__ == "__main__":
    main()