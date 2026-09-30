#!/usr/bin/env python3
"""coverage_baseline.py — regenerate audit_baseline.json from the standardized clips.

WHY THIS EXISTS
---------------
`run_pipeline.py` documents the coverage stage as producing audit_baseline.json
and lists it as that stage's artifact, but coverage.py contains no JSON write:
it emits three PNGs only. The artifact spec is a plain Path checked with
.exists(), so the stale file satisfies the check and the stage reports OK while
producing nothing. The producer was removed at some point and the spec was never
updated to notice.

The consequence: audit_baseline.json is pinned at
v4_10hz_common_grid_validity_gated while the clips are v4_4, so it still shows
Boreas at 110 clips, DurLAR at 4, Oxford at 70 with a 200 km/h maximum, and
4Seasons as EXCLUDED_MEASUREMENT_LIMITED — all pre-cleaning. Section 7.3 of the
thesis quotes those numbers.

This script restores the producer. The schema matches the existing file exactly
so nothing downstream needs changing.

FIELD DEFINITIONS (inferred from the v4 file and the documented gates)
  clips          number of standardized clip files
  frames         total rows across them
  max_kmh        max speed_kmh
  max_acc        max per-axis |a_long|, |a_lat|      (gate 15 m/s^2)
  max_jerk       max per-axis |jerk_long|, |jerk_lat| (gate 30 m/s^3)
  max_yaw_degs   max |yaw_rate| in deg/s             (gate 90 deg/s)
  n_spd_over     frames above the 200 km/h speed gate
  n_spd_108      frames above 108 km/h (the residual-contamination check)
  n_acc_over     frames above the 15 m/s^2 per-axis acceleration gate
  mean_speed     mean speed_kmh over valid frames
  speed_src      col_sources["speed"] from file metadata
  yaw_rate_src   col_sources["yaw_rate"] from file metadata
  version        processing_meta["pipeline_version"]
  gated_pct      frame-weighted mean of processing_meta["gated_frac_speed"], in %

Usage
    python coverage_baseline.py                 # write audit_baseline.json
    python coverage_baseline.py --dry-run       # print, write nothing
    python coverage_baseline.py --diff          # compare to the existing file
    python coverage_baseline.py --only nvidia   # one dataset (debugging)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/Users/edward/Downloads")
INERTIAL_DIR = PROJECT_ROOT / "dataset" / "inertial_data"
OUT_PATH = (PROJECT_ROOT / "bachelor-arbeit" / "04_exploratory_analysis"
            / "audit_baseline.json")

DATASETS = ["4seasons", "a2d2", "boreas_rt", "durlar", "ford", "hdd",
            "kitscenes", "kitti", "kitti360", "nuscene", "nvidia", "oxford",
            "waymo", "zod"]

# Published bounds (Section 8.1). Kept here as the reporting thresholds only;
# enforcement lives in standardize_inertial_data.py.
SPEED_GATE_KMH = 200.0
SPEED_WATCH_KMH = 108.0
ACC_GATE_MS2 = 15.0

NEEDED = ["speed_kmh", "a_long", "a_lat", "jerk_long", "jerk_lat", "yaw_rate"]
MAX_WORKERS = 8


# ---------------------------------------------------------------------------
def _nanmax(a: float, b: float) -> float:
    if math.isnan(a):
        return b
    if math.isnan(b):
        return a
    return max(a, b)


def _read_meta(path: Path) -> tuple[dict, dict]:
    """col_sources and processing_meta from a clip's file-level metadata."""
    md = pq.ParquetFile(path).schema_arrow.metadata or {}
    out = {}
    for k, v in md.items():
        key = k.decode() if isinstance(k, bytes) else k
        if key in ("col_sources", "processing_meta"):
            try:
                out[key] = json.loads(v.decode())
            except Exception:
                out[key] = {}
    return out.get("col_sources", {}), out.get("processing_meta", {})


def _scan_clip(path: Path) -> dict:
    """Per-clip accumulators. Never holds more than one clip in memory."""
    try:
        cols = set(pq.ParquetFile(path).schema_arrow.names)
    except Exception as exc:
        return {"error": f"{path.name}: {exc}"}
    use = [c for c in NEEDED if c in cols]
    df = pd.read_parquet(path, columns=use) if use else pd.DataFrame()

    def col(name):
        return (df[name].to_numpy(dtype=float)
                if name in df.columns else np.array([], dtype=float))

    spd = col("speed_kmh")
    acc = np.concatenate([np.abs(col("a_long")), np.abs(col("a_lat"))]) \
        if ("a_long" in df.columns or "a_lat" in df.columns) \
        else np.array([], dtype=float)
    jrk = np.concatenate([np.abs(col("jerk_long")), np.abs(col("jerk_lat"))]) \
        if ("jerk_long" in df.columns or "jerk_lat" in df.columns) \
        else np.array([], dtype=float)
    yaw = np.abs(col("yaw_rate")) * 180.0 / np.pi

    def mx(a):
        a = a[np.isfinite(a)]
        return float(a.max()) if a.size else float("nan")

    spd_f = spd[np.isfinite(spd)]
    _, pmeta = _read_meta(path)
    return {
        "frames": int(len(df)),
        "max_kmh": mx(spd),
        "max_acc": mx(acc),
        "max_jerk": mx(jrk),
        "max_yaw_degs": mx(yaw),
        "n_spd_over": int((spd_f > SPEED_GATE_KMH).sum()),
        "n_spd_108": int((spd_f > SPEED_WATCH_KMH).sum()),
        "n_acc_over": int((acc[np.isfinite(acc)] > ACC_GATE_MS2).sum()),
        "speed_sum": float(spd_f.sum()),
        "speed_n": int(spd_f.size),
        "gated_frac": float(pmeta.get("gated_frac_speed", 0.0) or 0.0),
    }


def scan_dataset(ds: str, quiet: bool = False) -> dict | None:
    d = INERTIAL_DIR / ds
    files = sorted(d.glob("*.parquet"))
    if not files:
        print(f"  {ds}: NO CLIPS at {d}", file=sys.stderr)
        return None

    src, pmeta = _read_meta(files[0])

    agg = {"frames": 0, "n_spd_over": 0, "n_spd_108": 0, "n_acc_over": 0,
           "speed_sum": 0.0, "speed_n": 0, "gated_weighted": 0.0}
    for k in ("max_kmh", "max_acc", "max_jerk", "max_yaw_degs"):
        agg[k] = float("nan")
    errors = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        it = ex.map(_scan_clip, files)
        if not quiet:
            it = tqdm(it, total=len(files), desc=f"  {ds:<10}",
                      unit="clip", leave=False)
        for r in it:
            if "error" in r:
                errors.append(r["error"])
                continue
            agg["frames"] += r["frames"]
            for k in ("n_spd_over", "n_spd_108", "n_acc_over"):
                agg[k] += r[k]
            for k in ("max_kmh", "max_acc", "max_jerk", "max_yaw_degs"):
                agg[k] = _nanmax(agg[k], r[k])
            agg["speed_sum"] += r["speed_sum"]
            agg["speed_n"] += r["speed_n"]
            agg["gated_weighted"] += r["gated_frac"] * r["frames"]

    if errors:
        print(f"  {ds}: {len(errors)} unreadable clip(s), first: {errors[0]}",
              file=sys.stderr)

    return {
        "clips": len(files),
        "frames": agg["frames"],
        "max_kmh": agg["max_kmh"],
        "max_acc": agg["max_acc"],
        "max_jerk": agg["max_jerk"],
        "max_yaw_degs": agg["max_yaw_degs"],
        "n_spd_over": agg["n_spd_over"],
        "n_spd_108": agg["n_spd_108"],
        "n_acc_over": agg["n_acc_over"],
        "mean_speed": (agg["speed_sum"] / agg["speed_n"]
                       if agg["speed_n"] else 0.0),
        "speed_src": src.get("speed", "UNKNOWN"),
        "yaw_rate_src": src.get("yaw_rate", "UNKNOWN"),
        "version": pmeta.get("pipeline_version", "UNKNOWN"),
        "gated_pct": (100.0 * agg["gated_weighted"] / agg["frames"]
                      if agg["frames"] else 0.0),
    }


# ---------------------------------------------------------------------------
def show(baseline: dict) -> None:
    print(f"\n{'dataset':<11}{'clips':>8}{'frames':>13}{'gated%':>9}"
          f"{'speed_src':>14}{'yaw_src':>12}{'max_kmh':>10}{'mean':>8}"
          f"{'>108':>9}")
    for ds, b in baseline.items():
        print(f"{ds:<11}{b['clips']:>8,}{b['frames']:>13,}"
              f"{b['gated_pct']:>9.2f}{str(b['speed_src'])[:13]:>14}"
              f"{str(b['yaw_rate_src'])[:11]:>12}{b['max_kmh']:>10.2f}"
              f"{b['mean_speed']:>8.2f}{b['n_spd_108']:>9,}")
    vers = sorted({b["version"] for b in baseline.values()})
    print(f"\npipeline version(s) present: {vers}")
    if len(vers) > 1:
        print("  WARNING: clips were produced by more than one pipeline "
              "version. Re-run standardization before trusting this baseline.")


def diff_against(new: dict, path: Path) -> None:
    if not path.exists():
        print(f"[diff] no existing file at {path}")
        return
    old = json.loads(path.read_text())
    print(f"\n{'dataset':<11}{'clips old/new':>18}{'frames old/new':>26}"
          f"{'version old':>36}")
    for ds in sorted(set(old) | set(new)):
        o, n = old.get(ds), new.get(ds)
        if o is None:
            print(f"{ds:<11}  ADDED")
            continue
        if n is None:
            print(f"{ds:<11}  REMOVED (in old file, no clips now)")
            continue
        flag = "  <-- moved" if (o["clips"] != n["clips"]
                                 or o["frames"] != n["frames"]) else ""
        print(f"{ds:<11}{o['clips']:>8,}/{n['clips']:<9,}"
              f"{o['frames']:>12,}/{n['frames']:<13,}"
              f"{str(o.get('version', '?'))[:35]:>36}{flag}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="print, do not write")
    ap.add_argument("--diff", action="store_true",
                    help="compare against the existing file before writing")
    ap.add_argument("--only", default=None, help="one dataset, for debugging")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    targets = [args.only] if args.only else DATASETS
    print(f"scanning {len(targets)} dataset(s) under {INERTIAL_DIR}\n")

    baseline = {}
    for ds in targets:
        r = scan_dataset(ds)
        if r is not None:
            baseline[ds] = r

    show(baseline)
    if args.diff:
        diff_against(baseline, Path(args.out))

    if args.dry_run:
        print("\n[dry-run] nothing written")
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(baseline, indent=1))
    print(f"\nwritten: {out}")
    print("Section 7.3's coverage paragraph can now be replaced with these "
          "post-cleaning values.")
    return 0


if __name__ == "__main__":
    sys.exit(main())