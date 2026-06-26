"""
check_std_data.py — sanity-check the v4 std_data layer (the A1 output).

Verifies the new feature-mean channels exist, sit inside their physical ranges,
and how complete they are, then prints a quick correlation PREVIEW of the
candidate vector. This is a verification probe, NOT the A2 feature gate:
no non-overlap filter, no Spearman, no VIF, no figure, no vector locking.
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k):
        return x

STD_DATA_DIR = Path("/Users/edward/Downloads/dataset/std_data")

# Candidate feature vector (Section 9.1). std_speed_kmh is reused from the
# variability channels; the other six are the v4 means.
FEATURES = ["mean_speed_kmh", "std_speed_kmh", "mean_acc_pos", "mean_acc_neg",
            "mean_abs_a_lat", "mean_abs_yaw_rate", "mean_abs_jerk"]

# Expected ranges — means of bounded / non-negative per-frame quantities.
EXPECTED = {
    "mean_speed_kmh":    (0.0, 200.0),
    "std_speed_kmh":     (0.0, 200.0),
    "mean_acc_pos":      (0.0,  15.0),
    "mean_acc_neg":      (0.0,  15.0),
    "mean_abs_a_lat":    (0.0,  15.0),
    "mean_abs_yaw_rate": (0.0,   1.57),
    "mean_abs_jerk":     (0.0,  42.43),
}

CORR_SAMPLE = 300_000   # cap pooled rows for the correlation preview


def _load(fp: Path):
    try:
        import pyarrow.parquet as pq
        have = set(pq.read_schema(fp).names)
        miss = [c for c in FEATURES if c not in have]
        cols = [c for c in FEATURES if c in have]
        df = pd.read_parquet(fp, columns=cols) if cols else pd.DataFrame()
        return fp.stem, df, miss
    except Exception as e:
        return fp.stem, None, [f"<read error: {e}>"]


def main():
    files = sorted(STD_DATA_DIR.glob("*.parquet"))
    if not files:
        print(f"[ERROR] no parquet files in {STD_DATA_DIR}")
        return

    with ThreadPoolExecutor(max_workers=min(8, len(files))) as ex:
        loaded = list(tqdm(ex.map(_load, files), total=len(files), desc="reading"))

    frames = {name: df for name, df, _ in loaded if df is not None and len(df.columns)}

    # ── per-dataset completeness ────────────────────────────────────────────
    print("\n" + "=" * 84)
    print("  PER-DATASET COMPLETENESS")
    print("=" * 84)
    print(f"  {'dataset':<14}{'windows':>10}   notes")
    print("  " + "-" * 80)
    total_w = 0
    for name, df, miss in sorted(loaded):
        if df is None:
            print(f"  {name:<14}{'—':>10}   {miss}")
            continue
        n = len(df)
        total_w += n
        note = "ok" if not miss else f"MISSING {miss}  (did this dataset get re-run?)"
        print(f"  {name:<14}{n:>10,}   {note}")
    print("  " + "-" * 80)
    print(f"  {'TOTAL':<14}{total_w:>10,}")

    if not frames:
        print("\n[ERROR] no feature columns found anywhere — rerun compute_std.py (v4).")
        return

    alldf = pd.concat(frames.values(), ignore_index=True)

    # ── per-feature summary + range check ───────────────────────────────────
    print("\n" + "=" * 84)
    print("  FEATURE SUMMARY  (pooled across all datasets)")
    print("=" * 84)
    print(f"  {'feature':<20}{'NaN%':>8}{'min':>11}{'max':>11}{'mean':>11}{'median':>11}  status")
    print("  " + "-" * 80)
    for c in FEATURES:
        if c not in alldf.columns:
            print(f"  {c:<20}{'—':>8}   COLUMN ABSENT")
            continue
        s = alldf[c]
        nanp = 100.0 * s.isna().mean()
        lo, hi = EXPECTED[c]
        vmin, vmax = float(s.min()), float(s.max())
        ok = (vmin >= lo - 1e-6) and (vmax <= hi + 1e-6)
        print(f"  {c:<20}{nanp:>7.1f}%{vmin:>11.3f}{vmax:>11.3f}"
              f"{float(s.mean()):>11.3f}{float(s.median()):>11.3f}  "
              f"{'OK' if ok else 'OUT OF RANGE'}")

    # ── correlation PREVIEW (not the gate; DurLAR excluded as its long. is noise)
    have = [c for c in FEATURES if c in alldf.columns]
    corr_frames = [df for name, df in frames.items() if name != "durlar"]
    cc = pd.concat(corr_frames, ignore_index=True)[have].dropna()
    if len(cc) > CORR_SAMPLE:
        cc = cc.sample(CORR_SAMPLE, random_state=0)

    print("\n" + "=" * 84)
    print(f"  PEARSON PREVIEW  (n={len(cc):,} complete rows, DurLAR excluded — PREVIEW, not the A2 gate)")
    print("=" * 84)
    corr = cc.corr(method="pearson").round(2)
    with pd.option_context("display.width", 110, "display.max_columns", 20):
        print(corr.to_string())

    if {"mean_abs_a_lat", "mean_abs_yaw_rate"} <= set(have):
        r = float(corr.loc["mean_abs_a_lat", "mean_abs_yaw_rate"])
        verdict = ("collinear as predicted (a_lat = v·ω) — the gate will likely drop one"
                   if abs(r) > 0.85 else
                   "below the 0.85 bar — both survive into the gate for now")
        print(f"\n  a_lat vs yaw_rate:  r = {r:+.2f}   -> {verdict}")
    print("=" * 84 + "\n")


if __name__ == "__main__":
    main()