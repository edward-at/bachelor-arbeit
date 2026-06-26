#!/usr/bin/env python3
"""
verify_section6.py — checks the Section 6 summary table against the raw
per-clip Parquet files in parquet_data/ (native Hz, pre-standardization).

  clips  = #(*.parquet) per dataset      -> HARD pass/fail vs Section 6
  rows   = sum of footer row counts      -> HARD pass/fail vs Section 6 (no data loaded)
  ncols  = native schema column count    -> printed for review
  Hz     = median timestamp delta -> Hz  -> printed for review (settles Ford 155 vs 157)

Reads parquet_data/, NOT inertial_data/ (that is the 43-col unified schema).
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pyarrow.parquet as pq

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k): return x

# ---------------------------------------------------------------- CONFIG
RAW_ROOT = Path("/Users/edward/Downloads/dataset/parquet_data")
N_WORKERS = 8
HZ_SAMPLE_FILES = 8
TS_CANDIDATES = ("timestamp", "ts_ns", "ts", "time", "t", "utime")

# Section 6 claims -> folder_name: (clips, rows, native_Hz)
EXPECTED = {
    "kitti":     (156,     47_962,      100),
    "kitti360":  (10,      940_262,     100),
    "a2d2":      (3,       438_131,     200),
    "nuscene":   (979,     1_906_954,   100),
    "4seasons":  (30,      31_243_319,  2000),
    "oxford":    (71,      1_361_928,   10),
    "boreas_rt": (110,     23_262_888,  200),
    "zod":       (29,      661_429,     100),
    "waymo":     (70_541,  13_967_118,  10),
    "nvidia":    (298_326, 60_222_852,  10),
    "hdd":       (137,     38_844_425,  100),
    "ford":      (29,      4_574_945,   157),
    "durlar":    (4,       1_074_550,   100),
    "kitscenes": (1_006,   181_501,     10),
}
EXPECTED_COLS = {  # only where Section 6 states a count
    "kitti": 30, "kitti360": 30, "nuscene": 22,
    "4seasons": 14, "durlar": 14, "kitscenes": 8, "nvidia": 7,
}
# ----------------------------------------------------------------------

def footer_rows(path):
    try:
        return pq.ParquetFile(path).metadata.num_rows
    except Exception:
        return 0

def pick_ts_col(names):
    for c in TS_CANDIDATES:
        if c in names:
            return c
    for c in names:
        lc = c.lower()
        if "time" in lc or lc.startswith("ts"):
            return c
    return None

def infer_hz(ts):
    ts = np.sort(np.asarray(ts, dtype="float64"))
    ts = ts[np.isfinite(ts)]
    if ts.size < 3:
        return None
    dt = np.median(np.diff(ts))
    if dt <= 0:
        return None
    for scale in (1.0, 1e3, 1e6, 1e9):     # s, ms, us, ns
        hz = scale / dt
        if 1.0 <= hz <= 3000.0:
            return hz
    return None

def analyze(ds_dir):
    files = sorted(ds_dir.rglob("*.parquet"))
    if not files:
        return None
    total = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        for r in tqdm(ex.map(footer_rows, files), total=len(files),
                      desc=f"{ds_dir.name:>10} rows", leave=False):
            total += r
    names = list(pq.ParquetFile(files[0]).schema.names)
    ts_col = pick_ts_col(names)
    hz = None
    if ts_col:
        hz_vals, taken = [], 0
        for f in files:
            if taken >= HZ_SAMPLE_FILES:
                break
            try:
                t = pq.read_table(f, columns=[ts_col]).column(0).to_numpy()
            except Exception:
                continue
            if t.size > 2:
                h = infer_hz(t)
                if h:
                    hz_vals.append(h)
                taken += 1
        if hz_vals:
            hz = float(np.median(hz_vals))
    return dict(name=ds_dir.name, clips=len(files), rows=total,
                ncols=len(names), names=names, ts_col=ts_col, hz=hz)

def main():
    if not RAW_ROOT.exists():
        print(f"!! {RAW_ROOT} does not exist — fix RAW_ROOT at top of script.")
        return
    folders = sorted([d for d in RAW_ROOT.iterdir() if d.is_dir()])
    print(f"Discovered {len(folders)} folders under {RAW_ROOT}:")
    print("  " + ", ".join(d.name for d in folders) + "\n")

    results = []
    for d in folders:
        res = analyze(d)
        if res is None:
            print(f"  [skip] {d.name}: no parquet files")
            continue
        results.append(res)

    g_clips = sum(r["clips"] for r in results)
    g_rows  = sum(r["rows"]  for r in results)

    print("\n===================== CLIPS & ROWS vs SECTION 6 =====================")
    hdr = f"{'folder':<12}{'clips':>9}{'exp':>9}{'':>3}{'rows':>15}{'exp':>15}{'':>3}"
    print(hdr); print("-" * len(hdr))
    for r in results:
        exp = EXPECTED.get(r["name"])
        if exp:
            ec, er, _ = exp
            cf = "OK" if r["clips"] == ec else "XX"
            rf = "OK" if r["rows"]  == er else "XX"
            print(f"{r['name']:<12}{r['clips']:>9,}{ec:>9,} {cf:>2}"
                  f"{r['rows']:>15,}{er:>15,} {rf:>2}")
        else:
            print(f"{r['name']:<12}{r['clips']:>9,}{'?':>9}   "
                  f"{r['rows']:>15,}{'?':>15}    << no Section-6 entry")
    print("-" * len(hdr))
    print(f"{'TOTAL':<12}{g_clips:>9,}{371431:>9,} "
          f"{'OK' if g_clips==371431 else 'XX':>2}"
          f"{g_rows:>15,}{178728264:>15,} "
          f"{'OK' if g_rows==178728264 else 'XX':>2}")

    print("\n===================== Hz & COLUMNS (review) =====================")
    print(f"{'folder':<12}{'Hz_est':>9}{'Hz_exp':>9}{'ncols':>7}{'exp':>5}  ts_col")
    print("-" * 60)
    for r in results:
        exp = EXPECTED.get(r["name"])
        hz_exp = exp[2] if exp else "?"
        ec = EXPECTED_COLS.get(r["name"], "?")
        hz_s = f"{r['hz']:.1f}" if r["hz"] else "n/a"
        print(f"{r['name']:<12}{hz_s:>9}{str(hz_exp):>9}"
              f"{r['ncols']:>7}{str(ec):>5}  {r['ts_col']}")

    print("\n===================== COLUMN NAMES =====================")
    for r in results:
        print(f"\n{r['name']} ({r['ncols']} cols):\n  {r['names']}")

if __name__ == "__main__":
    main()