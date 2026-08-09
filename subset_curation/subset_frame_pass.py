#!/usr/bin/env python3
"""subset_frame_pass.py — the chapter's SECOND disclosed frame-layer excursion.

One streaming pass over inertial_data/<dataset>/clip_*.parquet that tags every
frame with its non-overlapping 20 s window (win_start = floor(t_rel/20)*20,
key = "ds|clip|round(start*10)" — identical to the harness 'start' key) and,
per membership group (Frame + M01..M11 budget subsets, a 12-bit mask),
accumulates in a single read:

  1. fine per-signal histograms + moment sums  -> per-method frame histograms
     (the Chapter 8 distributional grammar, Section 8.1)
  2. the two phase-portrait 2D histograms      -> Section 8.5 grammar
  3. raw sequences for a seeded FAN_N-per-method window sample (+ sigma_speed
     from std_data and exemplar flags)          -> fan / time-series figures

After this pass the frame layer is never touched again (Section 9.6
compute-once; the first excursion was the Section 11.4 summary pass).
Re-run when curation subsets (C0x) exist — the mask simply grows.

Outputs (dataset/subset_data/eval/):
  frame_hist_fine.parquet, frame_stats.parquet, frame_portraits.npz,
  fan_frames.parquet, fan_index.parquet, frame_pass_meta.json

Runtime honesty: reads every clip file once (~371k files; NVIDIA's ~300k
small files dominate wall time). Expect 10-40 min. One-time.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("THESIS_ROOT", "/Users/edward/Downloads"))
INERTIAL_ROOT = PROJECT_ROOT / "dataset" / "inertial_data"
STD_DATA_DIR = PROJECT_ROOT / "dataset" / "std_data"
REGIME_DIR = PROJECT_ROOT / "dataset" / "regime_data"
SUBSET_DIR = PROJECT_ROOT / "dataset" / "subset_data"
BUDGET_N = 71_390                       # Section 11.3; single point of change
BUDGET_DIR = SUBSET_DIR / f"tier_n{BUDGET_N}"
EVAL_DIR = SUBSET_DIR / "eval"

WINDOW_S = 20.0
FAN_N = 5000            # windows sampled per group (Chapter 8 convention)
EX_N = 3                # exemplar windows per group (time-series figure)
FAN_SEED = 777
FINE_BINS = 1000        # fine accumulation grid per signal (rebinned at render)
N_WORKERS = 12
TASK_CLIPS = 200        # clip files per worker task

METHODS = ["M01", "M02", "M03", "M04", "M05",
           "M06", "M07", "M08", "M09", "M10", "M11"]
GROUPS = ["R0_frame"] + METHODS          # bit 0 = frame, bits 1..11 = methods
N_GROUPS = len(GROUPS)

# (signal_key, parquet_col, lo, hi, rad->deg) — verbatim distributional grammar
EXTRACT_SPEC = [
    ("speed",     "speed_kmh",  0.0,   200.0, False),
    ("lon_accel", "a_long",   -15.0,    15.0, False),
    ("lat_accel", "a_lat",    -15.0,    15.0, False),
    ("lon_jerk",  "jerk_long", -30.0,   30.0, False),
    ("lat_jerk",  "jerk_lat",  -30.0,   30.0, False),
    ("yaw_rate",  "yaw_rate",  -90.0,   90.0, True),
]
SIG_KEYS = [s[0] for s in EXTRACT_SPEC]
N_SIG = len(EXTRACT_SPEC)

# phase-portrait constants — verbatim phase_potrait.py
SPEED_MAX_MS, A_LONG_ABS, YAW_ABS_DEG = 40.0, 8.0, 60.0
N_BINS_SPEED = N_BINS_Y = 80
SPEED_EDGES = np.linspace(0.0, SPEED_MAX_MS, N_BINS_SPEED + 1)
ALONG_EDGES = np.linspace(-A_LONG_ABS, A_LONG_ABS, N_BINS_Y + 1)
YAW_EDGES = np.linspace(-YAW_ABS_DEG, YAW_ABS_DEG, N_BINS_Y + 1)

FRAME_COLS = ["t_rel_s", "speed_kmh", "a_long", "a_lat",
              "jerk_long", "jerk_lat", "yaw_rate", "acc_mag"]

EXPECTED_FRAME_WINDOWS = 402_421
EXPECTED_METHOD_WINDOWS = BUDGET_N


def log(m):
    print(m, flush=True)


# ---------------------------------------------------------------------------
# membership masks (start-keyed, identical construction to the harness)
# ---------------------------------------------------------------------------

def _key_series(df, ds_col, clip_col, start_col):
    st = np.round(df[start_col].astype(float) * 10).astype("int64").astype(str)
    return df[ds_col].astype(str) + "|" + df[clip_col].astype(str) + "|" + st


def _resolve(df, cands):
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    raise SystemExit(f"[frame_pass] cannot resolve any of {cands} "
                     f"in {list(df.columns)}")


def _std_bridge():
    """Map every window to its start-key via std_data (carries both keys),
    so the frame pass works whether assignments are start- or idx-keyed."""
    idx_to_start = {}
    start_set = set()
    for fp in sorted(STD_DATA_DIR.glob("*.parquet")):
        d = pd.read_parquet(fp, columns=["clip_id", "window_idx",
                                         "window_start_s"])
        ds = fp.stem
        st10 = np.round(d["window_start_s"].astype(float) * 10).astype("int64")
        skey = ds + "|" + d["clip_id"].astype(str) + "|" + st10.astype(str)
        ikey = ds + "|" + d["clip_id"].astype(str) + "|" + \
            d["window_idx"].astype("int64").astype(str)
        idx_to_start.update(dict(zip(ikey.values, skey.values)))
        start_set.update(skey.values)
    return idx_to_start, start_set


def _keys_as_start(df, ds_col, clip_col, bridge):
    """Return start-keys for a table keyed by either window_start_s or
    window_idx, translating idx->start through the std_data bridge."""
    low = {c.lower(): c for c in df.columns}
    ds = df[ds_col].astype(str)
    clip = df[clip_col].astype(str)
    idx_to_start, _ = bridge
    for cand in ("window_start_s", "win_start", "start_s"):
        if cand in low:
            st10 = np.round(df[low[cand]].astype(float) * 10).astype("int64")
            return (ds + "|" + clip + "|" + st10.astype(str)).values
    for cand in ("window_idx", "win_idx", "window_index", "widx"):
        if cand in low:
            ikey = ds + "|" + clip + "|" + \
                df[low[cand]].astype("int64").astype(str)
            return np.array([idx_to_start.get(k, k) for k in ikey.values])
    raise SystemExit(f"[frame_pass] no window key in {list(df.columns)}")


def build_masks():
    log("[1/4] Building the window -> group mask")
    bridge = _std_bridge()
    a = pd.read_parquet(REGIME_DIR / "cluster_assignments.parquet")
    ds_c = _resolve(a, ["dataset"])
    cl_c = _resolve(a, ["clip_id", "clip"])
    keys = _keys_as_start(a, ds_c, cl_c, bridge)
    mask = pd.Series(np.uint16(1), index=keys)                 # bit 0: frame
    log(f"  frame windows: {len(mask):,} (expected {EXPECTED_FRAME_WINDOWS:,})")

    per_method_keys = {}
    for bit, mid in enumerate(METHODS, start=1):
        fps = sorted(BUDGET_DIR.glob(f"{mid}_*.parquet"))
        if not fps:
            raise SystemExit(f"[frame_pass] missing budget subset for {mid}")
        s = pd.read_parquet(fps[0])
        k = _keys_as_start(s, _resolve(s, ["dataset"]),
                           _resolve(s, ["clip_id", "clip"]), bridge)
        per_method_keys[mid] = k
        missing = ~pd.Index(k).isin(mask.index)
        if missing.any():
            raise SystemExit(f"[frame_pass] {mid}: {int(missing.sum())} windows "
                             f"not in the frame")
        mask.loc[k] = mask.loc[k].values | np.uint16(1 << bit)
        log(f"  {mid}: {len(k):,} windows tagged")

    # sigma(speed) per window for the time-series band selection (start-keyed)
    sig = {}
    for fp in sorted(STD_DATA_DIR.glob("*.parquet")):
        d = pd.read_parquet(fp, columns=["clip_id", "window_start_s",
                                         "std_speed_kmh"])
        st = np.round(d["window_start_s"].astype(float) * 10).astype("int64")
        kk = fp.stem + "|" + d["clip_id"].astype(str) + "|" + st.astype(str)
        sig.update(dict(zip(kk.values, d["std_speed_kmh"].values)))

    return mask, per_method_keys, sig


def sample_fans(mask, per_method_keys, sigma):
    log("[2/4] Seeded fan / exemplar sampling")
    frame_keys = mask.index.values
    sample_mask = {}
    exemplar_mask = {}
    for bit, gid in enumerate(GROUPS):
        pool = frame_keys if gid == "R0_frame" else per_method_keys[gid]
        rng = np.random.default_rng(FAN_SEED + bit)
        take = pool if len(pool) <= FAN_N else rng.choice(pool, FAN_N,
                                                          replace=False)
        for j, k in enumerate(take):
            sample_mask[k] = sample_mask.get(k, 0) | (1 << bit)
            if j < EX_N:
                exemplar_mask[k] = exemplar_mask.get(k, 0) | (1 << bit)
        log(f"  {gid}: {len(take):,} fan windows")
    rows = []
    for kid, k in enumerate(sorted(sample_mask)):
        ds, clip, st10 = k.split("|")
        rows.append((kid, k, ds, clip, int(st10) / 10.0,
                     sample_mask[k], exemplar_mask.get(k, 0),
                     float(sigma.get(k, np.nan))))
    idx = pd.DataFrame(rows, columns=["key_id", "key", "dataset", "clip",
                                      "win_start", "sample_mask",
                                      "exemplar_mask", "std_speed_kmh"])
    log(f"  fan union: {len(idx):,} distinct windows")
    return idx


# ---------------------------------------------------------------------------
# streaming accumulation
# ---------------------------------------------------------------------------

def make_worker(ds, mask_dict, fan_kid):
    lo = np.array([s[2] for s in EXTRACT_SPEC])
    hi = np.array([s[3] for s in EXTRACT_SPEC])

    def worker(files):
        H1 = np.zeros((N_GROUPS, N_SIG, FINE_BINS), dtype=np.int64)
        Ha = np.zeros((N_GROUPS, N_BINS_SPEED, N_BINS_Y))
        Hy = np.zeros((N_GROUPS, N_BINS_SPEED, N_BINS_Y))
        st = np.zeros((N_GROUPS, N_SIG, 3))          # n, sum, sumsq
        n_tot = np.zeros(N_GROUPS, dtype=np.int64)
        fan_chunks = []
        for fp in files:
            try:
                df = pd.read_parquet(fp)
            except Exception:
                continue
            cols = set(df.columns)
            if "t_rel_s" not in cols or "speed_kmh" not in cols:
                continue
            if "clip_id" in cols:
                clip = str(df["clip_id"].iloc[0])
            else:
                m = re.search(r"clip_(\d+)", fp.name)
                clip = str(int(m.group(1))) if m else fp.stem
            t = df["t_rel_s"].to_numpy(float)
            ws = (np.floor(t / WINDOW_S) * WINDOW_S)
            st10 = np.round(ws * 10).astype("int64")
            keys = (ds + "|" + clip + "|"
                    + pd.Series(st10, dtype="int64").astype(str)).to_numpy()
            gm = np.fromiter((mask_dict.get(k, 0) for k in keys),
                             dtype=np.uint16, count=len(keys))
            inw = gm > 0
            if not inw.any():
                continue
            for b in range(N_GROUPS):
                n_tot[b] += int(((gm >> b) & 1).sum())
            # 1D fine hists + moments
            for si, (skey, col, l, h, deg) in enumerate(EXTRACT_SPEC):
                if col not in cols:
                    continue
                v = df[col].to_numpy(float)
                if deg:
                    v = v * (180.0 / math.pi)
                ok = inw & np.isfinite(v) & (v >= l) & (v <= h)
                if not ok.any():
                    continue
                vv, mm = v[ok], gm[ok]
                fi = np.minimum(((vv - l) / (h - l) * FINE_BINS)
                                .astype(np.int64), FINE_BINS - 1)
                for b in range(N_GROUPS):
                    sel = ((mm >> b) & 1).astype(bool)
                    if sel.any():
                        H1[b, si] += np.bincount(fi[sel],
                                                 minlength=FINE_BINS)
                        w = vv[sel]
                        st[b, si] += (len(w), w.sum(), (w * w).sum())
            # phase portraits (speed m/s x a_long, x yaw deg)
            sp = df["speed_kmh"].to_numpy(float) / 3.6
            al = (df["a_long"].to_numpy(float)
                  if "a_long" in cols else np.full(len(df), np.nan))
            yw = df["yaw_rate"].to_numpy(float) * (180.0 / math.pi)
            for b in range(N_GROUPS):
                sel = ((gm >> b) & 1).astype(bool)
                if not sel.any():
                    continue
                ma = sel & np.isfinite(sp) & np.isfinite(al)
                my = sel & np.isfinite(sp) & np.isfinite(yw)
                if ma.any():
                    Ha[b] += np.histogram2d(sp[ma], al[ma],
                                            bins=[SPEED_EDGES,
                                                  ALONG_EDGES])[0]
                if my.any():
                    Hy[b] += np.histogram2d(sp[my], yw[my],
                                            bins=[SPEED_EDGES, YAW_EDGES])[0]
            # fan sequences
            kid = np.fromiter((fan_kid.get(k, -1) for k in keys),
                              dtype=np.int64, count=len(keys))
            fs = kid >= 0
            if fs.any():
                am = (df["acc_mag"].to_numpy(float)
                      if "acc_mag" in cols else np.full(len(df), np.nan))
                fan_chunks.append(pd.DataFrame({
                    "key_id": kid[fs].astype(np.int32),
                    "t": (t[fs] - ws[fs]).astype(np.float32),
                    "speed_kmh": df["speed_kmh"].to_numpy(float)[fs]
                                    .astype(np.float32),
                    "yaw_rate": df["yaw_rate"].to_numpy(float)[fs]
                                    .astype(np.float32),
                    "acc_mag": am[fs].astype(np.float32),
                }))
        return H1, Ha, Hy, st, n_tot, fan_chunks
    return worker


def main():
    t0 = time.time()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    mask, per_method_keys, sigma = build_masks()
    fan_idx = sample_fans(mask, per_method_keys, sigma)
    mask_dict = dict(zip(mask.index.values, mask.values))
    fan_kid = dict(zip(fan_idx["key"].values, fan_idx["key_id"].values))

    H1 = np.zeros((N_GROUPS, N_SIG, FINE_BINS), dtype=np.int64)
    Ha = np.zeros((N_GROUPS, N_BINS_SPEED, N_BINS_Y))
    Hy = np.zeros((N_GROUPS, N_BINS_SPEED, N_BINS_Y))
    ST = np.zeros((N_GROUPS, N_SIG, 3))
    NT = np.zeros(N_GROUPS, dtype=np.int64)
    fan_all = []
    seen = set()

    log("[3/4] Streaming the frame layer (one pass)")
    datasets = [d for d in sorted(INERTIAL_ROOT.iterdir())
                if d.is_dir() and any(d.glob("clip_*.parquet"))]
    for d in datasets:
        clips = sorted(d.glob("clip_*.parquet"))
        worker = make_worker(d.name, mask_dict, fan_kid)
        tasks = [clips[i:i + TASK_CLIPS]
                 for i in range(0, len(clips), TASK_CLIPS)]
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            for h1, ha, hy, st, nt, fc in tqdm(
                    ex.map(worker, tasks), total=len(tasks),
                    desc=f"  {d.name:<10}"):
                H1 += h1; Ha += ha; Hy += hy; ST += st; NT += nt
                fan_all.extend(fc)

    log("[4/4] Writing the frame caches")
    b, s, i = np.nonzero(H1)
    pd.DataFrame({"method": np.array(GROUPS)[b], "signal":
                  np.array(SIG_KEYS)[s], "bin_idx": i,
                  "count": H1[b, s, i]}
                 ).to_parquet(EVAL_DIR / "frame_hist_fine.parquet",
                              index=False)
    rows = []
    for bi, g in enumerate(GROUPS):
        for si, k in enumerate(SIG_KEYS):
            n, sm, sq = ST[bi, si]
            rows.append((g, k, int(n), int(NT[bi]), sm, sq))
    pd.DataFrame(rows, columns=["method", "signal", "n_valid",
                                "n_total_frames", "sum", "sumsq"]
                 ).to_parquet(EVAL_DIR / "frame_stats.parquet", index=False)
    np.savez_compressed(EVAL_DIR / "frame_portraits.npz", Ha=Ha, Hy=Hy,
                        speed_edges=SPEED_EDGES, along_edges=ALONG_EDGES,
                        yaw_edges=YAW_EDGES, methods=np.array(GROUPS))
    fan = (pd.concat(fan_all, ignore_index=True)
           if fan_all else pd.DataFrame(columns=["key_id", "t", "speed_kmh",
                                                 "yaw_rate", "acc_mag"]))
    fan.sort_values(["key_id", "t"]).to_parquet(
        EVAL_DIR / "fan_frames.parquet", index=False)
    fan_idx.to_parquet(EVAL_DIR / "fan_index.parquet", index=False)

    got_frame = int(fan_idx.shape[0] and 0)  # placeholder keeps lint quiet
    kids_seen = set(fan["key_id"].unique().tolist())
    cover = len(kids_seen) / max(1, len(fan_idx))
    meta = {"groups": GROUPS, "fan_n": FAN_N, "ex_n": EX_N,
            "fan_seed": FAN_SEED, "fine_bins": FINE_BINS,
            "frames_tagged_frame_group": int(NT[0]),
            "fan_window_coverage": cover,
            "n_total_frames_per_group":
                {g: int(NT[i]) for i, g in enumerate(GROUPS)},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(EVAL_DIR / "frame_pass_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log(f"  frames tagged to the frame group: {NT[0]:,}")
    log(f"  fan sequence coverage: {cover:.1%} of sampled windows")
    if cover < 0.99:
        log("  [WARN] some sampled windows produced no frames — inspect "
            "fan_index vs fan_frames")
    log(f"Done in {(time.time() - t0)/60:.1f} min. Frame layer will not be "
        f"read again (compute-once).")


if __name__ == "__main__":
    main()