#!/usr/bin/env python3
"""
run_sampling.py — sampling method construction (thesis 4.9).

Generates eleven curated window-ID subsets (M01–M11) from the frozen window layer
(cluster_assignments + std_data [+ window_summaries cache]) under one shared
protocol: frame overlap guard (auto-verified), CONFOUNDED_ACCEL
guard, matched-N sizing, fixed seeds, largest-remainder integer allocation.

Outputs (all under SUBSET_DIR):
    M01_srs.parquet ... M11_neyman_allocation.parquet    (window-ID lists)
    supply_table.csv        regime x dataset supply, post-dedup
    allocation_table.csv    per-regime quotas for the allocation designs
                            (M02-M05, M11)
    manifest.json           full run metadata, config echo, per-method summaries

Usage:
    python run_sampling.py --inspect          # print schemas / resolved columns, exit
    python run_sampling.py                    # run everything possible
    python run_sampling.py --methods M01,M03  # run a subset of methods
    python run_sampling.py --n 5000           # override matched-N

Methods M07/M08 need the window_summaries.parquet cache
(built once by window_summary.py); they are skipped with a warning
if the cache is missing. Definitions and citations are in thesis Section 4.9.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # degrade to no-op per pipeline convention
    def tqdm(x, **kwargs):
        return x

# =========================================================================
# Canonical constants
# =========================================================================
sys.path.append(str(Path(__file__).resolve().parents[1]))
from thesis_paths import DATA_ROOT, REPO_ROOT  # noqa: E402
REGIME_DATA_DIR = DATA_ROOT / "regime_data"
STD_DATA_DIR = DATA_ROOT / "std_data"
SUBSET_DIR = DATA_ROOT / "subset_data"

# Primary input: the frozen window layer. If the file name differs, either fix
# this constant or let the fallback glob find it (it picks the parquet in
# REGIME_DATA_DIR that contains a regime-like column).
CLUSTER_ASSIGNMENTS_PATH = REGIME_DATA_DIR / "cluster_assignments.parquet"
SUMMARIES_PATH = SUBSET_DIR / "window_summaries.parquet"
# Outputs land here; switched to SUBSET_DIR/tier_n<N>/ when --n is used so a
# budget tier never overwrites the matched-N artifacts.
ACTIVE_OUT_DIR = SUBSET_DIR

GLOBAL_SEED = 42

# ---- Protocol decisions: DECISION CONSTANTS --------------------------------
# The assignment grid is ALREADY non-overlapping (thesis 4.8).
# "auto" verifies this at run time from the median
# within-clip start spacing and applies stride-2 thinning ONLY if an
# overlapping (10 s step) grid is detected. "on" forces thinning;
# "off" takes the frame as given.
DEDUP_MODE = "auto"         # "auto" | "on" | "off"
DEDUP_STRIDE = 2
DEDUP_PHASE = 0             # keep windows with (idx - clip_min_idx) % 2 == 0
WINDOW_S = 20.0
# Fidelity guard set (thesis 4.9.2): imported from the bias-quantification
# compute stage (thesis 4.8) so the two stages cannot diverge. Local fallback engages only if
# the import path is unavailable; the resolved source is logged and echoed
# in the manifest.
try:
    sys.path.insert(0, str(REPO_ROOT / "06_bias_quantification"))
    from bias_quant import CONFOUNDED_ACCEL
    CONFOUNDED_SRC = "bias_quant (imported)"
except Exception:
    CONFOUNDED_ACCEL = {"a2d2", "4seasons", "durlar"}
    CONFOUNDED_SRC = "local fallback — keep in sync with bias_quant.py" 
MATCHED_N_MULT = 5          # matched-N = MULT x rarest-regime supply (post-dedup)
OVERRIDE_N = None           # int to force a common N; None = matched-N rule

# ---- Method decisions — DECISION CONSTANTS ---------------------------------
BANKIER_Q = 0.5             # M04 power allocation exponent (primary value)
EVENT_SHARE = 0.5           # M07 target share of trigger-event windows
DECEL_TRIGGER_G = 0.6       # M07/M08 hard-decel threshold in g
                            #   0.6 g: 100-Car lineage (Dingus et al. 2006)
                            #   0.75 g: Perez et al. 2017
HIGH_YAW_DEG_S = 30.0       # M08 predicate: max |yaw rate| threshold (deg/s)
CRUISE_SPEED_KMH = 100.0    # M08 predicate: sustained high speed
STOPGO_STD_MIN = 5.0        # M08 predicate: std_speed >= t2 tier threshold
STOPGO_MEAN_MAX = 20.0      # M08 predicate: mean speed cap for stop-and-go
STATIONARY_MEAN_MAX = 2.0   # M08 predicate: near-stationary mean speed
STATIONARY_STD_MAX = 1.0    # M08 predicate: near-stationary std (t1 tier)
SCORE_P99_ON_GUARDED = 99.0 # M08 predicate: dynamic-score percentile
SUBCLUSTER_K = 10           # M10 sub-clusters per regime (selection layer ONLY;
                            # the frozen k=5 regime labels are never re-fitted)
MIN_SUMMARY_COVERAGE = 0.95 # M07/M08 refuse to run if any dataset's
                            # window_summaries coverage falls below this
                            # (prevents silently NaN-excluding datasets)
G_MS2 = 9.81
RAD2DEG = 180.0 / np.pi

# ---- Schema resolution candidates (first match wins, loudly printed) -------
REGIME_COL_CANDIDATES = ["regime", "regime_label", "cluster", "cluster_label",
                         "kmeans_label", "label", "regime_id", "cluster_id"]
DATASET_COL_CANDIDATES = ["dataset", "dataset_name", "source", "ds"]
CLIP_COL_CANDIDATES = ["clip_id", "clip", "clip_name"]
WIDX_COL_CANDIDATES = ["window_idx", "win_idx", "window_index", "widx"]
WSTART_COL_CANDIDATES = ["window_start_s", "start_s", "win_start_s"]
WEND_COL_CANDIDATES = ["window_end_s", "end_s", "win_end_s"]
SCORE_COL_CANDIDATES = ["dynamic_score", "dyn_score", "composite_score",
                        "score_dynamic", "composite_dynamic_score"]
STD_FEATURE_COLS = ["std_speed_kmh", "std_acc_mag", "std_yaw_rate"]

STEP_S = 20.0               # window step, used to derive window_idx if absent

METHOD_ORDER = ["M01", "M02", "M03", "M04", "M05",
                "M06", "M07", "M08", "M09", "M10", "M11"]
METHOD_FILES = {
    "M01": "M01_srs.parquet",
    "M02": "M02_stratified_proportional.parquet",
    "M03": "M03_stratified_equal.parquet",
    "M04": "M04_power_allocation.parquet",
    "M05": "M05_two_way_regime_source.parquet",
    "M06": "M06_pps_dynamic_score.parquet",
    "M07": "M07_kinematic_trigger.parquet",
    "M08": "M08_predicate_mining.parquet",
    "M09": "M09_kennard_stone.parquet",
    "M10": "M10_cluster_quota_fill.parquet",
    "M11": "M11_neyman_allocation.parquet",
}
ID_COLS = ["dataset", "clip_id", "window_idx",
           "window_start_s", "window_end_s", "regime"]


# =========================================================================
# Small utilities
# =========================================================================
def log(msg: str) -> None:
    print(f"[run_sampling] {msg}", flush=True)


def rng_for(method_num: int) -> np.random.Generator:
    return np.random.default_rng([GLOBAL_SEED, method_num])


def sk_seed(method_num: int) -> int:
    return GLOBAL_SEED * 100 + method_num


def resolve_col(cols, candidates, what, required=True):
    for c in candidates:
        if c in cols:
            return c
    if required:
        raise SystemExit(
            f"[run_sampling] FATAL: could not find a '{what}' column. "
            f"Tried {candidates}. Available: {sorted(cols)}. "
            f"Add the right name to the candidate list at the top of the script "
            f"and re-run."
        )
    return None


def json_default(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


# =========================================================================
# Loading & protocol
# =========================================================================
def find_assignments_path() -> Path:
    if CLUSTER_ASSIGNMENTS_PATH.exists():
        return CLUSTER_ASSIGNMENTS_PATH
    log(f"WARNING: {CLUSTER_ASSIGNMENTS_PATH} not found; scanning "
        f"{REGIME_DATA_DIR} for a parquet with a regime-like column ...")
    import pyarrow.parquet as pq
    candidates = sorted(REGIME_DATA_DIR.glob("*.parquet"))
    for p in candidates:
        try:
            names = set(pq.ParquetFile(p).schema_arrow.names)
        except Exception:
            continue
        if any(c in names for c in REGIME_COL_CANDIDATES):
            log(f"  -> using {p.name}")
            return p
    raise SystemExit(
        f"[run_sampling] FATAL: no parquet with a regime column found in "
        f"{REGIME_DATA_DIR}. Set CLUSTER_ASSIGNMENTS_PATH manually."
    )


def load_std_data() -> pd.DataFrame:
    files = sorted(STD_DATA_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit(f"[run_sampling] FATAL: no parquet files in {STD_DATA_DIR}")
    parts = []
    for p in tqdm(files, desc="std_data"):
        df = pd.read_parquet(p)
        df["dataset"] = p.stem
        parts.append(df)
    std = pd.concat(parts, ignore_index=True)
    log(f"std_data loaded: {len(std):,} rows from {len(files)} files")
    return std


def load_pool(inspect_only: bool = False):
    """Load the window layer, resolve schema, attach features/score/summaries."""
    path = find_assignments_path()
    asg = pd.read_parquet(path)
    log(f"assignments loaded: {path.name}  shape={asg.shape}")
    log(f"assignments columns: {list(asg.columns)}")

    colmap = {}
    colmap["regime"] = resolve_col(asg.columns, REGIME_COL_CANDIDATES, "regime")
    colmap["dataset"] = resolve_col(asg.columns, DATASET_COL_CANDIDATES, "dataset")
    colmap["clip_id"] = resolve_col(asg.columns, CLIP_COL_CANDIDATES, "clip_id")
    colmap["window_idx"] = resolve_col(asg.columns, WIDX_COL_CANDIDATES,
                                       "window_idx", required=False)
    colmap["window_start_s"] = resolve_col(asg.columns, WSTART_COL_CANDIDATES,
                                           "window_start_s", required=False)
    colmap["window_end_s"] = resolve_col(asg.columns, WEND_COL_CANDIDATES,
                                         "window_end_s", required=False)
    log(f"resolved column mapping: {colmap}")

    pool = asg.rename(columns={v: k for k, v in colmap.items() if v})

    # --- ensure window keys / std features, joining std_data if needed ------
    need_std_join = (
        "window_idx" not in pool.columns
        or "window_start_s" not in pool.columns
        or any(c not in pool.columns for c in STD_FEATURE_COLS)
    )
    if need_std_join:
        std = load_std_data()
        std_keep = ["dataset", "clip_id", "window_idx",
                    "window_start_s", "window_end_s"] + [
            c for c in STD_FEATURE_COLS if c in std.columns]
        std_keep = [c for c in std_keep if c in std.columns]
        std = std[std_keep].copy()
        if "window_idx" in pool.columns:
            keys = ["dataset", "clip_id", "window_idx"]
        elif "window_start_s" in pool.columns:
            keys = ["dataset", "clip_id", "window_start_s"]
        else:
            raise SystemExit(
                "[run_sampling] FATAL: assignments have neither window_idx "
                "nor window_start_s — cannot join std_data. Paste --inspect "
                "output back so the join keys can be fixed."
            )
        before = len(pool)
        overlap = [c for c in std.columns if c in pool.columns and c not in keys]
        pool = pool.merge(std.drop(columns=overlap), on=keys, how="left",
                          validate="one_to_one")
        matched = pool["window_start_s"].notna().sum() \
            if "window_start_s" in pool.columns else before
        log(f"std_data join on {keys}: {matched:,}/{before:,} rows matched")

    if "window_idx" not in pool.columns:
        pool["window_idx"] = (pool["window_start_s"] / STEP_S).round().astype(int)
        log("window_idx derived from window_start_s / STEP_S")

    missing_std = [c for c in STD_FEATURE_COLS if c not in pool.columns]
    if missing_std:
        log(f"WARNING: std feature columns missing even after join: {missing_std} "
            f"(feature-based methods will use what is available)")

    if "std_yaw_rate" in pool.columns:
        pool["std_yaw_rate_degs"] = pool["std_yaw_rate"] * RAD2DEG

    # --- dynamic score: reuse if present, else thesis 4.8 formula fallback -
    score_col = resolve_col(pool.columns, SCORE_COL_CANDIDATES, "score",
                            required=False)
    if score_col:
        pool["score"] = pool[score_col].astype(float)
        log(f"dynamic score: using existing column '{score_col}'")
    else:
        avail = [c for c in STD_FEATURE_COLS if c in pool.columns]
        if not avail:
            log("WARNING: no score column and no std features -> score unavailable; "
                "M06 and score-based pieces will be skipped")
            pool["score"] = np.nan
        else:
            s = np.zeros(len(pool))
            for c in avail:
                m = pool[c].mean(skipna=True)
                if m and np.isfinite(m) and m > 0:
                    s = s + pool[c].fillna(0).to_numpy() / m
            pool["score"] = s
            log(f"dynamic score: RECOMPUTED as sum(std_ch / pooled_mean_ch) over "
                f"{avail} (Ch.10 formula; pooled means on full pool). If a "
                f"canonical score column exists under another name, add it to "
                f"SCORE_COL_CANDIDATES instead — never keep two versions.")

    # --- summaries cache (M07/M08) -------------------------------------------
    have_summaries = SUMMARIES_PATH.exists()
    if have_summaries:
        summ = pd.read_parquet(SUMMARIES_PATH)
        keys = ["dataset", "clip_id", "window_idx"]
        overlap = [c for c in summ.columns if c in pool.columns and c not in keys]
        pool = pool.merge(summ.drop(columns=overlap), on=keys, how="left",
                          validate="one_to_one")
        cov = pool["max_decel_ms2"].notna().mean() \
            if "max_decel_ms2" in pool.columns else 0.0
        log(f"window_summaries joined: coverage {cov:.1%}")
    else:
        log(f"NOTE: {SUMMARIES_PATH} not found -> M07/M08 will be skipped. "
            f"Run build_window_summaries.py once to enable them.")

    # deterministic ordering regardless of file order on disk
    pool = pool.sort_values(["dataset", "clip_id", "window_idx"],
                            kind="mergesort").reset_index(drop=True)

    # sanity report
    regimes = sorted(pool["regime"].unique().tolist())
    log(f"pool: {len(pool):,} windows | {pool['dataset'].nunique()} datasets | "
        f"regimes {regimes}")
    if len(regimes) != 5:
        log(f"WARNING: expected 5 regimes, found {len(regimes)} — check labels")

    if inspect_only:
        log("--inspect: dataset counts:")
        print(pool["dataset"].value_counts().to_string())
        log("--inspect: regime counts:")
        print(pool["regime"].value_counts().sort_index().to_string())
        log("--inspect: dtypes:")
        print(pool.dtypes.to_string())
        sys.exit(0)

    return pool, colmap, have_summaries


def apply_dedup(pool: pd.DataFrame) -> pd.DataFrame:
    """Frame verification + overlap contingency (thesis 4.9.1).

    The assignment grid is expected to be non-overlapping already
    (thesis 4.8). This guard
    measures the median within-clip start spacing: ~WINDOW_S means the frame
    is taken as given; ~STEP_S means an overlapping grid was detected and
    stride-2 systematic thinning (Madow & Madow 1944) engages.
    """
    if DEDUP_MODE == "off":
        log("frame guard: OFF — frame taken as given")
        return pool
    diffs = (pool.groupby(["dataset", "clip_id"])["window_start_s"]
             .diff().dropna())
    med = float(diffs.median()) if len(diffs) else float("nan")
    overlapping = np.isfinite(med) and med < 0.75 * WINDOW_S
    log(f"frame guard: median within-clip start spacing = {med:.1f} s -> "
        f"{'OVERLAPPING grid' if overlapping else 'non-overlapping grid'}")
    if DEDUP_MODE == "auto" and not overlapping:
        log("frame guard: no thinning applied (grid already non-overlapping)")
        return pool
    if DEDUP_MODE == "auto":
        log("frame guard: overlapping grid detected -> stride-2 thinning "
            "contingency engaged")
    base = pool.groupby(["dataset", "clip_id"])["window_idx"].transform("min")
    keep = ((pool["window_idx"] - base) % DEDUP_STRIDE) == DEDUP_PHASE
    out = pool[keep].reset_index(drop=True)
    log(f"thinning: stride={DEDUP_STRIDE} phase={DEDUP_PHASE} -> "
        f"{len(out):,}/{len(pool):,} windows kept")
    return out


def compute_supply(pool: pd.DataFrame) -> pd.DataFrame:
    t = pool.pivot_table(index="regime", columns="dataset",
                         values="window_idx", aggfunc="count",
                         fill_value=0)
    t["TOTAL"] = t.sum(axis=1)
    return t


def matched_n(supply: pd.DataFrame) -> int:
    if OVERRIDE_N is not None:
        log(f"matched-N: OVERRIDDEN to {OVERRIDE_N}")
        return int(OVERRIDE_N)
    rarest = int(supply["TOTAL"].min())
    n = MATCHED_N_MULT * rarest
    log(f"matched-N: {MATCHED_N_MULT} x rarest regime supply ({rarest:,}) "
        f"= {n:,}")
    return n


# =========================================================================
# Allocation & selection engines
# =========================================================================
def allocate(weights, n: int, caps) -> np.ndarray:
    """Largest-remainder integer allocation with per-stratum caps.

    Practical stand-in for Wright's exact priority algorithm (Wright 2012,
    2017, 2020). At these stratum sizes the results coincide (thesis 4.9.4).
    """
    w = np.asarray(weights, dtype=float).copy()
    caps = np.asarray(caps, dtype=int)
    w[caps <= 0] = 0.0
    alloc = np.zeros(len(w), dtype=int)
    remaining = int(n)
    for _ in range(10_000):
        if remaining <= 0:
            break
        active = (caps > alloc) & (w > 0)
        if not active.any():
            break
        wa = np.where(active, w, 0.0)
        raw = wa / wa.sum() * remaining
        base = np.minimum(np.floor(raw).astype(int), caps - alloc)
        if base.sum() == 0:
            frac = np.where(active, raw - np.floor(raw), -1.0)
            order = np.argsort(-frac, kind="mergesort")
            for i in order:
                if remaining <= 0:
                    break
                if active[i] and alloc[i] < caps[i]:
                    alloc[i] += 1
                    remaining -= 1
        else:
            alloc += base
            remaining -= int(base.sum())
    return alloc


def equal_waterfill(caps: pd.Series, quota: int,
                    rng: np.random.Generator) -> pd.Series:
    """Equalize an integer quota across strata subject to supply caps."""
    ds = caps.sort_values(kind="mergesort").index.tolist()
    alloc = {d: 0 for d in ds}
    remaining = int(quota)
    m = len(ds)
    for i, d in enumerate(ds):
        share = remaining // (m - i)
        take = min(int(caps[d]), share)
        alloc[d] = take
        remaining -= take
    while remaining > 0:
        spare = [d for d in ds if alloc[d] < caps[d]]
        if not spare:
            break
        d = spare[int(rng.integers(len(spare)))]
        alloc[d] += 1
        remaining -= 1
    return pd.Series(alloc).reindex(caps.index)


def srs_within(pool: pd.DataFrame, group_col, quotas: dict,
               rng: np.random.Generator) -> pd.DataFrame:
    parts = []
    for g, q in quotas.items():
        if q <= 0:
            continue
        idx = pool.index[pool[group_col] == g].to_numpy()
        take = min(int(q), len(idx))
        parts.append(pool.loc[rng.choice(idx, size=take, replace=False)])
    return pd.concat(parts) if parts else pool.iloc[0:0]


def systematic_pps(weights: np.ndarray, n: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Randomized systematic PPS without replacement (Hartley & Rao 1962),
    with certainty extraction; operational engine for the fixed-size
    unequal-probability (conditional-Poisson-ideal) design."""
    w = np.asarray(weights, dtype=float)
    if (w < 0).any():
        raise ValueError("negative weights")
    idx_all = np.arange(len(w))
    positive = idx_all[w > 0]
    if len(positive) < n:
        raise ValueError(f"only {len(positive)} positive-weight units for n={n}")
    certain: list[int] = []
    remaining = positive
    while True:
        n_rem = n - len(certain)
        if n_rem <= 0:
            return np.array(sorted(certain[:n]))
        wr = w[remaining]
        pi = wr * n_rem / wr.sum()
        mask = pi >= 1.0
        if not mask.any():
            break
        certain.extend(remaining[mask].tolist())
        remaining = remaining[~mask]
    n_rem = n - len(certain)
    perm = rng.permutation(len(remaining))
    order = remaining[perm]
    c = np.cumsum(w[order])
    step = c[-1] / n_rem
    start = rng.uniform(0, step)
    points = start + step * np.arange(n_rem)
    pos = np.clip(np.searchsorted(c, points, side="left"), 0, len(c) - 1)
    sel = np.concatenate([np.array(certain, dtype=int), order[pos]])
    uniq = np.unique(sel)
    if len(uniq) != n:  # cannot happen with w_i < step; defensive top-up
        extra = rng.choice(np.setdiff1d(positive, uniq),
                           size=n - len(uniq), replace=False)
        uniq = np.concatenate([uniq, extra])
    return np.sort(uniq)


def kennard_stone(X: np.ndarray, k: int) -> np.ndarray:
    """Incremental max-min (Kennard & Stone 1969); start = farthest from mean."""
    n = X.shape[0]
    if k >= n:
        return np.arange(n)
    X = np.ascontiguousarray(X, dtype=np.float32)
    mu = X.mean(axis=0)
    d0 = ((X - mu) ** 2).sum(axis=1)
    first = int(np.argmax(d0))
    sel = np.empty(k, dtype=int)
    sel[0] = first
    mind = ((X - X[first]) ** 2).sum(axis=1)
    mind[first] = -1.0
    for j in tqdm(range(1, k), desc="KS", leave=False):
        nxt = int(np.argmax(mind))
        sel[j] = nxt
        dn = ((X - X[nxt]) ** 2).sum(axis=1)
        np.minimum(mind, dn, out=mind)
        mind[nxt] = -1.0
    return sel


def feature_matrix(pool: pd.DataFrame):
    """Z-scored feature matrix on the deduped pool (one scaler for M09/M10)."""
    cols = [c for c in ["std_speed_kmh", "std_acc_mag",
                        "std_yaw_rate_degs", "score"] if c in pool.columns]
    cols = [c for c in cols if pool[c].notna().any()]
    if not cols:
        return None, []
    M = pool[cols].fillna(0.0).to_numpy(dtype=np.float64)
    mu = M.mean(axis=0)
    sd = M.std(axis=0)
    sd[sd == 0] = 1.0
    return (M - mu) / sd, cols


# =========================================================================
# Methods M01–M10  (each: pool -> selected DataFrame with role columns)
# =========================================================================
def m01_srs(pool, N, rng):
    idx = rng.choice(pool.index.to_numpy(), size=min(N, len(pool)),
                     replace=False)
    return pool.loc[idx]


def m02_proportional(pool, N, rng, supply):
    regimes = supply.index.tolist()
    q = allocate(supply["TOTAL"].to_numpy(), N, supply["TOTAL"].to_numpy())
    return srs_within(pool, "regime", dict(zip(regimes, q)), rng), \
        dict(zip(regimes, q.tolist()))


def m03_equal(pool, N, rng, supply):
    regimes = supply.index.tolist()
    q = allocate(np.ones(len(regimes)), N, supply["TOTAL"].to_numpy())
    return srs_within(pool, "regime", dict(zip(regimes, q)), rng), \
        dict(zip(regimes, q.tolist()))


def m04_power(pool, N, rng, supply):
    regimes = supply.index.tolist()
    w = supply["TOTAL"].to_numpy(dtype=float) ** BANKIER_Q
    q = allocate(w, N, supply["TOTAL"].to_numpy())
    return srs_within(pool, "regime", dict(zip(regimes, q)), rng), \
        dict(zip(regimes, q.tolist()))


def m05_two_way(pool, N, rng, supply):
    regimes = supply.index.tolist()
    q_regime = allocate(np.ones(len(regimes)), N, supply["TOTAL"].to_numpy())
    parts, cell_alloc = [], {}
    for r, q in zip(regimes, q_regime):
        sub = pool[pool["regime"] == r]
        caps = sub["dataset"].value_counts()
        alloc = equal_waterfill(caps, int(q), rng)
        cell_alloc[str(r)] = alloc.to_dict()
        parts.append(srs_within(sub, "dataset", alloc.to_dict(), rng))
    return pd.concat(parts), cell_alloc


def m06_pps(pool, N, rng):
    guarded = pool[~pool["dataset"].isin(CONFOUNDED_ACCEL)]
    w = guarded["score"].to_numpy(dtype=float)
    if not np.isfinite(w).any() or np.nansum(w) <= 0:
        raise RuntimeError("M06: dynamic score unavailable/degenerate")
    w = np.nan_to_num(w, nan=0.0)
    sel = systematic_pps(w, min(N, int((w > 0).sum())), rng)
    return guarded.iloc[sel]


def check_summary_coverage(pool: pd.DataFrame) -> None:
    """M07/M08 guard: refuse to run on incomplete window_summaries.

    NaN summary values compare as False in every predicate, which would
    silently exclude whole datasets from rule-based subsets. Better to
    fail loudly than to write a plausible-looking biased artifact.
    """
    if "max_decel_ms2" not in pool.columns:
        raise RuntimeError("window_summaries not joined (max_decel_ms2 missing)")
    cov = pool["max_decel_ms2"].notna().groupby(pool["dataset"]).mean()
    bad = cov[cov < MIN_SUMMARY_COVERAGE]
    if len(bad):
        detail = ", ".join(f"{d}={v:.0%}" for d, v in bad.items())
        raise RuntimeError(
            f"window_summaries incomplete (gate {MIN_SUMMARY_COVERAGE:.0%}): "
            f"{detail} -> fix clip mapping, rebuild summaries, re-run."
        )


def m07_trigger(pool, N, rng):
    if "max_decel_ms2" not in pool.columns:
        raise RuntimeError("M07 needs window_summaries (max_decel_ms2)")
    check_summary_coverage(pool)
    guarded = pool[~pool["dataset"].isin(CONFOUNDED_ACCEL)]
    thr = DECEL_TRIGGER_G * G_MS2
    events = guarded[guarded["max_decel_ms2"] >= thr]
    n_event_target = int(round(N * EVENT_SHARE))
    if len(events) > n_event_target:
        ev = events.loc[rng.choice(events.index.to_numpy(),
                                   size=n_event_target, replace=False)]
    else:
        ev = events  # natural event supply — recorded in manifest
    base_pool = pool.drop(index=ev.index)
    n_base = min(N - len(ev), len(base_pool))
    ba = base_pool.loc[rng.choice(base_pool.index.to_numpy(),
                                  size=n_base, replace=False)]
    ev = ev.copy(); ev["role"] = "event"
    ba = ba.copy(); ba["role"] = "baseline"
    return pd.concat([ev, ba]), {"threshold_ms2": thr,
                                 "n_event": len(ev), "n_baseline": len(ba)}


def build_predicates(pool):
    """Declared predicate set (nuPlan-style). Each: name, mask, guarded flag."""
    preds = []

    def has(*cols):
        return all(c in pool.columns and pool[c].notna().any() for c in cols)

    if has("max_decel_ms2"):
        preds.append(("hard_decel",
                      pool["max_decel_ms2"] >= DECEL_TRIGGER_G * G_MS2, True))
    if has("max_abs_yaw_degs"):
        preds.append(("high_yaw",
                      pool["max_abs_yaw_degs"] >= HIGH_YAW_DEG_S, False))
    if has("mean_speed_kmh"):
        preds.append(("high_speed_cruise",
                      pool["mean_speed_kmh"] >= CRUISE_SPEED_KMH, False))
    if has("mean_speed_kmh", "std_speed_kmh"):
        preds.append(("stop_and_go",
                      (pool["std_speed_kmh"] >= STOPGO_STD_MIN)
                      & (pool["mean_speed_kmh"] <= STOPGO_MEAN_MAX), False))
        preds.append(("near_stationary",
                      (pool["mean_speed_kmh"] < STATIONARY_MEAN_MAX)
                      & (pool["std_speed_kmh"] < STATIONARY_STD_MAX), False))
    if "score" in pool.columns and pool["score"].notna().any():
        guarded_scores = pool.loc[~pool["dataset"].isin(CONFOUNDED_ACCEL),
                                  "score"]
        thr = np.nanpercentile(guarded_scores, SCORE_P99_ON_GUARDED)
        preds.append(("high_dynamic_score", pool["score"] >= thr, True))
    return preds


def m08_predicate_mining(pool, N, rng):
    check_summary_coverage(pool)
    preds = build_predicates(pool)
    if len(preds) < 2:
        raise RuntimeError("M08: fewer than 2 predicates computable "
                           "(is window_summaries built?)")
    cap = N // len(preds)
    confounded_mask = pool["dataset"].isin(CONFOUNDED_ACCEL)
    counts = {name: int((mask & ~(confounded_mask & g)).sum())
              for name, mask, g in preds}
    order = sorted(preds, key=lambda p: counts[p[0]])   # rarest type first
    selected = pd.Index([])
    per_type = {}
    for name, mask, guarded in order:
        elig = pool.index[mask & ~pool.index.isin(selected)
                          & ~(confounded_mask & guarded)]
        take = min(cap, len(elig))
        pick = rng.choice(elig.to_numpy(), size=take, replace=False)
        selected = selected.append(pd.Index(pick))
        per_type[name] = int(take)
    # second pass: refill remaining budget from leftover qualifiers
    remaining = N - len(selected)
    for name, mask, guarded in order:
        if remaining <= 0:
            break
        elig = pool.index[mask & ~pool.index.isin(selected)
                          & ~(confounded_mask & guarded)]
        take = min(remaining, len(elig))
        if take > 0:
            pick = rng.choice(elig.to_numpy(), size=take, replace=False)
            selected = selected.append(pd.Index(pick))
            per_type[name] += int(take)
            remaining -= take
    out = pool.loc[selected].copy()
    # tag each window with the first predicate it satisfies (rarest-first)
    out["predicate"] = "unknown"
    for name, mask, guarded in order[::-1]:
        out.loc[out.index.isin(pool.index[mask]), "predicate"] = name
    return out, {"caps_per_type": cap, "selected_per_type": per_type,
                 "qualifying_counts": counts}


def m09_kennard_stone(pool, N, rng, supply):
    Xz, cols = feature_matrix(pool)
    if Xz is None:
        raise RuntimeError("M09: no usable feature columns")
    guarded_idx = pool.index[~pool["dataset"].isin(CONFOUNDED_ACCEL)]
    regimes = supply.index.tolist()
    quota = {r: int(q) for r, q in
             zip(regimes, allocate(np.ones(len(regimes)), N,
                                   supply["TOTAL"].to_numpy()))}
    parts, shortfall = [], {}
    pos = pd.Series(np.arange(len(pool)), index=pool.index)
    for r in regimes:
        cand = pool.index[(pool["regime"] == r)].intersection(guarded_idx)
        k = min(quota[r], len(cand))
        if k < quota[r]:
            shortfall[str(r)] = quota[r] - k
        if k <= 0:
            continue
        Xr = Xz[pos[cand].to_numpy()]
        sel_local = kennard_stone(Xr, k)
        parts.append(pool.loc[cand[sel_local]])
    return pd.concat(parts), {"feature_cols": cols, "quota": quota,
                              "guarded_shortfall": shortfall}


def m10_cluster_quota_fill(pool, N, rng, supply):
    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError:
        raise RuntimeError("M10 needs scikit-learn (pip install scikit-learn)")
    Xz, cols = feature_matrix(pool)
    if Xz is None:
        raise RuntimeError("M10: no usable feature columns")
    regimes = supply.index.tolist()
    quota = {r: int(q) for r, q in
             zip(regimes, allocate(np.ones(len(regimes)), N,
                                   supply["TOTAL"].to_numpy()))}
    pos = pd.Series(np.arange(len(pool)), index=pool.index)
    parts = []
    for r in regimes:
        cand = pool.index[pool["regime"] == r]
        k_sub = min(SUBCLUSTER_K, len(cand))
        if k_sub == 0 or quota[r] == 0:
            continue
        km = MiniBatchKMeans(n_clusters=k_sub, random_state=sk_seed(10),
                             n_init=5, batch_size=4096)
        labels = km.fit_predict(Xz[pos[cand].to_numpy()])
        sub = pool.loc[cand].copy()
        sub["subcluster"] = labels
        caps = sub["subcluster"].value_counts()
        alloc = equal_waterfill(caps, quota[r], rng)
        parts.append(srs_within(sub, "subcluster", alloc.to_dict(), rng))
    out = pd.concat(parts)
    return out, {"feature_cols": cols, "quota": quota,
                 "subcluster_k": SUBCLUSTER_K}


# =========================================================================
# Orchestration
# =========================================================================
def summarize(sel: pd.DataFrame) -> dict:
    return {
        "n": int(len(sel)),
        "regime_counts": {str(k): int(v) for k, v in
                          sel["regime"].value_counts().sort_index().items()},
        "dataset_counts": {str(k): int(v) for k, v in
                           sel["dataset"].value_counts().items()},
    }


def write_subset(sel: pd.DataFrame, method_id: str, method_name: str) -> Path:
    keep = [c for c in ID_COLS + ["role", "predicate", "subcluster", "score"]
            if c in sel.columns]
    out = sel[keep].copy()
    out.insert(0, "method", method_id)
    dup = out.duplicated(subset=["dataset", "clip_id", "window_idx"]).sum()
    if dup:
        raise RuntimeError(f"{method_id}: {dup} duplicate window IDs — bug")
    path = ACTIVE_OUT_DIR / METHOD_FILES[method_id]
    out.to_parquet(path, index=False)
    log(f"{method_id} ({method_name}): wrote {len(out):,} windows -> {path.name}")
    return path


def m11_neyman(pool, N, rng, supply):
    """M11 — Neyman variance-optimal allocation (Neyman 1934), GUARDED.

    Quota_r proportional to S_r x SD_r of the declared variable (the composite
    dynamic score of thesis 4.9.4), integerised by the shared largest-remainder
    engine and filled by SRS within regime — the same machinery as M02-M04,
    differing in the weight vector alone.

    Contrast case, not a balancing design: Neyman optimises the precision of
    an estimated mean, spending budget where a regime's windows are most
    scattered; representation is indifferent to dispersion. Read in the evaluation
    (thesis 4.10) in that role, as M01 is read as the uncurated null.

    Guard (thesis 4.9.2, widened rule): this is the first design whose
    ALLOCATION rather than selection consumes an acceleration-derived
    quantity — the confounded datasets would inflate the within-regime
    dispersion and thereby the quotas. Supply, dispersion, AND the draw all
    use the guarded pool, so every quota is achievable by construction and
    no structural shortfall arises.
    """
    if "score" not in pool.columns or not pool["score"].notna().any():
        raise RuntimeError("M11: dynamic score unavailable")
    guarded = pool[~pool["dataset"].isin(CONFOUNDED_ACCEL)]
    regimes = supply.index.tolist()
    g_supply = (guarded.groupby("regime").size()
                .reindex(regimes).fillna(0).astype(int))
    g_sd = (guarded.groupby("regime")["score"].std()
            .reindex(regimes).fillna(0.0))
    w = g_supply.to_numpy(dtype=float) * g_sd.to_numpy(dtype=float)
    if not np.isfinite(w).any() or float(np.nansum(w)) <= 0:
        raise RuntimeError("M11: degenerate Neyman weights "
                           "(guarded score dispersion is zero everywhere)")
    q = allocate(w, min(int(N), int(g_supply.sum())), g_supply.to_numpy())
    quotas = dict(zip(regimes, q.tolist()))
    sel = srs_within(guarded, "regime", quotas, rng)
    return sel, {"quota": {str(r): int(v) for r, v in quotas.items()},
                 "declared_variable": "dynamic_score",
                 "guarded_sd_per_regime": {str(r): float(v)
                                           for r, v in g_sd.items()},
                 "guard": "pool + allocation (Section 11.2, widened rule)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true",
                    help="print schemas / resolved columns and exit")
    ap.add_argument("--methods", type=str, default=",".join(METHOD_ORDER))
    ap.add_argument("--n", type=int, default=None, help="override matched-N")
    args = ap.parse_args()

    global OVERRIDE_N, ACTIVE_OUT_DIR
    if args.n is not None:
        OVERRIDE_N = args.n
        ACTIVE_OUT_DIR = SUBSET_DIR / f"tier_n{args.n}"
        log(f"budget tier: all outputs -> {ACTIVE_OUT_DIR}")

    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    pool_full, colmap, have_summaries = load_pool(inspect_only=args.inspect)

    pool = apply_dedup(pool_full)
    supply = compute_supply(pool)
    supply.to_csv(ACTIVE_OUT_DIR / "supply_table.csv")
    log(f"supply table written; regime totals:\n"
        f"{supply['TOTAL'].to_string()}")
    N = matched_n(supply)
    log(f"fidelity guard set: {sorted(CONFOUNDED_ACCEL)} ({CONFOUNDED_SRC})")

    requested = [m.strip().upper() for m in args.methods.split(",") if m.strip()]
    manifest = {
        "run_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "global_seed": GLOBAL_SEED,
        "config": {
            "DEDUP_MODE": DEDUP_MODE, "DEDUP_STRIDE": DEDUP_STRIDE,
            "DEDUP_PHASE": DEDUP_PHASE,
            "CONFOUNDED_ACCEL": sorted(CONFOUNDED_ACCEL),
            "CONFOUNDED_SOURCE": CONFOUNDED_SRC,
            "MATCHED_N_MULT": MATCHED_N_MULT, "OVERRIDE_N": OVERRIDE_N,
            "BANKIER_Q": BANKIER_Q, "EVENT_SHARE": EVENT_SHARE,
            "DECEL_TRIGGER_G": DECEL_TRIGGER_G,
            "HIGH_YAW_DEG_S": HIGH_YAW_DEG_S,
            "CRUISE_SPEED_KMH": CRUISE_SPEED_KMH,
            "SUBCLUSTER_K": SUBCLUSTER_K,
        },
        "resolved_columns": colmap,
        "pool": {"n_raw": int(len(pool_full)), "n_dedup": int(len(pool)),
                 "n_target_matched": int(N)},
        "methods": {},
    }

    alloc_rows = {}
    runners = {
        "M01": ("simple random sampling",
                lambda rng: (m01_srs(pool, N, rng), None)),
        "M02": ("stratified proportional allocation",
                lambda rng: m02_proportional(pool, N, rng, supply)),
        "M03": ("stratified equal allocation",
                lambda rng: m03_equal(pool, N, rng, supply)),
        "M04": (f"power allocation (Bankier, q={BANKIER_Q})",
                lambda rng: m04_power(pool, N, rng, supply)),
        "M05": ("two-way controlled allocation (regime x source)",
                lambda rng: m05_two_way(pool, N, rng, supply)),
        "M06": ("PPS by dynamic score (systematic-PPS engine, guarded pool)",
                lambda rng: (m06_pps(pool, N, rng), None)),
        "M07": (f"kinematic trigger + baseline ({DECEL_TRIGGER_G} g)",
                lambda rng: m07_trigger(pool, N, rng)),
        "M08": ("predicate scenario mining (capped per type)",
                lambda rng: m08_predicate_mining(pool, N, rng)),
        "M09": ("Kennard-Stone within regime quotas (guarded pool)",
                lambda rng: m09_kennard_stone(pool, N, rng, supply)),
        "M10": ("cluster-based quota fill (Yen & Lee variant)",
                lambda rng: m10_cluster_quota_fill(pool, N, rng, supply)),
        "M11": ("Neyman variance-optimal allocation (guarded pool)",
                lambda rng: m11_neyman(pool, N, rng, supply)),
    }

    for i, mid in enumerate(METHOD_ORDER, start=1):
        if mid not in requested:
            continue
        name, fn = runners[mid]
        if mid in ("M07", "M08") and not have_summaries:
            log(f"{mid} SKIPPED (window_summaries.parquet missing)")
            manifest["methods"][mid] = {"name": name, "skipped":
                                        "window_summaries missing"}
            continue
        t0 = time.time()
        try:
            sel, extra = fn(rng_for(i))
        except Exception as e:
            log(f"{mid} FAILED: {e}")
            manifest["methods"][mid] = {"name": name, "error": str(e)}
            continue
        write_subset(sel, mid, name)
        entry = {"name": name, "seed": [GLOBAL_SEED, i],
                 "runtime_s": round(time.time() - t0, 2),
                 "n_target": int(N), **summarize(sel)}
        if entry["n"] < N:
            entry["natural_n"] = True
        if extra:
            entry["params"] = extra
        manifest["methods"][mid] = entry
        if mid in ("M02", "M03", "M04") and isinstance(extra, dict):
            alloc_rows[mid] = {str(k): v for k, v in extra.items()}
        if mid == "M05" and isinstance(extra, dict):
            alloc_rows[mid] = {r: int(sum(d.values()))
                               for r, d in extra.items()}
        if mid == "M11" and isinstance(extra, dict) and "quota" in extra:
            alloc_rows[mid] = dict(extra["quota"])

    if alloc_rows:
        pd.DataFrame(alloc_rows).T.to_csv(ACTIVE_OUT_DIR / "allocation_table.csv")
        log("allocation_table.csv written")

    with open(ACTIVE_OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=json_default)
    log(f"manifest.json written -> {ACTIVE_OUT_DIR}")
    log("DONE.")


if __name__ == "__main__":
    main()