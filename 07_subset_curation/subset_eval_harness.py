#!/usr/bin/env python3
"""
subset_eval_harness.py: Section 4.10 evaluation harness.

Purpose
-------
Re-points the Section 4.8 instruments at METHODS instead of datasets. Reads the
frozen Section 4.9 artifacts (window-ID subsets M01..M11 at two tiers), joins the
fixed data layers, computes every Section 4.10 metric once, and writes a versioned
cache that all figures read. No selection is re-run, nothing is re-fitted, no
figure computes anything itself (compute-once discipline).

curation_eval.py runs this harness with the curation rows added. A standalone
run of this file writes the caches without C01-C03.

Inputs (all read-only, all frozen before this script exists)
-----------------------------------------------------------
  dataset/regime_data/cluster_assignments.parquet   frame: 402,421 windows,
                                                    regime labels 0..4, frozen PCs
  dataset/subset_data/window_summaries.parquet      six per-window summaries (Section 4.9.4)
  dataset/std_data/<dataset>.parquet                std_speed_kmh / std_acc_mag /
                                                    std_yaw_rate (rad/s) per window
  dataset/subset_data/M01..M11_*.parquet            capability tier subsets
  dataset/subset_data/tier_n71390/M01..M11_*.parquet budget tier subsets (headline,
                                                    size = rarest regime supply)
  manifest.json in each tier dir                    decision constants + counts

Outputs (dataset/subset_data/eval/)
-----------------------------------
  metrics_long.parquet          every metric, long format:
                                (tier, method, variant, metric, value, sealed)
  master_table_<tier>.csv       displayed metrics only, wide, R0/R1 context rows
  srs_envelope_budget.parquet   K SRS replicate draws (null band)
  tautology_map.csv             per (metric, method) class: T / P / O / N
  run_meta.json                 resolved schema, unit decisions, canaries, seeds

Decisions applied (fixed before evaluation)
-------------------------------------------
  Population = the 402,421 non-overlapping frame; references R0 (frame),
       R1 (guarded frame, one guard-cost row), R2 (uniform regime target);
       Section 4.10 Wasserstein values are NOT comparable to the Section 4.8
       per-dataset values (those used the overlapping variability layer) — disclosed.
  Instrument set: I1 regime/JS/H/D2, I2 per-channel W1 (in-model | held-out),
       I3 dynamic content, I4 source composition (+ worst-regime top supplier),
       I5 event & predicate retention, I6 PC / cell coverage. PSI excluded,
       Hill D1 excluded, GMM entropy optional via GMM_ENTROPY_PATH (sealed).
  SRS null envelope K=100, seeds 1000+k, budget tier.
  Bias axes = low_dyn_share x top_source_share (figure stage).
  Sealed shelf = { w1_max_speed, w1_peak_acc_mag } (+ mean_gmm_entropy if
       the hook is armed). The ex-confounded variant of EVERY metric is cached
       sealed for the confounded-source slice. Sealed values are written,
       never printed without --reveal.

Canaries (hard unless --force)
------------------------------
  frame N == 402,421 ; guarded N == 401,230 ; 0.6 g trigger on guarded == 1,088
  (the trigger count doubles as the unit calibrator for peak_decel);
  soft: regime supplies multiset, predicate union == 146,223, per-regime top
  supplier (~0.89 for Highway Cruising), pooled top source (~0.74).

Usage
-----
  python subset_eval_harness.py                 # both tiers, envelope on budget
  python subset_eval_harness.py --inspect       # print input schemas and exit
  python subset_eval_harness.py --tiers budget  # one tier only
  python subset_eval_harness.py --reveal        # also print sealed metrics
  python subset_eval_harness.py --force         # downgrade hard canaries to warnings
  python subset_eval_harness.py --canaries-only # steps 1-3 only: persist the import-time
                                                # canaries to eval/harness_canaries.json and
                                                # stop; the cache is not touched (the
                                                # reproduced predicate union and the
                                                # unit-hypothesis counts are otherwise
                                                # console output only)

No matplotlib here, this is the compute stage. Figures are separate scripts
that read the cache.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from canonical import C

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

try:
    from tqdm import tqdm
except ImportError:  # tqdm degrades to a no-op per convention
    def tqdm(it, **kw):
        return it

# ============================================================================
# CONSTANTS — every tunable lives here
# ============================================================================

sys.path.append(str(Path(__file__).resolve().parents[1]))
from thesis_paths import DATA_ROOT, REPO_ROOT  # noqa: E402
STD_DATA_DIR = DATA_ROOT / "std_data"
REGIME_DIR   = DATA_ROOT / "regime_data"
SUBSET_DIR   = DATA_ROOT / "subset_data"
EVAL_DIR     = SUBSET_DIR / "eval"

ASSIGNMENTS_PARQUET = REGIME_DIR / "cluster_assignments.parquet"
SUMMARIES_PARQUET   = SUBSET_DIR / "window_summaries.parquet"

from canonical import C

TIER_DIRS = {
    "capability": SUBSET_DIR,          # N = C.capability_n
    "budget":     C.budget_dir,        # N = C.budget_n, headline population
}
TIER_ORDER = ["budget", "capability"]

from subset_eval_common import METHOD_SHORT as METHOD_NAMES
# single definition (subset_eval_common), so a standalone run also covers
# M11 Neyman
REF_ROWS = ["R0_frame", "R1_guarded"]           # context rows in every table

from subset_eval_common import CONFOUNDED as CONFOUNDED_ACCEL   # asterisk sources (Sections 4.4.5 and 4.8.2), mirror of bias_quant

# Regime conventions. LOW_DYN_REGIME feeds low_dyn_share and the landscape
# x-axis. Labels follow regime_style.py. Computation is by ID, so a label
# swap costs nothing downstream.
N_REGIMES = 5
LOW_DYN_REGIME = 0          # Calm/Stationary per the frozen Section 4.7 labelling

# Predicate constants — transcribed from Section 4.9.8 (M07/M08 definitions).
G_MS2 = 9.80665
HARD_DECEL_G     = 0.6      # peak deceleration trigger (100-Car lineage, Dingus et al. 2006)
HIGH_YAW_DEG     = 30.0     # peak |yaw| >= 30 deg/s
HIGH_SPEED_KMH   = 100.0    # mean speed >= 100 km/h
STOPGO_STD_KMH   = 5.0      # sigma(speed) >= 5 ...
STOPGO_MEAN_KMH  = 20.0     # ... AND mean <= 20
STAT_MEAN_KMH    = 2.0      # mean < 2 ...
STAT_STD_KMH     = 1.0      # ... AND sigma < 1
TOP_SCORE_Q      = 0.99     # score >= p99 of the GUARDED frame (frozen threshold)

# W1 channel map: metric name -> master column. First three = in-model group,
# next three = displayed held-out group, last two = SEALED shelf.
W1_CHANNELS = {
    "w1_std_speed":    "std_speed_kmh",    # km/h
    "w1_std_acc":      "std_acc_mag",      # m/s^2  (*)
    "w1_std_yaw":      "std_yaw_deg",      # deg/s (converted at load)
    "w1_mean_speed":   "mean_speed",       # km/h   (held-out)
    "w1_peak_decel":   "decel_g",          # g      (*, held-out)
    "w1_peak_abs_yaw": "peak_yaw_deg",     # deg/s  (held-out)
    "w1_max_speed":    "max_speed",        # km/h   (SEALED)
    "w1_peak_acc_mag": "peak_acc_mag",     # m/s^2  (*, SEALED)
}
SEALED_METRICS = {"w1_max_speed", "w1_peak_acc_mag", "mean_gmm_entropy"}

# Optional hook: per-window frozen-GMM responsibility entropies (Section
# 4.8). Set to the parquet path to arm; leave None to skip.
GMM_ENTROPY_PATH: Path | None = None

# SRS null envelope.
K_ENVELOPE = 100
ENVELOPE_SEED_BASE = 1000
ENVELOPE_TIERS = {"budget"}

# PC occupancy grid (I6): edges frozen on the R0 frame, declared resolution.
PC_GRID_BINS = 60

# Distribution cache for the boxplot/ridgeline figures: a seeded
# per-method sample of the raw per-window values, budget tier only.
DIST_TIERS = {"budget"}
DIST_SAMPLE = 25_000
DIST_SEED = 777
DIST_CHANNELS = ["score", "std_speed_kmh", "std_acc_mag", "std_yaw_deg",
                 "pc1", "pc2", "regime"]   # one index draw per method:
                 # channels row-paired (regime feeds the coloured PCA grid)

# within-regime mean-shift metrics (introduced-bias check, all methods)
SHIFT_CHANNELS = {"speed": "std_speed_kmh", "acc": "std_acc_mag",
                  "yaw": "std_yaw_deg"}

# Canary expectations, derived from the artifacts (canonical.py), never typed.
# Drift against an accepted corpus is checked by `python canonical.py --check`
# against the dated snapshot, which is the question worth asking.
EXPECTED = {
    "frame_n":         C.frame_n,
    "guarded_n":       C.guarded_n,
    "trigger_guarded": C.event_reserve_n,     # also calibrates decel units
    "predicate_union": None,                  # soft, no derived counterpart
    "regime_supplies": set(C.regime_supplies),
    "budget_n":        C.budget_n,
    "capability_n":    C.capability_n,
}

METRIC_VERSION = "12.1.2"

# --- schema resolver candidates (case-insensitive) --------------------------
# std_data schema is authoritative from the pipeline and is NOT resolved:
# clip_id, window_idx, window_start_s, window_end_s, n_frames,
# std_speed_kmh, std_acc_mag, std_yaw_rate (rad/s).
CAND_KEYS = {
    "dataset":   ["dataset", "dataset_name", "source", "ds", "dataset_key"],
    "clip":      ["clip_id", "clip", "clip_name", "sequence", "seq", "recording"],
    "win_idx":   ["window_idx", "win_idx", "window_index", "widx", "window_id", "win_id"],
    "win_start": ["window_start_s", "win_start", "start_s", "window_start", "t_start", "start"],
}
CAND_ASSIGN = {
    "regime": ["regime", "cluster", "regime_label", "cluster_label", "label", "assignment", "k_label"],
    "pc1":    ["pc1", "pc_1", "pca1", "pca_1", "comp1", "component_1"],
    "pc2":    ["pc2", "pc_2", "pca2", "pca_2", "comp2", "component_2"],
}
CAND_SUMMARY = {
    "mean_speed":   ["mean_speed", "speed_mean", "mean_speed_kmh", "v_mean"],
    "max_speed":    ["max_speed", "speed_max", "max_speed_kmh", "v_max"],
    "peak_decel":   ["peak_decel", "peak_deceleration", "max_decel",
                     "max_decel_ms2", "decel_peak", "peak_dec",
                     "max_deceleration", "min_a_long"],
    "peak_yaw":     ["peak_abs_yaw", "peak_yaw", "max_abs_yaw",
                     "max_abs_yaw_degs", "peak_yaw_rate",
                     "max_abs_yaw_rate", "yaw_peak", "max_yaw"],
    "peak_acc_mag": ["peak_acc_mag", "max_acc_mag", "acc_mag_peak", "peak_accel_mag",
                     "max_accel_mag", "acc_mag_max"],
}
CAND_ENTROPY = {
    "entropy": ["entropy", "gmm_entropy", "membership_entropy", "resp_entropy"],
}

STD_COLS = ["clip_id", "window_idx", "window_start_s",
            "std_speed_kmh", "std_acc_mag", "std_yaw_rate"]

# ============================================================================
# small utilities
# ============================================================================

_FORCE = False


def log(msg: str) -> None:
    print(msg, flush=True)


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}", flush=True)


def fail(msg: str) -> None:
    """Hard canary / guard failure. --force downgrades to a warning."""
    if _FORCE:
        warn(f"(forced past) {msg}")
    else:
        print(f"\n  [FAIL] {msg}\n  Re-run with --force only if you have "
              f"verified this is expected.", flush=True)
        sys.exit(1)


def check(name: str, got, expected, hard: bool, tol: float = 0.0) -> bool:
    """Compare a canary value; tol is relative for numerics."""
    ok = (got == expected) if tol == 0 else (
        abs(got - expected) <= tol * max(1, abs(expected)))
    line = f"canary {name}: got {got}  expected {expected}  -> {'OK' if ok else 'MISMATCH'}"
    if ok:
        log(f"  {line}")
    elif hard:
        fail(line)
    else:
        warn(line)
    return ok


def resolve_columns(df: pd.DataFrame, spec: dict, label: str) -> dict:
    """Case-insensitive alias resolution. Returns {canon: actual_col} for hits."""
    lower = {c.lower(): c for c in df.columns}
    out = {}
    for canon, aliases in spec.items():
        for a in aliases:
            if a.lower() in lower:
                out[canon] = lower[a.lower()]
                break
    return out


def require(resolved: dict, needed: list, label: str, df: pd.DataFrame) -> None:
    missing = [n for n in needed if n not in resolved]
    if missing:
        print(f"\n[SCHEMA FAIL] {label}: could not resolve {missing}.")
        print(f"  Available columns: {list(df.columns)}")
        print("  Fix: add the actual column name to the candidate lists at the "
              "top of subset_eval_harness.py and re-run.")
        sys.exit(1)


# ============================================================================
# loading
# ============================================================================

def load_assignments() -> tuple[pd.DataFrame, dict]:
    if not ASSIGNMENTS_PARQUET.exists():
        fail(f"missing {ASSIGNMENTS_PARQUET}")
    df = pd.read_parquet(ASSIGNMENTS_PARQUET)
    keys = resolve_columns(df, CAND_KEYS, "assignments")
    cols = resolve_columns(df, CAND_ASSIGN, "assignments")
    require(keys, ["dataset", "clip"], "cluster_assignments", df)
    if "win_idx" not in keys and "win_start" not in keys:
        require(keys, ["win_idx"], "cluster_assignments (needs a window key)", df)
    require(cols, ["regime", "pc1", "pc2"], "cluster_assignments", df)
    resolved = {**keys, **cols}
    log(f"  assignments schema resolved: {resolved}")
    return df, resolved


def load_summaries() -> tuple[pd.DataFrame, dict]:
    if not SUMMARIES_PARQUET.exists():
        fail(f"missing {SUMMARIES_PARQUET}")
    df = pd.read_parquet(SUMMARIES_PARQUET)
    keys = resolve_columns(df, CAND_KEYS, "summaries")
    cols = resolve_columns(df, CAND_SUMMARY, "summaries")
    require(keys, ["dataset", "clip"], "window_summaries", df)
    if "win_idx" not in keys and "win_start" not in keys:
        require(keys, ["win_idx"], "window_summaries (needs a window key)", df)
    require(cols, ["mean_speed", "max_speed", "peak_decel", "peak_yaw",
                   "peak_acc_mag"], "window_summaries", df)
    resolved = {**keys, **cols}
    log(f"  summaries schema resolved: {resolved}")
    return df, resolved


def _read_std_one(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=STD_COLS)
    df["dataset"] = path.stem
    return df


def load_std_data(datasets: list) -> pd.DataFrame:
    paths = []
    for d in datasets:
        p = STD_DATA_DIR / f"{d}.parquet"
        if not p.exists():
            fail(f"std_data missing for dataset '{d}': {p}")
        paths.append(p)
    frames = []
    with ThreadPoolExecutor() as ex:
        for df in tqdm(ex.map(_read_std_one, paths), total=len(paths),
                       desc="std_data"):
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_subset(path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(path)
    keys = resolve_columns(df, CAND_KEYS, path.name)
    require(keys, ["dataset", "clip"], path.name, df)
    if "win_idx" not in keys and "win_start" not in keys:
        require(keys, ["win_idx"], f"{path.name} (needs a window key)", df)
    return df, keys


def discover_methods(tier_dir: Path) -> dict:
    out = {}
    for p in sorted(tier_dir.glob("M[0-9][0-9]_*.parquet")):
        out[p.name[:3]] = p
    return out


def read_manifest(tier_dir: Path) -> dict:
    p = tier_dir / "manifest.json"
    if not p.exists():
        warn(f"no manifest.json in {tier_dir} — pooled-mean and count checks skipped")
        return {}
    with open(p) as f:
        return json.load(f)


def _walk_json(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_json(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_json(v, f"{path}[{i}]")
    else:
        yield path, obj


# ============================================================================
# key building and the master join
# ============================================================================

def make_key(df: pd.DataFrame, m: dict, strategy: str) -> pd.Series:
    ds = df[m["dataset"]].astype(str)
    clip = df[m["clip"]].astype(str)
    if strategy == "idx":
        w = df[m["win_idx"]].astype("int64").astype(str)
    else:  # start-time key, robust to reindexing, 0.1 s resolution
        w = np.round(df[m["win_start"]].astype(float) * 10).astype("int64").astype(str)
    return ds + "|" + clip + "|" + w


def build_master() -> tuple[pd.DataFrame, dict]:
    """Join assignments + std_data + summaries into one frame keyed by window.

    Key strategies are PER TABLE: on this corpus the assignment grid carries
    only window_start_s while window_summaries carries only window_idx.
    std_data carries both by pipeline contract, so it joins FIRST and acts as
    the idx<->start bridge; afterwards the master holds both key forms and
    every other table (summaries, subsets, entropy hook) joins on whichever
    key it supports.
    """
    log("[1/6] Loading fixed layers")
    assign, m_a = load_assignments()
    summ, m_s = load_summaries()

    n = len(assign)
    check("frame_n", n, EXPECTED["frame_n"], hard=True)

    primary = "idx" if "win_idx" in m_a else "start"
    s_strategy = "idx" if "win_idx" in m_s else "start"
    log(f"  window keys: assignments -> {primary}, summaries -> {s_strategy}, "
        f"std_data -> both (bridge)")

    master = pd.DataFrame({
        "dataset": assign[m_a["dataset"]].astype(str).values,
        "clip":    assign[m_a["clip"]].astype(str).values,
        "regime":  assign[m_a["regime"]].astype(int).values,
        "pc1":     assign[m_a["pc1"]].astype(float).values,
        "pc2":     assign[m_a["pc2"]].astype(float).values,
    })
    if "win_idx" in m_a:
        master["win_idx"] = assign[m_a["win_idx"]].astype("int64").values
    if "win_start" in m_a:
        master["win_start"] = assign[m_a["win_start"]].astype(float).values

    MM = {"dataset": "dataset", "clip": "clip",
          "win_idx": "win_idx", "win_start": "win_start"}
    master["key"] = make_key(master, MM, primary).values
    if master["key"].duplicated().any():
        fail("duplicate window keys inside cluster_assignments — frame corrupt")

    # -- std_data join FIRST: sigma channels + the missing key field ---------
    datasets = sorted(master["dataset"].unique().tolist())
    log(f"  frame covers {len(datasets)} datasets: {datasets}")
    missing_conf = CONFOUNDED_ACCEL - set(datasets)
    if missing_conf:
        fail(f"CONFOUNDED_ACCEL names not found in frame: {missing_conf} "
             f"(name drift — fix the constant)")
    std = load_std_data(datasets)
    std_map = {"dataset": "dataset", "clip": "clip_id",
               "win_idx": "window_idx", "win_start": "window_start_s"}
    std["key"] = make_key(std, std_map, primary).values
    std = std.drop_duplicates("key")
    master = master.merge(
        std[["key", "std_speed_kmh", "std_acc_mag", "std_yaw_rate",
             "window_idx", "window_start_s"]],
        on="key", how="left")
    miss = int(master["std_speed_kmh"].isna().sum())
    if miss:
        sample = master.loc[master["std_speed_kmh"].isna(), "key"].head(5).tolist()
        fail(f"std_data join incomplete: {miss} frame windows unmatched, "
             f"e.g. {sample}")

    # bridge: attach whichever key field the assignment grid lacked
    if "win_idx" not in master.columns:
        master["win_idx"] = master["window_idx"].astype("int64")
    if "win_start" not in master.columns:
        master["win_start"] = master["window_start_s"].astype(float)
    master = master.drop(columns=["window_idx", "window_start_s"])
    alt = "start" if primary == "idx" else "idx"
    master["key_alt"] = make_key(master, MM, alt).values
    if master["key_alt"].duplicated().any():
        fail("bridged alternate keys not unique — window_idx/window_start_s "
             "disagree between std_data and the assignment grid")

    # -- summaries join on ITS key (must be 100% coverage per Section 4.9.4) ---------
    s = pd.DataFrame({
        "jkey": make_key(summ, m_s, s_strategy).values,
        "mean_speed":   summ[m_s["mean_speed"]].astype(float).values,
        "max_speed":    summ[m_s["max_speed"]].astype(float).values,
        "peak_decel":   summ[m_s["peak_decel"]].astype(float).values,
        "peak_yaw":     summ[m_s["peak_yaw"]].astype(float).values,
        "peak_acc_mag": summ[m_s["peak_acc_mag"]].astype(float).values,
    })
    if s["jkey"].duplicated().any():
        s = s.drop_duplicates("jkey")
        warn("window_summaries contained duplicate keys — deduplicated")
    join_col = "key" if s_strategy == primary else "key_alt"
    master = master.merge(s, left_on=join_col, right_on="jkey", how="left")
    master = master.drop(columns=["jkey"])
    miss = int(master["mean_speed"].isna().sum())
    if miss:
        sample = master.loc[master["mean_speed"].isna(), join_col].head(5).tolist()
        fail(f"summaries join incomplete: {miss} frame windows unmatched, "
             f"e.g. {sample}")

    # -- optional hook: frozen-GMM per-window entropy ----------------------
    if GMM_ENTROPY_PATH is not None:
        gdf = pd.read_parquet(GMM_ENTROPY_PATH)
        gk = resolve_columns(gdf, CAND_KEYS, "gmm_entropy")
        gc = resolve_columns(gdf, CAND_ENTROPY, "gmm_entropy")
        require(gk, ["dataset", "clip"], "gmm_entropy", gdf)
        require(gc, ["entropy"], "gmm_entropy", gdf)
        if "win_idx" not in gk and "win_start" not in gk:
            require(gk, ["win_idx"], "gmm_entropy (needs a window key)", gdf)
        g_strategy = "idx" if "win_idx" in gk else "start"
        g_join = "key" if g_strategy == primary else "key_alt"
        gdf2 = pd.DataFrame({
            "jkey": make_key(gdf, gk, g_strategy).values,
            "gmm_entropy": gdf[gc["entropy"]].astype(float).values,
        }).drop_duplicates("jkey")
        master = master.merge(gdf2, left_on=g_join, right_on="jkey",
                              how="left").drop(columns=["jkey"])
        gm = int(master["gmm_entropy"].isna().sum())
        if gm:
            warn(f"gmm_entropy join left {gm} windows without a value")
        log("  S3 armed: mean_gmm_entropy will be computed (sealed)")

    meta = {"join_strategy": primary, "alt_join_strategy": alt,
            "assignments_schema": m_a, "summaries_schema": m_s}
    return master, meta


# ============================================================================
# unit calibration and verification guards
# ============================================================================

def calibrate_units(master: pd.DataFrame, meta: dict) -> None:
    log("[2/6] Unit calibration and frame verification")
    decisions = {}

    # yaw: std layer is rad/s by pipeline contract -> deg column for W1/figures.
    master["std_yaw_deg"] = master["std_yaw_rate"] * (180.0 / math.pi)
    decisions["std_yaw_rate"] = "rad/s (contract) -> std_yaw_deg = x180/pi"

    med_py = float(np.nanmedian(master["peak_yaw"].abs()))
    if med_py < 1.5:
        master["peak_yaw_deg"] = master["peak_yaw"].abs() * (180.0 / math.pi)
        decisions["peak_yaw"] = f"median {med_py:.3f} -> rad/s, converted to deg/s"
    else:
        master["peak_yaw_deg"] = master["peak_yaw"].abs()
        decisions["peak_yaw"] = f"median {med_py:.2f} -> already deg/s"

    p99_ms = float(np.nanpercentile(master["mean_speed"], 99))
    if p99_ms < 45.0:
        master["mean_speed"] = master["mean_speed"] * 3.6
        master["max_speed"] = master["max_speed"] * 3.6
        decisions["speed"] = f"p99 {p99_ms:.1f} -> m/s, converted to km/h"
    else:
        decisions["speed"] = f"p99 {p99_ms:.1f} -> km/h"

    # peak_decel: unit hypothesis chosen by the frozen 1,088 anchor (Section 4.9.3).
    guard = ~master["dataset"].isin(CONFOUNDED_ACCEL)
    d_abs = master["peak_decel"].abs()
    c_g = int(((d_abs >= HARD_DECEL_G) & guard).sum())
    c_ms = int(((d_abs >= HARD_DECEL_G * G_MS2) & guard).sum())
    exp = EXPECTED["trigger_guarded"]
    log(f"  decel hypothesis counts on guarded frame: g-units -> {c_g}, "
        f"m/s^2-units -> {c_ms} (expected {exp})")
    if abs(c_g - exp) <= abs(c_ms - exp):
        master["decel_g"] = d_abs
        chosen, got = "g", c_g
    else:
        master["decel_g"] = d_abs / G_MS2
        chosen, got = "m/s^2", c_ms
    decisions["peak_decel"] = f"stored in {chosen} (anchor-calibrated)"
    meta["decel_hypothesis_counts"] = {"g_units": c_g, "ms2_units": c_ms,
                                      "expected": int(exp), "chosen": chosen}
    check("trigger_guarded(0.6g)", got, exp, hard=True,
          tol=0.02)  # unit calibration is load-bearing for I5

    # guarded frame size
    check("guarded_n", int(guard.sum()), EXPECTED["guarded_n"], hard=True)

    # regime supplies (soft, multiset)
    supplies = master["regime"].value_counts().sort_index()
    log(f"  regime supplies: {supplies.to_dict()}")
    if set(supplies.values.tolist()) != EXPECTED["regime_supplies"]:
        warn("regime supply multiset differs from the frozen record — "
             "verify LOW_DYN_REGIME / labels against regime_style.py")

    # non-overlap spacing verification (mirror of Section 4.9.1), if starts available
    if "win_start" in master.columns:
        sp = (master.sort_values(["dataset", "clip", "win_start"])
                    .groupby(["dataset", "clip"])["win_start"].diff().dropna())
        if len(sp):
            med_sp = float(sp.median())
            log(f"  median within-clip start spacing: {med_sp:.1f} s "
                f"(20.0 confirms the non-overlapping grid)")
            meta["median_start_spacing_s"] = med_sp
            if abs(med_sp - 20.0) > 1.0:
                warn("spacing not ~20 s — verify the frame is the assignment "
                     "grid, not the overlapping variability layer")

    meta["unit_decisions"] = decisions


def add_score_and_predicates(master: pd.DataFrame, meta: dict,
                             manifests: dict) -> None:
    log("[3/6] Dynamic score and predicate masks")
    # score = sum_ch std_ch / pooled_mean(std_ch), pooled means on the FULL
    # frame, native units (Section 4.9.4 single-source rule; yaw ratio is unit-free).
    pooled = {
        "std_speed_kmh": float(master["std_speed_kmh"].mean()),
        "std_acc_mag":   float(master["std_acc_mag"].mean()),
        "std_yaw_rate":  float(master["std_yaw_rate"].mean()),
    }
    master["score"] = sum(master[c] / pooled[c] for c in pooled)
    log(f"  pooled means (native units): "
        f"{ {k: round(v, 6) for k, v in pooled.items()} }")

    # verify against manifest-echoed constants where discoverable
    matched = 0
    for tier, man in manifests.items():
        for path, val in _walk_json(man):
            if "pooled" in path.lower() and isinstance(val, (int, float)):
                hits = [k for k, v in pooled.items()
                        if math.isclose(v, float(val), rel_tol=1e-4)]
                if hits:
                    matched += 1
    if matched:
        log(f"  manifest pooled-mean check: {matched} value(s) matched — OK")
    else:
        warn("no pooled-mean constants found in manifests to verify against; "
             "recomputed values printed above — consider echoing them from "
             "the selection stage (design-note verification guard)")
    meta["pooled_means"] = pooled

    guard = ~master["dataset"].isin(CONFOUNDED_ACCEL)
    master["is_guarded"] = guard
    thr_score = float(master.loc[guard, "score"].quantile(TOP_SCORE_Q))
    meta["score_p99_guarded"] = thr_score
    log(f"  top-score threshold (p99, guarded frame): {thr_score:.4f}")

    # predicate masks exactly as M08 defines them (accel-consuming ones guarded)
    master["p_hard_decel"] = (master["decel_g"] >= HARD_DECEL_G) & guard
    master["p_high_yaw"]   = master["peak_yaw_deg"] >= HIGH_YAW_DEG
    master["p_high_speed"] = master["mean_speed"] >= HIGH_SPEED_KMH
    master["p_stop_go"]    = ((master["std_speed_kmh"] >= STOPGO_STD_KMH)
                              & (master["mean_speed"] <= STOPGO_MEAN_KMH))
    master["p_near_stat"]  = ((master["mean_speed"] < STAT_MEAN_KMH)
                              & (master["std_speed_kmh"] < STAT_STD_KMH))
    master["p_top_score"]  = (master["score"] >= thr_score) & guard

    union = int((master["p_hard_decel"] | master["p_high_yaw"]
                 | master["p_high_speed"] | master["p_stop_go"]
                 | master["p_near_stat"] | master["p_top_score"]).sum())
    if EXPECTED["predicate_union"] is not None:
        check("predicate_union", union, EXPECTED["predicate_union"],
              hard=False, tol=0.01)
    else:
        log(f"  predicate union: {union:,} "
            f"(no derived expectation; informational)")

    # supplier canaries (frame level): pooled top source, per-regime top source
    top_src = master["dataset"].value_counts(normalize=True)
    log(f"  frame top source: {top_src.index[0]} = {top_src.iloc[0]:.3f} "
        f"(record: ~0.74 NVIDIA)")
    per_reg = (master.groupby("regime")["dataset"]
                     .apply(lambda s: s.value_counts(normalize=True).iloc[0]))
    log(f"  per-regime top-supplier shares: "
        f"{ {int(k): round(float(v), 3) for k, v in per_reg.items()} } "
        f"(record: ~0.89 for Highway Cruising)")
    per_reg_name = (master.groupby("regime")["dataset"]
                          .apply(lambda s: s.value_counts(normalize=True).index[0]))
    # persisted canaries: everything this step prints and Section 5.7
    # reports, written as one JSON beside the cache so
    # that no number of the harness exists only in a console line.
    meta["canaries"] = {
        "frame_n": int(len(master)),
        "guarded_n": int(guard.sum()),
        "trigger_guarded": int(master["p_hard_decel"].sum()),
        "predicate_union_harness": union,
        "predicate_union_expected": EXPECTED["predicate_union"],
        "predicate_counts_frame": {p: int(master[p].sum()) for p in
                                   ["p_hard_decel", "p_high_yaw", "p_high_speed",
                                    "p_stop_go", "p_near_stat", "p_top_score"]},
        "score_p99_guarded": thr_score,
        "frame_top_source": str(top_src.index[0]),
        "frame_top_source_share": float(top_src.iloc[0]),
        "per_regime_top_supplier": {int(k): str(v) for k, v in per_reg_name.items()},
        "per_regime_top_supplier_share": {int(k): float(v) for k, v in per_reg.items()},
        "regime_supplies": {int(k): int(v) for k, v in
                            master["regime"].value_counts().sort_index().items()},
    }


# ============================================================================
# metric primitives
# ============================================================================

def shannon_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def hill_d2(p: np.ndarray) -> float:
    s = float((p ** 2).sum())
    return 1.0 / s if s > 0 else float("nan")


def gini(x: np.ndarray) -> float:
    x = np.sort(x[~np.isnan(x)])
    n = len(x)
    if n == 0:
        return float("nan")
    tot = x.sum()
    if tot <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * x).sum()) / (n * tot) - (n + 1.0) / n)


def pc_cells(pc1: np.ndarray, pc2: np.ndarray, edges: tuple) -> set:
    e1, e2 = edges
    i1 = np.clip(np.digitize(pc1, e1) - 1, 0, PC_GRID_BINS - 1)
    i2 = np.clip(np.digitize(pc2, e2) - 1, 0, PC_GRID_BINS - 1)
    return set((i1.astype(np.int64) * PC_GRID_BINS + i2).tolist())


PREDICATES = ["p_hard_decel", "p_high_yaw", "p_high_speed",
              "p_stop_go", "p_near_stat", "p_top_score"]
RET_NAMES = {p: "ret_" + p[2:] for p in PREDICATES}


class BaseCtx:
    """Pre-computed reference quantities for one variant (R0 or ex-confounded)."""

    def __init__(self, base: pd.DataFrame, pc_edges: tuple):
        self.n = len(base)
        self.regime_p = (base["regime"].value_counts(normalize=True)
                         .reindex(range(N_REGIMES), fill_value=0.0)
                         .values.astype(float))
        self.uniform = np.full(N_REGIMES, 1.0 / N_REGIMES)
        self.arrays = {col: base[col].dropna().values.astype(float)
                       for col in set(W1_CHANNELS.values())}
        self.pred_rate = {p: float(base[p].mean()) for p in PREDICATES}
        self.score_mean = float(base["score"].mean())
        self.pc_edges = pc_edges
        self.pc_cells = pc_cells(base["pc1"].values, base["pc2"].values, pc_edges)
        self.cellset = set(map(tuple, base[["regime", "dataset"]]
                               .drop_duplicates().itertuples(index=False)))
        self.datasets = sorted(base["dataset"].unique().tolist())
        self.reg_ch_mean = {
            (r, key): float(base.loc[base["regime"] == r, col].mean())
            for r in range(N_REGIMES) for key, col in SHIFT_CHANNELS.items()}


def compute_row(sub: pd.DataFrame, ctx: BaseCtx, with_gmm: bool) -> dict:
    out = {}
    n = len(sub)
    out["n_windows"] = float(n)
    out["frac_of_frame"] = n / ctx.n if ctx.n else float("nan")

    p = (sub["regime"].value_counts(normalize=True)
         .reindex(range(N_REGIMES), fill_value=0.0).values.astype(float))
    for r in range(N_REGIMES):
        out[f"share_r{r}"] = float(p[r])
    out["low_dyn_share"] = float(p[LOW_DYN_REGIME])
    out["entropy_bits"] = shannon_bits(p)
    out["entropy_norm"] = out["entropy_bits"] / math.log2(N_REGIMES)
    out["d2_regimes"] = hill_d2(p)
    out["js_to_frame"] = js_divergence(p, ctx.regime_p)
    out["js_to_uniform"] = js_divergence(p, ctx.uniform)

    for mname, col in W1_CHANNELS.items():
        vals = sub[col].dropna().values.astype(float)
        base = ctx.arrays[col]
        out[mname] = (float(wasserstein_distance(vals, base))
                      if len(vals) and len(base) else float("nan"))

    out["mean_score_ratio"] = (float(sub["score"].mean()) / ctx.score_mean
                               if ctx.score_mean else float("nan"))
    out["gini_sig_speed"] = gini(sub["std_speed_kmh"].values.astype(float))
    out["gini_sig_acc"] = gini(sub["std_acc_mag"].values.astype(float))
    out["gini_sig_yaw"] = gini(sub["std_yaw_deg"].values.astype(float))

    src = sub["dataset"].value_counts(normalize=True)
    shares = src.values.astype(float)
    out["volume_gini"] = gini(shares)
    out["d2_sources"] = hill_d2(shares)
    out["top_source_share"] = float(shares.max()) if len(shares) else float("nan")
    wr = (sub.groupby("regime")["dataset"]
             .apply(lambda s: float(s.value_counts(normalize=True).iloc[0])))
    out["worst_regime_top_source"] = float(wr.max()) if len(wr) else float("nan")
    for ds in ctx.datasets:
        out[f"src_share_{ds}"] = float(src.get(ds, 0.0))

    for r in range(N_REGIMES):
        sub_r = sub[sub["regime"] == r]
        for key, col in SHIFT_CHANNELS.items():
            base_m = ctx.reg_ch_mean[(r, key)]
            out[f"shift_r{r}_{key}"] = (float(sub_r[col].mean()) - base_m
                                        if len(sub_r) else float("nan"))

    for pcol in PREDICATES:
        base_rate = ctx.pred_rate[pcol]
        sub_rate = float(sub[pcol].mean()) if n else float("nan")
        out[RET_NAMES[pcol]] = (sub_rate / base_rate
                                if base_rate > 0 else float("nan"))

    cells = pc_cells(sub["pc1"].values, sub["pc2"].values, ctx.pc_edges)
    out["pc_coverage_share"] = (len(cells & ctx.pc_cells) / len(ctx.pc_cells)
                                if ctx.pc_cells else float("nan"))
    sub_cells = set(map(tuple, sub[["regime", "dataset"]]
                        .drop_duplicates().itertuples(index=False)))
    out["cell_coverage_share"] = (len(sub_cells & ctx.cellset) / len(ctx.cellset)
                                  if ctx.cellset else float("nan"))

    if with_gmm and "gmm_entropy" in sub.columns:
        out["mean_gmm_entropy"] = float(sub["gmm_entropy"].mean())
    return out


# ============================================================================
# tautology map: T = construction-determined, P = influenced,
# O = held-out, N = null anchor. Group defaults + metric-level overrides.
# ============================================================================

I1 = ["share_r0", "share_r1", "share_r2", "share_r3", "share_r4",
      "low_dyn_share", "entropy_bits", "entropy_norm", "d2_regimes",
      "js_to_frame", "js_to_uniform"]
I2 = list(W1_CHANNELS.keys())
I3 = ["mean_score_ratio", "gini_sig_speed", "gini_sig_acc", "gini_sig_yaw"]
I4 = ["volume_gini", "d2_sources", "top_source_share", "worst_regime_top_source"]
I5 = list(RET_NAMES.values())
I6 = ["pc_coverage_share", "cell_coverage_share"]

GROUP_CLS = {
    # group: {method: class}; M01 handled globally as N
    "I1": {"M02": "T", "M03": "T", "M04": "T", "M05": "T", "M06": "P",
           "M07": "O", "M08": "O", "M09": "T", "M10": "T"},
    "I2": {"M02": "O", "M03": "O", "M04": "O", "M05": "O", "M06": "P",
           "M07": "O", "M08": "P", "M09": "P", "M10": "P"},
    "I3": {"M02": "O", "M03": "O", "M04": "O", "M05": "O", "M06": "T",
           "M07": "P", "M08": "P", "M09": "P", "M10": "P"},
    "I4": {"M02": "O", "M03": "O", "M04": "O", "M05": "T", "M06": "P",
           "M07": "P", "M08": "P", "M09": "P", "M10": "O"},
    "I5": {"M02": "O", "M03": "O", "M04": "O", "M05": "O", "M06": "P",
           "M07": "T", "M08": "T", "M09": "P", "M10": "O"},
    "I6": {"M02": "O", "M03": "O", "M04": "O", "M05": "O", "M06": "P",
           "M07": "O", "M08": "O", "M09": "T", "M10": "P"},
}
CLS_OVERRIDES = {  # (metric, method) -> class
    ("w1_std_acc", "M07"): "P",      # trigger consumes the accel channel
    ("w1_peak_decel", "M07"): "T",   # the trigger variable itself
    ("w1_peak_decel", "M08"): "T",   # hard-decel predicate
    ("w1_peak_abs_yaw", "M08"): "T",  # high-yaw predicate
    ("w1_mean_speed", "M08"): "T",   # speed predicates consume the mean
}


def metric_group(metric: str) -> str:
    for g, lst in [("I1", I1), ("I2", I2), ("I3", I3),
                   ("I4", I4), ("I5", I5), ("I6", I6)]:
        if metric in lst:
            return g
    return "misc"


def build_tautology_rows() -> pd.DataFrame:
    rows = []
    metrics = I1 + I2 + I3 + I4 + I5 + I6
    for metric in metrics:
        g = metric_group(metric)
        for mid in METHOD_NAMES:
            if mid == "M01":
                cls = "N"
            else:
                cls = CLS_OVERRIDES.get((metric, mid),
                                        GROUP_CLS.get(g, {}).get(mid, "O"))
            rows.append({"metric": metric, "method": mid, "cls": cls})
    return pd.DataFrame(rows)


# ============================================================================
# per-tier orchestration
# ============================================================================

MASTER_COL_ORDER = (["n_windows", "frac_of_frame"] + I1
                    + [m for m in I2 if m not in SEALED_METRICS]
                    + I3 + I4 + I5 + I6)


def run_tier(tier: str, master: pd.DataFrame, meta: dict,
             manifests: dict, k_envelope: int) -> tuple[list, list]:
    tier_dir = TIER_DIRS[tier]
    log(f"[4/6] Tier '{tier}' — {tier_dir}")
    method_paths = discover_methods(tier_dir)
    if not method_paths:
        fail(f"no M??_*.parquet subsets found in {tier_dir}")
    log(f"  methods found: {sorted(method_paths.keys())}")

    # frozen PC-grid edges: min/max of the R0 frame (I6, declared)
    e1 = np.linspace(master["pc1"].min(), master["pc1"].max(), PC_GRID_BINS + 1)
    e2 = np.linspace(master["pc2"].min(), master["pc2"].max(), PC_GRID_BINS + 1)
    edges = (e1, e2)

    with_gmm = "gmm_entropy" in master.columns
    guarded_df = master[master["is_guarded"]]
    ctx_main = BaseCtx(master, edges)
    ctx_exc = BaseCtx(guarded_df, edges)

    key_pos = {meta["join_strategy"]:
               pd.Series(np.arange(len(master)), index=master["key"].values)}
    if meta.get("alt_join_strategy") and "key_alt" in master.columns:
        key_pos[meta["alt_join_strategy"]] = pd.Series(
            np.arange(len(master)), index=master["key_alt"].values)

    long_rows, env_rows, dist_chunks = [], [], []

    def dist_sample(name: str, df: pd.DataFrame, seed_off: int) -> None:
        if tier not in DIST_TIERS:
            return
        rng = np.random.default_rng(DIST_SEED + seed_off)
        n_rows = len(df)
        if n_rows > DIST_SAMPLE:
            idx = rng.choice(n_rows, size=DIST_SAMPLE, replace=False)
        else:
            idx = np.arange(n_rows)
        for ch in DIST_CHANNELS:
            v = df[ch].values[idx].astype(np.float32)
            dist_chunks.append(pd.DataFrame(
                {"tier": tier, "method": name, "channel": ch, "value": v}))

    def emit(method: str, variant: str, row: dict) -> None:
        for metric, value in row.items():
            sealed = (variant == "ex_confounded") or (metric in SEALED_METRICS)
            long_rows.append({"tier": tier, "method": method,
                              "variant": variant, "metric": metric,
                              "value": value, "sealed": sealed})

    # reference rows: R0 (frame vs itself) and R1 (guard-cost, Section 4.9.2)
    emit("R0_frame", "main", compute_row(master, ctx_main, with_gmm))
    emit("R1_guarded", "main", compute_row(guarded_df, ctx_main, with_gmm))
    dist_sample("R0_frame", master, 0)
    dist_sample("R1_guarded", guarded_df, 1)

    man = manifests.get(tier, {})
    for mid in tqdm(sorted(method_paths), desc=f"{tier} methods"):
        sdf, keys = load_subset(method_paths[mid])
        avail = [s for s in ("idx", "start")
                 if (("win_idx" if s == "idx" else "win_start") in keys)
                 and s in key_pos]
        if not avail:
            fail(f"{mid}: no window key shared with the frame "
                 f"(subset resolved keys: {sorted(keys)})")
        strat = (meta["join_strategy"] if meta["join_strategy"] in avail
                 else avail[0])
        skey = make_key(sdf, keys, strat)

        if skey.duplicated().any():
            fail(f"{mid}: duplicate window identifiers inside the subset")
        pos = key_pos[strat].reindex(skey.values)
        if pos.isna().any():
            missing = skey.values[pos.isna().values][:5].tolist()
            fail(f"{mid}: {int(pos.isna().sum())} windows not in the frame, "
                 f"e.g. {missing}")
        sub = master.iloc[pos.values.astype(int)]

        # manifest realized-count check (best effort, structure-agnostic)
        man_vals = [float(v) for p, v in _walk_json(man)
                    if mid in p and isinstance(v, (int, float))]
        if man_vals and not any(abs(x - len(sub)) < 0.5 for x in man_vals):
            warn(f"{mid}: realized n={len(sub):,} matches no numeric under "
                 f"'{mid}' in the manifest (best-effort check)")
        log(f"    {mid} ({METHOD_NAMES[mid]}): n = {len(sub):,}")

        emit(mid, "main", compute_row(sub, ctx_main, with_gmm))
        emit(mid, "ex_confounded",
             compute_row(sub[sub["is_guarded"]], ctx_exc, with_gmm))
        dist_sample(mid, sub, 10 + int(mid[1:]))

    # SRS null envelope: K draws at this tier's M01 size
    if k_envelope > 0 and tier in ENVELOPE_TIERS:
        n_target = int(next(v for r in long_rows
                            if r["method"] == "M01" and r["variant"] == "main"
                            and r["metric"] == "n_windows"
                            for v in [r["value"]]))
        log(f"[5/6] SRS envelope: K = {k_envelope} draws of n = {n_target:,} "
            f"(seeds {ENVELOPE_SEED_BASE}+k)")
        for k in tqdm(range(k_envelope), desc="envelope"):
            rng = np.random.default_rng(ENVELOPE_SEED_BASE + k)
            idx = rng.choice(len(master), size=n_target, replace=False)
            row = compute_row(master.iloc[idx], ctx_main, with_gmm=False)
            for metric, value in row.items():
                env_rows.append({"tier": tier, "draw": k,
                                 "metric": metric, "value": value})

    return long_rows, env_rows, dist_chunks


def write_outputs(all_rows: list, env_rows: list, dist_chunks: list,
                  meta: dict, reveal: bool) -> None:
    log("[6/6] Writing the eval cache")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    longdf = pd.DataFrame(all_rows)
    longdf.to_parquet(EVAL_DIR / "metrics_long.parquet", index=False)
    log(f"  {EVAL_DIR / 'metrics_long.parquet'}  "
        f"({len(longdf):,} metric values)")

    taut = build_tautology_rows()
    taut.to_csv(EVAL_DIR / "tautology_map.csv", index=False)
    log(f"  {EVAL_DIR / 'tautology_map.csv'}")

    for tier in longdf["tier"].unique():
        disp = longdf[(longdf["tier"] == tier)
                      & (longdf["variant"] == "main")
                      & (~longdf["sealed"])]
        wide = disp.pivot(index="method", columns="metric", values="value")
        order = [r for r in REF_ROWS + list(METHOD_NAMES) if r in wide.index]
        cols = [c for c in MASTER_COL_ORDER if c in wide.columns]
        wide = wide.loc[order, cols]
        out = EVAL_DIR / f"master_table_{tier}.csv"
        wide.to_csv(out, float_format="%.6g")
        log(f"  {out}")
        head_cols = [c for c in ["n_windows", "js_to_frame", "js_to_uniform",
                                 "entropy_norm", "top_source_share",
                                 "ret_hard_decel", "pc_coverage_share"]
                     if c in wide.columns]
        log(f"\n  === {tier} master table (headline columns) ===")
        log(wide[head_cols].round(4).to_string())
        log("")

    if dist_chunks:
        ddf = pd.concat(dist_chunks, ignore_index=True)
        ddf.to_parquet(EVAL_DIR / "dist_cache_budget.parquet", index=False)
        log(f"  {EVAL_DIR / 'dist_cache_budget.parquet'}  "
            f"({len(ddf):,} sampled values, seed {DIST_SEED})")

    if env_rows:
        envdf = pd.DataFrame(env_rows)
        envdf.to_parquet(EVAL_DIR / "srs_envelope_budget.parquet", index=False)
        log(f"  {EVAL_DIR / 'srs_envelope_budget.parquet'}  "
            f"({envdf['draw'].nunique()} draws)")

    meta["metric_version"] = METRIC_VERSION
    meta["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["k_envelope"] = K_ENVELOPE
    meta["pc_grid_bins"] = PC_GRID_BINS
    meta["sealed_metrics"] = sorted(SEALED_METRICS)
    meta["decisions"] = ("D14 population/references; D15 instruments; "
                         "D16 grammar+envelope; D17 scatter axes; D5 shelf")
    with open(EVAL_DIR / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    log(f"  {EVAL_DIR / 'run_meta.json'}")

    if reveal:
        sealed = longdf[longdf["sealed"] & (longdf["variant"] == "main")]
        log("\n  === SEALED SHELF (--reveal; 12.4 material) ===")
        if len(sealed):
            log(sealed.pivot_table(index="method", columns="metric",
                                   values="value").round(4).to_string())
        else:
            log("  (no sealed main-variant metrics present)")
        log("  ex-confounded variant rows are in metrics_long.parquet "
            "(variant == 'ex_confounded').")
    else:
        n_sealed = int(longdf["sealed"].sum())
        log(f"\n  Sealed rows written and hidden: {n_sealed:,} "
            f"(reveal in 12.4 with --reveal).")


# ============================================================================
# entry point
# ============================================================================

def inspect_inputs() -> None:
    def show(path: Path) -> None:
        if not path.exists():
            print(f"\n-- MISSING: {path}")
            return
        df = pd.read_parquet(path)
        print(f"\n-- {path}  shape={df.shape}")
        print(df.dtypes.to_string())

    show(ASSIGNMENTS_PARQUET)
    show(SUMMARIES_PARQUET)
    any_std = sorted(STD_DATA_DIR.glob("*.parquet"))
    if any_std:
        show(any_std[0])
    for tier, d in TIER_DIRS.items():
        ms = discover_methods(d)
        print(f"\n-- tier '{tier}' ({d}): {len(ms)} subset files")
        if ms:
            show(next(iter(ms.values())))


def main() -> None:
    global _FORCE, K_ENVELOPE
    ap = argparse.ArgumentParser(description="Chapter 12.1 evaluation harness")
    ap.add_argument("--tiers", default="both",
                    choices=["both", "budget", "capability"])
    ap.add_argument("--k-envelope", type=int, default=K_ENVELOPE)
    ap.add_argument("--reveal", action="store_true",
                    help="print sealed metrics (12.4 only)")
    ap.add_argument("--inspect", action="store_true",
                    help="print input schemas and exit")
    ap.add_argument("--force", action="store_true",
                    help="downgrade hard canary failures to warnings")
    ap.add_argument("--canaries-only", action="store_true",
                    help="persist the import-time canaries to eval/harness_canaries.json "
                         "and stop before any tier is evaluated (cache untouched)")
    args = ap.parse_args()
    _FORCE = args.force
    K_ENVELOPE = args.k_envelope

    if args.inspect:
        inspect_inputs()
        return

    t0 = time.time()
    log(f"subset_eval_harness  METRIC_VERSION={METRIC_VERSION}")

    master, meta = build_master()
    calibrate_units(master, meta)
    manifests = {t: read_manifest(d) for t, d in TIER_DIRS.items()}
    add_score_and_predicates(master, meta, manifests)

    if args.canaries_only:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        payload = {"metric_version": METRIC_VERSION,
                   "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "unit_decisions": meta.get("unit_decisions", {}),
                   "decel_hypothesis_counts": meta.get("decel_hypothesis_counts", {}),
                   "median_start_spacing_s": meta.get("median_start_spacing_s"),
                   "pooled_means": meta.get("pooled_means", {}),
                   **meta.get("canaries", {})}
        out = EVAL_DIR / "harness_canaries.json"
        with open(out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        log(f"\n  --canaries-only: wrote {out}; the eval cache was not touched.")
        return

    tiers = TIER_ORDER if args.tiers == "both" else [args.tiers]
    all_rows, env_rows, dist_chunks = [], [], []
    for tier in tiers:
        rows, env, dist = run_tier(tier, master, meta, manifests, K_ENVELOPE)
        all_rows.extend(rows)
        env_rows.extend(env)
        dist_chunks.extend(dist)

    write_outputs(all_rows, env_rows, dist_chunks, meta, reveal=args.reveal)
    log(f"\nDone in {time.time() - t0:.1f} s. Figures read only "
        f"{EVAL_DIR} — nothing is recomputed at render time.")


if __name__ == "__main__":
    main()