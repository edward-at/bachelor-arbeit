#!/usr/bin/env python3
"""general_info.py — every thesis-relevant number, from artifacts, in one file.

Place in bachelor-arbeit/ and run from there:

    python general_info.py

Prints to stdout AND writes general_info.txt next to itself. Upload the .txt
to the Claude Project as the numeric ground truth for all writing chats.

RULES THIS SCRIPT OBEYS
  * Read-only. It recomputes nothing statistical; it reads artifacts the
    closed analysis already produced. The only arithmetic is joins, counts,
    and unit conversion for display.
  * Every section is independently guarded: a missing artifact prints
    MISSING with the path, never a traceback.
  * Numbers that exist ONLY in run logs (never exported) are inventoried in
    the final section instead of being silently absent or hard-coded.
"""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("GENERAL_INFO_ROOT",
                                   "/Users/edward/Downloads"))
BA           = PROJECT_ROOT / "bachelor-arbeit"
DATA         = PROJECT_ROOT / "dataset"
STD_DIR      = DATA / "std_data"
REGIME_DIR   = DATA / "regime_data"
SUBSET_DIR   = DATA / "subset_data"
EVAL_DIR     = SUBSET_DIR / "eval"
TIER_DIR     = SUBSET_DIR / "tier_n71390"
STANDARDIZATION = BA / "03_standardization"
EXPLORATORY  = BA / "04_exploratory_analysis"
CURATION     = BA / "07_subset_curation"

OUT_TXT = Path(__file__).resolve().parent / "general_info.txt"

DATASETS = ["4seasons", "a2d2", "boreas_rt", "durlar", "ford", "hdd",
            "kitscenes", "kitti", "kitti360", "nuscene", "nvidia", "oxford",
            "waymo", "zod"]

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 300)
pd.set_option("display.float_format", lambda v: f"{v:.4f}")

_LINES: list[str] = []
def emit(*args):
    s = " ".join(str(a) for a in args)
    print(s)
    _LINES.append(s)

_SECTIONS = []
def section(title):
    def deco(fn):
        _SECTIONS.append((title, fn)); return fn
    return deco

def frame(df):        # print a DataFrame through emit
    for line in df.to_string().splitlines():
        emit(line)

def read_parquet(p, **kw):
    return pd.read_parquet(p, **kw)

def rows_only(p):
    """Row count without loading the table."""
    import pyarrow.parquet as pq
    return pq.ParquetFile(p).metadata.num_rows

# ===========================================================================
@section("0  provenance — snapshot, paths, timestamp")
def s0():
    emit(f"generated: {datetime.now():%Y-%m-%d %H:%M}")
    emit(f"project root: {PROJECT_ROOT}")
    snap = DATA / "canonical_snapshot.json"
    if snap.exists():
        emit(f"\ncanonical_snapshot.json ({datetime.fromtimestamp(snap.stat().st_mtime):%Y-%m-%d}):")
        emit(json.dumps(json.loads(snap.read_text()), indent=1))
    else:
        emit("canonical_snapshot.json: MISSING — run canonical.py --freeze")

@section("1  canonical corpus constants (derived live via canonical.py)")
def s1():
    sys.path.insert(0, str(CURATION))
    from canonical import C
    emit(f"frame_n (assign set)      : {C.frame_n:,}")
    emit(f"guarded_n                 : {getattr(C, 'guarded_n', 'see snapshot'):,}"
         if hasattr(C, "guarded_n") else "guarded_n                 : see snapshot section 0")
    emit(f"regime supplies           : {C.regime_supplies}")
    emit(f"budget N (rarest supply)  : {C.budget_n:,}")
    emit(f"capability N (5x rarest)  : {C.budget_n * 5:,}")
    emit(f"frame-to-budget ratio     : {C.frame_n / C.budget_n:.3f}")
    emit(f"event reserve (0.6 g)     : {C.event_reserve_n:,}")
    emit(f"evasive population (D24)  : {C.evasive_n:,}")
    emit(f"confounded-accel sources  : {sorted(C.confounded)}")

@section("2  standardization contract (clip metadata + script constants)")
def s2():
    import pyarrow.parquet as pq
    clip = DATA / "inertial_data" / "kitti" / "clip_0001.parquet"
    if clip.exists():
        md = pq.ParquetFile(clip).schema_arrow.metadata or {}
        pm = json.loads(md[b"processing_meta"].decode())
        keep = ["pipeline_version", "grid_hz", "f_cut_hz", "butter_order",
                "hampel_window_s", "hampel_t0", "sg_poly", "w_d1", "w_d2",
                "edge_mask_frames", "gap_max_s", "speed_max_kmh_gate",
                "a_max_ms2_gate", "j_max_ms3_gate", "yaw_rate_max_degs_gate",
                "gate_close_s", "min_valid_frac_clip", "gate_frac_drop"]
        for k in keep:
            emit(f"  {k:<24}: {pm.get(k)}")
    else:
        emit(f"  sample clip MISSING: {clip}")
    cs = STANDARDIZATION / "compute_std.py"
    if cs.exists():
        t = cs.read_text()
        for name in ("WINDOW_S", "STEP_S", "GRID_HZ", "MIN_VALID_FRAC"):
            m = re.search(rf"^{name}\s*=\s*([0-9.]+)", t, re.M)
            if m: emit(f"  {name:<24}: {m.group(1)}")
    else:
        emit(f"  compute_std.py MISSING at {cs}")

@section("3  coverage / audit baseline — the S7.3 numbers (post-cleaning)")
def s3():
    p = EXPLORATORY / "audit_baseline.json"
    if not p.exists():
        emit(f"MISSING: {p}"); return
    d = json.load(open(p))
    df = pd.DataFrame(d).T
    cols = ["clips", "frames", "gated_pct", "speed_src", "yaw_rate_src",
            "mean_speed", "max_kmh", "n_spd_108", "version"]
    frame(df[cols])
    emit(f"\ntotals: clips {int(df.clips.sum()):,} | frames {int(df.frames.sum()):,}")
    emit("\nfull blocks for the five S7.3 case studies:")
    for ds in ("a2d2", "4seasons", "hdd", "durlar", "oxford"):
        if ds in d:
            emit(f"--- {ds}"); emit(json.dumps(d[ds], indent=1))

@section("4  window layer — std_data counts and the two complete-case counts")
def s4():
    total = 0
    for ds in DATASETS:
        p = STD_DIR / f"{ds}.parquet"
        n = rows_only(p) if p.exists() else 0
        total += n
        emit(f"  {ds:<11}{n:>9,}")
    emit(f"  {'TOTAL':<11}{total:>9,}   (expected 406,130)")
    FEATS6 = ["mean_speed_kmh", "std_speed_kmh", "mean_acc_pos",
              "mean_acc_neg", "mean_abs_yaw_rate", "mean_abs_lat_acc"]
    import pyarrow.parquet as pq
    dfs = []
    for ds in DATASETS:
        p = STD_DIR / f"{ds}.parquet"
        if p.exists():
            names = set(pq.ParquetFile(p).schema_arrow.names)
            cols = [c for c in FEATS6 if c in names]
            dfs.append(read_parquet(p, columns=cols))
    if dfs:
        big = pd.concat(dfs, ignore_index=True)
        f6 = [c for c in FEATS6 if c in big.columns]
        f5 = [c for c in f6 if c != "mean_abs_lat_acc"]
        emit(f"\n  six-feature complete-case (gate population) : "
             f"{int(big[f6].notna().all(axis=1).sum()):,}  (expected 402,293)")
        emit(f"  five-feature complete-case (assign set)     : "
             f"{int(big[f5].notna().all(axis=1).sum()):,}  (expected 402,421)")
        emit("  reason for the difference: dropping |aLat| readmits windows")
        emit("  whose lateral channel is missing (127 KITScenes + 1 Waymo).")

@section("5  tier thresholds (D-5 frozen conventions) + pooled ECDF/tier shares")
def s5():
    rc = EXPLORATORY / "regime_classification.py"
    trip = {}
    if rc.exists():
        for m in re.finditer(r'"(std_\w+)":\s*\(([\d.]+),\s*([\d.]+),\s*([\d.]+)\)',
                             rc.read_text()):
            trip[m.group(1)] = tuple(float(m.group(i)) for i in (2, 3, 4))
    emit(f"  thresholds (t1/t2/t3): {trip or 'regime_classification.py MISSING'}")
    if not trip: return
    use = {"std_speed_kmh": 1.0, "std_acc_mag": 1.0,
           "std_yaw_rate": 180.0 / np.pi}          # rad/s -> deg/s for display
    dfs = [read_parquet(STD_DIR / f"{ds}.parquet",
                        columns=list(use)) for ds in DATASETS
           if (STD_DIR / f"{ds}.parquet").exists()]
    big = pd.concat(dfs, ignore_index=True)
    for ch, (t1, t2, t3) in trip.items():
        v = (big[ch] * use[ch]).dropna().to_numpy()
        e = [float((v <= t).mean()) for t in (t1, t2, t3)]
        sh = [e[0], e[1] - e[0], e[2] - e[1], 1 - e[2]]
        emit(f"  {ch:<15} ECDF@t: {e[0]:.3f}/{e[1]:.3f}/{e[2]:.3f}"
             f"   tier shares NS/LD/MD/HD %: "
             f"{sh[0]*100:.2f}/{sh[1]*100:.2f}/{sh[2]*100:.2f}/{sh[3]*100:.2f}")

@section("6  feature gate — correlations, VIF, locked vector")
def s6():
    for name in ("feature_gate_stats", "feature_gate_vector"):
        p = REGIME_DIR / f"{name}.parquet"
        if p.exists(): emit(f"--- {name}"); frame(read_parquet(p))
        else: emit(f"{name}: MISSING")

@section("7  clustering — PCA variance, k-selection, profile, supplies")
def s7():
    p = REGIME_DIR / "pca_variance.parquet"
    if p.exists(): emit("--- pca_variance"); frame(read_parquet(p))
    p = REGIME_DIR / "kselect_curves.parquet"
    if p.exists():
        k = read_parquet(p); emit("--- k-selection sweep"); frame(k)
        for col, best in (("silhouette", "idxmax"), ("gap", "idxmax")):
            if col in k.columns:
                emit(f"  {col} argbest -> k = {int(k.loc[getattr(k[col], best)(), 'k'])}")
    p = REGIME_DIR / "cluster_profile.parquet"
    if p.exists(): emit("--- regime profile (centroids in physical units)"); frame(read_parquet(p))
    p = REGIME_DIR / "cluster_assignments.parquet"
    if p.exists():
        a = read_parquet(p, columns=["regime"])
        vc = a.regime.value_counts().sort_index()
        emit("--- supplies from cluster_assignments")
        for r, n in vc.items():
            emit(f"  regime {r}: {n:,}  ({n/len(a)*100:.2f} %)")
        emit(f"  total: {len(a):,}")

@section("8  stability battery — caps, reseed, resample, validity, jaccard, consensus")
def s8():
    for name in ("cap_sensitivity_caps", "cap_sensitivity_rank",
                 "cap_sensitivity_drift", "reseed_stability",
                 "resample_stability", "validity_indices",
                 "bootstrap_jaccard"):
        p = REGIME_DIR / f"{name}.parquet"
        if p.exists(): emit(f"--- {name}"); frame(read_parquet(p))
        else: emit(f"{name}: MISSING")
    p = REGIME_DIR / "consensus_matrix.parquet"
    if p.exists():
        m = read_parquet(p).to_numpy()
        emit(f"--- consensus: diagonal mean {np.diag(m).mean():.3f} | "
             f"off-diagonal mean {m[~np.eye(len(m), dtype=bool)].mean():.3f}")

@section("9  GMM cross-check + HMM temporal structure")
def s9():
    p = REGIME_DIR / "gmm_selection.parquet"
    if p.exists(): emit("--- gmm_selection (BIC/AIC sweep)"); frame(read_parquet(p))
    p = REGIME_DIR / "hmm_transitions.parquet"
    if p.exists():
        t = read_parquet(p); emit("--- hmm transition matrix"); frame(t)
        m = t.to_numpy()
        if m.shape[0] == m.shape[1]:
            emit(f"  self-transitions: {[round(float(x),3) for x in np.diag(m)]}")
    p = REGIME_DIR / "hmm_durations.parquet"
    if p.exists(): emit("--- hmm dwell durations (s)"); frame(read_parquet(p))

@section("10  bias tier 1 — per-dataset shares, profiles, corpus scalars")
def s10():
    for name in ("bias_per_dataset", "bias_corpus_profiles",
                 "bias_corpus_scalars", "bias_sparse_regimes"):
        p = REGIME_DIR / f"{name}.parquet"
        if p.exists(): emit(f"--- {name}"); frame(read_parquet(p))
        else: emit(f"{name}: MISSING")

@section("11  bias tier 2 — diversity, volume, bootstrap, entropy, shape, supply, fingerprint")
def s11():
    for name in ("bias_diversity", "bias_volume", "bias_bootstrap_ci",
                 "bias_gmm_entropy", "bias_shape", "bias_regime_supply",
                 "bias_fingerprint", "bias_metric_agreement"):
        p = REGIME_DIR / f"{name}.parquet"
        if not p.exists(): emit(f"{name}: MISSING"); continue
        df = read_parquet(p); emit(f"--- {name}"); frame(df)
        if name == "bias_volume" and "vol_share" in df.columns:
            s = np.sort(df["vol_share"].to_numpy()); n = len(s)
            gini = float((2*np.arange(1, n+1) - n - 1) @ s / (n * s.sum()))
            d1 = float(np.exp(-(s[s > 0] * np.log(s[s > 0])).sum()))
            d2 = float(1.0 / (s @ s))
            emit(f"  derived: Gini {gini:.3f} | D1 {d1:.2f} | D2 {d2:.2f} of {n}")

@section("12  sampling construction — supplies, allocations, method slates")
def s12():
    for label, p in (("supply_table", SUBSET_DIR / "supply_table.csv"),
                     ("allocation capability", SUBSET_DIR / "allocation_table.csv"),
                     ("allocation budget", TIER_DIR / "allocation_table.csv")):
        if p.exists(): emit(f"--- {label}"); frame(pd.read_csv(p))
        else: emit(f"{label}: MISSING ({p})")
    for label, p in (("capability manifest", SUBSET_DIR / "manifest.json"),
                     ("budget manifest", TIER_DIR / "manifest.json")):
        if p.exists():
            d = json.load(open(p))
            emit(f"--- {label}: N={d.get('N', d.get('n', '?'))}, "
                 f"methods={sorted(d.get('methods', {}).keys()) if isinstance(d.get('methods'), dict) else d.get('methods')}")

@section("13  evaluation — master tables (both tiers), envelope, run meta")
def s13():
    for label, p in (("master_table_budget", EVAL_DIR / "master_table_budget.csv"),
                     ("master_table_capability", EVAL_DIR / "master_table_capability.csv"),
                     ("tautology_map", EVAL_DIR / "tautology_map.csv")):
        if p.exists():
            emit(f"--- {label}"); frame(pd.read_csv(p, index_col=0))
        else: emit(f"{label}: MISSING")
    p = EVAL_DIR / "srs_envelope_budget.parquet"
    if p.exists():
        env = read_parquet(p)
        emit("--- SRS envelope (budget): per-metric quantiles")
        num = env.select_dtypes("number")
        frame(num.describe(percentiles=[0.025, 0.5, 0.975]).T[
            ["mean", "2.5%", "50%", "97.5%"]])
    for name in ("run_meta.json", "frame_pass_meta.json"):
        p = EVAL_DIR / name
        if p.exists(): emit(f"--- {name}"); emit(json.dumps(json.load(open(p)), indent=1))

@section("14  curation — gates re-applied, manifests, replication, reserve composition")
def s14():
    sys.path.insert(0, str(CURATION))
    import curation_common as cc
    emit(f"gates (D4): {cc.GATES}")
    emit(f"tie-break : {getattr(cc, 'TIE_BREAK', 'see curation_common')}")
    emit(f"winner    : {cc.WINNER} | N_EVENT_RESERVE {cc.N_EVENT_RESERVE:,} | "
         f"EXPECTED_EVASIVE {cc.EXPECTED_EVASIVE:,}")
    mp = EVAL_DIR / "master_table_budget.csv"
    if mp.exists():
        mt = pd.read_csv(mp, index_col=0)
        emit("\n--- gate verdicts recomputed from master_table_budget")
        for m in mt.index:
            verdicts = []
            for metric, (op, thr, *_ ) in cc.GATES.items():
                if metric not in mt.columns: continue
                v = mt.loc[m, metric]
                ok = (v <= thr) if op in ("<=", "le") else (v >= thr)
                verdicts.append(f"{metric}={v:.4f}:{'PASS' if ok else 'fail'}")
            emit(f"  {m:<11} " + "  ".join(verdicts))
    for label, p in (("budget curation_manifest", TIER_DIR / "curation_manifest.json"),
                     ("capability curation_manifest", SUBSET_DIR / "curation_manifest.json")):
        if not p.exists(): emit(f"{label}: MISSING"); continue
        d = json.load(open(p))
        emit(f"\n--- {label}: tier={d['tier']} N={d['N']:,} reserve={d['n_event_reserve']:,}")
        for cid, mm in d["methods"].items():
            pr = mm["params"].get("per_regime", {})
            caps = {r: (v.get("top_share"), v.get("cap_breached"),
                        v.get("n_forced")) for r, v in pr.items()}
            emit(f"  {cid}: dataset top = "
                 f"{max(mm['dataset_counts'], key=mm['dataset_counts'].get)} "
                 f"{max(mm['dataset_counts'].values()):,}; "
                 f"per-regime (top_share, breached, forced) = {caps}")
    p = EVAL_DIR / "curation_replication_meta.json"
    if p.exists():
        emit("\n--- replication meta"); emit(json.dumps(json.load(open(p)), indent=1))
    # reserve composition: assignments -> std bridge -> summaries
    try:
        a = read_parquet(REGIME_DIR / "cluster_assignments.parquet",
                         columns=["dataset", "clip_id", "window_start_s", "regime"])
        parts = []
        for ds in DATASETS:
            sp = STD_DIR / f"{ds}.parquet"
            if not sp.exists(): continue
            s = read_parquet(sp, columns=["clip_id", "window_start_s", "window_idx"])
            s["dataset"] = ds; parts.append(s)
        bridge = pd.concat(parts, ignore_index=True)
        su = read_parquet(SUBSET_DIR / "window_summaries.parquet",
                          columns=["dataset", "clip_id", "window_idx",
                                   "max_decel_ms2", "max_abs_yaw_degs"])
        j = (a.merge(bridge, on=["dataset", "clip_id", "window_start_s"])
              .merge(su, on=["dataset", "clip_id", "window_idx"]))
        guarded = j[j["max_abs_yaw_degs"] <= 90.0]
        thr = 0.6 * 9.81
        res = guarded[guarded["max_decel_ms2"] >= thr]
        emit(f"\n--- reserve recomputed: guarded {len(guarded):,} | "
             f"events >= {thr:.3f} m/s^2: {len(res):,}")
        emit("  by source:"); frame(res.dataset.value_counts().to_frame("events"))
        emit("  by regime:"); frame(res.regime.value_counts().sort_index().to_frame("events"))
    except Exception as e:
        emit(f"reserve composition recompute failed: {e}")

@section("15  numbers that exist ONLY in run logs (not exported) — source inventory")
def s15():
    emit("The following quoted values have no artifact and come from script")
    emit("stdout; re-run the named script to regenerate, or cite the register:")
    emit("  -4.96 m/s^2 planted-event recovery ......... pipeline_regression_test.py")
    emit("  +5.3 % rectangular-window comparison ....... pipeline_regression_test.py")
    emit("  GMM-vs-kmeans ARI 0.316 .................... gmm.py")
    emit("  'top 20 % carry 44 % of speed mass' ........ bias_shape_metrics.py")
    emit("  evasive retention table (C01 = 0.804) ...... curation_evassive_check.py")
    emit("  ex-confounded gate check (C01 0.5035) ...... curation_reveal.py")
    emit("  seed-ARI 0.996 / cap-reseed 0.986 line ..... clustering.py")
    emit("  gate/assign complete-case reconciliation ... recomputed in section 4 here")

# ===========================================================================
def main() -> int:
    emit("=" * 78)
    emit("GENERAL INFO — thesis numeric ground truth (derived from artifacts)")
    emit("=" * 78)
    for title, fn in _SECTIONS:
        emit("\n" + "=" * 78); emit(f"[{title}]"); emit("=" * 78)
        try:
            fn()
        except SystemExit as e:
            emit(f"SECTION ABORTED (SystemExit): {e}")
        except Exception:
            emit("SECTION FAILED (continuing):")
            for line in traceback.format_exc().splitlines()[-3:]:
                emit("  " + line)
    OUT_TXT.write_text("\n".join(_LINES) + "\n")
    print(f"\nwritten: {OUT_TXT}")
    return 0

if __name__ == "__main__":
    sys.exit(main())