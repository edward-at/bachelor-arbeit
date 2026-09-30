"""
bias_tier2_common.py  —  shared constants, loaders and metric math for the
Tier-2 bias-quantification scripts (Chapter 10, §4.7).

Why this module exists: five Tier-2 scripts consume the same canonical
artifact (cluster_assignments.parquet) and the same metric definitions.
Duplicating the share-pivot / Hill / JS / PSI code five times is exactly the
divergence risk the compute-once architecture forbids, so it lives here once.

Position in the architecture:
    dataset/regime_data/cluster_assignments.parquet   (canonical, Chapter 9)
    dataset/regime_data/bias_per_dataset.parquet      (Tier-1, bias_quant.py)
    dataset/std_data/<ds>.parquet                     (windowed sigma layer)
                |
        bias_tier2_common.py  (this file: loaders + metric functions)
                |
    bias_diversity.py / bias_volume.py / bias_bootstrap.py /
    bias_fingerprint.py / bias_pca_map.py
                |
    dataset/regime_data/*.parquet  +  bias_quantification/figures/*

Schema alignment (verified against bias_quant.py, 2026-07-03):
  * ARTIFACT_DIR is dataset/regime_data — the SAME home as the Tier-1
    bias_* artifacts. Tier-2 artifacts join them there.
  * JS convention: Jensen-Shannon DISTANCE, base-2 logs, bounded [0, 1] —
    IDENTICAL to bias_quant.js_distance_to_reference (jensenshannon base=2).
    Recomputed values are therefore bit-comparable to `js_to_pooled`.
  * Shared metric names follow Tier-1 exactly: `low_dynamic_share`,
    `js_to_pooled`. One metric, one name, everywhere.
  * W1 columns in bias_per_dataset.parquet are `w_std_speed_kmh`,
    `w_std_acc_mag`, `w_std_yaw_rate` (yaw already in deg/s). The std_data
    fallback below reproduces bias_quant.wasserstein_per_channel exactly
    (all std_data windows — non-overlapping since 2026-08-29 — pooled includes self, NaN-drop only).
  * Equivalences to Tier-1 columns (disclosed in §10.4 of the note):
    Hill D1 = N_REGIMES ** entropy_norm  and  Hill D2 = 1 / hhi.
    cross_check_tier1() verifies all of this at runtime when the Tier-1
    artifact is present.

PSI caveat: 1e-4 floor on zero cells (industry convention). A dataset with a
genuinely empty regime is floor-dominated and exceeds the 0.25 "major shift"
threshold by construction — treated as a correct diagnosis (a missing regime
IS a major shift), not a bug.
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

try:
    from tqdm import tqdm
except Exception:  # tqdm is optional -> degrade to a no-op
    def tqdm(x, **kwargs):
        return x

from tum_colors import TUM_PRIMARY, TUM_SECONDARY, TUM_ACCENT, TUM_EXT

# Single sources of truth (same imports the Tier-1 figure scripts use):
from regime_style import N_REGIMES, LOW_DYNAMIC_REGIMES
from bias_quant import CONFOUNDED_ACCEL   # bias_quant is __main__-guarded

# ────────────────────────────── paths ──────────────────────────────
PROJECT_ROOT  = Path("/Users/edward/Downloads")
ARTIFACT_DIR  = PROJECT_ROOT / "dataset" / "regime_data"     # Tier-1 + Tier-2 artifacts
STD_DATA_DIR  = PROJECT_ROOT / "dataset" / "std_data"
FIG_ROOT      = PROJECT_ROOT / "bachelor-arbeit" / "06_bias_quantification" / "figures"

REGIME_ASSIGN      = ARTIFACT_DIR / "cluster_assignments.parquet"
TIER1_PER_DATASET  = ARTIFACT_DIR / "bias_per_dataset.parquet"

# ───────────────────────── display names ───────────────────────────
DISPLAY_NAMES = {
    "boreas_rt": "Boreas", "durlar": "DurLAR", "ford": "Ford", "kitti": "KITTI",
    "kitti360": "KITTI-360", "nuscene": "nuScenes", "nvidia": "NVIDIA",
    "oxford": "Oxford", "waymo": "Waymo", "zod": "ZOD",
    "4seasons": "4Seasons", "hdd": "HDD", "a2d2": "A2D2", "kitscenes": "KITScenes",
}

DATASET_COLORS = {
    "4seasons":  TUM_SECONDARY["blue2"],
    "a2d2":      TUM_SECONDARY["dark_gray"],
    "hdd":       TUM_SECONDARY["gray"],
    "kitscenes": TUM_PRIMARY["black"],
    "boreas_rt": TUM_EXT["maroon"],
    "durlar":    TUM_SECONDARY["blue1"],
    "ford":      TUM_EXT["orange"],
    "kitti":     TUM_EXT["yellow"],
    "kitti360":  TUM_EXT["olive"],
    "nuscene":   TUM_EXT["dark_green"],
    "nvidia":    TUM_EXT["teal"],
    "oxford":    TUM_EXT["navy"],
    "waymo":     TUM_EXT["purple"],
    "zod":       TUM_EXT["red"],
}
DATASET_COLOR_FALLBACK = TUM_SECONDARY["blue2"]

# ───────────────────────── constants ───────────────────────────────
RAD2DEG = 180.0 / np.pi
DPI = 300

JS_BASE = 2            # matches bias_quant.JS_BASE (JS DISTANCE, bounded [0,1])
PSI_EPS = 1e-4         # zero-cell floor, standard industry convention

# Wasserstein channels — DC5 (the three Chapter-8 ECDF channels, window grain)
W1_CHANNELS = ["std_speed_kmh", "std_acc_mag", "std_yaw_rate"]

# tier-2 internal key -> Tier-1 column in bias_per_dataset.parquet
W1_COLMAP = {
    "w1_speed": "w_std_speed_kmh",
    "w1_acc":   "w_std_acc_mag",
    "w1_yaw":   "w_std_yaw_rate",
}


# ══════════════════════════ small helpers ══════════════════════════
def label(ds):
    """Thesis display name, with the Tier-1 fidelity asterisk where due."""
    name = DISPLAY_NAMES.get(ds, ds)
    return name + "*" if ds in CONFOUNDED_ACCEL else name


def color(ds):
    return DATASET_COLORS.get(ds, DATASET_COLOR_FALLBACK)


# ═══════════════════════ canonical loaders ═════════════════════════
def load_assignments(columns=("dataset", "clip_id", "regime")):
    """Read the canonical window-regime artifact (402,421 x 20 s windows; derived, see canonical.py)."""
    if not REGIME_ASSIGN.exists():
        raise SystemExit(f"Canonical artifact not found: {REGIME_ASSIGN}")
    return pd.read_parquet(REGIME_ASSIGN, columns=list(columns))


def regime_counts(df=None):
    """Per-dataset regime count matrix (n_datasets x N_REGIMES), datasets
    sorted alphabetically. Same construction as bias_quant.regime_share_matrix."""
    if df is None:
        df = load_assignments()
    pivot = (df.groupby(["dataset", "regime"]).size()
               .unstack(fill_value=0)
               .reindex(columns=range(N_REGIMES), fill_value=0)
               .sort_index())
    return list(pivot.index), pivot.to_numpy(dtype=float)


def share_rows(counts):
    """Row-normalise a count matrix into per-dataset regime shares."""
    tot = counts.sum(axis=1, keepdims=True)
    return counts / np.where(tot > 0, tot, np.nan)


def pooled_shares(counts):
    """Window-weighted (pooled) reference — matches bias_quant.pooled_profile."""
    tot = counts.sum(axis=0)
    return tot / tot.sum()


def balanced_shares(counts):
    """Equal-dataset-weight reference — matches bias_quant.balanced_profile."""
    return share_rows(counts).mean(axis=0)


# ═══════════════════════════ metric math ═══════════════════════════
def hill(p, q):
    """Hill number D_q — the 'effective number of regimes' of order q.

    D_0 = richness, D_1 = exp(Shannon entropy), D_2 = 1/Simpson.
    Equivalences to Tier-1 columns: D_1 = N ** entropy_norm, D_2 = 1 / hhi.
    """
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return np.nan
    if q == 0:
        return float(p.size)
    if q == 1:
        return float(np.exp(-np.sum(p * np.log(p))))
    return float(np.sum(p ** q) ** (1.0 / (1.0 - q)))


def js_distance(p, q):
    """Jensen-Shannon DISTANCE, base 2 — identical convention to bias_quant."""
    return float(jensenshannon(p, q, base=JS_BASE))


def psi(p, ref, eps=PSI_EPS):
    """Population Stability Index of p against ref, with a zero-cell floor.

    Conventional reading: < 0.10 stable, 0.10-0.25 moderate, > 0.25 major shift.
    """
    p = np.clip(np.asarray(p, dtype=float), eps, None)
    r = np.clip(np.asarray(ref, dtype=float), eps, None)
    p, r = p / p.sum(), r / r.sum()
    return float(np.sum((p - r) * np.log(p / r)))


def gini(x):
    """Gini coefficient of a non-negative vector (0 = equal, 1 = one holder).

    NAMING RULE: in Chapter 10 prose this is always the qualified
    "volume-share Gini". The unqualified word "Gini" is reserved for the
    exposé §3.7 dynamic-content Gini (see bias_quant.py D4 note / DC10).
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n == 0 or x.sum() <= 0:
        return np.nan
    i = np.arange(1, n + 1)
    return float((2.0 * np.sum(i * x)) / (n * np.sum(x)) - (n + 1.0) / n)


# ═════════════════ continuous channels (W1 fallback) ═══════════════
def _read_std(ds):
    df = pd.read_parquet(STD_DATA_DIR / f"{ds}.parquet")
    out = {}
    for ch in W1_CHANNELS:
        if ch not in df.columns:
            out[ch] = np.empty(0)
            continue
        v = df[ch].to_numpy(dtype=float)
        if ch == "std_yaw_rate":
            v = v * RAD2DEG          # rad/s -> deg/s, matching bias_quant
        out[ch] = v[np.isfinite(v)]
    return ds, out


def load_std_channels(datasets):
    """Threaded read of the three DC5 channels for every dataset."""
    data = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for ds, chans in tqdm(ex.map(_read_std, datasets), total=len(datasets),
                              desc="std_data", leave=False):
            data[ds] = chans
    return data


def w1_vs_pooled(datasets):
    """1-Wasserstein per DC5 channel vs the pooled window set — reproduces
    bias_quant.wasserstein_per_channel (all std_data windows, non-overlapping, pooled
    includes the dataset itself, yaw in deg/s). Fallback path only."""
    data = load_std_channels(datasets)
    pooled = {ch: np.concatenate([data[d][ch] for d in datasets]) for ch in W1_CHANNELS}
    rows = {}
    for ds in tqdm(datasets, desc="wasserstein", leave=False):
        rows[ds] = {ch: (wasserstein_distance(data[ds][ch], pooled[ch])
                         if data[ds][ch].size else np.nan)
                    for ch in W1_CHANNELS}
    return rows


# ══════════════════ the Tier-2 metric table builder ════════════════
def build_bias_metric_table(verbose=True):
    """One row per dataset, every Chapter-10 scalar metric as a column.

    Categorical metrics (shares, Hill, JS, PSI) are recomputed from the
    canonical assignment artifact — deterministic, so identical to Tier-1
    by construction (verified via cross_check_tier1). W1 is read from the
    Tier-1 artifact when present and recomputed from std_data otherwise.
    """
    df = load_assignments()
    datasets, counts = regime_counts(df)
    shares = share_rows(counts)
    pooled = pooled_shares(counts)
    n_win = counts.sum(axis=1)
    low_idx = list(LOW_DYNAMIC_REGIMES)

    tab = pd.DataFrame(index=datasets)
    tab["n_windows"] = n_win.astype(int)
    tab["vol_share"] = n_win / n_win.sum()
    tab["low_dynamic_share"] = shares[:, low_idx].sum(axis=1)   # Tier-1 name
    tab["hill_d0"] = [hill(s, 0) for s in shares]
    tab["hill_d1"] = [hill(s, 1) for s in shares]
    tab["hill_d2"] = [hill(s, 2) for s in shares]
    tab["js_to_pooled"] = [js_distance(s, pooled) for s in shares]  # Tier-1 name
    tab["psi_pooled"] = [psi(s, pooled) for s in shares]

    # ---- W1: Tier-1 artifact first, canonical recompute as fallback ----
    got_tier1 = False
    if TIER1_PER_DATASET.exists():
        t1 = pd.read_parquet(TIER1_PER_DATASET)
        if all(c in t1.columns for c in W1_COLMAP.values()):
            for k, c in W1_COLMAP.items():
                tab[k] = t1[c].reindex(tab.index).to_numpy(dtype=float)
            got_tier1 = True
            if verbose:
                print(f"[tier2] W1 columns read from Tier-1 artifact: {TIER1_PER_DATASET}")
            cross_check_tier1(tab, t1, verbose=verbose)
        elif verbose:
            print(f"[tier2] Tier-1 artifact found but W1 columns "
                  f"{list(W1_COLMAP.values())} missing -> recomputing from std_data")
    elif verbose:
        print(f"[tier2] Tier-1 artifact not found at {TIER1_PER_DATASET} "
              f"-> recomputing W1 from std_data")

    if not got_tier1:
        w1 = w1_vs_pooled(datasets)
        tab["w1_speed"] = [w1[d]["std_speed_kmh"] for d in datasets]
        tab["w1_acc"]   = [w1[d]["std_acc_mag"]   for d in datasets]
        tab["w1_yaw"]   = [w1[d]["std_yaw_rate"]  for d in datasets]
    return tab


def cross_check_tier1(tab, t1, verbose=True):
    """Verify Tier-2 recomputation against the Tier-1 artifact:
      low_dynamic_share, js_to_pooled  -> should match to float precision;
      hill_d1 vs N**entropy_norm, hill_d2 vs 1/hhi -> exact equivalences."""
    t1 = t1.reindex(tab.index)
    checks = {}
    if "low_dynamic_share" in t1.columns:
        checks["low_dynamic_share"] = np.nanmax(
            np.abs(tab["low_dynamic_share"].values - t1["low_dynamic_share"].values))
    if "js_to_pooled" in t1.columns:
        checks["js_to_pooled"] = np.nanmax(
            np.abs(tab["js_to_pooled"].values - t1["js_to_pooled"].values))
    if "entropy_norm" in t1.columns:
        checks["hill_d1 vs N**entropy_norm"] = np.nanmax(
            np.abs(tab["hill_d1"].values - N_REGIMES ** t1["entropy_norm"].values))
    if "hhi" in t1.columns:
        checks["hill_d2 vs 1/hhi"] = np.nanmax(
            np.abs(tab["hill_d2"].values - 1.0 / t1["hhi"].values))
    if verbose and checks:
        print("[tier2] cross-check vs Tier-1 (max |diff|; expect ~1e-12):")
        for k, v in checks.items():
            flag = "" if v < 1e-6 else "   <-- INVESTIGATE"
            print(f"         {k}: {v:.3e}{flag}")
    return checks