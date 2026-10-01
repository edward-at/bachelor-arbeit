"""
bias_quant.py  —  Section 4.8 Tier-1 bias quantification (COMPUTE stage)
════════════════════════════════════════════════════════════════════════════
Compute-once → save-to-parquet → read-in-figures. This script does ALL the
number-crunching for Section 4.8; every figure script just reads its outputs.

INPUTS
  • ARTIFACT_DIR/cluster_assignments.parquet   (Section 4.7 output; the spine)
        cols: dataset, clip_id, window_start_s, window_end_s, regime, pc1,pc2,pc3
        one row per NON-OVERLAPPING 20 s window, tagged with dataset + regime.
  • STD_DATA_DIR/<dataset>.parquet             (windowed signals, Section 4.5)
        the three ECDF channels: std_speed_kmh, std_acc_mag, std_yaw_rate.

OUTPUTS  (parquet only, ARTIFACT_DIR)
  • bias_per_dataset.parquet        one row per dataset — shares + every scalar
  • bias_corpus_profiles.parquet    pooled / balanced / dataset_std regime vectors
  • bias_corpus_scalars.parquet     volume-gap, mean pairwise JS, counts
  • bias_sparse_regimes.parquet     long (dataset×regime) under-fill table for RQ3

METRIC PROVENANCE
  regime shares / entropy / JS / directional / sparse  ← Section 4.7 (regimes)
  per-channel Wasserstein                              ← Section 4.6 (distributions)

REFERENCES  (each metric, its own scoped reference)
  directional low-dynamic share : NONE (absolute, self-contained property)
  normalized Shannon entropy     : UNIFORM (entropy maximal at uniform)
  Jensen–Shannon distance        : POOLED empirical regime distribution
  per-channel Wasserstein        : POOLED empirical signal distribution

Nothing here is stochastic, so seed 42 is irrelevant to Tier 1 (kept in the
project only for the clustering stage). Reproducibility = code + fixed inputs.
════════════════════════════════════════════════════════════════════════════
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from regime_style import N_REGIMES, LOW_DYNAMIC_REGIMES

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  (data under DATA_ROOT, figures under REPO_ROOT, see thesis_paths.py)
# ─────────────────────────────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parents[1]))
from thesis_paths import DATA_ROOT, REPO_ROOT  # noqa: E402
ARTIFACT_DIR = DATA_ROOT / "regime_data"
STD_DATA_DIR = DATA_ROOT / "std_data"

ASSIGN_FILE = ARTIFACT_DIR / "cluster_assignments.parquet"

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SHARE_COLS = [f"share_{i}" for i in range(N_REGIMES)]
UNIFORM = np.full(N_REGIMES, 1.0 / N_REGIMES)
JS_BASE = 2  # bounds Jensen–Shannon DISTANCE to [0, 1]

RAD_TO_DEG = 180.0 / np.pi
# The three ECDF channels (Section 4.6.8) the Wasserstein metric quantifies.
# std_yaw_rate is stored in rad/s → convert to °/s on load (as ecdf_analysis.py does).
WASSERSTEIN_CHANNELS = ["std_speed_kmh", "std_acc_mag", "std_yaw_rate"]

# Population for the Wasserstein channels. "all" = every std_data window
# (the full non-overlapping std_data layer), reproducing the ECDFs of
# Section 4.6.8 exactly.
WASSERSTEIN_POPULATION = "all_windows"

# ── MEASUREMENT-FIDELITY FLAG (single source of truth, Sections 4.4.5 and 4.8.2)
# a_long is derived uniformly as the SG-derivative of speed (standardize
# pipeline, W_D1 = 0.7 s). Accel fidelity therefore equals SPEED-SOURCE
# fidelity, and the 14-dataset audit (frame-level a_long p1/p99) found three
# tiers:
#   ATTENUATED  a2d2      (±0.6)   — speed = ECU bus wheel-speed (pre-smoothed,
#                                    quantized) → accel compressed → regimes
#                                    Highway/Braking/Accelerating unreachable
#   PLAUSIBLE   11 datasets (±1.3–2.4)
#   INFLATED    4seasons  (±8.9), durlar (±6.4–8.2) — no speed channel; speed
#                                    position-derived (SLAM/GNSS poses, GPS
#                                    lat/lon) → a_long = 2nd derivative of
#                                    noisy position → rectified noise inflates
#                                    mean_acc_pos/neg → Braking/Accelerating
#                                    over-filled
# Regime-based metrics for flagged datasets are confounded by signal fidelity,
# not behaviour. Kept in ALL computations (completeness); figures mark them
# with an asterisk; rankings/headline claims must exclude them (§6.6).
CONFOUNDED_ACCEL = {"a2d2", "4seasons", "durlar"}


# ═════════════════════════════════════════════════════════════════════════════
# PER-DATASET REGIME METRICS  (all from cluster_assignments, Section 4.7)
# ═════════════════════════════════════════════════════════════════════════════
def regime_share_matrix(assign: pd.DataFrame) -> pd.DataFrame:
    """14×5 share matrix: cell = fraction of a dataset's windows in each regime.
    Missing regimes for a dataset are filled with 0 (single-scene datasets need
    not occupy all five). Returns share_0..share_{N-1} + n_windows, per dataset."""
    counts = (
        assign.groupby(["dataset", "regime"]).size()
        .unstack("regime")
        .reindex(columns=range(N_REGIMES))
        .fillna(0.0)
    )
    n = counts.sum(axis=1)
    shares = counts.div(n, axis=0)
    shares.columns = SHARE_COLS
    shares["n_windows"] = n.astype(int)
    return shares


def directional_low_dynamic_share(shares: pd.DataFrame) -> pd.Series:
    """RQ1 HEADLINE. Fraction of a dataset's windows in the two lowest-intensity
    regimes (Calm + Highway). Reference-free — an absolute property, not a
    distance. Concentration indices are direction-blind; this is not."""
    cols = [f"share_{i}" for i in LOW_DYNAMIC_REGIMES]
    return shares[cols].sum(axis=1)


def normalized_entropy(shares: pd.DataFrame) -> pd.Series:
    """Shannon entropy of the regime-share vector, ÷ ln(N) → [0,1].
    1 = mass spread evenly across regimes; 0 = collapsed into one.
    Implicit reference = UNIFORM (that is where entropy is maximal)."""
    P = shares[SHARE_COLS].values
    with np.errstate(divide="ignore", invalid="ignore"):
        logP = np.where(P > 0, np.log(P), 0.0)
    H = -(P * logP).sum(axis=1)
    return pd.Series(H / np.log(N_REGIMES), index=shares.index)


def js_distance_to_reference(shares: pd.DataFrame, ref: np.ndarray) -> pd.Series:
    """Jensen–Shannon DISTANCE (√ of the divergence, base 2 → [0,1]) between each
    dataset's categorical regime distribution and a reference distribution.
    Correct tool for nominal regime labels — needs no ground metric (unlike
    Wasserstein). Used with ref = POOLED empirical distribution."""
    P = shares[SHARE_COLS].values
    return pd.Series(
        [jensenshannon(p, ref, base=JS_BASE) for p in P], index=shares.index
    )


def hhi(shares: pd.DataFrame) -> pd.Series:
    """Herfindahl–Hirschman index = Σ pᵢ² ∈ [1/N, 1]. Concentration (APPENDIX
    robustness only — redundant in the body with entropy)."""
    return pd.Series((shares[SHARE_COLS].values ** 2).sum(axis=1), index=shares.index)


def hhi_normalized(shares: pd.DataFrame) -> pd.Series:
    """HHI rescaled to [0,1]: (HHI − 1/N)/(1 − 1/N)."""
    return (hhi(shares) - 1.0 / N_REGIMES) / (1.0 - 1.0 / N_REGIMES)


# NOTE: there is no regime-share Gini. "Gini" in this thesis refers
# exclusively to the Lorenz-based dynamic-content Gini of bias_shape_metrics.py
# (clustering-independent). A second Gini would be ambiguous.


# ═════════════════════════════════════════════════════════════════════════════
# CORPUS-LEVEL VIEWS  (aggregate — descriptive + the one reference-free "bias")
# ═════════════════════════════════════════════════════════════════════════════
def pooled_profile(assign: pd.DataFrame) -> np.ndarray:
    """Volume-weighted regime distribution over ALL windows = the field as it is
    actually pooled (large datasets dominate). DESCRIPTIVE — not labelled 'biased'
    without an external real-driving reference (see §6.6 limitation)."""
    c = assign["regime"].value_counts().reindex(range(N_REGIMES)).fillna(0.0).values
    return c / c.sum()


def balanced_profile(shares: pd.DataFrame) -> np.ndarray:
    """Equal-weight mean of the per-dataset share vectors = one vote per dataset."""
    return shares[SHARE_COLS].mean(axis=0).values


def volume_gap_js(pooled: np.ndarray, balanced: np.ndarray) -> float:
    """JS distance between pooled (volume-weighted) and balanced (equal-weight).
    The ONLY aggregate quantity allowed to wear the word 'bias': it is a purely
    internal comparison → volume-induced bias. Motivates RQ3."""
    return float(jensenshannon(pooled, balanced, base=JS_BASE))


def dispersion(shares: pd.DataFrame):
    """How spread out the 14 datasets are in regime space.
    Returns (per-regime std across datasets, mean pairwise JS distance)."""
    S = shares[SHARE_COLS].values
    per_regime_std = S.std(axis=0, ddof=1)
    pair_js = [jensenshannon(a, b, base=JS_BASE) for a, b in itertools.combinations(S, 2)]
    return per_regime_std, float(np.mean(pair_js))


def sparse_regime_table(shares: pd.DataFrame) -> pd.DataFrame:
    """Long (dataset×regime) table flagging under-filled regimes for RQ3.
    deficit_vs_uniform = max(0, 1/N − share) = how much a balanced subset would
    need to compensate. NOTE: uniform here is the compensation target, a
    methodological choice (same caveat as entropy, §6.6)."""
    rows = []
    for ds, r in shares.iterrows():
        for i in range(N_REGIMES):
            s = float(r[f"share_{i}"])
            deficit = max(0.0, 1.0 / N_REGIMES - s)
            rows.append(
                dict(dataset=ds, regime=i, share=s,
                     deficit_vs_uniform=deficit, is_sparse=deficit > 0.0)
            )
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# PER-CHANNEL WASSERSTEIN  (from std_data, Section 4.6 distributions)
# ═════════════════════════════════════════════════════════════════════════════
def load_std_channels(std_dir: Path, datasets) -> dict:
    """Read the three ECDF channels per dataset; convert yaw rad/s → °/s so the
    distances are in the same units Section 4.6 plotted. No physical-validity
    masking is applied to std values beyond NaN-drop, matching ecdf_analysis.py."""
    out = {}
    for ds in datasets:
        p = std_dir / f"{ds}.parquet"
        if not p.exists():
            print(f"  [WARN] std_data missing: {p}")
            continue
        df = pd.read_parquet(p)
        cols = [c for c in WASSERSTEIN_CHANNELS if c in df.columns]
        df = df[cols].copy()
        if "std_yaw_rate" in df.columns:
            df["std_yaw_rate"] = df["std_yaw_rate"] * RAD_TO_DEG
        out[ds] = df
    return out


def wasserstein_per_channel(std_data: dict) -> pd.DataFrame:
    """1-Wasserstein (earth-mover) distance of each dataset's per-channel
    distribution to the POOLED distribution (all datasets concatenated), per
    channel, in physical units. Wasserstein is valid HERE (continuous, ordered
    support) and never applied to the categorical regime labels (that is JS's
    job). The pooled reference includes the dataset itself; documented."""
    pooled = {
        ch: np.concatenate(
            [std_data[ds][ch].dropna().values for ds in std_data if ch in std_data[ds]]
        )
        for ch in WASSERSTEIN_CHANNELS
    }
    rows = {}
    for ds, df in std_data.items():
        rows[ds] = {}
        for ch in WASSERSTEIN_CHANNELS:
            if ch in df.columns:
                v = df[ch].dropna().values
                rows[ds][f"w_{ch}"] = (
                    wasserstein_distance(v, pooled[ch]) if len(v) else np.nan
                )
            else:
                rows[ds][f"w_{ch}"] = np.nan
    return pd.DataFrame(rows).T


# ═════════════════════════════════════════════════════════════════════════════
# DRIVER
# ═════════════════════════════════════════════════════════════════════════════
def compute_all(assign: pd.DataFrame, std_data: dict):
    """Pure function: assignments + std channels → (per_dataset, profiles,
    scalars, sparse) frames. No I/O, so it is unit-testable."""
    shares = regime_share_matrix(assign)

    per = shares.copy()
    per["low_dynamic_share"] = directional_low_dynamic_share(shares)
    per["entropy_norm"] = normalized_entropy(shares)
    pooled = pooled_profile(assign)
    balanced = balanced_profile(shares)
    per["js_to_pooled"] = js_distance_to_reference(shares, pooled)
    # COMPANION metric (veto-able): pooled is volume-weighted, hence dominated
    # by the largest datasets; js_to_balanced is the same distance against the
    # one-vote-per-dataset reference. Robustness column only. The reference
    # fixed for the main results remains POOLED.
    per["js_to_balanced"] = js_distance_to_reference(shares, balanced)
    per["hhi"] = hhi(shares)
    per["hhi_norm"] = hhi_normalized(shares)
    per["confounded_accel"] = per.index.isin(CONFOUNDED_ACCEL)

    w = wasserstein_per_channel(std_data)
    per = per.join(w)  # aligns on dataset index; missing std_data → NaN + warn
    if per[[c for c in per.columns if c.startswith("w_")]].isna().any().any():
        print("  [WARN] some Wasserstein channels are NaN (dataset missing from std_data?)")

    balanced2 = balanced  # already computed above for the companion JS column
    per_regime_std, mean_pw_js = dispersion(shares)
    profiles = pd.DataFrame(
        {"pooled": pooled, "balanced": balanced2, "dataset_std": per_regime_std},
        index=SHARE_COLS,
    ).T
    low_cols = [f"share_{i}" for i in LOW_DYNAMIC_REGIMES]
    profiles["low_dynamic_share"] = profiles[low_cols].sum(axis=1)
    # summing per-regime std is not a share — blank it to avoid misreading.
    profiles.loc["dataset_std", "low_dynamic_share"] = np.nan

    scalars = pd.DataFrame([dict(
        volume_gap_js=volume_gap_js(pooled, balanced),
        mean_pairwise_js=mean_pw_js,
        n_datasets=int(len(shares)),
        total_windows=int(shares["n_windows"].sum()),
    )])

    sparse = sparse_regime_table(shares)
    # RQ3 guard: deficits of confounded datasets are NOT actionable targets —
    # their under-filled regimes reflect signal fidelity, not missing behaviour.
    sparse["confounded_accel"] = sparse["dataset"].isin(CONFOUNDED_ACCEL)
    return per, profiles, scalars, sparse


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading assignments … {ASSIGN_FILE}")
    assign = pd.read_parquet(ASSIGN_FILE)
    datasets = sorted(assign["dataset"].unique())
    print(f"  {len(assign):,} windows across {len(datasets)} datasets")

    print("Reading std_data channels …")
    std_data = load_std_channels(STD_DATA_DIR, datasets)

    per, profiles, scalars, sparse = compute_all(assign, std_data)

    per.to_parquet(ARTIFACT_DIR / "bias_per_dataset.parquet")
    profiles.to_parquet(ARTIFACT_DIR / "bias_corpus_profiles.parquet")
    scalars.to_parquet(ARTIFACT_DIR / "bias_corpus_scalars.parquet")
    sparse.to_parquet(ARTIFACT_DIR / "bias_sparse_regimes.parquet")

    print("\nWrote:")
    for f in ["bias_per_dataset", "bias_corpus_profiles",
              "bias_corpus_scalars", "bias_sparse_regimes"]:
        print(f"  {ARTIFACT_DIR / (f + '.parquet')}")

    # console sanity summary (NOT stored; helps eyeball before figures)
    with pd.option_context("display.width", 180, "display.max_columns", 40):
        print("\nPer-dataset (sorted by low-dynamic share; * = accel-confounded):")
        show = per.sort_values("low_dynamic_share", ascending=False)[
            SHARE_COLS + ["n_windows", "low_dynamic_share", "entropy_norm",
                          "js_to_pooled", "js_to_balanced"]
        ].copy()
        show.index = [d + ("*" if d in CONFOUNDED_ACCEL else "") for d in show.index]
        print(show.round(3))
        vol = per["n_windows"] / per["n_windows"].sum()
        print("\nVolume share of pooled reference (top 3):")
        print((vol.sort_values(ascending=False).head(3) * 100).round(1).to_string())
        print("\n[NOTE] * datasets: regime metrics confounded by accel-signal "
              "fidelity (a2d2 attenuated; 4seasons/durlar inflated). Keep in "
              "tables, EXCLUDE from ranking claims. See CONFOUNDED_ACCEL.")
        print("\nCorpus profiles:")
        print(profiles.round(3))
        print("\nCorpus scalars:")
        print(scalars.round(4).to_string(index=False))


if __name__ == "__main__":
    main()