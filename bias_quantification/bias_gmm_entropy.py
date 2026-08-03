"""
bias_gmm_entropy.py  —  Tier-3 boundary-bias instrument (Chapter 10, §4.7).

THE question k-means cannot answer: does a dataset occupy the clean CORES of
the behavioural regimes, or does it pile into the OVERLAP ZONES between them?
K-means assigns every window to exactly one regime with no notion of
ambiguity; the Chapter-9 GMM stored, per window, the posterior entropy over
its five components and the maximum posterior. This script aggregates those
into a per-dataset boundary-bias profile:

  1. Mean posterior entropy per dataset (PRIMARY — threshold-free).
     High = the dataset's windows sit where components overlap; low = the
     dataset lives in unambiguous regime cores.
  2. Ambiguous-window share: fraction of windows with max_posterior < 0.8
     (COMPANION; tau = 0.8 is a disclosed convention, with 0.5 / 0.9
     sensitivity printed so the choice is auditable, DC11).

Honest caveats (write these into the methodology):
  * Boundary bias is defined in the GMM's geometry. The GMM<->k-means
    correspondence is partial (ARI = 0.30, Chapter 9) — this measures
    ambiguity in the mixture model's view of feature space, not distance to
    the k-means partition. Disclosed, not hidden.
  * Composition confound: part of a dataset's mean entropy is inherited from
    WHICH regions it occupies (some components are intrinsically fuzzier).
    The per-component pooled entropy breakdown is printed so this is visible.
  * Entropy units are as stored by Chapter 9 (natural log assumed; max for
    five components = ln 5 ~ 1.609). The script verifies the observed max
    against that bound before printing a normalised column.
  * Accel-confounded datasets carry the asterisk: the GMM features include
    acceleration-derived dimensions, so their boundary profile inherits the
    fidelity artifact.

Reads : ARTIFACT_DIR/gmm_assignments.parquet
        (dataset, clip_id, window_start_s, window_end_s, gmm_regime,
         max_posterior, entropy, pc1, pc2)
Writes: ARTIFACT_DIR/bias_gmm_entropy.parquet
Output: figures/gmm_entropy.png

Strict matplotlib: pure defaults, dpi=300, TUM colours, asterisked labels.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bias_tier2_common import (
    ARTIFACT_DIR, REGIME_ASSIGN, FIG_ROOT, DPI, TUM_SECONDARY,
    label, color,
)

OUT_DIR = FIG_ROOT
GMM_ASSIGN = ARTIFACT_DIR / "gmm_assignments.parquet"

TAU_PRIMARY = 0.8            # DC11: ambiguity threshold (companion metric)
TAU_SENSITIVITY = (0.5, 0.8, 0.9)
N_COMPONENTS = 5
MAX_ENTROPY_NATS = np.log(N_COMPONENTS)   # 1.609..., assuming natural log


# ══════════════════════════════ build ══════════════════════════════
def build_table():
    if not GMM_ASSIGN.exists():
        raise SystemExit(f"GMM artifact not found: {GMM_ASSIGN}")
    df = pd.read_parquet(
        GMM_ASSIGN, columns=["dataset", "gmm_regime", "max_posterior", "entropy"])

    # coverage sanity vs the canonical k-means artifact
    n_kmeans = len(pd.read_parquet(REGIME_ASSIGN, columns=["regime"]))
    if len(df) != n_kmeans:
        print(f"  [WARN] gmm_assignments has {len(df):,} windows but "
              f"cluster_assignments has {n_kmeans:,} — populations differ.")

    # entropy-units guard (DC11): verify the natural-log assumption
    obs_max = float(df["entropy"].max())
    units_ok = obs_max <= MAX_ENTROPY_NATS * 1.001
    if not units_ok:
        print(f"  [WARN] observed max entropy {obs_max:.3f} exceeds ln(5) = "
              f"{MAX_ENTROPY_NATS:.3f} — normalised column SKIPPED; verify the "
              f"Chapter-9 entropy convention before quoting normalised values.")

    g = df.groupby("dataset")
    tab = pd.DataFrame(index=sorted(df["dataset"].unique()))
    tab["n_windows"] = g.size()
    tab["mean_entropy"] = g["entropy"].mean()
    tab["median_entropy"] = g["entropy"].median()
    if units_ok:
        tab["mean_entropy_norm"] = tab["mean_entropy"] / MAX_ENTROPY_NATS
    for tau in TAU_SENSITIVITY:
        tab[f"amb_share_{int(tau * 100)}"] = g["max_posterior"].apply(
            lambda v, t=tau: float(np.mean(v < t)))

    pooled_mean = float(df["entropy"].mean())
    comp_break = df.groupby("gmm_regime")["entropy"].mean()
    return tab, pooled_mean, comp_break, obs_max, units_ok


# ═════════════════════════════ figure ══════════════════════════════
def fig_entropy(tab, pooled_mean):
    t = tab.sort_values("mean_entropy")            # barh: largest ends on top
    y = np.arange(len(t))
    amb_col = f"amb_share_{int(TAU_PRIMARY * 100)}"

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), sharey=True)

    axes[0].barh(y, t["mean_entropy"].to_numpy(),
                 color=[color(d) for d in t.index],
                 edgecolor="black", linewidth=0.5)
    axes[0].axvline(pooled_mean, color=TUM_SECONDARY["gray"], linewidth=1.4,
                    label=f"pooled mean ({pooled_mean:.2f})")
    axes[0].set_title("Mean GMM posterior entropy\n(higher = more boundary-seated)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([label(d) for d in t.index])
    axes[0].grid(axis="x", alpha=0.3)
    axes[0].set_axisbelow(True)
    axes[0].legend(loc="lower right")

    axes[1].barh(y, t[amb_col].to_numpy(),
                 color=[color(d) for d in t.index],
                 edgecolor="black", linewidth=0.5)
    axes[1].set_title(f"Ambiguous-window share\n(max posterior < {TAU_PRIMARY})")
    axes[1].grid(axis="x", alpha=0.3)
    axes[1].set_axisbelow(True)

    fig.suptitle("Boundary bias: regime-core vs overlap-zone occupancy per dataset")
    fig.tight_layout()
    out = OUT_DIR / "gmm_entropy.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# ═══════════════════════════════ main ══════════════════════════════
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    tab, pooled_mean, comp_break, obs_max, units_ok = build_table()

    pd.set_option("display.width", 220, "display.max_columns", 40)
    shown = tab.copy()
    shown.index = [label(d) for d in shown.index]
    print("\nBoundary bias metrics (* = confounded acceleration):\n")
    print(shown.round(4).sort_values("mean_entropy", ascending=False).to_string())

    print(f"\nPooled mean entropy = {pooled_mean:.4f} "
          f"(observed max {obs_max:.4f}; ln(5) = {MAX_ENTROPY_NATS:.4f}"
          f"{', units check OK' if units_ok else ', UNITS CHECK FAILED'})")
    print("\nPer-component pooled mean entropy (composition-confound context):")
    print(comp_break.round(4).to_string())

    artifact = ARTIFACT_DIR / "bias_gmm_entropy.parquet"
    tab.reset_index(names="dataset").to_parquet(artifact, index=False)
    print(f"\nArtifact: {artifact}")

    p = fig_entropy(tab, pooled_mean)
    print(f"Output: {p}")


if __name__ == "__main__":
    main()