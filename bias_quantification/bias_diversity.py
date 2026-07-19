"""
bias_diversity.py  —  Tier-2 diversity metrics (Chapter 10, §4.7).

Two instruments, one artifact:

  1. Hill diversity profile D_q for q in {0, 1, 2} per dataset — the
     "effective number of regimes" family. NOT independent new metrics:
     D_1 = N ** entropy_norm and D_2 = 1 / hhi, i.e. the q=1 and q=2 slices
     are exact transforms of the Tier-1 entropy and HHI columns. The family
     UNIFIES them in interpretable effective-regime units and adds one
     genuinely new member, D_0 (richness), which ties directly to the
     sparse-regime table. DC8: the per-dataset regime-share Gini stays
     dropped; this family is the diversity instrument.

  2. PSI of each dataset's regime distribution against the pooled reference,
     plotted against the conventional 0.10 / 0.25 thresholds. DC9: PSI is a
     CALIBRATION COMPANION to the primary JS metric — it borrows an external
     severity convention; it never replaces JS. Zero-cell datasets are
     floor-dominated by construction and read as "major shift", which is the
     correct diagnosis for a missing regime.

Reads : ARTIFACT_DIR/cluster_assignments.parquet   (canonical)
        ARTIFACT_DIR/bias_per_dataset.parquet      (cross-check only, optional)
Writes: ARTIFACT_DIR/bias_diversity.parquet
Output: figures/hill_profiles.png, figures/psi_thresholds.png

Strict matplotlib: pure defaults, dpi=300, TUM dataset colours; confounded
datasets carry the Tier-1 asterisk in every label.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bias_tier2_common import (
    ARTIFACT_DIR, TIER1_PER_DATASET, FIG_ROOT, DPI, N_REGIMES,
    load_assignments, regime_counts, share_rows, pooled_shares,
    hill, psi, js_distance, label, color, cross_check_tier1,
)

OUT_DIR = FIG_ROOT
Q_GRID = [0, 1, 2]
PSI_MODERATE, PSI_MAJOR = 0.10, 0.25


# ══════════════════════════════ build ══════════════════════════════
def build_table():
    df = load_assignments()
    datasets, counts = regime_counts(df)
    shares = share_rows(counts)
    pooled = pooled_shares(counts)

    tab = pd.DataFrame(index=datasets)
    tab["n_windows"] = counts.sum(axis=1).astype(int)
    for q in Q_GRID:
        tab[f"hill_d{q}"] = [hill(s, q) for s in shares]
    tab["psi_pooled"] = [psi(s, pooled) for s in shares]
    tab["js_to_pooled"] = [js_distance(s, pooled) for s in shares]
    tab["low_dynamic_share"] = shares[:, :2].sum(axis=1)  # for cross-check only
    return tab


# ═════════════════════════════ figures ═════════════════════════════
def fig_hill_profiles(tab):
    fig, ax = plt.subplots(figsize=(9, 6))
    order = tab["hill_d1"].sort_values(ascending=False).index  # legend readability
    for ds in order:
        ax.plot(Q_GRID, [tab.loc[ds, f"hill_d{q}"] for q in Q_GRID],
                color=color(ds), linewidth=1.8, alpha=0.9, label=label(ds))
    ax.set_xticks(Q_GRID)
    ax.set_xticklabels(["q = 0\n(richness)", "q = 1\n(Shannon)", "q = 2\n(Simpson)"])
    ax.set_ylim(0.0, N_REGIMES + 0.4)
    ax.axhline(N_REGIMES, color="black", linewidth=1, alpha=0.4,
               label=f"maximum ({N_REGIMES} regimes)")
    ax.set_xlabel("Diversity order q")
    ax.set_ylabel("Effective number of regimes  $D_q$")
    ax.set_title("Hill diversity profiles: regime diversity per dataset")
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", ncol=2)
    fig.tight_layout()
    out = OUT_DIR / "hill_profiles.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def fig_psi_thresholds(tab):
    t = tab.sort_values("psi_pooled")          # barh: largest ends on top
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(y, t["psi_pooled"].to_numpy(),
            color=[color(d) for d in t.index], edgecolor="black", linewidth=0.5)
    ax.axvline(PSI_MODERATE, color="black", linewidth=1, alpha=0.6,
               label="PSI = 0.10 (moderate shift)")
    ax.axvline(PSI_MAJOR, color="black", linewidth=2,
               label="PSI = 0.25 (major shift)")
    ax.set_yticks(y)
    ax.set_yticklabels([label(d) for d in t.index])
    ax.set_xlabel("PSI vs pooled reference")
    ax.set_title("Population Stability Index against the pooled corpus")
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = OUT_DIR / "psi_thresholds.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# ═══════════════════════════════ main ══════════════════════════════
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    tab = build_table()

    # optional verification against the Tier-1 artifact (equivalences + names)
    if TIER1_PER_DATASET.exists():
        cross_check_tier1(tab, pd.read_parquet(TIER1_PER_DATASET))

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\nDiversity metrics (real values; * = confounded acceleration):\n")
    shown = tab.drop(columns=["low_dynamic_share"]).copy()
    shown.index = [label(d) for d in shown.index]
    print(shown.round(4).sort_values("hill_d1", ascending=False).to_string())

    n_major = int((tab["psi_pooled"] > PSI_MAJOR).sum())
    print(f"\n{n_major} of {len(tab)} datasets exceed the conventional "
          f"PSI major-shift threshold (0.25).")
    zero_cell = tab.index[tab["hill_d0"] < N_REGIMES].tolist()
    if zero_cell:
        print(f"Zero-cell (floor-dominated PSI) datasets: "
              f"{', '.join(label(d) for d in zero_cell)}")

    artifact = ARTIFACT_DIR / "bias_diversity.parquet"
    tab.drop(columns=["low_dynamic_share"]).reset_index(names="dataset") \
       .to_parquet(artifact, index=False)
    print(f"\nArtifact: {artifact}")

    p1 = fig_hill_profiles(tab)
    p2 = fig_psi_thresholds(tab)
    print(f"Output: {p1}")
    print(f"Output: {p2}")


if __name__ == "__main__":
    main()