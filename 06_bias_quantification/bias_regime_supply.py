"""
bias_regime_supply.py  —  regime provenance map (Section 4.8 / RQ3 bridge).

The share heatmap answers "what does each DATASET look like?" (rows sum to
100 %). This figure answers the transposed question RQ3 actually needs:
"who SUPPLIES each REGIME's windows?" — columns sum to 100 %. Same count
matrix, opposite normalisation: one is dataset composition, the other is
corpus provenance.

Why it matters for RQ3: a regime can look healthy in the pooled corpus while
being supplied almost entirely by one dataset — that regime's diversity is
an illusion of volume. This is the literal shopping map for the curated
subset: for every under-represented regime, read down the column to see who
has stock.

Caveats:
  * Supply shares are volume-driven BY CONSTRUCTION — NVIDIA dominating
    every column is the volume-bias finding restated at regime granularity,
    not a bug. The interesting cells are where domination BREAKS.
  * Confounded datasets' supply of Braking/Accelerating is fidelity-tinted
    (asterisked rows, same policy as everywhere).

Reads : ARTIFACT_DIR/cluster_assignments.parquet   (canonical)
Writes: ARTIFACT_DIR/bias_regime_supply.parquet
Output: 06_bias_quantification/figures/bias_regime_supply.png
        (supporting figure, not part of the thesis body)

Strict matplotlib: pure defaults, dpi=300. Single small in-cell font size —
the documented heatmap exception.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from regime_style import N_REGIMES, regime_label
from bias_tier2_common import (
    ARTIFACT_DIR, FIG_ROOT, DPI,
    load_assignments, regime_counts, label,
)

OUT_DIR = FIG_ROOT
CELL_FS = 8   # documented heatmap exception: in-cell annotation font
TOP_K = 3     # suppliers listed per regime in the console summary


# ══════════════════════════════ build ══════════════════════════════
def build():
    df = load_assignments(columns=("dataset", "regime"))
    datasets, counts = regime_counts(df)

    supply = counts / counts.sum(axis=0, keepdims=True) * 100.0  # columns -> 100 %
    order = np.argsort(-counts.sum(axis=1))                      # by volume, largest top
    datasets = [datasets[i] for i in order]
    supply = supply[order]
    counts = counts[order]
    return datasets, counts, supply


# ═════════════════════════════ render ══════════════════════════════
def render(datasets, supply):
    n_ds = len(datasets)
    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(supply, cmap="Blues", aspect="auto",
                   vmin=0.0, vmax=float(supply.max()))

    ax.set_xticks(range(N_REGIMES))
    ax.set_xticklabels([regime_label(i) for i in range(N_REGIMES)],
                       rotation=30, ha="right")
    ax.set_yticks(range(n_ds))
    ax.set_yticklabels([label(d) for d in datasets])
    ax.set_title("Regime supply, who provides each regime's windows")

    thresh = 0.55 * supply.max()
    for r in range(n_ds):
        for c in range(N_REGIMES):
            v = supply[r, c]
            ax.text(c, r, f"{v:.1f}" if v < 10 else f"{v:.0f}",
                    ha="center", va="center", fontsize=CELL_FS,
                    color="white" if v > thresh else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Share of the regime's pooled windows (%)")

    fig.tight_layout()
    out = OUT_DIR / "bias_regime_supply.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# ═══════════════════════════════ main ══════════════════════════════
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    datasets, counts, supply = build()

    pd.set_option("display.width", 200, "display.max_columns", 30)
    tab = pd.DataFrame(supply, index=datasets,
                       columns=[f"supply_{i}" for i in range(N_REGIMES)])
    tab["n_windows"] = counts.sum(axis=1).astype(int)
    shown = tab.copy()
    shown.index = [label(d) for d in shown.index]
    print("\nRegime supply (% of each regime's pooled windows) ")
    print(shown.round(2).to_string())

    print(f"\nTop {TOP_K} suppliers per regime:")
    for i in range(N_REGIMES):
        col = tab[f"supply_{i}"].sort_values(ascending=False).head(TOP_K)
        parts = ", ".join(f"{label(d)} {v:.0f}%" for d, v in col.items())
        print(f"  {i} {regime_label(i)}: {parts}")

    artifact = ARTIFACT_DIR / "bias_regime_supply.parquet"
    tab.reset_index(names="dataset").to_parquet(artifact, index=False)
    print(f"\nArtifact: {artifact}")

    p = render(datasets, supply)
    print(f"Output: {p}")


if __name__ == "__main__":
    main()