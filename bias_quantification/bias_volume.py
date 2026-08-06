"""
bias_volume.py  —  Tier-2 volume-bias instrument (Chapter 10, §4.7).

The Tier-1 volume finding (volume_gap_js: the pooled reference is whatever
NVIDIA and Waymo say it is) gets its own axis here:

  * Lorenz curve of window counts across the 14 datasets, with the
    VOLUME-SHARE GINI of corpus concentration.
  * Effective dataset count via Hill numbers on the VOLUME shares:
    D_1 = exp(Shannon), D_2 = 1/Simpson. D_2 is the headline — "the pooled
    corpus behaves like ~D_2 datasets out of 14" — the single strongest
    quantitative motivation for the RQ3 curated subset.

NAMING RULE (updated with DC10 resolved): TWO qualified Ginis exist in
Chapter 10 — this "volume-share Gini" and the "dynamic-content Gini" of
bias_shape_metrics.py (exposé §3.7). The unqualified word "Gini" is never
used. Two unqualified Ginis = examiner bait.

Honest caveats:
  * A Gini over n = 14 units is a coarse descriptor. It is used descriptively
    for THIS corpus, not as an estimator of anything beyond it.
  * Volume concentration is corpus-level composition, not a per-dataset
    bias: a dataset is not "worse" for being small. The fingerprint carries
    vol_share as a context column for the same reason.

Reads : ARTIFACT_DIR/cluster_assignments.parquet   (canonical)
Writes: ARTIFACT_DIR/bias_volume.parquet
Output: figures/lorenz_volume.png

Strict matplotlib: pure defaults, dpi=300, TUM colours, asterisked labels.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bias_tier2_common import (
    ARTIFACT_DIR, FIG_ROOT, DPI, TUM_PRIMARY,
    load_assignments, regime_counts, hill, gini, label, color,
)

OUT_DIR = FIG_ROOT
AXIS_PAD = 0.015     # headroom so the corner marker (largest dataset) is not
                     # clipped by the axes frame at (1, 1)


# ══════════════════════════════ build ══════════════════════════════
def build_table():
    df = load_assignments(columns=("dataset", "regime"))
    datasets, counts = regime_counts(df)
    n_win = counts.sum(axis=1)
    shares = n_win / n_win.sum()

    tab = pd.DataFrame({"n_windows": n_win.astype(int), "vol_share": shares},
                       index=datasets).sort_values("vol_share")
    scalars = {
        "volume_gini": gini(tab["vol_share"].to_numpy()),
        "eff_datasets_d1": hill(tab["vol_share"].to_numpy(), 1),
        "eff_datasets_d2": hill(tab["vol_share"].to_numpy(), 2),
        "n_datasets": len(tab),
        "n_windows_total": int(n_win.sum()),
    }
    return tab, scalars


# ═════════════════════════════ figure ══════════════════════════════
def fig_lorenz(tab, scalars):
    shares = tab["vol_share"].to_numpy()            # ascending
    cum_x = np.arange(len(shares) + 1) / len(shares)
    cum_y = np.concatenate([[0.0], np.cumsum(shares)])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], color="black", linewidth=1, alpha=0.5, zorder=1,
            label="perfect equality")
    ax.plot(cum_x, cum_y, color=TUM_PRIMARY["tum_blue"], linewidth=2, zorder=2,
            label="corpus Lorenz curve")

    # one point per dataset at its cumulative position, largest drawn last.
    # zorder=3: scatter defaults BELOW plot, which hid the markers under the curve.
    for i, ds in enumerate(tab.index):
        ax.scatter(cum_x[i + 1], cum_y[i + 1], color=color(ds),
                   edgecolor="black", linewidth=0.5, zorder=3, label=label(ds))

    ax.set_xlim(-AXIS_PAD, 1.0 + AXIS_PAD)
    ax.set_ylim(-AXIS_PAD, 1.0 + AXIS_PAD)
    ax.set_xlabel("Cumulative fraction of datasets (smallest first)")
    ax.set_ylabel("Cumulative fraction of 20 s windows")
    # two-line title: the one-line version clipped at the figure edge
    ax.set_title("Corpus volume concentration\n"
                 f"volume-share Gini = {scalars['volume_gini']:.2f}, "
                 f"effective datasets $D_2$ \u2248 {scalars['eff_datasets_d2']:.1f} "
                 f"of {scalars['n_datasets']}")
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    out = OUT_DIR / "lorenz_volume.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# ═══════════════════════════════ main ══════════════════════════════
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    tab, sc = build_table()

    pd.set_option("display.width", 200)
    shown = tab.copy()
    shown.index = [label(d) for d in shown.index]
    print("\nVolume shares (ascending; * = confounded acceleration):\n")
    print(shown.round(4).to_string())
    print(f"\nvolume-share Gini = {sc['volume_gini']:.3f} | effective datasets: "
          f"D1 = {sc['eff_datasets_d1']:.2f}, D2 = {sc['eff_datasets_d2']:.2f} "
          f"of {sc['n_datasets']} | total windows = {sc['n_windows_total']:,}")

    out_tab = tab.reset_index(names="dataset")
    for k, v in sc.items():
        out_tab[k] = v            # scalars repeated per row, self-contained artifact
    artifact = ARTIFACT_DIR / "bias_volume.parquet"
    out_tab.to_parquet(artifact, index=False)
    print(f"\nArtifact: {artifact}")

    p = fig_lorenz(tab, sc)
    print(f"Output: {p}")


if __name__ == "__main__":
    main()