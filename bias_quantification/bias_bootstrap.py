"""
bias_bootstrap.py  —  Tier-2 uncertainty quantification (Chapter 10, §4.7).

DC7 (locked): clip-level nonparametric bootstrap, B = 1000 replicates,
percentile 95 % confidence intervals, seed 42, POOLED REFERENCE HELD FIXED
(resampling the reference too would answer a different question and is
computationally brutal). Metrics covered: the RQ1 headline low_dynamic_share,
js_to_pooled, and Hill D_1 (effective regimes) — every number the chapter
makes a ranking claim about carries an interval. Metric names match the
Tier-1 bias_per_dataset.parquet columns exactly.

DC14 addition — rank probabilities: for the headline metric, the fraction of
replicates strictly above the pooled and balanced references is stored as
p_above_pooled / p_above_balanced. This turns CI-vs-line eyeballing into a
quotable number ("Ford exceeds the pooled corpus in 100 % of resamples;
KITTI in only 74 %"). It is a PRESENTATION of the existing uncertainty, not
a new bias axis.

Why clips, not windows: windows within a clip share a route, a driver and a
recording session; resampling windows would pretend they are independent and
produce dishonestly narrow intervals. Resampling whole clips preserves the
within-clip correlation structure.

Honest caveats:
  * Clip exchangeability within a dataset is itself an approximation — clips
    from one campaign share routes. The intervals are best read as lower
    bounds on the real uncertainty.
  * Datasets with very few clips (A2D2 after cleaning) produce enormous
    intervals. That is a FEATURE: the statistics themselves declare the rank
    unstable, independently of the acceleration-fidelity flag.

Reads : ARTIFACT_DIR/cluster_assignments.parquet   (canonical)
Writes: ARTIFACT_DIR/bias_bootstrap_ci.parquet
Output: figures/low_dyn_share_ci.png

Strict matplotlib: pure defaults, dpi=300, TUM colours, asterisked labels.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

from bias_tier2_common import (
    ARTIFACT_DIR, FIG_ROOT, DPI, N_REGIMES, LOW_DYNAMIC_REGIMES,
    TUM_PRIMARY, TUM_ACCENT,
    load_assignments, regime_counts, share_rows, pooled_shares,
    balanced_shares, hill, js_distance, label, color,
)

OUT_DIR = FIG_ROOT

B = 1000            # DC7
SEED = 42           # DC7
CI = (2.5, 97.5)    # DC7: percentile 95 %
MIN_CLIPS_WARN = 30

METRICS = ["low_dynamic_share", "js_to_pooled", "hill_d1"]   # Tier-1 names


# ═══════════════════════════ bootstrap ═════════════════════════════
def _metrics(counts_vec, pooled):
    tot = counts_vec.sum()
    if tot <= 0:
        return np.nan, np.nan, np.nan
    s = counts_vec / tot
    low = float(s[list(LOW_DYNAMIC_REGIMES)].sum())
    return low, js_distance(s, pooled), hill(s, 1)


def bootstrap_dataset(df_ds, pooled, refs, rng):
    """Clip-level bootstrap for one dataset.

    Builds the (n_clips x N_REGIMES) regime-count matrix once, then each
    replicate resamples clip rows with replacement and sums — O(n_clips) per
    replicate, no window-level copies, tractable even for NVIDIA (~298k clips).
    Also returns DC14 rank probabilities: fraction of replicates whose
    headline low_dynamic_share lies strictly above each fixed reference.
    """
    C = (df_ds.groupby(["clip_id", "regime"]).size()
              .unstack(fill_value=0)
              .reindex(columns=range(N_REGIMES), fill_value=0)
              .to_numpy(dtype=np.int64))
    n_clips = C.shape[0]

    point = _metrics(C.sum(axis=0).astype(float), pooled)
    reps = np.empty((B, 3), dtype=float)
    for b in range(B):
        idx = rng.integers(0, n_clips, n_clips)
        reps[b] = _metrics(C[idx].sum(axis=0).astype(float), pooled)

    lo = np.nanpercentile(reps, CI[0], axis=0)
    hi = np.nanpercentile(reps, CI[1], axis=0)
    probs = {"p_above_pooled": float(np.mean(reps[:, 0] > refs["pooled_low_dyn"])),
             "p_above_balanced": float(np.mean(reps[:, 0] > refs["balanced_low_dyn"]))}
    return n_clips, point, lo, hi, probs


def build_table():
    df = load_assignments()
    datasets, counts = regime_counts(df)
    pooled = pooled_shares(counts)          # fixed reference (DC7)
    balanced = balanced_shares(counts)
    low_idx = list(LOW_DYNAMIC_REGIMES)
    refs = {"pooled_low_dyn": float(pooled[low_idx].sum()),
            "balanced_low_dyn": float(balanced[low_idx].sum())}
    rng = np.random.default_rng(SEED)

    rows = []
    for ds in tqdm(datasets, desc="bootstrap"):
        n_clips, pt, lo, hi, probs = bootstrap_dataset(
            df[df["dataset"] == ds], pooled, refs, rng)
        rec = {"dataset": ds, "n_clips": n_clips,
               "n_windows": int(df["dataset"].eq(ds).sum())}
        for j, m in enumerate(METRICS):
            rec[f"{m}_pt"], rec[f"{m}_lo"], rec[f"{m}_hi"] = pt[j], lo[j], hi[j]
        rec.update(probs)                    # DC14
        rows.append(rec)

    tab = pd.DataFrame(rows).set_index("dataset")
    return tab, refs


# ═════════════════════════════ figure ══════════════════════════════
def fig_share_ci(tab, refs):
    t = tab.sort_values("low_dynamic_share_pt")          # largest ends on top
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(9, 6.5))

    for yi, ds in zip(y, t.index):
        ax.plot([t.loc[ds, "low_dynamic_share_lo"], t.loc[ds, "low_dynamic_share_hi"]],
                [yi, yi], color=color(ds), linewidth=2.5, alpha=0.9)
        ax.scatter(t.loc[ds, "low_dynamic_share_pt"], yi, color=color(ds),
                   edgecolor="black", linewidth=0.6)

    ax.axvline(refs["pooled_low_dyn"], color=TUM_PRIMARY["tum_blue"],
               linewidth=1.5, alpha=0.8,
               label=f"pooled corpus ({refs['pooled_low_dyn']:.3f})")
    ax.axvline(refs["balanced_low_dyn"], color=TUM_ACCENT["orange"],
               linewidth=1.5, alpha=0.8,
               label=f"balanced corpus ({refs['balanced_low_dyn']:.3f})")

    ax.set_yticks(y)
    ax.set_yticklabels([label(d) for d in t.index])
    ax.set_xlabel("Low-dynamic share (regimes {0, 1}) with 95% clip-bootstrap CI")
    ax.set_title(f"Headline metric with uncertainty (B = {B}, clip-level resampling)")
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = OUT_DIR / "low_dyn_share_ci.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# ═══════════════════════════════ main ══════════════════════════════
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    tab, refs = build_table()

    pd.set_option("display.width", 240, "display.max_columns", 40)
    shown = tab.copy()
    shown.index = [label(d) for d in shown.index]
    print("\nBootstrap CIs and rank probabilities "
          "(real values; * = confounded acceleration):\n")
    print(shown.round(4).sort_values("low_dynamic_share_pt", ascending=False).to_string())

    small = tab.index[tab["n_clips"] < MIN_CLIPS_WARN].tolist()
    if small:
        print(f"\nWARNING — fewer than {MIN_CLIPS_WARN} clips (intervals are the finding): "
              f"{', '.join(label(d) for d in small)}")

    # DC14 quotable lines: unambiguous vs unrankable datasets
    sure_hi = tab.index[tab["p_above_pooled"] >= 0.975].tolist()
    sure_lo = tab.index[tab["p_above_pooled"] <= 0.025].tolist()
    unclear = tab.index[(tab["p_above_pooled"] > 0.025)
                        & (tab["p_above_pooled"] < 0.975)].tolist()
    print(f"\nAbove pooled in >=97.5 % of resamples: "
          f"{', '.join(label(d) for d in sure_hi) or '-'}")
    print(f"Below pooled in >=97.5 % of resamples: "
          f"{', '.join(label(d) for d in sure_lo) or '-'}")
    print(f"Direction unresolved vs pooled: "
          f"{', '.join(label(d) for d in unclear) or '-'}")

    artifact = ARTIFACT_DIR / "bias_bootstrap_ci.parquet"
    tab.reset_index().to_parquet(artifact, index=False)
    print(f"\nArtifact: {artifact}")

    p = fig_share_ci(tab, refs)
    print(f"Output: {p}")


if __name__ == "__main__":
    main()