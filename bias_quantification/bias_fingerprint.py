"""
bias_fingerprint.py  —  Chapter-10 synthesis figure: the bias fingerprint.

One heatmap, every Chapter-10 scalar bias metric:

  Rows    : the 14 datasets, ordered by the RQ1 headline metric
            (low_dynamic_share, descending). DC6: there is deliberately NO
            composite bias score and no index spine — a weighted composite
            invites "why these weights" attacks and hides the multi-axis
            structure. The headline metric IS the ordering.
  Columns : Headline | Diversity | Divergence | Distributional W1 | Volume.
  Colour  : per-column min-max normalised intensity, Blues colormap, oriented
            so darker = stronger bias expression on that axis (Hill numbers
            are inverted: FEWER effective regimes = more concentrated = more
            biased). Real values printed in every cell — read the numbers,
            not just the shade (same contract as the Chapter-8 scorecard).

DC14 addition — metric-agreement table: Spearman rank correlations between
all fingerprint columns are computed, printed and stored
(bias_metric_agreement.parquet). This QUANTIFIES the "columns are correlated
by construction" disclosure and pre-answers the viva question "aren't your
ten metrics really one metric?" — appendix material, no figure.

Honest caveats (carry into the methodology):
  * Columns are correlated by construction; the repeating light/dark stripe
    across a row is a finding, not redundancy to hide.
  * n = 14: Spearman values are noisy; report as descriptive, appendix-only.
  * D1/D2 are transforms of Tier-1 entropy_norm/hhi (unified family); JS is
    the Tier-1 js_to_pooled value itself. The fingerprint REPRESENTS the
    chapter's metrics; it does not mint new ones.
  * Vol % is corpus COMPOSITION context, not a per-dataset defect — a small
    dataset is not "worse". It sits behind its own separator for that reason.
  * W1 values come from the Tier-1 artifact (w_std_* columns) when available,
    otherwise recomputed from std_data via the identical procedure
    (announced loudly at runtime).

Reads : cluster_assignments.parquet (+ bias_per_dataset.parquet or std_data)
Writes: ARTIFACT_DIR/bias_fingerprint.parquet
        ARTIFACT_DIR/bias_metric_agreement.parquet
Output: figures/bias_fingerprint.png

Strict matplotlib: pure defaults, dpi=300. The only deliberate deviation is a
single small font size for in-cell text and column-group labels — the same
documented latitude heatmap_summary.py uses.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from bias_tier2_common import (
    ARTIFACT_DIR, FIG_ROOT, DPI,
    build_bias_metric_table, label,
)

OUT_DIR = FIG_ROOT
CELL_FS = 8   # documented heatmap exception: in-cell + group-label font

# key, on-figure label, facet group, invert (True: LOW raw value = dark), format
COLUMNS = [
    dict(key="low_dynamic_share", label="LowDyn", group="Headline",   invert=False, fmt="%.2f"),
    dict(key="hill_d1",           label="$D_1$",  group="Diversity",  invert=True,  fmt="%.2f"),
    dict(key="hill_d2",           label="$D_2$",  group="Diversity",  invert=True,  fmt="%.2f"),
    dict(key="js_to_pooled",      label="JS",     group="Divergence", invert=False, fmt="%.3f"),
    dict(key="psi_pooled",        label="PSI",    group="Divergence", invert=False, fmt="%.2f"),
    dict(key="w1_speed",          label="W1 Spd", group="W1",         invert=False, fmt="%.2f"),
    dict(key="w1_acc",            label="W1 Acc", group="W1",         invert=False, fmt="%.3f"),
    dict(key="w1_yaw",            label="W1 Yaw", group="W1",         invert=False, fmt="%.2f"),
    dict(key="vol_share_pct",     label="Vol %",  group="Volume",     invert=False, fmt="%.1f"),
]


# ══════════════════════════════ build ══════════════════════════════
def build():
    tab = build_bias_metric_table(verbose=True)
    tab["vol_share_pct"] = tab["vol_share"] * 100.0
    tab = tab.sort_values("low_dynamic_share", ascending=False)   # DC6 row order
    raw = tab[[c["key"] for c in COLUMNS]].to_numpy(dtype=float)
    return tab, raw


def normalise(raw):
    """Per-column min-max -> bias intensity in [0, 1]; NaN cells stay missing."""
    inten = np.full_like(raw, np.nan)
    for j, c in enumerate(COLUMNS):
        col = raw[:, j]
        finite = np.isfinite(col)
        if not finite.any():
            continue
        lo, hi = np.nanmin(col), np.nanmax(col)
        t = (col - lo) / (hi - lo) if hi > lo else np.zeros_like(col)
        if c["invert"]:
            t = 1.0 - t
        t = np.asarray(t, dtype=float)
        t[~finite] = np.nan
        inten[:, j] = t
    return inten


def metric_agreement(raw):
    """DC14: Spearman rank-correlation matrix across the fingerprint columns.
    Descriptive appendix QA (n = 14, noisy) — quantifies the correlated-
    columns disclosure."""
    rho, _ = spearmanr(raw, axis=0, nan_policy="omit")
    names = [c["key"] for c in COLUMNS]
    return pd.DataFrame(rho, index=names, columns=names)


# ═════════════════════════════ render ══════════════════════════════
def render(tab, raw, inten):
    datasets = list(tab.index)
    n_ds, n_col = len(datasets), len(COLUMNS)

    fig_w = max(10.0, 1.15 * n_col + 3.0)
    fig_h = max(6.0, 0.46 * n_ds + 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(inten, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(n_col))
    ax.set_xticklabels([c["label"] for c in COLUMNS])
    ax.set_yticks(np.arange(n_ds))
    ax.set_yticklabels([label(d) for d in datasets])

    # headroom above row 0 for the facet-group labels (origin is upper)
    ax.set_ylim(n_ds - 0.5, -1.5)

    # in-cell real values
    for i in range(n_ds):
        for j, c in enumerate(COLUMNS):
            v = raw[i, j]
            if np.isfinite(v):
                shade = inten[i, j]
                tc = "white" if np.isfinite(shade) and shade > 0.55 else "black"
                ax.text(j, i, c["fmt"] % v, ha="center", va="center",
                        color=tc, fontsize=CELL_FS)

    # facet-group separators + labels
    spans, start = [], 0
    for j in range(1, n_col + 1):
        if j == n_col or COLUMNS[j]["group"] != COLUMNS[start]["group"]:
            spans.append((COLUMNS[start]["group"], start, j - 1))
            start = j
    for _, a, b in spans[1:]:
        ax.axvline(a - 0.5, color="black", linewidth=1)
    ax.axvline(n_col - 1.5, color="black", linewidth=2)  # composition context divider

    ax.axhline(-1.5, color="black", linewidth=1)
    ax.axhline(-0.5, color="black", linewidth=1)
    for name, a, b in spans:
        ax.text((a + b) / 2.0, -0.95, name, ha="center", va="center", fontsize=CELL_FS)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("column-normalised bias intensity (darker = stronger)")

    ax.set_title("Bias fingerprint across datasets — rows ordered by low-dynamic share; "
                 "* = accel-confounded")
    fig.tight_layout()

    out = OUT_DIR / "bias_fingerprint.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# ═══════════════════════════════ main ══════════════════════════════
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    tab, raw = build()
    inten = normalise(raw)

    pd.set_option("display.width", 220, "display.max_columns", 40)
    shown = tab[[c["key"] for c in COLUMNS] + ["n_windows"]].copy()
    shown.index = [label(d) for d in shown.index]
    print("\nFingerprint metrics (real values; * = confounded acceleration):\n")
    print(shown.round(4).to_string())

    agree = metric_agreement(raw)
    print("\nMetric-agreement Spearman matrix (DC14; n = 14, descriptive, "
          "appendix only):\n")
    print(agree.round(2).to_string())

    artifact = ARTIFACT_DIR / "bias_fingerprint.parquet"
    tab.reset_index(names="dataset").to_parquet(artifact, index=False)
    agree_path = ARTIFACT_DIR / "bias_metric_agreement.parquet"
    agree.reset_index(names="metric").to_parquet(agree_path, index=False)
    print(f"\nArtifact: {artifact}")
    print(f"Artifact: {agree_path}")

    p = render(tab, raw, inten)
    print(f"Output: {p}")


if __name__ == "__main__":
    main()