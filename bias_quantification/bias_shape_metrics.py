"""
bias_shape_metrics.py  —  dynamic-content concentration (Chapter 10, §4.7).

DC10 RESOLVED: BUILD. This is the exposé-§3.7 Lorenz/Gini instrument whose
name bias_quant.py's D4 note reserved — the "dynamic-content Gini". It is
clustering-INDEPENDENT: it never touches the regime labels, so it
cross-validates the regime-based story from a second, model-free angle.

The question: how unequally is dynamic content distributed across a
dataset's windows? Wealth-inequality analogy: if the top 20 % of windows
hold 70 % of all the speed-variability mass, the dataset is a long-tail
corpus — most windows are dead calm and a few carry all the action.

Instruments:
  * Per-dataset Lorenz curve of window-level std_speed_kmh (the HEADLINE
    channel: speed fidelity is plausible for all 14 datasets, so this Gini
    is fidelity-clean everywhere — DC12).
  * Dynamic-content Gini per dataset and for the pooled corpus, computed on
    all three DC5 channels (acc and yaw stored in the artifact; the
    acc-channel Gini of confounded datasets reflects fidelity, not
    behaviour, same as the W1 reading).

NAMING RULE (updated): TWO qualified Ginis now exist in Chapter 10 —
"dynamic-content Gini" (this script) and "volume-share Gini"
(bias_volume.py). The unqualified word "Gini" is never used.

Honest caveats:
  * Population = all std_data windows (50 % overlap), the Chapter-8 ECDF
    population — consistent with the W1 metric, disclosed as such.
  * A high Gini is not "bad": real driving IS long-tailed. The metric
    quantifies HOW long-tailed each corpus is, which is exactly what uniform
    sampling then inherits (the RQ1 mechanism, stated on the continuous
    signal instead of regime labels).
  * Curves are quantile-downsampled to 512 points per dataset for rendering;
    Gini values are computed on the full window population.

Reads : STD_DATA_DIR/<ds>.parquet   (via the shared threaded loader)
Writes: ARTIFACT_DIR/bias_shape.parquet
Output: figures/dynamic_lorenz.png

Strict matplotlib: pure defaults, dpi=300, TUM colours, asterisked labels.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bias_tier2_common import (
    ARTIFACT_DIR, FIG_ROOT, DPI, TUM_PRIMARY, W1_CHANNELS,
    load_assignments, load_std_channels, gini, label, color,
)

OUT_DIR = FIG_ROOT
HEADLINE_CH = "std_speed_kmh"      # DC12: fidelity-clean for all 14 datasets
CURVE_POINTS = 512                  # rendering downsample; Gini uses full data


# ══════════════════════════════ build ══════════════════════════════
def build():
    datasets = sorted(load_assignments(columns=("dataset",))["dataset"].unique())
    data = load_std_channels(datasets)

    rows = []
    for ds in datasets:
        rec = {"dataset": ds}
        for ch in W1_CHANNELS:
            v = data[ds][ch]
            rec[f"gini_{ch}"] = gini(v) if v.size else np.nan
            rec[f"n_windows_{ch}"] = int(v.size)
        rows.append(rec)

    pooled_rec = {"dataset": "pooled"}
    for ch in W1_CHANNELS:
        pooled = np.concatenate([data[d][ch] for d in datasets])
        pooled_rec[f"gini_{ch}"] = gini(pooled)
        pooled_rec[f"n_windows_{ch}"] = int(pooled.size)
    rows.append(pooled_rec)

    tab = pd.DataFrame(rows).set_index("dataset")
    return datasets, data, tab


def lorenz_curve(v, n_points=CURVE_POINTS):
    """(x, y) of the Lorenz curve of a non-negative vector, downsampled."""
    v = np.sort(v[np.isfinite(v)])
    if v.size == 0 or v.sum() <= 0:
        return None, None
    cum = np.concatenate([[0.0], np.cumsum(v) / v.sum()])
    x = np.arange(v.size + 1) / v.size
    idx = np.unique(np.linspace(0, v.size, n_points).astype(int))
    return x[idx], cum[idx]


# ═════════════════════════════ figure ══════════════════════════════
def fig_lorenz(datasets, data, tab):
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot([0, 1], [0, 1], color="black", linewidth=1, alpha=0.5,
            label="perfect equality")

    order = tab.loc[datasets, f"gini_{HEADLINE_CH}"] \
               .sort_values(ascending=False).index
    for ds in order:
        x, y = lorenz_curve(data[ds][HEADLINE_CH])
        if x is None:
            continue
        g = tab.loc[ds, f"gini_{HEADLINE_CH}"]
        ax.plot(x, y, color=color(ds), linewidth=1.6, alpha=0.9,
                label=f"{label(ds)} ({g:.2f})")

    xp, yp = lorenz_curve(
        np.concatenate([data[d][HEADLINE_CH] for d in datasets]))
    gp = tab.loc["pooled", f"gini_{HEADLINE_CH}"]
    ax.plot(xp, yp, color=TUM_PRIMARY["black"], linewidth=2.5,
            label=f"pooled ({gp:.2f})")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Cumulative fraction of 20 s windows (calmest first)")
    ax.set_ylabel("Cumulative fraction of speed-variability mass")
    ax.set_title("Dynamic concentration: Lorenz curves of window "
                 "\u03c3-speed\n(Legend: dynamic-content Gini per dataset)")
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    out = OUT_DIR / "dynamic_lorenz.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# ═══════════════════════════════ main ══════════════════════════════
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    datasets, data, tab = build()

    pd.set_option("display.width", 220, "display.max_columns", 40)
    shown = tab.copy()
    shown.index = [label(d) if d != "pooled" else "POOLED" for d in shown.index]
    gini_cols = [f"gini_{ch}" for ch in W1_CHANNELS]
    print("\nDynamic-content Gini per channel (* = confounded acceleration; "
          "acc-channel Gini of * datasets reflects fidelity):\n")
    print(shown[gini_cols + [f"n_windows_{HEADLINE_CH}"]]
          .round(4).sort_values(f"gini_{HEADLINE_CH}", ascending=False).to_string())

    # one ready-to-quote sentence on the headline channel, pooled corpus
    pooled_v = np.sort(np.concatenate([data[d][HEADLINE_CH] for d in datasets]))
    cum = np.cumsum(pooled_v) / pooled_v.sum()
    top20_mass = 1.0 - float(np.interp(0.8, np.arange(pooled_v.size) / pooled_v.size, cum))
    print(f"\nPooled corpus: the most dynamic 20 % of windows carry "
          f"{top20_mass * 100:.0f} % of the total speed-variability mass.")

    artifact = ARTIFACT_DIR / "bias_shape.parquet"
    tab.reset_index().to_parquet(artifact, index=False)
    print(f"\nArtifact: {artifact}")

    p = fig_lorenz(datasets, data, tab)
    print(f"Output: {p}")


if __name__ == "__main__":
    main()