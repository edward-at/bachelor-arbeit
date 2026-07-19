#!/usr/bin/env python3
"""F6 — source composition per sampling method (budget tier).

The second bias axis: did the method fix supplier concentration or just
reshuffle regimes inside NVIDIA? Stacked source shares in the thesis-wide
DATASET_COLORS (bias_tier2_common convention), stacked in descending frame
share so the dominant supplier sits at the bottom of every bar; Frame leads
behind a separator. Above each bar: volume-share Gini over effective sources
D2. Legend outside the axes so no bar is ever covered.
Reads src_share_* from metrics_long. Output:
figures/subset_eval/subset_source_composition.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from subset_eval_common import (DATASET_COLORS, DATASET_COLOR_FALLBACK,
                                DISPLAY_NAMES, DPI, ROW_LABELS, TIER,
                                BUDGET_N, ensure_outdir, load_master,
                                metric_matrix, present_methods)

ANNOT_FS = 8   # documented fontsize exception: per-bar Gini/D2 annotations


def main() -> None:
    out_dir = ensure_outdir()
    mt = load_master(TIER)
    methods = present_methods(mt)
    rows = ["R0_frame"] + methods

    src = metric_matrix("src_share_", TIER).reindex(rows).fillna(0.0)
    src.columns = [c.replace("src_share_", "") for c in src.columns]
    order = src.loc["R0_frame"].sort_values(ascending=False).index.tolist()

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(13.5, 6.5))
    bottom = np.zeros(len(rows))
    for ds in order:
        vals = src[ds].values.astype(float)
        ax.bar(x, vals, bottom=bottom,
               color=DATASET_COLORS.get(ds, DATASET_COLOR_FALLBACK),
               edgecolor="white", linewidth=0.3,
               label=DISPLAY_NAMES.get(ds, ds))
        bottom += vals

    for i, r in enumerate(rows):
        ax.text(i, 1.015,
                f"{mt.loc[r, 'volume_gini']:.2f}\n{mt.loc[r, 'd2_sources']:.1f}",
                ha="center", va="bottom", fontsize=ANNOT_FS)

    ax.axvline(0.5, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([ROW_LABELS[r] for r in rows], rotation=45, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Source share")
    ax.set_title(f"Source composition per sampling method — budget tier "
                 f"(N = {BUDGET_N:,}); annotations: volume Gini / D\u2082")
    ax.legend(loc="center left", bbox_to_anchor=(1.005, 0.5))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = out_dir / "subset_source_composition.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {out}")


if __name__ == "__main__":
    main()