#!/usr/bin/env python3
"""Regime composition per sampling method (budget tier).

Mirrors the Section 4.8 headline grammar (bias_stacked_bar.py): HORIZONTAL
stacked bars in percent, regime colours from regime_style, white segment
edges, and the reference bars BELOW a separator — here Frame (R0) and the
guarded frame (R1) take the place of Pooled/Balanced. One deliberate
deviation, stated: rows keep the fixed M01..M11 slate order instead of a
headline sort, because ordering methods by outcome before the decision rule
exists would presuppose the rule. Identical bars for M03/M05/M09/M10
are the construction (all four use the equal regime quotas of M03; M05, M09
and M10 change only WHICH windows fill them), pre-stated in Section 4.9.9.
Reads ONLY the eval cache. Output: figures/subset_eval/subset_stacked_bar.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tum_colors import TUM_SECONDARY
from subset_eval_common import (DPI, N_REGIMES, ROW_LABELS, TIER, BUDGET_N,
                                ensure_outdir, load_master, present_methods,
                                regime_color, regime_label)


def main() -> None:
    out_dir = ensure_outdir()
    mt = load_master(TIER)
    methods = present_methods(mt)

    labels, mat = [], []
    for m in methods:                                   # slate order, top-down
        labels.append(ROW_LABELS[m])
        mat.append([mt.loc[m, f"share_r{r}"] * 100 for r in range(N_REGIMES)])
    labels.append("")                                   # spacer (invisible)
    mat.append([0.0] * N_REGIMES)
    for ref in ["R0_frame", "R1_guarded"]:
        labels.append(ROW_LABELS[ref])
        mat.append([mt.loc[ref, f"share_r{r}"] * 100 for r in range(N_REGIMES)])
    mat = np.array(mat)

    n = len(labels)
    y = np.arange(n)[::-1]                              # first row at the top
    fig, ax = plt.subplots(figsize=(12, 7.5))
    left = np.zeros(n)
    for i in range(N_REGIMES):
        ax.barh(y, mat[:, i], left=left, color=regime_color(i),
                edgecolor="white", linewidth=0.6, label=regime_label(i))
        left += mat[:, i]

    ax.axhline(y[-3] + 0.5, color=TUM_SECONDARY["gray"], linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.set_ylim(-2.6, y.max() + 0.8)     # empty band so the legend covers no bar
    ax.set_xlabel("Share of 20 s windows (%)")
    ax.set_title(f"Per-method dynamic-regime composition")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", ncol=N_REGIMES)

    fig.tight_layout()
    out = out_dir / "subset_stacked_bar.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  [SAVED] {out}")


if __name__ == "__main__":
    main()