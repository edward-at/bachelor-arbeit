#!/usr/bin/env python3
"""D17 figure (amended) — the Chapter-10 bias axes as paired bars.

Replaces the landscape scatter: with half the slate x-pinned at 0.20 by
construction, the 2D view collapses into one unreadable vertical strip.
The same two numbers in the thesis barh grammar instead — one panel per
bias axis, fixed slate order, guarded-frame context bar below a separator.
Reference lines: black = frame value; grey (left panel only) = the uniform
20 % balance target. At 12.3 the proposed method's two bars against the
frame lines carry the "reaches the good corner" reading directly.
Reads ONLY the eval cache. Output: figures/subset_eval/subset_bias_axes.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tum_colors import TUM_SECONDARY
from subset_eval_common import (DPI, LOW_DYN_REGIME, METHOD_COLORS,
                                N_REGIMES, ROW_LABELS, TIER, BUDGET_N,
                                ensure_outdir, load_master, present_methods,
                                regime_label)

UNIFORM = 1.0 / N_REGIMES


def main() -> None:
    out_dir = ensure_outdir()
    mt = load_master(TIER)
    methods = present_methods(mt)

    labels, rows = [], []
    for m in methods:
        labels.append(ROW_LABELS[m])
        rows.append(m)
    labels.append("")
    rows.append(None)
    labels.append(ROW_LABELS["R1_guarded"])
    rows.append("R1_guarded")

    n = len(labels)
    y = np.arange(n)[::-1]
    colors = [("white" if r is None else
               (TUM_SECONDARY["gray"] if r == "R1_guarded"
                else METHOD_COLORS[r])) for r in rows]

    panels = [
        ("low_dyn_share",
         f"Calm share\n({regime_label(LOW_DYN_REGIME)}, "
         f"regime {LOW_DYN_REGIME})", True),  # block 5F: the calm share of Section 4.10.6, not the two-regime headline of Section 4.8.3
        ("top_source_share", "Top-source share", False),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    for ax, (col, title, show_uniform) in zip(axes, panels):
        vals = [0.0 if r is None else float(mt.loc[r, col]) for r in rows]
        frame_v = float(mt.loc["R0_frame", col])
        ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.axvline(frame_v, color="black", linewidth=1.2)
        if show_uniform:
            ax.axvline(UNIFORM, color=TUM_SECONDARY["gray"],
                       linewidth=1.0, alpha=0.9)
        ax.axhline(y[-2] + 0.5, color=TUM_SECONDARY["gray"], linewidth=1.0)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)

    fig.suptitle(f"Bias axes per sampling method, (Black line = frame), (Grey line = "
                 f"uniform {UNIFORM:.0%} target)")
    fig.tight_layout()
    out = out_dir / "subset_bias_axes.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  [SAVED] {out}")


if __name__ == "__main__":
    main()