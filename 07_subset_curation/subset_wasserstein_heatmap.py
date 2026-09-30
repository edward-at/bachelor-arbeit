#!/usr/bin/env python3
"""F5 — per-channel W1 distance to the frame (two figures, budget tier).

Mirrors bias_wasserstein.py: horizontal-bar panels, one per channel, separate
axes because units differ (never a shared colour scale), black bar edges,
x-grid, suptitle carrying the asterisk note. Two figures: the IN-MODEL sigma
channels (the score's ingredients — construction-influenced for the
score/geometry designs) and the HELD-OUT summary channels no allocation
touches. max_speed and peak_acc_mag* are absent by design: sealed shelf (D5).
Rows keep the fixed slate order with the guarded frame as a context bar
below a separator (guard cost, D14). Bars in design-family colours.
Outputs: figures/subset_eval/subset_wasserstein_inmodel.png, _heldout.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tum_colors import TUM_SECONDARY
from subset_eval_common import (DPI, METHOD_COLORS, ROW_LABELS, TIER,
                                BUDGET_N, ensure_outdir, load_master,
                                present_methods)

FIGS = [
    ("inmodel", "in-model \u03c3 channels (score ingredients)", [
        ("w1_std_speed", "Speed variability\n1-Wasserstein (km/h)"),
        ("w1_std_acc",   "Accel-magnitude variability\n1-Wasserstein (m/s\u00b2) *"),
        ("w1_std_yaw",   "Yaw-rate variability\n1-Wasserstein (\u00b0/s)"),
    ]),
    ("heldout", "held-out summary channels", [
        ("w1_mean_speed",   "Mean speed\n1-Wasserstein (km/h)"),
        ("w1_peak_decel",   "Peak deceleration\n1-Wasserstein (g) *"),  # block 5F: the harness computes this W1 on decel_g (Section 4.10.6)
        ("w1_peak_abs_yaw", "Peak |yaw|\n1-Wasserstein (\u00b0/s)"),
    ]),
]


def plot_group(tag, subtitle, panels, mt, methods, out_dir):
    labels, rows = [], []
    for m in methods:
        labels.append(ROW_LABELS[m])
        rows.append(m)
    labels.append("")                       # spacer
    rows.append(None)
    labels.append(ROW_LABELS["R1_guarded"])
    rows.append("R1_guarded")

    n = len(labels)
    y = np.arange(n)[::-1]
    colors = [("white" if r is None else
               (TUM_SECONDARY["gray"] if r == "R1_guarded"
                else METHOD_COLORS[r])) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 7), sharey=True)
    for ax, (col, title) in zip(axes, panels):
        vals = [0.0 if r is None else float(mt.loc[r, col]) for r in rows]
        ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(y[-2] + 0.5, color=TUM_SECONDARY["gray"], linewidth=1.0)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)

    fig.suptitle(f"Per-channel 1-Wasserstein distance to the frame, "
                 f"{subtitle} (*)")
    fig.tight_layout()
    out = out_dir / f"subset_wasserstein_{tag}.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  [SAVED] {out}")


def main() -> None:
    out_dir = ensure_outdir()
    mt = load_master(TIER)
    methods = present_methods(mt)
    for tag, subtitle, panels in FIGS:
        cols = [c for c, _ in panels]
        if not all(c in mt.columns for c in cols):
            print(f"  [SKIP] {tag}: columns missing {cols}")
            continue
        plot_group(tag, subtitle, panels, mt, methods, out_dir)


if __name__ == "__main__":
    main()