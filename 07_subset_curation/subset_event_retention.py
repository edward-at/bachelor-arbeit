#!/usr/bin/env python3
"""F7 — event & predicate retention per method (budget tier).

Six panels, one per M08 predicate (the 0.6 g trigger is the hard-decel
panel): bars of retention ratio = (rate in subset) / (rate in frame).
Ratio 1 = exactly what random keeps (black line, grey SRS envelope band);
the thin upper line is the full-capture ceiling N_frame/N_subset — a method
touching it holds EVERY qualifying window. Guarded predicates (* panels)
are evaluated exactly as M08 defines them: confounded-accel sources are
ineligible for the accel-consuming predicates.
Output: figures/subset_eval/subset_event_retention.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from subset_eval_common import (DPI, MAIN_BLUE, TIER, BUDGET_N,
                                envelope_band, ensure_outdir, load_master,
                                present_methods)

PANELS = [
    ("ret_hard_decel", "hard decel \u2265 0.6 g *"),
    ("ret_high_yaw", "high yaw \u2265 30 \u00b0/s"),
    ("ret_high_speed", "sustained \u2265 100 km/h"),
    ("ret_stop_go", "stop-and-go"),
    ("ret_near_stat", "near-stationary"),
    ("ret_top_score", "top-percentile score *"),
]


def main() -> None:
    out_dir = ensure_outdir()
    mt = load_master(TIER)
    methods = present_methods(mt)
    ceiling = float(mt.loc["R0_frame", "n_windows"]) / BUDGET_N

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    x = np.arange(len(methods))
    for ax, (metric, title) in zip(axes.ravel(), PANELS):
        vals = mt.loc[methods, metric].values.astype(float)
        band = envelope_band(metric)
        if band is not None:
            ax.axhspan(band[0], band[1], color="gray", alpha=0.25)
        ax.axhline(1.0, color="black", linewidth=1.0)
        ax.axhline(ceiling, color="gray", linewidth=0.8, alpha=0.7)
        ax.bar(x, vals, color=MAIN_BLUE, edgecolor="black",
               linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=0)
        ax.set_title(title)
        ax.set_ylim(0, ceiling * 1.08)
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("Retention ratio vs frame")
    fig.suptitle(f"Predicate retention per sampling method "
                 f"(line = SRS null), (band = K-draw envelope), "
                 f"(upper line = full-capture ceiling {ceiling:.2f})")
    fig.tight_layout()
    out = out_dir / "subset_event_retention.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()