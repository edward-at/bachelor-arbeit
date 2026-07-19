#!/usr/bin/env python3
"""F2 — per-method regime profiles (small-multiple grid, budget tier).

Mirrors bias_regime_grid.py verbatim: one panel per method, five regime bars
in the canonical regime colours, black bar edges, a thin grey line at the
uniform share (1/N = 20 %), shared y-axis with a dynamic ceiling, regime key
in the suptitle. The Frame profile is the leading panel. Panels follow the
fixed slate order (see subset_stacked_bar.py for the D4 grounds).
Reads ONLY the eval cache. Output: figures/subset_eval/subset_regime_grid.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tum_colors import TUM_SECONDARY
from subset_eval_common import (DPI, N_REGIMES, ROW_LABELS, TIER, BUDGET_N,
                                ensure_outdir, load_master, present_methods,
                                regime_color, regime_label)

NROWS, NCOLS = 2, 6
UNIFORM_PCT = 100.0 / N_REGIMES


def main() -> None:
    out_dir = ensure_outdir()
    mt = load_master(TIER)
    rows = ["R0_frame"] + present_methods(mt)

    share_cols = [f"share_r{r}" for r in range(N_REGIMES)]
    ymax = min(100.0, np.ceil((float(mt.loc[rows, share_cols].values.max())
                               * 100.0 + 5.0) / 5.0) * 5.0)

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(16, 6.5), sharey=True)
    flat = axes.ravel()
    for k, r in enumerate(rows):
        ax = flat[k]
        vals = [mt.loc[r, c] * 100 for c in share_cols]
        ax.bar(range(N_REGIMES), vals,
               color=[regime_color(i) for i in range(N_REGIMES)],
               edgecolor="black", linewidth=0.5)
        ax.axhline(UNIFORM_PCT, color=TUM_SECONDARY["gray"],
                   linewidth=1.0, alpha=0.8)
        ax.set_title(ROW_LABELS[r])
        ax.set_xticks(range(N_REGIMES))
        ax.set_xticklabels([str(i) for i in range(N_REGIMES)])
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    for k in range(len(rows), NROWS * NCOLS):
        flat[k].axis("off")
    axes[0][0].set_ylabel("share of windows (%)")
    axes[1][0].set_ylabel("share of windows (%)")

    key = " \u00b7 ".join(f"{i} {regime_label(i)}" for i in range(N_REGIMES))
    fig.suptitle(f"Per-method regime profiles — budget tier "
                 f"(N = {BUDGET_N:,}); grey line = uniform 20 %\n"
                 f"Regimes: {key}")
    fig.tight_layout()
    out = out_dir / "subset_regime_grid.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  [SAVED] {out}")


if __name__ == "__main__":
    main()