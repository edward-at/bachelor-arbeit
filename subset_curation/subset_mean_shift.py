#!/usr/bin/env python3
"""Within-regime mean shift — the introduced-bias check, all methods.

Three panels (one per sigma channel, units differ so scales never mix):
methods x regimes heatmap of (subset within-regime mean minus frame
within-regime mean) in physical units, diverging colormap centred at zero
(explicit-diverging exception), values printed. Answers: did selection drag
a regime's average kinematics away from the corpus — e.g. k-center pulling
regime means upward — even where the regime SHARES are fixed by quota?
Reads the shift_* metrics of harness v12.1.2. Output:
figures/subset_eval/subset_mean_shift.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from subset_eval_common import (DPI, N_REGIMES, ROW_LABELS, TIER, BUDGET_N,
                                ensure_outdir, load_master, metric_matrix,
                                present_methods, regime_label)

CELL_FS = 8   # documented fontsize exception: in-cell annotations
PANELS = [("speed", "\u0394 mean \u03c3(speed) [km/h]"),
          ("acc",   "\u0394 mean \u03c3(acc_mag) [m/s\u00b2] *"),
          ("yaw",   "\u0394 mean \u03c3(yaw) [\u00b0/s]")]


def main() -> None:
    out_dir = ensure_outdir()
    mt = load_master(TIER)
    methods = present_methods(mt)
    shifts = metric_matrix("shift_", TIER)
    if shifts.empty:
        raise SystemExit("shift_* metrics missing — re-run "
                         "subset_eval_harness.py v12.1.2+")

    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2), sharey=True)
    for ax, (key, title) in zip(axes, PANELS):
        cols = [f"shift_r{r}_{key}" for r in range(N_REGIMES)]
        mat = shifts.reindex(methods)[cols].to_numpy(dtype=float)
        vmax = max(1e-9, float(np.nanmax(np.abs(mat))))
        ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                ax.text(j, i, "\u2013" if not np.isfinite(v) else f"{v:+.2f}",
                        ha="center", va="center", fontsize=CELL_FS)
        ax.set_xticks(range(N_REGIMES))
        ax.set_xticklabels([regime_label(r) for r in range(N_REGIMES)],
                           rotation=30, ha="right")
        ax.set_title(title)
    axes[0].set_yticks(range(len(methods)))
    axes[0].set_yticklabels([ROW_LABELS[m] for m in methods])

    fig.suptitle(f"Within-regime mean shift vs the frame — budget tier "
                 f"(N = {BUDGET_N:,}); diverging scale per panel, "
                 f"zero = no introduced bias")
    fig.tight_layout()
    out = out_dir / "subset_mean_shift.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  [SAVED] {out}")


if __name__ == "__main__":
    main()