#!/usr/bin/env python3
"""Per-method phase portraits (Section 8.5 grammar, methods as units).

Two figures per group (speed x longitudinal acceleration, speed x yaw rate)
from the cached 2D histograms of subset_frame_pass.py: fraction-normalised,
viridis on a shared cross-method log colour scale per view (the mechanism
that made the 28 per-dataset portraits comparable, re-pointed), white zero
line, colorbar 'fraction of frames (log)'.
Reads ONLY the frame caches. Outputs: figures/subset_eval/phase/…
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from subset_eval_common import DPI, EVAL_DIR, ROW_LABELS, ensure_outdir

VMIN_RATIO = 1.0e4
CMAP = "viridis"
FIG_W, FIG_H = 10, 6
VIEWS = [("speed_along", "Ha", "long. accel. [m/s\u00b2]",
          "Speed \u00d7 longitudinal acceleration"),
         ("speed_yaw", "Hy", "yaw rate [\u00b0/s]", "Speed \u00d7 yaw rate")]


def main():
    out_dir = ensure_outdir() / "phase"
    out_dir.mkdir(exist_ok=True)
    z = np.load(EVAL_DIR / "frame_portraits.npz", allow_pickle=True)
    methods = [str(m) for m in z["methods"]]
    speed_edges = z["speed_edges"]
    y_edges = {"speed_along": z["along_edges"], "speed_yaw": z["yaw_edges"]}

    for tag, arr, ylab, title in VIEWS:
        H = z[arr]
        frac = np.array([h / h.sum() if h.sum() > 0 else h for h in H])
        vmax = max(frac.max(), 1e-12)
        norm = LogNorm(vmin=vmax / VMIN_RATIO, vmax=vmax)
        ye = y_edges[tag]
        extent = [speed_edges[0], speed_edges[-1], ye[0], ye[-1]]
        for i, m in enumerate(methods):
            if frac[i].sum() == 0:
                continue
            fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
            masked = np.ma.masked_where(frac[i] <= 0, frac[i])
            im = ax.imshow(masked.T, origin="lower", extent=extent,
                           aspect="auto", cmap=CMAP, norm=norm)
            ax.axhline(0.0, color="white", linewidth=0.6, alpha=0.5)
            ax.set_xlabel("speed [m/s]")
            ax.set_ylabel(ylab)
            ax.set_title(f"{title} ({ROW_LABELS.get(m, m)})")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04
                         ).set_label("fraction of frames (log)")
            fig.tight_layout()
            rid = "frame" if m == "R0_frame" else m
            out = out_dir / f"{rid}_{tag}.png"
            fig.savefig(out, dpi=DPI, bbox_inches="tight")
            plt.close(fig)
        print(f"  [SAVED] {tag}: {len(methods)} portraits")


if __name__ == "__main__":
    main()