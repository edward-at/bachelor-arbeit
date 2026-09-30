"""
state_timeline.py  —  Section 4.6.9 figure
Reads hmm_states.parquet → the decoded most-likely regime sequence along each
example drive, the succession of regimes made legible as a trace through time.
One stacked panel per example clip; the y-axis spans all regimes so panels are
comparable. Thin reader (k read from hmm_transitions.parquet for the y-axis).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tum_colors import TUM_SECONDARY
from regime_style import regime_label, regime_color

PROJECT_ROOT    = Path("/Users/edward/Downloads")
REGIME_DATA_DIR = PROJECT_ROOT / "dataset" / "regime_data"
OUT_DIR         = PROJECT_ROOT / "bachelor-arbeit" / "05_regime_discovery" / "figures" / "hmm"

KEY        = ["dataset", "clip_id"]
# short names of Section 4.2.3 for the panel labels (block 5D); folder keys otherwise
DISPLAY_NAMES = {"4seasons": "4Seasons", "a2d2": "A2D2", "boreas_rt": "Boreas", "durlar": "DurLAR",
                 "ford": "Ford", "hdd": "HDD", "kitscenes": "KITScenes", "kitti": "KITTI",
                 "kitti360": "KITTI-360", "nuscene": "nuScenes", "nvidia": "NVIDIA",
                 "oxford": "Oxford", "waymo": "Waymo", "zod": "ZOD"}
GAP_S      = 1.5         # a jump larger than the 1 s step means dropped sub-windows: the line breaks there
TIMELINE_S = 10          # marker size — same legibility latitude as the s=4 scatter exception
DPI        = 300


def main():
    tl = pd.read_parquet(REGIME_DATA_DIR / "hmm_states.parquet")
    k  = pd.read_parquet(REGIME_DATA_DIR / "hmm_transitions.parquet").shape[0]

    # clips in first-appearance order (hmm.py wrote the longest first)
    clips = list(dict.fromkeys(map(tuple, tl[KEY].to_numpy())))
    n = len(clips)

    fig, axes = plt.subplots(n, 1, figsize=(11, 2.3 * n))
    if n == 1:
        axes = [axes]

    for ax, (ds, cid) in zip(axes, clips):
        g = tl[(tl["dataset"] == ds) & (tl["clip_id"] == cid)].sort_values("window_start_s")
        t = (g["window_start_s"] - g["window_start_s"].min()) / 60.0   # minutes
        # break the connecting line where sub-windows were dropped (4.7.11 validity
        # rule), so a gap is not drawn as a run of the regime on either side (block 5D)
        y = g["regime"].to_numpy(float)
        gaps = np.flatnonzero(np.diff(g["window_start_s"].to_numpy()) > GAP_S)
        t_line, y_line = t.to_numpy(float), y.copy()
        if len(gaps):
            t_line = np.insert(t_line, gaps + 1, np.nan)
            y_line = np.insert(y_line, gaps + 1, np.nan)
        ax.plot(t_line, y_line, color=TUM_SECONDARY["gray"], linewidth=0.8, alpha=0.6)
        for r in range(k):
            gr = g[g["regime"] == r]
            if len(gr):
                tr = (gr["window_start_s"] - g["window_start_s"].min()) / 60.0
                ax.scatter(tr, gr["regime"], color=regime_color(r), s=TIMELINE_S,
                           edgecolor="none")
        ax.set_yticks(range(k))
        ax.set_yticklabels([regime_label(r) for r in range(k)])
        ax.set_ylim(-0.5, k - 0.5)
        ax.set_ylabel(f"{DISPLAY_NAMES.get(ds, ds)}\nclip {int(cid)}")
        ax.grid(True); ax.set_axisbelow(True)
    axes[-1].set_xlabel("Time (min)")
    fig.suptitle("Decoded regime timelines (longest example drives)")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "state_timeline.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  → {out}")


if __name__ == "__main__":
    main()