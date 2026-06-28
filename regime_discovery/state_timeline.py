"""
state_timeline.py  —  Section 4.6.9 figure
Reads hmm_states.parquet → the decoded most-likely regime sequence along each
example drive, the succession of regimes made legible as a trace through time.
One stacked panel per example clip; the y-axis spans all regimes so panels are
comparable. Thin reader (k read from hmm_transitions.parquet for the y-axis).
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from tum_colors import TUM_SECONDARY
from regime_style import regime_label, regime_color

PROJECT_ROOT    = Path("/Users/edward/Downloads")
REGIME_DATA_DIR = PROJECT_ROOT / "dataset" / "regime_data"
OUT_DIR         = PROJECT_ROOT / "bachelor-arbeit" / "regime_discovery" / "figures" / "hmm"

KEY        = ["dataset", "clip_id"]
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
        ax.plot(t, g["regime"], color=TUM_SECONDARY["gray"], linewidth=0.8, alpha=0.6)
        for r in range(k):
            gr = g[g["regime"] == r]
            if len(gr):
                tr = (gr["window_start_s"] - g["window_start_s"].min()) / 60.0
                ax.scatter(tr, gr["regime"], color=regime_color(r), s=TIMELINE_S,
                           edgecolor="none")
        ax.set_yticks(range(k))
        ax.set_yticklabels([regime_label(r) for r in range(k)])
        ax.set_ylim(-0.5, k - 0.5)
        ax.set_ylabel(f"{ds}\nclip {int(cid)}")
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