#!/usr/bin/env python3
"""Per-method exemplar speed traces (Section 8.8 grammar, adapted).

One figure per group: the EX_N exemplar windows flagged in the frame pass,
drawn as speed-vs-time traces on a shared time axis [0, 20 s] and a shared
y-range, so a method's typical within-window speed profile is legible and
comparable across methods. The parent figure faceted by regime BAND; here the
exemplars are the method's own seeded sample (the subset is a bag of windows,
not a per-band selection), which is the honest adaptation to the selection
grain. A light grey band shows the method's fan-sample envelope (10th-90th
speed percentile per time step) behind the exemplars.
Reads ONLY the fan caches. Outputs: figures/subset_eval/timeseries/…
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tum_colors import TUM_ACCENT
from subset_eval_common import (DPI, EVAL_DIR, ROW_LABELS, TUM_SECONDARY,
                                ensure_outdir)

WINDOW_S = 20.0
GRID_HZ = 10.0
GROUPS = ["R0_frame"] + [f"M{i:02d}" for i in range(1, 12)]
REP_COLORS = [TUM_ACCENT["blue"], TUM_ACCENT["orange"], TUM_ACCENT["green"]]
FIG_W, FIG_H = 9.0, 4.5


def main():
    out_dir = ensure_outdir() / "timeseries"
    out_dir.mkdir(exist_ok=True)
    frames = pd.read_parquet(EVAL_DIR / "fan_frames.parquet")
    idx = pd.read_parquet(EVAL_DIR / "fan_index.parquet").set_index("key_id")
    if frames.empty:
        raise SystemExit("fan_frames empty — run subset_frame_pass.py first")

    grid = np.linspace(0, WINDOW_S, int(WINDOW_S * GRID_HZ) + 1)
    y_hi = 0.0
    traces_by_group = {g: [] for g in GROUPS}
    env_by_group = {g: [] for g in GROUPS}

    for kid, seq in frames.groupby("key_id"):
        t = seq["t"].to_numpy(float)
        v = seq["speed_kmh"].to_numpy(float)
        o = np.argsort(t)
        t, v = t[o], v[o]
        m = np.isfinite(t) & np.isfinite(v)
        if m.sum() < 5:
            continue
        vg = np.interp(grid, t[m], v[m], left=np.nan, right=np.nan)
        smask = int(idx.loc[kid, "sample_mask"])
        emask = int(idx.loc[kid, "exemplar_mask"])
        for b, g in enumerate(GROUPS):
            if (smask >> b) & 1:
                env_by_group[g].append(vg)
            if (emask >> b) & 1:
                traces_by_group[g].append(vg)
                y_hi = max(y_hi, np.nanmax(vg))

    for g in GROUPS:
        traces = traces_by_group[g]
        if not traces:
            continue
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        env = np.array(env_by_group[g])
        if len(env) >= 10:
            lo = np.nanpercentile(env, 10, axis=0)
            hi = np.nanpercentile(env, 90, axis=0)
            ax.fill_between(grid, lo, hi, color=TUM_SECONDARY["gray"],
                            alpha=0.25, label="sample 10\u201390th %ile")
        for i, tr in enumerate(traces[:3]):
            ax.plot(grid, tr, color=REP_COLORS[i], linewidth=1.4,
                    label=f"exemplar {i + 1}")
        ax.set_xlim(0, WINDOW_S)
        ax.set_ylim(0, max(y_hi * 1.05, 1.0))
        ax.set_xlabel("Time within window (s)")
        ax.set_ylabel("Speed (km/h)")
        ax.set_title(f"Exemplar speed traces ({ROW_LABELS[g]})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        fig.tight_layout()
        rid = "frame" if g == "R0_frame" else g
        fig.savefig(out_dir / f"timeseries_{rid}.png", dpi=DPI,
                    bbox_inches="tight")
        plt.close(fig)
    print(f"  [SAVED] exemplar traces for "
          f"{sum(1 for g in GROUPS if traces_by_group[g])} groups")


if __name__ == "__main__":
    main()