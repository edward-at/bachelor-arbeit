#!/usr/bin/env python3
"""T2 — ECDF overlays for the finalists (budget tier).

Four panels (dynamic score, sigma speed, sigma acc*, sigma yaw): empirical
CDFs of the frame (black) and the finalist methods. Tier-2 by design: ECDF
overlays are only readable at 3-5 lines, which is why the all-methods
distribution view is the ridgeline and THIS figure waits for a finalist
pick.

FINALISTS below is FIXED by register entry D27 (frame + M01 + M05 + M07)
(M05 = only source-breaker, M08 = balance without quotas + full event
capture, M09 = coverage champion). Confirm or replace it after reviewing
the T1 figures; the choice is analytic, not mine to lock.
Reads ONLY the eval cache. Output: figures/subset_eval/subset_ecdf_finalists.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from subset_eval_common import (DPI, ROW_LABELS, TIER, BUDGET_N, TUM_ACCENT,
                                TUM_EXT, dist_arrays, ensure_outdir,
                                load_dist, load_master, present_methods)

# PROVISIONAL finalist pick — edit after reviewing the Tier-1 results.
FINALISTS = ["M01", "M05", "M07"]   # D27: frame + M01 + M05 + M07; the winner joins via the Ch.13 renderer
PALETTE = [TUM_ACCENT["blue"], TUM_ACCENT["orange"], TUM_ACCENT["green"],
           TUM_EXT["purple"], TUM_EXT["red"]]

PANELS = [
    ("score",         "Dynamic score"),
    ("std_speed_kmh", "\u03c3(speed) [km/h]"),
    ("std_acc_mag",   "\u03c3(acc_mag) [m/s\u00b2] *"),
    ("std_yaw_deg",   "\u03c3(yaw) [\u00b0/s]"),
]
X_CLIP_Q = 99.5


def ecdf(vals: np.ndarray):
    x = np.sort(vals)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def main() -> None:
    out_dir = ensure_outdir()
    dist = load_dist(TIER)
    have = set(present_methods(load_master(TIER)))
    finalists = [m for m in FINALISTS if m in have]
    rows = ["R0_frame"] + finalists

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (ch, label) in zip(axes.ravel(), PANELS):
        arrays = dist_arrays(dist, ch, rows)
        arrays = [a[np.isfinite(a)] for a in arrays]
        x_hi = float(np.percentile(arrays[0], X_CLIP_Q))
        for row, vals, color in zip(
                rows, arrays,
                ["black"] + PALETTE[:len(finalists)]):
            if len(vals) < 10:
                continue
            x, y = ecdf(vals)
            lw = 1.8 if row == "R0_frame" else 1.3
            ax.plot(x, y, color=color, linewidth=lw,
                    label=ROW_LABELS[row])
        ax.set_xlim(0, x_hi)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"{label} (clipped at frame p{X_CLIP_Q})")
        ax.set_ylabel("ECDF")
        ax.legend()
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3)
    # 2026-08-23: title cleaned to the thesis text rules (no em dash, no
    # semicolon, no tier name) and the finalist list is fixed by register
    # entry D27, no longer provisional. Content unchanged.
    fig.suptitle("ECDF overlays, frame and finalists "
                 "(finalist register D27)")
    fig.tight_layout()
    out = out_dir / "subset_ecdf_finalists.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"saved {out}  (finalists: {finalists})")


if __name__ == "__main__":
    main()