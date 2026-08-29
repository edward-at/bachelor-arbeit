#!/usr/bin/env python3
"""curation_winner_pca.py, the winner's occupancy of the frozen PC plane (13.3).

Every other figure of the winner family reports marginals. A subset can
match six marginal distributions and still sit in a different region of
the joint feature space, so this figure reports the joint. Two panels, one
pass, both drawn from the same seeded sample of the winner's windows:

  left   the winner on the frozen principal-component plane over the
         frame's sample as a light-gray background, which answers where
         the subset sits
  right  the identical points coloured by their frozen regime label under
         the canonical Chapter 9 palette, which answers how the regimes
         are arranged inside that occupancy

The basis is Chapter 9's and is never re-fitted, and the labels are the
frozen k = 5 labels of Section 9.5. Conventions inherited verbatim: the
central-99.95-per-cent frame crop of Section 12.3 on both axes, with the
number of winner points falling outside it counted and printed, and the
seeded interleaved draw order so no regime overdraws another.

The panels carry no individual titles. The left panel is identified by the
figure title and the right by its legend, which is the only difference
between them. The coverage scalar is a table quantity and is reported
there rather than annotated here.

Reads the frozen caches only. Nothing is fitted, sampled afresh, or
recomputed.

Usage:  python curation_winner_pca.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import curation_common as cc
from regime_style import regime_color, regime_label

TIER = "budget"
SAMPLE_N = 12_000        # points per group, seeded
DRAW_SEED = 42
CROP_PCT = 99.95         # Section 12.3 convention
CROP_PAD = 0.03
POINT_S = 3
FRAME_ALPHA = 0.20
WIN_ALPHA = 0.5


def crop_bounds(frame: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(frame, [(100 - CROP_PCT) / 2,
                                   100 - (100 - CROP_PCT) / 2])
    pad = CROP_PAD * (hi - lo)
    return lo - pad, hi + pad


def main() -> None:
    win = cc.WINNER
    if not win:
        raise SystemExit("[pca] curation_common.WINNER is unset")

    cls = pd.read_parquet(cc.PROJECT_ROOT / "dataset" / "regime_data"
                          / "cluster_assignments.parquet")
    need = ["pc1", "pc2", "regime"]
    missing = [c for c in need if c not in cls.columns]
    if missing:
        raise SystemExit(f"[pca] assignment table lacks {missing}")

    keys = [c for c in ["dataset", "clip_id", "window_start_s"]
            if c in cls.columns]
    sub = pd.read_parquet(cc.PROJECT_ROOT / "dataset" / "subset_data"
                          / f"tier_n{cc.BUDGET_N}" / cc.C_FILES[win])
    use = [k for k in keys if k in sub.columns]
    joined = sub[use].merge(cls[use + need], on=use, how="left")
    n_lost = int(joined["pc1"].isna().sum())
    if n_lost:
        print(f"[pca] {n_lost} winner windows failed the key join, excluded")
        joined = joined.dropna(subset=need)

    rng = np.random.default_rng(DRAW_SEED)
    fs = cls.iloc[rng.choice(len(cls), size=min(SAMPLE_N, len(cls)),
                             replace=False)]
    ws = joined.iloc[rng.choice(len(joined),
                                size=min(SAMPLE_N, len(joined)),
                                replace=False)]
    print(f"[pca] frame sample {len(fs):,} | {win} sample {len(ws):,} "
          f"of {len(joined):,} joined windows")

    lims = [crop_bounds(cls[c].to_numpy(dtype=float)) for c in
            ("pc1", "pc2")]
    inside = np.ones(len(ws), dtype=bool)
    for c, (lo, hi) in zip(("pc1", "pc2"), lims):
        v = ws[c].to_numpy(dtype=float)
        inside &= (v >= lo) & (v <= hi)
    n_out = int((~inside).sum())

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.8),
                             sharex=True, sharey=True)
    for col, ax in enumerate(axes):
        ax.scatter(fs["pc1"], fs["pc2"], s=POINT_S, c=cc.GHOST,
                   alpha=FRAME_ALPHA, edgecolor="none")
        order = rng.permutation(len(ws))          # interleave, Section 12.3
        w = ws.iloc[order]
        if col == 0:
            ax.scatter(w["pc1"], w["pc2"], s=POINT_S,
                       c=cc.CURATION_COLOR, alpha=WIN_ALPHA,
                       edgecolor="none")
        else:
            for r in sorted(w["regime"].dropna().unique()):
                g = w[w["regime"] == r]
                ax.scatter(g["pc1"], g["pc2"], s=POINT_S,
                           c=[regime_color(int(r))], alpha=WIN_ALPHA,
                           edgecolor="none")
            handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                                  color=regime_color(i), markersize=5,
                                  label=regime_label(i)) for i in range(5)]
            ax.legend(handles=handles, fontsize=8, loc="upper right")
        ax.set_xlabel("PC 1")
        ax.set_xlim(*lims[0])
        ax.set_ylim(*lims[1])
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.4)
    axes[0].set_ylabel("PC 2")

    fig.suptitle(f"Occupancy of the frozen principal-component plane, "
                 f"{win} over the frame sample")
    fig.tight_layout()
    out = cc.ensure_curation_outdir() / "curation_winner_pca.png"
    fig.savefig(out, dpi=cc.DPI)
    plt.close(fig)
    print(f"[pca] wrote {out.name}")
    print(f"[pca] {n_out} winner points fall outside the central "
          f"{CROP_PCT:g}% frame crop and are drawn off-axis")
    print("[pca] the basis is Chapter 9's and was not re-fitted, and the "
          "labels are\nthe frozen k = 5 labels of Section 9.5.")


if __name__ == "__main__":
    main()