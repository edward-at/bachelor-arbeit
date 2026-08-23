#!/usr/bin/env python3
"""curation_feasibility.py — per-dataset internal-balance retention (13.1).

The figure behind the pooling justification: a regime-balanced subset of a
single dataset can keep at most 5 x (its rarest regime's share) of that
dataset. Pooled, the frame keeps 88.4 per cent; most individual datasets
collapse to a few per cent because they lack entire regimes. One bar per
dataset, the pooled frame as the reference line. Winner-independent, so it
may be produced before the gates run (D25 untouched: it reads no method).

Output: figures/curation/curation_feasibility.png (300 dpi)
"""

from __future__ import annotations

import matplotlib.pyplot as plt

import run_sampling as rs
from curation_common import (CONFOUNDED, DISPLAY_NAMES, MAIN_BLUE, DPI,
                             ensure_curation_outdir)

N_REGIMES = 5


def main() -> None:
    pool_full, _, _ = rs.load_pool()
    pool = rs.apply_dedup(pool_full)

    shares = (pool.pivot_table(index="dataset", columns="regime",
                               values="window_idx", aggfunc="count",
                               fill_value=0))
    totals = shares.sum(axis=1)
    retention = (N_REGIMES * shares.min(axis=1) / totals).clip(upper=1.0)
    retention = retention.sort_values()

    frame_ret = min(1.0, N_REGIMES
                    * pool["regime"].value_counts().min() / len(pool))

    labels = [DISPLAY_NAMES.get(d, d) + ("*" if d in CONFOUNDED else "")
              for d in retention.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, 100 * retention.values, color=MAIN_BLUE,
            edgecolor="black", linewidth=0.5)
    ax.axvline(100 * frame_ret, color="black", linestyle="--", linewidth=1,
               label=f"pooled frame ({100 * frame_ret:.1f}%)")
    ax.set_xlabel("share of dataset retained by internal regime "
                  "balancing (%)")
    ax.set_title("Per-dataset feasibility of internal balancing\n"
                 "retention = 5 x rarest-regime share")
    ax.set_axisbelow(True)
    ax.grid(True, axis="x", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    out = ensure_curation_outdir() / "curation_feasibility.png"
    fig.savefig(out, dpi=DPI)
    print(f"[feasibility] wrote {out}")
    print(retention.mul(100).round(1).rename("retained_%").to_string())


if __name__ == "__main__":
    main()