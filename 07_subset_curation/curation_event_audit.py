#!/usr/bin/env python3
"""curation_event_audit.py — pre-selection diagnostics for the curation stage (thesis 4.11).

Purely diagnostic: reads the frozen selection frame, writes nothing, selects
nothing, and touches no gate metric, so running it before the gates are
applied is clean. Two questions, both console tables:

  A. Where do the guarded 0.6 g reserve windows come from?
     Per-dataset and per-regime composition of the event reserve that C01
     and C03 will force-include. The realised count on the current frame
     is printed and becomes the frozen constant for the build.

  B. What supply does the evasive conjunction have?
     peak|yaw| >= lat_g * g / mean_speed, swept over EVASIVE_SWEEP_G, with
     overlap against the hard-decel reserve and per-dataset composition at
     the primary candidate. The naive M08-intersection (30 deg/s AND
     100 km/h) is counted too, to document that physics
     empties it.

Usage:  python curation_event_audit.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import run_sampling as rs
from curation_common import (CONFOUNDED, DISPLAY_NAMES, EVASIVE_SWEEP_G,
                             EVASIVE_LAT_G, evasive_mask)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)


def name(ds: str) -> str:
    star = "*" if ds in CONFOUNDED else ""
    return DISPLAY_NAMES.get(ds, ds) + star


def main() -> None:
    pool_full, _, have_summaries = rs.load_pool()
    if not have_summaries:
        raise SystemExit("[audit] window_summaries.parquet missing — the "
                         "audit needs the Section 11.4 summary layer")
    pool = rs.apply_dedup(pool_full)
    rs.check_summary_coverage(pool)
    n_frame = len(pool)
    guarded = pool[~pool["dataset"].isin(rs.CONFOUNDED_ACCEL)]
    print(f"frame (dedup) = {n_frame:,} | guarded = {len(guarded):,}")

    # ---- A: the reserve -------------------------------------------------
    thr = rs.DECEL_TRIGGER_G * rs.G_MS2
    ev = guarded[guarded["max_decel_ms2"] >= thr]
    print(f"\n=== A. Hard-decel reserve ({rs.DECEL_TRIGGER_G} g, guarded) ===")
    print(f"n_event = {len(ev):,}  ({100 * len(ev) / len(guarded):.3f}% of "
          f"guarded frame)  [superseded canaries: 1,124 pre-restoration, 1,115 pre-v4_4]")

    by_ds = (ev.groupby("dataset").size().sort_values(ascending=False)
             .rename("events").to_frame())
    by_ds["share_%"] = 100 * by_ds["events"] / len(ev)
    g_supply = guarded.groupby("dataset").size()
    by_ds["rate_in_dataset_%"] = 100 * by_ds["events"] / g_supply
    by_ds.index = [name(d) for d in by_ds.index]
    print("\nreserve by SOURCE (D22 imports this composition verbatim):")
    print(by_ds.round(2).to_string())
    top_share = by_ds["share_%"].iloc[0]
    print(f"-> top event supplier holds {top_share:.1f}% of the reserve; "
          f"C01's per-regime cap applies to TOTALS incl. forced windows, "
          f"so a breach caused by forced events alone is disclosed, "
          f"never resolved by dropping events (D22/D23).")

    by_r = ev.groupby("regime").size().rename("events").to_frame()
    by_r["share_%"] = 100 * by_r["events"] / len(ev)
    q = None  # per-regime debit preview at the budget quota
    print("\nreserve by REGIME (these counts are debited from the quotas):")
    print(by_r.round(2).to_string())

    # ---- B: evasive conjunction supply ---------------------------------
    print(f"\n=== B. Evasive conjunction (D24 rule, clean channels, "
          f"all 14 datasets) ===")
    rows = []
    for g in EVASIVE_SWEEP_G:
        m = evasive_mask(pool, g)
        both = m & pool.index.isin(ev.index)
        rows.append({"lat_g": g, "n_evasive": int(m.sum()),
                     "%_of_frame": 100 * m.sum() / n_frame,
                     "overlap_with_reserve": int(both.sum())})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    m0 = evasive_mask(pool, EVASIVE_LAT_G)
    comp = (pool[m0].groupby("dataset").size().sort_values(ascending=False)
            .rename(f"n at {EVASIVE_LAT_G} g").to_frame())
    comp.index = [name(d) for d in comp.index]
    print(f"\ncomposition at the primary candidate ({EVASIVE_LAT_G} g):")
    print(comp.to_string())

    naive = ((pool["max_abs_yaw_degs"] >= rs.HIGH_YAW_DEG_S)
             & (pool["mean_speed_kmh"] >= rs.CRUISE_SPEED_KMH))
    print(f"\nnaive M08 intersection (>= {rs.HIGH_YAW_DEG_S:.0f} deg/s AND "
          f">= {rs.CRUISE_SPEED_KMH:.0f} km/h): n = {int(naive.sum())}  "
          f"(physics predicts ~0: implied a_lat at 100 km/h and 30 deg/s "
          f"is ~14.5 m/s^2, beyond the friction limit)")

    print("\n[audit] paste this output into the chat; D24's constant is "
          "fixed from table B, and D22's realised n_event is frozen from "
          "section A before curation_build.py runs.")


if __name__ == "__main__":
    main()