#!/usr/bin/env python3
"""curation_replicate.py — seed replication of the winner (step 1 of replication and reveal).

Resolves the limitation stated in the evaluation (thesis 4.10): every method
was realised by a single seeded draw, so the tie-break value (C01's
max within-regime shift) had no error bar. This script rebuilds
the winner R times under a declared replicate seed family and evaluates
each draw through the HARNESS'S OWN compute_row — the identical
instrument, not a reimplementation — so replicated values are directly
comparable to the recorded single-draw numbers.

Declared before any curation subset was evaluated:
  replicate seed family = (GLOBAL_SEED, REPLICATE_SEED_BASE + r),
  REPLICATE_SEED_BASE = 1200, r = 0..R-1, R = 20 by default.
  The original draw (42, 12) is NOT one of the replicates; it is plotted
  and reported alongside them as the realised subset.
  Reported: mean, standard deviation and the central-95 per cent interval
  of every gate metric and of the tie-break, plus the share of replicates
  that pass all four gates.

The reserve is a census (inclusion probability 1), so it is identical in
every replicate by construction. Replication therefore varies only the
fill, which is the disclosure attached to the event reserve: the reported
spread understates total variance, and this script prints that statement
with the numbers.

Nothing sealed is read. No artifact is overwritten: replicate subsets are
built in memory and discarded, and the summary is written beside the eval
cache as its own file.

Usage:
  python curation_replicate.py                 # R = 20, budget tier
  python curation_replicate.py --r 40
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import run_sampling as rs
import subset_eval_harness as h
import curation_build as cb
from curation_common import (BUDGET_N, GATES, TIE_BREAK, N_EVENT_RESERVE,
                             WINNER, C_SEED_INDEX)

REPLICATE_SEED_BASE = 1200          # declared replicate seed family
DEFAULT_R = 20


def harness_context():
    """Reproduce the harness's own setup so compute_row is byte-identical.

    Reuses curation_eval.patch() first, because build_master() needs the
    same canary re-anchoring the eval driver applies. Same patch, same
    values, same loud old->new log.
    """
    import curation_eval as ce
    ce.patch()
    master, meta = h.build_master()
    h.calibrate_units(master, meta)
    manifests = {t: h.read_manifest(d) for t, d in h.TIER_DIRS.items()}
    h.add_score_and_predicates(master, meta, manifests)
    e1 = np.linspace(master["pc1"].min(), master["pc1"].max(),
                     h.PC_GRID_BINS + 1)
    e2 = np.linspace(master["pc2"].min(), master["pc2"].max(),
                     h.PC_GRID_BINS + 1)
    ctx = h.BaseCtx(master, (e1, e2))
    strategy = meta["join_strategy"]
    key_pos = pd.Series(np.arange(len(master)), index=master["key"].values)
    return master, meta, ctx, key_pos, strategy


def subset_rows(sel: pd.DataFrame, master, key_pos, strategy):
    """Map a built subset onto master rows using the harness join key."""
    m = {"dataset": "dataset", "clip": "clip_id",
         "win_start": "window_start_s", "win_idx": "window_idx"}
    key = h.make_key(sel, m, strategy)
    pos = key_pos.reindex(key.values).dropna().astype(int).values
    if len(pos) != len(sel):
        raise RuntimeError(f"replicate join lost {len(sel) - len(pos)} rows")
    return master.iloc[pos]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=DEFAULT_R)
    args = ap.parse_args()

    if WINNER != "C01":
        raise SystemExit(f"[replicate] WINNER is {WINNER!r}; this script "
                         f"replicates the C01 composite. Update it "
                         f"deliberately, not by accident.")

    print(f"[replicate] D28: winner {WINNER}, R = {args.r}, seed family "
          f"(42, {REPLICATE_SEED_BASE}+r); original draw "
          f"(42, {C_SEED_INDEX['C01']}) reported alongside, not counted.")

    # ---- selection-stage inputs (same as curation_build) ----------------
    pool_full, _, have = rs.load_pool()
    if not have:
        raise SystemExit("[replicate] window_summaries.parquet missing")
    pool = rs.apply_dedup(pool_full)
    rs.check_summary_coverage(pool)
    supply = rs.compute_supply(pool)
    ev = cb.reserve(pool)
    if len(ev) != N_EVENT_RESERVE:
        raise RuntimeError(f"reserve {len(ev)} != frozen {N_EVENT_RESERVE}")

    master, meta, ctx, key_pos, strategy = harness_context()
    with_gmm = "gmm_entropy" in master.columns

    rows = []
    for r in range(args.r):
        rng = np.random.default_rng([rs.GLOBAL_SEED, REPLICATE_SEED_BASE + r])
        sel, _ = cb.build_capped(pool, BUDGET_N, rng, supply, ev)
        if len(sel) != BUDGET_N:
            raise RuntimeError(f"replicate {r}: n = {len(sel)}")
        sub = subset_rows(sel, master, key_pos, strategy)
        row = h.compute_row(sub, ctx, with_gmm)
        sh = [abs(row[k]) for k in row if k.startswith("shift_r")
              and k.endswith("_speed")]
        row[TIE_BREAK] = max(sh) if sh else np.nan
        row["replicate"] = r
        rows.append(row)
        print(f"  r={r:>3}  shift={row[TIE_BREAK]:.4f}  "
              f"top_src={row.get('top_source_share', np.nan):.4f}  "
              f"ret_hd={row.get('ret_hard_decel', np.nan):.3f}")

    rep = pd.DataFrame(rows)
    cols = list(GATES) + [TIE_BREAK]
    summ = rep[cols].agg(["mean", "std", "min", "max"]).T
    summ["p2_5"] = rep[cols].quantile(0.025)
    summ["p97_5"] = rep[cols].quantile(0.975)

    print(f"\n=== C01 across R = {args.r} replicates (budget tier) ===")
    print(summ.round(4).to_string())

    passes = np.ones(len(rep), dtype=bool)
    for met, (op, thr, _) in GATES.items():
        passes &= (rep[met] <= thr) if op == "<=" else (rep[met] >= thr)
    print(f"\nreplicates passing ALL FOUR gates: {int(passes.sum())} / "
          f"{args.r}")
    print(f"tie-break spread: {rep[TIE_BREAK].min():.4f} to "
          f"{rep[TIE_BREAK].max():.4f} "
          f"(limit {GATES['top_source_share'][1]:.3f}; "
          f"single realised draw: gate table, Section 13.2)")
    print(f"\nDISCLOSURE (D22, print this in the prose): the {N_EVENT_RESERVE:,}-window "
          "reserve is a census with inclusion probability 1, identical in "
          "every replicate, so the spread above varies only the fill and "
          "UNDERSTATES total selection variance.")

    out = h.EVAL_DIR / "curation_replication_budget.parquet"
    rep.to_parquet(out, index=False)
    with open(h.EVAL_DIR / "curation_replication_meta.json", "w") as f:
        json.dump({"winner": WINNER, "R": args.r,
                   "seed_family": [rs.GLOBAL_SEED, REPLICATE_SEED_BASE],
                   "reserve_n": int(len(ev)),
                   "gates_passed_all": int(passes.sum()),
                   "summary": json.loads(summ.to_json())}, f, indent=2)
    print(f"\n[replicate] wrote {out.name} + curation_replication_meta.json")
    print("[replicate] NEXT and only now: curation_reveal.py")


if __name__ == "__main__":
    main()