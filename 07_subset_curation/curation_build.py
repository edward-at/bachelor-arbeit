#!/usr/bin/env python3
"""curation_build.py — construct the curation subsets (thesis 4.11).

Builds the 2x2 slate's three new cells by IMPORTING the frozen
sampling machinery (thesis 4.9) from run_sampling.py — loaders, dedup, largest-
remainder allocator, SRS engine, writer, seed keying — and composing it.
run_sampling.py itself is never edited; its module globals are patched at
run time (METHOD_FILES for the C names, OVERRIDE_N / ACTIVE_OUT_DIR for
the tier), which is the same one-constant tier mechanism its own main uses.

  C01  equal regime quotas -> guarded 0.6 g reserve force-included and
       debited -> per-regime top-source cap at 0.50, cap-not-
       equalise, proportional draw for everyone else -> seeded SRS
       within each (regime, source) cell.
  C02  C01 without the reserve (source-lever ablation).
  C03  C01 without the cap   (event-lever ablation; regime-level SRS).

Construction-determined by definition (class O in the tautology map): C01/C02/C03
regime shares; C01/C03 event retention; C01/C02 top-source cap. The
harness measures everything else; nothing here reads a gate metric.

Usage:
  python curation_build.py --expect-events <n_from_audit>       # both tiers
  python curation_build.py --tiers budget --expect-events <n>
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import run_sampling as rs
from curation_common import (C_FILES, C_NAMES, C_SEED_INDEX, SOURCE_CAP_FRAC,
                             BUDGET_N, N_EVENT_RESERVE)

rs.METHOD_FILES.update(C_FILES)          # writer now knows the C names


# ---------------------------------------------------------------------------
# allocator: cap, do not equalise
# ---------------------------------------------------------------------------
def cap_allocate(residual_supply: pd.Series, forced: pd.Series, quota: int,
                 cap_frac: float, rng) -> tuple[pd.Series, dict]:
    """Per-source DRAW counts for one regime under the top-source cap.

    Draw is allocated proportionally to residual supply (cap-not-equalise);
    any source whose TOTAL (forced + draw) exceeds cap_frac * quota is
    shaved down to the cap and the excess re-spread proportionally over
    sources with headroom. Forced windows are never shaved: a breach
    caused by forced counts alone is disclosed, not resolved.
    """
    ds = residual_supply.index
    forced = forced.reindex(ds).fillna(0).astype(int)
    draw_total = int(quota) - int(forced.sum())
    if draw_total < 0:
        raise RuntimeError("reserve exceeds regime quota — impossible on "
                           "this corpus; check inputs")
    draw = pd.Series(
        rs.allocate(residual_supply.to_numpy(dtype=float), draw_total,
                    residual_supply.to_numpy()), index=ds, dtype=int)
    cap_units = int(np.floor(cap_frac * quota))
    breached = forced_breach = False
    for _ in range(30):
        totals = draw + forced
        over = totals[totals > cap_units]
        shave = pd.Series(0, index=ds, dtype=int)
        for d in over.index:
            reducible = max(0, int(draw[d]) - max(0, cap_units - int(forced[d])))
            shave[d] = min(reducible, int(totals[d]) - cap_units)
            if forced[d] > cap_units:
                forced_breach = True
        excess = int(shave.sum())
        if excess == 0:
            break
        draw = draw - shave
        headroom = np.minimum(residual_supply - draw,
                              cap_units - (draw + forced)).clip(lower=0)
        room = int(headroom.sum())
        put = min(excess, room)
        if put > 0:
            add = pd.Series(
                rs.allocate(residual_supply.where(headroom > 0, 0.0)
                            .to_numpy(dtype=float), put, headroom.to_numpy()),
                index=ds, dtype=int)
            draw = draw + add
        rest = excess - put
        if rest > 0:                      # cap infeasible: give back, disclose
            breached = True
            give = pd.Series(
                rs.allocate(residual_supply.where(shave > 0, 0.0)
                            .to_numpy(dtype=float), rest,
                            (residual_supply - draw).to_numpy()),
                index=ds, dtype=int)
            draw = draw + give
            break
    totals = draw + forced
    if int(totals.sum()) != int(quota):
        raise RuntimeError(f"cap_allocate lost windows: {totals.sum()} != "
                           f"{quota}")
    top = totals.idxmax()
    info = {"cap_units": cap_units, "top_source": str(top),
            "top_total": int(totals[top]),
            "top_share": round(float(totals[top]) / quota, 4),
            "cap_breached": bool(breached or forced_breach),
            "forced_breach": bool(forced_breach)}
    return draw, info


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def reserve(pool: pd.DataFrame) -> pd.DataFrame:
    guarded = pool[~pool["dataset"].isin(rs.CONFOUNDED_ACCEL)]
    return guarded[guarded["max_decel_ms2"] >= rs.DECEL_TRIGGER_G * rs.G_MS2]


def equal_quotas(supply: pd.DataFrame, N: int) -> dict:
    q = rs.allocate(np.ones(len(supply)), N, supply["TOTAL"].to_numpy())
    return dict(zip(supply.index.tolist(), q.tolist()))


def build_capped(pool, N, rng, supply, ev: pd.DataFrame | None):
    """Shared core of C01 (ev != None) and C02 (ev == None)."""
    quotas = equal_quotas(supply, N)
    parts, per_regime = [], {}
    for r, q in quotas.items():
        sub = pool[pool["regime"] == r]
        ev_r = (ev[ev["regime"] == r] if ev is not None
                else sub.iloc[0:0])
        forced = ev_r.groupby("dataset").size()
        residual = (sub["dataset"].value_counts()
                    .sub(forced, fill_value=0).astype(int))
        draw, info = cap_allocate(residual, forced, int(q),
                                  SOURCE_CAP_FRAC, rng)
        fill = rs.srs_within(sub.drop(index=ev_r.index), "dataset",
                             draw.to_dict(), rng)
        ev_r = ev_r.copy(); ev_r["role"] = "event"
        fill = fill.copy(); fill["role"] = "fill"
        parts.append(pd.concat([ev_r, fill]))
        info["quota"] = int(q); info["n_forced"] = int(forced.sum())
        per_regime[str(r)] = info
    return pd.concat(parts), {"quotas": {str(k): int(v)
                                         for k, v in quotas.items()},
                              "cap_frac": SOURCE_CAP_FRAC,
                              "per_regime": per_regime}


def build_c03(pool, N, rng, supply, ev: pd.DataFrame):
    quotas = equal_quotas(supply, N)
    parts, dbg = [], {}
    for r, q in quotas.items():
        sub = pool[pool["regime"] == r]
        ev_r = ev[ev["regime"] == r]
        rest = sub.drop(index=ev_r.index)
        n_fill = int(q) - len(ev_r)
        idx = rng.choice(rest.index.to_numpy(), size=min(n_fill, len(rest)),
                         replace=False)
        ev_r = ev_r.copy(); ev_r["role"] = "event"
        fill = rest.loc[idx].copy(); fill["role"] = "fill"
        parts.append(pd.concat([ev_r, fill]))
        dbg[str(r)] = {"quota": int(q), "n_forced": int(len(ev_r))}
    return pd.concat(parts), {"quotas": {str(k): int(v)
                                         for k, v in quotas.items()},
                              "per_regime": dbg}


# ---------------------------------------------------------------------------
def run_tier(tier: str, pool, supply, ev, expect_events, manifest_all):
    if tier == "budget":
        rs.OVERRIDE_N = BUDGET_N
        rs.ACTIVE_OUT_DIR = rs.SUBSET_DIR / f"tier_n{BUDGET_N}"
    else:
        rs.OVERRIDE_N = None
        rs.ACTIVE_OUT_DIR = rs.SUBSET_DIR
    rs.ACTIVE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    N = rs.matched_n(supply)
    rs.log(f"[{tier}] N = {N:,} -> {rs.ACTIVE_OUT_DIR}")

    runners = {
        "C01": lambda rng: build_capped(pool, N, rng, supply, ev),
        "C02": lambda rng: build_capped(pool, N, rng, supply, None),
        "C03": lambda rng: build_c03(pool, N, rng, supply, ev),
    }
    tier_entry = {"tier": tier, "N": int(N), "n_event_reserve": int(len(ev)),
                  "expect_events": expect_events, "methods": {}}
    for cid, fn in runners.items():
        t0 = time.time()
        sel, extra = fn(rs.rng_for(C_SEED_INDEX[cid]))
        if len(sel) != N:
            raise RuntimeError(f"{cid} [{tier}]: n = {len(sel):,} != N — "
                               f"exact-N contract broken")
        if cid in ("C01", "C03"):
            kept = int((sel["role"] == "event").sum())
            if kept != len(ev):
                raise RuntimeError(f"{cid}: reserve not fully retained "
                                   f"({kept} of {len(ev)})")
        rs.write_subset(sel, cid, C_NAMES[cid])
        tier_entry["methods"][cid] = {
            "name": C_NAMES[cid], "seed": [rs.GLOBAL_SEED, C_SEED_INDEX[cid]],
            "runtime_s": round(time.time() - t0, 2),
            **rs.summarize(sel), "params": extra}
    with open(rs.ACTIVE_OUT_DIR / "curation_manifest.json", "w") as f:
        json.dump(tier_entry, f, indent=2, default=rs.json_default)
    rs.log(f"[{tier}] curation_manifest.json written")
    manifest_all[tier] = tier_entry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="both",
                    choices=["both", "budget", "capability"])
    ap.add_argument("--expect-events", type=int, default=N_EVENT_RESERVE,
                    help="realised reserve size (frozen 2026-08-20 audit); "
                         "asserted on every run")
    args = ap.parse_args()

    pool_full, _, have_summaries = rs.load_pool()
    if not have_summaries:
        raise SystemExit("[build] window_summaries.parquet missing")
    pool = rs.apply_dedup(pool_full)
    rs.check_summary_coverage(pool)
    supply = rs.compute_supply(pool)
    ev = reserve(pool)
    rs.log(f"reserve: {len(ev):,} guarded {rs.DECEL_TRIGGER_G} g windows "
           f"(D22; provisional threshold, Section 11.8 disclosure carries)")
    if args.expect_events is not None and len(ev) != args.expect_events:
        raise RuntimeError(f"reserve {len(ev):,} != audited "
                           f"{args.expect_events:,} — inputs changed; re-run "
                           f"curation_event_audit.py first")

    manifest_all = {}
    tiers = (["budget", "capability"] if args.tiers == "both"
             else [args.tiers])
    for t in tiers:
        run_tier(t, pool, supply, ev, args.expect_events, manifest_all)
    rs.log("DONE — C-artifacts written; next: curation_eval.py")


if __name__ == "__main__":
    main()