#!/usr/bin/env python3
"""curation_evasive_check.py — the D24 held-out metric, finally reported.

Register entry D24 defines an evasive-maneuver retention under the
implied-lateral rule (peak |yaw| >= 0.75 g of lateral acceleration at the
window's mean speed) and promises it is reported for every method, the
winner included, while being optimized by nothing. The supply was counted
in the audit (expected 2,279 windows, 0.57 percent of the frame), but the
retention itself was never computed anywhere, because the Chapter 12
harness's metric set is frozen. This script closes that gap.

For every budget-tier artifact: retention = (evasive share in the subset)
divided by (evasive share in the frame), the same ratio grammar as the
hard-deceleration metric, so 1.0 is the random-selection null. Also
reported for C01: the overlap between its retained evasive windows and
its forced hard-deceleration reserve, so lateral and longitudinal
responses can be discussed as the distinct populations the audit showed
them to be.

Read-only. No selection consequence is possible: the winner was declared,
replicated, and revealed before this number existed, which is precisely
what makes it evidence.

Usage:  python curation_evasive_check.py
"""

from __future__ import annotations

import pandas as pd

import run_sampling as rs
import curation_common as cc

from curation_common import N_EVENT_RESERVE, EXPECTED_EVASIVE
# single definition: curation_common derives both via canonical; deriving them
# again here was a third copy of numbers that must agree


def main() -> None:
    pool_full, _, have = rs.load_pool()
    if not have:
        raise SystemExit("[evasive] window_summaries.parquet missing")
    pool = rs.apply_dedup(pool_full).reset_index(drop=True)
    rs.check_summary_coverage(pool)

    mask = cc.evasive_mask(pool, cc.EVASIVE_LAT_G)
    n_ev = int(mask.sum())
    frame_rate = n_ev / len(pool)
    print(f"[evasive] rule: peak |yaw| >= {cc.EVASIVE_LAT_G} g / v "
          f"(clean channels, physical low-speed floor)")
    print(f"[evasive] frame supply: {n_ev:,} of {len(pool):,} "
          f"({100 * frame_rate:.3f}%)  [audit canary: "
          f"{EXPECTED_EVASIVE:,}]")
    if n_ev != EXPECTED_EVASIVE:
        print("[evasive] WARNING: supply differs from the frozen audit "
              "count. Inputs changed. Stop and diagnose before quoting.")

    thr = rs.DECEL_TRIGGER_G * rs.G_MS2
    hard = (pool["max_decel_ms2"] >= thr) \
        & (~pool["dataset"].isin(rs.CONFOUNDED_ACCEL))

    keys = [c for c in ["dataset", "clip_id", "window_start_s",
                        "window_idx"] if c in pool.columns]
    pool_keyed = pool[keys].copy()
    pool_keyed["_evasive"] = mask
    pool_keyed["_hard"] = hard.to_numpy()

    tier_dir = rs.SUBSET_DIR / f"tier_n{cc.BUDGET_N}"
    winner_note = None
    rows = []
    for p in sorted(list(tier_dir.glob("M[0-9][0-9]_*.parquet"))
                    + list(tier_dir.glob("C[0-9][0-9]_*.parquet"))):
        mid = p.name[:3]
        sel = pd.read_parquet(p)
        use = [k for k in keys if k in sel.columns]
        j = sel[use].merge(pool_keyed, on=use, how="left")
        if j["_evasive"].isna().any():
            print(f"[evasive] {mid}: {int(j['_evasive'].isna().sum())} "
                  f"windows failed the key join, excluded")
            j = j.dropna(subset=["_evasive"])
        rate = float(j["_evasive"].mean())
        rows.append({"method": mid, "n_evasive": int(j["_evasive"].sum()),
                     "ret_evasive": rate / frame_rate})
        if mid == cc.WINNER:
            both = int((j["_evasive"] & j["_hard"].fillna(False)).sum())
            winner_note = (f"[evasive] {mid}: {both} of its retained "
                           f"evasive windows are also reserve windows, "
                           f"the rest are lateral-only responses")
    tab = pd.DataFrame(rows).set_index("method").sort_index()
    print("\n=== D24 held-out evasive retention (budget tier, "
          "1.0 = random-selection null) ===")
    print(tab.round(3).to_string())
    print()
    if winner_note:
        print(winner_note)
    else:
        print(f"[evasive] note: winner {cc.WINNER} not found in "
              f"{tier_dir.name}; overlap line skipped")
    print("\n[evasive] paste this table into the chat. The winner's value "
          "is genuine\nheld-out evidence either way: retention near or "
          "above 1.0 means the\ncomposite protects lateral responses it "
          "never targeted, retention below\n1.0 is an honest limitation "
          "and ready-made future work.")


if __name__ == "__main__":
    main()