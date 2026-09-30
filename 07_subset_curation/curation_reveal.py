#!/usr/bin/env python3
"""curation_reveal.py — the sealed shelf and the ex-confounded slice
(step 2 of replication and reveal). RUN ONLY AFTER curation_replicate.py.

Two things have been computed since the evaluation stage (thesis 4.10) and
never read:

  1. SEALED SHELF (defined in thesis 4.10): the Wasserstein
     distances of the maximum-speed and peak-acceleration-magnitude
     summaries. Mechanically chosen: they are the only kinematic
     summaries consumed by NO method's selection criterion, which is what
     makes reading them now evidence rather than theatre.
  2. EX-CONFOUNDED SLICE: every displayed metric recomputed on the
     guarded frame, the prepared answer to "does the result survive
     dropping A2D2*, 4Seasons* and DurLAR*?".

PRE-REGISTERED EXPECTATION, recorded before the reveal:
C01 optimises no extreme — its levers are equal quotas, a source cap and
an event census — so no mechanism predicts it should win or lose the
shelf. The prior is NEUTRAL. A large deviation in either direction is
therefore informative, and a null result is a genuine null, not a
disappointment.

The script refuses to run until the replication artifact exists, which is
the ordering that keeps the reveal honest.

Usage:  python curation_reveal.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import subset_eval_harness as h
from curation_common import (WINNER, GATES, TIE_BREAK, METHODS, C_METHODS,
                             load_long)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
SHELF = ["w1_max_speed", "w1_peak_acc_mag"]
TIER = "budget"


def guard() -> None:
    p = h.EVAL_DIR / "curation_replication_budget.parquet"
    if not p.exists():
        raise SystemExit(
            "[reveal] replication artifact not found. Section 13.3 runs "
            "replication FIRST, then the reveal — that ordering is what "
            "makes the sealed evidence count. Run curation_replicate.py.")


def main() -> None:
    guard()
    print("=" * 74)
    print("SEALED SHELF REVEAL — Section 13.3. Sealed 2026-07-08, opened "
          "2026-08-20,\nafter the winner was declared and replicated.")
    print("D29 pre-registered expectation: NEUTRAL. C01 targets no extreme, "
          "so\nneither a win nor a loss on the shelf is predicted by its "
          "construction.")
    print("=" * 74)

    d = load_long()
    d = d[d["tier"] == TIER]
    order = [m for m in ["R0_frame", "R1_guarded"] + METHODS + C_METHODS]

    # ---- 1. the shelf --------------------------------------------------
    sh = d[(d["variant"] == "main") & (d["metric"].isin(SHELF))]
    piv = sh.pivot(index="method", columns="metric", values="value")
    piv = piv.reindex([m for m in order if m in piv.index])
    print("\n[1] Sealed shelf: 1-Wasserstein distance to the frame, "
          "physical units\n    (max speed in km/h, peak |a| in m/s^2; the "
          "second carries the asterisk)")
    print(piv.round(4).to_string())
    if WINNER in piv.index:
        m_only = piv.loc[[m for m in METHODS if m in piv.index]]
        print(f"\n    {WINNER} rank among the 11 sampling methods "
              f"(1 = closest to the frame):")
        for met in SHELF:
            if met in piv.columns:
                better = int((m_only[met] < piv.at[WINNER, met]).sum())
                print(f"      {met:<18} {piv.at[WINNER, met]:.4f}  "
                      f"-> rank {better + 1} of {len(m_only) + 1}")

    # ---- 2. ex-confounded slice ----------------------------------------
    print("\n[2] Ex-confounded slice: every gate metric and the tie-break, "
          "recomputed\n    on the guarded frame (A2D2*, 4Seasons*, DurLAR* "
          "removed)")
    cols = list(GATES)
    out = {}
    for variant, label in [("main", "all 14"), ("ex_confounded", "ex-conf")]:
        v = d[d["variant"] == variant]
        w = v.pivot(index="method", columns="metric", values="value")
        s = v[v["metric"].str.match(r"shift_r\d_speed$")]
        if len(s):
            w[TIE_BREAK] = (s.pivot(index="method", columns="metric",
                                    values="value").abs().max(axis=1))
        for c in cols + [TIE_BREAK]:
            if c in w.columns:
                out[(c, label)] = w[c]
    tab = pd.DataFrame(out)
    tab = tab.reindex([m for m in order if m in tab.index])
    print(tab.round(4).to_string())

    if WINNER in tab.index:
        print(f"\n    {WINNER} gate check on the ex-confounded slice:")
        ok = True
        for met, (op, thr, _) in GATES.items():
            if (met, "ex-conf") not in tab.columns:
                continue
            val = tab.at[WINNER, (met, "ex-conf")]
            p = (val <= thr) if op == "<=" else (val >= thr)
            ok &= bool(p) and np.isfinite(val)
            print(f"      {met:<18} {val:.4f} {op} {thr}  "
                  f"{'PASS' if p else 'FAIL'}")
        tb = tab.at[WINNER, (TIE_BREAK, "ex-conf")] \
            if (TIE_BREAK, "ex-conf") in tab.columns else np.nan
        print(f"      {TIE_BREAK:<18} {tb:.4f} (limit 0.500)")
        word = "SURVIVES" if ok else "DOES NOT SURVIVE"
        print(f"    -> the winner's verdict {word} removal of the three")
        print("       fidelity-confounded sources.")

    print("\n[reveal] Shelf and slice are now read. Nothing further is "
          "sealed.\nNext: 13.3 prose — replication spread, this reveal, the "
          "standalone\ncharacterization, then the verdict.")


if __name__ == "__main__":
    main()