#!/usr/bin/env python3
"""curation_gates.py — apply the selection rule mechanically and declare the winner.

Tables only, no figures (numbers-first rule): the rule decides on numbers,
plots never vote.

Sequence, printed in reading order:
  1. The selection rule text with thresholds and grounds (declared before
     any curation subset was evaluated).
  2. The 2x2 ablation table (M03 / C02 / C03 / C01) on every gate metric,
     the tie-break, and the diagnostics — each cell tagged with its
     tautology class so construction-determined values cannot be read as
     findings, and each metric shown with the K=100 SRS null band.
  3. Gate pass/fail for the full slate (M01..M11 + C01..C03).
  4. Tie handling: survivors within the tie tolerance on the tie-break
     metric are declared tied (both are replicated); otherwise the
     minimiser wins. Tolerance = half-width of the central-95% SRS null
     band for the tie-break metric, TIE_FALLBACK_TOL if the band is absent
     (disclosed).
  5. The winner declaration line, whose values are recorded in
     curation_common.py (WINNER / FINALISTS).

Nothing sealed is read: only main-variant unsealed rows are touched.

Usage:  python curation_gates.py            # budget tier (headline)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from curation_common import (GATES, TIE_BREAK, TIE_FALLBACK_TOL, DIAGNOSTICS,
                             ABLATION_ORDER, C_METHODS, METHODS, load_long,
                             load_cls, envelope_band)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
TIER = "budget"


def table(tier: str) -> pd.DataFrame:
    d = load_long()
    d = d[(d["tier"] == tier) & (d["variant"] == "main") & (~d["sealed"])]
    wide = d.pivot(index="method", columns="metric", values="value")
    sh = d[d["metric"].str.match(r"shift_r\d_speed$")]
    wide[TIE_BREAK] = (sh.pivot(index="method", columns="metric",
                                values="value").abs().max(axis=1))
    return wide


def main() -> None:
    print("=" * 74)
    print("D4 — dated 2026-08-20, declared before any curation artifact was "
          "evaluated.\nGates (all must pass), thresholds reference-derived:")
    for m, (op, thr, ground) in GATES.items():
        print(f"  {m:<18} {op} {thr:<6}  [{ground}]")
    print(f"Tie-break: min {TIE_BREAK} among survivors, measured on "
          f"channels no method optimises.")
    print("=" * 74)

    w = table(TIER)
    missing = [c for c in C_METHODS if c not in w.index]
    if missing:
        raise SystemExit(f"[gates] {missing} absent from the eval cache — "
                         f"run curation_eval.py first")

    cls = load_cls()
    gate_cols = list(GATES) + [TIE_BREAK] + [c for c in DIAGNOSTICS
                                             if c in w.columns]

    def fmt(mid, met):
        v = w.at[mid, met] if met in w.columns else np.nan
        mark = {"T": "#", "P": "~", "O": "#", "N": ""}.get(
            cls.get((met, mid), "N"), "")
        return f"{v:.3f}{mark}" if np.isfinite(v) else "-"

    print("\n[2] 2x2 ablation (rows: no-lever -> cap-only -> events-only -> "
          "composite). '#' = construction-determined, '~' = influenced — "
          "such cells are definitions, not findings:")
    ab = pd.DataFrame({met: {mid: fmt(mid, met) for mid in ABLATION_ORDER}
                       for met in gate_cols}).T
    print(ab.to_string())
    print("\nSRS null band (central 95%, K=100):")
    for met in list(GATES) + [TIE_BREAK]:
        band = envelope_band(met)
        print(f"  {met:<22} " + (f"[{band[0]:.3f}, {band[1]:.3f}]"
                                 if band else "band unavailable"))

    print("\n[3] gates over the full slate:")
    rows = []
    for mid in [m for m in METHODS + C_METHODS if m in w.index]:
        res, ok = {}, True
        for met, (op, thr, _) in GATES.items():
            v = w.at[mid, met] if met in w.columns else np.nan
            p = (v <= thr) if op == "<=" else (v >= thr)
            p = bool(p) and np.isfinite(v)
            res[met] = f"{v:.3f} {'PASS' if p else 'fail'}"
            ok &= p
        rows.append({"method": mid, **res,
                     TIE_BREAK: round(float(w.at[mid, TIE_BREAK]), 3),
                     "ALL GATES": "PASS" if ok else "-"})
    full = pd.DataFrame(rows).set_index("method")
    print(full.to_string())

    surv = full.index[full["ALL GATES"] == "PASS"].tolist()
    print(f"\n[4] survivors: {surv if surv else 'NONE'}")
    if not surv:
        print("No method passes all four gates. Under D4 this is the "
              "incompatibility branch: report the priced trade-off (headline "
              "(c)); no winner is declared and 13.3 replicates the nearest "
              "miss for the record.")
        return
    band = envelope_band(TIE_BREAK)
    tol = (band[1] - band[0]) / 2 if band else TIE_FALLBACK_TOL
    src = "SRS-null half-width" if band else f"fallback {TIE_FALLBACK_TOL}"
    tb = full.loc[surv, TIE_BREAK].astype(float).sort_values()
    best = tb.index[0]
    tied = tb.index[(tb - tb.iloc[0]) <= tol].tolist()
    print(f"tie tolerance = {tol:.4f} ({src}); tie-break values: "
          f"{tb.round(3).to_dict()}")
    if len(tied) > 1:
        print(f"[5] TIED within tolerance: {tied} — replicate ALL of them "
              f"in 13.3, then decide on the replicated means (pre-authorised "
              f"by D4).")
    else:
        print(f"[5] WINNER: {best}  (dated register entry; paste into "
              f"curation_common.py -> WINNER = \"{best}\")")
    print("    FINALISTS for the overlays: M01 + the best-performing "
          "sampling methods per the master table + the winner — fix the "
          "list by a dated entry, then set curation_common.FINALISTS.")


if __name__ == "__main__":
    main()