#!/usr/bin/env python3
"""curation_render_winner.py — figures for the declared winner ONLY (13.2/13.3).

Refuses to run until curation_common.WINNER is set by the dated register
entry that curation_gates.py prints (D25: numbers first, one method plotted).
Chapter 12 figure code is driven by patching, never edited; everything
lands in figures/curation/ so no Chapter 12 figure is overwritten.

Default run:
  * finalist ECDF overlays via the frozen subset_ecdf_finalist module,
    FINALISTS = curation_common.FINALISTS + WINNER (highlighted in the
    curation colour)
  * native winner regime-profile bars (vs frame, uniform line)
  * native winner source-composition bars (vs frame, cap line, asterisks)

--frame-pass   re-runs subset_frame_pass with the mask widened by the winner
               (the one extra frame-layer read Section 12.1 budgeted for)
--dossier      then drives the Ch.8-grain dossier modules (histogram grids,
               phase portraits, fans, exemplars) into figures/curation/

Usage:  python curation_render_winner.py [--frame-pass] [--dossier]
"""

from __future__ import annotations

import argparse
import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import curation_common as cc
import subset_eval_common as sec

DOSSIER_MODULES = ["subset_histogram", "subset_phase_potrait",
                   "subset_fans", "subset_time_series"]


def require_winner() -> str:
    if not cc.WINNER:
        raise SystemExit("[render] curation_common.WINNER is unset — run "
                         "curation_gates.py, record the dated entry, set "
                         "WINNER (and FINALISTS), then re-run.")
    return cc.WINNER


def patch_common(winner: str) -> None:
    sec.OUT_DIR = cc.ensure_curation_outdir()          # figures -> curation/
    if winner not in sec.METHODS:
        sec.METHODS.append(winner)
    sec.METHOD_SHORT[winner] = cc.C_SHORT.get(winner, winner)
    sec.METHOD_COLORS[winner] = cc.CURATION_COLOR
    sec.ROW_LABELS[winner] = f"{winner} {sec.METHOD_SHORT[winner]}"
    print(f"[render] patched: OUT_DIR -> {sec.OUT_DIR}, slate += {winner}")


def finalist_overlays(winner: str) -> None:
    mod = importlib.import_module("subset_ecdf_finalist")
    mod.FINALISTS = [m for m in cc.FINALISTS if m != winner] + [winner]
    for attr in ("METHOD_COLORS", "COLORS", "LINE_COLORS", "PALETTE"):
        try:
            getattr(mod, attr)[winner] = cc.CURATION_COLOR
        except (AttributeError, TypeError, KeyError):
            pass
    print(f"[render] finalists: {mod.FINALISTS}")
    mod.main()


def winner_profile(winner: str) -> None:
    m = cc.load_master("budget")
    shares = [c for c in m.columns if c.startswith("share_r")]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(shares))
    ax.bar(x - 0.2, 100 * m.loc["R0_frame", shares].astype(float), 0.4,
           label="Frame", color=cc.GHOST,
           edgecolor="black", linewidth=0.5)
    ax.bar(x + 0.2, 100 * m.loc[winner, shares].astype(float), 0.4,
           label=winner, color=cc.CURATION_COLOR,
           edgecolor="black", linewidth=0.5)
    ax.axhline(20, color="black", linestyle="--", linewidth=1,
               label="uniform target")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}" for i in range(len(shares))])
    ax.set_xlabel("regime")
    ax.set_ylabel("share of windows (%)")
    ax.set_title(f"{winner}: regime composition")
    ax.set_axisbelow(True); ax.grid(True, axis="y", alpha=0.4); ax.legend()
    fig.tight_layout()
    out = sec.OUT_DIR / "curation_winner_regime_profile.png"
    fig.savefig(out, dpi=cc.DPI); print(f"[render] wrote {out}")


def winner_sources(winner: str) -> None:
    d = cc.load_long()
    d = d[(d["tier"] == "budget") & (d["variant"] == "main")
          & (~d["sealed"]) & d["metric"].str.startswith("src_share_")]
    piv = d.pivot(index="method", columns="metric", values="value")
    piv.columns = [c.replace("src_share_", "") for c in piv.columns]
    order = piv.loc["R0_frame"].sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(order))
    ax.bar(x - 0.2, 100 * piv.loc["R0_frame", order].astype(float), 0.4,
           label="Frame", color=cc.GHOST, edgecolor="black",
           linewidth=0.5)
    ax.bar(x + 0.2, 100 * piv.loc[winner, order].astype(float), 0.4,
           label=winner, color=cc.CURATION_COLOR, edgecolor="black",
           linewidth=0.5)
    ax.axhline(100 * cc.SOURCE_CAP_FRAC / 5, linewidth=0)  # no-op spacing
    labels = [cc.DISPLAY_NAMES.get(ds, ds)
              + ("*" if ds in cc.CONFOUNDED else "") for ds in order]
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("share of windows (%)")
    ax.set_title(f"{winner}: source composition vs frame (per-regime "
                 f"cap {cc.SOURCE_CAP_FRAC:.0%} applies within regimes)")
    ax.set_axisbelow(True); ax.grid(True, axis="y", alpha=0.4); ax.legend()
    fig.tight_layout()
    out = sec.OUT_DIR / "curation_winner_source_composition.png"
    fig.savefig(out, dpi=cc.DPI); print(f"[render] wrote {out}")


def run_frame_pass(winner: str) -> None:
    import subset_frame_pass as fp
    fp.BUDGET_DIR = fp.SUBSET_DIR / f"tier_n{cc.BUDGET_N}"
    orig = fp.discover_methods

    def widened():
        methods = list(orig())
        hits = sorted(fp.BUDGET_DIR.glob(f"{winner}_*.parquet"))
        if not hits:
            raise SystemExit(f"[render] {winner} artifact missing in "
                             f"{fp.BUDGET_DIR}")
        if winner not in methods:
            methods.append(winner)
        return methods

    fp.discover_methods = widened
    print("[render] frame pass with widened mask — this streams every clip "
          "once; expect the Section 12.1 runtime.")
    fp.main()


def run_dossier() -> None:
    for name in DOSSIER_MODULES:
        try:
            importlib.import_module(name).main()
            print(f"[render] dossier module done: {name}")
        except Exception as e:                      # keep going, report
            print(f"[render] {name} FAILED: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-pass", action="store_true")
    ap.add_argument("--dossier", action="store_true")
    args = ap.parse_args()

    winner = require_winner()
    patch_common(winner)
    finalist_overlays(winner)
    winner_profile(winner)
    winner_sources(winner)
    if args.frame_pass:
        run_frame_pass(winner)
    if args.dossier:
        run_dossier()
    print("[render] done. Sealed shelf: reveal in 13.3 only "
          "(curation_show_eval.py --reveal-seal).")


if __name__ == "__main__":
    main()