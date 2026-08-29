#!/usr/bin/env python3
"""curation_common.py — shared constants for every Chapter 13 script.

Chapter 13 rule: scripts are prefixed curation_, figures go to
figures/curation/, data artifacts keep C0x_*.parquet naming inside the
Chapter 11 subset directories so the harness and frame pass discover them.

Everything Chapter 12 already defines is IMPORTED from subset_eval_common,
never redefined (one-definition rule). This module adds only what is new
in Chapter 13: the C-slate, its seeds, the D4 gate constants, the evasive
held-out metric rule, and the curation figure directory.

D-register (declared 2026-08-20, before any C-artifact was evaluated):
  D4   selection rule, filled: four gates then fidelity tie-break, see
       GATES / TIE_BREAK below. Tie tolerance = central-95% half-width of
       the K=100 SRS null for the tie-break metric (fallback 0.01 km/h,
       disclosed if used).
  D20  fairness definition P1-P5; excluded properties written up in
       S2.7.9 (concepts), Ch.13 (operational table), S6.6 (two ceilings).
  D21  slate = C01 full composite, C02 no-events ablation, C03 no-cap
       ablation; M03 is the free fourth cell of the 2x2.
  D22  event reserve = ALL guarded 0.6 g windows (expected N_EVENT_RESERVE (1,088 on v4_4; superseded: 1,124, 1,115)),
       debited, role-tagged, class O; never dropped for any cap.
  D23  source lever = per-regime top-source cap at 0.50 of the regime
       quota, cap-not-equalise; feasibility relaxation disclosed, never
       silent; d2_sources reported as diagnostic, not gated.
  D24  evasive held-out metric = implied-lateral derivation rule
       peak|yaw| >= EVASIVE_LAT_G * g / mean_speed, constant fixed after
       curation_event_audit.py counts; optimised by nothing.
  D25  numbers-first selection: gates applied on tables + null band;
       figures for the winner only; duplication rule cross-ref /
       finalist overlay / named-reference overlay.
  D26  pipeline stated in two modes (multi-corpus with Stage 0, single-
       corpus entering at Stage 1); step-by-step section closes Ch.13.
"""

from __future__ import annotations

from pathlib import Path

from subset_eval_common import (  # noqa: F401  (re-exported on purpose)
    PROJECT_ROOT, EVAL_DIR, FIG_ROOT, DPI, BUDGET_N, METHODS, METHOD_SHORT,
    METHOD_COLORS, ROW_LABELS, DISPLAY_NAMES, DATASET_COLORS, CONFOUNDED,
    MAIN_BLUE, load_master, load_long, envelope_band, load_cls, load_dist,
)

# ---------------------------------------------------------------------------
# Chapter 13 slate — IDs, files, seeds, labels, colour
# ---------------------------------------------------------------------------
C_METHODS = ["C01", "C02", "C03"]
C_FILES = {
    "C01": "C01_event_capped_equal.parquet",
    "C02": "C02_capped_equal.parquet",       # ablation: no event reserve
    "C03": "C03_event_equal.parquet",        # ablation: no source cap
}
C_NAMES = {
    "C01": "event-anchored capped equal allocation (full composite)",
    "C02": "capped equal allocation (no event reserve)",
    "C03": "event-anchored equal allocation (no source cap)",
}
C_SHORT = {"C01": "Composite", "C02": "Cap only", "C03": "Events only"}
# seed indices continue the Chapter 11 keying (M01..M11 = 1..11)
C_SEED_INDEX = {"C01": 12, "C02": 13, "C03": 14}
from tum_colors import TUM_SECONDARY

CURATION_COLOR = MAIN_BLUE              # = TUM_SECONDARY["blue2"], #005293.
                                        # Edward 2026-08-23: the winner uses
                                        # the thesis single-series colour, so
                                        # C-figures match the Ch.12 convention
                                        # by definition, not by coincidence.
GHOST = TUM_SECONDARY["light_gray"]     # #CCCCCC, 20% black — the frame
                                        # ghost fill in every C-figure
FRAME_LINE = GHOST                      # Edward 2026-08-23: frame lines use
                                        # the SAME light_gray as the fills.
                                        # Visibility comes from linewidth
                                        # (thick pale underlay, thin blue2
                                        # series on top), not from a darker
                                        # hue. Flip to TUM_SECONDARY["gray"]
                                        # here if it proves too faint.
# Edward 2026-08-23: every colour in Chapter 13 figures comes from the TUM
# palette. Only black and white may be literal.

# the 2x2 ablation grid (rows in slate order; M03 is the free corner)
ABLATION_ORDER = ["M03", "C02", "C03", "C01"]

# ---------------------------------------------------------------------------
# D4 gate constants (reference-derived; see runbook for grounds)
# ---------------------------------------------------------------------------
GATES = {
    # metric               (direction, threshold, ground)
    "js_to_uniform":    ("<=", 0.005, "uniform regime target R2"),
    "top_source_share": ("<=", 0.500, "half the frame's 0.741"),
    "ret_hard_decel":   (">=", 1.000, "random-selection null"),
    "ret_near_stat":    (">=", 0.500, "half the random-selection null"),
}
TIE_BREAK = "max_abs_shift_speed"       # min over survivors wins
TIE_FALLBACK_TOL = 0.01                 # km/h, used only if null band absent
DIAGNOSTICS = ["worst_regime_top_source", "d2_sources", "low_dyn_share"]

# ---------------------------------------------------------------------------
# C01/C02 source lever
# ---------------------------------------------------------------------------
SOURCE_CAP_FRAC = 0.50                  # per-regime top-source cap (D23)

# ---------------------------------------------------------------------------
# D24 — evasive held-out metric (derivation rule, not a constant threshold)
# implied lateral acceleration: a_lat = v * yaw  =>  yaw_thr(v) = a_g*g / v
# EVASIVE_LAT_G is provisional until curation_event_audit.py counts are read
# (same provisional status as the 0.6 g longitudinal trigger, Section 11.8).
# ---------------------------------------------------------------------------
G_MS2 = 9.80665

EVASIVE_LAT_G = 0.75                    # FIXED 2026-08-20, re-audited 2026-08-29:
                                        # n=2,294 (0.57% of frame) at 0.75 g;
                                        # 0.50 g admits 3.7% (brisk cornering,
                                        # not evasion). Provisional against the
                                        # NDS lateral-trigger PDFs, same D5
                                        # duty as the 0.6 g constant.
EVASIVE_SWEEP_G = [0.40, 0.50, 0.60, 0.75]
YAW_PHYS_BOUND_DEGS = 90.0              # Section 7 validity bound; self-floors
                                        # the rule at low speed (no speed floor)

# Derived audit constants. Placed BELOW the D24 constants on purpose:
# canonical._evasive_consts() imports EVASIVE_LAT_G / YAW_PHYS_BOUND_DEGS /
# G_MS2 from THIS module, so importing C above them was a circular import that
# failed inside canonical's try/except and fell back to literals silently.
# The order of these lines is load-bearing.
from canonical import C
N_EVENT_RESERVE = C.event_reserve_n        # 1,088 on the v4_4 frame (was 1,115)
EXPECTED_EVASIVE = C.evasive_n             # 2,294 at 0.75 g (was 2,279)

# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
CURATION_FIG_DIR = FIG_ROOT / "curation"

# winner + finalist registers — filled by dated entries AFTER the gates run.
# curation_gates.py prints the values to paste here; scripts refuse to run
# with the placeholders still in place.
WINNER: str | None = "C01"              # D4 executed 2026-08-20: sole survivor
FINALISTS: list = ["M01", "M05", "M07"] # D27 (2026-08-20): null + best
                                        # single-lever rivals per axis


def ensure_curation_outdir() -> Path:
    CURATION_FIG_DIR.mkdir(parents=True, exist_ok=True)
    return CURATION_FIG_DIR


def evasive_mask(df, lat_g: float = EVASIVE_LAT_G):
    """Boolean mask of the evasive conjunction under the D24 rule.

    Uses only unconfounded channels (yaw, speed): peak |yaw| must exceed the
    yaw rate implied by lat_g at the window's mean speed. Windows whose
    implied threshold exceeds the physical yaw bound can never qualify,
    which is the built-in low-speed floor (no invented constant).
    """
    import numpy as np
    v = df["mean_speed_kmh"].to_numpy(dtype=float) / 3.6          # m/s
    with np.errstate(divide="ignore", invalid="ignore"):
        thr_degs = np.degrees(lat_g * G_MS2 / np.maximum(v, 1e-9))
    thr_degs = np.where(v <= 0, np.inf, thr_degs)
    qualifies = thr_degs <= YAW_PHYS_BOUND_DEGS
    return qualifies & (df["max_abs_yaw_degs"].to_numpy(dtype=float)
                        >= thr_degs)