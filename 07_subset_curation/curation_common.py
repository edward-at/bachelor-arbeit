#!/usr/bin/env python3
"""curation_common.py — shared constants for every curation script (thesis 4.11, 4.12).

Convention: scripts are prefixed curation_, figures go to
figures/curation/, data artifacts keep C0x_*.parquet naming inside the
sampling subset directories (thesis 4.9) so the harness and frame pass
discover them.

Everything the evaluation stage (thesis 4.10) already defines is IMPORTED
from subset_eval_common, never redefined (one-definition rule). This module
adds only what is new in the curation stage: the C-slate, its seeds, the gate
constants, the evasive held-out metric rule, and the curation figure directory.

Curation protocol (thesis 4.11, 4.12), declared before any curation subset
was evaluated:
  Selection rule: four gates then fidelity tie-break, see GATES / TIE_BREAK
       below. Tie tolerance = central-95% half-width of the K=100 SRS null
       for the tie-break metric (fallback 0.01 km/h, disclosed if used).
  Fairness definition P1-P5. Excluded properties are discussed in thesis
       4.11 (operational table) and Section 6.6 (two ceilings).
  Slate = C01 full composite, C02 no-events ablation, C03 no-cap ablation.
       M03 is the free fourth cell of the 2x2.
  Event reserve = ALL guarded 0.6 g windows (N_EVENT_RESERVE, derived by
       canonical.py), debited, role-tagged, class O, never dropped for any cap.
  Source lever = per-regime top-source cap at 0.50 of the regime quota,
       cap-not-equalise. A feasibility relaxation is disclosed, never
       silent. d2_sources is reported as diagnostic, not gated.
  Evasive held-out metric = implied-lateral derivation rule
       peak|yaw| >= EVASIVE_LAT_G * g / mean_speed, constant fixed after the
       curation_event_audit.py counts, optimised by nothing.
  Numbers-first selection: gates applied on tables + null band, figures
       for the winner only, duplication rule cross-ref / finalist overlay /
       named-reference overlay.
  The pipeline is stated in two modes (multi-corpus with Stage 0,
       single-corpus entering at Stage 1).
"""

from __future__ import annotations

from pathlib import Path

from subset_eval_common import (  # noqa: F401  (re-exported on purpose)
    DATA_ROOT, REPO_ROOT, EVAL_DIR, FIG_ROOT, DPI, BUDGET_N, METHODS, METHOD_SHORT,
    METHOD_COLORS, ROW_LABELS, DISPLAY_NAMES, DATASET_COLORS, CONFOUNDED,
    MAIN_BLUE, load_master, load_long, envelope_band, load_cls, load_dist,
)

# ---------------------------------------------------------------------------
# curation slate: IDs, files, seeds, labels, colour
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
# seed indices continue the sampling-stage keying (M01..M11 = 1..11)
C_SEED_INDEX = {"C01": 12, "C02": 13, "C03": 14}
from tum_colors import TUM_SECONDARY

CURATION_COLOR = MAIN_BLUE              # = TUM_SECONDARY["blue2"], #005293.
                                        # The winner uses the thesis
                                        # single-series colour, so C-figures
                                        # match the evaluation-figure
                                        # convention by definition, not by
                                        # coincidence.
GHOST = TUM_SECONDARY["light_gray"]     # #CCCCCC, 20% black — the frame
                                        # ghost fill in every C-figure
FRAME_LINE = GHOST                      # frame lines use the SAME
                                        # light_gray as the fills.
                                        # Visibility comes from linewidth
                                        # (thick pale underlay, thin blue2
                                        # series on top), not from a darker
                                        # hue.
# Every colour in the curation figures comes from the TUM palette. Only
# black and white may be literal.

# the 2x2 ablation grid (rows in slate order; M03 is the free corner)
ABLATION_ORDER = ["M03", "C02", "C03", "C01"]

# ---------------------------------------------------------------------------
# gate constants (reference-derived, grounds stated in thesis 4.11, 4.12)
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
SOURCE_CAP_FRAC = 0.50                  # per-regime top-source cap

# ---------------------------------------------------------------------------
# evasive held-out metric (derivation rule, not a constant threshold)
# implied lateral acceleration: a_lat = v * yaw  =>  yaw_thr(v) = a_g*g / v
# EVASIVE_LAT_G was fixed after the curation_event_audit.py counts were read
# (same literature status as the 0.6 g longitudinal trigger, thesis 4.9.8).
# ---------------------------------------------------------------------------
G_MS2 = 9.80665

EVASIVE_LAT_G = 0.75                    # declared before any curation subset
                                        # was evaluated. n=2,294 (0.57% of
                                        # frame) at 0.75 g, while 0.50 g
                                        # admits 3.7% (brisk cornering, not
                                        # evasion). Literature lineage: the
                                        # NDS lateral-trigger studies, as for
                                        # the 0.6 g constant.
EVASIVE_SWEEP_G = [0.40, 0.50, 0.60, 0.75]
YAW_PHYS_BOUND_DEGS = 90.0              # physical validity bound, self-floors
                                        # the rule at low speed (no speed floor)

# Derived audit constants. Placed BELOW the evasive constants on purpose:
# canonical._evasive_consts() imports EVASIVE_LAT_G / YAW_PHYS_BOUND_DEGS /
# G_MS2 from THIS module, so importing C above them would be a circular
# import that canonical's try/except turns into a silent literal fallback.
# The order of these lines is load-bearing.
from canonical import C
N_EVENT_RESERVE = C.event_reserve_n        # 1,088 on the current frame
EXPECTED_EVASIVE = C.evasive_n             # 2,294 at 0.75 g

# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
CURATION_FIG_DIR = FIG_ROOT / "curation"

# winner + finalist registers, filled AFTER the gates run with the values
# curation_gates.py prints. Scripts refuse to run with the placeholders
# still in place.
WINNER: str | None = "C01"              # selection rule applied: sole survivor
FINALISTS: list = ["M01", "M05", "M07"] # fixed before evaluation: null + best
                                        # single-lever rivals per axis


def ensure_curation_outdir() -> Path:
    CURATION_FIG_DIR.mkdir(parents=True, exist_ok=True)
    return CURATION_FIG_DIR


def evasive_mask(df, lat_g: float = EVASIVE_LAT_G):
    """Boolean mask of the evasive conjunction under the evasive rule.

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