#!/usr/bin/env python3
"""subset_eval_common.py — shared constants and loaders for the 12.1 figures.

Every figure script imports from here, so method order, labels, colours and
cache paths have exactly ONE definition. All figures read ONLY the eval cache
written by subset_eval_harness.py (compute-once discipline).

Style sources of truth (mirrored, never re-invented):
  * regime_style.py        — regime labels/colours (direct import). Settles
                             the 3-vs-4 question: 3 = Braking, 4 = Accelerating.
  * bias_tier2_common.py   — DATASET_COLORS / DISPLAY_NAMES convention.
  * tum_colors.py          — palette; the single-series colour everywhere is
                             TUM_PRIMARY['tum_blue'] (#0065BD), never a light
                             accent shade.
Method-series colouring: one colour per DESIGN FAMILY (replica / allocation /
probability / rule / coverage) so a method's family is readable across every
figure; flip METHOD_COLORS to {m: MAIN_BLUE} for a uniform view.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from tum_colors import TUM_PRIMARY, TUM_SECONDARY, TUM_ACCENT, TUM_EXT
from regime_style import REGIME_LABELS, REGIME_COLORS, regime_label, regime_color

try:                                    # optional exports in newer versions
    from regime_style import N_REGIMES
except ImportError:
    N_REGIMES = len(REGIME_LABELS)
try:
    from regime_style import LOW_DYNAMIC_REGIMES
except ImportError:
    LOW_DYNAMIC_REGIMES = None

# ---------------------------------------------------------------------------
# paths (THESIS_ROOT env override exists only for sandbox testing)
# figures moved: regime_discovery -> subset_curation (Edward, 2026-07-08)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("THESIS_ROOT", "/Users/edward/Downloads"))
EVAL_DIR = PROJECT_ROOT / "dataset" / "subset_data" / "eval"
FIG_ROOT = PROJECT_ROOT / "bachelor-arbeit" / "subset_curation" / "figures"
OUT_DIR = FIG_ROOT / "subset_eval"
DPI = 300

# Budget-tier size = one rarest-regime supply (Section 11.3). Recomputed on
# the current 402,421-window frame, where the rarest regime is regime 4.
# CHANGE HERE ONLY: the tier directory name is derived from it downstream.
TIER = "budget"                      # headline population
BUDGET_N = 71_390

MAIN_BLUE = TUM_SECONDARY["blue2"]  # the thesis single-series colour

# ---------------------------------------------------------------------------
# method slate: fixed M-order everywhere (never performance-ordered pre-rule)
# ---------------------------------------------------------------------------
METHODS = ["M01", "M02", "M03", "M04", "M05",
           "M06", "M07", "M08", "M09", "M10", "M11"]
METHOD_SHORT = {
    "M01": "SRS", "M02": "Proportional", "M03": "Equal",
    "M04": "Power q=0.5", "M05": "Two-way", "M06": "PPS score",
    "M07": "Trigger", "M08": "Predicates", "M09": "Kennard-Stone",
    "M10": "Cluster fill", "M11": "Neyman",
}
ROW_LABELS = {"R0_frame": "Frame", "R1_guarded": "Guarded frame",
              **{m: f"{m} {METHOD_SHORT[m]}" for m in METHODS}}

# design-family colour (Section 11.5-11.9 families)
METHOD_COLORS = {
    "M01": TUM_SECONDARY["gray"],      # replicas
    "M02": TUM_SECONDARY["gray"],
    "M03": TUM_ACCENT["blue"],     # allocation
    "M04": TUM_ACCENT["blue"],
    "M05": TUM_ACCENT["blue"],
    "M06": TUM_ACCENT["green"],            # probability
    "M07": TUM_ACCENT["orange"],          # rule-based
    "M08": TUM_ACCENT["orange"],
    "M09": TUM_SECONDARY["light_gray"],          # coverage / geometric
    "M10": TUM_SECONDARY["light_gray"],
    "M11": TUM_EXT["purple"],          # variance-optimal (Section 11.10)
}
FAMILY_LEGEND = [("Replica", TUM_SECONDARY["gray"]),
                 ("Allocation", TUM_ACCENT["blue"]),
                 ("Probability", TUM_ACCENT["green"]),
                 ("Rule-based", TUM_ACCENT["orange"]),
                 ("Coverage", TUM_SECONDARY["light_gray"]),
                 ("Variance-optimal", TUM_EXT["purple"])]

CONFOUNDED = {"a2d2", "4seasons", "durlar"}

# 14-dataset palette / names — verbatim from bias_tier2_common.py
DISPLAY_NAMES = {
    "boreas_rt": "Boreas", "durlar": "DurLAR", "ford": "Ford", "kitti": "KITTI",
    "kitti360": "KITTI-360", "nuscene": "nuScenes", "nvidia": "NVIDIA",
    "oxford": "Oxford", "waymo": "Waymo", "zod": "ZOD",
    "4seasons": "4Seasons", "hdd": "HDD", "a2d2": "A2D2",
    "kitscenes": "KITScenes",
}
DATASET_COLORS = {
    "4seasons":  TUM_SECONDARY["blue2"],
    "a2d2":      TUM_SECONDARY["dark_gray"],
    "hdd":       TUM_SECONDARY["gray"],
    "kitscenes": TUM_PRIMARY["black"],
    "boreas_rt": TUM_EXT["maroon"],
    "durlar":    TUM_SECONDARY["blue1"],
    "ford":      TUM_EXT["orange"],
    "kitti":     TUM_EXT["yellow"],
    "kitti360":  TUM_EXT["olive"],
    "nuscene":   TUM_EXT["dark_green"],
    "nvidia":    TUM_EXT["teal"],
    "oxford":    TUM_EXT["navy"],
    "waymo":     TUM_EXT["purple"],
    "zod":       TUM_EXT["red"],
}
DATASET_COLOR_FALLBACK = TUM_SECONDARY["blue2"]

LOW_DYN_REGIME = 0
if LOW_DYNAMIC_REGIMES is not None and set(LOW_DYNAMIC_REGIMES) != {0}:
    print(f"[subset_eval_common] NOTE: regime_style.LOW_DYNAMIC_REGIMES = "
          f"{sorted(LOW_DYNAMIC_REGIMES)}, but the Chapter-12 harness metric "
          f"low_dyn_share is the regime-0 share only. Keep the two names "
          f"distinct in prose.")

# ---------------------------------------------------------------------------
# cache loaders (everything reads EVAL_DIR only)
# ---------------------------------------------------------------------------
_LONG = None
_ENV = None


def load_master(tier: str = TIER) -> pd.DataFrame:
    path = EVAL_DIR / f"master_table_{tier}.csv"
    if not path.exists():
        raise SystemExit(f"[subset_eval_common] missing {path} — run "
                         f"subset_eval_harness.py first")
    return pd.read_csv(path, index_col=0)


def present_methods(df: pd.DataFrame) -> list:
    return [m for m in METHODS if m in df.index]


def load_long() -> pd.DataFrame:
    global _LONG
    if _LONG is None:
        _LONG = pd.read_parquet(EVAL_DIR / "metrics_long.parquet")
    return _LONG


def metric_matrix(prefix: str, tier: str = TIER) -> pd.DataFrame:
    d = load_long()
    d = d[(d["tier"] == tier) & (d["variant"] == "main")
          & d["metric"].str.startswith(prefix)]
    return d.pivot(index="method", columns="metric", values="value")


def envelope_band(metric: str, lo: float = 2.5, hi: float = 97.5):
    global _ENV
    path = EVAL_DIR / "srs_envelope_budget.parquet"
    if _ENV is None:
        if not path.exists():
            return None
        _ENV = pd.read_parquet(path)
    v = _ENV.loc[_ENV["metric"] == metric, "value"].dropna().values
    if len(v) == 0:
        return None
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def load_cls() -> dict:
    path = EVAL_DIR / "tautology_map.csv"
    if not path.exists():
        return {}
    t = pd.read_csv(path)
    return {(r.metric, r.method): r.cls for r in t.itertuples(index=False)}


CLS_MARK = {"T": "\u25cf", "P": "\u25d0", "O": "", "N": ""}


def load_dist(tier: str = TIER) -> pd.DataFrame:
    path = EVAL_DIR / f"dist_cache_{tier}.parquet"
    if not path.exists():
        raise SystemExit(f"[subset_eval_common] missing {path} — re-run "
                         f"subset_eval_harness.py (v12.1.1+ writes it)")
    return pd.read_parquet(path)


def dist_arrays(dist: pd.DataFrame, channel: str, rows: list) -> list:
    g = dist[dist["channel"] == channel].groupby("method")["value"]
    have = {k: v.values.astype(float) for k, v in g}
    return [have.get(r, np.array([])) for r in rows]


def ensure_outdir() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR