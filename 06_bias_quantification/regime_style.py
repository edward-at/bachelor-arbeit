"""
regime_style.py
────────────────────────────────────────────────────────────────────────────
SINGLE SOURCE OF TRUTH for the five multivariate k-means REGIMES recovered in
Section 4.7 (RQ2). Import label/colour helpers from here; never hard-code regime
names or colours anywhere else.

TERMINOLOGY: "regime" = the 5 clustering categories (RQ2). Do NOT confuse with
the 4 fixed-threshold "variability tiers" of Section 4.5.6 (those live with the
TIER_* constants in the exploratory-analysis scripts).

Regime order is FIXED by clustering.py::_order_regimes — ascending mean of the
four z-scored change features (speed excluded). So regime 0 is the calmest and
regime 4 the most dynamically intense. The two LOWEST-intensity regimes are
0 (Calm/Stationary) and 1 (Highway Cruising): both are low on the change
features (Highway is fast but kinematically quiet), which is why they form the
"low-dynamic" group used by the RQ1 directional metric in Section 4.8.
────────────────────────────────────────────────────────────────────────────
"""

from tum_colors import TUM_SECONDARY, TUM_EXT

N_REGIMES = 5

# Ordered 0 (calmest) → 4 (most intense); order matches clustering.py.
REGIME_LABELS = [
    "Calm / Stationary",   # 0
    "Highway Cruising",    # 1
    "Turning",             # 2
    "Braking",             # 3
    "Accelerating",        # 4
]

REGIME_COLORS = [
    TUM_SECONDARY["blue2"],   # 0 Calm / Stationary
    TUM_EXT["maroon"],        # 1 Highway Cruising
    TUM_EXT["orange"],        # 2 Turning
    TUM_EXT["dark_green"],    # 3 Braking
    TUM_EXT["purple"],        # 4 Accelerating
]

# The two lowest-intensity regimes. Membership here defines the RQ1
# directional "low-dynamic share". Grounded in the clustering's own intensity
# ordering, NOT an arbitrary hand-drawn cut.
LOW_DYNAMIC_REGIMES = (0, 1)


def regime_label(i: int) -> str:
    return REGIME_LABELS[int(i)]


def regime_color(i: int) -> str:
    return REGIME_COLORS[int(i)]