"""
regime_style.py  —  single source of truth for regime labels and colours
────────────────────────────────────────────────────────────────────────────
Every regime figure (the PCA scatters, the profile heatmap, and the validity /
GMM / HMM figures) reads its labels and colours from here, so the regimes are
named and coloured identically across the whole chapter. Change a label or a
colour ONCE, here, and every figure follows.

Index = regime number, ordered 0 = calmest .. k-1 = most dynamic (the same
deterministic intensity ordering clustering.py applies). Names are BEHAVIOURAL,
never tier-levels (low/medium/high), which are reserved for the variability tiers.

NOTE on regime 0 ("Calm / Stationary"): in the 20 s k-means layer this regime's
centroid is low-speed calm driving (~12 km/h); in the 2 s HMM the same aligned
regime's emission collapses to ~0 km/h (parked / idling), because at fine
resolution the stationary sub-windows separate out from low-speed-but-moving
ones. The label spans both — it is the calmest regime, from stationary through
low-speed — so it is honest at either resolution. (The 0 km/h HMM emission must
not be labelled "Calm Driving" alone.)
────────────────────────────────────────────────────────────────────────────
"""

from tum_colors import TUM_SECONDARY, TUM_EXT

REGIME_LABELS = ["Calm / Stationary", "Highway Cruising", "Turning", "Braking", "Accelerating"]
REGIME_COLORS = [TUM_SECONDARY["blue2"], TUM_EXT["maroon"], TUM_EXT["orange"],
                 TUM_EXT["dark_green"], TUM_EXT["purple"]]


def regime_label(i: int) -> str:
    return REGIME_LABELS[i] if 0 <= i < len(REGIME_LABELS) else f"Regime {i}"


def regime_color(i: int):
    return REGIME_COLORS[i % len(REGIME_COLORS)]