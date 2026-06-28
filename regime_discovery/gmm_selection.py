"""
gmm_selection.py  —  Section 4.6.8 figure
Reads gmm_selection.parquet → BIC and AIC across the component-count sweep.

HONEST FRAMING: BIC/AIC are generative criteria; on a continuous feature space
they keep preferring more components (the curve is still descending at the range
edge — over-clustering of a continuum, NOT evidence for that many regimes). The
regime count is therefore set by the SEPARATION criteria (silhouette, with
Davies–Bouldin and Calinski–Harabasz in §4.6.7), marked here as the selected k.
The BIC curve is shown to demonstrate that the generative criterion does not
bottom out within the tested range, which is reported as such — it does not
corroborate the count, and the figure must not imply the range-edge minimum is
the answer. Thin reader.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from tum_colors import TUM_ACCENT, TUM_EXT, TUM_SECONDARY

PROJECT_ROOT    = Path("/Users/edward/Downloads")
REGIME_DATA_DIR = PROJECT_ROOT / "dataset" / "regime_data"
OUT_DIR         = PROJECT_ROOT / "bachelor-arbeit" / "regime_discovery" / "figures" / "gmm"
DPI             = 300


def main():
    df = pd.read_parquet(REGIME_DATA_DIR / "gmm_selection.parquet")
    n_bic = int(df["bic_selected_n"].iloc[0])
    k_sel = int(df["selected_k"].iloc[0])
    n_max = int(df["n"].max())
    still_falling = n_bic >= n_max          # BIC min is at the range edge → not a true minimum

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.plot(df["n"], df["bic"], color=TUM_ACCENT["blue"], linewidth=1, label="BIC")
    ax.plot(df["n"], df["aic"], color=TUM_ACCENT["orange"], linewidth=1, label="AIC")

    # selected value (prominent) — from the separation criteria, not BIC
    ax.axvline(k_sel, color=TUM_EXT["red"], linewidth=2.2,
               label=f"selected k = {k_sel} (separation criteria)")
    # BIC behaviour (de-emphasised) — keeps decreasing, does not select a count
    bic_lbl = (f"BIC still decreasing at range edge (n = {n_bic})" if still_falling
               else f"BIC minimum (n = {n_bic})")
    ax.axvline(n_bic, color=TUM_SECONDARY["dark_gray"], linewidth=1.4, alpha=0.9, label=bic_lbl)

    ax.set_xlabel("Number of components")
    ax.set_ylabel("Information criterion")
    ax.set_title("Gaussian-mixture component count: generative vs separation criteria")
    ax.grid(True); ax.set_axisbelow(True); ax.legend()
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "gmm_selection.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  → {out}")


if __name__ == "__main__":
    main()