"""
regime_stability.py  —  Section 4.6.7 figure
Reads bootstrap_jaccard.parquet → per-regime mean bootstrap-Jaccard, coloured by
regime, with Hennig's reference lines (≥0.75 stable, ≥0.85 highly stable). Legend
placed lower-right so it does not sit on top of the (near-full-height) bars.
Thin reader.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from tum_colors import TUM_SECONDARY
from regime_style import regime_label, regime_color

PROJECT_ROOT    = Path("/Users/edward/Downloads")
REGIME_DATA_DIR = PROJECT_ROOT / "dataset" / "regime_data"
OUT_DIR         = PROJECT_ROOT / "bachelor-arbeit" / "regime_discovery" / "figures" / "validity"
DPI             = 300

J_STABLE, J_HIGH = 0.75, 0.85


def main():
    df = pd.read_parquet(REGIME_DATA_DIR / "bootstrap_jaccard.parquet").sort_values("regime")
    regs = df["regime"].astype(int).tolist()

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.bar(regs, df["mean_jaccard"], color=[regime_color(r) for r in regs],
           edgecolor="black", linewidth=0.5)
    ax.axhline(J_HIGH, color=TUM_SECONDARY["dark_gray"], alpha=0.9,
               label=f"highly stable (≥{J_HIGH})")
    ax.axhline(J_STABLE, color=TUM_SECONDARY["gray"], alpha=0.9,
               label=f"stable (≥{J_STABLE})")
    ax.set_xticks(regs)
    ax.set_xticklabels([regime_label(r) for r in regs], rotation=30, ha="right")
    ax.set_ylabel("Mean bootstrap-Jaccard")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Per-regime cluster stability")
    ax.grid(True); ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "regime_stability.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  → {out}")


if __name__ == "__main__":
    main()