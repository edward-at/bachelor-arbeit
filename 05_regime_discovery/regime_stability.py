"""
regime_stability.py  —  Sections 4.7.8–4.7.9 figure
Reads bootstrap_jaccard.parquet → per-regime mean bootstrap-Jaccard, coloured by
regime, with Hennig's reference lines (≥0.75 stable, ≥0.85 highly stable). Legend
placed lower-right so it does not sit on top of the (near-full-height) bars.

The mean value is annotated above each bar: the scores sit close together near
the top of the scale, where bar height alone does not separate them. Thin reader.
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from tum_colors import TUM_SECONDARY
from regime_style import regime_label, regime_color

sys.path.append(str(Path(__file__).resolve().parents[1]))
from thesis_paths import DATA_ROOT, REPO_ROOT  # noqa: E402
REGIME_DATA_DIR = DATA_ROOT / "regime_data"
OUT_DIR         = REPO_ROOT / "05_regime_discovery" / "figures" / "validity"
DPI             = 300

J_STABLE, J_HIGH = 0.75, 0.85
ANNOT_FS         = 9        # documented annotation-font latitude


def main():
    df = pd.read_parquet(REGIME_DATA_DIR / "bootstrap_jaccard.parquet").sort_values("regime")
    regs = df["regime"].astype(int).tolist()
    vals = df["mean_jaccard"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.bar(regs, vals, color=[regime_color(r) for r in regs],
           edgecolor="black", linewidth=0.5)

    # annotate the mean above each bar — the bars are near-equal in height
    for r, v in zip(regs, vals):
        ax.text(r, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=ANNOT_FS)

    ax.axhline(J_HIGH, color=TUM_SECONDARY["dark_gray"], alpha=0.9,
               label=f"highly stable (≥{J_HIGH})")
    ax.axhline(J_STABLE, color=TUM_SECONDARY["gray"], alpha=0.9,
               label=f"stable (≥{J_STABLE})")
    ax.set_xticks(regs)
    ax.set_xticklabels([regime_label(r) for r in regs], rotation=30, ha="right")
    ax.set_ylabel("Mean bootstrap-Jaccard")
    ax.set_ylim(0.0, 1.08)          # headroom for the annotations
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