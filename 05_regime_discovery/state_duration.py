"""
state_durations.py  —  Section 4.6.9 figure
Reads hmm_durations.parquet → MEDIAN dwell time per regime (seconds), coloured by
regime. Median, not mean: the dwell distributions are heavily right-skewed (a few
very long recordings inflate the mean), so the median is the honest "typical
dwell before switching". The full per-regime spread (p25–p90, max, and the mean)
is carried in the parquet. Thin reader.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from regime_style import regime_label, regime_color

PROJECT_ROOT    = Path("/Users/edward/Downloads")
REGIME_DATA_DIR = PROJECT_ROOT / "dataset" / "regime_data"
OUT_DIR         = PROJECT_ROOT / "bachelor-arbeit" / "05_regime_discovery" / "figures" / "hmm"
DPI             = 300


def main():
    df = pd.read_parquet(REGIME_DATA_DIR / "hmm_durations.parquet").sort_values("regime")
    regs = df["regime"].astype(int).tolist()

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.bar(regs, df["median_s"], color=[regime_color(r) for r in regs],
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(regs)
    ax.set_xticklabels([regime_label(r) for r in regs], rotation=30, ha="right")
    ax.set_ylabel("Median dwell time (s)")
    ax.set_title("Per-regime dwell time before switching (HMM)")
    ax.grid(True); ax.set_axisbelow(True)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "state_durations.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  → {out}")


if __name__ == "__main__":
    main()