"""
regime_profile_heatmap.py  —  Section 4.6.5 figure (the RQ2 interpretability evidence)
────────────────────────────────────────────────────────────────────────────
Renders the regime-by-feature profile grid: which behaviour each recovered
regime represents, read straight off its feature centroid. Colour encodes each
feature standardised across the regimes (so the regime that stands highest on a
feature is darkest in that column), while the cell annotation is the mean in
physical units. This is the figure that turns the regime labels from assertions
into something the reader can verify.

Row labels come from regime_style.py (the single source of truth) so this figure
names regime 0 identically to every other figure in the chapter — never "Regime 0".

Thin reader of dataset/regime_data/cluster_profile.parquet — no recomputation,
no model JSON. Same arrangement as the Section 4.5 figures over std_data.
────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from regime_style import regime_label            # ← single source of truth for labels

PROJECT_ROOT    = Path("/Users/edward/Downloads")
REGIME_DATA_DIR = PROJECT_ROOT / "dataset" / "regime_data"
OUT_DIR         = PROJECT_ROOT / "bachelor-arbeit" / "regime_discovery" / "figures" / "regime_profile"
DPI             = 300

LABELS = {"mean_speed_kmh": "mean speed", "std_speed_kmh": "σ speed",
          "mean_acc_pos": "acc⁺", "mean_acc_neg": "acc⁻",
          "mean_abs_a_lat": "|a_lat|", "mean_abs_yaw_rate": "|yaw rate|",
          "mean_abs_jerk": "|jerk|"}


def _fmt(v: float) -> str:
    """Physical-unit cell label: more decimals for the small (rad/s) channels."""
    return f"{v:.3f}" if abs(v) < 1 else f"{v:.1f}"


def main():
    df = pd.read_parquet(REGIME_DATA_DIR / "cluster_profile.parquet").sort_values("regime")
    feats = [c for c in df.columns if c not in ("regime", "n_windows", "share_pct")]

    M = df[feats].to_numpy(float)                       # regimes × features, physical units
    std = M.std(0); std[std == 0] = 1.0                 # guard a constant feature column
    Z = (M - M.mean(0)) / std                           # standardise each feature across regimes

    fig, ax = plt.subplots(figsize=(1.4 * len(feats) + 2.0, 1.0 * len(df) + 1.5))
    im = ax.imshow(Z, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(feats)))
    ax.set_yticks(range(len(df)))
    ax.set_xticklabels([LABELS.get(f, f) for f in feats], rotation=30, ha="right")
    # behavioural names from regime_style — identical to every other figure
    ax.set_yticklabels([f"{regime_label(int(r))}\n({s:.0f}%)"
                        for r, s in zip(df["regime"], df["share_pct"])])
    ax.set_title("Dynamic regime feature profiles")

    # documented heatmap exception: a single small fontsize for the in-cell values
    for i in range(len(df)):
        for j in range(len(feats)):
            ax.text(j, i, _fmt(M[i, j]), ha="center", va="center", fontsize=8)

    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label("standardised across regimes (z-score)")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "regime_profile_heatmap.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  → {out}")


if __name__ == "__main__":
    main()