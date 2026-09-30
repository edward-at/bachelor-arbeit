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
OUT_DIR         = PROJECT_ROOT / "bachelor-arbeit" / "05_regime_discovery" / "figures" / "regime_profile"
DPI             = 300

LABELS = {"mean_speed_kmh": "mean speed (km/h)", "std_speed_kmh": "σ speed (km/h)",
          "mean_acc_pos": "acc⁺ (m/s²)", "mean_acc_neg": "acc⁻ (m/s²)",
          "mean_abs_a_lat": "|a_lat| (m/s²)", "mean_abs_yaw_rate": "|yaw rate| (°/s)",
          "mean_abs_jerk": "|jerk| (m/s³)"}

# display convention of Section 4.6.1: yaw rate is stored in rad/s and shown in °/s.
# The conversion is applied to the printed cell only; the standardization across
# regimes that drives the color is scale-invariant, so no color changes (block 5D).
DISPLAY_SCALE = {"mean_abs_yaw_rate": 180.0 / np.pi}
# one fixed precision per feature so a column reads uniformly
CELL_FMT = {"mean_speed_kmh": "{:.1f}", "std_speed_kmh": "{:.1f}",
            "mean_acc_pos": "{:.3f}", "mean_acc_neg": "{:.3f}",
            "mean_abs_a_lat": "{:.3f}", "mean_abs_yaw_rate": "{:.2f}",
            "mean_abs_jerk": "{:.3f}"}


def _fmt(v: float, feat: str = "") -> str:
    """Physical-unit cell label with the feature's fixed precision."""
    return CELL_FMT.get(feat, "{:.3f}" if abs(v) < 1 else "{:.1f}").format(v)


def main():
    df = pd.read_parquet(REGIME_DATA_DIR / "cluster_profile.parquet").sort_values("regime")
    feats = [c for c in df.columns if c not in ("regime", "n_windows", "share_pct")]

    for f, k in DISPLAY_SCALE.items():                  # display units only (block 5D)
        if f in df.columns:
            df[f] = df[f] * k
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
            ax.text(j, i, _fmt(M[i, j], feats[j]), ha="center", va="center", fontsize=8)

    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label("standardized across regimes (z-score)")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "regime_profile_heatmap.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  → {out}")


if __name__ == "__main__":
    main()