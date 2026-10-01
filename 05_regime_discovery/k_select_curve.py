"""
k_select_curve.py  —  Section 4.7.5 figure (cluster-number selection)
────────────────────────────────────────────────────────────────────────────
Shows why k was chosen. Left panel: the mean silhouette against k together with
its null baseline (k-means on independently shuffled features), so the reader
sees both that the silhouette peaks and that it sits well above the separation
k-means imposes on structureless data. Right panel: the within-cluster inertia
against k, the elbow corroborating the choice. The selected k is marked in both.

Thin reader of dataset/regime_data/kselect_curves.parquet — no recomputation.
────────────────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from tum_colors import TUM_SECONDARY, TUM_ACCENT

sys.path.append(str(Path(__file__).resolve().parents[1]))
from thesis_paths import DATA_ROOT, REPO_ROOT  # noqa: E402
REGIME_DATA_DIR = DATA_ROOT / "regime_data"
OUT_DIR         = REPO_ROOT / "05_regime_discovery" / "figures" / "model_selection"
DPI             = 300


def main():
    df = pd.read_parquet(REGIME_DATA_DIR / "kselect_curves.parquet").sort_values("k")
    k_sel = int(df.loc[df["silhouette"].idxmax(), "k"])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))

    axL.plot(df["k"], df["silhouette"], color=TUM_ACCENT["green"], linewidth=2, label="silhouette")
    axL.plot(df["k"], df["null"], color=TUM_ACCENT["blue"], linewidth=2, label="null baseline")
    axL.axvline(k_sel, color=TUM_SECONDARY["dark_gray"], linewidth=1.5, label=f"selected k = {k_sel}")
    axL.set_xlabel("number of clusters k")
    axL.set_ylabel("mean silhouette")
    axL.set_title("Cluster separation vs k")
    axL.grid(True); axL.set_axisbelow(True); axL.legend()

    axR.plot(df["k"], df["inertia"], color=TUM_ACCENT["orange"], linewidth=2,
             label="within-cluster inertia")
    axR.axvline(k_sel, color=TUM_SECONDARY["dark_gray"], linewidth=1.5, label=f"selected k = {k_sel}")
    axR.set_xlabel("number of clusters k")
    axR.set_ylabel("within-cluster inertia (WCSS)")
    axR.set_title("Compactness vs k")
    axR.grid(True); axR.set_axisbelow(True); axR.legend()

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "kselect_curve.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  → {out}")


if __name__ == "__main__":
    main()