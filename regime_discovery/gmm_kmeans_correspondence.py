"""
kmeans_gmm_correspondence.py  —  Section 4.6.8 figure
────────────────────────────────────────────────────────────────────────────
The correspondence between the k-means partition and the Gaussian mixture, made
visible. Section 4.6.8 reports that correspondence as a single adjusted Rand
index; this figure shows WHERE the two agree and where they part.

Three panels:
  1. the projection coloured by the k-means regime          (the partition)
  2. the same windows coloured by the GMM hard assignment   (the cross-check)
  3. the correspondence matrix — rows = k-means regime, columns = GMM
     component, each row normalised to 100 %, so cell (i, j) reads "this
     percentage of k-means regime i was placed in GMM component j".

Panel 3 is the quantitative core: a strong diagonal means the two methods
recover the same regimes, and off-diagonal mass localises exactly which pair of
regimes the two draw differently — the structure the scalar index compresses
into one number.

The GMM components were already aligned to the k-means regime numbering in
gmm.py (Hungarian matching on the component means), so the two labellings share
an index and the diagonal is meaningful.

Thin reader of cluster_assignments.parquet + gmm_assignments.parquet, merged on
the window back-reference. No recomputation, no model file.
────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import adjusted_rand_score

from regime_style import regime_label, regime_color

PROJECT_ROOT    = Path("/Users/edward/Downloads")
REGIME_DATA_DIR = PROJECT_ROOT / "dataset" / "regime_data"
OUT_DIR         = PROJECT_ROOT / "bachelor-arbeit" / "regime_discovery" / "figures" / "gmm"

BACKREF  = ["dataset", "clip_id", "window_start_s", "window_end_s"]
N_PLOT   = 30000       # subsample for rendering only (matrix uses all windows)
SEED     = 42          # same subsample as the other projection figures
CROP_PCT = (1, 99)     # same crop as the other projection figures
CELL_FS  = 9           # documented heatmap annotation-font latitude
DPI      = 300


def _scatter(ax, s, col, title):
    for r in sorted(s[col].unique()):
        g = s[s[col] == r]
        # `s` (marker size) — same single deviation as the other scatters
        ax.scatter(g["pc1"], g["pc2"], color=regime_color(int(r)),
                   alpha=0.45, s=4, edgecolor="none")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title)
    ax.grid(True); ax.set_axisbelow(True)


def main():
    km = pd.read_parquet(REGIME_DATA_DIR / "cluster_assignments.parquet",
                         columns=BACKREF + ["regime", "pc1", "pc2"])
    gm = pd.read_parquet(REGIME_DATA_DIR / "gmm_assignments.parquet",
                         columns=BACKREF + ["gmm_regime"])
    df = km.merge(gm, on=BACKREF, how="inner")
    print(f"  merged {len(df):,} windows "
          f"(k-means {len(km):,} · gmm {len(gm):,})")

    k = int(max(df["regime"].max(), df["gmm_regime"].max())) + 1
    ari = adjusted_rand_score(df["regime"].to_numpy(), df["gmm_regime"].to_numpy())
    print(f"  ARI(k-means, GMM) = {ari:.4f}")

    # ── correspondence matrix on ALL windows, rows normalised to 100 % ────────
    C = (pd.crosstab(df["regime"], df["gmm_regime"])
           .reindex(index=range(k), columns=range(k), fill_value=0)
           .to_numpy(dtype=float))
    R = 100.0 * C / np.where(C.sum(1, keepdims=True) > 0, C.sum(1, keepdims=True), np.nan)

    print(f"\n  Correspondence (% of each k-means regime, by GMM component):")
    print("          " + "".join(f"{j:>8}" for j in range(k)))
    for i in range(k):
        print(f"    {i:>5} " + "".join(f"{R[i, j]:>8.1f}" for j in range(k)))

    # ── projections (subsampled + cropped for rendering only) ─────────────────
    x_lo, x_hi = np.percentile(df["pc1"], CROP_PCT)
    y_lo, y_hi = np.percentile(df["pc2"], CROP_PCT)
    px, py = 0.05 * (x_hi - x_lo), 0.05 * (y_hi - y_lo)
    s = df.sample(min(N_PLOT, len(df)), random_state=SEED)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))

    _scatter(axes[0], s, "regime", "k-means partition")
    _scatter(axes[1], s, "gmm_regime", "Gaussian-mixture hard assignment")
    for ax in axes[:2]:
        ax.set_xlim(x_lo - px, x_hi + px)
        ax.set_ylim(y_lo - py, y_hi + py)

    handles = [Line2D([0], [0], marker="o", linestyle="none",
                      color=regime_color(r), alpha=1.0, label=regime_label(r))
               for r in range(k)]
    axes[0].legend(handles=handles, fontsize=8)

    im = axes[2].imshow(R, cmap="Blues", vmin=0.0, vmax=100.0)
    axes[2].set_xticks(range(k))
    axes[2].set_xticklabels([regime_label(j) for j in range(k)],
                            rotation=30, ha="right", fontsize=8)
    axes[2].set_yticks(range(k))
    axes[2].set_yticklabels([regime_label(i) for i in range(k)], fontsize=8)
    axes[2].set_xlabel("Gaussian-mixture component")
    axes[2].set_ylabel("k-means regime")
    axes[2].set_title("Correspondence (row %)")
    for i in range(k):
        for j in range(k):
            if np.isfinite(R[i, j]):
                axes[2].text(j, i, f"{R[i, j]:.0f}", ha="center", va="center",
                             color=("white" if R[i, j] > 55 else "black"),
                             fontsize=CELL_FS)
    fig.colorbar(im, ax=axes[2], fraction=0.046, label="% of the k-means regime")

    fig.suptitle(f"Correspondence between the k-means partition and the "
                 f"Gaussian mixture (ARI = {ari:.3f})")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "kmeans_gmm_correspondence.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"\n  → {out}")


if __name__ == "__main__":
    main()