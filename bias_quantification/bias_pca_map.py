"""
bias_pca_map.py  —  exploratory map of the 14 datasets in bias-metric space.

Takes the kinematic bias metrics from the fingerprint table
(low_dynamic_share, Hill D1/D2, js_to_pooled, PSI, the three W1 channels),
z-scores each column, and projects the 14 datasets onto the first two
principal components. The result is a "map of dataset bias space": datasets
that express bias the same way sit together (e.g. the Ford/ZOD low-dynamic
cluster), and structural outliers (e.g. KITTI-360's turning-dominated route)
separate visibly.

Deliberate choices:
  * vol_share is EXCLUDED — volume is corpus composition, already carried by
    the Lorenz/volume-share-Gini figure; z-scored it would make PC1 an
    "is NVIDIA" axis and flatten the kinematic structure this map is for.
  * numpy SVD, no fitted model beyond centring/scaling — the chapter's
    "closed-form functionals only" defence line stays intact.

Honest caveats:
  * 14 samples x 8 features is EXPLORATORY. The map supports grouping
    statements only; no distances are quantified and no claims are made
    beyond what the underlying metric columns already show.
  * D1/D2 duplicate the information in low_dynamic_share only partially, but
    the feature set IS internally correlated — PC1 will largely be a
    "general bias expression" axis. That is expected and disclosed.
  * Confounded-acceleration datasets carry artifact-tinted W1 Acc values;
    their points are drawn with a black edge and an asterisked label.

Reads : the same sources as bias_fingerprint.py (via the shared builder)
Writes: ARTIFACT_DIR/bias_pca_coords.parquet
Output: figures/bias_pca_map.png

Strict matplotlib: pure defaults, dpi=300, TUM colours.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bias_tier2_common import (
    ARTIFACT_DIR, FIG_ROOT, DPI, CONFOUNDED_ACCEL,
    build_bias_metric_table, label, color,
)

OUT_DIR = FIG_ROOT

FEATURES = ["low_dynamic_share", "hill_d1", "hill_d2", "js_to_pooled",
            "psi_pooled", "w1_speed", "w1_acc", "w1_yaw"]


# ══════════════════════════════ build ══════════════════════════════
def build():
    tab = build_bias_metric_table(verbose=True)
    X = tab[FEATURES].to_numpy(dtype=float)

    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd == 0] = 1.0                      # constant column guard
    Z = (X - mu) / sd
    Z = np.nan_to_num(Z, nan=0.0)          # a missing cell contributes the column mean

    # PCA via SVD on the centred, scaled matrix
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    var = S ** 2 / (Z.shape[0] - 1)
    evr = var / var.sum()
    pcs = U * S                            # sample coordinates

    coords = pd.DataFrame({"dataset": tab.index,
                           "pc1": pcs[:, 0], "pc2": pcs[:, 1]})
    loadings = pd.DataFrame(Vt[:2].T, index=FEATURES, columns=["PC1", "PC2"])
    return coords, loadings, evr


# ═════════════════════════════ figure ══════════════════════════════
def fig_map(coords, evr):
    fig, ax = plt.subplots(figsize=(9, 7))
    for _, row in coords.iterrows():
        ds = row["dataset"]
        edge = "black" if ds in CONFOUNDED_ACCEL else "none"
        ax.scatter(row["pc1"], row["pc2"], color=color(ds),
                   edgecolor=edge, linewidth=1.2, label=label(ds))
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.3)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.3)
    ax.set_xlabel(f"PC1 ({evr[0] * 100:.0f} % of variance)")
    ax.set_ylabel(f"PC2 ({evr[1] * 100:.0f} % of variance)")
    ax.set_title("Datasets in bias-metric space (z-scored, kinematic metrics only)")
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    out = OUT_DIR / "bias_pca_map.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# ═══════════════════════════════ main ══════════════════════════════
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    coords, loadings, evr = build()

    pd.set_option("display.width", 200)
    print("\nExplained variance ratio (PC1..):",
          np.round(evr[: min(4, evr.size)], 3))
    print("\nLoadings (which metrics drive each axis):\n")
    print(loadings.round(3).to_string())
    print("\nCoordinates:\n")
    shown = coords.copy()
    shown["dataset"] = [label(d) for d in shown["dataset"]]
    print(shown.round(3).to_string(index=False))

    artifact = ARTIFACT_DIR / "bias_pca_coords.parquet"
    coords.to_parquet(artifact, index=False)
    print(f"\nArtifact: {artifact}")

    p = fig_map(coords, evr)
    print(f"Output: {p}")


if __name__ == "__main__":
    main()