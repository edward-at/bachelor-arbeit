"""
bias_stacked_bar.py  —  Section 4.8 HEADLINE figure
════════════════════════════════════════════════════════════════════════════
Horizontal stacked bars: each dataset's five regime shares, datasets ordered by
low-dynamic share (RQ1). Two reference bars appended below a separator — POOLED
(volume-weighted) and BALANCED (one vote per dataset) — so the corpus-level view
and the volume-induced gap are visible in the same figure.

Reads : ARTIFACT_DIR/bias_per_dataset.parquet, bias_corpus_profiles.parquet
Output: 06_bias_quantification/figures/bias_stacked_bar.png
════════════════════════════════════════════════════════════════════════════
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tum_colors import TUM_PRIMARY, TUM_SECONDARY, TUM_ACCENT, TUM_EXT
from regime_style import N_REGIMES, regime_label, regime_color
from bias_quant import CONFOUNDED_ACCEL


def dname(ds, names):
    """Display name with '*' for accel-confounded datasets (see bias_quant)."""
    n = names.get(ds, ds)
    return n + "*" if ds in CONFOUNDED_ACCEL else n

# ── paths and constants ──────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parents[1]))
from thesis_paths import DATA_ROOT, REPO_ROOT  # noqa: E402
ARTIFACT_DIR = DATA_ROOT / "regime_data"
# bias-quantification code + figures live under 06_bias_quantification/
OUT_DIR      = REPO_ROOT / "06_bias_quantification" / "figures"

DISPLAY_NAMES = {
    "boreas_rt": "Boreas", "durlar": "DurLAR", "ford": "Ford", "kitti": "KITTI",
    "kitti360": "KITTI-360", "nuscene": "nuScenes", "nvidia": "NVIDIA",
    "oxford": "Oxford", "waymo": "Waymo", "zod": "ZOD",
    "4seasons": "4Seasons", "hdd": "HDD", "a2d2": "A2D2", "kitscenes": "KITScenes",
}
DPI = 300

SHARE_COLS = [f"share_{i}" for i in range(N_REGIMES)]
SORT_DESCENDING = True   # most low-dynamic-skewed dataset at the top


def _rows(per, profiles):
    order = per.sort_values("low_dynamic_share", ascending=not SORT_DESCENDING).index
    labels, mat = [], []
    for ds in order:
        labels.append(dname(ds, DISPLAY_NAMES))
        mat.append([per.loc[ds, c] * 100 for c in SHARE_COLS])
    labels.append("")                       # spacer row (invisible)
    mat.append([0.0] * N_REGIMES)
    labels.append("Pooled (volume-weighted)")
    mat.append([profiles.loc["pooled", c] * 100 for c in SHARE_COLS])
    labels.append("Balanced (per-dataset mean)")
    mat.append([profiles.loc["balanced", c] * 100 for c in SHARE_COLS])
    return labels, np.array(mat)


def plot_stacked_bar(per, profiles, out_dir):
    labels, mat = _rows(per, profiles)
    n = len(labels)
    y = np.arange(n)[::-1]                    # first row at the top

    fig, ax = plt.subplots(figsize=(12, 8))
    left = np.zeros(n)
    for i in range(N_REGIMES):
        ax.barh(y, mat[:, i], left=left, color=regime_color(i),
                edgecolor="white", linewidth=0.6, label=regime_label(i))
        left += mat[:, i]

    # separator line above the two reference bars (spacer is the 3rd-from-bottom row)
    ax.axhline(y[-3] + 0.5, color=TUM_SECONDARY["gray"], linewidth=1.0)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    # reserve an empty band BELOW the bottom bar so the legend does not cover it
    ax.set_ylim(-2.6, y.max() + 0.8)
    ax.set_xlabel("Share of 20 s windows (%) ")
    ax.set_title("Per-dataset dynamic-regime composition, ordered by low-dynamic share")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", ncol=N_REGIMES)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bias_stacked_bar.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  [SAVED] {out_path}")


def main():
    per = pd.read_parquet(ARTIFACT_DIR / "bias_per_dataset.parquet")
    profiles = pd.read_parquet(ARTIFACT_DIR / "bias_corpus_profiles.parquet")
    plot_stacked_bar(per, profiles, OUT_DIR)


if __name__ == "__main__":
    main()