"""
Script: coverage_heatmaps.py
=============================
Bachelor Thesis: "Identifying Dynamic Biases in Large-Scale Public Datasets
via Distributional Analysis"

Produces three coverage heatmaps for all 13 datasets × 6 kinematic signals:

  1. signal_coverage.png  — % non-NaN values per signal per dataset
  2. non_zero.png         — % non-zero values (among non-NaN) per signal per dataset
  3. boundary.png         — % values within physical bounds per signal per dataset

Physical bounds (same as distributional_analysis.py):
  speed_kmh : [0,   200]  km/h
  a_long    : [-15,  15]  m/s²
  a_lat     : [-15,  15]  m/s²
  jerk_long : [-30,  30]  m/s³
  jerk_lat  : [-30,  30]  m/s³
  yaw_rate  : [-1.57, 1.57] rad/s  (stored in rad/s, bounds applied before conversion)

Output: figures/Coverage/signal_coverage.png
                          non_zero.png
                          boundary.png

Usage:
    python coverage_heatmaps.py
    python coverage_heatmaps.py /path/to/inertial_exploratory_analysis
"""

import sys
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import os as _os
_os.environ.setdefault("PYTHONWARNINGS", "ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# 6 active signals — column names as stored in parquet
SIGNALS = [
    {"col": "speed_kmh",  "label": "Speed\n(km/h)"},
    {"col": "a_long",     "label": "Long. Accel\n(m/s²)"},
    {"col": "a_lat",      "label": "Lat. Accel\n(m/s²)"},
    {"col": "jerk_long",  "label": "Long. Jerk\n(m/s³)"},
    {"col": "jerk_lat",   "label": "Lat. Jerk\n(m/s³)"},
    {"col": "yaw_rate",   "label": "Yaw Rate\n(rad/s)"},
]

# Physical bounds — applied to raw stored values (yaw_rate still in rad/s here)
BOUNDS = {
    "speed_kmh":  (0.0,    200.0),
    "a_long":     (-15.0,   15.0),
    "a_lat":      (-15.0,   15.0),
    "jerk_long":  (-30.0,   30.0),
    "jerk_lat":   (-30.0,   30.0),
    "yaw_rate":   (-1.57,   1.57),   # rad/s
}

# Display order for datasets (rows), top to bottom
DATASET_ORDER = [
    "4seasons", "a2d2", "boreas_rt", "durlar", "ford",
    "hdd", "kitti", "kitti360", "nuscene", "nvidia",
    "oxford", "waymo", "zod", "kitscenes"
]

# Sample at most this many frames per dataset to keep runtime reasonable.
# Set to None to read all frames (slow for NVIDIA/HDD/Boreas).
MAX_FRAMES_PER_DATASET = None


# ─────────────────────────────────────────────────────────────────────────────
# PATH RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────

def _find_project_root() -> Path:
    candidates = []
    if len(sys.argv) > 1:
        candidates.append(Path(sys.argv[1]).expanduser().resolve())
    script_dir = Path(__file__).parent.resolve()
    candidates += [
        script_dir,
        Path.cwd().resolve(),
        script_dir.parent,
        Path("/Users/edward/Downloads/bachelor-arbeit/inertial_exploratory_analysis"),
        Path.home() / "Downloads" / "bachelor-arbeit" / "inertial_exploratory_analysis",
        Path.home() / "Downloads" / "dataset",
    ]
    for root in candidates:
        if (root / "inertial_data").is_dir():
            return root
    checked = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not find 'inertial_data/' in any of:\n  " + checked
    )


# ─────────────────────────────────────────────────────────────────────────────
# PER-DATASET WORKER
# ─────────────────────────────────────────────────────────────────────────────

def _compute_dataset_coverage(args: tuple) -> dict | None:
    """
    Worker: reads all clips for one dataset, computes three coverage metrics
    for each of the 6 signals. Returns a dict keyed by signal col name.
    """
    ds_dir, ds_name = args
    import pyarrow.parquet as pq

    cols = [s["col"] for s in SIGNALS]

    parquet_files = sorted(ds_dir.glob("clip_*.parquet"))
    if not parquet_files:
        return None

    # Peek schema once
    try:
        schema_cols = set(pq.read_schema(parquet_files[0]).names)
        cols_to_read = [c for c in cols if c in schema_cols]
    except Exception:
        cols_to_read = cols

    chunks = []
    total_read = 0

    for fp in parquet_files:
        if MAX_FRAMES_PER_DATASET and total_read >= MAX_FRAMES_PER_DATASET:
            break
        try:
            df = pd.read_parquet(fp, columns=cols_to_read)
            chunks.append(df)
            total_read += len(df)
        except Exception:
            continue

    if not chunks:
        return None

    data = pd.concat(chunks, ignore_index=True)

    # Fill missing columns with NaN
    for col in cols:
        if col not in data.columns:
            data[col] = np.nan

    n_total = len(data)
    results = {"dataset": ds_name, "n_frames": n_total}

    for sig in SIGNALS:
        col = sig["col"]
        arr = data[col].values.astype(float)
        lo, hi = BOUNDS[col]

        n_non_nan  = int(np.sum(~np.isnan(arr)))
        n_non_zero = int(np.sum((arr != 0) & ~np.isnan(arr)))
        n_in_bounds = int(np.sum((arr >= lo) & (arr <= hi) & ~np.isnan(arr)))

        results[f"{col}__coverage"] = 100.0 * n_non_nan   / max(1, n_total)
        results[f"{col}__nonzero"]  = 100.0 * n_non_zero  / max(1, n_total)
        results[f"{col}__boundary"] = 100.0 * n_in_bounds / max(1, n_total)

    print(f"  ✓ {ds_name:<15}  {n_total:>12,} frames sampled")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# HEATMAP PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def _plot_heatmap(matrix: np.ndarray,
                  row_labels: list[str],
                  col_labels: list[str],
                  title: str,
                  out_path: Path,
                  cmap: str = "RdYlGn") -> None:
    """
    Plot a heatmap where:
      - rows = datasets
      - cols = signals
      - cell value = percentage (0–100)
      - cell text shows the percentage value
    Standard matplotlib style: box with grid lines from cell borders.
    """
    n_rows, n_cols = matrix.shape

    fig_w = max(10.0, n_cols * 1.6)
    fig_h = max(5.0,  n_rows * 0.7)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    norm = mcolors.Normalize(vmin=0, vmax=100)
    im   = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("% of frames", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Axis ticks
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(col_labels, fontsize=8.5, ha="center")
    ax.set_yticklabels(row_labels, fontsize=9)

    # Cell text — white on dark cells, black on light cells
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            # Determine text colour based on background brightness
            rgba      = plt.get_cmap(cmap)(norm(val))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            txt_color = "white" if luminance < 0.5 else "black"
            ax.text(j, i, f"{val:.1f}%",
                    ha="center", va="center",
                    fontsize=8, color=txt_color, fontweight="bold")

    # Grid lines between cells
    ax.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Signal", fontsize=9, labelpad=6)
    ax.set_ylabel("Dataset", fontsize=9, labelpad=6)

    fig.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")


def main():
    import os

    root          = _find_project_root()
    inertial_root = root / "inertial_data"
    fig_root      = (Path.home() / "Downloads" / "bachelor-arbeit"
                     / "inertial_exploratory_analysis" / "figures" / "coverage")
    fig_root.mkdir(parents=True, exist_ok=True)

    # Discover dataset folders
    found = {d.name: d for d in inertial_root.iterdir()
             if d.is_dir() and not d.name.startswith(".")}

    # Respect DATASET_ORDER; include any extras at the end
    ordered_names = [n for n in DATASET_ORDER if n in found]
    extras        = sorted(n for n in found if n not in DATASET_ORDER)
    dataset_names = ordered_names + extras

    if not dataset_names:
        print(f"[ERROR] No dataset folders found in {inertial_root}")
        sys.exit(1)

    n_workers = min(len(dataset_names), os.cpu_count() or 4)

    print(f"\n{'='*60}")
    print(f"  coverage_heatmaps.py — 3 heatmaps × {len(dataset_names)} datasets × {len(SIGNALS)} signals")
    print(f"  Input   → {inertial_root}")
    print(f"  Output  → {fig_root}")
    print(f"  Workers → {n_workers}")
    print(f"  Started → {_now()}")
    print(f"{'='*60}\n")

    # Parallel per-dataset coverage computation
    args = [(found[n], n) for n in dataset_names]
    results_by_ds = {}

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_compute_dataset_coverage, a): a[1] for a in args}
        for fut in as_completed(futures):
            ds_name = futures[fut]
            try:
                res = fut.result()
                if res is not None:
                    results_by_ds[ds_name] = res
            except Exception as e:
                print(f"  [ERROR] {ds_name}: {e}")

    # Build matrices — rows = datasets in order, cols = signals
    present_names = [n for n in dataset_names if n in results_by_ds]
    n_rows = len(present_names)
    n_cols = len(SIGNALS)

    mat_coverage = np.full((n_rows, n_cols), np.nan)
    mat_nonzero  = np.full((n_rows, n_cols), np.nan)
    mat_boundary = np.full((n_rows, n_cols), np.nan)

    for i, ds_name in enumerate(present_names):
        res = results_by_ds[ds_name]
        for j, sig in enumerate(SIGNALS):
            col = sig["col"]
            mat_coverage[i, j] = res.get(f"{col}__coverage", np.nan)
            mat_nonzero[i, j]  = res.get(f"{col}__nonzero",  np.nan)
            mat_boundary[i, j] = res.get(f"{col}__boundary", np.nan)

    col_labels = [s["label"] for s in SIGNALS]
    row_labels = present_names

    print(f"\n{'='*60}\n  Saving figures\n{'='*60}")

    _plot_heatmap(
        mat_coverage, row_labels, col_labels,
        title="Signal Coverage — % Non-NaN Values",
        out_path=fig_root / "signal_coverage.png",
        cmap="RdYlGn",
    )

    _plot_heatmap(
        mat_nonzero, row_labels, col_labels,
        title="Non-Zero Coverage — % Non-Zero Values (of all frames)",
        out_path=fig_root / "non_zero.png",
        cmap="RdYlGn",
    )

    _plot_heatmap(
        mat_boundary, row_labels, col_labels,
        title="Boundary Coverage — % Values Within Physical Bounds",
        out_path=fig_root / "boundary.png",
        cmap="RdYlGn",
    )

    print(f"\n{'='*60}")
    print(f"  ✓ DONE  —  {len(present_names)}/{len(dataset_names)} datasets  ·  3 PNGs  ·  {_now()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()