"""
Script: heatmap_analysis.py
============================
Bachelor Thesis: "Identifying Dynamic Biases in Large-Scale Public Datasets
via Distributional Analysis"

Section 4.5.4 — Summary Heatmap

Reads per-window std data from:
    std_data/<dataset>.parquet

For each dataset × signal:
    - Computes median(std(window)) → 10×6 raw matrix (real physical units)
    - Applies per-column min-max normalization → drives colormap
    - Sorts rows by rank-sum (most dynamic on top)

Cell annotation:
    - Line 1 (large): normalized value (0.00–1.00)
    - Line 2 (small): real median std value with unit

Single shared colorbar on the right (YlOrRd).

Output:
    figures/heatmap/summary_heatmap.png

Usage:
    python heatmap_analysis.py
    python heatmap_analysis.py /path/to/inertial_exploratory_analysis
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = [
    "4seasons", "a2d2", "hdd", "boreas_rt", "durlar", "ford", "kitti", "kitti360",
    "nuscene", "nvidia", "oxford", "waymo", "zod", "kitscenes",
]

# (parquet_column, display_label, unit_string)
SIGNALS = [
    ("std_speed_kmh",  "σ(speed)",    "km/h"),
    ("std_acc_mag",    "σ(acc_mag)",  "m/s²"),
    ("std_yaw_rate",   "σ(yaw_rate)", "°/s"),
]

# Physical validity bounds — applied at visualization time only (NaN masking).
# std_yaw_rate bound is in °/s (applied AFTER rad→deg conversion).
# All other bounds in native signal units. Never modifies source data.
SIGNAL_BOUNDS = {
    "std_speed_kmh":  (0.0, 200.0),
    "std_acc_mag":    (0.0,  20.0),
}
# Yaw rate bound expressed in °/s — applied after rad→deg conversion
YAW_RATE_BOUND_DEG = 90.0

# Yaw rate: stored in rad/s in std_data → convert to °/s for display
YAW_RATE_COL = "std_yaw_rate"
RAD_TO_DEG   = 180.0 / np.pi

# Stationary window filter (toggle — keep False to preserve true bias)
FILTER_STATIONARY = False
MIN_STD_SPEED_KMH = 0.5   # only used if FILTER_STATIONARY = True

WINDOW_S = 20
STEP_S   = 10
DPI      = 300

COLORMAP = "Blues"

# ─────────────────────────────────────────────────────────────────────────────
# PATH RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────

def _find_roots() -> tuple[Path, Path]:
    """
    Returns (project_root, data_root).
    project_root → bachelor-arbeit/inertial_exploratory_analysis/
    data_root    → dataset/std_data/
    """
    if len(sys.argv) > 1:
        project_root = Path(sys.argv[1]).expanduser().resolve()
    else:
        script_dir = Path(__file__).resolve().parent
        candidates = [
            script_dir,
            script_dir.parent,
            Path.home() / "Downloads" / "bachelor-arbeit" / "inertial_exploratory_analysis",
        ]
        project_root = next(
            (p for p in candidates if (p / "figures").exists() or p.name == "inertial_exploratory_analysis"),
            script_dir,
        )

    # std_data lives in the dataset folder, sibling to bachelor-arbeit
    data_candidates = [
        project_root.parent.parent / "dataset" / "std_data",
        project_root.parent / "dataset" / "std_data",
        Path.home() / "Downloads" / "dataset" / "std_data",
    ]
    data_root = next((p for p in data_candidates if p.exists()), data_candidates[-1])

    return project_root, data_root


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_std_data(data_root: Path) -> dict[str, pd.DataFrame]:
    """Load std_data parquet for each dataset. Returns dict dataset → DataFrame."""
    data = {}
    for ds in DATASETS:
        path = data_root / f"{ds}.parquet"
        if not path.exists():
            print(f"  [WARN] Missing: {path} — skipping {ds}")
            continue
        df = pd.read_parquet(path)

        # Apply physical validity bounds (NaN masking — source data never modified)
        for col, (lo, hi) in SIGNAL_BOUNDS.items():
            if col in df.columns:
                mask = (df[col] < lo) | (df[col] > hi)
                n_masked = mask.sum()
                if n_masked > 0:
                    df.loc[mask, col] = np.nan
                    print(f"  [BOUNDS] {ds}.{col}: masked {n_masked:,} out-of-range values")

        # Convert yaw_rate rad/s → °/s, then apply °/s bound
        if YAW_RATE_COL in df.columns:
            df[YAW_RATE_COL] = df[YAW_RATE_COL] * RAD_TO_DEG
            mask = df[YAW_RATE_COL] > YAW_RATE_BOUND_DEG
            n_masked = mask.sum()
            if n_masked > 0:
                df.loc[mask, YAW_RATE_COL] = np.nan
                print(f"  [BOUNDS] {ds}.{YAW_RATE_COL}: masked {n_masked:,} values > {YAW_RATE_BOUND_DEG}°/s")

        # Optional stationary filter
        if FILTER_STATIONARY and "std_speed_kmh" in df.columns:
            before = len(df)
            df = df[df["std_speed_kmh"] >= MIN_STD_SPEED_KMH]
            after = len(df)
            if before > after:
                print(f"  [FILTER] {ds}: removed {before - after:,} stationary windows "
                      f"({100*(before-after)/before:.1f}%)")

        data[ds] = df
        print(f"  [OK] {ds}: {len(df):,} windows loaded")

    return data


# ─────────────────────────────────────────────────────────────────────────────
# BUILD RAW MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def build_raw_matrix(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build 10×6 raw matrix of median(std(window)) per dataset per signal.
    Rows = datasets, columns = signal display labels.
    Values in real physical units.
    """
    rows = {}
    for ds in DATASETS:
        if ds not in data:
            rows[ds] = {label: np.nan for _, label, _ in SIGNALS}
            continue
        df = data[ds]
        row = {}
        for col, label, _ in SIGNALS:
            if col in df.columns:
                row[label] = df[col].median()
            else:
                print(f"  [WARN] {ds}: column '{col}' not found — NaN")
                row[label] = np.nan
        rows[ds] = row

    matrix = pd.DataFrame(rows).T   # datasets × signals
    matrix.index.name = "dataset"
    return matrix


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZE & SORT
# ─────────────────────────────────────────────────────────────────────────────

def normalize_per_column(matrix: pd.DataFrame) -> pd.DataFrame:
    """Per-column min-max normalization → all values in [0, 1]."""
    norm = matrix.copy()
    for col in norm.columns:
        col_min = norm[col].min()
        col_max = norm[col].max()
        denom = col_max - col_min
        if denom == 0:
            norm[col] = 0.0
        else:
            norm[col] = (norm[col] - col_min) / denom
    return norm


def sort_by_rank_sum(matrix: pd.DataFrame, norm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sort rows by sum of per-column ranks (descending = most dynamic on top).
    Applies same order to both raw matrix and normalized matrix.
    """
    ranks = norm.rank(axis=0, ascending=True, na_option="bottom")
    rank_sums = ranks.sum(axis=1)
    order = rank_sums.sort_values(ascending=False).index
    return matrix.loc[order], norm.loc[order]


# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────

# Dataset display names (title-cased, consistent with rest of pipeline)
DATASET_DISPLAY = {
    "4seasons":   "4 Seasons",
    "a2d2":      "A2D2",
    "hdd":       "HDD",
    "boreas_rt": "Boreas",
    "durlar":    "DurLAR",
    "ford":      "Ford",
    "kitti":     "KITTI",
    "kitti360":  "KITTI-360",
    "nuscene":   "nuScenes",
    "nvidia":    "NVIDIA",
    "oxford":    "Oxford",
    "waymo":     "Waymo",
    "zod":       "ZOD",
    "kitscenes": "KITScenes",
}

# Unit lookup for annotation
SIGNAL_UNITS = {label: unit for _, label, unit in SIGNALS}


def plot_heatmap(matrix: pd.DataFrame, norm: pd.DataFrame, out_path: Path) -> None:
    """
    Plot the 10×6 summary heatmap.
    Color driven by normalized values, cell annotation shows both.
    """
    n_rows, n_cols = matrix.shape

    fig, ax = plt.subplots(figsize=(13, 7))

    # ── draw colored cells ──────────────────────────────────────────────────
    cmap = plt.get_cmap(COLORMAP)
    norm_vals = norm.values.astype(float)   # shape (n_rows, n_cols)

    for r in range(n_rows):
        for c in range(n_cols):
            nv = norm_vals[r, c]
            color = cmap(nv) if not np.isnan(nv) else (0.85, 0.85, 0.85, 1.0)
            rect = plt.Rectangle([c, r], 1, 1, color=color)
            ax.add_patch(rect)

            # ── cell text ───────────────────────────────────────────────────
            raw_val  = matrix.values[r, c]
            col_name = matrix.columns[c]
            unit     = SIGNAL_UNITS.get(col_name, "")

            # text color: white on dark cells, black on light cells
            text_color = "white" if (not np.isnan(nv) and nv > 0.5) else "black"

            if np.isnan(raw_val):
                ax.text(c + 0.5, r + 0.5, "N/A",
                        ha="center", va="center",
                        fontsize=9, color=text_color)
            else:
                # Line 1 (top, bold): normalized value — drives color
                ax.text(c + 0.5, r + 0.65, f"{nv:.2f}",
                        ha="center", va="center",
                        fontsize=10, fontweight="bold", color=text_color)
                # Line 2 (bottom, small): real value + unit — reference
                ax.text(c + 0.5, r + 0.30, f"{raw_val:.2f} {unit}",
                        ha="center", va="center",
                        fontsize=7.5, color=text_color)

    # ── axes setup ──────────────────────────────────────────────────────────
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)

    # x-axis: signal labels
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(matrix.columns, fontsize=10)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # y-axis: dataset display names (bottom row = index 0 in imshow coords,
    #         but we drew row 0 at y=0 and the most dynamic is first in matrix)
    # Flip so most dynamic (row 0) appears at the top visually
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(
        [DATASET_DISPLAY.get(ds, ds) for ds in matrix.index],
        fontsize=10
    )
    ax.invert_yaxis()   # row 0 → top (most dynamic)

    # white grid lines between cells
    for x in range(n_cols + 1):
        ax.axvline(x, color="white", linewidth=1.5)
    for y in range(n_rows + 1):
        ax.axhline(y, color="white", linewidth=1.5)

    ax.set_xlabel("Kinematic Signal", fontsize=11, labelpad=8)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Dataset", fontsize=11, labelpad=8)

    # remove default ticks
    ax.tick_params(left=False, top=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── colorbar ────────────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Normalized median σ (per signal)", fontsize=9)
    cbar.set_ticks([0.0, 0.25, 0.50, 0.75, 1.0])
    cbar.set_ticklabels(["0.00\n(min)", "0.25", "0.50", "0.75", "1.00\n(max)"])

    # ── title ────────────────────────────────────────────────────────────────
    fig.suptitle(
        f"Per-Dataset Dynamic Variability — Median Within-Window σ "
        f"({WINDOW_S}s windows, {STEP_S}s step)",
        fontsize=11, y=1.02
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    project_root, data_root = _find_roots()
    out_path = project_root / "figures" / "heatmap" / "summary_heatmap.png"

    print(f"Project root : {project_root}")
    print(f"std_data root: {data_root}")
    print(f"Output       : {out_path}")
    print()

    print("Loading std_data...")
    data = load_std_data(data_root)
    print()

    print("Building raw matrix (median per dataset per signal)...")
    matrix = build_raw_matrix(data)
    print(matrix.to_string())
    print()

    print("Normalizing per column (min-max)...")
    norm = normalize_per_column(matrix)

    print("Sorting rows by rank-sum (most dynamic on top)...")
    matrix, norm = sort_by_rank_sum(matrix, norm)
    print("Row order:", list(matrix.index))
    print()

    print("Plotting heatmap...")
    plot_heatmap(matrix, norm, out_path)
    print(f"✓ Heatmap saved → {out_path}")


if __name__ == "__main__":
    main()