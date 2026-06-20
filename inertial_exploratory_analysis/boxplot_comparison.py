"""
cross_dataset_boxplot.py  (v2)
========================
Section 4.5.3 — Cross-Dataset Variability Boxplots

v2 CHANGES:
  1. Violin KDE is trimmed to where density actually exists (was drawn over
     the full physical-bound grid, producing a zero-density hairline spanning
     the whole axis for every dataset).
  2. Per-figure y-limit from a robust statistic (max of per-dataset p99,
     never below the tallest whisker). Datasets whose tail exceeds the limit
     get an explicit "▲ max ≈ X" annotation — tails are disclosed, not hidden.
  3. DATASET_COLORS completed (durlar, zod were falling back to the same blue).
  4. Optional LOG_SCALE toggle for a supplementary view.

Reads pre-computed per-window std data from std_data/<dataset>.parquet and
produces 6 publication-quality PNG figures (one per signal) comparing
within-window signal variability across all 10 qualified datasets.

Plot design per figure:
  - Half-violin (right side, KDE clipped at physical bounds) in dataset color
  - Standard boxplot overlaid (IQR box, whiskers 1.5×IQR, no fliers)
  - Median line (dataset color, thicker)
  - Mean dot (white fill, black edge)
  - Datasets sorted by median ascending (order differs per figure)
  - X-tick labels: "KITTI\n(n=8,204 windows)"

Usage:
  python cross_dataset_boxplot.py

Toggle FILTER_STATIONARY = True to exclude near-stationary windows.
"""

import pathlib
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# PATHS  (edit if your folder layout differs)
# ---------------------------------------------------------------------------
STD_DATA_DIR = pathlib.Path(
    "/Users/edward/Downloads/dataset/std_data"
)
FIG_DIR = pathlib.Path(
    "/Users/edward/Downloads/bachelor-arbeit/"
    "inertial_exploratory_analysis/figures/boxplot"
)

# ---------------------------------------------------------------------------
# FILTER TOGGLE
# ---------------------------------------------------------------------------
FILTER_STATIONARY = False       # default: keep ALL windows (shows true bias)
MIN_STD_SPEED     = 0.05        # km/h — only applied when FILTER_STATIONARY=True

# ---------------------------------------------------------------------------
# DATASETS
# ---------------------------------------------------------------------------
DATASETS = [
    "4seasons", "a2d2", "boreas_rt",
    "hdd", "durlar",
    "ford", "kitti", "kitti360", 
    "nuscene", "nvidia", "oxford", "waymo", "zod", "kitscenes",
]

DISPLAY_NAMES = {
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

# ---------------------------------------------------------------------------
# COLORS  (must match all other thesis scripts exactly)
# ---------------------------------------------------------------------------

from tum_colors import TUM_PRIMARY, TUM_SECONDARY, TUM_ACCENT, TUM_EXT
DATASET_COLORS = {
    "4seasons":   TUM_SECONDARY['blue2'],
    "a2d2":      TUM_SECONDARY['dark_gray'],
    "hdd":       TUM_SECONDARY['gray'],
    "kitscenes": TUM_SECONDARY['light_gray'],
    "boreas_rt": TUM_EXT['maroon'],
    "durlar":    TUM_SECONDARY['blue1'],
    "ford":      TUM_EXT['orange'],
    "kitti":     TUM_EXT['yellow'],
    "kitti360":  TUM_EXT['olive'],
    "nuscene":   TUM_EXT['dark_green'],
    "nvidia":    TUM_EXT['teal'],
    "oxford":    TUM_EXT['navy'],
    "waymo":     TUM_EXT['purple'],
    "zod":       TUM_EXT['red'],
}
DATASET_COLOR_FALLBACK = TUM_SECONDARY["blue2"]


def _color(ds: str) -> str:
    """Return a color for dataset ds, with fallback.
    Kept as small helper to mirror other scripts' API."""
    return DATASET_COLORS.get(ds, DATASET_COLOR_FALLBACK)


# ---------------------------------------------------------------------------
# SIGNALS
# ---------------------------------------------------------------------------
SIGNALS = [
    {
        "key":       "speed",
        "col":       "std_speed_kmh",
        "title":     "Standard Deviation of Speed Comparison",
        "ylabel":    "σ(speed) [km/h]",
        "clip_lo":   0.0,
        "clip_hi":   200.0,
        "yaw_conv":  False,
    },
    {
        "key":       "acc_mag",
        "col":       "std_acc_mag",
        "title":     "Standard Deviation of Acceleration Magnitude Comparison",
        "ylabel":    "σ(acc_mag) [m/s²]",
        "clip_lo":   0.0,
        "clip_hi":   20.0,
        "yaw_conv":  False,
    },
    {
        "key":       "yaw_rate",
        "col":       "std_yaw_rate",
        "title":     "Standard Deviation of Yaw Rate Comparison",
        "ylabel":    "σ(yaw_rate) [°/s]",
        "clip_lo":   0.0,
        "clip_hi":   90.0,
        "yaw_conv":  True,   # stored as rad/s → convert to °/s
    },
]

# ---------------------------------------------------------------------------
# VISUAL PARAMETERS  (adjust after seeing first output)
# ---------------------------------------------------------------------------
FIGSIZE        = (16, 7)
DPI            = 300
VIOLIN_WIDTH   = 0.35    # max half-width of violin in x-axis units
BOX_WIDTH      = 0.12    # full width of IQR box
WHISKER_WIDTH  = 0.08    # end cap width of whiskers
MEAN_DOT_SIZE  = 7       # marker size for mean dot
VIOLIN_ALPHA   = 0.40    # transparency of violin fill
BOX_LINEWIDTH  = 1.6
MEDIAN_LW      = 2.2
WHISKER_LW     = 1.4
KDE_POINTS     = 400     # resolution of KDE curve

# Y-axis limit policy (v2):
#   y_top = HEADROOM × max over datasets of percentile(vals, Y_TOP_PCTL),
#   raised to the tallest whisker if needed. Data above y_top is annotated.
Y_TOP_PCTL     = 99.0
CONSENSUS_PCTL = 80.0    # y_top from the 80th pctl of per-dataset p99s, so a
                         # single pathological dataset cannot dictate the axis
HEADROOM       = 1.08
DENSITY_FLOOR  = 0.005   # violin drawn only where density > floor × max
LOG_SCALE      = False   # True → symlog supplementary view (suffix _log)

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_all(std_data_dir: pathlib.Path) -> dict[str, pd.DataFrame]:
    """Load all dataset parquets.  Returns {dataset_name: DataFrame}."""
    data = {}
    for ds in DATASETS:
        path = std_data_dir / f"{ds}.parquet"
        if not path.exists():
            print(f"  [WARN] {ds}: file not found — skipping ({path})")
            continue
        df = pd.read_parquet(path)
        if FILTER_STATIONARY and "std_speed_kmh" in df.columns:
            before = len(df)
            df = df[df["std_speed_kmh"] >= MIN_STD_SPEED].copy()
            removed = before - len(df)
            if removed:
                print(f"  [FILTER] {ds}: removed {removed:,} near-stationary windows")
        data[ds] = df
    return data


# ---------------------------------------------------------------------------
# HALF-VIOLIN DRAWING
# ---------------------------------------------------------------------------

def _draw_half_violin(ax, x_center: float, values: np.ndarray,
                      clip_lo: float, clip_hi: float,
                      color: str) -> None:
    """Draw a right-side-only half violin at x_center."""
    # Clip values to physical bounds before KDE
    vals = values[(values >= clip_lo) & (values <= clip_hi)]
    if len(vals) < 10:
        return

    # v2: KDE grid spans the DATA range (not the full physical-bound range) —
    # evaluating up to clip_hi drew a zero-density hairline across the axis.
    lo = max(clip_lo, float(vals.min()))
    hi = min(clip_hi, float(np.percentile(vals, 99.5)))
    if hi <= lo:
        hi = min(clip_hi, float(vals.max()))
    if hi <= lo:
        return
    y_grid = np.linspace(lo, hi, KDE_POINTS)
    try:
        kde = gaussian_kde(vals, bw_method="scott")
        density = kde(y_grid)
    except Exception:
        return

    # Normalise to VIOLIN_WIDTH
    max_density = density.max()
    if max_density == 0:
        return

    # v2: draw only where density is meaningfully non-zero
    support = np.where(density > DENSITY_FLOOR * max_density)[0]
    if len(support) == 0:
        return
    sl = slice(support.min(), support.max() + 1)
    y_grid  = y_grid[sl]
    density = density[sl]
    x_offsets = (density / max_density) * VIOLIN_WIDTH

    # Right side: x_center + offset
    x_right = x_center + x_offsets
    x_base  = np.full_like(y_grid, x_center)

    # Fill polygon
    ax.fill_betweenx(y_grid, x_base, x_right,
                     color=color, alpha=VIOLIN_ALPHA, linewidth=0)
    # Outline
    ax.plot(x_right, y_grid, color=color, linewidth=0.8, alpha=0.7)


# ---------------------------------------------------------------------------
# BOXPLOT DRAWING
# ---------------------------------------------------------------------------

def _draw_box(ax, x_center: float, values: np.ndarray, color: str,
              clip_lo: float = 0.0, clip_hi: float = np.inf) -> None:
    """Draw IQR box + whiskers + median line + mean dot at x_center.
    Values are clipped to [clip_lo, clip_hi] before all statistics,
    matching the physical bounds applied to the violin."""
    vals = values[np.isfinite(values)]
    vals = vals[(vals >= clip_lo) & (vals <= clip_hi)]
    if len(vals) < 4:
        return

    q1, median, q3 = np.percentile(vals, [25, 50, 75])
    iqr = q3 - q1
    mean = vals.mean()

    # Whisker ends (1.5 × IQR, capped at data range)
    lo_fence = q1 - 1.5 * iqr
    hi_fence = q3 + 1.5 * iqr
    lo_whisker = vals[vals >= lo_fence].min() if np.any(vals >= lo_fence) else q1
    hi_whisker = vals[vals <= hi_fence].max() if np.any(vals <= hi_fence) else q3

    half_box  = BOX_WIDTH / 2
    half_whis = WHISKER_WIDTH / 2

    # IQR box
    rect = mpatches.FancyBboxPatch(
        (x_center - half_box, q1),
        BOX_WIDTH, iqr,
        boxstyle="square,pad=0",
        linewidth=BOX_LINEWIDTH,
        edgecolor=color,
        facecolor="white",
        zorder=3,
    )
    ax.add_patch(rect)

    # Median line (inside box)
    ax.plot([x_center - half_box, x_center + half_box],
            [median, median],
            color=color, linewidth=MEDIAN_LW, zorder=4)

    # Lower whisker
    ax.plot([x_center, x_center], [lo_whisker, q1],
            color=color, linewidth=WHISKER_LW, zorder=3)
    ax.plot([x_center - half_whis, x_center + half_whis],
            [lo_whisker, lo_whisker],
            color=color, linewidth=WHISKER_LW, zorder=3)

    # Upper whisker
    ax.plot([x_center, x_center], [q3, hi_whisker],
            color=color, linewidth=WHISKER_LW, zorder=3)
    ax.plot([x_center - half_whis, x_center + half_whis],
            [hi_whisker, hi_whisker],
            color=color, linewidth=WHISKER_LW, zorder=3)

    # Mean dot (white fill, black edge — on top of everything)
    ax.plot(x_center, mean,
            marker="o", markersize=MEAN_DOT_SIZE,
            markerfacecolor="white", markeredgecolor="black",
            markeredgewidth=1.3, zorder=5)


# ---------------------------------------------------------------------------
# ONE FIGURE PER SIGNAL
# ---------------------------------------------------------------------------

def plot_signal(sig: dict, data: dict[str, pd.DataFrame],
                fig_dir: pathlib.Path) -> None:
    col      = sig["col"]
    clip_lo  = sig["clip_lo"]
    clip_hi  = sig["clip_hi"]
    yaw_conv = sig["yaw_conv"]

    # ── 1. Collect per-dataset arrays, convert yaw if needed ────────────────
    arrays: dict[str, np.ndarray] = {}
    for ds, df in data.items():
        if col not in df.columns:
            print(f"  [WARN] {ds}: column '{col}' missing — skipping dataset")
            continue
        vals = df[col].dropna().to_numpy(dtype=float)
        if yaw_conv:
            vals = vals * (180.0 / math.pi)
        # Remove non-finite values
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            print(f"  [WARN] {ds}: no valid values for '{col}' — skipping")
            continue
        arrays[ds] = vals

    if not arrays:
        print(f"  [SKIP] {sig['key']}: no data in any dataset")
        return

    # ── 2. Sort datasets by median ascending ────────────────────────────────
    order = sorted(arrays.keys(), key=lambda ds: float(np.median(arrays[ds])))

    # ── 3. Set up figure ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.set_title(sig["title"], fontsize=13, pad=10)
    ax.set_ylabel(sig["ylabel"], fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)

    # ── 4. Draw each dataset ─────────────────────────────────────────────────
    x_positions = []
    x_labels    = []

    for i, ds in enumerate(order):
        vals   = arrays[ds]
        color  = _color(ds)
        x_pos  = i + 1   # 1-indexed so there's margin on both sides

        _draw_half_violin(ax, x_pos, vals, clip_lo, clip_hi, color)
        _draw_box(ax, x_pos, vals, color, clip_lo, clip_hi)

        n_windows = len(vals)
        label = f"{DISPLAY_NAMES.get(ds, ds)}\n(n={n_windows:,})"
        x_positions.append(x_pos)
        x_labels.append(label)

    # ── 5. Axes formatting ───────────────────────────────────────────────────
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_xlim(0.3, len(order) + VIOLIN_WIDTH + 0.3)

    # ── v2: robust y-limit + disclosure of clipped tails ─────────────────────
    def _upper_whisker(vals):
        q1, q3 = np.percentile(vals, [25, 75])
        fence  = q3 + 1.5 * (q3 - q1)
        inside = vals[vals <= fence]
        return float(inside.max()) if len(inside) else float(q3)

    # Consensus limit: a single pathological dataset (e.g. corrupted tails)
    # is clipped + annotated instead of dictating the axis for all others.
    p99_list  = [float(np.percentile(arrays[ds], Y_TOP_PCTL)) for ds in order]
    whisk_list = [_upper_whisker(arrays[ds]) for ds in order]
    med_list  = [float(np.median(arrays[ds])) for ds in order]
    y_top = HEADROOM * max(
        float(np.percentile(p99_list,  CONSENSUS_PCTL)),
        float(np.percentile(whisk_list, CONSENSUS_PCTL)),
        1.5 * max(med_list),          # every median always clearly visible
    )
    if not np.isfinite(y_top) or y_top <= 0:
        y_top = clip_hi

    if LOG_SCALE:
        ax.set_yscale("symlog", linthresh=max(y_top * 1e-3, 1e-3))
        ax.set_ylim(0, clip_hi)
    else:
        ax.set_ylim(0, y_top)
        # Annotate datasets whose tail exceeds the visible range
        def _fmt(v):
            return f"{v:.0f}" if v >= 100 else (f"{v:.1f}" if v >= 10 else f"{v:.2f}")
        for x_pos, ds in zip(x_positions, order):
            vals  = arrays[ds]
            v_max = float(vals.max())
            if v_max > y_top:
                v_med = float(np.median(vals))
                v_q3  = float(np.percentile(vals, 75))
                if v_q3 > y_top:   # even the box is clipped — disclose more
                    note = f"▲ med {_fmt(v_med)} · q3 {_fmt(v_q3)} · max {_fmt(v_max)}"
                else:
                    note = f"▲ max ≈ {_fmt(v_max)}"
                ax.text(x_pos, y_top * 0.985, note, ha="center", va="top",
                        fontsize=7.5, color="black")

    # ── 6. Legend ────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor="grey", alpha=VIOLIN_ALPHA, label="Density (half-violin)"),
        mpatches.Patch(facecolor="white", edgecolor="grey",
                       linewidth=BOX_LINEWIDTH, label="IQR box (25th–75th %ile)"),
        plt.Line2D([0], [0], color="grey", linewidth=MEDIAN_LW, label="Median"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="white", markeredgecolor="black",
                   markeredgewidth=1.3, markersize=MEAN_DOT_SIZE, label="Mean"),
        plt.Line2D([0], [0], color="grey", linewidth=WHISKER_LW,
                   label="Whiskers (1.5 × IQR)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left",
              frameon=True, facecolor="white", fontsize=9)

    fig.tight_layout()

    # ── 7. Save ──────────────────────────────────────────────────────────────
    fig_dir.mkdir(parents=True, exist_ok=True)
    suffix   = "_log" if LOG_SCALE else ""
    out_path = fig_dir / f"boxplot_{sig['key']}{suffix}.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ saved: {out_path.name}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("cross_dataset_boxplot.py — Section 4.5.3")
    print("=" * 60)
    print(f"  std_data dir : {STD_DATA_DIR}")
    print(f"  output dir   : {FIG_DIR}")
    print(f"  FILTER_STATIONARY = {FILTER_STATIONARY}")
    print()

    print("Loading std_data parquets ...")
    data = load_all(STD_DATA_DIR)
    if not data:
        print("[ERROR] No datasets loaded. Check STD_DATA_DIR path.")
        return
    print(f"  Loaded {len(data)} datasets: {', '.join(data.keys())}")
    print()

    print("Generating figures ...")
    for sig in SIGNALS:
        print(f"  [{sig['key']}]")
        plot_signal(sig, data, FIG_DIR)

    print()
    print("✓ DONE — all figures saved to:", FIG_DIR)


if __name__ == "__main__":
    main()