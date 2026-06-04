"""
Script: distributional_analysis.py
===================================
Bachelor Thesis: "Identifying Dynamic Biases in Large-Scale Public Datasets
via Distributional Analysis"

Reads standardized parquet clips from:
    inertial_data/<dataset>/clip_XXXX.parquet

Plots log-scale histograms for 6 signals:
    1.  speed                → 'speed'      (m/s, used directly)
    2.  longitudinal accel   → 'a_long'     (m/s²)
    3.  lateral accel        → 'a_lat'      (m/s²)
    4.  longitudinal jerk    → 'jerk_long'  (m/s³)
    5.  lateral jerk         → 'jerk_lat'   (m/s³)
    6.  yaw rate             → 'yaw_rate'   (rad/s)

Design choices:
  - Log y-scale: reveals tail structure even in highly leptokurtic distributions
  - Data-driven x-axis: limits set to [p01 - pad, p99 + pad] per plot
  - White background, standard matplotlib style
  - One PNG per dataset per signal

Output:
    figures/distributions/<dataset>_<signal>.png

Usage:
    python distributional_analysis.py
    python distributional_analysis.py /path/to/inertial_exploratory_analysis
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Suppress pandas/numexpr/bottleneck version warnings in worker subprocesses.
# These fire on every worker import before our warnings.filterwarnings() runs.
import os as _os
_os.environ.setdefault("PYTHONWARNINGS", "ignore")

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
        "Could not find 'inertial_data/' in any of:\n  " + checked +
        "\n\nPass the path as argument:\n    python distributional_analysis.py /path/to/folder"
    )


# ─────────────────────────────────────────────────────────────────────────────
# COLOR CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

from tum_colors import TUM_PRIMARY, TUM_SECONDARY, TUM_ACCENT, TUM_EXT

# ── Choose coloring mode ──────────────────────────────────────────────────────
# "per_signal"  → every dataset uses the same color per signal key
# "per_dataset" → every signal in a dataset uses the same color per dataset name
COLOR_MODE = "per_dataset"

# ── Colors per signal (used when COLOR_MODE = "per_signal") ──────────────────
SIGNAL_COLORS = {
    "speed":     TUM_SECONDARY['blue2'],
    "lon_accel": TUM_SECONDARY['blue2'],
    "lat_accel": TUM_SECONDARY['blue2'],
    "lon_jerk":  TUM_SECONDARY['blue2'],
    "lat_jerk":  TUM_SECONDARY['blue2'],
    "yaw_rate":  TUM_SECONDARY['blue2'],
}

# ── Colors per dataset (used when COLOR_MODE = "per_dataset") ─────────────────
# Add or remove datasets freely. If a dataset name is not listed here,
# it falls back to DATASET_COLOR_FALLBACK.
DATASET_COLORS = {
    "4seasons":  TUM_EXT['maroon'],
    "a2d2":      TUM_EXT['orange'],
    "boreas_rt": TUM_EXT['yellow'],
    "durlar":    TUM_EXT['dark_green'],
    "ford":      TUM_EXT['teal'],
    "hdd":       TUM_EXT['navy'],
    "kitti":     TUM_EXT['purple'],
    "kitti360":  TUM_EXT['gold'],
    "nuscene":   TUM_EXT['olive'],
    "nvidia":    TUM_EXT['maroon'],
    "oxford":    TUM_EXT['gold'],
    "waymo":     TUM_PRIMARY['blue2'],
    "zod":       TUM_PRIMARY['gray'],
}
DATASET_COLOR_FALLBACK = TUM_SECONDARY['blue2']   # used if dataset not in dict above

MEAN_LINE_COLOR = TUM_PRIMARY['black']


def _get_bar_color(dataset_name: str, signal_key: str) -> str:
    """Return the bar color based on the active COLOR_MODE."""
    if COLOR_MODE == "per_dataset":
        return DATASET_COLORS.get(dataset_name, DATASET_COLOR_FALLBACK)
    else:
        return SIGNAL_COLORS.get(signal_key, DATASET_COLOR_FALLBACK)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
#
# clip_lo / clip_hi: hard physical outlier limits applied BEFORE plotting
# x-axis limits are then set to data-driven p01/p99 ± padding

SIGNALS = [
    {
        "key":      "speed",
        "label":    "Speed",
        "unit":     "m/s",
        "clip_lo":  0.0,
        "clip_hi":  50.0,
        "xlabel":   "Speed (m/s)",
    },
    {
        "key":      "lon_accel",
        "label":    "Longitudinal Acceleration",
        "unit":     "m/s²",
        "clip_lo":  -15.0,
        "clip_hi":   15.0,
        "xlabel":   "Longitudinal Acceleration (m/s²)",
    },
    {
        "key":      "lat_accel",
        "label":    "Lateral Acceleration",
        "unit":     "m/s²",
        "clip_lo":  -15.0,
        "clip_hi":   15.0,
        "xlabel":   "Lateral Acceleration (m/s²)",
    },
    {
        "key":      "lon_jerk",
        "label":    "Longitudinal Jerk",
        "unit":     "m/s³",
        "clip_lo":  -30.0,
        "clip_hi":   30.0,
        "xlabel":   "Longitudinal Jerk (m/s³)",
    },
    {
        "key":      "lat_jerk",
        "label":    "Lateral Jerk",
        "unit":     "m/s³",
        "clip_lo":  -30.0,
        "clip_hi":   30.0,
        "xlabel":   "Lateral Jerk (m/s³)",
    },
    {
        "key":      "yaw_rate",
        "label":    "Yaw Rate",
        "unit":     "rad/s",
        "clip_lo":  -1.5,
        "clip_hi":   1.5,
        "xlabel":   "Yaw Rate (rad/s)",
    },
]

N_SIGNALS = len(SIGNALS)
N_BINS    = 125
FIG_W     = 10.0
FIG_H     = 6.0

# Padding multiplier added to p01/p99 to set x-axis limits
X_PAD_FRAC = 0.10

# Y-axis cap: the linear axis is capped at this multiple of the 2nd tallest bin.
# This prevents a dominant zero-spike from flattening all other bars.
# The spike is still shown but clipped with a visual break marker (triangle).
YLIN_CAP_FACTOR = 1.20   # cap = 2nd tallest bin × 1.20

# Temporal downsampling window.
# Raw frames are averaged into non-overlapping time windows of this length.
# Each window produces exactly one data point per signal.
# This removes frame-rate bias (200 Hz vs 10 Hz datasets become comparable)
# and naturally smooths out transient zero-spikes from stationary frames.
# Change this value to resample at a different resolution, e.g. 0.5 or 1.0.
WINDOW_S = 0.1   # seconds per window


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

# Full schema reference (from Signal Coverage Matrix):
#   TIME:        timestamp_ns, timestamp_s, t_rel_ns, t_rel_s, dt_ns, dt_s
#   VELOCITY:    speed, speed_kmh, vx, vy, vz
#   ACCEL:       a_long, a_lat, deceleration, acc_mag
#   JERK:        jerk_long, jerk_lat, jerk_mag, abs_jerk_long, abs_jerk_mag
#   ORIENTATION: roll, pitch, yaw, yaw_unwrapped, yaw_rate
#   HEADING:     heading_change, abs_heading_change
#   CURVATURE:   curvature, abs_curvature, curvature_valid
REQUIRED_COLS = {"timestamp_s", "speed", "a_long", "a_lat", "jerk_long", "jerk_lat", "yaw_rate"}


def load_clips(dataset_dir: Path, dataset_name: str) -> pd.DataFrame | None:
    """
    Concatenate all clip_*.parquet files for one dataset.
    Returns None if no usable data is found.
    """
    import pyarrow.parquet as pq

    parquet_files = sorted(dataset_dir.glob("clip_*.parquet"))
    if not parquet_files:
        print(f"  [WARN] No clip_*.parquet files found in {dataset_dir}")
        return None

    chunks = []
    missing_warned = False

    for fp in parquet_files:
        try:
            schema_cols  = set(pq.read_schema(fp).names)
            cols_to_read = list(REQUIRED_COLS & schema_cols)
            df = pd.read_parquet(fp, columns=cols_to_read)
        except Exception as e:
            print(f"  [WARN] Could not read {fp.name}: {e}")
            continue

        missing = REQUIRED_COLS - set(df.columns)
        if missing and not missing_warned:
            print(f"  [WARN] {dataset_name}: columns {missing} not found — filled with NaN")
            missing_warned = True
        for col in REQUIRED_COLS:
            if col not in df.columns:
                df[col] = np.nan

        chunks.append(df[list(REQUIRED_COLS)])

    if not chunks:
        return None

    combined = pd.concat(chunks, ignore_index=True)
    if len(combined) < 10:
        print(f"  [WARN] {dataset_name}: only {len(combined)} rows total — skipping")
        return None

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_signals(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Temporally downsample all 6 signals into non-overlapping WINDOW_S windows.

    For each clip, frames are grouped by:
        window_id = floor((timestamp_s - clip_start_s) / WINDOW_S)
    and the mean of each signal is taken within each window. Each window
    produces exactly one data point regardless of the dataset's frame rate.

    This makes all datasets comparable (200 Hz and 10 Hz produce the same
    number of data points per second of recording) and naturally smooths
    transient zeros from brief stationary moments.

    Hard clip limits are applied AFTER windowing to remove physical outliers.
    """
    SIGNAL_COLS = ["speed", "a_long", "a_lat", "jerk_long", "jerk_lat", "yaw_rate"]
    CLIP_LIMITS = {
        "speed":     (0,    50),
        "a_long":  (-15,   15),
        "a_lat":   (-15,   15),
        "jerk_long":(-30,  30),
        "jerk_lat": (-30,  30),
        "yaw_rate": (-1.5, 1.5),
    }

    ts = df["timestamp_s"].values.astype(float)

    # Build window IDs relative to the start of the whole dataset.
    # Clips are independent recordings, so we reset relative time per clip
    # by using timestamp continuity: a gap > 1 s signals a new clip boundary.
    dt = np.diff(ts, prepend=ts[0]) # type: ignore
    clip_boundary = np.where(np.abs(dt) > 1.0)[0]
    clip_ids = np.zeros(len(ts), dtype=np.int64)
    for b in clip_boundary:
        clip_ids[b:] += 1

    # Relative time within each clip
    t_rel = np.empty_like(ts)
    for cid in np.unique(clip_ids):
        mask = clip_ids == cid
        t_rel[mask] = ts[mask] - ts[mask][0]

    window_ids = (t_rel // WINDOW_S).astype(np.int64)

    # Global unique window key: combine clip_id and window_id
    global_key = clip_ids * 10_000_000 + window_ids

    # Group by window key and compute mean for each signal
    df_work = pd.DataFrame({col: df[col].values.astype(float)
                            for col in SIGNAL_COLS})
    df_work["_key"] = global_key
    windowed = df_work.groupby("_key")[SIGNAL_COLS].mean()

    def clip_valid(arr, lo, hi):
        a = np.asarray(arr, dtype=float)
        return a[(a >= lo) & (a <= hi) & ~np.isnan(a)]

    return {
        "speed":     clip_valid(windowed["speed"].values,      *CLIP_LIMITS["speed"]),
        "lon_accel": clip_valid(windowed["a_long"].values,     *CLIP_LIMITS["a_long"]),
        "lat_accel": clip_valid(windowed["a_lat"].values,      *CLIP_LIMITS["a_lat"]),
        "lon_jerk":  clip_valid(windowed["jerk_long"].values,  *CLIP_LIMITS["jerk_long"]),
        "lat_jerk":  clip_valid(windowed["jerk_lat"].values,   *CLIP_LIMITS["jerk_lat"]),
        "yaw_rate":  clip_valid(windowed["yaw_rate"].values,   *CLIP_LIMITS["yaw_rate"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")


def _format_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1e6:.1f}M"
    elif n >= 1_000:
        return f"{n/1e3:.0f}k"
    return str(n)


def _data_xlim(values: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    """
    Compute data-driven x-axis limits: [p01 - pad, p99 + pad],
    clamped to [lo, hi]. Falls back to [lo, hi] for degenerate data.
    """
    if len(values) < 10:
        return lo, hi
    p01 = float(np.percentile(values,  1))
    p99 = float(np.percentile(values, 99))
    rng = p99 - p01
    if rng < 1e-9:
        center = float(np.median(values))
        return max(lo, center - 1.0), min(hi, center + 1.0)
    pad = rng * X_PAD_FRAC
    return max(lo, p01 - pad), min(hi, p99 + pad)


def _ylin_cap(density: np.ndarray) -> float | None:
    """
    Return a y-axis cap for the linear axis based on the 2nd tallest bin.
    Returns None if no capping is needed (distribution is not spike-dominated).

    A spike is considered dominant when the tallest bin is more than 3×
    the 2nd tallest bin — i.e. it would visually crush everything else.
    """
    if len(density) < 2:
        return None
    sorted_d = np.sort(density)[::-1]   # descending
    tallest  = sorted_d[0]
    second   = sorted_d[1]
    if second < 1e-9:
        return None
    if tallest > 3.0 * second:
        return second * YLIN_CAP_FACTOR
    return None


def _draw_cap_markers(ax, density: np.ndarray, centers: np.ndarray,
                      cap: float, color: str) -> None:
    """
    Draw small downward-pointing triangles on top of every bar that was
    clipped by the y-axis cap, to signal that the bar continues upward.
    """
    clipped = centers[density > cap]
    for x in clipped:
        ax.annotate(
            "▼",
            xy=(x, cap),
            xytext=(x, cap * 1.01),
            ha="center", va="bottom",
            fontsize=7, color="black", alpha=0.8,
        )


def plot_signal(dataset_name: str,
                sig: dict,
                values: np.ndarray,
                out_path: Path,
                total_frames: int,
                bar_color: str) -> None:
    """
    Single linear histogram per dataset per signal.
    Y-axis is capped at 2nd tallest bin × YLIN_CAP_FACTOR when spike-dominated;
    clipped bars get a ▼ marker. Bar color is passed in from _process_dataset
    based on the active COLOR_MODE.
    """
    lo  = sig["clip_lo"]
    hi  = sig["clip_hi"]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    if len(values) < 5:
        ax.text(0.5, 0.5, "Insufficient data",
                ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="gray")
    else:
        xlim_lo, xlim_hi = _data_xlim(values, lo, hi)
        bins    = np.linspace(xlim_lo, xlim_hi, N_BINS + 1)
        counts, edges = np.histogram(values, bins=bins)
        density = counts / max(1, counts.sum())
        centers = 0.5 * (edges[:-1] + edges[1:])
        width   = edges[1] - edges[0]

        mean_v = float(np.mean(values))
        std_v  = float(np.std(values))
        p05_v  = float(np.percentile(values,  5))
        p95_v  = float(np.percentile(values, 95))

        # ── Bars ──────────────────────────────────────────────────────────
        ax.bar(centers, density, width=width,
               color=bar_color, alpha=0.75, linewidth=0, zorder=2)

        cap = _ylin_cap(density)
        if cap is not None:
            ax.set_ylim(0, cap)
            _draw_cap_markers(ax, density, centers, cap, bar_color)

        # ── Reference lines ───────────────────────────────────────────────
        if lo < 0 and xlim_lo < 0 < xlim_hi:
            ax.axvline(0, color=MEAN_LINE_COLOR, lw=0.5, ls="-", alpha=1, zorder=1)
        ax.axvline(mean_v, color=MEAN_LINE_COLOR, lw=1, ls="--", alpha=1, zorder=4)

        # ── Stats box ─────────────────────────────────────────────────────
        stats_text = (
            f"μ = {mean_v:+.3f}   σ = {std_v:.3f}\n"
            f"p05 = {p05_v:+.3f}   p95 = {p95_v:+.3f}\n"
            f"n = {len(values):,} / {total_frames:,} frames"
        )
        ax.text(0.97, 0.97, stats_text,
                ha="right", va="top",
                transform=ax.transAxes,
                fontsize=8.5, color=TUM_PRIMARY["black"],
                linespacing=1.6,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#cccccc", alpha=0.90, lw=0.6),
                zorder=5)

        ax.set_xlim(xlim_lo, xlim_hi)

    ax.set_title(
        f"{sig['label']}  ·  {dataset_name.upper()}  "
        f"({int(WINDOW_S * 1000)} ms windows)",
        fontsize=11, fontweight="bold", pad=8,
    )
    ax.set_xlabel(sig["xlabel"], fontsize=9, labelpad=4)
    ax.set_ylabel("Fraction of frames", fontsize=9, labelpad=4)
    ax.tick_params(axis="both", labelsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL WORKER
# ─────────────────────────────────────────────────────────────────────────────

def _process_dataset(args: tuple) -> dict | None:
    """
    Worker function executed in a separate process for one dataset.
    Loads all clips, computes windowed signals, plots all 6 PNGs.
    Returns a summary dict, or None if the dataset has no usable data.
    """
    ds_dir, ds_name, fig_root, position = args

    df = load_clips(ds_dir, ds_name)
    if df is None:
        return None

    n_frames = len(df)
    sig_data = compute_signals(df)

    results = []
    for sig in SIGNALS:
        vals      = sig_data.get(sig["key"], np.array([]))
        out_path  = fig_root / f"{ds_name}_{sig['key']}.png"
        bar_color = _get_bar_color(ds_name, sig["key"])
        plot_signal(ds_name, sig, vals, out_path, n_frames, bar_color)
        results.append({
            "key":    sig["key"],
            "unit":   sig["unit"],
            "n":      len(vals),
            "pct":    100.0 * len(vals) / max(1, n_frames),
            "mean":   float(np.mean(vals)) if len(vals) > 0 else float("nan"),
            "fname":  out_path.name,
        })

    return {
        "dataset":  ds_name,
        "n_frames": n_frames,
        "signals":  results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import concurrent.futures
    import os

    root          = _find_project_root()
    inertial_root = root / "inertial_data"
    fig_root      = (Path.home() / "Downloads" / "bachelor-arbeit"
                     / "inertial_exploratory_analysis" / "figures" / "temporal")
    fig_root.mkdir(parents=True, exist_ok=True)

    datasets = sorted(
        d for d in inertial_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    if not datasets:
        print(f"[ERROR] No dataset folders found in {inertial_root}")
        sys.exit(1)

    # One worker per dataset, capped at CPU count.
    # Datasets with many clips (NVIDIA 298k, 4seasons 31M) are I/O + CPU heavy,
    # so process-level parallelism avoids Python's GIL entirely.
    n_workers   = min(len(datasets), os.cpu_count() or 4)
    total_plots = len(datasets) * N_SIGNALS

    print(f"\n{'='*60}")
    print(f"  distributional_analysis.py — {N_SIGNALS} signals, {int(WINDOW_S*1000)} ms windows")
    print(f"  Found {len(datasets)} dataset(s) → {total_plots} PNGs")
    print(f"  Workers: {n_workers}  (CPU cores: {os.cpu_count()})")
    print(f"  Output  → {fig_root}")
    print(f"  Started → {_now()}")
    print(f"{'='*60}\n")

    args = [(ds_dir, ds_dir.name, fig_root, i) for i, ds_dir in enumerate(datasets)]

    summary_rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_dataset, a): a[1] for a in args}
        for fut in concurrent.futures.as_completed(futures):
            ds_name = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                print(f"  [ERROR] {ds_name}: {e}")
                continue

            if result is None:
                print(f"── {ds_name.upper()}: no usable data — skipped\n")
                continue

            print(f"── {ds_name.upper()} {'─' * max(1, 40 - len(ds_name))}  [{_now()}]")
            print(f"   Loaded {_format_count(result['n_frames'])} frames")
            for r in result["signals"]:
                mean_s = f"{r['mean']:+.3f}" if not np.isnan(r["mean"]) else "N/A"
                print(f"   ✓ {r['fname']:<45}  "
                      f"{r['n']:>9,} valid ({r['pct']:5.1f}%)  "
                      f"mean={mean_s} {r['unit']}")
            print()
            summary_rows.append({
                "dataset":  result["dataset"],
                "n_frames": result["n_frames"],
                **{r["key"]: r["n"] for r in result["signals"]},
            })

    # Summary table — sort by dataset name for consistent output
    summary_rows.sort(key=lambda r: r["dataset"])
    if summary_rows:
        col_w  = 12
        header = f"  {'Dataset':<15} {'Total':>10}" + "".join(
            f"  {s['key'][:col_w]:>{col_w}}" for s in SIGNALS
        )
        print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for r in summary_rows:
            row = f"  {r['dataset']:<15} {r['n_frames']:>10,}"
            row += "".join(f"  {r[s['key']]:>{col_w},}" for s in SIGNALS)
            print(row)

    print(f"\nAll figures saved to: {fig_root}\n")
    print(f"{'='*60}")
    print(f"  ✓ DONE  —  {len(summary_rows)}/{len(datasets)} datasets  "
          f"·  {len(summary_rows) * N_SIGNALS} PNGs saved  ·  {_now()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()