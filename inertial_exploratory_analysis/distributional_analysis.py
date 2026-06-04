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
# TUM COLOR PALETTE
# ─────────────────────────────────────────────────────────────────────────────

from tum_colors import TUM_PRIMARY, TUM_SECONDARY, TUM_ACCENT, TUM_EXT

# ── Signal colors ─────────────────────────────────────────────────────────────
#
# PHASE 1 — uniform (use now, while dataset count is uncertain)
#
SIGNAL_COLORS = {
    "speed":     TUM_SECONDARY['blue2'],
    "lon_accel": TUM_SECONDARY['blue2'],
    "lat_accel": TUM_SECONDARY['blue2'],
    "lon_jerk":  TUM_SECONDARY['blue2'],
    "lat_jerk":  TUM_SECONDARY['blue2'],
    "yaw_rate":  TUM_SECONDARY['blue2'],
}

# PHASE 2 — colorful (uncomment when final dataset set is fixed)
#
# SIGNAL_COLORS = {
#     "speed":     TUM_SECONDARY['blue2'],    
#     "lon_accel": TUM_ACCENT['green'],       
#     "lat_accel": TUM_SECONDARY['gray'],        
#     "lon_jerk":  TUM_ACCENT['blue1'],      
#     "lat_jerk":  TUM_SECONDARY['light_gray'],             
#     "yaw_rate":  TUM_ACCENT['orange'],            
# }

MEAN_LINE_COLOR = TUM_PRIMARY['black']  # dark gray for mean + zero lines


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
REQUIRED_COLS = {"speed", "a_long", "a_lat", "jerk_long", "jerk_lat", "yaw_rate"}


def load_clips(dataset_dir: Path, dataset_name: str) -> pd.DataFrame | None:
    """
    Concatenate all clip_*.parquet files for one dataset.
    Returns None if no usable data is found.
    """
    parquet_files = sorted(dataset_dir.glob("clip_*.parquet"))
    if not parquet_files:
        print(f"  [WARN] No clip_*.parquet files found in {dataset_dir}")
        return None

    chunks = []
    missing_warned = False

    for fp in parquet_files:
        try:
            df = pd.read_parquet(fp)
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
    Extract all 6 signals from pre-computed parquet columns.
    speed is in m/s already — used directly.
    All signals kept signed to preserve distribution shape.

    Column definitions (from pipeline spec):
      speed      = scalar speed                         [m/s]
      a_long     = d(speed)/dt_s                        [m/s²]  +accel / -brake
      a_lat      = speed × yaw_rate                     [m/s²]  +left / -right
      jerk_long  = d(a_long)/dt_s                       [m/s³]
      jerk_lat   = d(a_lat)/dt_s                        [m/s³]
      yaw_rate   = d(yaw_unwrapped)/dt_s                [rad/s] +left / -right

    Hard clip limits (physically motivated):
      speed:      0–50 m/s   (0–180 km/h)
      a_long:    ±15 m/s²    (~1.5g)
      a_lat:     ±15 m/s²    (~1.5g lateral)
      jerk_long: ±30 m/s³    (standard outlier threshold)
      jerk_lat:  ±30 m/s³
      yaw_rate:  ±1.5 rad/s  (~86°/s)
    """
    def clip_valid(arr, lo, hi):
        a = np.asarray(arr, dtype=float)
        return a[(a >= lo) & (a <= hi) & ~np.isnan(a)]

    return {
        "speed":     clip_valid(df["speed"].values,         0,    50),
        "lon_accel": clip_valid(df["a_long"].values,      -15,    15),
        "lat_accel": clip_valid(df["a_lat"].values,       -15,    15),
        "lon_jerk":  clip_valid(df["jerk_long"].values,   -30,    30),
        "lat_jerk":  clip_valid(df["jerk_lat"].values,    -30,    30),
        "yaw_rate":  clip_valid(df["yaw_rate"].values,    -1.5,  1.5),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

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
            fontsize=7, color=color, alpha=0.8,
        )


def plot_signal(dataset_name: str,
                sig: dict,
                values: np.ndarray,
                out_path: Path,
                total_frames: int) -> None:
    """
    Single figure:
      Linear bar histogram only.
      Y-axis is optionally capped at 2nd tallest bin × YLIN_CAP_FACTOR
      when the distribution is spike-dominated.
      Clipped bars get a ▼ marker to indicate truncation.
    """
    lo        = sig["clip_lo"]
    hi        = sig["clip_hi"]
    color_bar = SIGNAL_COLORS[sig["key"]]

    fig, ax_lin = plt.subplots(figsize=(FIG_W, FIG_H))

    if len(values) < 5:
        ax_lin.text(0.5, 0.5, "Insufficient data",
                    ha="center", va="center",
                    transform=ax_lin.transAxes, fontsize=11, color="gray")
    else:
        # ── Shared x limits ──────────────────────────────────────────────
        xlim_lo, xlim_hi = _data_xlim(values, lo, hi)
        bins    = np.linspace(xlim_lo, xlim_hi, N_BINS + 1)
        counts, edges = np.histogram(values, bins=bins)
        density = counts / max(1, counts.sum())
        centers = 0.5 * (edges[:-1] + edges[1:])
        width   = edges[1] - edges[0]

        # ── Stats ─────────────────────────────────────────────────────────
        mean_v = float(np.mean(values))
        std_v  = float(np.std(values))
        p05_v  = float(np.percentile(values,  5))
        p95_v  = float(np.percentile(values, 95))

        # ── Linear histogram bars ─────────────────────────────────────────
        ax_lin.bar(
            centers,
            density,
            width=width,
            color=color_bar,
            alpha=0.75,
            linewidth=0,
            label="Distribution",
            zorder=2
        )

        # ── Optional y-axis cap for spike-dominated distributions ─────────
        cap = _ylin_cap(density)
        if cap is not None:
            ax_lin.set_ylim(0, cap)
            _draw_cap_markers(ax_lin, density, centers, cap, color_bar)

        # ── Reference lines ───────────────────────────────────────────────
        if lo < 0 and xlim_lo < 0 < xlim_hi:
            ax_lin.axvline(
                0,
                color=MEAN_LINE_COLOR,
                lw=0.5,
                ls="-",
                alpha=1,
                zorder=1
            )

        ax_lin.axvline(
            mean_v,
            color=MEAN_LINE_COLOR,
            lw=1,
            ls="--",
            alpha=1,
            zorder=4,
            label="Mean"
        )

        # ── Stats box ─────────────────────────────────────────────────────
        stats_text = (
            f"μ = {mean_v:+.3f}   σ = {std_v:.3f}\n"
            f"p05 = {p05_v:+.3f}   p95 = {p95_v:+.3f}\n"
            f"n = {len(values):,} / {total_frames:,} frames"
        )

        ax_lin.text(
            0.97,
            0.97,
            stats_text,
            ha="right",
            va="top",
            transform=ax_lin.transAxes,
            fontsize=8.5,
            color=TUM_PRIMARY["black"],
            linespacing=1.6,
            bbox=dict(
                boxstyle="round,pad=0.3",
                fc="white",
                ec=TUM_PRIMARY["black"],
                alpha=0.90,
                lw=0.6
            ),
            zorder=5
        )

        ax_lin.set_xlim(xlim_lo, xlim_hi)

        # ── Legend ────────────────────────────────────────────────────────
        ax_lin.legend(
            fontsize=8,
            loc="upper left",
            framealpha=0.85,
            edgecolor=TUM_PRIMARY["black"]
        )

    # ── Axis labels & styling ─────────────────────────────────────────────
    ax_lin.set_title(
        f"{sig['label']}  ·  {dataset_name.upper()}",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )

    ax_lin.set_xlabel(sig["xlabel"], fontsize=9, labelpad=4)
    ax_lin.set_ylabel("Fraction of frames", fontsize=9, labelpad=4)

    ax_lin.tick_params(axis="both", labelsize=8.5)

    ax_lin.spines["top"].set_visible(False)
    ax_lin.spines["right"].set_visible(False)

    fig.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    root          = _find_project_root()
    inertial_root = root / "inertial_data"
    fig_root      = Path.home() / "Downloads" / "bachelor-arbeit" / "inertial_exploratory_analysis" / "figures" / "distributions"
    fig_root.mkdir(parents=True, exist_ok=True)

    datasets = sorted(
        d for d in inertial_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    if not datasets:
        print(f"[ERROR] No dataset folders found in {inertial_root}")
        sys.exit(1)

    total_plots = len(datasets) * N_SIGNALS
    print(f"\n{'='*60}")
    print(f"  distributional_analysis.py — 6 signals, linear histograms")
    print(f"  Found {len(datasets)} dataset(s) → {total_plots} PNGs")
    print(f"  Output → {fig_root}")
    print(f"{'='*60}\n")

    summary_rows = []

    for ds_dir in datasets:
        ds_name = ds_dir.name
        print(f"── {ds_name.upper()} {'─' * max(1, 40 - len(ds_name))}")

        df = load_clips(ds_dir, ds_name)
        if df is None:
            print(f"   [SKIP] No usable data.\n")
            continue

        n_frames = len(df)
        print(f"   Loaded {_format_count(n_frames)} frames")
        sig_data = compute_signals(df)

        for sig in SIGNALS:
            vals    = sig_data.get(sig["key"], np.array([]))
            n_valid = len(vals)
            pct     = 100.0 * n_valid / max(1, n_frames)
            mean_s  = f"{np.mean(vals):+.3f}" if n_valid > 0 else "N/A"

            out_path = fig_root / f"{ds_name}_{sig['key']}.png"
            plot_signal(ds_name, sig, vals, out_path, n_frames)

            print(f"   ✓ {out_path.name:<45}  "
                  f"{n_valid:>9,} valid ({pct:5.1f}%)  mean={mean_s} {sig['unit']}")

        print()
        summary_rows.append({
            "dataset":  ds_name,
            "n_frames": n_frames,
            **{s["key"]: len(sig_data.get(s["key"], [])) for s in SIGNALS},
        })

    # Summary table
    if summary_rows:
        col_w = 12
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


if __name__ == "__main__":
    main()