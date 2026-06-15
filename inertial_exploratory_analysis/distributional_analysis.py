"""
distributional_analysis.py  (v2 — adapted to the 10 Hz common-grid pipeline)
────────────────────────────────────────────────────────────────────────────
Per-dataset signal histograms (thesis Section 4.5.1 / 8.1).

─── CHANGES vs v1 ──────────────────────────────────────────────────────────
  1. The 0.1 s temporal-downsampling windowing (_window_df) is REMOVED.
     It existed to eliminate frame-rate bias across 10–200 Hz datasets;
     the v3 standardization already resamples every dataset onto a common
     uniform 10 Hz grid, so the windowing degenerated to an expensive no-op
     (one frame per 0.1 s window, groupby over tens of millions of rows).
     Histograms are now per-frame on the 10 Hz grid — same information,
     none of the compute.
  2. Streaming mode is leaner: bounds-filtering happens per batch and only
     filtered float32 arrays are kept, instead of concatenating full frames.
  3. DATASET_COLORS completed with durlar (gold) and zod (red) — previously
     both silently fell back to the same blue.
────────────────────────────────────────────────────────────────────────────
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

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

from tum_colors import TUM_PRIMARY, TUM_SECONDARY, TUM_EXT

COLOR_MODE = "per_dataset"

SIGNAL_COLORS = {
    "speed":     TUM_SECONDARY['blue2'],
    "lon_accel": TUM_SECONDARY['blue2'],
    "lat_accel": TUM_SECONDARY['blue2'],
    "lon_jerk":  TUM_SECONDARY['blue2'],
    "lat_jerk":  TUM_SECONDARY['blue2'],
    "yaw_rate":  TUM_SECONDARY['blue2'],
}

DATASET_COLORS = {
    "4seasons":  TUM_SECONDARY['blue2'],
    "a2d2":      TUM_SECONDARY['dark_gray'],
    "hdd":       TUM_SECONDARY['gray'],
    "kitscenes": TUM_SECONDARY['light_gray'],
    "boreas_rt": TUM_EXT['maroon'],
    "durlar":    TUM_EXT['gold'],
    "ford":      TUM_EXT['orange'],
    "kitti":     TUM_EXT['yellow'],
    "kitti360":  TUM_EXT['olive'],
    "nuscene":   TUM_EXT['dark_green'],
    "nvidia":    TUM_EXT['teal'],
    "oxford":    TUM_EXT['navy'],
    "waymo":     TUM_EXT['purple'],
    "zod":       TUM_EXT['red'],
}
DATASET_COLOR_FALLBACK = TUM_SECONDARY['blue2']

MEAN_LINE_COLOR = TUM_PRIMARY['black']


def _get_bar_color(dataset_name: str, signal_key: str) -> str:
    if COLOR_MODE == "per_dataset":
        return DATASET_COLORS.get(dataset_name, DATASET_COLOR_FALLBACK)
    else:
        return SIGNAL_COLORS.get(signal_key, DATASET_COLOR_FALLBACK)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

SIGNALS = [
    {
        "key":      "speed",
        "label":    "Speed",
        "unit":     "km/h",
        "clip_lo":  0.0,
        "clip_hi":  200.0,
        "xlabel":   "Speed (km/h)",
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
        "unit":     "°/s",
        "clip_lo":  -90.0,
        "clip_hi":   90.0,
        "xlabel":   "Yaw Rate (°/s)",
    },
]

N_SIGNALS = len(SIGNALS)
N_BINS    = 125
FIG_W     = 10.0
FIG_H     = 6.0

X_PAD_FRAC      = 0.10
YLIN_CAP_FACTOR = 1.20
GRID_HZ         = 10        # sampling rate of v3 standardized clips

BATCH_SIZE = 5_000          # clip files concatenated per streaming batch


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING + SIGNAL EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

SIGNAL_COLS = ["speed_kmh", "a_long", "a_lat", "jerk_long", "jerk_lat", "yaw_rate"]

# (signal_key, parquet_col, lo, hi, rad→deg conversion)
EXTRACT_SPEC = [
    ("speed",     "speed_kmh",  0.0,   200.0, False),
    ("lon_accel", "a_long",   -15.0,    15.0, False),
    ("lat_accel", "a_lat",    -15.0,    15.0, False),
    ("lon_jerk",  "jerk_long", -30.0,   30.0, False),
    ("lat_jerk",  "jerk_lat",  -30.0,   30.0, False),
    ("yaw_rate",  "yaw_rate",  -90.0,   90.0, True),   # bounds in °/s
]


def _extract_signal_values(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Bounds-filter each signal (visualization-only masking) and return
    compact float32 arrays. yaw_rate is converted rad/s → °/s first."""
    out = {}
    for key, col, lo, hi, to_deg in EXTRACT_SPEC:
        a = df[col].to_numpy(dtype=float)
        if to_deg:
            a = a * (180.0 / np.pi)
        a = a[(a >= lo) & (a <= hi) & ~np.isnan(a)]
        out[key] = a.astype(np.float32)
    return out


def load_signal_values(dataset_dir: Path,
                       dataset_name: str) -> tuple[dict[str, np.ndarray] | None, int]:
    """
    Read all clips in batches, extract + bounds-filter signals per batch,
    keep only filtered float32 arrays (memory-lean for NVIDIA's 60M rows).
    Returns ({signal_key: values}, n_total_frames).
    """
    import pyarrow.parquet as pq

    parquet_files = sorted(dataset_dir.glob("clip_*.parquet"))
    if not parquet_files:
        print(f"  [WARN] No clip_*.parquet files found in {dataset_dir}")
        return None, 0

    try:
        first_schema = set(pq.read_schema(parquet_files[0]).names)
        cols_to_read = [c for c in SIGNAL_COLS if c in first_schema]
    except Exception:
        cols_to_read = list(SIGNAL_COLS)

    missing_warned = False

    def _read_file(fp):
        nonlocal missing_warned
        try:
            df = pd.read_parquet(fp, columns=cols_to_read)
        except Exception as e:
            print(f"  [WARN] Could not read {fp.name}: {e}")
            return None
        missing = set(SIGNAL_COLS) - set(df.columns)
        if missing and not missing_warned:
            print(f"  [WARN] {dataset_name}: columns {missing} not found — filled with NaN")
            missing_warned = True
        for col in SIGNAL_COLS:
            if col not in df.columns:
                df[col] = np.nan
        return df[SIGNAL_COLS]

    parts = {key: [] for key, *_ in EXTRACT_SPEC}
    n_raw = 0

    n_batches = (len(parquet_files) + BATCH_SIZE - 1) // BATCH_SIZE
    if n_batches > 1:
        print(f"  [INFO] {dataset_name}: {len(parquet_files):,} clips — "
              f"streaming in {n_batches} batches")

    for batch_start in range(0, len(parquet_files), BATCH_SIZE):
        batch = parquet_files[batch_start: batch_start + BATCH_SIZE]
        chunks = [r for fp in batch if (r := _read_file(fp)) is not None]
        if not chunks:
            continue
        batch_df = pd.concat(chunks, ignore_index=True)
        n_raw += len(batch_df)
        batch_vals = _extract_signal_values(batch_df)
        for key, arr in batch_vals.items():
            parts[key].append(arr)
        del batch_df, chunks, batch_vals

    if n_raw < 10:
        print(f"  [WARN] {dataset_name}: only {n_raw} rows — skipping")
        return None, 0

    values = {key: (np.concatenate(p) if p else np.array([], dtype=np.float32))
              for key, p in parts.items()}
    return values, n_raw


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
    if len(density) < 2:
        return None
    sorted_d = np.sort(density)[::-1]
    tallest  = sorted_d[0]
    second   = sorted_d[1]
    if second < 1e-9:
        return None
    if tallest > 2.0 * second:
        return second * YLIN_CAP_FACTOR
    return None


def _draw_cap_markers(ax, density: np.ndarray, centers: np.ndarray,
                      cap: float, color: str) -> None:
    clipped = centers[density > cap]
    for x in clipped:
        ax.annotate(
            "▼",
            xy=(x, cap),
            xytext=(x, cap),
            ha="center", va="bottom",
            fontsize=6, color="black", alpha=1,
        )


def plot_signal(dataset_name: str,
                sig: dict,
                values: np.ndarray,
                out_path: Path,
                total_frames: int,
                bar_color: str) -> None:
    lo  = sig["clip_lo"]
    hi  = sig["clip_hi"]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    if len(values) < 5:
        ax.text(0.5, 0.5, "Insufficient data",
                ha="center", va="center",
                transform=ax.transAxes, color="gray")
    else:
        values = values.astype(np.float64)
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

        ax.bar(centers, density, width=width, color=bar_color, zorder=2)

        cap = _ylin_cap(density)
        if cap is not None:
            ax.set_ylim(0, cap)
            _draw_cap_markers(ax, density, centers, cap, bar_color)

        if lo < 0 and xlim_lo < 0 < xlim_hi:
            ax.axvline(0, color=MEAN_LINE_COLOR, lw=0.5, ls="-", alpha=1, zorder=1)
        ax.axvline(mean_v, color=MEAN_LINE_COLOR, lw=1, ls="--", alpha=1, zorder=4)

        stats_text = (
            f"μ = {mean_v:+.3f}   σ = {std_v:.3f}\n"
            f"p05 = {p05_v:+.3f}   p95 = {p95_v:+.3f}\n"
            f"n = {len(values):,} valid frames  |  {total_frames:,} total frames"
        )
        ax.text(
            0.97, 0.97, stats_text,
            ha="right",
            va="top",
            transform=ax.transAxes,
            color=TUM_PRIMARY["black"],
            fontsize=9,
            bbox=dict(
                facecolor="white",
                edgecolor="#cccccc",
                alpha=0.90
            ),
            zorder=5
        )

        ax.set_xlim(xlim_lo, xlim_hi)

    ax.set_title(f"{sig['label']} Distribution — {dataset_name.upper()} ")
    ax.set_xlabel(sig["xlabel"])
    ax.set_ylabel("Fraction of frames")
    ax.grid(True)
    ax.set_axisbelow(True)

    fig.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL WORKER
# ─────────────────────────────────────────────────────────────────────────────

def _process_dataset(args: tuple) -> dict | None:
    ds_dir, ds_name, fig_root, position = args

    sig_data, n_raw = load_signal_values(ds_dir, ds_name)
    if sig_data is None:
        return None

    results = []
    for sig in SIGNALS:
        vals      = sig_data.get(sig["key"], np.array([]))
        out_path  = fig_root / f"{ds_name}_{sig['key']}.png"
        bar_color = _get_bar_color(ds_name, sig["key"])
        plot_signal(ds_name, sig, vals, out_path, n_raw, bar_color)
        results.append({
            "key":    sig["key"],
            "unit":   sig["unit"],
            "n":      len(vals),
            "mean":   float(np.mean(vals)) if len(vals) > 0 else float("nan"),
            "fname":  out_path.name,
        })

    return {
        "dataset":  ds_name,
        "n_frames": n_raw,
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
                     / "inertial_exploratory_analysis" / "figures" / "distribution")
    fig_root.mkdir(parents=True, exist_ok=True)

    datasets = sorted(
        d for d in inertial_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    if not datasets:
        print(f"[ERROR] No dataset folders found in {inertial_root}")
        sys.exit(1)

    n_workers   = min(len(datasets), os.cpu_count() or 4)
    total_plots = len(datasets) * N_SIGNALS

    print(f"\n{'='*60}")
    print(f"  distributional_analysis.py  (v2 — {GRID_HZ} Hz common grid)")
    print(f"  {N_SIGNALS} signals, per-frame histograms")
    print(f"  Found {len(datasets)} dataset(s) → {total_plots} PNGs")
    print(f"  Workers: {n_workers}  (CPU cores: {os.cpu_count()})")
    print(f"  Input   → {inertial_root}")
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
            print(f"   Frames: {_format_count(result['n_frames'])} on the {GRID_HZ} Hz grid")
            for r in result["signals"]:
                mean_s = f"{r['mean']:+.3f}" if not np.isnan(r["mean"]) else "N/A"
                print(f"   ✓ {r['fname']:<45}  "
                      f"{r['n']:>9,} frames  "
                      f"mean={mean_s} {r['unit']}")
            print()
            summary_rows.append({
                "dataset":  result["dataset"],
                "n_frames": result["n_frames"],
                **{r["key"]: r["n"] for r in result["signals"]},
            })

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