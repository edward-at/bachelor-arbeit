import sys
import warnings
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import os as _os
_os.environ.setdefault("PYTHONWARNINGS", "ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS  (edit here only)
# ─────────────────────────────────────────────────────────────────────────────

MIN_STD          = 0.05                    # discard near-stationary windows
TIER_PERCENTILES = [0.25, 0.50, 0.75]     # 3 representatives per tier
N_REPS           = len(TIER_PERCENTILES)

DPI              = 300
FIG_W            = 15.0                   # wide: 3 columns
FIG_H            = 4.5                    # short: 1 row

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

INERTIAL_ROOT = Path("/Users/edward/Downloads/dataset/inertial_data")
STD_ROOT      = Path("/Users/edward/Downloads/dataset/std_data")
FIG_ROOT      = Path("/Users/edward/Downloads/bachelor-arbeit/inertial_exploratory_analysis/figures/time_series")

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

SIGNALS = [
    {"col": "speed_kmh",  "std_col": "std_speed_kmh",   "label": "Speed",                    "unit": "km/h",  "clip_lo":   0.0, "clip_hi": 200.0},
    {"col": "a_long",     "std_col": "std_a_long",      "label": "Longitudinal Acceleration","unit": "m/s²",  "clip_lo": -15.0, "clip_hi":  15.0},
    {"col": "a_lat",      "std_col": "std_a_lat",       "label": "Lateral Acceleration",     "unit": "m/s²",  "clip_lo": -15.0, "clip_hi":  15.0},
    {"col": "jerk_long",  "std_col": "std_jerk_long",   "label": "Longitudinal Jerk",        "unit": "m/s³",  "clip_lo": -30.0, "clip_hi":  30.0},
    {"col": "jerk_lat",   "std_col": "std_jerk_lat",    "label": "Lateral Jerk",             "unit": "m/s³",  "clip_lo": -30.0, "clip_hi":  30.0},
    {"col": "yaw_rate",   "std_col": "std_yaw_rate",    "label": "Yaw Rate",                 "unit": "°/s",   "clip_lo": -90.0, "clip_hi":  90.0},
]

TIER_LABELS = ["Low Dynamic", "Medium Dynamic", "High Dynamic"]

# ─────────────────────────────────────────────────────────────────────────────
# COLORS
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from tum_colors import TUM_ACCENT

# 3 lines: 25%, 50%, 75% of tier — blue, green, orange
REP_COLORS = [
    TUM_ACCENT["blue"],
    TUM_ACCENT["green"],
    TUM_ACCENT["orange"],
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _clip_path(ds_name: str, clip_id: int) -> Path:
    return INERTIAL_ROOT / ds_name / f"clip_{clip_id:04d}.parquet"


def _read_window_data(ds_name: str, clip_id: int,
                      window_start_s: float, window_end_s: float) -> pd.DataFrame | None:
    """
    Recover exact raw frames for a window using the back-reference contract:
      open clip_XXXX.parquet, filter t_rel_s >= window_start_s and < window_end_s
    Returns df with t_win_s column added (time relative to window start).
    """
    fp = _clip_path(ds_name, clip_id)
    if not fp.exists():
        print(f"  [WARN] {fp.name} not found")
        return None
    try:
        import pyarrow.parquet as pq
        needed = ["t_rel_s"] + [s["col"] for s in SIGNALS]
        schema_cols = set(pq.read_schema(fp).names)
        cols = [c for c in needed if c in schema_cols]
        df = pd.read_parquet(fp, columns=cols)
        for c in needed:
            if c not in df.columns:
                df[c] = np.nan
    except Exception as e:
        print(f"  [WARN] Could not read {fp.name}: {e}")
        return None

    # short clip window: use full clip
    if window_start_s == df["t_rel_s"].min() and window_end_s == df["t_rel_s"].max():
        w_df = df.copy()
    else:
        mask = (df["t_rel_s"] >= window_start_s) & (df["t_rel_s"] < window_end_s)
        w_df = df[mask].copy()

    if len(w_df) < 2:
        return None

    w_df["t_win_s"] = w_df["t_rel_s"] - window_start_s
    return w_df


# ─────────────────────────────────────────────────────────────────────────────
# TIER SELECTION  (reads from std_data)
# ─────────────────────────────────────────────────────────────────────────────

def _select_representatives(std_df: pd.DataFrame, std_col: str) -> tuple[dict, int] | None:
    """
    Read pre-computed std scores from std_data parquet.
    Filter near-stationary, sort, divide into thirds, pick 3 representatives
    per tier at TIER_PERCENTILES [25%, 50%, 75%].

    Returns (tier_windows, n_eligible) where tier_windows is:
      {"low": [...], "medium": [...], "high": [...]}
    Each item: (std_val, clip_id, window_start_s, window_end_s)
    """
    # Drop rows where this signal's std is NaN
    valid = std_df[std_df[std_col].notna()].copy()

    # Filter near-stationary
    filtered = valid[valid[std_col] >= MIN_STD].copy()
    n_eligible = len(filtered)
    if n_eligible < 3:
        return None

    # Sort ascending by std
    filtered = filtered.sort_values(std_col).reset_index(drop=True)

    # Split into three equal thirds
    third = n_eligible // 3
    tiers_raw = [
        filtered.iloc[:third],
        filtered.iloc[third: 2 * third],
        filtered.iloc[2 * third:],
    ]

    result = {}
    for tier_name, tier_df in zip(["low", "medium", "high"], tiers_raw):
        n = len(tier_df)
        indices = [max(0, min(n - 1, int(round(p * (n - 1))))) for p in TIER_PERCENTILES]
        items = []
        for idx in indices:
            row = tier_df.iloc[idx]
            items.append((
                float(row[std_col]),
                int(row["clip_id"]),
                float(row["window_start_s"]),
                float(row["window_end_s"]),
            ))
        result[tier_name] = items

    return result, n_eligible


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def _plot_signal(dataset_name: str,
                 sig: dict,
                 tier_windows: dict,
                 out_path: Path) -> None:
    """
    1 row × 3 columns. Each column = one dynamic tier.
    Each subplot: N_REPS lines, each a different representative window.
    Signal values outside clip_lo/clip_hi are masked as NaN (visualization only).
    """
    col = sig["col"]
    lo  = sig["clip_lo"]
    hi  = sig["clip_hi"]

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H), sharey=False)

    for ax, tier_name, tier_label in zip(axes, ["low", "medium", "high"], TIER_LABELS):
        items = tier_windows[tier_name]
        stds  = []

        for i, (std_val, clip_id, win_start, win_end) in enumerate(items):
            color = REP_COLORS[i]

            w_df = _read_window_data(dataset_name, clip_id, win_start, win_end)
            if w_df is None or col not in w_df.columns:
                continue
            if w_df[col].isna().all():
                continue

            t = w_df["t_win_s"].values
            y = w_df[col].values.astype(float)

            # Convert yaw_rate from rad/s to °/s
            if col == "yaw_rate":
                y = y * (180.0 / np.pi)

            # Mask physically implausible values (visualization only)
            y = np.where((y >= lo) & (y <= hi), y, np.nan)

            pct_label = f"{int(round(TIER_PERCENTILES[i] * 100))}%"
            ax.plot(t, y, color=color, label=f"{pct_label} (σ={std_val:.3f})")
            stds.append(std_val)

        if stds:
            title = f"{tier_label}\nσ range: {min(stds):.3f} – {max(stds):.3f}"
        else:
            title = tier_label

        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"{sig['label']} ({sig['unit']})")
        ax.legend(frameon=True, facecolor="white")
        ax.grid(True)
        ax.set_axisbelow(True)

    fig.suptitle(f"{dataset_name.upper()} — {sig['label']}")
    fig.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# WORKER  (one per dataset)
# ─────────────────────────────────────────────────────────────────────────────

def _process_dataset(args: tuple) -> dict | None:
    ds_name, fig_root = args

    std_path = STD_ROOT / f"{ds_name}.parquet"
    if not std_path.exists():
        print(f"  [WARN] {ds_name}: std_data file not found at {std_path}")
        return None

    std_df = pd.read_parquet(std_path)

    # Plot one PNG per signal — each signal uses its own std for tier selection
    for sig in SIGNALS:
        selection = _select_representatives(std_df, sig["std_col"])
        if selection is None:
            print(f"  [WARN] {ds_name} / {sig['col']}: not enough eligible windows "
                  f"({sig['std_col']} >= {MIN_STD})")
            continue

        tier_windows, n_eligible = selection

        low_stds  = [item[0] for item in tier_windows["low"]]
        med_stds  = [item[0] for item in tier_windows["medium"]]
        high_stds = [item[0] for item in tier_windows["high"]]
        print(
            f"── {ds_name.upper():<12} {sig['col']:<12} [{_now()}]  "
            f"eligible: {n_eligible:,}  |  "
            f"low σ=[{min(low_stds):.3f}–{max(low_stds):.3f}]  "
            f"med σ=[{min(med_stds):.3f}–{max(med_stds):.3f}]  "
            f"high σ=[{min(high_stds):.3f}–{max(high_stds):.3f}]"
        )

        out_path = fig_root / f"{ds_name}_{sig['col']}.png"
        _plot_signal(ds_name, sig, tier_windows, out_path)

    return {"dataset": ds_name}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import os

    FIG_ROOT.mkdir(parents=True, exist_ok=True)

    # Discover datasets from std_data folder
    std_files = sorted(STD_ROOT.glob("*.parquet"))
    if not std_files:
        print(f"[ERROR] No parquet files found in {STD_ROOT}")
        print(f"        Run compute_std.py first.")
        sys.exit(1)

    ds_names   = [f.stem for f in std_files]
    n_workers  = min(len(ds_names), os.cpu_count() or 4)
    total_pngs = len(ds_names) * len(SIGNALS)
    t_start    = datetime.now()

    print(f"\n{'='*65}")
    print(f"  time_series_analysis.py")
    print(f"  Min std: {MIN_STD}   Reps per tier: {N_REPS} at {[int(p*100) for p in TIER_PERCENTILES]}%")
    print(f"  Datasets: {len(ds_names)}   Signals: {len(SIGNALS)}   PNGs: {total_pngs}")
    print(f"  Workers:  {n_workers}  (CPU cores: {os.cpu_count()})")
    print(f"  std_data → {STD_ROOT}")
    print(f"  Output   → {FIG_ROOT}")
    print(f"  Started  → {_now()}")
    print(f"{'='*65}\n")

    args = [(ds_name, FIG_ROOT) for ds_name in ds_names]

    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_dataset, a): a[0] for a in args}
        for fut in as_completed(futures):
            ds_name = futures[fut]
            try:
                result = fut.result()
                if result is not None:
                    done += 1
            except Exception as e:
                print(f"  [ERROR] {ds_name}: {e}")

    elapsed = (datetime.now() - t_start).total_seconds()
    print(f"\n{'='*65}")
    print(f"  ✓ DONE  —  {done}/{len(ds_names)} datasets  "
          f"·  {done * len(SIGNALS)} PNGs saved  "
          f"·  {elapsed:.1f}s  ·  {_now()}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()