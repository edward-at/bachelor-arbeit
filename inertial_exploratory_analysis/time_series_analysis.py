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

WINDOW_S         = 20.0                        # seconds per window
STEP_S           = WINDOW_S * 0.5              # sliding step (50 % overlap)
MIN_STD          = 0.05                        # m/s — discard near-stationary windows
TIER_PERCENTILES = [0.10, 0.325, 0.55, 0.775, 0.90]   # 5 representatives per tier
N_REPS           = len(TIER_PERCENTILES)

DPI              = 200
FIG_W            = 15.0                        # wide: 3 columns
FIG_H            = 4.5                         # short: 1 row

LARGE_CLIP_THRESHOLD = 10_000                  # stream clip-by-clip above this

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

SIGNALS = [
    {"col": "speed",      "label": "Speed",                   "unit": "m/s"},
    {"col": "a_long",     "label": "Longitudinal Acceleration","unit": "m/s²"},
    {"col": "a_lat",      "label": "Lateral Acceleration",     "unit": "m/s²"},
    {"col": "jerk_long",  "label": "Longitudinal Jerk",        "unit": "m/s³"},
    {"col": "jerk_lat",   "label": "Lateral Jerk",             "unit": "m/s³"},
    {"col": "yaw_rate",   "label": "Yaw Rate",                 "unit": "rad/s"},
]

TIER_LABELS = ["Low Dynamic", "Medium Dynamic", "High Dynamic"]

# ─────────────────────────────────────────────────────────────────────────────
# COLORS
# ─────────────────────────────────────────────────────────────────────────────

import os as _os2
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from tum_colors import TUM_EXT

# 5 lines: 10 % → 90 % percentile of tier, mapped to these colors in order
REP_COLORS = [
    TUM_EXT["red"],
    TUM_EXT["orange"],
    TUM_EXT["yellow"],
    TUM_EXT["dark_green"],
    TUM_EXT["navy"],
]

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
    ]
    for root in candidates:
        if (root / "inertial_data").is_dir():
            return root
    checked = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not find 'inertial_data/' in any of:\n  " + checked
    )


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ─────────────────────────────────────────────────────────────────────────────
# WINDOW EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_windows_from_df(df: pd.DataFrame) -> list[dict]:
    """
    Slide a WINDOW_S window with STEP_S step across a single clip DataFrame.

    If t_max < WINDOW_S: 1 clip = 1 window, full clip used as-is (no sliding).
    If t_max >= WINDOW_S: normal sliding window with STEP_S step.
    """
    if "t_rel_s" not in df.columns:
        return []

    t_max = df["t_rel_s"].max()

    # Short clip (under WINDOW_S): 1 clip = 1 window, use full clip as-is
    if t_max < WINDOW_S:
        if len(df) < 10:
            return []
        w_df = df.copy()
        w_df["t_win_s"] = w_df["t_rel_s"]
        return [{"t_start": 0.0, "data": w_df}]

    # Normal case: sliding window with STEP_S
    windows = []
    t_start = 0.0
    while t_start + WINDOW_S <= t_max + 1e-6:
        mask = (df["t_rel_s"] >= t_start) & (df["t_rel_s"] < t_start + WINDOW_S)
        w_df = df[mask].copy()
        if len(w_df) >= 10:
            w_df["t_win_s"] = w_df["t_rel_s"] - t_start
            windows.append({"t_start": t_start, "data": w_df})
        t_start += STEP_S

    return windows


def _score_window(w: dict) -> float:
    """Speed std as dynamic score."""
    if "speed" not in w["data"].columns:
        return 0.0
    return float(w["data"]["speed"].std())


# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADING  (two modes: normal vs large)
# ─────────────────────────────────────────────────────────────────────────────

NEEDED_COLS = ["t_rel_s"] + [s["col"] for s in SIGNALS]


def _read_clip(fp: Path) -> pd.DataFrame | None:
    try:
        import pyarrow.parquet as pq
        schema_cols = set(pq.read_schema(fp).names)
        cols = [c for c in NEEDED_COLS if c in schema_cols]
        df = pd.read_parquet(fp, columns=cols)
        # Fill missing signal cols with NaN
        for c in NEEDED_COLS:
            if c not in df.columns:
                df[c] = np.nan
        return df
    except Exception as e:
        print(f"  [WARN] Could not read {fp.name}: {e}")
        return None


def _collect_windows_normal(clip_files: list[Path]) -> list[dict]:
    """Load all clips, extract all windows — for small datasets."""
    all_windows = []
    for fp in clip_files:
        df = _read_clip(fp)
        if df is None:
            continue
        all_windows.extend(_extract_windows_from_df(df))
    return all_windows


def _collect_windows_large(clip_files: list[Path]) -> list[dict]:
    """
    Stream clip by clip.
    Pass 1: score every window, store only (score, clip_fp, t_start).
    Pass 2: load only the 3×N_REPS selected clips fully for data retrieval.
    """
    print(f"    [stream] scoring {len(clip_files):,} clips one by one ...")

    scored = []   # list of (std, clip_fp, t_start)
    for fp in clip_files:
        df = _read_clip(fp)
        if df is None:
            continue
        for w in _extract_windows_from_df(df):
            s = _score_window(w)
            scored.append((s, fp, w["t_start"]))

    return scored   # caller handles tier selection differently


# ─────────────────────────────────────────────────────────────────────────────
# TIER SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def _select_representatives(windows_scored: list[tuple]) -> dict | None:
    """
    windows_scored: list of (std, *payload)
      - normal mode:  (std, window_dict)
      - large mode:   (std, clip_fp, t_start)

    Returns dict with keys 'low', 'medium', 'high', each a list of N_REPS items.
    Returns None if not enough windows after filtering.
    """
    # Filter near-stationary
    filtered = [w for w in windows_scored if w[0] >= MIN_STD]
    n_eligible = len(filtered)
    if n_eligible < 3:
        return None

    # Sort ascending by std
    filtered.sort(key=lambda x: x[0])

    # Split into three equal thirds
    third = n_eligible // 3
    tiers_raw = [
        filtered[:third],
        filtered[third: 2 * third],
        filtered[2 * third:],
    ]

    result = {}
    for tier_name, tier_items in zip(["low", "medium", "high"], tiers_raw):
        n = len(tier_items)
        indices = [int(round(p * (n - 1))) for p in TIER_PERCENTILES]
        # Clamp to valid range
        indices = [max(0, min(n - 1, i)) for i in indices]
        result[tier_name] = [tier_items[i] for i in indices]

    return result, n_eligible


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVE WINDOW DATA  (large mode needs re-loading)
# ─────────────────────────────────────────────────────────────────────────────

def _get_window_data_large(clip_fp: Path, t_start: float) -> pd.DataFrame | None:
    """Re-load a single clip and extract the specific window by t_start."""
    df = _read_clip(clip_fp)
    if df is None:
        return None
    mask = (df["t_rel_s"] >= t_start) & (df["t_rel_s"] < t_start + WINDOW_S)
    w_df = df[mask].copy()
    if len(w_df) < 10:
        return None
    w_df["t_win_s"] = w_df["t_rel_s"] - t_start
    return w_df


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def _plot_signal(dataset_name: str,
                 sig: dict,
                 tier_windows: dict,   # {"low": [w1..w5], "medium": [...], "high": [...]}
                 out_path: Path,
                 is_large: bool) -> None:
    """
    1 row × 3 columns. Each column = one dynamic tier.
    Each subplot: N_REPS lines from that tier for the given signal.
    """
    col = sig["col"]

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H), sharey=False)

    for ax, tier_name, tier_label in zip(axes, ["low", "medium", "high"], TIER_LABELS):
        items = tier_windows[tier_name]
        stds = []

        for i, item in enumerate(items):
            color = REP_COLORS[i]

            # Retrieve data depending on mode
            if is_large:
                std_val, clip_fp, t_start = item
                w_df = _get_window_data_large(clip_fp, t_start)
            else:
                std_val, w_dict = item
                w_df = w_dict["data"]

            if w_df is None or col not in w_df.columns:
                continue
            if w_df[col].isna().all():
                continue

            t = w_df["t_win_s"].values
            y = w_df[col].values

            pct_label = f"{int(round(TIER_PERCENTILES[i] * 100))}%"
            ax.plot(t, y, color=color, label=f"{pct_label} (σ={std_val:.3f})")
            stds.append(std_val)

        # Subplot title with std range of tier
        if stds:
            title = f"{tier_label}\nσ range: {min(stds):.3f} – {max(stds):.3f}"
        else:
            title = tier_label
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"{sig['label']} ({sig['unit']})")
        ax.set_xlim(0, WINDOW_S)
        ax.legend()
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
    ds_dir, ds_name, fig_root = args

    clip_files = sorted(ds_dir.glob("clip_*.parquet"))
    if not clip_files:
        print(f"  [WARN] {ds_name}: no clip_*.parquet files found")
        return None

    is_large = len(clip_files) > LARGE_CLIP_THRESHOLD

    # ── collect scored windows ───────────────────────────────────────────────
    if is_large:
        scored_raw = _collect_windows_large(clip_files)
        # scored_raw: list of (std, clip_fp, t_start)
        windows_scored = [(s, fp, ts) for s, fp, ts in scored_raw]
    else:
        all_windows = _collect_windows_normal(clip_files)
        # scored: list of (std, window_dict)
        windows_scored = [(_score_window(w), w) for w in all_windows]

    # ── select representatives ───────────────────────────────────────────────
    selection = _select_representatives(windows_scored)
    if selection is None:
        print(f"  [WARN] {ds_name}: not enough eligible windows (≥ {MIN_STD} m/s std)")
        return None

    tier_windows, n_eligible = selection

    # ── summary print ────────────────────────────────────────────────────────
    ts = _now()
    low_stds  = [item[0] for item in tier_windows["low"]]
    med_stds  = [item[0] for item in tier_windows["medium"]]
    high_stds = [item[0] for item in tier_windows["high"]]
    print(
        f"── {ds_name.upper():<12} [{ts}]  "
        f"eligible: {n_eligible:,}  |  "
        f"low σ=[{min(low_stds):.3f}–{max(low_stds):.3f}]  "
        f"med σ=[{min(med_stds):.3f}–{max(med_stds):.3f}]  "
        f"high σ=[{min(high_stds):.3f}–{max(high_stds):.3f}]"
    )

    # ── plot one PNG per signal ──────────────────────────────────────────────
    for sig in SIGNALS:
        out_path = fig_root / f"{ds_name}_{sig['col']}.png"
        _plot_signal(ds_name, sig, tier_windows, out_path, is_large)

    return {"dataset": ds_name, "n_eligible": n_eligible}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import os

    inertial_root = Path("/Users/edward/Downloads/dataset/inertial_data")
    fig_root      = Path("/Users/edward/Downloads/bachelor-arbeit/inertial_exploratory_analysis/figures/time_series")
    fig_root.mkdir(parents=True, exist_ok=True)

    datasets = sorted(
        d for d in inertial_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    if not datasets:
        print(f"[ERROR] No dataset folders found in {inertial_root}")
        sys.exit(1)

    n_workers   = min(len(datasets), os.cpu_count() or 4)
    total_pngs  = len(datasets) * len(SIGNALS)
    t_start     = datetime.now()

    print(f"\n{'='*65}")
    print(f"  representative_windows.py")
    print(f"  Window: {WINDOW_S}s   Step: {STEP_S}s   Min std: {MIN_STD} m/s")
    print(f"  Reps per tier: {N_REPS}  at percentiles {[int(p*100) for p in TIER_PERCENTILES]}%")
    print(f"  Datasets: {len(datasets)}   Signals: {len(SIGNALS)}   PNGs: {total_pngs}")
    print(f"  Workers:  {n_workers}  (CPU cores: {os.cpu_count()})")
    print(f"  Output  → {fig_root}")
    print(f"  Started → {_now()}")
    print(f"{'='*65}\n")

    args = [(ds_dir, ds_dir.name, fig_root) for ds_dir in datasets]

    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_dataset, a): a[1] for a in args}
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
    print(f"  ✓ DONE  —  {done}/{len(datasets)} datasets  "
          f"·  {done * len(SIGNALS)} PNGs saved  "
          f"·  {elapsed:.1f}s  ·  {_now()}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()