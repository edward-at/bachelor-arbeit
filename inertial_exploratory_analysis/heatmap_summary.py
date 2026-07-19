"""
heatmap_summary.py  —  Chapter-8 "scorecard" heatmap.

A single summary figure for the whole exploratory chapter.

  Rows    : the auto-discovered datasets (one std_data parquet each).
  Columns : one headline metric per facet, GROUPED and traceable to a section
            in 8.1-8.8, oriented so that "more dynamic" -> darker.
  Spine   : a Dynamic-Index column (mean of the per-column normalised
            intensities) that sets the row order.
  Colour  : per-column min-max normalised intensity, Blues colormap
            (darker = more dynamic). The REAL value is printed in every cell;
            colour only encodes the within-column ranking.

Most columns are cheap and come straight from std_data, which is the single
source of truth for windowed analysis. One facet that std_data simply cannot
express -- frame-level stationarity (8.1) -- needs one streaming pass over the
frame layer. Set COMPUTE_FRAME_METRICS = False to skip that pass entirely and
drop that column for a fast, std_data-only run.

Honest caveats (carried into the §8.9 methodology):
  * The columns are correlated by construction (a calm dataset is calm on most
    of them). That redundancy is a finding, not a bug -- the eye should SEE the
    same light/dark stripe repeat across a row.
  * Per-column min-max stretches the contrast: a column where all datasets are
    nearly identical still spans light->dark. That is why the real value is
    printed in every cell -- read the numbers, not just the shade.
  * sigma Yaw (8.6) doubles as the cornering / geometry proxy for 8.4, and IQR
    of sigma-speed (8.3) is a homogeneity proxy, not the literal per-time band
    width.

Strict matplotlib: pure defaults, dpi=300. The only deliberate deviation is a
single small font size for in-cell text and the column-group labels, which is
the documented latitude every packed grid figure in this thesis uses.
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:  # tqdm is optional -> degrade to a no-op
    def tqdm(x, **kwargs):
        return x

# ────────────────────────────── paths ──────────────────────────────
PROJECT_ROOT = Path("/Users/edward/Downloads")
INERTIAL_DIR = PROJECT_ROOT / "dataset" / "inertial_data"
STD_DATA_DIR = PROJECT_ROOT / "dataset" / "std_data"
OUT_DIR = PROJECT_ROOT / "bachelor-arbeit" / "inertial_exploratory_analysis" / "figures" / "summary"

# ───────────────────────── display names ───────────────────────────
DISPLAY_NAMES = {
    "boreas_rt": "Boreas", "durlar": "DurLAR", "ford": "Ford", "kitti": "KITTI",
    "kitti360": "KITTI-360", "nuscene": "nuScenes", "nvidia": "NVIDIA",
    "oxford": "Oxford", "waymo": "Waymo", "zod": "ZOD",
    "4seasons": "4Seasons", "hdd": "HDD", "a2d2": "A2D2", "kitscenes": "KITScenes",
}

# ───────────────── tier thresholds (speed sigma, km/h) ──────────────
# near-stationary < t1 ; low [t1,t2) ; medium [t2,t3) ; high >= t3
T1_SPEED, T2_SPEED, T3_SPEED = 1.0, 5.0, 15.0

# ─────────────────────── frame-pass config ─────────────────────────
COMPUTE_FRAME_METRICS = True     # False -> skip the frame layer, drop Stat %
FRAME_SAMPLE_CLIPS = 3000        # cap clips read per dataset (None = all). Estimates
                                 # for Stat % are stable under sampling, and this
                                 # keeps NVIDIA (~298k clips) and Waymo tractable.
N_FRAME_WORKERS = 8
STATIONARY_KMH = 0.5             # a frame is "stationary" below this speed (8.1)

# ────────────── physical-validity bounds (NaN-mask only) ────────────
SPEED_MAX_KMH = 200.0
RAD2DEG = 180.0 / np.pi

# ──────────────────────────── render ───────────────────────────────
DPI = 300
CELL_FS = 8                      # documented heatmap exception: in-cell + group-label font

# ─────────────────────── column definitions ────────────────────────
# key, on-figure label, section, facet group, invert?, needs frame layer?, print format
COLUMNS = [
    dict(key="stat_frac",     label="Stat %",   group="Stationarity", invert=True,  frame=True,  fmt="%.1f"),
    dict(key="ns_pct",        label="NS %",     group="Stationarity", invert=True,  frame=False, fmt="%.0f"),
    dict(key="med_sig_speed", label="\u03c3 Spd",   group="Central \u03c3", invert=False, frame=False, fmt="%.1f"),
    dict(key="med_sig_acc",   label="\u03c3 Acc",   group="Central \u03c3", invert=False, frame=False, fmt="%.2f"),
    dict(key="med_sig_yaw",   label="\u03c3 Yaw",   group="Central \u03c3", invert=False, frame=False, fmt="%.1f"),
    dict(key="high_pct",      label="High %",   group="Tail",         invert=False, frame=False, fmt="%.0f"),
    dict(key="iqr_sig_speed", label="IQR Spd",  group="Homog.",       invert=False, frame=False, fmt="%.1f"),
]


# ══════════════════════════ std_data metrics ═══════════════════════
def std_metrics(parquet_path):
    """Cheap per-window-sigma summaries from one dataset's std_data parquet."""
    df = pd.read_parquet(parquet_path, columns=["std_speed_kmh", "std_acc_mag", "std_yaw_rate"])
    spd = df["std_speed_kmh"].to_numpy(dtype=float)
    acc = df["std_acc_mag"].to_numpy(dtype=float)
    yaw = df["std_yaw_rate"].to_numpy(dtype=float) * RAD2DEG  # rad/s -> deg/s

    spd = spd[np.isfinite(spd)]
    acc = acc[np.isfinite(acc)]
    yaw = yaw[np.isfinite(yaw)]

    out = {k: np.nan for k in
           ("ns_pct", "high_pct", "med_sig_speed", "med_sig_acc", "med_sig_yaw", "iqr_sig_speed")}
    if spd.size:
        out["ns_pct"] = 100.0 * float(np.mean(spd < T1_SPEED))
        out["high_pct"] = 100.0 * float(np.mean(spd >= T3_SPEED))
        out["med_sig_speed"] = float(np.median(spd))
        q25, q75 = np.percentile(spd, [25, 75])
        out["iqr_sig_speed"] = float(q75 - q25)
    if acc.size:
        out["med_sig_acc"] = float(np.median(acc))
    if yaw.size:
        out["med_sig_yaw"] = float(np.median(yaw))
    return out


# ═══════════════════════════ frame metrics ═════════════════════════
def _accum_clip(clip_path):
    """Return (n_low_speed, n_valid) for one clip."""
    try:
        df = pd.read_parquet(clip_path, columns=["speed_kmh"])
    except Exception:
        df = pd.read_parquet(clip_path)

    spd = df["speed_kmh"].to_numpy(dtype=float) if "speed_kmh" in df.columns else np.empty(0)

    valid_spd = np.isfinite(spd) & (spd >= 0.0) & (spd <= SPEED_MAX_KMH)
    n_valid = int(valid_spd.sum())
    n_low = int((valid_spd & (spd < STATIONARY_KMH)).sum())
    return n_low, n_valid


def frame_metrics(dataset_dir):
    """Stat % (8.1) from one streaming pass over clips."""
    clips = sorted(dataset_dir.glob("clip_*.parquet"))
    if not clips:
        return {"stat_frac": np.nan}

    if FRAME_SAMPLE_CLIPS and len(clips) > FRAME_SAMPLE_CLIPS:
        rng = np.random.default_rng(0)
        pick = sorted(rng.choice(len(clips), size=FRAME_SAMPLE_CLIPS, replace=False))
        clips = [clips[i] for i in pick]

    n_low = n_valid = 0
    with ThreadPoolExecutor(max_workers=N_FRAME_WORKERS) as ex:
        for nl, nv in tqdm(ex.map(_accum_clip, clips), total=len(clips),
                           desc=f"frames:{dataset_dir.name}", leave=False):
            n_low += nl
            n_valid += nv

    stat = 100.0 * n_low / n_valid if n_valid else np.nan
    return {"stat_frac": stat}


# ══════════════════════════════ build ══════════════════════════════
def build_table():
    datasets = [p.stem for p in sorted(STD_DATA_DIR.glob("*.parquet"))]
    if not datasets:
        raise SystemExit(f"No std_data parquet files found in {STD_DATA_DIR}")

    records = {}
    for ds in tqdm(datasets, desc="datasets"):
        rec = std_metrics(STD_DATA_DIR / f"{ds}.parquet")
        if COMPUTE_FRAME_METRICS:
            rec.update(frame_metrics(INERTIAL_DIR / ds))
        records[ds] = rec

    cols = [c for c in COLUMNS if (COMPUTE_FRAME_METRICS or not c["frame"])]
    raw = np.array([[records[ds].get(c["key"], np.nan) for c in cols] for ds in datasets], dtype=float)
    return datasets, cols, raw


def normalise(raw, cols):
    """Per-column min-max -> dynamism intensity in [0, 1]; index = row mean."""
    inten = np.full_like(raw, np.nan)
    for j, c in enumerate(cols):
        col = raw[:, j]
        finite = np.isfinite(col)
        if not finite.any():
            continue
        lo, hi = np.nanmin(col), np.nanmax(col)
        t = (col - lo) / (hi - lo) if hi > lo else np.zeros_like(col)
        if c["invert"]:
            t = 1.0 - t
        t = np.asarray(t, dtype=float)
        t[~finite] = np.nan          # keep missing cells missing, never count them as 0
        inten[:, j] = t
    index = np.nanmean(inten, axis=1)
    return inten, index


# ═════════════════════════════ render ══════════════════════════════
def render(datasets, cols, raw, inten, index):
    # order rows by the Dynamic Index (descending); NaN-index datasets sink to the bottom
    key = np.where(np.isfinite(index), index, -np.inf)
    order = np.argsort(-key)
    datasets = [datasets[i] for i in order]
    raw, inten, index = raw[order], inten[order], index[order]

    n_ds, n_col = len(datasets), len(cols)
    disp = np.column_stack([inten, index])  # last column = the index spine

    fig_w = max(11.0, 1.15 * (n_col + 1) + 3.0)
    fig_h = max(6.0, 0.46 * n_ds + 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(disp, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")

    # axis ticks / labels
    xpos = np.arange(n_col + 1)
    xlabels = [f'{c["label"]}' for c in cols] + ["Index\n(0\u2013100)"]
    ax.set_xticks(xpos)
    ax.set_xticklabels(xlabels)
    ax.set_yticks(np.arange(n_ds))
    ax.set_yticklabels([DISPLAY_NAMES.get(d, d) for d in datasets])

    # headroom above row 0 for the facet-group labels (origin is upper => top is negative)
    ax.set_ylim(n_ds - 0.5, -1.5)

    # in-cell real values
    for i in range(n_ds):
        for j, c in enumerate(cols):
            v = raw[i, j]
            if np.isfinite(v):
                tc = "white" if disp[i, j] > 0.55 else "black"
                ax.text(j, i, c["fmt"] % v, ha="center", va="center", color=tc, fontsize=CELL_FS)
        iv = index[i]
        if np.isfinite(iv):
            tc = "white" if iv > 0.55 else "black"
            ax.text(n_col, i, "%.0f" % (iv * 100.0), ha="center", va="center", color=tc, fontsize=CELL_FS)

    # facet-group separators + labels
    spans, start = [], 0
    for j in range(1, n_col + 1):
        if j == n_col or cols[j]["group"] != cols[start]["group"]:
            spans.append((cols[start]["group"], start, j - 1))
            start = j
    for _, a, b in spans[1:]:
        ax.axvline(a - 0.5, color="black", linewidth=1)
    ax.axvline(n_col - 0.5, color="black", linewidth=2)  # divide facets from the index spine

    ax.axhline(-1.5, color="black", linewidth=1)   # top border
    ax.axhline(-0.5, color="black", linewidth=1)   # line between labels and heatmap

    for name, a, b in spans:
        ax.text((a + b) / 2.0, -0.95, name, ha="center", va="center", fontsize=CELL_FS)
    ax.text(n_col, -0.95, "Index", ha="center", va="center", fontsize=CELL_FS)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("column-normalised intensity (darker = more dynamic)")

    ax.set_title("Dynamic-content summary across datasets")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "heatmap_summary.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ═══════════════════════════════ main ══════════════════════════════
def main():
    datasets, cols, raw = build_table()

    # echo the assembled numbers so they can be eyeballed and reused in prose
    table = pd.DataFrame(raw, index=[DISPLAY_NAMES.get(d, d) for d in datasets],
                         columns=[c["key"] for c in cols])
    inten, index = normalise(raw, cols)
    table["INDEX_0_100"] = np.round(index * 100.0, 1)
    table = table.sort_values("INDEX_0_100", ascending=False)
    pd.set_option("display.width", 200, "display.max_columns", 50)
    print("\nSummary metrics (real values):\n")
    print(table.round(3).to_string())

    out_path = render(datasets, cols, raw, inten, index)
    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()