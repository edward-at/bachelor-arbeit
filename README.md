# Identifying Dynamic Biases in Large-Scale Public Datasets via Distributional Analysis

Bachelor thesis (TUM) — analysis code for quantifying **dynamic bias** in 14 public
autonomous-driving ego-motion datasets, recovering **dynamic regimes** by clustering,
and building a balanced curated subset.

This repository holds only the **analysis and plotting code**. The data lives in a
separate tree (`~/Downloads/dataset`, see [Data layout](#data-layout)); it is not
version-controlled.

## Research questions

- **RQ1 — Quantify** the low-dynamic skew (near-stationary / low-speed bias) across the 14 datasets.
- **RQ2 — Recover** dynamic regimes from per-window behaviour via clustering.
- **RQ3 — Curate** a balanced subset that corrects the skew without discarding measurement data.

## Quick start

```bash
# 1. From the repo root, create and activate the environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies (matplotlib, numpy, pandas, scipy, scikit-learn, pyarrow, ...)
pip install -r requirements.txt

# 3. Run one figure script (each script writes into its module's figures/ folder)
cd inertial_exploratory_analysis
python ecdf_analysis.py

# 4. Or run the whole exploratory batch in dependency order
python run_all.py
```

Every script is standalone: it reads a data layer from `~/Downloads/dataset/...`,
computes, and saves PNGs (`dpi=300`) into a `figures/<type>/` subfolder. Nothing is
interactive and nothing mutates the stored data.

> **Note.** Scripts resolve the dataset tree by absolute path (`/Users/edward/Downloads/dataset`).
> If you move the data, update the `*_DIR` constants at the top of each script (they are all
> top-level constants).

## Pipeline overview

The project runs as a linear pipeline. Each stage reads the previous layer and writes a new one.

```
raw per-dataset dumps
        │  standardize_inertial_data.py   (v3, per-clip)
        ▼
inertial_data/   ── 43-column unified clips on a common 10 Hz grid
        │  compute_std.py                 (v4, per-window)
        ▼
std_data/        ── per-window stds + feature means (20 s window, 10 s step, 50% overlap)
        │
        ├── inertial_exploratory_analysis/   → RQ1: distributional analysis + variability tiers
        │
        └── regime_discovery/                → RQ2: feature gate → clustering → validation
                 │
                 ▼
            regime_data/  ── cluster assignments, profiles, GMM/HMM, validity indices
```

Two foundational scripts are **run once** and deliberately excluded from `run_all.py`:

- **`standardize_inertial_data.py`** — the ingestion pipeline. Per clip it does timestamp
  cleanup, GPS cleaning, velocity derivation from low-passed positions, Hampel outlier
  removal, a gap policy (interpolate gaps ≤ 0.5 s, mask longer gaps and NaN-runs), a
  zero-phase 3 Hz Butterworth anti-alias filter, resampling to a 10 Hz grid,
  Savitzky-Golay derivatives per valid segment, derivative/edge masking, and curvature.
  Output is the 43-column schema.
- **`compute_std.py`** — collapses the frame layer into the windowed `std_data` layer.
  In a single per-window pass it emits both the **variability channels** (`std_speed_kmh`,
  `std_acc_mag`, `std_yaw_rate`) used in RQ1 and the **feature-mean channels**
  (`mean_speed_kmh`, `mean_acc_pos`, `mean_acc_neg`, `mean_abs_a_lat`, `mean_abs_yaw_rate`,
  `mean_abs_jerk`) used as the RQ2 clustering vector. A window's statistic is `NaN`
  unless at least 50% of its expected frames survive masking.

## Repository structure

| Path | Purpose |
|------|---------|
| `dataset_analysis/` | Metadata-level analysis of the dataset *pool* (not ego-motion signals): sensor counts, recording countries, real/synthetic mix, availability, scoring. Cleans a CSV pool and renders descriptive figures. |
| `dataset_analysis/total_dataset_analysis/` | Full pool (scripts `00`–`10`): year trends, cumulative growth, sensor/vehicle info, country maps. |
| `dataset_analysis/filtered_dataset_analysis/` | Filtered pool (scripts `00`–`16`): sensor modality counts, IMU grade/frequency, road/weather/time-of-day, quality scores. |
| `inertial_exploratory_analysis/` | **RQ1.** Standardization + std computation, then distributional figures: ECDF, ridgeline, boxplot, heatmap summary, phase portrait, density band, trajectory fan, time series, and the four fixed variability **tiers**. |
| `regime_discovery/` | **RQ2.** Feature gating, k-means clustering (with k-selection and a shuffled null baseline), GMM and HMM alternatives, PCA scatters, consensus/stability, state durations/timelines, transition matrices, and validity indices. |
| `bias_quantification/` | RQ3 workspace (currently empty — reserved for the balanced-subset construction). |
| `tum_colors.py` | Official TUM colour palette; imported directly by every plotting script (one copy per module). |
| `run_all.py` | Runs the exploratory scripts in dependency order, stops on first failure. |
| `requirements.txt` | Pinned dependencies. |

## Data layout

The data tree is separate from the code and is **not** in git.

| Layer | Location | What it is |
|-------|----------|------------|
| `parquet_data/<dataset>/` | `~/Downloads/dataset/parquet_data` | Native-Hz per-clip parquet, pre-standardization. |
| `inertial_data/<dataset>/clip_XXXX.parquet` | `~/Downloads/dataset/inertial_data` | 43-column unified clips on the 10 Hz grid. |
| `std_data/<dataset>.parquet` | `~/Downloads/dataset/std_data` | **Single source of truth for windowed analysis** (per-window stds + feature means). |
| `regime_data/*.parquet` | `~/Downloads/dataset/regime_data` | Clustering outputs: `cluster_assignments`, `cluster_profile`, `feature_gate_vector`, `gmm_*`, `hmm_*`, `validity_indices`, etc. |
| `csv_data/`, `removed_inertial/` | `~/Downloads/dataset` | Intermediate CSVs and quarantined clips. |

**The 14 datasets:** Boreas, DurLAR, Ford, KITTI, KITTI-360, nuScenes, NVIDIA, Oxford,
Waymo, ZOD, 4Seasons, HDD, A2D2, KITScenes. Each has one fixed display name and one fixed
TUM colour, reused across every figure.

**Back-reference contract:** `(clip_id, window_start_s, window_end_s)` recovers a window's
original frames from its clip, so downstream scripts never re-window.

## Key conventions

- **Terminology.** The four fixed-threshold bins (near-stationary / low / medium / high) are
  **variability tiers** — a descriptive, single-signal device. The word **"regime"** is
  reserved for the multivariate behavioural categories recovered by clustering in RQ2.
- **Windowing.** 20 s windows, 10 s step (50% overlap), 10 Hz grid, `MIN_VALID_FRAC = 0.5`.
- **Physical-validity bounds** (e.g. `speed_kmh ∈ [0, 200]`, `yaw_rate ∈ [-90, 90] °/s`) are
  applied as **NaN masking at load/compute/viz time only** — stored data is never modified.
- **`yaw_rate`** is stored in rad/s; multiply by `180/π` at visualization time.
- **`acc_mag` / `jerk_mag`** are precomputed upstream (`sqrt(a_long² + a_lat²)`, etc.) and
  read directly — never re-derived.
- **Plot style.** Pure default matplotlib, `dpi=300`, colours only from `tum_colors`.
- **Reproducibility.** Clustering runs once under a fixed seed; downstream figures read the
  saved `regime_data` layer and never re-fit.

## Requirements

Python 3 with the packages pinned in `requirements.txt`. The core scientific stack:
`numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `pyarrow`. Progress bars use
`tqdm` where present and degrade gracefully when it is missing.

## Notes for contributors / future me

- Every tunable is a top-level constant; datasets are auto-discovered from `std_data/`
  where possible rather than hard-coded.
- File/parquet reads are parallelised with `ThreadPoolExecutor`; CPU-bound per-dataset
  rendering uses `ProcessPoolExecutor` (matplotlib is not thread-safe within a process).
- For the giant datasets (NVIDIA, Waymo), read the schema once from the first clip, then
  stream/batch the rest to bound memory.
- Re-run `standardize_inertial_data.py` and `compute_std.py` only when the ingestion logic
  changes — everything else consumes their outputs.
