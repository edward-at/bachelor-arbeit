# Identifying Dynamic Biases in Large-Scale Public Datasets via Distributional Analysis

Bachelor thesis (TUM, Professorship of Autonomous Vehicle Systems). Analysis code for
quantifying **dynamic bias** in 14 public autonomous-driving ego-motion datasets (RQ1),
recovering **dynamic regimes** by unsupervised clustering (RQ2), and constructing a
**dynamically balanced curated subset** (RQ3).

This repository holds only the analysis and plotting code. The data lives in a separate
tree (`~/Downloads/dataset`, see [Data layout](#data-layout)) and is not version-controlled.

**Status: the analysis pipeline is closed (2026-08-30).** The full pipeline was re-run
end to end from cleared outputs on the v4_4 standardization and the non-overlapping
window grid, every stage completed green, and the corpus snapshot is frozen in
`dataset/canonical_snapshot.json`. All remaining thesis work is prose.

## Research questions

- **RQ1 — Quantify** the low-dynamic skew (near-stationary and low-speed bias) across the 14 datasets.
- **RQ2 — Recover** interpretable dynamic regimes from per-window behavior via clustering.
- **RQ3 — Curate** a balanced subset that corrects the skew without discarding measurement data.

## Headline numbers (frozen)

| Quantity | Value |
|---|---|
| Datasets | 14 (Boreas, DurLAR, Ford, KITTI, KITTI-360, nuScenes, NVIDIA, Oxford, Waymo, ZOD, 4Seasons, HDD, A2D2, KITScenes) |
| Raw corpus | 371,431 clips, 178,778,906 native-rate frames |
| Standardization | v4_4, 10 Hz common grid, validity gated |
| Window layer | 406,130 windows (20 s, non-overlapping) |
| Analysis frame | 402,421 windows (five-feature complete case) |
| Guarded frame | 401,230 windows (yaw validity bound) |
| Regimes (k-means, k = 5, seed 42) | supplies 89,420 / 89,499 / 78,883 / 73,229 / 71,390 |
| Budget tier N | 71,390 (rarest regime supply) |
| Capability tier N | 356,950 (5 x rarest) |
| Event reserve (0.6 g decel) | 1,088 windows |
| Evasive population (0.75 g lateral) | 2,294 windows |
| Curation winner | C01, sole survivor of the pre-registered D4 gates |

`bachelor-arbeit/general_info.py` regenerates every thesis-relevant number from the
artifacts and writes `general_info.txt`. `subset_curation/canonical.py` derives the
canonical constants live and verifies them against the frozen snapshot
(`python canonical.py --check`).

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt      # 8 pinned packages, see Requirements

# one figure script (each writes PNGs into its module's figures/ tree)
cd inertial_exploratory_analysis
python ecdf_analysis.py

# or a whole stage through its runner
python run_pipeline.py --from coverage
```

Every script is standalone. It reads a data layer from `~/Downloads/dataset/...`,
computes, and saves PNGs at `dpi=300` into a `figures/<type>/` subfolder. Nothing is
interactive and nothing mutates stored data.

> Scripts resolve the dataset tree by absolute path (`/Users/edward/Downloads/dataset`).
> If the data moves, update the `*_DIR` top-level constants.

## Pipeline overview

```
raw per-dataset dumps (parquet_data/)
        |  standardize_inertial_data.py     v4_4, per clip: timestamp cleanup, GPS
        |                                   cleaning, velocity from band-limited
        |                                   positions (central differences), Hampel,
        |                                   gap policy, zero-phase 3 Hz Butterworth,
        |                                   10 Hz resample, Savitzky-Golay derivatives,
        |                                   edge masks, validity gates
        v
inertial_data/    43-column unified clips on the common 10 Hz grid
        |  compute_std.py                   20 s windows advanced 20 s (non-overlapping,
        |                                   D-2), variability + feature-mean channels,
        v                                   MIN_VALID_FRAC = 0.5
std_data/         one parquet per dataset, 406,130 windows
        |
        |-- inertial_exploratory_analysis/  RQ1 distributional analysis + coverage audit
        |
        |-- regime_discovery/               RQ2 feature gate -> k-means -> validation
        |          v
        |     regime_data/                  assignments, profiles, GMM/HMM, stability
        |
        |-- bias_quantification/            RQ1 instruments on the regime structure
        |
        '-- subset_curation/                RQ3 sampling methods (M01-M11), curation
                   v                        methods (C01-C03), gates, winner dossier
              subset_data/                  subsets (2 tiers), eval caches, manifests
```

Execution order for a full re-run (dependency order, not preference):

```bash
cd inertial_exploratory_analysis
python pipeline_regression_test.py            # synthetic gate, quotes the -4.96 value
python run_pipeline.py --from coverage        # skip standardize unless raw data changed

cd ../regime_discovery
python regime_run_pipeline.py

cd ../bias_quantification
python bias_run_pipeline.py

cd ../subset_curation
python window_summary.py
python run_sampling.py                        # capability tier
python run_sampling.py --n 71390              # budget tier
python curation_run_pipeline.py               # includes the widened frame pass
python curation_evassive_check.py             # D24 evidence, not in the runner
python subset_run_pipeline.py                 # Chapter 12 figures, renders from caches
python canonical.py --check                   # oracle: must report no drift
python curation_info_dump.py                  # Chapter 5 number set
```

Why the Chapter 13 runner comes before the Chapter 12 renderer: `curation_eval.py`
evaluates the entire slate (M01-M11 and C01-C03) in one pass and produces the shared
eval caches. `subset_run_pipeline.py` computes nothing, it renders those caches.

## Repository structure

| Path | Purpose |
|---|---|
| `dataset_analysis/` | Chapter 5/6 survey of the dataset pool (metadata, not ego-motion signals). Exempt from the pipeline. |
| `dataset_analysis/total_dataset_analysis/` | Full pool, scripts 00-10 (+ 01a-01d, 04a-04b, 06, 07a-07b): year trends, cumulative growth, sensors, countries, availability, originality. Figures in `total_dataset_analysis_figures/`. |
| `dataset_analysis/filtered_dataset_analysis/` | Filtered pool, scripts 00-16 (+ 08a-08b): sensor counts, IMU grade and frequency, GPS type, CAN, road, weather, time of day, citations, QS/DS/DSS scores. Figures in `filtered_dataset_analysis_figures/`. |
| `inertial_exploratory_analysis/` | RQ1. Standardization, windowing, coverage audit, distributional figures, variability tiers. Runner: `run_pipeline.py`. |
| `regime_discovery/` | RQ2. Feature gate, k-means with k-selection, full stability battery, GMM and HMM cross-checks. Runner: `regime_run_pipeline.py`. |
| `bias_quantification/` | RQ1 instruments. Tier-1 composition metrics and Tier-2 diagnostics on the regime structure. Runner: `bias_run_pipeline.py`. |
| `subset_curation/` | RQ3. Sampling methods, curation methods, pre-registered gates, replication, winner dossier, Chapter 12 figure set. Runners: `curation_run_pipeline.py`, `subset_run_pipeline.py`, plus `run_sampling.py`. |
| `general_info.py` | Reads every artifact and writes `general_info.txt`, the numeric ground truth for the write-up. |
| `requirements.txt` | The 8 pinned direct dependencies. |
| `thesis_audit.py` | Personal audit helper, not part of the pipeline. |

`tum_colors.py` (official palette) is duplicated into each module by design, since every
module imports its local copy. `regime_style.py` (regime names, colors, order) is shared
by the regime, bias, and subset modules.

### inertial_exploratory_analysis

Producers: `standardize_inertial_data.py` (run once per raw-data change),
`compute_std.py`, `coverage.py` + `coverage_baseline.py` (writes `audit_baseline.json`,
the post-cleaning Section 7.3 numbers), `regime_classification.py` (the D-5 frozen
variability tiers). `pipeline_regression_test.py` is the synthetic gate for the
standardization code. `standardize_nvidia.py` is a one-off raw converter, already applied.

Figure tree (`figures/`): `boxplot/`, `coverage/`, `density_band/` (one per dataset),
`distribution/` (per dataset and signal), `ecdf/`, `phase_portrait/` (speed-along and
speed-yaw per dataset), `regime/` (tier figures), `ridgeline/`, `summary/`,
`time_series/`, `trajectory_fans/` (trajectory and speed-change fans per dataset).
About 184 figures.

Runner flags: `--dry-run`, `--from <stage>`, `--only <stage>`, `--skip <stage>`,
`--keep-going`. Stages: standardize, coverage, compute_std, thresholds, frame_figs,
window_figs, time_series, summary.

### regime_discovery

`feature_gate.py` locks the five-feature clustering vector (drops |aLat| for |yaw|,
VIF after <= 2.96). `clustering.py` is the single stochastic fit (seed 42, cap 5,000
per dataset, 10 intensity strata) and writes `cluster_assignments.parquet`, the spine
of everything downstream. Then the evidence battery: `k_select_curve`,
`validity_indices`, `validity`, `regime_stability` (bootstrap Jaccard),
`reseed_stability`, `resample_stability`, `cap_sensitivity`, `consensus_matrix`, the
PCA scatters (2D and 3D, pre- and post-labeling), `regime_heatmap` and
`regime_heatmap_sweep`, the GMM branch (`gmm_selection`, `gmm`, `membership_entropy`,
`gmm_kmeans_correspondence`, `covariance_contours`), and the HMM branch (`hmm`,
`transition_matrix`, `state_duration`, `state_timeline` on 2 s / 1 s sub-windows).
About 20 figures in `figures/`.

### bias_quantification

Stages: compute (`bias_quant.py`, the Tier-1 core), tier1_figs (`bias_stacked_bar`,
`bias_regime_grid`, `bias_share_heatmap`, `bias_balance_metrics`, `bias_wasserstein`),
tier2 (`bias_diversity` Hill numbers and PSI, `bias_volume` Gini and effective sources,
`bias_bootstrap` clip-level CIs and rank probabilities, `bias_gmm_entropy` boundary
occupancy, `bias_shape_metrics` dynamic-content Gini and Lorenz, `bias_regime_supply`
per-regime suppliers), fingerprint (`bias_fingerprint` 10-metric table plus the
Spearman agreement matrix). `bias_tier2_common.py` is the shared loader with the
Tier-1 cross-check (agreement to 1e-12). 14 figures.

### subset_curation

Construction: `window_summary.py` (per-window physical summaries with a keys-plus-mtime
freshness cache), `run_sampling.py` (M01-M11 at both tiers: SRS, proportional, equal,
power q=0.5, two-way regime x source, PPS, kinematic trigger, predicate mining,
Kennard-Stone, cluster quota fill, Neyman), `curation_build.py` (C01-C03, D22 event
reserve, D23 per-regime source cap).

Protocol (enforced order in `curation_run_pipeline.py`): `curation_stale_audit` ->
`curation_event_audit` -> `curation_build` -> `curation_eval` (patches the Chapter 12
harness, evaluates the full slate, writes the shared caches) -> `curation_gates`
(pre-registered D4 gates, dated 2026-08-20) -> `curation_feasibility` ->
`curation_replicate` (R = 20 seeds) -> `curation_reveal` (sealed shelf and the
ex-confounded slice) -> `curation_render_winner` (winner figures and the widened frame
pass) -> `curation_winner_dossier` -> `curation_winner_pca`.

Evaluation and rendering: `subset_eval_harness.py` (metric engine, METRIC_VERSION
12.1.2), `subset_eval_common.py` (shared constants), `subset_frame_pass.py` (streams
the frame layer once, records the group list in `frame_pass_meta.json`), and the
figure scripts behind `subset_run_pipeline.py`: stacked bar, regime grid, boxplots,
ridgeline, Wasserstein heatmaps (in-model and held-out), source composition, event
retention, bias axes, mean shift, ECDF grid, density grid, PCA grids, fingerprint,
histograms, phase portraits, fans, time series, finalist ECDF overlay.

`canonical.py` derives the canonical constants from the artifacts (`--freeze` writes
the snapshot, `--check` verifies against it). `curation_info_dump.py` prints the
Chapter 5 number set. `curation_evassive_check.py` is the held-out D24 evidence and
runs outside the runner.

Figures split across `figures/subset_eval/` (Chapter 12, M-slate) and
`figures/curation/` (Chapter 13, winner dossier, feasibility, finalists with C01,
PCA overlay). C01 pages that appear in the subset_eval dossier folders are surplus,
the canonical copies live in `figures/curation/`.

## Data layout

Separate from the code, not in git.

| Layer | Location | What it is |
|---|---|---|
| `parquet_data/<ds>/` | `~/Downloads/dataset/parquet_data` | Native-rate per-clip parquet, pre-standardization. |
| `inertial_data/<ds>/clip_XXXX.parquet` | `.../inertial_data` | 43-column v4_4 clips on the 10 Hz grid, `processing_meta` and `col_sources` in the file metadata. |
| `std_data/<ds>.parquet` | `.../std_data` | The windowed layer, single source of truth for all window analysis. |
| `regime_data/*.parquet` | `.../regime_data` | Assignments, profiles, k-selection, stability, GMM/HMM, all bias_* result tables. |
| `subset_data/` | `.../subset_data` | Capability-tier subsets (M01-M11, C01-C03), `window_summaries.parquet`, `_summaries_partial/` cache, `supply_table.csv`, `allocation_table.csv`, manifests. |
| `subset_data/tier_n71390/` | | The budget tier, same file set at N = 71,390. |
| `subset_data/eval/` | | Shared eval caches: master tables (both tiers), `metrics_long`, SRS envelope, tautology map, frame-pass caches (`fan_*`, `frame_*`), replication meta, `run_meta.json`. |
| `canonical_snapshot.json` | `~/Downloads/dataset` | The frozen oracle. Do not delete. Re-freeze only after a clean `canonical.py --check`. |
| `csv_data/` | | Intermediate conversion CSVs. |

Back-reference contract: `(dataset, clip_id, window_start_s)` keys the assignments,
`(dataset, clip_id, window_idx)` keys the summaries, and `std_data` carries both, so
joins bridge through it and no script ever re-windows.

## Key conventions

- **Terminology.** The four fixed-threshold bins (near-stationary, low, medium, high) are
  **variability tiers**, a descriptive single-signal device. **Regime** is reserved for
  the five multivariate clusters of RQ2. **Sampling methods** are Chapters 11-12,
  **curation methods** are Chapter 13.
- **Windowing (D-2, 2026-08-29).** 20 s windows advanced 20 s, non-overlapping, on the
  10 Hz grid, `MIN_VALID_FRAC = 0.5`.
- **Tier thresholds (D-5, 2026-08-29).** Frozen as stated round conventions, speed
  (1, 5, 15) km/h, acceleration (0.2, 0.5, 1.5) m/s^2, yaw (1, 5, 15) deg/s. Lineage
  disclosed in the thesis, never recalibrated.
- **Physical-validity bounds** (speed in [0, 200] km/h, |yaw| <= 90 deg/s, per-axis
  |a| <= 15 m/s^2, per-axis |jerk| <= 30 m/s^3) are enforced during standardization
  and applied as NaN masking downstream. Stored data is never modified.
- **Units.** `yaw_rate` stored in rad/s, converted by 180/pi at visualization.
  `max_decel_ms2` is m/s^2. Gravity is 9.81 in `run_sampling.py` and 9.80665 in
  `curation_common.py` by documented decision, since unifying would move the 0.6 g
  trigger and the event counts.
- **Seeds.** Clustering and every stochastic construction run under seed 42 (replication
  family 1200 + r). Downstream figures read saved artifacts and never re-fit.
- **Plot style.** Pure default matplotlib, `dpi=300`, TUM palette only, no rcParams.
- **Code style.** All tunables are top-level constants. Parquet reads parallelized with
  ThreadPoolExecutor, matplotlib rendering single-threaded. tqdm on long operations.

## Requirements

Python 3.12. The 8 direct dependencies, pinned in `requirements.txt`:
numpy, pandas, pyarrow, matplotlib, scipy, scikit-learn, hmmlearn, tqdm.
`pip install -r requirements.txt` on a clean machine rebuilds the analysis stack.
Raw-data conversion (one-off, long since applied) used separate tooling and is not
required to reproduce the analysis.

## Not part of the pipeline

`Test.py` (scratch, three copies), `inspect_clip_mapping.py` (diagnostic),
`check_tiers.py` (superseded by D-5), `standardize_nvidia.py` (one-off converter),
`thesis_audit.py` (personal audit). Safe to ignore.