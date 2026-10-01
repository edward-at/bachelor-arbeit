# Identifying Dynamic Biases in Large-Scale Public Datasets via Distributional Analysis

Analysis code of the bachelor thesis of Edward Anderson Tamin, Professorship of Autonomous
Vehicle Systems, TUM School of Engineering and Design, Technical University of Munich.
Supervised by Prof. Dr.-Ing. Johannes Betz and Finn Schäfer, M.Sc. Submitted on 1 October 2026.

The pipeline measures the ego-motion (kinematic) content of 14 public autonomous-driving
datasets. It standardizes every recording to a common 10 Hz grid, cuts it into 20 s windows,
describes the corpus distributionally, discovers driving regimes by clustering (RQ2),
quantifies the low-dynamic skew of each dataset and of the pooled corpus (RQ1), and builds and
evaluates regime-balanced subsets under a pre-registered selection rule (RQ3).

The repository holds code, figures, and the survey tables. It holds no dataset content. Section
and figure numbers below refer to the submitted thesis.

## Contents

1. [Repository layout](#repository-layout)
2. [The pipeline](#the-pipeline)
3. [Data](#data)
4. [Environments](#environments)
5. [Running the pipeline](#running-the-pipeline)
6. [Frozen constants](#frozen-constants)
7. [Checks](#checks)
8. [Where each figure and table is produced](#where-each-figure-and-table-is-produced)
9. [Known issues](#known-issues)
10. [Citation](#citation)

## Repository layout

```
thesis_paths.py                   REPO_ROOT and DATA_ROOT, imported by every script
requirements.txt                  environment "venv" (see Environments)
requirements-standardization.txt  environment "standardization"
01_survey_and_scoring/            dataset survey, filtering, DSS scoring (Sections 4.1, 5.1)
02_acquisition_and_extraction/    description only, see its README (Section 4.2)
03_standardization/               10 Hz standardization and the 20 s window layer (4.3 to 4.5)
04_exploratory_analysis/          coverage audit, variability tiers, exploratory figures (4.4, 4.6, 5.2, 5.3)
05_regime_discovery/              feature screen, k-means regimes, validation, GMM, HMM (4.7, 5.4)
06_bias_quantification/           bias instruments on the regime structure (4.8, 5.5)
07_subset_curation/               sampling methods, evaluation harness, curation, gates (4.9 to 4.12, 5.6 to 5.8)
08_reporting/                     number dump of all artifacts, tiled appendix figures
archive/                          diagnostics that are not part of the pipeline
```

Every stage folder keeps its scripts flat, next to a `figures/` folder with the PNG outputs.
`tum_colors.py` and `regime_style.py` are shared style modules; each stage has its own copy.

## The pipeline

| Stage | Folder | Scripts, in run order | Reads | Writes |
|---|---|---|---|---|
| 1 Survey and scoring | `01_survey_and_scoring` | `total_dataset_analysis/00_cleaning_total_dataset.py`, then `01a` to `10`; `filtered_dataset_analysis/00_cleaning_filtered_dataset.py`, then `01` to `16` (Qs `14`, Ds `15`, DSS `16`) | survey CSVs in the stage folder | `*_cleaned.csv`, `*_figures/*.png` |
| 2 Acquisition and extraction | `02_acquisition_and_extraction` | no code in this repository | native dataset files | `DATA_ROOT/parquet_data/<dataset>/*.parquet` |
| 3 Standardization | `03_standardization` | `pipeline_regression_test.py`, `standardize_inertial_data.py`, `compute_std.py` | `parquet_data` | `inertial_data/<dataset>/clip_XXXX.parquet` (43 columns, 10 Hz), `std_data/<dataset>.parquet` (one row per 20 s window) |
| 4 Exploratory analysis | `04_exploratory_analysis` | runner `run_pipeline.py` (also runs stage 3): `coverage.py`, `coverage_baseline.py`, `regime_classification.py`, `distributional_analysis.py`, `trajectory_fan.py`, `phase_potrait.py`, `ridgeline.py`, `density_band_analysis.py`, `boxplot_comparison.py`, `ecdf_analysis.py`, `time_series_analysis.py`, `heatmap_summary.py` | `inertial_data`, `std_data` | `audit_baseline.json`, `figures/<type>/` |
| 5 Regime discovery | `05_regime_discovery` | runner `regime_run_pipeline.py`: `feature_gate.py`, `clustering.py`, `cap_sensitivity.py`, `validity.py`, `gmm.py`, `hmm.py`, then the figure scripts (including `reseed_stability.py` and `resample_stability.py`) | `std_data`, `inertial_data` (HMM) | `regime_data/*.parquet`, `figures/` |
| 6 Bias quantification | `06_bias_quantification` | runner `bias_run_pipeline.py`: `bias_quant.py`, five tier-1 figure scripts, `bias_diversity.py`, `bias_volume.py`, `bias_bootstrap.py`, `bias_gmm_entropy.py`, `bias_shape_metrics.py`, `bias_regime_supply.py`, `bias_fingerprint.py` | `regime_data`, `std_data` | `regime_data/bias_*.parquet`, `figures/` |
| 7 Subset curation and evaluation | `07_subset_curation` | `window_summary.py`, `run_sampling.py` (two tiers), runner `curation_run_pipeline.py` (stale audit, event audit, build, eval, gates, feasibility, replicate, reveal, render, dossier, pca), `curation_evassive_check.py`, runner `subset_run_pipeline.py`, `canonical.py --check` | `regime_data`, `std_data`, `inertial_data` | `subset_data/` (subsets, manifests, `eval/` caches), `figures/curation/`, `figures/subset_eval/` |
| Reporting | `08_reporting` | `general_info.py`, `compile_grids.py` | all artifacts, stage 7 figures | `general_info.txt`, `compiled_figures/` |

`subset_eval_harness.py` and `subset_frame_pass.py` are run through `curation_eval.py` and
`curation_render_winner.py` (or `subset_run_pipeline.py --frame-pass`); they are not started
on their own. The runners stop at the first failing script. The runners of stages 4 to 6
accept `--dry-run`, `--from <stage>`, `--only <stage>`, `--skip <stage>` and `--keep-going`;
`curation_run_pipeline.py` accepts `--from`, `--skip`, `--frame-pass` and `--dossier`, and
`subset_run_pipeline.py` accepts `--from` and `--frame-pass`.

## Data

### The fourteen datasets

The data are not part of this repository and must be obtained from the providers under their
own terms of use. Access routes are described in Section 4.2.1 and the datasets excluded at
acquisition in Section 4.2.2 of the thesis.

| Folder key | Dataset | Source | Access | Native ego-motion source |
|---|---|---|---|---|
| `4seasons` | 4Seasons | https://www.4seasons-dataset.com | open | `imu.txt` and `GNSSPoses.txt` |
| `a2d2` | A2D2 | https://www.a2d2.audi | open | bus-signal JSON |
| `boreas_rt` | Boreas Road Trip | https://www.boreas.utias.utoronto.ca, paper https://arxiv.org/abs/2602.16870 | open | `applanix/gps_post_process.csv` |
| `durlar` | DurLAR | https://github.com/l1997i/DurLAR | registration (Hugging Face) | IMU and GPS CSV from the tar parts |
| `ford` | Ford Multi-AV Seasonal | https://avdata.ford.com | open | ROS bag |
| `hdd` | Honda Research Institute Driving Dataset | https://usa.honda-ri.com/hdd | access request | per-signal timestamped files |
| `kitscenes` | KITScenes | https://arxiv.org/abs/2606.02956 | registration (Hugging Face) | `poses.txt` |
| `kitti` | KITTI (raw data) | https://www.cvlibs.net/datasets/kitti/raw_data.php | registration | OXTS text |
| `kitti360` | KITTI-360 | https://www.cvlibs.net/datasets/kitti-360 | registration | OXTS text |
| `nuscene` | nuScenes (CAN bus expansion) | https://www.nuscenes.org/nuscenes | registration | JSON (`ms_imu`, `pose`, `ego_pose`) |
| `nvidia` | NVIDIA PhysicalAI Autonomous Vehicles (v26.03) | https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles | registration (Hugging Face) | `egomotion.offline` parquet |
| `oxford` | Oxford RobotCar | https://robotcar-dataset.robots.ox.ac.uk | registration | `rtk.csv` |
| `waymo` | Waymo Open Motion Dataset (`training_20s`) | https://waymo.com/open/data/motion | registration | TFRecord, one scenario per clip |
| `zod` | Zenseact Open Dataset (drives) | https://zod.zenseact.com | registration | `oxts.hdf5` |

### Data root and layers

All scripts find the data through `thesis_paths.py`. `DATA_ROOT` defaults to a folder named
`dataset` next to the repository and is set with the environment variable `THESIS_DATA_ROOT`:

```bash
export THESIS_DATA_ROOT=/path/to/dataset
```

| Layer under `DATA_ROOT` | Written by | Size | Needed for |
|---|---|---:|---|
| `parquet_data/<dataset>/` | stage 2 (not in this repository) | 13 GB | stage 3 |
| `inertial_data/<dataset>/clip_XXXX.parquet` | `standardize_inertial_data.py` | 31 GB | frame-level figures of stage 4, the HMM, `window_summary.py`, `subset_frame_pass.py` |
| `std_data/<dataset>.parquet` | `compute_std.py` | 34 MB | stages 4 to 7 |
| `regime_data/*.parquet` | stages 5 and 6 | 28 MB | stages 5 to 7 |
| `subset_data/` (subsets, manifests, `eval/`) | stage 7 | 289 MB | stage 7 figures, `08_reporting` |
| `canonical_snapshot.json` | `canonical.py --freeze` | 4 KB | `canonical.py --check` |

With `std_data`, `regime_data`, `subset_data` and `canonical_snapshot.json` (about 350 MB) every
number of stages 5 to 7 can be checked and every figure of stages 5 to 7 re-rendered without the
raw data. The frame-level stages need `inertial_data` (31 GB).

The extraction from the native formats into `parquet_data` (stage 2, Section 4.2.3) was done
per dataset outside this code base and is not included. Each file in `parquet_data/<dataset>/`
holds one clip (one contiguous recording; one scenario for Waymo) at the native rate, with a
nanosecond timestamp column and whichever of position, velocity, acceleration and attitude
columns the dataset provides. The loaders in `standardize_inertial_data.py` document the columns
read per dataset.

## Environments

Two Python 3.12.2 environments produced the thesis results. The artifacts record which one
wrote them (the pandas version in the parquet metadata, the matplotlib version in the PNG
metadata).

| File | Produced |
|---|---|
| `requirements-standardization.txt` (numpy 1.26.4, pandas 2.3.2, scikit-learn 1.5.1, matplotlib 3.10.0) | `inertial_data`, `feature_gate.py`, `cap_sensitivity.py`, the figures of `01_survey_and_scoring` and most figures of `04_exploratory_analysis` |
| `requirements.txt` (numpy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, matplotlib 3.10.9, hmmlearn 0.3.3) | `std_data`, every other artifact of stages 5 to 7, and their figures |

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Figures re-rendered in the other environment show the same content with small anti-aliasing
differences, so a byte-level comparison of PNGs needs the environment that produced them.

## Running the pipeline

The commands below run the full pipeline from `parquet_data`. Every script can be started from
any working directory; the runners start their scripts in the script folder. Durations are for
an Apple M4 Pro (14 cores, 48 GB). Stages marked "measured" were re-run for this handover;
the others are taken from the file timestamps of the last complete run.

| Step | Command | Duration |
|---|---|---|
| Survey figures | `for f in 01_survey_and_scoring/total_dataset_analysis/[0-9]*.py 01_survey_and_scoring/filtered_dataset_analysis/[0-9]*.py; do python "$f"; done` | 20 s, measured |
| Regression test | `python 03_standardization/pipeline_regression_test.py` | 1 s, measured |
| Stages 3 and 4 | `python 04_exploratory_analysis/run_pipeline.py` | standardization about 1.5 h, windowing about 15 min, exploratory figures up to 1 h (the frame-level figures read all 371,425 clips) |
| Stage 5 | `python 05_regime_discovery/regime_run_pipeline.py` | about 1 h (HMM about 15 min, reseed and resample stability about 12 min) |
| Stage 6 | `python 06_bias_quantification/bias_run_pipeline.py` | about 1 min |
| Stage 7, frame summaries | `python 07_subset_curation/window_summary.py` | about 10 min |
| Stage 7, sampling | `python 07_subset_curation/run_sampling.py` and `python 07_subset_curation/run_sampling.py --n 71390` | 1 to 4 min each |
| Stage 7, curation | `python 07_subset_curation/curation_run_pipeline.py --frame-pass --dossier` | about 30 min (frame pass 10 to 40 min) |
| Stage 7, evasive retention | `python 07_subset_curation/curation_evassive_check.py` | under 1 min |
| Stage 7, evaluation figures | `python 07_subset_curation/subset_run_pipeline.py` | 2 min, measured |
| Snapshot check | `python 07_subset_curation/canonical.py --check` | 2 s, measured |
| Number dump and tiled figures | `python 08_reporting/general_info.py` and `python 08_reporting/compile_grids.py` | 5 s, measured |

The order inside stage 7 matters: `curation_run_pipeline.py` evaluates M01 to M11 and C01 to C03
in one pass and writes the evaluation caches that `subset_run_pipeline.py` renders.
`subset_run_pipeline.py` refuses to start if a cache is older than
`regime_data/cluster_assignments.parquet`.

Scripts that only read existing artifacts and write figures can be run on their own, for
example `python 06_bias_quantification/bias_stacked_bar.py`.

## Frozen constants

| Constant | Value | Defined in code | Thesis section |
|---|---|---|---|
| Grid rate | 10 Hz | `standardize_inertial_data.py` `GRID_HZ` | 4.3.1, 4.3.5 |
| Anti-alias low-pass | 4th-order Butterworth, 3.0 Hz, zero phase | `F_CUT_HZ`, `BUTTER_ORDER` | 4.3.5 |
| Gap policy | interpolate gaps up to 0.5 s, mask longer gaps | `GAP_MAX_S` | 4.3.5 |
| Hampel filter | 0.5 s window, 3 scaled MADs | `HAMPEL_WINDOW_S`, `HAMPEL_T0` | 4.3.4 |
| Savitzky-Golay derivatives | order 3, 7 frames (first), 11 frames (second) | `SG_POLY`, `W_D1`, `W_D2` | 4.3.6 |
| Physical validity bounds | speed 200 km/h, acceleration 15 m/s², jerk 30 m/s³, yaw rate 90 °/s | `SPEED_MAX_MS`, `A_MAX_MS2`, `J_MAX_MS3`, `YAW_RATE_MAX_RADS` | 4.3.4, 4.3.6, 4.5.4 |
| Window | 20 s, step 20 s (non-overlapping), 200 frames | `compute_std.py` `WINDOW_S`, `STEP_S` | 4.5.1 |
| Minimum validity per window | 50 % of the expected frames, at least 2 | `MIN_VALID_FRAC`, `MIN_VALID_ABS` | 4.5.4 |
| Variability tier cuts | σ speed 1 / 5 / 15 km/h, σ acceleration 0.2 / 0.5 / 1.5 m/s², σ yaw 1 / 5 / 15 °/s | `regime_classification.py` `THRESHOLDS` (copies in `ecdf_analysis.py`, `time_series_analysis.py`, `heatmap_summary.py`) | 4.5.6 |
| Feature screen | abs. Pearson or Spearman above 0.85, VIF above 5, VIF tie margin 0.5 | `feature_gate.py` | 4.7.2 |
| Fit cap | 5,000 windows per dataset, 10 intensity strata | `clustering.py` `CAP`, `N_STRATA` | 4.7.3 |
| k-means | k from 2 to 10 by maximum silhouette (k = 5), 10 restarts, seed 42 | `K_RANGE`, `N_INIT`, `SEED` | 4.7.5 |
| Stability seeds | reseeds 43 to 62, fit-population redraws 242 to 261 | `reseed_stability.py`, `resample_stability.py` | 4.7.7 |
| Clip bootstrap | B = 1,000, seed 42, 95 % interval | `bias_bootstrap.py` | 4.8.10 |
| Hard-deceleration trigger | 0.6 g | `run_sampling.py` `DECEL_TRIGGER_G`, `subset_eval_harness.py` `HARD_DECEL_G` | 4.9.8, 4.11.5 |
| Sampling seeds | method i uses the generator (42, i); curation methods continue with 12 to 14 | `run_sampling.py`, `curation_common.py` `C_SEED_INDEX` | 4.9, 4.11 |
| Budget tier | N = 71,390 (supply of the rarest regime), capability tier 356,950 | derived in `canonical.py` | 4.9.3 |
| Replicate null | K = 100 random draws, seeds 1,000 + k | `subset_eval_harness.py` `K_ENVELOPE`, `ENVELOPE_SEED_BASE` | 4.10.4 |
| Selection gates | JS to uniform ≤ 0.005, top-source share ≤ 0.500, hard-deceleration retention ≥ 1.0, near-stationary retention ≥ 0.5; tie-break on the within-regime speed-variability shift | `curation_common.py` `GATES`, `TIE_BREAK` | 4.11.1, 4.12.1 |
| Per-regime source cap | 0.50 | `curation_common.py` `SOURCE_CAP_FRAC` | 4.11.6 |
| Evasive threshold | 0.75 g implied lateral acceleration | `curation_common.py` `EVASIVE_LAT_G` | 4.11.7 |
| Seed replication | R = 20, seeds (42, 1,200 + r) | `curation_replicate.py` | 4.12.4 |
| DSS | 0.60 Qs + 0.25 Ds + 0.15 Ps, admission at DSS ≥ 0.675 | `16_dss_score.py` (threshold applied in the survey table) | 4.1.3, 4.1.4 |

## Checks

**Regression test of the standardization.** `03_standardization/pipeline_regression_test.py`
plants known defects into synthetic clips (rectangular and raised-cosine braking pulses, speed
spikes, timestamp gaps, missing-value runs, yaw wrapping, short clips, position-only input,
non-monotonic timestamps) and asserts how `build_standard_frame` handles them (Section 4.4.6).
It needs no data.

```bash
python 03_standardization/pipeline_regression_test.py
```

Expected end of the output:

```
T1a planted -5.0 m/s^2 rectangular brake -> measured peak -5.263 m/s^2 (deviation +5.3 %)
...
T10 monotonic/dt filters dropped 3 planted bad frames; grid dt uniform at 0.1 s

ALL TESTS PASSED
```

The test covers the standardization only. For the later stages,
`python 07_subset_curation/canonical.py --check` recomputes the corpus quantities (frame size,
guarded frame, regime supplies, budget N, source shares, event reserve, evasive population) from
the artifacts and compares them with `DATA_ROOT/canonical_snapshot.json`.

## Where each figure and table is produced

Paths are relative to the stage folder named in the script column. Figures of the same kind are
grouped. "console" means the script prints the table.

### Chapter 5

| Thesis | Script | Output |
|---|---|---|
| Fig. 5.1 | `01_survey_and_scoring/total_dataset_analysis/01d_dataset_to_year_total.py` | `total_dataset_analysis_figures/01d_dataset_to_year_total.png` |
| Fig. 5.2 | `…/total_dataset_analysis/08_vehicle_info.py` | `total_dataset_analysis_figures/08_vehicle_info.png` |
| Fig. 5.3 | `…/filtered_dataset_analysis/08b_zoomed_citation.py` | `filtered_dataset_analysis_figures/08b_citation_zoom.png` |
| Fig. 5.4 | `…/filtered_dataset_analysis/16_dss_score.py` | `filtered_dataset_analysis_figures/16a_dss_bar.png` |
| Table 5.1 | `16_dss_score.py` (console), survey table | `dataset_survey_table/4_dataset_scoring.csv` |
| Fig. 5.5, B.1, B.2 | `04_exploratory_analysis/coverage.py` | `figures/coverage/signal_coverage.png`, `non_zero.png`, `boundary.png`, `coverage_matrices.csv` |
| Table 5.2 | `coverage_baseline.py`, `08_reporting/general_info.py` | `audit_baseline.json`, `general_info.txt` |
| Fig. 5.6, 5.7, C.1 to C.4 | `distributional_analysis.py` | `figures/distribution/hist_grid_{speed,lon_accel,lat_accel,lon_jerk,lat_jerk,yaw_rate}.png`, `distribution_grid_stats.csv` |
| Fig. 5.8, C.6, C.7 | `ridgeline.py` | `figures/ridgeline/ridgeline_{speed,acc_mag,yaw}.png` |
| Fig. 5.9, C.8, C.9 | `density_band_analysis.py` | `figures/density_band/density_bands_grid_{acc_mag,speed_kmh,yaw_rate}.png`, `density_band_stats.csv` |
| Fig. 5.10, 5.11 | `trajectory_fan.py` | `figures/trajectory_fans/trajectory_fan_xy_grid_w20.png`, `speed_change_fan_grid_w20.png` |
| Fig. 5.12, C.10, Table C.1 | `phase_potrait.py` | `figures/phase_portrait/phase_portrait_grid_{speed_along,speed_yaw}.png`, `phase_portrait_disclosure.csv` |
| Table 5.3, Fig. C.15 to C.17 | `regime_classification.py` | `figures/regime/regime_{speed_kmh,acc_mag,yaw_rate}.png`, tier shares in `general_info.txt` |
| Fig. 5.13, C.13, C.14 | `ecdf_analysis.py` | `figures/ecdf/ecdf_{speed_kmh,acc_mag,yaw_rate}.png` |
| Fig. C.5, C.11, C.12 | `boxplot_comparison.py` | `figures/boxplot/boxplot_{speed,acc_mag,yaw_rate}.png` |
| Fig. 5.14 | `heatmap_summary.py` | `figures/summary/heatmap_summary.png`, `heatmap_summary.csv` |
| Table 5.4, Fig. D.1 | `05_regime_discovery/feature_gate.py` | `regime_data/feature_gate_{stats,correlations}.parquet`, `figures/feature_gate/feature_correlation_gate.png` |
| Fig. 5.15, Table 5.5 | `clustering.py`, `k_select_curve.py` | `regime_data/kselect_curves.parquet`, `figures/model_selection/kselect_curve.png` |
| Fig. 5.16, Table 5.6 | `clustering.py`, `regime_heatmap.py` | `regime_data/cluster_profile.parquet`, `figures/regime_profile/regime_profile_heatmap.png` |
| Fig. 5.17, D.2 | `pca_scatter_labeled.py`, `pca_scatter_preclustering.py` | `figures/pca/pca_scatter_labelled.png`, `pca_scatter_preclustering.png` |
| Fig. 5.18, D.4 | `resample_stability.py`, `reseed_stability.py` | `figures/cap_sensitivity/resample_stability.png`, `reseed_stability.png` |
| Table 5.7, Fig. D.5 | `cap_sensitivity.py` | `regime_data/cap_sensitivity_*.parquet`, `figures/cap_sensitivity/cap_sensitivity_ari.png` |
| Tables 5.8, 5.9, Fig. D.6 to D.8 | `validity.py`, `validity_indices.py`, `regime_stability.py`, `consensus_matrix.py` | `regime_data/validity_indices.parquet`, `bootstrap_jaccard.parquet`, `consensus_*.parquet`, `figures/validity/` |
| Table 5.10, Fig. D.9 | `gmm.py`, `gmm_selection.py` | `regime_data/gmm_selection.parquet`, `figures/gmm/gmm_selection.png` |
| Fig. 5.19, Table 5.11 | `gmm_kmeans_correspondence.py` | `figures/gmm/kmeans_gmm_correspondence.png` |
| Fig. 5.20, 5.21 | `membership_entropy.py`, `covariance_contours.py` | `figures/gmm/membership_entropy.png`, `covariance_contours.png` |
| Tables 5.12 to 5.14, Fig. 5.22 | `hmm.py`, `state_timeline.py` | `regime_data/hmm_{alignment,transitions,durations}.parquet`, `figures/hmm/state_timeline.png` |
| Table 5.15 | `feature_gate.py`, `hmm.py` | `regime_data/feature_gate_population.parquet`, `hmm_run_summary.parquet` |
| Fig. 5.23, Tables 5.16, 5.17 | `06_bias_quantification/bias_quant.py`, `bias_stacked_bar.py` | `regime_data/bias_per_dataset.parquet`, `bias_corpus_profiles.parquet`, `figures/bias_stacked_bar.png` |
| Fig. 5.24, E.1 | `bias_regime_grid.py`, `bias_share_heatmap.py` | `figures/bias_regime_grid.png`, `bias_share_heatmap.png` |
| Fig. 5.25, E.2, Table 5.18 | `bias_diversity.py` | `figures/hill_profiles.png`, `psi_thresholds.png`, `regime_data/bias_diversity.parquet` |
| Fig. 5.26, 5.27, Table 5.19 | `bias_balance_metrics.py`, `bias_wasserstein.py`, `bias_shape_metrics.py` | `figures/bias_balance_metrics.png`, `bias_wasserstein.png`, `regime_data/bias_shape.parquet` |
| Fig. 5.28, 5.29 | `bias_shape_metrics.py`, `bias_volume.py` | `figures/dynamic_lorenz.png`, `lorenz_volume.png` |
| Fig. 5.30, Table 5.20 | `bias_gmm_entropy.py` | `figures/gmm_entropy.png`, `regime_data/bias_gmm_entropy.parquet` |
| Fig. 5.31, E.3, Table 5.21 | `bias_bootstrap.py` | `figures/low_dyn_share_ci.png`, `js_to_pooled_ci.png`, `regime_data/bias_bootstrap_ci.parquet` |
| Fig. 5.32, Table E.1 | `bias_fingerprint.py` | `figures/bias_fingerprint.png`, `regime_data/bias_metric_agreement.parquet` |
| Fig. E.4 | `bias_regime_supply.py` | `figures/bias_regime_supply.png` |
| Tables 5.22 to 5.24, F.1 | `07_subset_curation/run_sampling.py` | `subset_data/supply_table.csv`, `allocation_table.csv`, `manifest.json` (both tiers) |
| Tables 5.25, 5.26, F.2 | `curation_eval.py` (runs `subset_eval_harness.py`) | `subset_data/eval/master_table_{budget,capability}.csv`, `tautology_map.csv` |
| Fig. 5.33 to 5.41 | `subset_ecdf_grid.py`, `subset_stacked_bar.py`, `subset_mean_shift.py`, `subset_wasserstein_heatmap.py`, `subset_source_composition.py`, `subset_event_retention.py`, `subset_bias_axes.py`, `subset_fingerprint.py` | `figures/subset_eval/subset_*.png` of the same names (5.36 `subset_wasserstein_inmodel.png`, 5.37 `subset_wasserstein_heldout.png`) |
| Table 5.27 | written by hand in the thesis | |
| Fig. 5.42, Table 5.28 | `curation_feasibility.py` | `figures/curation/curation_feasibility.png`, console |
| Table 5.29 | `curation_build.py` | `subset_data/curation_manifest.json` (both tiers) |
| Tables 5.30, 5.31 | `curation_gates.py` | console |
| Fig. 5.43, 5.44 | `curation_render_winner.py` | `figures/curation/curation_winner_regime_profile.png`, `curation_winner_source_composition.png` |
| Table 5.32 | `curation_replicate.py` | `subset_data/eval/curation_replication_{budget.parquet,meta.json}` |
| Tables 5.33, 5.34 | `curation_reveal.py` | console |
| Fig. 5.45, 5.46 | `curation_winner_dossier.py` | `figures/curation/curation_winner_fidelity.png`, `curation_winner_w1_departure.png` |
| Table 5.35 | `curation_evassive_check.py` | console |
| Fig. 5.47 | `curation_render_winner.py` (renders `subset_ecdf_finalist.py` with C01) | `figures/curation/subset_ecdf_finalists.png` |

Tables 4.1 and 4.2 are written by hand in the thesis.

### Appendices

| Thesis | Script | Output |
|---|---|---|
| Fig. A.1 to A.5 | `01_survey_and_scoring/total_dataset_analysis/02_…`, `03_…`, `04a_…`, `04b_…`, `05_…` | `total_dataset_analysis_figures/02_cumulative_datasets.png`, `03_avg_sensors_per_year.png`, `04a_recording_country_map.png`, `04b_recording_country_bar.png`, `05_issuing_country_bar.png` |
| Fig. A.6, A.7 | `07a_cameras.py`, `07b_sensors.py` | `07a_sensor_cameras.png`, `07b_sensor_others.png` |
| Fig. A.8 to A.14, A.19, A.20 | `filtered_dataset_analysis/06_weather.py`, `13_road_type.py`, `12_imu_frequency.py`, `08a_full_citation.py`, `14_qs_score.py`, `15_ds_score.py`, `16_dss_score.py`, `04_task_support.py`, `10_imu_grade.py` | `filtered_dataset_analysis_figures/06_weather.png`, `13_road_type.png`, `12_imu_frequency.png`, `08a_citation_full.png`, `14_qs_score.png`, `15_ds_score.png`, `16b_dss_heatmap.png`, `04_task_support.png`, `10_imu_grade.png` |
| Fig. A.15 to A.18 | tiles of `01a`, `01b`, `01c`; of `06`, `09`, `10` (pool pies); of `05`, `09`, `11` (candidate pies); of `01`, `02`, `03`, `07` (candidate counts) | `08_reporting/compiled_figures/01abc_dataset_to_year_by_type.png`, `pool_pies_grid.png`, `candidate_pies_grid.png`, `candidate_counts_grid.png` (see Known issues) |
| Fig. B.1, B.2 | `04_exploratory_analysis/coverage.py` | see Fig. 5.5 |
| Fig. C.18, C.19 | tiles of `time_series_analysis.py` output `figures/time_series/<dataset>_speed_kmh.png` | `08_reporting/compiled_figures/tier_timeseries_grid_{1,2}.png` (see Known issues) |
| Fig. D.3 | tile of `pca_scatter_preclustering_3d.png` and `pca_scatter_3d.png` (`pca_scatter_preclustering_3D.py`, `pca_scatter_labeled_3D.py`) | `08_reporting/compiled_figures/pca_scatter_3d_pair.png` (see Known issues) |
| Fig. D.10 | tile of `transition_matrix.png` and `state_durations.png` (`transition_matrix.py`, `state_duration.py`) | `08_reporting/compiled_figures/hmm_matrix_dwell_pair.png` (see Known issues) |
| Fig. F.1 | `07_subset_curation/subset_regime_grid.py` | `figures/subset_eval/subset_regime_grid.png` |
| Fig. F.2 to F.7 | `subset_histogram.py` | `figures/subset_eval/subset_hist_grid_{speed,lon_accel,lat_accel,lon_jerk,lat_jerk,yaw_rate}.png` |
| Fig. F.8 to F.10 | `subset_boxplot.py` | `figures/subset_eval/subset_boxplot_{speed,acc_mag,yaw_rate}.png` |
| Fig. F.11 to F.14 | `subset_ridgeline.py`, `subset_density_grid.py`, `subset_pca_grid.py` | `figures/subset_eval/subset_ridgeline.png`, `subset_density_grid.png`, `subset_pca_grid.png`, `subset_pca_grid_regime.png` |
| Fig. F.15 to F.19 | `subset_fans.py`, `subset_phase_potrait.py`, `subset_time_series.py`, tiled by `08_reporting/compile_grids.py` | `08_reporting/compiled_figures/dossier_{trajectory_fans,speed_change_fans,portraits_speed_along,portraits_speed_yaw,exemplar_traces}.png` |
| Fig. G.1 to G.5 | `curation_winner_dossier.py` | `figures/curation/curation_winner_{hist_grid,ecdf_grid,phase_portraits,trajectory_fan,speed_fan}.png` |
| Fig. G.6 | `curation_winner_pca.py` | `figures/curation/curation_winner_pca.png` |

The per-dataset and per-method single figures that the grids are built from (84 histograms,
14 density bands, 28 fans, 28 phase portraits, 14 tier time series, and the per-method
singles in `figures/subset_eval/{hist,fans,phase,timeseries}/`) stay in the figure folders.

## Known issues

These are documented and left unchanged, because every change would alter a thesis output.

- **Evaluation caches.** A standalone run of `subset_eval_harness.py` evaluates M01 to M11 only
  and writes caches without C01 to C03, after which `curation_gates.py` and
  `curation_winner_dossier.py` stop with an error. Always build the caches with
  `curation_eval.py` (or `curation_run_pipeline.py --from eval`).
- **Tiled appendix figures without a script.** Fig. A.15 to A.18, C.18, C.19, D.3 and D.10 are
  tiles of existing single figures that were assembled outside this code base. The tiles are kept
  in `08_reporting/compiled_figures/`, the singles they contain are listed in the figure table.
- **Survey figure filters.** `03_avg_sensors_per_year.py` combines its conditions as
  `(real and original) or extension`, so extension datasets of every type are counted.
  `08b_zoomed_citation.py` lists "Waymo Open Dataset", which does not match the name in the
  survey table, so only five datasets are excluded from the zoomed view.
- **Duplicated constants.** The tier cuts are defined in four files, gravity is 9.81 m/s² in
  `run_sampling.py` and 9.80665 m/s² in `curation_common.py`, and the two copies of
  `regime_style.py` differ (the copy in `06_bias_quantification` also defines `N_REGIMES` and
  `LOW_DYNAMIC_REGIMES`). `canonical.py` adds the stage folders to `sys.path`, so a script that
  imports it may load a style module from another stage folder.
- **`canonical.py --check` on another machine.** The snapshot stores the absolute path of the
  budget-tier folder, so `budget_dir` is reported as changed when `DATA_ROOT` differs.
- **Console text.** Runtime messages and some figure titles use the working numbering of the
  analysis (Chapters 7 to 13) instead of the thesis numbering. Section 4 of the thesis
  corresponds to Chapters 7 to 13 of that numbering.
- **Regime runner.** `regime_run_pipeline.py` lists `covariance_overlay.py` and
  `regime_heatmap_sweep.py` (moved to `archive/`) and skips both. Its closing hand-check message
  quotes PCA variance shares that differ slightly from `pca_variance.parquet` (37.7, 59.4,
  78.9 % cumulative).

## Citation

```bibtex
@thesis{tamin2026dynamicbias,
  author      = {Tamin, Edward Anderson},
  title       = {Identifying Dynamic Biases in Large-Scale Public Datasets via Distributional Analysis},
  type        = {Bachelor's thesis},
  institution = {Technical University of Munich, TUM School of Engineering and Design,
                 Professorship of Autonomous Vehicle Systems},
  address     = {Garching, Germany},
  year        = {2026},
  month       = oct
}
```

`CITATION.cff` holds the same reference in machine-readable form.
