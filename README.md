# Identifying Dynamic Biases in Large-Scale Public Datasets via Distributional Analysis

Bachelor thesis, Technical University of Munich, Professorship of Autonomous Vehicle
Systems. This repository holds the analysis and plotting code of a kinematic
(ego-motion) analysis pipeline across 14 public autonomous-driving datasets. The pipeline
quantifies the low-dynamic skew of the corpus (RQ1), recovers interpretable driving
regimes by unsupervised clustering (RQ2), and constructs a dynamically balanced curated
subset (RQ3).

The repository contains code, figures, and the survey tables only. The data lives in a
separate tree (`~/Downloads/dataset`, 742,973 files, 43.2 GB, see
[Data layout](#data-layout)) and is not version-controlled.

**Status: the analysis is closed (2026-09-06). No further analysis will be run, the
write-up is in progress.** The last accepted end-to-end run was executed from cleared
outputs on the v4_4 standardization and the non-overlapping 20 s window grid, and every
stage completed green. `general_info.txt` (regenerated 2026-08-30 02:10 directly from
the artifacts) is the numeric ground truth for every number quoted below.

The frozen oracle `dataset/canonical_snapshot.json` was lost after the original freeze
of 2026-08-30 and has been rebuilt, carrying `frozen_on: 2026-09-07`. It is an anchor
for future drift checks rather than a record of the closed run, but the values it holds
were verified before it was written: `general_info.py` was regenerated on 2026-09-07 and
every constant `canonical.py` derives live from the artifacts matched its 2026-08-30
value, so nothing drifted in between. `canonical.py --check` runs again, and every
number in this file is reproducible from the artifacts today.

## Research questions

- **RQ1.** Quantify the dynamic skew (near-stationary and low-speed bias) of the 14-dataset corpus, per dataset and pooled.
- **RQ2.** Recover interpretable driving regimes from per-window kinematics by unsupervised clustering, and verify that the regimes are stable.
- **RQ3.** Construct a dynamically balanced curated subset that corrects the skew under pre-registered rules, without discarding measurement data.

Each of the four analysis modules serves one question, and the runners are ordered so
that no stage reads an artifact that a later stage would have to regenerate.

## Headline numbers (frozen)

| Quantity | Value | Source |
|---|---|---|
| Datasets | 14 (4Seasons, A2D2, Boreas, DurLAR, Ford, HDD, KITScenes, KITTI, KITTI-360, nuScenes, NVIDIA, Oxford, Waymo, ZOD) | `run_pipeline.py` DATASETS |
| Raw clip files (`parquet_data/`) | 371,430 parquet files, one per clip, 178,778,906 rows at native rate | parquet footers, 2026-09-07 |
| Standardized clips (`inertial_data/`) | 371,425 clips, 81,252,848 frames on the 10 Hz grid | `audit_baseline.json` |
| Standardization | `v4_4_10hz_common_grid_validity_gated`, 3 Hz band limit, 10 Hz grid, validity gated | `standardize_inertial_data.py` |
| Window layer (`std_data/`) | 406,130 windows of 20 s, non-overlapping (D-2) | `compute_std.py` |
| Six-feature complete case (gate population) | 402,293 windows | `feature_gate.py`, verified 2026-09-07 |
| Five-feature complete case (assign set, the frame R0) | 402,421 windows | `cluster_assignments.parquet` |
| Guarded frame R1 | 401,230 windows, the frame minus A2D2, 4Seasons, and DurLAR | `subset_eval_harness.py` |
| Regimes | k = 5 k-means, seed 42, fit cap 5,000 windows per dataset | `clustering.py` |
| Regime supplies (0 to 4) | 89,420 / 89,499 / 78,883 / 73,229 / 71,390 | `cluster_assignments.parquet` |
| Budget tier N | 71,390 (the rarest regime, Accelerating) | `canonical.py` |
| Capability tier N | 356,950 (5 x 71,390), 88.7 % of the frame | `canonical.py` |
| Event reserve (0.6 g hard deceleration on the guarded frame) | 1,088 windows (0.27 % of the frame) | `curation_event_audit.py` |
| Evasive population (D24, 0.75 g implied lateral) | 2,294 windows (0.57 % of the frame) | `curation_event_audit.py` |
| Top source | NVIDIA, 74.13 % of the frame (top two sources 91.66 %) | `bias_volume.parquet` |
| Structural zeros | 1 (Oxford has no Highway Cruising window) | `bias_per_dataset.parquet` |
| Curation winner | C01, sole survivor of the D4 gates (0.0000 / 0.5000 / 5.6369 / 0.9873), 20 of 20 replicates pass, tie-break 0.116 km/h with standard deviation 0.027 km/h | `curation_gates.py`, `curation_replicate.py` |

`bachelor-arbeit/general_info.py` reads every artifact and writes `general_info.txt`
(1,408 lines, 15 sections). `subset_curation/canonical.py` derives the canonical
constants live from the artifacts and verifies them against the frozen snapshot
(`python canonical.py --check`). Neither script contains a typed number.

## The 14 datasets

Folder keys are the names used in every path, column, and figure file. The native source
column follows section 6.2 of `Thesis_Personal_Methodology_Notes.md`, since the pipeline
itself starts from `parquet_data/` and the native format never enters the code.
Clip and frame counts come from `audit_baseline.json`, window counts from
`general_info.txt` section 4, assigned windows and volume shares from
`bias_per_dataset.parquet`.

| Folder key | Name | Native source | Raw clips | Native rows | Std. clips | 10 Hz frames | std_data windows | Assigned windows | Speed source | Yaw source | Mean speed km/h | Frames gated % | Confounded accel |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|
| `4seasons` | 4Seasons | imu.txt and GNSSPoses.txt | 30 | 31,243,319 | 30 | 156,244 | 766 | 707 | DERIVED | RAW | 32.4 | 0.00 | yes |
| `a2d2` | A2D2 | bus-signal JSON | 3 | 438,131 | 3 | 21,910 | 108 | 106 | RAW | RAW | 17.2 | 0.00 | yes |
| `boreas_rt` | Boreas | gps_post_process.csv | 110 | 23,262,888 | 109 | 1,155,293 | 5,721 | 5,721 | RAW | RAW | 30.4 | 0.00 | no |
| `durlar` | DurLAR | imu and gps CSV from tar parts | 4 | 1,074,550 | 3 | 81,220 | 404 | 378 | DERIVED | RAW | 33.8 | 7.74 | yes |
| `ford` | Ford | ROS bag | 29 | 4,574,945 | 29 | 245,447 | 1,212 | 1,205 | RAW | RAW | 48.1 | 0.00 | no |
| `hdd` | HDD | per-signal timestamped files | 137 | 38,844,425 | 137 | 3,885,386 | 19,361 | 15,752 | RAW | DERIVED | 27.5 | 0.00 | no |
| `kitscenes` | KITScenes | poses.txt (8 fields) | 1,006 | 181,501 | 1,006 | 181,501 | 1,040 | 1,040 | DERIVED | DERIVED | 26.8 | 0.00 | no |
| `kitti` | KITTI | OXTS text | 156 | 47,962 | 156 | 49,774 | 300 | 297 | DERIVED | RAW | 26.7 | 0.00 | no |
| `kitti360` | KITTI-360 | OXTS text | 10 | 940,262 | 10 | 94,074 | 466 | 466 | DERIVED | RAW | 28.2 | 0.00 | no |
| `nuscene` | nuScenes | JSON (ms_imu, pose, ego_pose) | 979 | 1,906,954 | 979 | 192,275 | 979 | 977 | RAW | RAW | 18.3 | 0.00 | no |
| `nvidia` | NVIDIA | egomotion.offline parquet (pose and quaternion only) | 298,326 | 60,246,690 | 298,326 | 59,794,939 | 298,326 | 298,325 | DERIVED | DERIVED | 40.4 | 0.0014 | no |
| `oxford` | Oxford | rtk.csv | 70 | 1,355,769 | 67 | 1,324,765 | 6,588 | 6,588 | DERIVED | DERIVED | 13.4 | 0.00 | no |
| `waymo` | Waymo | TFRecord, one scenario per clip | 70,541 | 14,000,081 | 70,541 | 14,003,851 | 70,541 | 70,541 | RAW | DERIVED | 20.4 | 0.00 | no |
| `zod` | ZOD | oxts.hdf5 | 29 | 661,429 | 29 | 66,169 | 318 | 318 | RAW | RAW | 39.8 | 0.00 | no |
| **Total** | | | **371,430** | **178,778,906** | **371,425** | **81,252,848** | **406,130** | **402,421** | | | | | |

Five raw clips do not survive standardization: the three Oxford files named in
`KNOWN_BAD_SOURCES` (`oxford_clip_0013`, `_0039`, `_0051`, with raw maximum speeds of
636, 11,559, and 138,439 km/h), one Boreas clip, and one DurLAR clip. The standardizer
has only three other drop paths (the timestamp-cleanup minimum, `MIN_VALID_FRAC_CLIP`
0.60, and `GATE_FRAC_DROP` 0.15), and the run log states which one fired for the Boreas
and DurLAR clips. A clip is one contiguous recording file with no further splitting,
except Waymo, where each `scenario_id` is one clip.

`CONFOUNDED_ACCEL = {a2d2, 4seasons, durlar}` (owner: `bias_quant.py`, mirrored in
`subset_eval_common.py`) marks the sources whose acceleration channels are contaminated
by in-band position noise. They stay in every computation, are asterisked in figures,
and are excluded from rankings and headline claims. The guarded frame R1 is the frame
without them.

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
interactive and nothing mutates stored data. Long scripts are run with
`caffeinate -i python <script>` so the Mac does not sleep mid-pass.

> This repository is archived as a thesis artifact tied to the author's machine
> layout, not as a portable package. Reproducing it elsewhere means editing the root
> constants below.
>
> Scripts resolve the dataset tree by absolute path (`/Users/edward/Downloads`).
> `canonical.py` and `subset_eval_common.py` honor a `THESIS_ROOT` environment
> override, `standardize_inertial_data.py` searches argv, the script directory, the
> working directory, and `~/Downloads/dataset` for a `parquet_data/` folder. Every
> other script hardcodes `PROJECT_ROOT`. If the data moves, update the `*_DIR` and
> `*_ROOT` top-level constants.

## Pipeline overview

```
parquet_data/<ds>/              raw per-clip parquet, native rate (10 to 2,000 Hz)
        |
        |  standardize_inertial_data.py    v4_4, per clip: timestamp cleanup, GPS
        |                                  cleaning, velocity from band-limited
        |                                  positions, Hampel, gap policy, zero-phase
        |                                  3 Hz Butterworth, 10 Hz resample,
        |                                  Savitzky-Golay derivatives, edge masks,
        |                                  physical-validity gates, clip guards
        v
inertial_data/<ds>/clip_XXXX.parquet     43-column clips on the common 10 Hz grid
        |
        |  compute_std.py                  20 s windows advanced 20 s (non-overlapping,
        |                                  D-2 2026-08-29), 3 variability stds and
        v                                  6 feature means, MIN_VALID_FRAC = 0.5
std_data/<ds>.parquet                    one parquet per dataset, 406,130 windows
        |
        +-- inertial_exploratory_analysis/  RQ1 coverage audit, tiers, 184 figures
        |
        +-- regime_discovery/               RQ2 feature gate -> k-means -> battery
        |          v
        |     regime_data/                  21 regime artifacts (assignments, profile,
        |                                   k-select, stability, GMM, HMM)
        |
        +-- bias_quantification/            RQ1 instruments on the regime structure
        |          v
        |     regime_data/                  12 bias_* artifacts join the 21 above
        |
        +-- subset_curation/                RQ3 sampling methods M01 to M11, curation
                   v                        methods C01 to C03, gates, replication
              subset_data/                  subsets at 2 tiers, eval caches, manifests
```

Full re-run in dependency order (this is the order the artifacts require, not a
preference):

```bash
cd inertial_exploratory_analysis
python pipeline_regression_test.py            # synthetic gate, no real data, run first
python run_pipeline.py --from coverage        # skip standardize unless raw data changed

cd ../regime_discovery
python regime_run_pipeline.py                 # gate -> cluster -> models -> figures

cd ../bias_quantification
python bias_run_pipeline.py                   # compute -> tier1_figs -> tier2 -> fingerprint

cd ../subset_curation
python window_summary.py                      # frame-layer summary pass, compute-once
python run_sampling.py                        # M01 to M11, capability tier
python run_sampling.py --n 71390              # M01 to M11, budget tier
python curation_run_pipeline.py --frame-pass --dossier   # Chapter 13, protocol order
python curation_evassive_check.py             # D24 evidence, deliberately not in the runner
python subset_run_pipeline.py                 # Chapter 12 figures, renders from caches
python canonical.py --check                   # oracle: must report no drift
python curation_info_dump.py                  # paste-back of every owed number
cd ..
python general_info.py                        # rewrites general_info.txt
```

The Chapter 13 runner comes before the Chapter 12 renderer because `curation_eval.py`
evaluates the entire slate (M01 to M11 and C01 to C03) in one pass and writes the shared
eval caches. `subset_run_pipeline.py` computes nothing, it renders those caches and
refuses to start if any cache is older than `cluster_assignments.parquet`.

## Repository structure

The repository tree (generated 2026-09-06 23:02) holds 27,061 files and 1.8 GB. The
visible tree is about 550 files, the rest sits in hidden directories that the tree
tool does not expand (`.venv`, `.git`).

| Path | Purpose |
|---|---|
| `dataset_analysis/` | Survey of the dataset pool (metadata, not ego-motion signals). Exempt from the pipeline and from the stale audit. |
| `dataset_analysis/total_dataset_analysis/` | Full pool of 274 datasets. `00_cleaning_total_dataset.py` encodes the categorical columns, scripts `01a` to `10` draw 15 figures into `total_dataset_analysis_figures/`. |
| `dataset_analysis/filtered_dataset_analysis/` | The 34 pre-filtered candidates. `00_cleaning_filtered_dataset.py` normalizes the sensor columns, scripts `01` to `16` draw 18 figures into `filtered_dataset_analysis_figures/`, including the Qs, Ds, and DSS scores. |
| `dataset_analysis/tum_colors.py` | Palette copy imported by the survey scripts through `sys.path.append('..')`. |
| `dataset_survey_table/` | The five hand-built survey tables, stages 0 to 4 of the dataset selection. |
| `inertial_exploratory_analysis/` | RQ1. Standardization, coverage audit, windowing, variability tiers, 184 distributional figures. Runner: `run_pipeline.py`. |
| `regime_discovery/` | RQ2. Feature gate, k-means with k-selection, stability battery, GMM and HMM cross-checks, 20 figures. Runner: `regime_run_pipeline.py`. |
| `bias_quantification/` | RQ1 instruments. Tier-1 composition metrics and Tier-2 diagnostics on the regime structure, 14 figures. Runner: `bias_run_pipeline.py`. |
| `subset_curation/` | RQ3. Sampling methods, evaluation harness, curation methods, pre-registered gates, replication, winner dossier, 173 figures. Runners: `curation_run_pipeline.py`, `subset_run_pipeline.py`, plus `run_sampling.py`. |
| `general_info.py` | Reads every artifact and writes `general_info.txt`, the numeric ground truth for the write-up. |
| `general_info.txt` | 86.2 KB, 15 sections, generated 2026-08-30 02:10. Read-only, artifact-fed. |
| `requirements.txt` | The 8 pinned direct dependencies. |
| `README.md` | This file. |

Shared modules that are imported, never executed: `tum_colors.py` (official TUM
palette, duplicated into `dataset_analysis/`, `inertial_exploratory_analysis/`,
`regime_discovery/`, `bias_quantification/`, and `subset_curation/` because every module
imports its local copy) and `regime_style.py` (regime labels, colors, and order,
duplicated into `regime_discovery/`, `bias_quantification/`, and `subset_curation/`).
The `bias_quantification/` copy of `regime_style.py` (2.3 KB) additionally exports
`N_REGIMES = 5` and `LOW_DYNAMIC_REGIMES`, which `bias_quant.py` and
`bias_tier2_common.py` import unconditionally, while the two 2.0 KB copies do not.
`subset_eval_common.py` guards those two imports with a try/except, so the subset
module works with either copy.

### dataset_analysis and dataset_survey_table

The survey is the dataset-selection chain that produced the 14-dataset working set.
`dataset_survey_table/` keeps it as five tables. `0_total_dataset_pool.csv` is the full
pool of 274 datasets with 22 metadata columns (year, links, issuing and recording
country, real or synthetic, availability, originality, and 12 sensor flags).
`1_filtered_dataset_1st_stage.csv` keeps 41 after the hard filters (real recordings,
publicly downloadable, road vehicles, original releases, ego trajectory and inertial
data present). `2_ filtered_dataset_2nd_stage.csv` (the filename carries a space) keeps
34. `3_detailed_sensorr_modalities.csv` (the filename carries the typo) records 42
detailed sensor and task columns for those 34. `4_dataset_scoring.csv` holds the
Dynamic Scenario Suitability Score.

The scoring is implemented in `14_qs_score.py`, `15_ds_score.py`, and
`16_dss_score.py`: Qs = 0.5 x IMU grade (Navigation 1.00, Tactical 0.75, Industrial
0.50, Consumer 0.25, highest listed grade) + 0.5 x GPS type (RTK 1.0, otherwise 0.0),
Ds = 0.5 x road-type coverage (five groups, baseline 0.2 when undocumented) + 0.5 x
weather coverage (five conditions, baseline 0.4 when undocumented), Ps = 1.0 with CAN
bus and 0.0 without, and DSS = 0.60 Qs + 0.25 Ds + 0.15 Ps. Citation count and total
duration were removed from the score. The admission threshold DSS >= 0.675 passes 20 of
the 34 candidates (HDD 0.850 at the top, Waymo and ZOD at exactly 0.675 as the last
admitted), and 6 of the 20 were excluded at acquisition (not downloadable, no public
inertial data, redundant with an included dataset, or no answer to the access request),
which yields the 14-dataset working set.

The `total_dataset_analysis` figures cover datasets per year (real, synthetic, mixed,
total), cumulative growth, average sensors per year, recording-country map and bar,
issuing-country bar, real-versus-synthetic pie, camera and other-sensor counts, vehicle
information, availability, and originality. The `filtered_dataset_analysis` figures
cover camera, LiDAR, and radar counts, task support, time of day, weather, LiDAR beams,
citations (full and zoomed), CAN bus, IMU grade, GPS type, IMU frequency, road type, the
Qs and Ds bars, and the DSS bar and heatmap. `16_dss_score.py` saves the DSS bar at
`dpi=150`, the one exception to the 300 dpi rule in the whole tree.

### inertial_exploratory_analysis

**Standardization.** `standardize_inertial_data.py` (79 KB, pipeline version
`v4_4_10hz_common_grid_validity_gated`) converts every raw clip in `parquet_data/<ds>/`
into a 43-column clip on the common 10 Hz grid in `inertial_data/<ds>/clip_XXXX.parquet`.
Output clip indices renumber whenever a clip is dropped, so the blocklist is keyed on raw
source filenames. Every clip carries `processing_meta` (pipeline version, every
constant, frame drop counts, gated fractions, drop reason) and `col_sources` (RAW,
DERIVED, NOT_AVAILABLE, FLAG, or EXCLUDED_MEASUREMENT_LIMITED per column) in the parquet
schema metadata. Loading is parallel over clips with a ThreadPoolExecutor
(`MAX_WORKERS = min(cpu_count, 8)`) and a global tqdm bar.

| Stage | Operation | Constants |
|---|---|---|
| 1 | Timestamp cleanup: monotonic filter, drop near-zero dt frames | `DT_MIN_FRAC` 0.1 (drop dt below 10 % of the median dt) |
| 2 | GPS cleaning: sustained-offset interpolation for DurLAR and NVIDIA positions | `GPS_SPEED_MAX` 50 m/s |
| 3 | Velocity derivation by central differences of low-passed positions when no raw velocity exists | |
| 4 | Hampel filter on speed, point outliers only, tails preserved | window 0.5 s, `HAMPEL_T0` 3.0 MAD |
| 5 | Gap policy: interpolate short gaps, mask timestamp gaps and NaN runs | `GAP_MAX_S` 0.5 s |
| 6 | Anti-alias low-pass: zero-phase Butterworth at native rate | `F_CUT_HZ` 3.0, order 4 through `sosfiltfilt` (effective order 8) |
| 7 | Resample onto the common grid, masked gaps stay NaN in every level signal | `GRID_HZ` 10 |
| 8 | Savitzky-Golay first and second derivatives per contiguous valid segment | `SG_POLY` 3, `W_D1` 7 frames (0.7 s), `W_D2` 11 frames (1.1 s) |
| 9 | Derivative masking inside dilated gap regions and at clip edges | `EDGE_MASK` and `GAP_DILATION` 6 frames |
| 10 | Curvature validity floor | `CURV_SPEED_MIN` 2.0 m/s |
| v4 gates | Physical-validity gates with mask closing | speed <= 200 km/h, per-axis abs acceleration <= 15 m/s^2, per-axis abs jerk <= 30 m/s^3, abs yaw rate <= 90 deg/s, `GATE_CLOSE_S` 2.0 s |
| v4 guards | Clip drop instead of repair | `MIN_VALID_FRAC_CLIP` 0.60, `GATE_FRAC_DROP` 0.15, `KNOWN_BAD_SOURCES` (three Oxford files) |
| v4_4 | 4Seasons measurement-limited exclusion disabled (decision 2026-07-15), 4Seasons retained in full and asterisked | `MEASUREMENT_LIMITED` empty |

The 43 output columns are `timestamp_ns, timestamp_s, t_rel_ns, t_rel_s, dt_ns, dt_s,
x, y, z, lat, lon, x_rel, y_rel, x_aligned, y_aligned, heading_initial, vx, vy, vz,
speed, speed_kmh, ax, ay, az, a_long, a_lat, deceleration, acc_mag, jerk_long,
jerk_lat, jerk_mag, abs_jerk_long, abs_jerk_mag, roll, pitch, yaw, yaw_unwrapped,
yaw_rate, heading_change, abs_heading_change, curvature, abs_curvature,
curvature_valid`. `acc_mag` is `sqrt(a_long^2 + a_lat^2)` and `jerk_mag` is
`sqrt(jerk_long^2 + jerk_lat^2)`, both computed here and never re-derived downstream.
`yaw_rate` is stored in rad/s.

`pipeline_regression_test.py` is the synthetic gate for the standardization code: it
plants ten failure modes (T1a rectangular and T1b raised-cosine braking pulses of
-5.0 m/s^2, T2 single-frame speed spikes of +80 m/s, T3 timestamp gaps, T4 NaN-run
dropouts, T5 yaw wrapping, T6 deceleration NaN preservation, T7 edge masking, T8 short
clips, T9 position-only constant speed, T10 non-monotonic frames) into generated clips
and asserts the output. It needs no real data and runs first. The recovered peak of the
planted event (-4.96 m/s^2) and the +5.3 % rectangular-window comparison exist only in
its stdout, not in any artifact.

**Coverage audit.** `coverage.py` reads `inertial_data/` and draws three heatmaps over
the 6 kinematic signals (`speed_kmh, a_long, a_lat, jerk_long, jerk_lat, yaw_rate`):
`signal_coverage.png` (non-NaN share), `non_zero.png` (non-zero share among non-NaN),
and `boundary.png` (share within the physical bounds). `coverage_baseline.py` is the
restored producer of `audit_baseline.json` (the earlier producer was lost and the
existence-only artifact check kept passing on a stale file). Per dataset it records
clips, frames, maximum speed, maximum per-axis acceleration and jerk, maximum yaw rate,
frames above the 200 km/h and 108 km/h marks, frames above 15 m/s^2, mean speed, the
speed and yaw-rate sources, the pipeline version, and the frame-weighted gated share.
These are the post-cleaning coverage numbers for the thesis (analysis-track section 7.3).

**Window layer.** `compute_std.py` reads the standardized clips (ProcessPoolExecutor
over datasets) and writes one parquet per dataset into `std_data/`. Windows are 20 s
advanced 20 s within each clip (`WINDOW_S` 20.0, `STEP_S` 20.0, non-overlapping since
D-2 on 2026-08-29), never across clips. A window statistic is NaN unless at least
`MIN_VALID_FRAC` 0.5 of the expected 200 frames (and at least `MIN_VALID_ABS` 2) are
non-NaN for that signal. `APPLY_BOUNDS` masks impossible values at read time (speed
[0, 200] km/h, `acc_mag` [0, 21.3] m/s^2, `yaw_rate` [-1.57, 1.57] rad/s, `a_long` and
`a_lat` [-15, 15] m/s^2, `jerk_mag` [0, 42.43] m/s^3) without touching the clips. The
schema is `clip_id, window_idx, window_start_s, window_end_s, n_frames`, the three
variability channels `std_speed_kmh, std_acc_mag, std_yaw_rate`, and the six feature
means `mean_speed_kmh, mean_acc_pos, mean_acc_neg, mean_abs_a_lat, mean_abs_yaw_rate,
mean_abs_jerk`. This single per-window pass feeds the exploratory chapter (stds) and
the clustering chapter (means), so the two can never disagree about which window is
which.

**Variability tiers.** `regime_classification.py` sorts every window into four
variability tiers (near-stationary, low-dynamic, medium-dynamic, high-dynamic) by fixed
cuts on one variability signal, frozen as round conventions under D-5 on 2026-08-29:
speed 1.0 / 5.0 / 15.0 km/h, acceleration magnitude 0.2 / 0.5 / 1.5 m/s^2, yaw rate
1.0 / 5.0 / 15.0 deg/s. The pooled tier shares are 8.13 / 29.89 / 47.10 / 14.88 % for
speed, 15.31 / 36.66 / 47.46 / 0.56 % for acceleration, and 48.55 / 24.08 / 26.63 /
0.74 % for yaw rate. `ecdf_analysis.py` and `time_series_analysis.py` import these
thresholds, so the thresholds stage runs before either. The word regime is reserved for
the clustered categories of RQ2, the filename and the `figures/regime/` folder were kept
for path stability only.

**Figures (184, all under `figures/`).**

| Folder | Script | Content | Count |
|---|---|---|---:|
| `distribution/` | `distributional_analysis.py` | Per-frame histograms, 14 datasets x 6 signals (speed, lon_accel, lat_accel, lon_jerk, lat_jerk, yaw_rate) | 84 |
| `trajectory_fans/` | `trajectory_fan.py` | XY trajectory fans and speed-change fans per dataset, one line per non-overlapping 20 s window, subsample seed 0 | 28 |
| `phase_portrait/` | `phase_potrait.py` | Speed x longitudinal acceleration and speed x yaw rate occupancy per dataset, bins below a frame floor masked | 28 |
| `density_band/` | `density_band_analysis.py` | Per dataset, three stacked panels (speed, acc_mag, yaw) with median, interquartile, and 10 to 90 % bands over the within-window time axis, subsample seed 0 | 14 |
| `time_series/` | `time_series_analysis.py` | Per dataset speed traces of selected windows against the tier thresholds | 14 |
| `ridgeline/` | `ridgeline.py` | Per-window variability ridgelines, one figure per signal, datasets ordered by median | 3 |
| `boxplot/` | `boxplot_comparison.py` | Cross-dataset variability boxplots with trimmed violins and disclosed tails, one per signal | 3 |
| `ecdf/` | `ecdf_analysis.py` | ECDF overlays of the three windowed signals with the three tier cuts | 3 |
| `regime/` | `regime_classification.py` | Stacked tier-share bars per signal | 3 |
| `coverage/` | `coverage.py` | The three coverage heatmaps | 3 |
| `summary/` | `heatmap_summary.py` | The Chapter 8 scorecard: one headline metric per facet, Blues colormap, Dynamic-Index spine ordering the rows | 1 |

Runner flags for `run_pipeline.py`: `--dry-run`, `--from <stage>`, `--only <stage>`,
`--skip <stage>`, `--keep-going`. Stages in order: `standardize`, `coverage`,
`compute_std`, `thresholds`, `frame_figs`, `window_figs`, `time_series`, `summary`. Each
stage verifies its artifacts by counting files, because an existence check on a folder
would not have caught the emptied NVIDIA `inertial_data/` directory that once passed
silently.

### regime_discovery

`regime_run_pipeline.py` runs four stages: `gate`, `cluster`, `models`, `figures`.

**Feature gate.** `feature_gate.py` screens six candidate features (`mean_speed_kmh,
std_speed_kmh, mean_acc_pos, mean_acc_neg, mean_abs_a_lat, mean_abs_yaw_rate`) for
redundancy with Pearson, Spearman, and VIF (`CORR_THRESHOLD` 0.85, `VIF_THRESHOLD` 5.0,
`VIF_TIE_MARGIN` 0.5). Jerk is excluded a priori on signal-quality grounds. The lateral
acceleration intensity is dropped (abs Pearson 0.90 with the yaw-rate intensity, VIF
3.49 against 3.62, a tie broken toward the directly sensed feature), every retained VIF
is at most 2.96, and the locked five-feature vector is persisted to
`feature_gate_vector.parquet`. The screening operates on the 402,293 windows complete
in all six candidate features, while the assignment set comprises the 402,421 windows
complete in the five retained ones. The 128 additional windows (127 KITScenes, 1 Waymo,
verified against `std_data` on 2026-09-07) are exactly those whose lateral channel was
missing, so they could not have influenced the decision to drop that channel. Every downstream script reads the vector from that file
and never hardcodes it.

**Clustering.** `clustering.py` is the single stochastic fit. The FIT population caps
every dataset at `CAP` 5,000 windows by stratified sampling over `N_STRATA` 10
intensity strata, so the regimes are a property of driving in general rather than of
NVIDIA. The ASSIGN population is every window with all five features present (402,421).
k-means runs on the full standardized vector with `N_INIT` 10 and `SEED` 42 over
`K_RANGE` 2 to 10, k is chosen by maximum silhouette (subsample 10,000) against a
shuffled-feature null, and PCA serves the scatter figures only, with component signs
pinned to named feature anchors. Regime indices are ordered by intensity, 0 calmest.
Outputs: `cluster_assignments.parquet` (the spine of everything downstream, keyed
`dataset, clip_id, window_start_s, window_end_s` with `regime, pc1, pc2, pc3`),
`cluster_profile.parquet`, `kselect_curves.parquet`, `pca_variance.parquet`. No model
file is persisted, the seed regenerates the identical partition.

| Regime | Label | Mean speed km/h | Std speed km/h | acc+ m/s^2 | acc- m/s^2 | Mean abs yaw rad/s (deg/s) | Supply | Share |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | Calm / Stationary | 14.47 | 3.24 | 0.135 | 0.137 | 0.0130 (0.74) | 89,420 | 22.22 % |
| 1 | Highway Cruising | 77.70 | 3.63 | 0.178 | 0.162 | 0.0146 (0.84) | 89,499 | 22.24 % |
| 2 | Turning | 24.33 | 7.16 | 0.325 | 0.281 | 0.1146 (6.57) | 78,883 | 19.60 % |
| 3 | Braking | 29.41 | 14.14 | 0.177 | 0.642 | 0.0223 (1.28) | 73,229 | 18.20 % |
| 4 | Accelerating | 28.96 | 15.27 | 0.681 | 0.139 | 0.0420 (2.41) | 71,390 | 17.74 % |

Silhouette over the sweep is 0.2668, 0.2957, 0.3123, 0.3471, 0.2978, 0.3068, 0.3072,
0.3110, 0.3150 for k = 2 to 10 (null 0.2199 at k = 5), so k = 5 is the argmax. PCA
explains 37.74 / 21.62 / 19.52 % on the first three components.

**Stability battery (`models` stage).** `cap_sensitivity.py` refits at caps 2,500 /
5,000 / 10,000 (ARI to the 5,000 reference 0.9674 and 0.9620, Spearman rank correlation
of the dataset ordering 1.0000 and 0.9868, maximum centroid drift 0.0965 pooled sigma).
`reseed_stability.py` refits under 20 reseeds 43 to 62 (ARI 0.9945 to 0.9983).
`resample_stability.py` redraws the stratified fit population 20 times under seeds 242
to 261 (ARI 0.9705 to 0.9916). `validity.py` computes the gap statistic (`B_GAP` 10
uniform references, natural logarithm), Davies-Bouldin, and Calinski-Harabasz over the
sweep (Davies-Bouldin minimum 1.0409 and Calinski-Harabasz maximum 11,966.68 both at
k = 5, the gap argmax sits at k = 10), per-regime bootstrap Jaccard over `B_BOOT` 50
resamples (mean 0.9797 to 0.9870, minimum 0.9507 in regime 3), and a consensus matrix
over `B_CONS` 50 resamples (diagonal mean 1.000, off-diagonal mean 0.234). `gmm.py`
fits a Gaussian mixture within the same fit-and-assign firewall, sweeps BIC and AIC
over 2 to 10 components (BIC keeps falling to the sweep limit), and reports the
posterior entropy per window and an ARI of 0.3165 between the GMM hard labels and
k-means at k = 5. `hmm.py` re-derives the five features on 2 s sub-windows advanced 1 s
within each clip (`CAP_SUBWIN_PER_DATASET` 30,000, accumulated as whole clips), aligns
the Gaussian HMM states to the k-means regimes by nearest emission mean, and writes the
transition matrix, dwell durations, and decoded timelines. The fitted self-transitions
are 0.945 / 0.848 / 0.748 / 0.781 / 0.789 and the mean dwell times 13.2 / 6.1 / 3.9 /
4.3 / 4.7 s for regimes 0 to 4, so at the 2 s grain regime 2 (Turning) is the least
persistent state and regime 0 by far the most. `hmm.py` walks the frame layer and is
the slowest script in the module.

**Figures (20, under `figures/`).** `model_selection/kselect_curve.png`,
`feature_gate/feature_correlation_gate.png`, `pca/` (2D and 3D scatters before and
after labeling, seed 42 subsample), `regime_profile/regime_profile_heatmap.png` (plus the
`regime_heatmap_sweep.py` variant), `validity/` (`validity_indices.png`,
`regime_stability.png`, `consensus_matrix.png`), `cap_sensitivity/`
(`cap_sensitivity_ari.png`, `reseed_stability.png`, `resample_stability.png`), `gmm/`
(`gmm_selection.png`, `membership_entropy.png`, `covariance_contours.png`,
`kmeans_gmm_correspondence.png`), and `hmm/` (`transition_matrix.png`,
`state_durations.png`, `state_timeline.png`). Every figure reads labels and colors from
`regime_style.py` at render time, so a label change plus `--only figures` is enough.

### bias_quantification

`bias_run_pipeline.py` runs `compute`, `tier1_figs`, `tier2`, `fingerprint`, with the
fingerprint strictly last because it merges every earlier artifact.

**Tier 1 (`bias_quant.py`).** Reads `cluster_assignments.parquet` and the three
`std_data` channels, computes per dataset the five regime shares, the low-dynamic share
(regimes 0 and 1 together, `LOW_DYNAMIC_REGIMES`), normalized Shannon entropy (natural
logarithm, divided by ln 5), Jensen-Shannon distance to the pooled and to the balanced
regime profile (base-2 logarithms, `JS_BASE = 2`, bounded to [0, 1]), the
Herfindahl-Hirschman index, and the per-channel Wasserstein-1 distance of each dataset's
variability distribution to the pooled distribution (yaw in deg/s). Nothing here is
stochastic. Outputs: `bias_per_dataset.parquet`, `bias_corpus_profiles.parquet`
(pooled, balanced, dataset_std vectors), `bias_corpus_scalars.parquet` (volume gap JS
0.1466, mean pairwise JS 0.3297), `bias_sparse_regimes.parquet` (the dataset x regime
deficit table that feeds RQ3).

| Dataset | Windows | Volume share % | Low-dynamic share | Entropy (norm.) | JS to pooled | JS to balanced | Hill D1 | PSI to pooled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4Seasons* | 707 | 0.18 | 0.2277 | 0.9315 | 0.2289 | 0.2912 | 4.478 | 0.3002 |
| A2D2* | 106 | 0.03 | 0.4717 | 0.7822 | 0.3658 | 0.2438 | 3.521 | 0.8509 |
| Boreas | 5,721 | 1.42 | 0.5076 | 0.9723 | 0.0977 | 0.0829 | 4.782 | 0.0532 |
| DurLAR* | 378 | 0.09 | 0.4206 | 0.9838 | 0.0900 | 0.1446 | 4.871 | 0.0451 |
| Ford | 1,205 | 0.30 | 0.6880 | 0.8671 | 0.2395 | 0.3028 | 4.037 | 0.3242 |
| HDD | 15,752 | 3.91 | 0.5529 | 0.9308 | 0.1821 | 0.1880 | 4.473 | 0.1867 |
| KITScenes | 1,040 | 0.26 | 0.4269 | 0.9665 | 0.1336 | 0.0752 | 4.738 | 0.0998 |
| KITTI | 297 | 0.07 | 0.5488 | 0.7729 | 0.3494 | 0.2094 | 3.469 | 0.7055 |
| KITTI-360 | 466 | 0.12 | 0.2275 | 0.6825 | 0.4213 | 0.3385 | 3.000 | 1.0565 |
| nuScenes | 977 | 0.24 | 0.5466 | 0.7233 | 0.4034 | 0.2819 | 3.203 | 1.0905 |
| NVIDIA | 298,325 | 74.13 | 0.4392 | 0.9896 | 0.0647 | 0.1868 | 4.917 | 0.0232 |
| Oxford | 6,588 | 1.64 | 0.6606 | 0.5338 | 0.5260 | 0.4016 | 2.361 | 2.7201 |
| Waymo | 70,541 | 17.53 | 0.4151 | 0.9263 | 0.2222 | 0.2294 | 4.441 | 0.2828 |
| ZOD | 318 | 0.08 | 0.6635 | 0.8582 | 0.2635 | 0.2441 | 3.980 | 0.4108 |

Asterisks mark the confounded-acceleration sources. The pooled low-dynamic share is
0.4446 and the balanced (equal-weight) share 0.4855. Because NVIDIA is 74.13 % of the
frame, `js_to_pooled` nearly measures distance to NVIDIA, which is why
`js_to_balanced` is carried alongside as the robustness check.

**Tier 2 (`bias_tier2_common.py` plus six instruments).** The shared loader re-derives
the shares from `cluster_assignments.parquet` and cross-checks them against Tier 1 to
1e-12 at runtime (Hill D1 = 5 ^ entropy_norm, Hill D2 = 1 / HHI). `bias_diversity.py`
writes Hill numbers D0, D1, D2 and the population stability index (natural logarithm,
1e-4 floor on empty cells, so Oxford's missing regime shows as PSI 2.72 by
construction). `bias_volume.py` writes the volume share, the volume Gini (0.8626), and
the effective number of sources (D1 2.38, D2 1.72 of 14). `bias_bootstrap.py` writes
clip-level bootstrap confidence intervals for the low-dynamic share and `js_to_pooled`
and the rank probabilities against the pooled and balanced references.
`bias_gmm_entropy.py` writes the mean and median posterior entropy per dataset and the
ambiguous-window shares at 50 / 80 / 90 %. `bias_shape_metrics.py` writes the Gini of
each variability channel per dataset and pooled (0.4218 / 0.3298 / 0.6148) and the
Lorenz curves. `bias_regime_supply.py` writes each dataset's share of every regime's
supply (NVIDIA supplies 88.86 % of Highway Cruising), a real RQ3 input whose PNG is not
a thesis figure. `bias_fingerprint.py` merges everything into a 10-metric table plus
the Spearman agreement matrix between metrics (Hill D1 against `js_to_pooled` -0.9824,
`js_to_pooled` against PSI 0.9956).

**Figures (14, flat under `figures/`).** Tier 1: `bias_stacked_bar.png`,
`bias_regime_grid.png`, `bias_share_heatmap.png`, `bias_balance_metrics.png`,
`bias_wasserstein.png`. Tier 2: `hill_profiles.png`, `psi_thresholds.png`,
`lorenz_volume.png`, `low_dyn_share_ci.png`, `js_to_pooled_ci.png`, `gmm_entropy.png`,
`dynamic_lorenz.png`, `bias_regime_supply.png`. Synthesis: `bias_fingerprint.png`.

### subset_curation

**Construction (`run_sampling.py`, analysis-track Chapter 11).** Reads the frozen
window layer (`cluster_assignments.parquet`, `std_data`, and the `window_summaries`
cache) and writes eleven window-ID subsets under one protocol: overlap guard
(`DEDUP_MODE` auto, verifies the grid is already non-overlapping), `CONFOUNDED_ACCEL`
guard, matched-N sizing (`MATCHED_N_MULT` 5 x the rarest regime supply = 356,950),
`--n 71390` for the budget tier (written to `tier_n71390/` so it never overwrites the
capability tier), fixed seeds (`np.random.default_rng([42, m])` and scikit-learn seed
4200 + m for method m), and largest-remainder integer allocation. Also written:
`supply_table.csv` (regime x dataset), `allocation_table.csv` (per-regime quotas of
M02 to M05 and M11), `manifest.json`.

| ID | File | Family | Definition |
|---|---|---|---|
| M01 | `M01_srs.parquet` | replica | simple random sample |
| M02 | `M02_stratified_proportional.parquet` | replica | regime-stratified, proportional quotas (budget 15,864 / 15,878 / 13,994 / 12,990 / 12,664) |
| M03 | `M03_stratified_equal.parquet` | allocation | regime-stratified, equal quotas (14,278 per regime) |
| M04 | `M04_power_allocation.parquet` | allocation | Bankier power allocation, `BANKIER_Q` 0.5 |
| M05 | `M05_two_way_regime_source.parquet` | allocation | two-way regime x source cells |
| M06 | `M06_pps_dynamic_score.parquet` | probability | probability proportional to the dynamic score |
| M07 | `M07_kinematic_trigger.parquet` | rule-based | 0.6 g hard-deceleration trigger, `EVENT_SHARE` 0.5 |
| M08 | `M08_predicate_mining.parquet` | rule-based | union of predicates (abs yaw >= 30 deg/s, cruise >= 100 km/h, stop-and-go std >= 5 with mean <= 20 km/h, near-stationary mean <= 2 with std <= 1 km/h, score >= p99 of the guarded frame) |
| M09 | `M09_kennard_stone.parquet` | coverage | Kennard-Stone maximin in feature space |
| M10 | `M10_cluster_quota_fill.parquet` | coverage | `SUBCLUSTER_K` 10 sub-clusters per regime, selection layer only, regime labels never refit |
| M11 | `M11_neyman_allocation.parquet` | variance-optimal | Neyman allocation (budget 11,584 / 10,412 / 16,590 / 14,544 / 18,260) |

M07 and M08 need `window_summaries.parquet` and refuse to run if any dataset's summary
coverage falls below `MIN_SUMMARY_COVERAGE` 0.95. M08 is bounded by its predicate union
(146,223 windows), so its capability-tier subset holds 145,962 windows rather than
356,950, and M09 holds 356,779. `window_summary.py` is the first disclosed frame-layer
pass: for every `std_data` window it reads the frames through the back-reference
contract and writes `mean_speed_kmh, max_speed_kmh, max_decel_ms2, max_abs_yaw_degs,
max_acc_mag, n_frames_used`, with per-dataset partials in `_summaries_partial/` that are
reused only when their key set matches the current `std_data` exactly.

**Evaluation harness (`subset_eval_harness.py`, `METRIC_VERSION` 12.1.2).** Re-points
the Chapter 10 instruments at methods instead of datasets. For every subset at both
tiers, plus the reference rows R0 (frame) and R1 (guarded frame), it computes six
instrument groups (I1 regime shares, low-dynamic share as regime 0, entropy in bits,
Hill D2, JS to frame and to uniform, I2 per-channel Wasserstein-1 in-model and
held-out, I3 dynamic content, I4 source composition with the worst-regime top supplier,
I5 event and predicate retention as ratios to the random-selection null, I6 PC and cell
coverage), a K = 100 SRS null envelope at the budget tier (seeds 1000 + k), and the
tautology map that tags every (metric, method) cell as T (construction-determined), P
(influenced), O (held-out), or N (null anchor). Three metrics are sealed
(`w1_max_speed`, `w1_peak_acc_mag`, `mean_gmm_entropy`) and written but never printed
without `--reveal`. The harness canaries (frame 402,421, guarded 401,230, 0.6 g trigger
1,088) now derive from `canonical.py`. The harness is never run standalone in the closed
state: `curation_eval.py` patches its module globals (tier directory, method names, C
discovery, re-anchored canaries) and runs it so the C-rows land in the same caches.

**Frame pass (`subset_frame_pass.py`).** The second disclosed frame-layer excursion. One
streaming read of every clip tags each frame with its 20 s window and, per membership
group (13 groups: the frame, M01 to M11, and C01 at the budget tier), accumulates fine
histograms and moments per signal (1,000 bins), the two phase-portrait 2D histograms,
and raw sequences for a seeded sample of `FAN_N` 5,000 windows per group (seed 777).
80,304,243 frames belong to the frame group. Expect 10 to 40 minutes, NVIDIA's 298,326
small files dominate.

**Curation (`curation_run_pipeline.py`, analysis-track Chapter 13).** The protocol
order is enforced: `audit` (`curation_stale_audit.py`, per-chapter freshness verdicts,
read-only) -> `diagnostics` (`curation_event_audit.py`, reserve and evasive counts) ->
`build` (`curation_build.py`) -> `eval` (`curation_eval.py`) -> `gates`
(`curation_gates.py`) -> `feasibility` (`curation_feasibility.py`) -> `replicate`
(`curation_replicate.py`) -> `reveal` (`curation_reveal.py`) -> `render`
(`curation_render_winner.py`) -> `dossier` (`curation_winner_dossier.py`) -> `pca`
(`curation_winner_pca.py`). Every stage keeps its own guards, the runner adds ordering
and bookkeeping only. Re-running the gates and the reveal reprints frozen dated rules,
the D4 declaration and the seal opening remain the register events of 2026-08-20.

The slate (D21) is a 2 x 2 ablation with M03 as the free fourth cell. C01
(`C01_event_capped_equal.parquet`, seed index 12) is the full composite: equal regime
quotas, then the guarded 0.6 g reserve force-included and debited (D22, all 1,088
windows, inclusion probability 1), then a per-regime top-source cap at
`SOURCE_CAP_FRAC` 0.50 of the regime quota, cap-not-equalize, proportional draw for
everyone else (D23), then seeded SRS within each (regime, source) cell. C02
(`C02_capped_equal.parquet`, seed index 13) drops the reserve, C03
(`C03_event_equal.parquet`, seed index 14) drops the cap. `curation_build.py` imports
the frozen `run_sampling.py` machinery and patches its globals rather than editing it.
The budget manifest records C01's top source as NVIDIA at exactly 35,695 windows
(0.5000) with forced reserve counts 9 / 406 / 64 / 476 / 133 per regime, while at the
capability tier the cap is breached in regimes 1 to 4 (realized top shares 0.8603 /
0.7996 / 0.7098 / 0.7035) because the supply cannot honor 0.50 there, and the manifest
discloses every breach.

The D4 rule (declared 2026-08-20, before any C-artifact was evaluated) applies four
gates on the budget-tier tables, then a fidelity tie-break (`max_abs_shift_speed`, the
largest within-regime shift of mean sigma speed against the frame, minimum wins, tie
tolerance from the K = 100 SRS null band). Plots never vote.

| Row | JS to uniform <= 0.005 | Top-source share <= 0.500 | Hard-decel retention >= 1.000 | Near-stationary retention >= 0.500 | Verdict |
|---|---:|---:|---:|---:|---|
| R0_frame | 0.0017 | 0.7413 fail | 1.0000 | 1.0000 | fail |
| R1_guarded | 0.0017 | 0.7435 fail | 1.0030 | 1.0008 | fail |
| M01 | 0.0015 | 0.7412 fail | 0.9274 fail | 0.9776 | fail |
| M02 | 0.0017 | 0.7389 fail | 1.0517 | 0.9838 | fail |
| M03 | 0.0000 | 0.7386 fail | 0.9844 fail | 0.9074 | fail |
| M04 | 0.0004 | 0.7410 fail | 1.0880 | 0.9381 | fail |
| M05 | 0.0000 | 0.2997 | 0.6373 fail | 1.0976 | fail |
| M06 | 0.0411 fail | 0.7727 fail | 1.8030 | 0.0383 fail | fail |
| M07 | 0.0016 | 0.7441 fail | 5.6369 | 0.9744 | fail |
| M08 | 0.0380 fail | 0.6685 fail | 5.6369 | 5.6369 | fail |
| M09 | 0.0000 | 0.8112 fail | 4.7924 | 0.0556 fail | fail |
| M10 | 0.0000 | 0.7699 fail | 3.9738 | 0.4745 fail | fail |
| M11 | 0.0078 fail | 0.7394 fail | 0.9481 fail | 0.7219 | fail |
| **C01** | 0.0000 | 0.5000 | 5.6369 | 0.9873 | **pass** |
| C02 | 0.0000 | 0.5000 | 0.7772 fail | 0.9756 | fail |
| C03 | 0.0000 | 0.7416 fail | 5.6369 | 0.8863 | fail |

A hard-deceleration retention of 5.6369 is the frame-to-budget ratio 402,421 / 71,390,
the value any subset reaches when it keeps every reserve window. C01 is the sole
survivor, so no tie-break was needed. `curation_replicate.py` rebuilds C01 R = 20 times
under the seed family (42, 1200 + r) and evaluates each draw through the harness's own
`compute_row`: all 20 pass the four gates, the tie-break metric has mean 0.1162 km/h,
standard deviation 0.0274 km/h, minimum 0.0798 km/h and maximum 0.1611 km/h, and the
near-stationary retention has mean 0.9839 with standard deviation 0.0107. Because the
reserve is a census, replication varies only the fill, and the reported spread
understates total variance, which the script prints. `curation_reveal.py` opens the
sealed shelf (the two held-out Wasserstein summaries, D5, neutral prior recorded under
D29) and recomputes every displayed metric on the guarded frame, and refuses to run
before the replication artifact exists. `curation_evassive_check.py` reports the D24
evasive retention for every budget-tier artifact (C01 = 0.804, run log only) and is
kept outside the runner on purpose. `curation_feasibility.py` shows that a
regime-balanced subset of one dataset keeps at most 5 x its rarest-regime share (the
pooled frame keeps 88.7 %). `curation_info_dump.py` gathers every owed number,
including the coverage blocks and a best-effort search of the Dingus and Perez PDFs for
the 0.6 g and 0.75 g thresholds (D5, still a manual read).

**Canonical constants (`canonical.py`).** One derivation of every canonical quantity
from the artifacts at import time (`from canonical import C`): `frame_n`, `guarded_n`,
`regime_supplies`, `rarest_regime`, `budget_n`, `regime_quota` 14,278, `capability_n`,
`frame_to_budget` 5.637, `top_source_share`, `d2_sources`, `structural_zeros`,
`decel_trigger_ms2` 5.886, `event_reserve_n`, `evasive_n`, `budget_dir`. `--freeze`
writes the dated snapshot, `--check` compares live values to it and exits 1 on any
drift. The module discovers its sibling modules by marker files and honors
`CANONICAL_MODULE_DIR`. It replaced typed literals that had already diverged in the
tree (402,975 against 402,421).

**Figures (173).** `figures/curation/` (12) holds the winner dossier:
`curation_feasibility.png`, `curation_winner_regime_profile.png`,
`curation_winner_source_composition.png`, `curation_winner_hist_grid.png`,
`curation_winner_ecdf_grid.png`, `curation_winner_trajectory_fan.png`,
`curation_winner_speed_fan.png`, `curation_winner_phase_portraits.png`,
`curation_winner_fidelity.png` (per-regime abs shift of mean sigma speed against the
0.5 km/h P5 limit with the replication spread), `curation_winner_w1_departure.png`
(held-out W1 against the SRS band), `curation_winner_pca.png` (two panels on the frozen
PC plane), and `subset_ecdf_finalists.png` (finalists M01, M05, M07 per D27 plus C01
highlighted). Every C-figure draws the winner in `TUM_SECONDARY["blue2"]` over the frame
as a light-gray ghost. `figures/subset_eval/` (161) holds the Chapter 12 M-slate set:
24 comparative figures at the top level (stacked bar, regime grid, three boxplots,
ridgeline, Wasserstein heatmaps in-model and held-out, source composition, event
retention, bias axes, mean shift, ECDF grid, density grid, PCA grid and PCA grid by
regime, fingerprint, six histogram grids, finalist ECDF overlay), `hist/` (72, six
signals x frame and M01 to M11), `phase/` (26, two portraits x 13 groups), `fans/` (26,
trajectory and speed-change fans x 13 groups), `timeseries/` (13). The C01 pages in the
`subset_eval` dossier folders are surplus produced by the widened frame mask, the
canonical C01 dossier lives in `figures/curation/`.

## Data layout

Separate from the code, not in git. The tree generated 2026-09-07 02:42 listed 742,974
files and 43.2 GB, the stray KITScenes CSV in `parquet_data/` has since been removed,
leaving 742,973 files, every one of them a parquet clip, a derived artifact, or a
manifest. Sizes below are from that tree.

| Layer | Location | What it is | Size |
|---|---|---|---|
| `parquet_data/<ds>/` | `~/Downloads/dataset/parquet_data` | Native-rate per-clip parquet, the pipeline input. Raw conversion from the native formats was a one-off with separate tooling and is not needed to reproduce the analysis. | 4Seasons 30 files 418.0 MB, A2D2 3 files 5.5 MB, Boreas 110 files 3.6 GB, DurLAR 4 files 84.1 MB, Ford 29 files 856.5 MB, HDD 137 files 701.3 MB, KITScenes 1,006 files 14.9 MB, KITTI 156 files 12.9 MB, KITTI-360 10 files 201.7 MB, nuScenes 979 files 258.3 MB, NVIDIA 298,326 files 4.5 GB, Oxford 70 files 122.6 MB, Waymo 70,541 files 1.4 GB, ZOD 29 files 78.2 MB |
| `inertial_data/<ds>/clip_XXXX.parquet` | `.../inertial_data` | The 43-column v4_4 clips on the 10 Hz grid, `processing_meta` and `col_sources` in the file metadata. Larger than the input for the pose-only datasets because 8 input columns become 43. | 4Seasons 30 files 42.8 MB, A2D2 3 files 4.4 MB, Boreas 109 files 394.0 MB, DurLAR 3 files 27.4 MB, Ford 29 files 83.9 MB, HDD 137 files 955.5 MB, KITScenes 1,006 files 75.4 MB, KITTI 156 files 20.4 MB, KITTI-360 10 files 32.7 MB, nuScenes 979 files 76.3 MB, NVIDIA 298,326 files 23.4 GB, Oxford 67 files 411.2 MB, Waymo 70,541 files 5.2 GB, ZOD 29 files 22.9 MB |
| `std_data/<ds>.parquet` | `.../std_data` | The windowed layer, 14 files, single source of truth for all window analysis. | 34 MB in total, NVIDIA 24.3 MB, Waymo 6.5 MB, HDD 1.5 MB, the rest under 600 KB each |
| `regime_data/*.parquet` | `.../regime_data` | 33 parquet artifacts, the 21 regime artifacts and the 12 bias artifacts listed below. | 28.1 MB |
| `subset_data/` | `.../subset_data` | Capability-tier subsets, summaries, tables, manifests. | 14 subset parquets of 2.5 to 5.6 MB each, `window_summaries.parquet` 18.6 MB |
| `subset_data/tier_n71390/` | | The budget tier, same file set at N = 71,390. | 14 subset parquets of 1.1 to 1.3 MB each |
| `subset_data/_summaries_partial/` | | Per-dataset partials of the summary pass, reused only on an exact key match. | 14 files, NVIDIA 14.2 MB |
| `subset_data/eval/` | | The shared eval caches, see below. | `fan_frames.parquet` 150.9 MB, `dist_cache_budget.parquet` 11.4 MB, the rest small |
| `canonical_snapshot.json` | `~/Downloads/dataset` | The frozen oracle, written by `canonical.py --freeze`. Rebuilt 2026-09-07 after the 2026-08-30 original was lost, verified against the artifacts before writing. Do not delete, re-freeze only after a clean `--check`. | 1 file, 19 constants |

**`regime_data/` inventory (33 files).** Regime discovery: `feature_gate_vector`,
`feature_gate_stats`, `cluster_assignments`, `cluster_profile`, `kselect_curves`,
`pca_variance`, `validity_indices`, `bootstrap_jaccard`, `consensus_matrix`,
`consensus_rows`, `reseed_stability`, `resample_stability`, `cap_sensitivity_caps`,
`cap_sensitivity_rank`, `cap_sensitivity_drift`, `gmm_selection`, `gmm_assignments`,
`gmm_components_2d`, `hmm_transitions`, `hmm_durations`, `hmm_states`. Bias
quantification: `bias_per_dataset`, `bias_corpus_profiles`, `bias_corpus_scalars`,
`bias_sparse_regimes`, `bias_diversity`, `bias_volume`, `bias_bootstrap_ci`,
`bias_gmm_entropy`, `bias_shape`, `bias_regime_supply`, `bias_fingerprint`,
`bias_metric_agreement`.

**`subset_data/` and `subset_data/tier_n71390/` inventory.** `M01_srs.parquet` to
`M11_neyman_allocation.parquet`, `C01_event_capped_equal.parquet`,
`C02_capped_equal.parquet`, `C03_event_equal.parquet`, `supply_table.csv`,
`allocation_table.csv`, `manifest.json` (M-slate run metadata), `curation_manifest.json`
(C-slate caps, breaches, forced counts). The capability directory additionally holds
`window_summaries.parquet` and `_summaries_partial/`.

**`subset_data/eval/` inventory (15 files).** From `curation_eval.py` through the
harness: `metrics_long.parquet` (every metric, long format, with tier, variant, sealed
flag), `master_table_budget.csv` and `master_table_capability.csv` (displayed metrics,
wide, with the R0 and R1 rows), `srs_envelope_budget.parquet` (K = 100 SRS draws),
`tautology_map.csv`, `dist_cache_budget.parquet` (sampled values for the distribution
figures, seed 777), `run_meta.json` (metric version 12.1.2, `score_p99_guarded` 8.173,
`k_envelope` 100, `pc_grid_bins` 60, the sealed set, decisions D14 to D17 and D5,
timestamp 2026-08-30 01:03:10). From the frame pass: `frame_hist_fine.parquet`,
`frame_stats.parquet`, `frame_portraits.npz`, `fan_frames.parquet`, `fan_index.parquet`,
`frame_pass_meta.json` (13 groups, `fan_n` 5,000, `fan_seed` 777, `fine_bins` 1,000,
frames per group, timestamp 2026-08-30 01:26:31). From the replication:
`curation_replication_budget.parquet` and `curation_replication_meta.json`.

**Back-reference contract.** `(dataset, clip_id, window_start_s)` keys the assignments,
`(dataset, clip_id, window_idx)` keys the summaries, and `std_data` carries both, so
every join bridges through `std_data` and no script ever re-windows. Frames are
recovered from a window through `(clip_id, window_start_s, window_end_s)` against the
standardized clips. `window_summary.py` normalizes numeric clip ids so `7` and
`clip_7` both resolve to `clip_0007.parquet`.

## Key conventions

- **Terminology.** The four fixed-threshold bins (near-stationary, low, medium, high) are variability tiers, a descriptive single-signal device. Regime is reserved for the five multivariate clusters of RQ2. Sampling methods are M01 to M11 (analysis-track Chapters 11 and 12), curation methods are C01 to C03 (Chapter 13). The docstrings label the modules with the analysis-track numbering (Chapters 7 and 8 for the exploratory module, 9 for regimes, 10 for bias, 11 to 13 for subsets) and some also carry the thesis-section labels 4.5 to 4.7.
- **Windowing (D-2, 2026-08-29).** 20 s windows advanced 20 s, non-overlapping, on the 10 Hz grid, `MIN_VALID_FRAC = 0.5`. The earlier 10 s step is superseded everywhere, and `run_sampling.py` verifies the grid at run time before it would thin anything.
- **Tier thresholds (D-5, 2026-08-29).** Frozen round conventions, speed (1, 5, 15) km/h, acceleration (0.2, 0.5, 1.5) m/s^2, yaw (1, 5, 15) deg/s. Lineage disclosed in the thesis, never recalibrated.
- **Physical-validity bounds.** Speed in [0, 200] km/h, abs yaw rate <= 90 deg/s, per-axis abs acceleration <= 15 m/s^2, per-axis abs jerk <= 30 m/s^3, implied magnitude bounds 21.3 m/s^2 and 42.43 m/s^3. Enforced during standardization with mask closing, and applied again as NaN masking wherever a layer is read. Stored data is never modified.
- **Units.** `yaw_rate` is stored in rad/s and converted by 180 / pi at visualization and in the Wasserstein channels. `max_decel_ms2` is m/s^2. Gravity is 9.81 m/s^2 in `run_sampling.py` and 9.80665 m/s^2 in `curation_common.py` and the harness by documented decision, since unifying would move the 0.6 g trigger (5.886 m/s^2) and the event counts.
- **Logarithm bases (taken from the code).** Shannon entropy in Tier 1 and the GMM entropy are natural (normalized by ln 5), PSI natural, gap statistic natural, BIC natural, Jensen-Shannon base 2, harness entropy in bits (2.3219 bits at uniform).
- **Seeds.** Clustering seed 42 with 10 restarts. Reseed battery 43 to 62, resample battery 242 to 261. Sampling methods `default_rng([42, m])` and scikit-learn seed 4200 + m, C-slate seed indices 12 to 14, replication family (42, 1200 + r), SRS envelope seeds 1000 + k, frame-pass fans and the distribution cache seed 777, EDA subsamples in `density_band_analysis.py` and `trajectory_fan.py` seed 0. Downstream figures read saved artifacts and never re-fit.
- **Compute-once.** Every stage computes into parquet or a cache and every figure reads that cache. No figure recomputes a metric, and the two frame-layer excursions (`window_summary.py`, `subset_frame_pass.py`) are the only passes over `inertial_data/` after the exploratory chapter.
- **Plot style.** Pure default matplotlib, `dpi=300`, TUM palette only (black and white may be literal), `Blues` as the default colormap, no rcParams. Regime colors and labels from `regime_style.py`, dataset colors and display names from `bias_tier2_common.py` and `subset_eval_common.py`.
- **Code style.** All tunables are top-level constants. Parquet loading is parallel (ThreadPoolExecutor in most scripts, ProcessPoolExecutor in `compute_std.py`, `coverage.py`, `distributional_analysis.py`, and `time_series_analysis.py`), matplotlib rendering is single-threaded, tqdm on every long operation. Every runner is fail-fast by default.
- **Numbers.** Period as decimal marker, comma as thousands separator, exact values with units.

## Requirements

Python 3.12. The 8 direct dependencies, pinned exactly in `requirements.txt`:

```
numpy==2.4.4          scipy==1.17.1
pandas==3.0.2         scikit-learn==1.8.0
pyarrow==24.0.0       hmmlearn==0.3.3
matplotlib==3.10.9    tqdm==4.67.3
```

`pip install -r requirements.txt` on a clean machine rebuilds the analysis stack.
hmmlearn is needed by `hmm.py` only. Raw-data conversion (one-off, long since applied) used separate
tooling, including a Docker container for the Waymo TFRecords, and is not required to
reproduce the analysis from `parquet_data/`.

## Not part of the pipeline

`subset_curation/inspect_clip_mapping.py` is a 60-second diagnostic that prints the
clip-id resolution evidence and writes nothing. It is the only non-pipeline script left
in the tree. `Test.py`, `check_tiers.py`, `standardize_nvidia.py`, and
`thesis_audit.py`, which older versions of this file and the runners' skip lists still
name, are no longer present. `nvidia_imu.csv` was a wrongly converted file and is
ignored entirely, the NVIDIA input is the egomotion parquet set only.

## Open verification items and known inconsistencies

These are recorded here so nobody rediscovers them. None of them changes a frozen
number, but each needs one sentence or one edit before submission. Items 4, 5, and 6
are not open questions, they are known-wrong values still standing in draft prose.

1. **`general_info.py` section 4 misreports the gate population.** The complete-case counts themselves are settled (VERIFY-8 closed 2026-09-07 by a direct pass over `std_data`: six-feature 402,293, five-feature 402,421, the 128 readmitted windows confined to KITScenes 127 and Waymo 1). The script still prints 402,421 for both lines while noting 402,293 as expected, because it computes the gate population on the five retained features instead of the six candidates. Fix the line to use the six-feature candidate set, then regenerate `general_info.txt`. Two further defects in the same script surfaced in the 2026-09-07 regeneration and should be fixed in the same pass: section 13 labels its SRS envelope block "per-metric quantiles" but aggregates every metric together with the draw index, so it prints a meaningless mean of 1,082, and section 12 reads `N=?` from both sampling manifests instead of the tier size.
2. **Clip totals.** Three numbers circulate: 371,430 raw parquet clips (the tree), 371,431 (older notes and the writing skill), and 371,425 standardized clips (the audit). The 371,431 was a file count that included the stray KITScenes CSV, which has been deleted, so it no longer matches anything on disk and must not be quoted. The thesis should quote 371,425 for the analyzed corpus and state the five drops.
3. **Native and gridded row counts are different populations.** The native total of 178,778,906 rows is measured over all 371,430 raw clips, the 10 Hz total of 81,252,848 frames over the 371,425 clips that survived standardization, so the two are not a clean before-and-after pair and the thesis must not present them as one. The approximately 142 million figure carried in older notes is wrong and should be struck. Two per-dataset cases also need one sentence each: KITScenes is identical at both layers (181,501), and Waymo has more frames on the 10 Hz grid (14,003,851) than raw rows (14,000,081), which is expected when a source already at roughly 10 Hz is resampled onto a uniform grid but should be stated rather than left for a reader to notice.
4. **HMM persistence attribution.** Section 9 of `general_info.txt` assigns the lowest self-transition (0.748) and the shortest mean dwell (3.9 s) to regime 2 (Turning), not to regime 4 (Accelerating, 0.789 and 4.7 s). Regime 4 is the rarest by supply. Any prose that pairs the 0.748 / 3.9 s figures with Accelerating needs correcting.
5. **Replication numbers.** The artifact values are mean 0.1162 km/h and standard deviation 0.0274 km/h for the tie-break and 0.9873 for C01's near-stationary retention. Earlier notes carry 0.120, 0.024, and 0.989.
6. **Reserve recount label.** `general_info.txt` section 14 prints "guarded 402,421, events >= 5.886 m/s^2: 1,534". That population is the full frame, not the guarded frame. The by-source table in the same section resolves it: 1,534 total, of which 4Seasons contributes 347 and DurLAR 99 while A2D2 contributes none, leaving exactly the 1,088 guarded events. Relabel the line, the count itself is right.
7. **`bootstrap_jaccard` carries a share column that disagrees with the profile.** Its `share_pct` reads 36.00 / 14.82 / 19.36 / 16.12 / 13.70 for regimes 0 to 4, while the cluster profile and the assignments both give 22.22 / 22.24 / 19.60 / 18.20 / 17.74. The Jaccard values themselves are unaffected and no thesis claim rests on that column, but establish which population it counts before any figure or caption quotes it.
8. **Stale docstrings.** `curation_winner_dossier.py` promises `curation_winner_w1_nullband.png` but writes `curation_winner_w1_departure.png`. `subset_frame_pass.py` describes an 11-bit mask over M01 to M10 while `frame_pass_meta.json` records 13 groups. `subset_eval_harness.py` still cites `tier_n71215` and a 402,975-window frame in its header (patched at run time by `curation_eval.py`). `coverage.py` says 13 datasets. `bias_tier2_common.py` lists the retired `bias_pca_map.py`. `run_sampling.py` refers to a `METHODOLOGY.md` that is not in the tree.
9. **Survey filenames.** `2_ filtered_dataset_2nd_stage.csv` carries a space and `3_detailed_sensorr_modalities.csv` a typo. Renaming is safe, no script reads either file.
10. **D5 thresholds.** The 0.6 g longitudinal trigger (Dingus et al. 2006 lineage, 0.75 g in Perez et al. 2017) and the 0.75 g lateral constant are still marked VERIFY against the primary PDFs in `run_sampling.py`, `subset_eval_harness.py`, and `curation_common.py`.
11. **Two low-dynamic definitions.** Tier 1 `low_dynamic_share` is the share of regimes 0 and 1, the Chapter 12 harness `low_dyn_share` is regime 0 only (`LOW_DYN_REGIME = 0`). Both are correct for their chapter, the captions must not mix them.
12. **`regime_style.py` copies.** The `bias_quantification/` copy is the only one exporting `N_REGIMES` and `LOW_DYNAMIC_REGIMES`. Editing a label means editing three files.
13. **Post-run checks still printed by `run_pipeline.py`.** The §7.3 coverage paragraph replacement with the post-cleaning numbers from `audit_baseline.json` is a remaining prose task, and `heatmap_summary.py` should be confirmed to have run its full frame pass rather than the sampled mode for NVIDIA.