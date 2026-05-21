import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE  = os.path.join(BASE_DIR, 'filtered_dataset.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'filtered_dataset_cleaned.csv')

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE)

# ── Clean: citation ───────────────────────────────────────────────────────────
# '-' means no citation data → NaN
df['Citation'] = pd.to_numeric(df['Citation'], errors='coerce')

# ── Clean: numeric count columns ─────────────────────────────────────────────
# 0 = none / not present
for col in ['camera', 'lidar', 'radar', 'object_categories']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

# ── Clean: total_duration ─────────────────────────────────────────────────────
# Keep as float, NaN = not documented
df['total_duration'] = pd.to_numeric(df['total_duration'], errors='coerce')

# ── Clean: can_bus → 1/0 ─────────────────────────────────────────────────────
# 1 = Yes, 0 = No
df['can_bus'] = df['can_bus'].map({'Yes': 1, 'No': 0})

# ── Clean: weather ────────────────────────────────────────────────────────────
# 0 and '0' mean unknown/not stated → 'Unknown'
df['weather'] = df['weather'].astype(str).str.strip()
df['weather'] = df['weather'].replace({'0': 'Unknown', 'nan': 'Unknown'})

# ── Clean: lidar_beam ─────────────────────────────────────────────────────────
# Keep as string — values like '32; 64' parsed per chart. 0 = no LiDAR
df['lidar_beam'] = df['lidar_beam'].astype(str).str.strip()
df['lidar_beam'] = df['lidar_beam'].replace({'nan': '0'})

# ── Clean: imu_freq ───────────────────────────────────────────────────────────
# Keep as string — multi-value entries like '100; 10' parsed per chart
df['imu_freq'] = df['imu_freq'].astype(str).str.strip()
df['imu_freq'] = df['imu_freq'].replace({'nan': '0'})

# ── Clean: imu_grade ──────────────────────────────────────────────────────────
# Keep as string — multi-value entries like 'Tactical; Industrial' parsed per chart
df['imu_grade'] = df['imu_grade'].astype(str).str.strip()
df['imu_grade'] = df['imu_grade'].replace({'nan': 'Unknown'})

# ── Clean: gps_gnss_freq ──────────────────────────────────────────────────────
# Strip trailing spaces, keep as string — multi-value entries parsed per chart
df['gps_gnss_freq'] = df['gps_gnss_freq'].astype(str).str.strip()
df['gps_gnss_freq'] = df['gps_gnss_freq'].replace({'nan': '0'})

# ── Clean: gps_type ───────────────────────────────────────────────────────────
# Keep as string — '-' means not documented
df['gps_type'] = df['gps_type'].astype(str).str.strip()
df['gps_type'] = df['gps_type'].replace({'nan': 'Unknown', '-': 'Unknown'})

# ── Clean: totd (time of day) ─────────────────────────────────────────────────
# Keep as string — 'Day' or 'Day; Night'
df['totd'] = df['totd'].astype(str).str.strip()

# ── Clean: road_type ─────────────────────────────────────────────────────────
# Keep as string — semicolon-separated categories parsed per chart
df['road_type'] = df['road_type'].astype(str).str.strip()

# ── Clean: task columns → ensure int ─────────────────────────────────────────
# 1 = supported, 0 = not supported
task_cols = [
    'Object Detection (2D / 3D)',
    'Multi-Object Tracking (2D / 3D)',
    'Segmentation (Semantic / Instance / Panoptic / 3D Point Cloud)',
    'Depth Estimation',
    'Odometry /SLAM',
    'Motion / Trajectory Prediction',
    'Planning (Behavior / Motion)',
    'HD / Online Mapping',
    'Lane Detection',
]
for col in task_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_FILE, index=False)
print(f'Cleaned CSV saved → {OUTPUT_FILE}')
print(f'Shape: {df.shape[0]} rows x {df.shape[1]} columns')
print(f'\nNaN check:')
check_cols = ['camera', 'lidar', 'radar', 'object_categories', 'can_bus'] + task_cols
print(df[check_cols].isna().sum())