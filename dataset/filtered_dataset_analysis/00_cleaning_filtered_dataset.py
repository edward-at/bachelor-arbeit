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

# ── Clean: imu_grade_top — highest grade per dataset ─────────────────────────
# Hierarchy: Navigation (4) > Tactical (3) > Industrial (2) > Consumer (1)
# Scores noted for future use: Navigation=1.0, Tactical=0.75, Industrial=0.5, Consumer=0.25
GRADE_RANK = {'Navigation': 4, 'Tactical': 3, 'Industrial': 2, 'Consumer': 1}

def highest_grade(val):
    if pd.isna(val) or str(val).strip() in ('nan', 'Unknown', ''):
        return None
    parts = [p.strip() for p in str(val).split(';')]
    ranked = [(GRADE_RANK.get(p, 0), p) for p in parts if p in GRADE_RANK]
    return max(ranked)[1] if ranked else None

df['imu_grade_top'] = df['imu_grade'].apply(highest_grade)

# ── Clean: imu_freq_top — highest IMU frequency per dataset (Hz) ──────────────
# For multi-value entries like '100; 10' or '400; 50', take the maximum
def highest_freq(val):
    if pd.isna(val) or str(val).strip() in ('nan', '0', ''):
        return None
    parts = [p.strip() for p in str(val).split(';')]
    nums = []
    for p in parts:
        try:
            nums.append(float(p))
        except ValueError:
            pass
    return max(nums) if nums else None

df['imu_freq_top'] = df['imu_freq'].apply(highest_freq)
# Note: imu_freq_top is kept for reference but not used in scoring
# Qs = 0.5 * imu_grade + 0.5 * gps_type (frequency removed due to high missing rate)

# ── Clean: gps_gnss_freq_top — highest GPS frequency per dataset (Hz) ─────────
df['gps_gnss_freq_top'] = df['gps_gnss_freq'].apply(highest_freq)

# ── Clean: gps_type ───────────────────────────────────────────────────────────
# Merge RTK and RTK/PPP → RTK, '-' → Unknown
def clean_gps_type(val):
    if pd.isna(val) or str(val).strip() in ('nan', '-', ''):
        return 'Unknown'
    parts = [p.strip() for p in str(val).split(';')]
    cleaned = ['RTK' if 'RTK' in p else p for p in parts]
    return '; '.join(sorted(set(cleaned)))

df['gps_type'] = df['gps_type'].apply(clean_gps_type)

# ── Clean: totd (time of day) ─────────────────────────────────────────────────
df['totd'] = df['totd'].astype(str).str.strip()

# ── Clean: road_type ─────────────────────────────────────────────────────────
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
print(f'\nNaN check (key columns):')
check_cols = ['camera', 'lidar', 'radar', 'object_categories', 'can_bus',
              'imu_grade_top', 'imu_freq_top', 'gps_gnss_freq_top'] + task_cols
print(df[check_cols].isna().sum())