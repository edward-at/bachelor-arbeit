import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE  = os.path.join(BASE_DIR, 'filtered_dataset.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'filtered_dataset_cleaned.csv')

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE)

# ── Clean: numeric columns ────────────────────────────────────────────────────
# 0 = none / not present / not annotated
for col in ['camera', 'lidar', 'radar', 'object_categories', 'temporal']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

# ── Clean: citation ───────────────────────────────────────────────────────────
# '-' means no citation data → NaN
df['citation'] = pd.to_numeric(df['citation'], errors='coerce')

# ── Clean: weather ────────────────────────────────────────────────────────────
# 0 and '0' mean unknown/not stated → 'Unknown'
df['weather'] = df['weather'].astype(str).str.strip()
df['weather'] = df['weather'].replace({'0': 'Unknown', 'nan': 'Unknown'})

# ── Clean: lidar_beam ─────────────────────────────────────────────────────────
# Keep as string — values like '32; 64' parsed per chart. 0 = no LiDAR
df['lidar_beam'] = df['lidar_beam'].astype(str).str.strip()
df['lidar_beam'] = df['lidar_beam'].replace({'nan': '0'})

# ── Clean: task columns → ensure int ─────────────────────────────────────────
# 1 = supported, 0 = not supported
task_cols = [
    'object_detection', 'multi_object_tracking', 'segmentation',
    'depth_estimation', 'odometry_slam', 'prediction',
    'planning', 'mapping', 'lane_detection',
]
for col in task_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_FILE, index=False)
print(f'Cleaned CSV saved → {OUTPUT_FILE}')
print(f'Shape: {df.shape[0]} rows x {df.shape[1]} columns')
print(f'\nNaN check:')
print(df[['camera', 'lidar', 'radar', 'object_categories', 'temporal'] + task_cols].isna().sum())