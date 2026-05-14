import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'total_dataset_pool.csv'))

# Real / Synthetic / Mixed
# 0 = Real, 1 = Synthetic, 2 = Mixed
df['Real / Synthetic / Mixed'] = df['Real / Synthetic / Mixed'].map(
    {'Real': 0, 'Synthetic': 1, 'Mixed': 2}
)

# Public Availability
# 0 = Public-Free, 1 = Public-Registration, 2 = Not-Available
df['Public Availability'] = df['Public Availability'].map(
    {'Public-Free': 0, 'Public-Registration': 1, 'Not-Available': 2}
)

# Original / Derived / Extension / Aggregated
# 0 = Original, 1 = Derived, 2 = Extension, 3 = Aggregated
df['Original / Derived / Extension / Aggregated'] = df['Original / Derived / Extension / Aggregated'].map(
    {'Original': 0, 'Derived': 1, 'Extension': 2, 'Aggregated': 3}
)

# Sensors & vehicle info: Yes = 1, No = 0
binary_cols = [
    'Monocular Camera', 'Stereo Camera', 'Fisheye Camera',
    'Event Camera', 'Thermal Camera', 'LiDAR', 'Radar', 'IMU', 'GPS_GNSS',
    'Ego-motion IMU / INS', 'CAN Bus / Vehicle Odometry', 'Ego Pose / Trajectory',
]
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

df.to_csv(os.path.join(BASE_DIR, 'dataset_cleaned.csv'), index=False)
print(f'Done. Shape: {df.shape}')
print(f'NaN check:\n{df.iloc[:, 7:].isna().sum()}')