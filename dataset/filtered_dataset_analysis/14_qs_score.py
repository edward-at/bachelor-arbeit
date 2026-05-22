import pandas as pd
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tum_colors import TUM_SECONDARY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'filtered_dataset_cleaned.csv'))

# ── IMU Grade Score ───────────────────────────────────────────────────────────
# Navigation=1.0, Tactical=0.75, Industrial=0.5, Consumer=0.25, Unknown=0
GRADE_SCORE = {'Navigation': 1.0, 'Tactical': 0.75, 'Industrial': 0.5, 'Consumer': 0.25}
df['imu_grade_score'] = df['imu_grade_top'].map(GRADE_SCORE).fillna(0)

# ── GPS Type Score ────────────────────────────────────────────────────────────
# RTK=1.0, Standard=0.0, Unknown=0.0
df['gps_score'] = df['gps_type'].map({'RTK': 1.0, 'Standard': 0.0, 'Unknown': 0.0}).fillna(0)

# ── IMU Frequency Score ───────────────────────────────────────────────────────
# >200=1.0, 100-199=0.67, 50-99=0.33, 1-49=0.17, unknown/0=0
def imu_hz_score(val):
    if pd.isna(val) or val == 0:
        return 0
    if val > 200:
        return 1.0
    elif val >= 100:
        return 0.67
    elif val >= 50:
        return 0.33
    else:
        return 0.17

df['imu_hz_score'] = df['imu_freq_top'].apply(imu_hz_score)

# ── Qs = 0.4*imu_grade + 0.4*gps_type + 0.2*imu_hz ──────────────────────────
df['Qs'] = 0.4 * df['imu_grade_score'] + 0.4 * df['gps_score'] + 0.2 * df['imu_hz_score']

# Sort by Qs descending
df_sorted = df[['dataset', 'Qs']].sort_values('Qs')

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.figure(figsize=(12, 10))
plt.barh(df_sorted['dataset'], df_sorted['Qs'],
         edgecolor='black', color=TUM_SECONDARY['blue2'])
plt.xlim(0, 1)
plt.title('Signal Quality Score (Qs) per Dataset')
plt.xlabel('Qs Score (0–1)')
plt.grid(True)
plt.gca().set_axisbelow(True)
plt.tight_layout()

out = os.path.join(BASE_DIR, 'filtered_dataset_analysis_figures', '14_qs_score.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')
print()
print(df[['dataset', 'imu_grade_score', 'gps_score', 'imu_hz_score', 'Qs']].sort_values('Qs', ascending=False).to_string())