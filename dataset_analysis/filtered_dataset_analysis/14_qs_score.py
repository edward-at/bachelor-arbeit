import pandas as pd
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tum_colors import TUM_SECONDARY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'filtered_dataset_cleaned.csv'))

# ── Qs = 0.5 * imu_grade + 0.5 * gps_type ────────────────────────────────────
# IMU frequency removed from scoring due to high missing rate (6/34 undocumented)
GRADE_SCORE = {'Navigation': 1.0, 'Tactical': 0.75, 'Industrial': 0.5, 'Consumer': 0.25}
df['imu_grade_score'] = df['imu_grade_top'].map(GRADE_SCORE).fillna(0)
df['gps_score'] = df['gps_type'].map({'RTK': 1.0, 'Standard': 0.0, 'Unknown': 0.0}).fillna(0)
df['Qs'] = 0.5 * df['imu_grade_score'] + 0.5 * df['gps_score']

df_sorted = df[['dataset', 'Qs']].sort_values('Qs', ascending=True)

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
print(df[['dataset', 'imu_grade_score', 'gps_score', 'Qs']].sort_values('Qs', ascending=False).to_string())