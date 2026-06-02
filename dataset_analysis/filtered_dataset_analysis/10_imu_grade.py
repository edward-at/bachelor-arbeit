import pandas as pd
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tum_colors import TUM_SECONDARY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'filtered_dataset_cleaned.csv'))

# imu_grade_top already contains the highest grade per dataset
order = ['Consumer', 'Industrial', 'Tactical', 'Navigation']
counts = df['imu_grade_top'].value_counts()
counts = counts.reindex([g for g in order if g in counts.index])

plt.figure(figsize=(10, 4))
plt.barh(counts.index, counts.values, edgecolor='black', color=TUM_SECONDARY['blue2'])
plt.title('IMU Grade Distribution (Highest Grade per Dataset)')
plt.xlabel('Number of Datasets')
plt.grid(True)
plt.gca().set_axisbelow(True)
plt.tight_layout()

out = os.path.join(BASE_DIR, 'filtered_dataset_analysis_figures', '10_imu_grade.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')