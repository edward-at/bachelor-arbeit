import pandas as pd
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tum_colors import TUM_SECONDARY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'filtered_dataset_cleaned.csv'))

# imu_freq_top already contains the highest IMU frequency per dataset
# NaN = not documented, excluded
counts = df['imu_freq_top'].dropna().value_counts().sort_index()

plt.figure(figsize=(10, 6))
plt.bar(counts.index.astype(str), counts.values, edgecolor='black', color=TUM_SECONDARY['blue2'])
plt.title('IMU Frequency Distribution (Highest Frequency per Dataset)')
plt.xlabel('IMU Frequency (Hz)')
plt.ylabel('Number of Datasets')
plt.grid(True)
plt.gca().set_axisbelow(True)
plt.tight_layout()

out = os.path.join(BASE_DIR, 'filtered_dataset_analysis_figures', '12_imu_frequency.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')