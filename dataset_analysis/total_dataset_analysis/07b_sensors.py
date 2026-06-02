import pandas as pd
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tum_colors import TUM_SECONDARY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'dataset_cleaned.csv'))

filtered = df[
    (df['Real / Synthetic / Mixed'] == 0) &
    (df['Original / Derived / Extension / Aggregated'].isin([0, 2]))
].copy()

sensor_cols = ['LiDAR', 'Radar', 'IMU', 'GPS_GNSS']
counts = filtered[sensor_cols].sum().sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(counts.index, counts.values, color=TUM_SECONDARY['blue2'], edgecolor='black')
ax.set_title('Number of Datasets Using Each Sensor Type\n(Real + Original + Extension only)')
ax.set_xlabel('Number of Datasets')
ax.grid(True)
ax.set_axisbelow(True)

plt.tight_layout()
out = os.path.join(BASE_DIR, 'total_dataset_analysis_figures', '07b_sensor_others.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')