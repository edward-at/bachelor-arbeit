import pandas as pd
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tum_colors import TUM_SECONDARY, TUM_EXT

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'dataset_cleaned.csv'))

filtered = df[
    (df['Real / Synthetic / Mixed'] == 0) &
    (df['Original / Derived / Extension / Aggregated'] == 0) |
    (df['Original / Derived / Extension / Aggregated'] == 2)
].copy()

sensor_cols = [
    'Monocular Camera', 'Stereo Camera', 'Fisheye Camera',
    'Event Camera', 'Thermal Camera', 'LiDAR', 'Radar', 'IMU', 'GPS_GNSS'
]
filtered['sensor_count'] = filtered[sensor_cols].sum(axis=1)

stats = (filtered.groupby('Year')['sensor_count']
                 .agg(['mean', 'count'])
                 .reindex(range(2006, 2027), fill_value=0))
stats.columns = ['mean', 'count']
valid = stats[stats['count'] > 0]

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()
ax1.grid(True, alpha=0.4, zorder=1)
ax1.set_axisbelow(True)
ax2.bar(stats.index, stats['count'], color=TUM_SECONDARY['blue2'], edgecolor='black', label='Number of Datasets', zorder=2, alpha=0.5)
ax1.plot(valid.index, valid['mean'], color=TUM_EXT['red'], linewidth=2, marker='o', markerfacecolor='white', markeredgecolor=TUM_EXT['red'], label='Average Sensors', zorder=3)
ax1.set_title('Average Sensor Count per Year (Real & Original+Extension Datasets only)')
ax1.set_xlabel('Year')
ax1.set_ylabel('Average Number of Sensors')
ax2.set_ylabel('Number of Datasets')
ax1.set_ylim(0, 6)
ax1.set_xlim(2005.5, 2026.5)
ax1.set_xticks(list(range(2006, 2027)))
ax1.set_xticklabels(list(range(2006, 2027)), rotation=45, ha='right')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
ax2.set_ylim(0, stats['count'].max() * 3)
plt.tight_layout()

out = os.path.join(BASE_DIR, 'total_dataset_analysis_figures', '03_avg_sensors_per_year.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')