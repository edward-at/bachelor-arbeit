import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'dataset_cleaned.csv'))

# ── Filter: Real (0) + Original (0) only ─────────────────────────────────────
# Synthetic excluded: no physical sensors, would drag average down artificially
# Mixed excluded: unknown real/synthetic ratio per dataset, uncontrollable variable
# Derived/Extension/Aggregated excluded: reuse existing sensors, causes double counting

filtered = df[
    (df['Real / Synthetic / Mixed'] == 0) &
    (df['Original / Derived / Extension / Aggregated'] == 0)
].copy()

# ── Compute sensor count per dataset ─────────────────────────────────────────
sensor_cols = [
    'Monocular Camera', 'Stereo Camera', 'Fisheye Camera',
    'Event Camera', 'Thermal Camera', 'LiDAR', 'Radar', 'IMU', 'GPS_GNSS'
]
filtered['sensor_count'] = filtered[sensor_cols].sum(axis=1)

# ── Aggregate per year ────────────────────────────────────────────────────────
stats = (filtered.groupby('Year')['sensor_count']
                 .agg(['mean', 'count'])
                 .reindex(range(df['Year'].min(), filtered['Year'].max() + 1), fill_value=0))
stats.columns = ['mean', 'count']

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(14, 6))

# Background bars: number of datasets per year (secondary context)
ax2 = ax1.twinx()
ax2.bar(stats.index, stats['count'], color='#d0d0d0', width=0.6, zorder=2, label='# datasets')
ax2.set_ylabel('Number of Datasets (Real + Original)', fontsize=10, color='#999999')
ax2.tick_params(axis='y', colors='#999999', labelsize=9)
ax2.spines[['top', 'right']].set_color('#d0d0d0')
ax2.set_ylim(0, stats['count'].max() * 3.5)

# Line: average sensor count (only for years with at least 1 dataset)
valid = stats[stats['count'] > 0]
ax1.plot(valid.index, valid['mean'],
         color='#1f77b4', linewidth=2.5, marker='o', markersize=5, zorder=4, label='Avg sensors')

# Value labels on each point
for year, row in valid.iterrows():
    ax1.text(year, row['mean'] + 0.1, f"{row['mean']:.1f}",
             ha='center', va='bottom', fontsize=7.5, color='#1f77b4')

# ── Styling ───────────────────────────────────────────────────────────────────
ax1.set_title('Average Sensor Count per Year\n(Real + Original datasets only, out of 9 sensors)',
              fontsize=14, fontweight='bold', pad=12)
ax1.set_xlabel('Year', fontsize=11)
ax1.set_ylabel('Average Number of Sensors (out of 9)', fontsize=11)
ax1.set_ylim(0, 9)
ax1.yaxis.grid(True, linestyle='--', linewidth=0.7, alpha=0.7, zorder=0)
ax1.set_axisbelow(True)
ax1.spines[['top', 'right']].set_visible(False)
ax1.xaxis.set_major_locator(mticker.MultipleLocator(1))
ax1.set_xlim(2005.5, 2026.5)
plt.xticks(rotation=45, ha='right', fontsize=9)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, frameon=False, loc='upper left')

plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
out = os.path.join(BASE_DIR, 'total_dataset_analysis_figures', '03_avg_sensors_per_year.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')