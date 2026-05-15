import pandas as pd
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'dataset_cleaned.csv'))

# Filter: Real (0) + Original (0) + Extension (2) only
# Same filter as chart 03
filtered = df[
    (df['Real / Synthetic / Mixed'] == 0) &
    (df['Original / Derived / Extension / Aggregated'].isin([0, 2]))
].copy()

camera_cols = ['Monocular Camera', 'Stereo Camera', 'Fisheye Camera', 'Event Camera', 'Thermal Camera']
counts = filtered[camera_cols].sum().sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(counts.index, counts.values, edgecolor='black')
ax.set_title('Number of Datasets Using Each Camera Type\n(Real + Original + Extension only)')
ax.set_xlabel('Number of Datasets')
ax.grid(True)
ax.set_axisbelow(True)

plt.tight_layout()
out = os.path.join(BASE_DIR, 'total_dataset_analysis_figures', '07a_sensor_cameras.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')