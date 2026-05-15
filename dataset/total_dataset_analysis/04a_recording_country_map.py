import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
import os
import pyogrio
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'dataset_cleaned.csv'))

# ── Filter: Real only for recording map ──────────────────────────────────────
df_map = df[df['Real / Synthetic / Mixed'] == 0].copy()

# ── Cleaning rules ────────────────────────────────────────────────────────────
REMOVE = {'Multiple', 'Europe', 'Mixed', 'Synthetic', 'Global'}
REMAP  = {'North America': 'USA', 'UK': 'United Kingdom'}

def normalize(name):
    name = name.strip()
    return REMAP.get(name, name)

# ── Equal split for multi-country entries ─────────────────────────────────────
country_counts = defaultdict(float)
for val in df_map['Recording Country'].dropna():
    parts = [normalize(p) for p in val.split(';')]
    parts = [p for p in parts if p not in REMOVE]
    if not parts:
        continue
    share = 1.0 / len(parts)
    for country in parts:
        country_counts[country] += share

# ── Map to shapefile names ────────────────────────────────────────────────────
NAME_TO_SHP = {
    'USA':            'United States of America',
    'Czech Republic': 'Czechia',
    'Hong Kong':      'China',
    'Singapore':      'Malaysia',
    'UAE':            'United Arab Emirates',
}
def to_shp_name(name):
    return NAME_TO_SHP.get(name, name)

# ── Load shapefile ────────────────────────────────────────────────────────────
SHP = os.path.join(os.path.dirname(pyogrio.__file__), 'tests', 'fixtures', 'naturalearth_lowres', 'naturalearth_lowres.shp')
world = gpd.read_file(SHP)
world['count'] = world['name'].map(
    {to_shp_name(k): v for k, v in country_counts.items()}
).fillna(0)

# ── Plot with LOG scale (change norm=mcolors.LogNorm(vmin=1) to remove for linear) ──
fig, ax = plt.subplots(figsize=(14, 7))
world.plot(column='count', ax=ax, cmap='YlOrRd',
           linewidth=0.4, edgecolor='grey',
           legend=True,
           norm=mcolors.LogNorm(vmin=1, vmax=world['count'].max()),  # ← remove this line for linear scale
           missing_kwds={'color': 'lightgrey'},
           legend_kwds={'label': 'Number of datasets (log scale)', 'shrink': 0.6})  # ← change label for linear
ax.set_title('Recording Country Heatmap (log scale)')
ax.set_axis_off()

plt.tight_layout()
out = os.path.join(BASE_DIR, 'total_dataset_analysis_figures', '04a_recording_country_map.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')