import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
import os
from collections import defaultdict


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'dataset_cleaned.csv'))

# ── Filter: exclude Synthetic (1) and Mixed (2) for recording map ─────────────
df_map = df[df['Real / Synthetic / Mixed'] == 0].copy()

# ── Cleaning rules for Recording Country ─────────────────────────────────────
REMOVE   = {'Multiple', 'Europe', 'Mixed', 'Synthetic', 'Global'}
REMAP    = {
    'North America': 'USA',
    'UK':            'United Kingdom',
}

def normalize(name):
    name = name.strip()
    if name in REMAP:
        return REMAP[name]
    return name

# ── Equal split for multi-country entries ─────────────────────────────────────
# e.g. "USA; Germany; Japan" → each gets 1/3
country_counts = defaultdict(float)

for val in df_map['Recording Country'].dropna():
    parts = [normalize(p) for p in val.split(';')]
    parts = [p for p in parts if p not in REMOVE]
    if not parts:
        continue
    share = 1.0 / len(parts)
    for country in parts:
        country_counts[country] += share

# ── For the count table: all datasets (Real + Synthetic + Mixed) ──────────────
country_counts_all = defaultdict(float)
for val in df['Recording Country'].dropna():
    parts = [normalize(p) for p in val.split(';')]
    parts = [p for p in parts if p not in REMOVE]
    if not parts:
        continue
    share = 1.0 / len(parts)
    for country in parts:
        country_counts_all[country] += share

# ── Map country names to shapefile names ──────────────────────────────────────
NAME_TO_SHP = {
    'USA':          'United States of America',
    'Czech Republic': 'Czechia',
    'Hong Kong':    'China',       # not in shapefile, merged into China
    'Singapore':    'Malaysia',    # not in shapefile, nearest proxy
    'UAE':          'United Arab Emirates',
}
def to_shp_name(name):
    return NAME_TO_SHP.get(name, name)

# ── Load world shapefile ──────────────────────────────────────────────────────
SHP = r'C:\Users\edwar\Downloads\bachelor-arbeit\.venv\Lib\site-packages\pyogrio\tests\fixtures\naturalearth_lowres\naturalearth_lowres.shp'
world = gpd.read_file(SHP)
world['count'] = world['name'].map(
    {to_shp_name(k): v for k, v in country_counts.items()}
).fillna(0)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax_map, ax_table) = plt.subplots(1, 2, figsize=(16, 7),
                                        gridspec_kw={'width_ratios': [2, 1]})

# Heatmap
world.plot(column='count', ax=ax_map, cmap='YlOrRd',
           linewidth=0.4, edgecolor='grey',
           legend=True,
           missing_kwds={'color': 'lightgrey'},
           legend_kwds={'label': 'Number of datasets recorded', 'shrink': 0.6})
ax_map.set_title('Recording Country Heatmap\n(Real datasets only)')
ax_map.set_axis_off()

# Count table (all datasets)
sorted_counts = sorted(country_counts_all.items(), key=lambda x: x[1], reverse=True)
table_text = [['Country', 'Count']] + [[c, f'{v:.1f}'] for c, v in sorted_counts]
ax_table.axis('off')
table = ax_table.table(cellText=table_text[1:],
                       colLabels=table_text[0],
                       loc='center',
                       cellLoc='left')
table.auto_set_font_size(True)
table.scale(1, 1.2)
ax_table.set_title('Dataset Count per Country\n(All datasets)', pad=12)

plt.tight_layout()
out = os.path.join(BASE_DIR, 'total_dataset_analysis_figures', '04_recording_country.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')