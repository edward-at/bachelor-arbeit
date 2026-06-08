import pandas as pd
import matplotlib.pyplot as plt
import os, sys
from collections import defaultdict
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tum_colors import TUM_SECONDARY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'filtered_dataset_cleaned.csv'))

# Keyword to group mapping
# 'traffic' and 'maneuvers' excluded — too vague to categorize
ROAD_MAP = {
    'urban':        'Urban',
    'city':         'Urban',
    'downtown':     'Urban',
    'residential':  'Urban',
    'suburban':     'Urban',
    'suburb':       'Urban',
    'suburbs':      'Urban',
    'highway':      'Highway',
    'freeway':      'Highway',
    'skyway':       'Highway',
    'regional':     'Highway',
    'rural':        'Rural',
    'country':      'Rural',
    'countryside':  'Rural',
    'farm':         'Rural',
    'forest':       'Rural',
    'nature':       'Rural',
    'hills':        'Rural',
    'campus':       'Controlled',
    'airport':      'Controlled',
    'parking':      'Controlled',
    'bridge':       'Infrastructure',
    'tunnel':       'Infrastructure',
    'construction': 'Infrastructure',
    'industrial':   'Infrastructure',
    'structures':   'Infrastructure',
}
EXCLUDE = {'traffic', 'maneuvers'}

# Equal split per dataset across matched groups
road_counts = defaultdict(float)
for val in df['road_type'].dropna():
    if str(val).strip() in ('nan', ''):
        continue
    tokens = [t.strip().lower() for t in val.split(';')]
    matched = list({ROAD_MAP[t] for t in tokens if t in ROAD_MAP})
    if not matched:
        continue
    share = 1.0 / len(matched)
    for group in matched:
        road_counts[group] += share

# Sort descending
order = ['Urban', 'Highway', 'Rural', 'Controlled', 'Infrastructure']
sorted_counts = {g: road_counts.get(g, 0) for g in order if road_counts.get(g, 0) > 0}

plt.figure(figsize=(10, 5))
plt.barh(list(sorted_counts.keys()), list(sorted_counts.values()),
         edgecolor='black', color=TUM_SECONDARY['blue2'])
plt.title('Road Type Coverage')
plt.xlabel('Number of Datasets (weighted)')
plt.grid(True)
plt.gca().set_axisbelow(True)
plt.tight_layout()

out = os.path.join(BASE_DIR, 'filtered_dataset_analysis_figures', '13_road_type.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')