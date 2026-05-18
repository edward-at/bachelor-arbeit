import pandas as pd
import matplotlib.pyplot as plt
import os, sys
from collections import defaultdict
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tum_colors import TUM_SECONDARY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'dataset_cleaned.csv'))

REMOVE = {'Multiple'}
REMAP  = {'UK': 'United Kingdom', 'Hongkong': 'Hong Kong'}

def normalize(name):
    name = name.strip()
    return REMAP.get(name, name)

country_counts = defaultdict(float)
for val in df['Issuing Country'].dropna():
    parts = [normalize(p) for p in val.split(';')]
    parts = [p for p in parts if p not in REMOVE]
    if not parts:
        continue
    share = 1.0 / len(parts)
    for country in parts:
        country_counts[country] += share

sorted_counts = sorted(country_counts.items(), key=lambda x: x[1], reverse=True)
countries, values = zip(*sorted_counts)

fig, ax = plt.subplots(figsize=(10, len(countries) * 0.35 + 1))
ax.barh(list(reversed(countries)), list(reversed(values)), color=TUM_SECONDARY['blue2'], edgecolor='black')
ax.set_title('Number of Datasets per Issuing Country')
ax.set_xlabel('Number of Datasets')
ax.grid(True)
ax.set_axisbelow(True)

plt.tight_layout()
out = os.path.join(BASE_DIR, 'total_dataset_analysis_figures', '05_issuing_country_bar.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')