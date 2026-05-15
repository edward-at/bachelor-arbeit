import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'dataset_cleaned.csv'))

# ── All datasets (Real + Synthetic + Mixed) for bar chart ────────────────────
REMOVE = {'Multiple', 'Europe', 'Global'}
REMAP  = {'North America': 'USA', 'UK': 'United Kingdom'}

def normalize(name):
    name = name.strip()
    return REMAP.get(name, name)

country_counts = defaultdict(float)
for val in df['Recording Country'].dropna():
    parts = [normalize(p) for p in val.split(';')]
    parts = [p for p in parts if p not in REMOVE]
    if not parts:
        continue
    share = 1.0 / len(parts)
    for country in parts:
        country_counts[country] += share

# ── Sort descending ───────────────────────────────────────────────────────────
sorted_counts = sorted(country_counts.items(), key=lambda x: x[1], reverse=True)
countries, values = zip(*sorted_counts)

fig, ax = plt.subplots(figsize=(10, len(countries) * 0.35 + 1))
ax.barh(list(reversed(countries)), list(reversed(values)), edgecolor='black')
ax.set_title('Number of Datasets per Recording Country')
ax.set_xlabel('Number of Datasets')
ax.grid(True)
ax.set_axisbelow(True)

plt.tight_layout()
out = os.path.join(BASE_DIR, 'total_dataset_analysis_figures', '04b_recording_country_bar.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')