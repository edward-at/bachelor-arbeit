import pandas as pd
import matplotlib.pyplot as plt
import os, sys
from collections import Counter
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tum_colors import TUM_SECONDARY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'filtered_dataset_cleaned.csv'))

beam_counts = Counter()
for val in df['lidar_beam'].dropna():
    if val == '0':
        continue
    for b in val.split(';'):
        b = b.strip()
        if b != '0':
            beam_counts[int(b)] += 1

sorted_counts = dict(sorted(beam_counts.items()))
labels = [str(k) for k in sorted_counts.keys()]

plt.figure(figsize=(5, 6))
plt.bar(labels, list(sorted_counts.values()), edgecolor='black', color=TUM_SECONDARY['blue2'])
plt.title('LiDAR Beam Count Distribution')
plt.xlabel('Number of Beams')
plt.ylabel('Count')
plt.xticks(labels)
plt.grid(True)
plt.gca().set_axisbelow(True)
plt.tight_layout()

out = os.path.join(BASE_DIR, 'filtered_dataset_analysis_figures', '07_lidar_beam.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')