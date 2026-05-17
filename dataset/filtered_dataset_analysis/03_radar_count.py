import pandas as pd
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'filtered_dataset_cleaned.csv'))

counts = df['radar'].value_counts().sort_index()

plt.figure(figsize=(5, 6))
plt.bar(counts.index, counts.values, edgecolor='black')
plt.title('Number of Radars per Dataset')
plt.xlabel('Number of Radars')
plt.ylabel('Number of Datasets')
plt.xticks(counts.index)
plt.grid(True)
plt.gca().set_axisbelow(True)
plt.tight_layout()

out = os.path.join(BASE_DIR, 'filtered_dataset_analysis_figures', '03_radar_count.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')