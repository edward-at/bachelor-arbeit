import pandas as pd
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tum_colors import TUM_SECONDARY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'filtered_dataset_cleaned.csv'))

counts = df['camera'].value_counts().sort_index()

plt.figure(figsize=(5, 6))
plt.bar(counts.index, counts.values, edgecolor='black', color=TUM_SECONDARY['blue2'])
plt.title('Number of Cameras per Dataset')
plt.xlabel('Number of Cameras')
plt.ylabel('Number of Datasets')
plt.xticks(counts.index)
plt.grid(True)
plt.gca().set_axisbelow(True)
plt.tight_layout()

out = os.path.join(BASE_DIR, 'filtered_dataset_analysis_figures', '01_camera_count.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')