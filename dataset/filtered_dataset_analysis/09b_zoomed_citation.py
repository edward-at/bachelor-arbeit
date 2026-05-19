import pandas as pd
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tum_colors import TUM_SECONDARY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'filtered_dataset_cleaned.csv'))
df_sorted = df[['dataset', 'citation']].dropna().sort_values('citation')

# Exclude top 4 dominant datasets for readability
EXCLUDE = {'KITTI', 'nuScenes', 'Waymo Open Dataset', 'BDD100K (Berkley Deep Drive 100k)', 'Oxford RobotCar', 'KITTI-360'}
df_zoom = df_sorted[~df_sorted['dataset'].isin(EXCLUDE)]

plt.figure(figsize=(15, 10))
plt.barh(df_zoom['dataset'], df_zoom['citation'],
         edgecolor='black', color=TUM_SECONDARY['blue2'])
plt.xlim(0, 700)
plt.title('Citation Count per Dataset (Excluding Top 6, Zoomed 0–700)')
plt.xlabel('Number of Citations')
plt.grid(True)
plt.gca().set_axisbelow(True)
plt.tight_layout()

out = os.path.join(BASE_DIR, 'filtered_dataset_analysis_figures', '09b_citation_zoom.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')