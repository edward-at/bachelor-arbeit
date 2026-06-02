import pandas as pd
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tum_colors import TUM_SECONDARY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'filtered_dataset_cleaned.csv'))

task_cols = {
    'Object Detection (2D / 3D)':                                      'Object Detection',
    'Multi-Object Tracking (2D / 3D)':                                 'Multi-Object Tracking',
    'Segmentation (Semantic / Instance / Panoptic / 3D Point Cloud)':  'Segmentation',
    'Depth Estimation':                                                 'Depth Estimation',
    'Odometry /SLAM':                                                   'Odometry / SLAM',
    'Motion / Trajectory Prediction':                                   'Motion Prediction',
    'Planning (Behavior / Motion)':                                     'Planning',
    'HD / Online Mapping':                                              'HD / Online Mapping',
    'Lane Detection':                                                   'Lane Detection',
}

counts = {label: df[col].sum() for col, label in task_cols.items()}
counts = dict(sorted(counts.items(), key=lambda x: x[1]))

plt.figure(figsize=(10, 6))
plt.barh(list(counts.keys()), list(counts.values()), edgecolor='black', color=TUM_SECONDARY['blue2'])
plt.title('Task Support Coverage')
plt.xlabel('Number of Datasets')
plt.grid(True)
plt.gca().set_axisbelow(True)
plt.tight_layout()

out = os.path.join(BASE_DIR, 'filtered_dataset_analysis_figures', '04_task_support.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')