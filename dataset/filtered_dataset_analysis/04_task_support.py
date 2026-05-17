import pandas as pd
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'filtered_dataset_cleaned.csv'))

task_cols = {
    'object_detection':       'Object Detection',
    'multi_object_tracking':  'Multi-Object Tracking',
    'segmentation':           'Segmentation',
    'depth_estimation':       'Depth Estimation',
    'odometry_slam':          'Odometry / SLAM',
    'prediction':             'Motion / Trajectory Prediction',
    'planning':               'Planning',
    'mapping':                'HD / Online Mapping',
    'lane_detection':         'Lane Detection',
}

counts = {label: df[col].sum() for col, label in task_cols.items()}
counts = dict(sorted(counts.items(), key=lambda x: x[1]))

plt.figure(figsize=(10, 6))
plt.barh(list(counts.keys()), list(counts.values()), edgecolor='black')
plt.title('Task Support Coverage')
plt.xlabel('Number of Datasets')
plt.grid(True)
plt.gca().set_axisbelow(True)
plt.tight_layout()

out = os.path.join(BASE_DIR, 'filtered_dataset_analysis_figures', '04_task_support.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')