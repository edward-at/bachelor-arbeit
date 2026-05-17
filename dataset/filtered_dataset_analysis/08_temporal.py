import pandas as pd
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'filtered_dataset_cleaned.csv'))

counts = df['temporal'].value_counts().sort_index()
labels = {0: 'Static', 1: 'Video'}
names = [labels[k] for k in counts.index]

plt.figure()
plt.pie(counts.values, labels=names, autopct='%1.1f%%', wedgeprops={'edgecolor': 'black'}, colors=['C1', 'C0'])
plt.title('Temporal Coverage (Video vs Static)')
plt.tight_layout()

out = os.path.join(BASE_DIR, 'filtered_dataset_analysis_figures', '08_temporal.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')
