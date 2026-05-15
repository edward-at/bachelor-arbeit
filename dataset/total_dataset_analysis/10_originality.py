import pandas as pd
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'dataset_cleaned.csv'))

counts = df['Original / Derived / Extension / Aggregated'].value_counts().sort_index()
labels = {0: 'Original', 1: 'Derived', 2: 'Extension', 3: 'Aggregated'}
sizes = [counts.get(k, 0) for k in [0, 1, 2, 3]]
names = [labels[k] for k in [0, 1, 2, 3]]

plt.figure()
plt.pie(sizes, labels=names, autopct='%1.1f%%', wedgeprops={'edgecolor': 'black'})
plt.title('Original / Derived / Extension / Aggregated Datasets')
plt.tight_layout()

out = os.path.join(BASE_DIR, 'total_dataset_analysis_figures', '10_pie_originality.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')