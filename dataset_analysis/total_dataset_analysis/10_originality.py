import pandas as pd
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tum_colors import TUM_SECONDARY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'dataset_cleaned.csv'))

counts = df['Original / Derived / Extension / Aggregated'].value_counts().sort_index()
labels = {0: 'Original', 1: 'Derived', 2: 'Extension', 3: 'Aggregated'}
sizes = [counts.get(k, 0) for k in [0, 1, 2, 3]]
names = [labels[k] for k in [0, 1, 2, 3]]

DARK_COLORS = {TUM_SECONDARY['blue1'], TUM_SECONDARY['blue2']}
pie_colors = [TUM_SECONDARY['blue2'], TUM_SECONDARY['gray'], TUM_SECONDARY['light_gray'], TUM_SECONDARY['blue1']]  # adjust per chart

plt.figure()
wedges, texts, autotexts = plt.pie(sizes, labels=names, autopct='%1.1f%%',
                                    colors=pie_colors,
                                    wedgeprops={'edgecolor': 'black'})
for i, autotext in enumerate(autotexts):
    if pie_colors[i] in DARK_COLORS:
        autotext.set_color('white')
    else:
        autotext.set_color('black')
plt.title('Original / Derived / Extension / Aggregated Datasets')
plt.tight_layout()

out = os.path.join(BASE_DIR, 'total_dataset_analysis_figures', '10_pie_originality.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')