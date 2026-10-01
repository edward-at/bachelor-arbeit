import pandas as pd
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tum_colors import TUM_SECONDARY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'dataset_cleaned.csv'))

counts = df['Real / Synthetic / Mixed'].value_counts().sort_index()
labels = {0: 'Real', 1: 'Synthetic', 2: 'Mixed'}
sizes = [counts.get(k, 0) for k in [0, 1, 2]]
names = [labels[k] for k in [0, 1, 2]]

DARK_COLORS = {TUM_SECONDARY['blue1'], TUM_SECONDARY['blue2']}
pie_colors = [TUM_SECONDARY['blue2'], TUM_SECONDARY['gray'], TUM_SECONDARY['light_gray'], TUM_SECONDARY['blue1']]

plt.figure()
wedges, texts, autotexts = plt.pie(sizes, labels=names, autopct='%1.1f%%',
                                    colors=pie_colors,
                                    wedgeprops={'edgecolor': 'black'})
for i, autotext in enumerate(autotexts):
    if pie_colors[i] in DARK_COLORS:
        autotext.set_color('white')
    else:
        autotext.set_color('black')
plt.title('Real / Synthetic / Mixed Datasets')
plt.tight_layout()

out = os.path.join(BASE_DIR, 'total_dataset_analysis_figures', '06_pie_real_synth_mix.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')

