import pandas as pd
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tum_colors import TUM_SECONDARY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'filtered_dataset_cleaned.csv'))

counts = df['temporal'].value_counts().sort_index()
labels = {0: 'Static', 1: 'Video'}
names = [labels[k] for k in counts.index]
pie_colors = [TUM_SECONDARY['light_gray'], TUM_SECONDARY['blue2']]
DARK_COLORS = {TUM_SECONDARY['blue2']}

plt.figure()
wedges, texts, autotexts = plt.pie(counts.values, labels=names, autopct='%1.1f%%',
                                    colors=pie_colors,
                                    wedgeprops={'edgecolor': 'black'})
for i, autotext in enumerate(autotexts):
    autotext.set_color('white' if pie_colors[i] in DARK_COLORS else 'black')
plt.title('Temporal Coverage (Video vs Static)')
plt.tight_layout()

out = os.path.join(BASE_DIR, 'filtered_dataset_analysis_figures', '08_temporal.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')