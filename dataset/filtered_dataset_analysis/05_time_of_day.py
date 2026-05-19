import pandas as pd
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tum_colors import TUM_SECONDARY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'filtered_dataset_cleaned.csv'))

counts = df['time_of_day'].value_counts()
label_map = {'Day': 'Day', 'Day; Night': 'Day & Night'}
labels = [label_map.get(l, l) for l in counts.index]
pie_colors = [TUM_SECONDARY['blue2'], TUM_SECONDARY['light_gray']]
DARK_COLORS = {TUM_SECONDARY['blue2']}

plt.figure()
wedges, texts, autotexts = plt.pie(counts.values, labels=labels, autopct='%1.1f%%',
                                    colors=pie_colors,
                                    wedgeprops={'edgecolor': 'black'})
for i, autotext in enumerate(autotexts):
    autotext.set_color('white' if pie_colors[i] in DARK_COLORS else 'black')
plt.title('Time of Day Coverage')
plt.tight_layout()

out = os.path.join(BASE_DIR, 'filtered_dataset_analysis_figures', '05_time_of_day.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')