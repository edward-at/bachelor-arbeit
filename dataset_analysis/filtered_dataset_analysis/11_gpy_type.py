import pandas as pd
import matplotlib.pyplot as plt
import os, sys
from collections import defaultdict
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tum_colors import TUM_SECONDARY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'filtered_dataset_cleaned.csv'))

# Merge RTK and RTK/PPP → RTK, exclude Unknown
gps_counts = defaultdict(float)
for val in df['gps_type'].dropna():
    if val == 'Unknown':
        continue
    parts = [p.strip() for p in val.split(';')]
    share = 1.0 / len(parts)
    for p in parts:
        # Merge RTK/PPP into RTK
        label = 'RTK/PPP' if 'RTK' in p else p
        gps_counts[label] += share

pie_colors = [TUM_SECONDARY['blue2'], TUM_SECONDARY['light_gray']]
DARK_COLORS = {TUM_SECONDARY['blue2']}

labels = list(gps_counts.keys())
values = list(gps_counts.values())

plt.figure()
wedges, texts, autotexts = plt.pie(values, labels=labels, autopct='%1.1f%%',
                                    colors=pie_colors[:len(labels)],
                                    wedgeprops={'edgecolor': 'black'})
for i, autotext in enumerate(autotexts):
    autotext.set_color('white' if pie_colors[i] in DARK_COLORS else 'black')
plt.title('GPS / GNSS Type (RTK/PPP vs Standard)')
plt.tight_layout()

out = os.path.join(BASE_DIR, 'filtered_dataset_analysis_figures', '11_gps_type.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')