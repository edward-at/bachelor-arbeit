import pandas as pd
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'filtered_dataset_cleaned.csv'))

counts = df['time_of_day'].value_counts()

plt.figure()
label_map = {'Day': 'Day', 'Day; Night': 'Day & Night'}
labels = [label_map.get(l, l) for l in counts.index]

plt.pie(counts.values, labels=labels, autopct='%1.1f%%', wedgeprops={'edgecolor': 'black'})
plt.title('Time of Day Coverage')
plt.tight_layout()

out = os.path.join(BASE_DIR, 'filtered_dataset_analysis_figures', '05_time_of_day.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')