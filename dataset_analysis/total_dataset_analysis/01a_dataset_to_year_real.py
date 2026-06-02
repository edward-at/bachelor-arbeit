import pandas as pd
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tum_colors import TUM_SECONDARY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'dataset_cleaned.csv'))

real = df[df['Real / Synthetic / Mixed'] == 0]
counts = real.groupby('Year').size().reindex(range(2006, 2027), fill_value=0)

plt.figure(figsize=(10, 6))
plt.bar(counts.index, counts.values, edgecolor='black', color=TUM_SECONDARY['blue2'])
plt.title('Number of Real Datasets Published per Year')
plt.xlabel('Year')
plt.ylabel('Number of Datasets')
plt.xticks(counts.index, rotation=45)
plt.grid(True)
plt.gca().set_axisbelow(True)
plt.tight_layout()

out = os.path.join(BASE_DIR, 'total_dataset_analysis_figures', '01a_dataset_to_year_real.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')