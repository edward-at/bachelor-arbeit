import pandas as pd
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tum_colors import TUM_SECONDARY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'filtered_dataset_cleaned.csv'))
df_sorted = df[['dataset', 'Citation']].dropna().sort_values('Citation')

plt.figure(figsize=(15, 10))
plt.barh(df_sorted['dataset'], df_sorted['Citation'],
         edgecolor='black', color=TUM_SECONDARY['blue2'])
plt.xlim(0, 14000)
plt.title('Citation Count per Dataset')
plt.xlabel('Number of Citations')
plt.grid(True)
plt.gca().set_axisbelow(True)
plt.tight_layout()

out = os.path.join(BASE_DIR, 'filtered_dataset_analysis_figures', '08a_citation_full.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')