import pandas as pd
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tum_colors import TUM_SECONDARY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'dataset_cleaned.csv'))

type_map = {0: 'Real', 1: 'Synthetic', 2: 'Mixed'}
df['Type'] = df['Real / Synthetic / Mixed'].map(type_map)

years = range(2006, 2027)
counts = (df.groupby(['Year', 'Type'])
            .size()
            .unstack(fill_value=0)
            .reindex(years, fill_value=0)
            .reindex(columns=['Real', 'Synthetic', 'Mixed'], fill_value=0))

cumul = counts.cumsum()
cumul['Total'] = cumul['Real'] + cumul['Synthetic'] + cumul['Mixed']

plt.figure(figsize=(10, 6))
plt.stackplot(cumul.index,
              cumul['Real'], cumul['Synthetic'], cumul['Mixed'],
              labels=['Real', 'Synthetic', 'Mixed'],
              colors=[TUM_SECONDARY['blue2'], TUM_SECONDARY['gray'], TUM_SECONDARY['light_gray']], alpha=0.8)
plt.plot(cumul.index, cumul['Total'], color='black', linewidth=2, label='Total')
plt.title('Cumulative Number of Datasets Over Time')
plt.xlabel('Year')
plt.ylabel('Cumulative Number of Datasets')
plt.xlim(2006, 2026)
plt.xticks(list(years), rotation=45)
plt.legend()
plt.grid(True)
plt.gca().set_axisbelow(True)
plt.tight_layout()

out = os.path.join(BASE_DIR, 'total_dataset_analysis_figures', '02_cumulative_datasets.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')