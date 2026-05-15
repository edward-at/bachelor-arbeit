import pandas as pd
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'dataset_cleaned.csv'))

type_map = {0: 'Real', 1: 'Synthetic', 2: 'Mixed'}
df['Type'] = df['Real / Synthetic / Mixed'].map(type_map)

all_years = range(2006, 2027)
counts = (df.groupby(['Year', 'Type'])
            .size()
            .unstack(fill_value=0)
            .reindex(all_years, fill_value=0)
            .reindex(columns=['Real', 'Synthetic', 'Mixed'], fill_value=0))

years = counts.index.tolist()

plt.figure(figsize=(10, 6))
plt.bar(years, counts['Real'], label='Real', edgecolor='black', color='C0')
plt.bar(years, counts['Synthetic'], label='Synthetic', bottom=counts['Real'], edgecolor='black', color='C1')
plt.bar(years, counts['Mixed'], label='Mixed', bottom=counts['Real'] + counts['Synthetic'], edgecolor='black', color='C2')
plt.title('Total Number of Datasets Published per Year')
plt.xlabel('Year')
plt.ylabel('Number of Datasets')
plt.xticks(years, rotation=45)
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.gca().set_axisbelow(True)

out = os.path.join(BASE_DIR, 'total_dataset_analysis_figures', '01d_dataset_to_year_total.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')