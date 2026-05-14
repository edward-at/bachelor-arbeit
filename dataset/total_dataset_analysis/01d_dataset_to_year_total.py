import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'dataset_cleaned.csv'))

type_map = {0: 'Real', 1: 'Synthetic', 2: 'Mixed'}
df['Type'] = df['Real / Synthetic / Mixed'].map(type_map)

all_years = range(df['Year'].min(), df['Year'].max()+1)
counts = (df.groupby(['Year', 'Type'])
            .size()
            .unstack(fill_value=0)
            .reindex(all_years, fill_value=0)
            .reindex(columns=['Real', 'Synthetic', 'Mixed'], fill_value=0))

years = counts.index.tolist()

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(years, counts['Real'],      color='#1f77b4', label='Real',      zorder=3)
ax.bar(years, counts['Synthetic'], color='#ff7f0e', label='Synthetic',
       bottom=counts['Real'],      zorder=3)
ax.bar(years, counts['Mixed'],     color='#2ca02c', label='Mixed',
       bottom=counts['Real'] + counts['Synthetic'], zorder=3)

# Total label on top of each bar
for year in years:
    total = counts.loc[year, ['Real', 'Synthetic', 'Mixed']].sum()
    if total > 0:
        ax.text(year, total + 0.15, str(int(total)), ha='center', va='bottom', fontsize=8)

ax.set_title('Total Number of Datasets Published per Year (Stacked)', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Number of Datasets', fontsize=11)
ax.legend(fontsize=10, frameon=False)
ax.yaxis.grid(True, linestyle='--', linewidth=0.7, alpha=0.7, zorder=0)
ax.set_axisbelow(True)
ax.spines[['top', 'right']].set_visible(False)
ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.xticks(rotation=45, ha='right', fontsize=9)

plt.tight_layout()
out = os.path.join(BASE_DIR, 'total_dataset_analysis_figures', '01d_data_to_year_total.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')