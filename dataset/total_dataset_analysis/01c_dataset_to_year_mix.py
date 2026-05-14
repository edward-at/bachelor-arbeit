import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'dataset_cleaned.csv'))

# Only Mixed datasets (2 = Mixed)
mixed = df[df['Real / Synthetic / Mixed'] == 2]
counts = mixed.groupby('Year').size().reindex(range(df['Year'].min(), df['Year'].max()+1), fill_value=0)

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(counts.index, counts.values, color='#2ca02c', zorder=3)

for x, y in zip(counts.index, counts.values):
    if y > 0:
        ax.text(x, y + 0.1, str(y), ha='center', va='bottom', fontsize=8)

ax.set_title('Number of Mixed Datasets Published per Year', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Number of Datasets', fontsize=11)
ax.yaxis.grid(True, linestyle='--', linewidth=0.7, alpha=0.7, zorder=0)
ax.set_axisbelow(True)
ax.spines[['top', 'right']].set_visible(False)
ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.xticks(rotation=45, ha='right', fontsize=9)

plt.tight_layout()
out = os.path.join(BASE_DIR, 'figures', '01c_data_to_year_mix.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')