import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'dataset_cleaned.csv'))

# ── Prepare data ──────────────────────────────────────────────────────────────
type_map = {0: 'Real', 1: 'Synthetic', 2: 'Mixed'}
df['Type'] = df['Real / Synthetic / Mixed'].map(type_map)

years = range(df['Year'].min(), df['Year'].max()+1)

counts = (df.groupby(['Year', 'Type'])
            .size()
            .unstack(fill_value=0)
            .reindex(years, fill_value=0)
            .reindex(columns=['Real', 'Synthetic', 'Mixed'], fill_value=0))

# Cumulative sum per category
cumul = counts.cumsum()
cumul['Total'] = cumul['Real'] + cumul['Synthetic'] + cumul['Mixed']

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

# Stacked area: Real at bottom, Synthetic middle, Mixed on top
ax.stackplot(cumul.index,
             cumul['Real'],
             cumul['Synthetic'],
             cumul['Mixed'],
             labels=['Real', 'Synthetic', 'Mixed'],
             colors=['#1f77b4', '#ff7f0e', '#2ca02c'],
             alpha=0.7)

# Total cumulative line on top
ax.plot(cumul.index, cumul['Total'],
        color='black', linewidth=2, linestyle='-', label='Total', zorder=5)

# ── Styling ───────────────────────────────────────────────────────────────────
ax.set_title('Cumulative Number of Datasets Over Time', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Cumulative Number of Datasets', fontsize=11)
ax.legend(fontsize=10, frameon=False, loc='upper left')
ax.yaxis.grid(True, linestyle='--', linewidth=0.7, alpha=0.7, zorder=0)
ax.set_axisbelow(True)
ax.spines[['top', 'right']].set_visible(False)
ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
ax.set_xlim(2006, 2026)
plt.xticks(rotation=45, ha='right', fontsize=9)

plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
out = os.path.join(BASE_DIR, 'total_dataset_analysis_figures', '02_cumulative_datasets.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')