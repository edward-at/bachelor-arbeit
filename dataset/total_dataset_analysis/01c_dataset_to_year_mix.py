import pandas as pd
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'dataset_cleaned.csv'))

mixed = df[df['Real / Synthetic / Mixed'] == 2]
counts = mixed.groupby('Year').size().reindex(range(2006, 2027), fill_value=0)

plt.figure(figsize=(10, 6))
plt.bar(counts.index, counts.values, edgecolor='black', color='C2')
plt.title('Number of Mixed Datasets Published per Year')
plt.xlabel('Year')
plt.ylabel('Number of Datasets')
plt.xticks(counts.index, rotation=45)
plt.tight_layout()
plt.grid(True)
plt.gca().set_axisbelow(True)

out = os.path.join(BASE_DIR, 'total_dataset_analysis_figures', '01c_dataset_to_year_mix.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')