import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'filtered_dataset_cleaned.csv'))

# Count how many datasets include each weather type
# Unknown entries are excluded
weather_counts = Counter()
for val in df['weather'].dropna():
    if val == 'Unknown':
        continue
    for w in val.split(';'):
        weather_counts[w.strip()] += 1

# Sort ascending for horizontal bar
sorted_counts = dict(sorted(weather_counts.items(), key=lambda x: x[1]))

plt.figure(figsize=(10, 5))
plt.barh(list(sorted_counts.keys()), list(sorted_counts.values()), edgecolor='black')
plt.title('Number of Datasets per Weather Condition')
plt.xlabel('Number of Datasets')
plt.grid(True)
plt.gca().set_axisbelow(True)
plt.tight_layout()

out = os.path.join(BASE_DIR, 'filtered_dataset_analysis_figures', '06_weather.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')