import pandas as pd
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tum_colors import TUM_SECONDARY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'filtered_dataset_cleaned.csv'))

ROAD_MAP = {
    'urban': 'Urban', 'city': 'Urban', 'downtown': 'Urban',
    'residential': 'Urban', 'suburban': 'Urban', 'suburb': 'Urban', 'suburbs': 'Urban',
    'highway': 'Highway', 'freeway': 'Highway', 'skyway': 'Highway', 'regional': 'Highway',
    'rural': 'Rural', 'country': 'Rural', 'countryside': 'Rural',
    'farm': 'Rural', 'forest': 'Rural', 'nature': 'Rural', 'hills': 'Rural',
    'campus': 'Controlled', 'airport': 'Controlled', 'parking': 'Controlled',
    'bridge': 'Infrastructure', 'tunnel': 'Infrastructure',
    'construction': 'Infrastructure', 'industrial': 'Infrastructure', 'structures': 'Infrastructure',
}
VALID_WEATHER = {'Sun', 'Rain', 'Snow', 'Cloud', 'Fog'}

def road_score(val):
    if pd.isna(val) or str(val).strip() in ('nan', ''):
        return 0.2   # minimum baseline — road type exists but undocumented
    tokens = [t.strip().lower() for t in str(val).split(';')]
    groups = {ROAD_MAP[t] for t in tokens if t in ROAD_MAP}
    return len(groups) / 5

def weather_score(val):
    if pd.isna(val) or str(val).strip() in ('nan', 'Unknown'):
        return 0.4   # minimum baseline — Sun and Cloud assumed as default conditions
    types = {w.strip() for w in str(val).split(';') if w.strip() in VALID_WEATHER}
    return len(types) / 5

df['road_score']    = df['road_type'].apply(road_score)
df['weather_score'] = df['weather'].apply(weather_score)
df['Ds'] = 0.5 * df['road_score'] + 0.5 * df['weather_score']

df_sorted = df[['dataset', 'Ds']].sort_values('Ds', ascending=True)

plt.figure(figsize=(12, 10))
plt.barh(df_sorted['dataset'], df_sorted['Ds'],
         edgecolor='black', color=TUM_SECONDARY['blue2'])
plt.xlim(0, 1)
plt.title('Dynamic Coverage Score (Ds) per Dataset')
plt.xlabel('Ds Score (0–1)')
plt.grid(True)
plt.gca().set_axisbelow(True)
plt.tight_layout()

out = os.path.join(BASE_DIR, 'filtered_dataset_analysis_figures', '15_ds_score.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')