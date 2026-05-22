import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tum_colors import TUM_SECONDARY, TUM_PRIMARY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'filtered_dataset_cleaned.csv'))

CURRENT_YEAR = 2026

# ── Qs ────────────────────────────────────────────────────────────────────────
GRADE_SCORE = {'Navigation': 1.0, 'Tactical': 0.75, 'Industrial': 0.5, 'Consumer': 0.25}
df['imu_grade_score'] = df['imu_grade_top'].map(GRADE_SCORE).fillna(0)
df['gps_score'] = df['gps_type'].map({'RTK': 1.0, 'Standard': 0.0, 'Unknown': 0.0}).fillna(0)

def imu_hz_score(val):
    if pd.isna(val) or val == 0:
        return 0
    if val > 200:   return 1.0
    elif val >= 100: return 0.67
    elif val >= 50:  return 0.33
    else:            return 0.17

df['imu_hz_score'] = df['imu_freq_top'].apply(imu_hz_score)
df['Qs'] = 0.4 * df['imu_grade_score'] + 0.4 * df['gps_score'] + 0.2 * df['imu_hz_score']

# ── Ds ────────────────────────────────────────────────────────────────────────
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
        return 0
    tokens = [t.strip().lower() for t in str(val).split(';')]
    groups = {ROAD_MAP[t] for t in tokens if t in ROAD_MAP}
    return len(groups) / 5

def weather_score(val):
    if pd.isna(val) or str(val).strip() in ('nan', 'Unknown'):
        return 0
    types = {w.strip() for w in str(val).split(';') if w.strip() in VALID_WEATHER}
    return len(types) / 5

df['road_score'] = df['road_type'].apply(road_score)
df['weather_score'] = df['weather'].apply(weather_score)
df['Ds'] = 0.5 * df['road_score'] + 0.5 * df['weather_score']

# ── Ps ────────────────────────────────────────────────────────────────────────
# can_bus already 1/0 from cleaning
df['Ps'] = df['can_bus'].fillna(0)

# ── Cs ────────────────────────────────────────────────────────────────────────
df['Ct'] = df['Citation'].fillna(0)

def age_adjusted(row):
    if row['year'] == CURRENT_YEAR:
        return row['Ct']
    return row['Ct'] / (CURRENT_YEAR - row['year'])

df['Ca'] = df.apply(age_adjusted, axis=1)

def log_normalize(series):
    log_vals = np.log(series + 1)
    min_val, max_val = log_vals.min(), log_vals.max()
    if max_val == min_val:
        return pd.Series(0, index=series.index)
    return (log_vals - min_val) / (max_val - min_val)

df['Ct_norm'] = log_normalize(df['Ct'])
df['Ca_norm'] = log_normalize(df['Ca'])
df['Cs'] = 0.5 * df['Ct_norm'] + 0.5 * df['Ca_norm']

# ── DSS = 0.50*Qs + 0.20*Ds + 0.20*Ps + 0.10*Cs ─────────────────────────────
df['DSS'] = 0.50 * df['Qs'] + 0.20 * df['Ds'] + 0.20 * df['Ps'] + 0.10 * df['Cs']

# ── Plot A — Horizontal bar (DSS sorted) ─────────────────────────────────────
df_sorted = df[['dataset', 'DSS']].sort_values('DSS')

plt.figure(figsize=(12, 10))
plt.barh(df_sorted['dataset'], df_sorted['DSS'],
         edgecolor='black', color=TUM_SECONDARY['blue2'])
plt.xlim(0, 1)
plt.title('Dynamic Scenario Suitability Score (DSS) per Dataset')
plt.xlabel('DSS Score (0–1)')
plt.grid(True)
plt.gca().set_axisbelow(True)
plt.tight_layout()

out_a = os.path.join(BASE_DIR, 'filtered_dataset_analysis_figures', '17a_dss_bar.png')
plt.savefig(out_a, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {out_a}')

# ── Plot B — Heatmap (datasets × score components) ───────────────────────────
heat_df = df[['dataset', 'Qs', 'Ds', 'Ps', 'Cs', 'DSS']].set_index('dataset')
heat_df = heat_df.sort_values('DSS', ascending=False)

fig, ax = plt.subplots(figsize=(10, 12))
im = ax.imshow(heat_df.values, aspect='auto', cmap='Blues', vmin=0, vmax=1)
ax.set_xticks(range(len(heat_df.columns)))
ax.set_xticklabels(heat_df.columns, fontsize=11)
ax.set_yticks(range(len(heat_df.index)))
ax.set_yticklabels(heat_df.index, fontsize=9)
plt.colorbar(im, ax=ax, label='Score (0–1)', shrink=0.5)

# Add value annotations
for i in range(len(heat_df.index)):
    for j in range(len(heat_df.columns)):
        val = heat_df.values[i, j]
        color = 'white' if val > 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7.5, color=color)

ax.set_title('Score Components Heatmap per Dataset\n(sorted by DSS)', fontsize=13, pad=12)
plt.tight_layout()

out_b = os.path.join(BASE_DIR, 'filtered_dataset_analysis_figures', '17b_dss_heatmap.png')
plt.savefig(out_b, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {out_b}')
print()
print(df[['dataset', 'Qs', 'Ds', 'Ps', 'Cs', 'DSS']].sort_values('DSS', ascending=False).to_string())