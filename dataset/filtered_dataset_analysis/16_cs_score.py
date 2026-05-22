import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tum_colors import TUM_SECONDARY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'filtered_dataset_cleaned.csv'))

CURRENT_YEAR = 2026

# ── Ct: raw citation count ────────────────────────────────────────────────────
# NaN citation (NVIDIA) → 0
df['Ct'] = df['Citation'].fillna(0)

# ── Ca: age-adjusted citation count ──────────────────────────────────────────
# If year == 2026: Ca = Ct (no age adjustment, just published)
# If year != 2026: Ca = Ct / (2026 - year)
def age_adjusted(row):
    if row['year'] == CURRENT_YEAR:
        return row['Ct']
    else:
        return row['Ct'] / (CURRENT_YEAR - row['year'])

df['Ca'] = df.apply(age_adjusted, axis=1)

# ── Log normalization ─────────────────────────────────────────────────────────
# Ct_norm = (log(Ct) - min(log(Ct))) / (max(log(Ct)) - min(log(Ct)))
# Ca_norm = (log(Ca) - min(log(Ca))) / (max(log(Ca)) - min(log(Ca)))
# Add 1 to avoid log(0)
def log_normalize(series):
    log_vals = np.log(series + 1)
    min_val = log_vals.min()
    max_val = log_vals.max()
    if max_val == min_val:
        return pd.Series(0, index=series.index)
    return (log_vals - min_val) / (max_val - min_val)

df['Ct_norm'] = log_normalize(df['Ct'])
df['Ca_norm'] = log_normalize(df['Ca'])

# ── Cs = 0.5*Ct_norm + 0.5*Ca_norm ──────────────────────────────────────────
df['Cs'] = 0.5 * df['Ct_norm'] + 0.5 * df['Ca_norm']

# Sort by Cs descending
df_sorted = df[['dataset', 'Cs']].sort_values('Cs')

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.figure(figsize=(12, 10))
plt.barh(df_sorted['dataset'], df_sorted['Cs'],
         edgecolor='black', color=TUM_SECONDARY['blue2'])
plt.xlim(0, 1)
plt.title('Citation Score (Cs) per Dataset')
plt.xlabel('Cs Score (0–1)')
plt.grid(True)
plt.gca().set_axisbelow(True)
plt.tight_layout()

out = os.path.join(BASE_DIR, 'filtered_dataset_analysis_figures', '16_cs_score.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')
print()
print(df[['dataset', 'year', 'Ct', 'Ca', 'Ct_norm', 'Ca_norm', 'Cs']].sort_values('Cs', ascending=False).to_string())