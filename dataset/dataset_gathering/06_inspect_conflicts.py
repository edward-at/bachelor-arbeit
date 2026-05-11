import pandas as pd
from pathlib import Path

FINAL_DIR = Path("dataset_gathering_final")

conflict_file = FINAL_DIR / "conflict_report.csv"

df = pd.read_csv(conflict_file, dtype=str)

print("=" * 80)
print(f"Total conflict rows: {len(df)}")
print("=" * 80)

print("\nConflicts by attribute:")
print(df["Attribute"].value_counts().to_string())

print("\n" + "=" * 80)
print("First 30 conflict rows:")
print("=" * 80)

cols_to_show = [
    "Dataset",
    "Attribute",
    "Values_Found",
    "Value_Sources",
    "Raw_Name_Variants"
]

cols_to_show = [c for c in cols_to_show if c in df.columns]

print(df[cols_to_show].head(30).to_string(index=False))
