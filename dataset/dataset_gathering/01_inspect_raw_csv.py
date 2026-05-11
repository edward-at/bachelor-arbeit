import pandas as pd
from pathlib import Path

RAW_DIR = Path("dataset_gathering_raw")

csv_files = list(RAW_DIR.glob("*.csv"))

print(f"Found {len(csv_files)} CSV files.\n")

for file in csv_files:
    print("=" * 80)
    print(f"File: {file.name}")

    try:
        df = pd.read_csv(file)
        print(f"Rows: {len(df)}")
        print(f"Columns ({len(df.columns)}):")

        for col in df.columns:
            print(f"  - {col}")

        print("\nFirst 3 rows:")
        print(df.head(3).to_string(index=False))

    except Exception as e:
        print(f"ERROR reading file: {e}")

print("=" * 80)
print("Inspection complete.")