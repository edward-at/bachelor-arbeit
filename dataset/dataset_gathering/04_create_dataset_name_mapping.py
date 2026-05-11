import pandas as pd
from pathlib import Path

FINAL_DIR = Path("dataset_gathering_final")

input_file = FINAL_DIR / "duplicate_dataset_candidates_reviewed.csv"
output_file = FINAL_DIR / "dataset_name_mapping.csv"

df = pd.read_csv(input_file, dtype=str)

# Support both possible column names
same_col = "Same_Dataset?"
if same_col not in df.columns:
    same_col = "Same_Dataset"

# Keep only confirmed duplicates
yes_rows = df[df[same_col].fillna("").str.lower().str.strip() == "yes"].copy()

mapping_rows = []

for _, row in yes_rows.iterrows():
    canonical_id = row["Canonical_Dataset_ID"]
    canonical_name = row["Canonical_Dataset"]

    if pd.isna(canonical_id) or pd.isna(canonical_name):
        print("WARNING: yes row missing canonical info:")
        print(row)
        continue

    # Map dataset 1 to canonical
    mapping_rows.append({
        "Raw_Dataset_ID": row["Dataset_ID_1"],
        "Raw_Dataset": row["Dataset_1"],
        "Canonical_Dataset_ID": canonical_id,
        "Canonical_Dataset": canonical_name,
        "Notes": row.get("Notes", "")
    })

    # Map dataset 2 to canonical
    mapping_rows.append({
        "Raw_Dataset_ID": row["Dataset_ID_2"],
        "Raw_Dataset": row["Dataset_2"],
        "Canonical_Dataset_ID": canonical_id,
        "Canonical_Dataset": canonical_name,
        "Notes": row.get("Notes", "")
    })

mapping = pd.DataFrame(mapping_rows)

# Remove duplicate mapping rows
mapping = mapping.drop_duplicates(subset=["Raw_Dataset_ID", "Canonical_Dataset_ID"])

mapping.to_csv(output_file, index=False)

print(f"Saved: {output_file}")
print(f"Mappings created: {len(mapping)}")