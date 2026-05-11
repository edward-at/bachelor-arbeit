import pandas as pd
from pathlib import Path

FINAL_DIR = Path("dataset_gathering_final")

conflict_file = FINAL_DIR / "conflict_report.csv"
output_file = FINAL_DIR / "conflict_resolutions.csv"

df = pd.read_csv(conflict_file, dtype=str)


def suggest_resolution(row):
    attr = row["Attribute"]
    values = str(row["Values_Found"])

    if attr in ["Annotation", "Tasks"]:
        return "combine_or_choose_more_detailed"

    if attr in ["Paper_Link", "Related_Paper"]:
        return "choose_cleaner_link"

    if attr in ["Year", "Real_or_Synthetic", "LiDAR"]:
        return "manual_check_required"

    return "manual_check_required"


def suggest_final_value(row):
    attr = row["Attribute"]
    values = str(row["Values_Found"])

    parts = [v.strip() for v in values.split("|")]

    if attr in ["Annotation", "Tasks"]:
        # Put all values together as a first draft.
        # You can edit this manually in Excel.
        return "; ".join(parts)

    if attr in ["Paper_Link", "Related_Paper"]:
        # Choose first link as draft.
        # You can replace it manually if needed.
        return parts[0]

    # For risky attributes, leave blank.
    return ""


df["Suggested_Resolution"] = df.apply(suggest_resolution, axis=1)
df["Suggested_Final_Value"] = df.apply(suggest_final_value, axis=1)

# You manually edit this column
df["Final_Value"] = ""

# Notes column for your reasoning
df["Manual_Notes"] = ""

# Reorder useful columns
preferred_cols = [
    "Dataset_ID",
    "Dataset",
    "Attribute",
    "Values_Found",
    "Value_Sources",
    "Suggested_Resolution",
    "Suggested_Final_Value",
    "Final_Value",
    "Manual_Notes",
    "Raw_Name_Variants",
    "Source_Tables",
]

existing_cols = [c for c in preferred_cols if c in df.columns]
other_cols = [c for c in df.columns if c not in existing_cols]

df = df[existing_cols + other_cols]

df.to_csv(output_file, index=False)

print(f"Saved: {output_file}")
print(f"Rows: {len(df)}")