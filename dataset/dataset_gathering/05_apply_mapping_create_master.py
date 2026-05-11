import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("dataset_gathering_processed")
FINAL_DIR = Path("dataset_gathering_final")

combined_file = PROCESSED_DIR / "combined_raw_all_tables.csv"
mapping_file = FINAL_DIR / "dataset_name_mapping.csv"

mapped_output = PROCESSED_DIR / "combined_raw_mapped.csv"
master_output = FINAL_DIR / "dataset_master_draft.csv"
conflict_output = FINAL_DIR / "conflict_report.csv"


# -----------------------------
# Helper functions
# -----------------------------

def is_empty_value(x):
    if pd.isna(x):
        return True

    x = str(x).strip()

    if x == "":
        return True

    if x.lower() in ["nan", "none", "n/a", "na", "-", "null"]:
        return True

    return False


def unique_non_empty_values(series):
    values = []

    for x in series:
        if is_empty_value(x):
            continue

        x = str(x).strip()

        if x not in values:
            values.append(x)

    return values


def aggregate_sources(group):
    if "Source_Table" not in group.columns:
        return ""

    sources = unique_non_empty_values(group["Source_Table"])
    return "; ".join(sources)


def source_map_for_attribute(group, attribute):
    """
    Returns something like:
    2012 -> Sheet6.csv; 2013 -> Sheet2.csv
    """
    parts = []

    for value in unique_non_empty_values(group[attribute]):
        source_rows = group[group[attribute].astype(str).str.strip() == str(value).strip()]

        if "Source_Table" in source_rows.columns:
            sources = unique_non_empty_values(source_rows["Source_Table"])
            source_text = "; ".join(sources)
        else:
            source_text = ""

        parts.append(f"{value} -> {source_text}")

    return " | ".join(parts)


# -----------------------------
# 1. Load combined raw table
# -----------------------------

combined = pd.read_csv(combined_file, dtype=str)

print(f"Loaded combined raw table: {combined_file}")
print(f"Rows: {len(combined)}")
print(f"Columns: {len(combined.columns)}")


# -----------------------------
# 2. Apply dataset name mapping
# -----------------------------

if mapping_file.exists():
    mapping = pd.read_csv(mapping_file, dtype=str)

    required_cols = [
        "Raw_Dataset_ID",
        "Canonical_Dataset_ID",
        "Canonical_Dataset"
    ]

    missing_cols = [col for col in required_cols if col not in mapping.columns]

    if missing_cols:
        raise ValueError(f"Mapping file missing columns: {missing_cols}")

    combined = combined.merge(
        mapping[required_cols].drop_duplicates(),
        left_on="Dataset_ID",
        right_on="Raw_Dataset_ID",
        how="left"
    )

    combined["Final_Dataset_ID"] = combined["Canonical_Dataset_ID"].fillna(combined["Dataset_ID"])
    combined["Final_Dataset"] = combined["Canonical_Dataset"].fillna(combined["Dataset"])

    combined = combined.drop(columns=[
        col for col in [
            "Raw_Dataset_ID",
            "Canonical_Dataset_ID",
            "Canonical_Dataset"
        ] if col in combined.columns
    ])

else:
    print("WARNING: dataset_name_mapping.csv not found. Using original Dataset_IDs.")
    combined["Final_Dataset_ID"] = combined["Dataset_ID"]
    combined["Final_Dataset"] = combined["Dataset"]


# Move final ID/name to front
front_cols = ["Final_Dataset_ID", "Final_Dataset", "Dataset_ID", "Dataset", "Raw_Dataset_Name", "Source_Table"]
front_cols = [col for col in front_cols if col in combined.columns]
other_cols = [col for col in combined.columns if col not in front_cols]
combined = combined[front_cols + other_cols]

combined.to_csv(mapped_output, index=False)

print(f"\nSaved mapped raw table: {mapped_output}")
print(f"Rows: {len(combined)}")
print(f"Unique datasets before mapping: {combined['Dataset_ID'].nunique()}")
print(f"Unique datasets after mapping: {combined['Final_Dataset_ID'].nunique()}")


# -----------------------------
# 3. Create master draft + conflict report
# -----------------------------

ignore_cols = {
    "Dataset_ID",
    "Dataset",
    "Raw_Dataset_Name",
    "Source_Table",
    "Final_Dataset_ID",
    "Final_Dataset"
}

attribute_cols = [col for col in combined.columns if col not in ignore_cols]

master_rows = []
conflict_rows = []

for final_id, group in combined.groupby("Final_Dataset_ID", dropna=False):
    final_name_values = unique_non_empty_values(group["Final_Dataset"])

    final_name = final_name_values[0] if final_name_values else final_id

    master_row = {
        "Dataset_ID": final_id,
        "Dataset": final_name,
        "Source_Tables": aggregate_sources(group),
        "Raw_Name_Variants": "; ".join(unique_non_empty_values(group["Dataset"])) if "Dataset" in group.columns else ""
    }

    for col in attribute_cols:
        values = unique_non_empty_values(group[col])

        if len(values) == 0:
            master_row[col] = ""

        elif len(values) == 1:
            master_row[col] = values[0]

        else:
            # Conflict: leave master value blank and report conflict separately
            master_row[col] = ""

            conflict_rows.append({
                "Dataset_ID": final_id,
                "Dataset": final_name,
                "Attribute": col,
                "Values_Found": " | ".join(values),
                "Value_Sources": source_map_for_attribute(group, col),
                "Raw_Name_Variants": "; ".join(unique_non_empty_values(group["Dataset"])) if "Dataset" in group.columns else "",
                "Source_Tables": aggregate_sources(group),
                "Resolution": "",
                "Final_Value": "",
                "Notes": ""
            })

    master_rows.append(master_row)


master = pd.DataFrame(master_rows)
conflicts = pd.DataFrame(conflict_rows)

master = master.sort_values("Dataset").reset_index(drop=True)

if len(conflicts) > 0:
    conflicts = conflicts.sort_values(["Dataset", "Attribute"]).reset_index(drop=True)

master.to_csv(master_output, index=False)
conflicts.to_csv(conflict_output, index=False)

print(f"\nSaved master draft: {master_output}")
print(f"Master rows: {len(master)}")
print(f"Master columns: {len(master.columns)}")

print(f"\nSaved conflict report: {conflict_output}")
print(f"Conflict rows: {len(conflicts)}")