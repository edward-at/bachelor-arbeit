import pandas as pd
from pathlib import Path

FINAL_DIR = Path("dataset_gathering_final")

master_file = FINAL_DIR / "dataset_master_draft.csv"
resolution_file = FINAL_DIR / "conflict_resolutions_reviewed.csv"

final_master_file = FINAL_DIR / "dataset_master_final.csv"
unresolved_file = FINAL_DIR / "unresolved_conflicts.csv"
change_log_file = FINAL_DIR / "applied_conflict_resolutions_log.csv"


def is_empty(x):
    if pd.isna(x):
        return True

    x = str(x).strip()

    return x == "" or x.lower() in ["nan", "none", "n/a", "na", "-", "null"]


# -----------------------------
# Load files
# -----------------------------

master = pd.read_csv(master_file, dtype=str)
resolutions = pd.read_csv(resolution_file, dtype=str)

print(f"Loaded master draft: {master_file}")
print(f"Master rows: {len(master)}")
print(f"Master columns: {len(master.columns)}")

print(f"\nLoaded reviewed resolutions: {resolution_file}")
print(f"Resolution rows: {len(resolutions)}")


# -----------------------------
# Check required columns
# -----------------------------

required_resolution_cols = ["Dataset_ID", "Attribute", "Final_Value"]

missing_cols = [col for col in required_resolution_cols if col not in resolutions.columns]

if missing_cols:
    raise ValueError(f"Missing columns in conflict_resolutions_reviewed.csv: {missing_cols}")

if "Dataset_ID" not in master.columns:
    raise ValueError("dataset_master_draft.csv must contain Dataset_ID column.")


# -----------------------------
# Apply resolutions
# -----------------------------

change_log = []
unresolved_rows = []

for _, row in resolutions.iterrows():
    dataset_id = row["Dataset_ID"]
    attribute = row["Attribute"]
    final_value = row["Final_Value"]

    if is_empty(final_value):
        unresolved_rows.append(row.to_dict())
        continue

    if attribute not in master.columns:
        print(f"⚠️ Attribute not found in master table: {attribute}")
        unresolved_rows.append(row.to_dict())
        continue

    mask = master["Dataset_ID"] == dataset_id

    if not mask.any():
        print(f"⚠️ Dataset_ID not found in master table: {dataset_id}")
        unresolved_rows.append(row.to_dict())
        continue

    old_values = master.loc[mask, attribute].fillna("").unique().tolist()

    master.loc[mask, attribute] = final_value

    change_log.append({
        "Dataset_ID": dataset_id,
        "Dataset": row.get("Dataset", ""),
        "Attribute": attribute,
        "Old_Master_Value": " | ".join(old_values),
        "Applied_Final_Value": final_value,
        "Values_Found": row.get("Values_Found", ""),
        "Manual_Notes": row.get("Manual_Notes", "")
    })


# -----------------------------
# Save outputs
# -----------------------------

master.to_csv(final_master_file, index=False)

pd.DataFrame(change_log).to_csv(change_log_file, index=False)

if unresolved_rows:
    pd.DataFrame(unresolved_rows).to_csv(unresolved_file, index=False)
else:
    pd.DataFrame(columns=resolutions.columns).to_csv(unresolved_file, index=False)

print("\n✅ Saved final master table:")
print(final_master_file)

print("\n✅ Saved applied resolution log:")
print(change_log_file)
print(f"Applied resolutions: {len(change_log)}")

print("\n✅ Saved unresolved conflicts:")
print(unresolved_file)
print(f"Unresolved conflicts: {len(unresolved_rows)}")