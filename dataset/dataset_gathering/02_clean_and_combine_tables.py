import pandas as pd
import re
from pathlib import Path

RAW_DIR = Path("dataset_gathering_raw")
PROCESSED_DIR = Path("dataset_gathering_processed")
FINAL_DIR = Path("dataset_gathering_final")

PROCESSED_DIR.mkdir(exist_ok=True)
FINAL_DIR.mkdir(exist_ok=True)


# -----------------------------
# Cleaning functions
# -----------------------------

def clean_dataset_name(name):
    if pd.isna(name):
        return None

    name = str(name).strip()

    # Remove numeric references only: KITTI [12] -> KITTI
    # This will NOT remove [CARLA] because CARLA is not numeric.
    name = re.sub(r"\[\d+\]", "", name)

    # Normalize spaces
    name = re.sub(r"\s+", " ", name).strip()

    return name


def make_dataset_id(name):
    if pd.isna(name) or name is None:
        return None

    name = str(name).lower().strip()

    # Remove numeric reference markers
    name = re.sub(r"\[\d+\]", "", name)

    # Normalize symbols
    name = name.replace("&", "and")
    name = name.replace("+", "plus")

    # Create clean ID
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")

    return name


def clean_value(x):
    if pd.isna(x):
        return None

    x = str(x).strip()

    if x.lower() in ["nan", "none", ""]:
        return None

    return x


# -----------------------------
# Column standardization
# -----------------------------

COLUMN_RENAME = {
    "Link": "Dataset_Link",

    "Time": "Year",

    "Lidar": "LiDAR",
    "lidar": "LiDAR",

    "2D Lidar": "2D_LiDAR",
    "3D Lidar": "3D_LiDAR",

    "Spinning LiDAR": "Spinning_LiDAR",
    "Solid State LiDAR": "Solid_State_LiDAR",

    "Sensing domain": "Sensing_Domain",

    "Real/Synthetic": "Real_or_Synthetic",

    "Sensortypes": "Sensor_Types",
    "Provides raw data": "Provides_Raw_Data",
    "Related Paper": "Related_Paper",

    "Annotation": "Annotation",
    "Annotations": "Annotation",

    "Paper": "Paper_Link",

    "Task": "Tasks",
    "Tasks": "Tasks",

    "Monocular_Camera": "Mono_Camera",
    "Stereo_Camera": "Stereo_Camera",

    "2D_Det": "OD_2D",
    "3D_Det": "OD_3D",
    "Seg": "Segmentation",
    "Pred": "Prediction",
}


# -----------------------------
# Process all CSV files
# -----------------------------

all_tables = []

csv_files = list(RAW_DIR.glob("*.csv"))

print(f"Found {len(csv_files)} CSV files.\n")

for file in csv_files:
    print("=" * 80)
    print(f"Processing: {file.name}")

    df = pd.read_csv(file, dtype=str)

    # Clean column names
    df.columns = [str(col).strip() for col in df.columns]

    # Drop empty unnamed columns
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # Rename columns
    df = df.rename(columns=COLUMN_RENAME)

    # Clean all cell values lightly
    for col in df.columns:
        df[col] = df[col].apply(clean_value)

    # Clean Dataset column
    if "Dataset" not in df.columns:
        print("❌ Skipped: no Dataset column found.")
        continue

    df["Raw_Dataset_Name"] = df["Dataset"]
    df["Dataset"] = df["Dataset"].apply(clean_dataset_name)
    df["Dataset_ID"] = df["Dataset"].apply(make_dataset_id)

    # Add source table
    df["Source_Table"] = file.name

    # Move important columns to front
    front_cols = ["Dataset_ID", "Dataset", "Raw_Dataset_Name", "Source_Table"]
    other_cols = [col for col in df.columns if col not in front_cols]
    df = df[front_cols + other_cols]

    # Save individual cleaned file
    out_file = PROCESSED_DIR / file.name
    df.to_csv(out_file, index=False)

    all_tables.append(df)

    print(f"✅ Saved cleaned file: {out_file}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")


# -----------------------------
# Combine all cleaned tables
# -----------------------------

combined = pd.concat(all_tables, ignore_index=True, sort=False)

combined.to_csv(PROCESSED_DIR / "combined_raw_all_tables.csv", index=False)

print("=" * 80)
print("✅ Saved combined raw table:")
print(PROCESSED_DIR / "combined_raw_all_tables.csv")
print(f"Total rows: {len(combined)}")
print(f"Total columns: {len(combined.columns)}")


# -----------------------------
# Create dataset review list
# -----------------------------

dataset_review = (
    combined[["Dataset_ID", "Dataset"]]
    .drop_duplicates()
    .sort_values("Dataset_ID")
    .reset_index(drop=True)
)

dataset_review.to_csv(FINAL_DIR / "dataset_name_review.csv", index=False)

print("✅ Saved dataset name review file:")
print(FINAL_DIR / "dataset_name_review.csv")
print(f"Unique dataset IDs: {len(dataset_review)}")