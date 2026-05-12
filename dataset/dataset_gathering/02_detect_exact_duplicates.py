from pathlib import Path
import pandas as pd
import re


# ============================================================
# PATH CONFIG
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = ROOT_DIR / "dataset_gathering_processed"

INPUT_FILE = (
    PROCESSED_DIR
    / "01_combined_raw"
    / "01_combined_raw_long_table.csv"
)

OUTPUT_DIR = PROCESSED_DIR / "02_duplicate_detection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# OUTPUT FILES
# ============================================================

DUPLICATE_SUMMARY_OUTPUT = OUTPUT_DIR / "02_exact_duplicate_dataset_summary.csv"
DUPLICATE_ROWS_OUTPUT = OUTPUT_DIR / "02_exact_duplicate_dataset_rows.csv"
UNIQUE_NAMES_OUTPUT = OUTPUT_DIR / "02_unique_dataset_names.csv"
BASIC_STATS_OUTPUT = OUTPUT_DIR / "02_duplicate_detection_stats.csv"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def basic_dataset_key(value):
    """
    Basic duplicate key.

    This does NOT do aggressive merging.
    It only:
    - strips spaces
    - lowercases
    - collapses multiple spaces

    Example:
    'KITTI' and ' kitti ' become the same key.
    """

    value = str(value).strip()
    value = value.lower()
    value = re.sub(r"\s+", " ", value)

    return value


def is_empty(value):
    text = str(value).strip()

    return text == "" or text.lower() in [
        "nan",
        "none",
        "null",
        "n/a",
        "na",
        "-"
    ]


# ============================================================
# MAIN
# ============================================================

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}\n\n"
            "Run 01_combine_raw_csvs.py first."
        )

    print("=" * 80)
    print("STEP 02: DETECT EXACT DUPLICATE DATASET NAMES")
    print("=" * 80)

    df = pd.read_csv(
        INPUT_FILE,
        dtype=str,
        keep_default_na=False
    )

    if "dataset" not in df.columns:
        raise ValueError(
            "Column 'dataset' not found in combined table.\n"
            f"Available columns: {list(df.columns)}"
        )

    # Keep only rows with dataset names
    df["dataset_raw"] = df["dataset"].astype(str).str.strip()
    df = df[~df["dataset_raw"].apply(is_empty)].copy()

    # Create basic duplicate key
    df["dataset_key_basic"] = df["dataset_raw"].apply(basic_dataset_key)

    # Count duplicates
    counts = df["dataset_key_basic"].value_counts()

    duplicate_keys = counts[counts > 1].index.tolist()

    duplicate_rows = df[df["dataset_key_basic"].isin(duplicate_keys)].copy()

    # Build duplicate summary
    summary_rows = []

    for dataset_key in duplicate_keys:
        group = df[df["dataset_key_basic"] == dataset_key].copy()

        raw_names = sorted(
            group["dataset_raw"].dropna().astype(str).str.strip().unique(),
            key=lambda x: x.lower()
        )

        source_files = sorted(
            group["source_file"].dropna().astype(str).str.strip().unique(),
            key=lambda x: x.lower()
        )

        source_locations = []

        for _, row in group.iterrows():
            source_locations.append(
                f"{row.get('source_file', '')}: row {row.get('source_row_number', '')}"
            )

        summary_rows.append({
            "dataset_key_basic": dataset_key,
            "recommended_display_name": raw_names[0] if raw_names else "",
            "row_count": len(group),
            "raw_name_variants": " | ".join(raw_names),
            "source_files": " | ".join(source_files),
            "source_locations": " | ".join(source_locations),
        })

    duplicate_summary = pd.DataFrame(summary_rows)

    if not duplicate_summary.empty:
        duplicate_summary = duplicate_summary.sort_values(
            by=["row_count", "recommended_display_name"],
            ascending=[False, True]
        )

    # Unique names table
    unique_rows = []

    for dataset_key, group in df.groupby("dataset_key_basic"):
        raw_names = sorted(
            group["dataset_raw"].dropna().astype(str).str.strip().unique(),
            key=lambda x: x.lower()
        )

        unique_rows.append({
            "dataset_key_basic": dataset_key,
            "recommended_display_name": raw_names[0] if raw_names else "",
            "raw_name_variants": " | ".join(raw_names),
            "row_count": len(group),
        })

    unique_names = pd.DataFrame(unique_rows).sort_values(
        by="recommended_display_name"
    )

    # Stats
    stats = pd.DataFrame([
        {
            "metric": "total_rows_with_dataset_name",
            "value": len(df),
        },
        {
            "metric": "unique_dataset_names_basic",
            "value": df["dataset_key_basic"].nunique(),
        },
        {
            "metric": "duplicate_dataset_keys_basic",
            "value": len(duplicate_keys),
        },
        {
            "metric": "rows_that_are_part_of_duplicates",
            "value": len(duplicate_rows),
        },
    ])

    # Save outputs
    duplicate_summary.to_csv(
        DUPLICATE_SUMMARY_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    duplicate_rows.to_csv(
        DUPLICATE_ROWS_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    unique_names.to_csv(
        UNIQUE_NAMES_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    stats.to_csv(
        BASIC_STATS_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Input rows: {len(df)}")
    print(f"Unique dataset names: {df['dataset_key_basic'].nunique()}")
    print(f"Duplicate dataset keys found: {len(duplicate_keys)}")
    print(f"Rows involved in duplicates: {len(duplicate_rows)}")

    print("\nSaved:")
    print(DUPLICATE_SUMMARY_OUTPUT)
    print(DUPLICATE_ROWS_OUTPUT)
    print(UNIQUE_NAMES_OUTPUT)
    print(BASIC_STATS_OUTPUT)

    print("\nDONE")


if __name__ == "__main__":
    main()