from pathlib import Path
import pandas as pd
import re


# ============================================================
# PATH CONFIG
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = ROOT_DIR / "dataset_gathering_processed"

COMBINED_INPUT = (
    PROCESSED_DIR
    / "01_combined_raw"
    / "01_combined_raw_long_table.csv"
)

MAPPING_INPUT = (
    PROCESSED_DIR
    / "03_manual_similarity_mapping"
    / "03_manual_similar_dataset_mapping.csv"
)

OUTPUT_DIR = PROCESSED_DIR / "04_mapped_dataset_names"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# OUTPUT FILES
# ============================================================

MAPPED_LONG_TABLE_OUTPUT = OUTPUT_DIR / "04_mapped_long_table.csv"
MAPPING_REPORT_OUTPUT = OUTPUT_DIR / "04_mapping_application_report.csv"
UNUSED_MAPPING_OUTPUT = OUTPUT_DIR / "04_unused_manual_mapping_rows.csv"
APPLIED_MAPPING_ROWS_OUTPUT = OUTPUT_DIR / "04_rows_changed_by_mapping.csv"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_for_matching(value):
    """
    Conservative normalization for matching manual raw names.

    This does:
    - strip spaces
    - lowercase
    - collapse repeated spaces

    It does NOT aggressively remove words.
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


def clean_mapping_decision(value):
    return str(value).strip().lower()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("STEP 04: APPLY MANUAL DATASET NAME MAPPING")
    print("=" * 80)

    if not COMBINED_INPUT.exists():
        raise FileNotFoundError(
            f"Combined input file not found:\n{COMBINED_INPUT}\n\n"
            "Run 01_combine_raw_csvs.py first."
        )

    if not MAPPING_INPUT.exists():
        raise FileNotFoundError(
            f"Manual mapping file not found:\n{MAPPING_INPUT}\n\n"
            "Run 03_create_manual_similarity_mapping_template.py first."
        )

    # Load combined long table
    df = pd.read_csv(
        COMBINED_INPUT,
        dtype=str,
        keep_default_na=False
    )

    if "dataset" not in df.columns:
        raise ValueError(
            "Column 'dataset' not found in combined table.\n"
            f"Available columns: {list(df.columns)}"
        )

    # Load manual mapping
    mapping_df = pd.read_csv(
        MAPPING_INPUT,
        dtype=str,
        keep_default_na=False
    )

    required_cols = ["raw_name", "canonical_name", "merge_decision"]

    for col in required_cols:
        if col not in mapping_df.columns:
            raise ValueError(
                f"Column '{col}' not found in mapping file.\n"
                f"Available columns: {list(mapping_df.columns)}"
            )

    # Keep only mapping rows marked as merge
    mapping_df["merge_decision_clean"] = mapping_df["merge_decision"].apply(
        clean_mapping_decision
    )

    active_mapping_df = mapping_df[
        mapping_df["merge_decision_clean"] == "merge"
    ].copy()

    # Remove empty raw/canonical rows
    active_mapping_df = active_mapping_df[
        ~active_mapping_df["raw_name"].apply(is_empty)
        & ~active_mapping_df["canonical_name"].apply(is_empty)
    ].copy()

    # Build mapping dictionary
    manual_mapping = {}

    duplicate_mapping_conflicts = []

    for _, row in active_mapping_df.iterrows():
        raw_name = str(row["raw_name"]).strip()
        canonical_name = str(row["canonical_name"]).strip()

        raw_key = normalize_for_matching(raw_name)

        if raw_key in manual_mapping and manual_mapping[raw_key] != canonical_name:
            duplicate_mapping_conflicts.append({
                "raw_name": raw_name,
                "first_canonical_name": manual_mapping[raw_key],
                "second_canonical_name": canonical_name,
            })

        manual_mapping[raw_key] = canonical_name

    if duplicate_mapping_conflicts:
        conflict_df = pd.DataFrame(duplicate_mapping_conflicts)
        conflict_output = OUTPUT_DIR / "04_duplicate_mapping_conflicts.csv"
        conflict_df.to_csv(conflict_output, index=False, encoding="utf-8-sig")

        raise ValueError(
            "Your manual mapping file contains conflicting mappings.\n"
            f"Check this file:\n{conflict_output}"
        )

    # Apply mapping
    df["dataset_original"] = df["dataset"].astype(str).str.strip()
    df["dataset_match_key"] = df["dataset_original"].apply(normalize_for_matching)

    df["canonical_dataset"] = df["dataset_original"]

    df["mapping_applied"] = "No"

    for idx, row in df.iterrows():
        key = row["dataset_match_key"]

        if key in manual_mapping:
            df.at[idx, "canonical_dataset"] = manual_mapping[key]
            df.at[idx, "mapping_applied"] = "Yes"

    # Create source info for mapped rows
    changed_rows = df[df["mapping_applied"] == "Yes"].copy()

    # Detect mapping rows that were not used
    dataset_keys_present = set(df["dataset_match_key"].tolist())

    unused_mapping_rows = []

    for _, row in active_mapping_df.iterrows():
        raw_name = str(row["raw_name"]).strip()
        raw_key = normalize_for_matching(raw_name)

        if raw_key not in dataset_keys_present:
            unused_mapping_rows.append({
                "raw_name": raw_name,
                "canonical_name": row["canonical_name"],
                "merge_decision": row["merge_decision"],
                "note": row.get("note", ""),
                "reason": "raw_name not found in combined raw table",
            })

    unused_mapping_df = pd.DataFrame(unused_mapping_rows)

    # Report
    report = pd.DataFrame([
        {
            "metric": "total_rows_in_combined_table",
            "value": len(df),
        },
        {
            "metric": "active_manual_mapping_rows",
            "value": len(active_mapping_df),
        },
        {
            "metric": "rows_changed_by_mapping",
            "value": len(changed_rows),
        },
        {
            "metric": "unique_original_dataset_names",
            "value": df["dataset_original"].nunique(),
        },
        {
            "metric": "unique_canonical_dataset_names_after_mapping",
            "value": df["canonical_dataset"].nunique(),
        },
        {
            "metric": "unused_manual_mapping_rows",
            "value": len(unused_mapping_df),
        },
    ])

    # Reorder key columns to front
    front_cols = [
        "source_file",
        "source_row_number",
        "dataset_original",
        "canonical_dataset",
        "mapping_applied",
        "dataset_match_key",
    ]

    other_cols = [col for col in df.columns if col not in front_cols]

    df = df[front_cols + other_cols]

    # Save outputs
    df.to_csv(
        MAPPED_LONG_TABLE_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    report.to_csv(
        MAPPING_REPORT_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    changed_rows.to_csv(
        APPLIED_MAPPING_ROWS_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    unused_mapping_df.to_csv(
        UNUSED_MAPPING_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Total rows: {len(df)}")
    print(f"Active manual mapping rows: {len(active_mapping_df)}")
    print(f"Rows changed by mapping: {len(changed_rows)}")
    print(f"Unique original dataset names: {df['dataset_original'].nunique()}")
    print(f"Unique canonical names after mapping: {df['canonical_dataset'].nunique()}")
    print(f"Unused manual mapping rows: {len(unused_mapping_df)}")

    print("\nSaved:")
    print(MAPPED_LONG_TABLE_OUTPUT)
    print(MAPPING_REPORT_OUTPUT)
    print(APPLIED_MAPPING_ROWS_OUTPUT)
    print(UNUSED_MAPPING_OUTPUT)

    print("\nDONE")


if __name__ == "__main__":
    main()