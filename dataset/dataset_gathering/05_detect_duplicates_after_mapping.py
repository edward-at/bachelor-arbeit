from pathlib import Path
import pandas as pd


# ============================================================
# PATH CONFIG
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = ROOT_DIR / "dataset_gathering_processed"

INPUT_FILE = (
    PROCESSED_DIR
    / "04_mapped_dataset_names"
    / "04_mapped_long_table.csv"
)

OUTPUT_DIR = PROCESSED_DIR / "05_duplicates_after_mapping"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# OUTPUT FILES
# ============================================================

DUPLICATE_SUMMARY_OUTPUT = OUTPUT_DIR / "05_duplicate_canonical_dataset_summary.csv"
DUPLICATE_ROWS_OUTPUT = OUTPUT_DIR / "05_duplicate_canonical_dataset_rows.csv"
UNIQUE_CANONICAL_OUTPUT = OUTPUT_DIR / "05_unique_canonical_dataset_names.csv"
STATS_OUTPUT = OUTPUT_DIR / "05_duplicates_after_mapping_stats.csv"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

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


def join_unique(values):
    cleaned = []

    for value in values:
        text = str(value).strip()

        if not is_empty(text) and text not in cleaned:
            cleaned.append(text)

    return " | ".join(cleaned)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("STEP 05: DETECT DUPLICATES AFTER MANUAL MAPPING")
    print("=" * 80)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}\n\n"
            "Run 04_apply_manual_name_mapping.py first."
        )

    df = pd.read_csv(
        INPUT_FILE,
        dtype=str,
        keep_default_na=False
    )

    if "canonical_dataset" not in df.columns:
        raise ValueError(
            "Column 'canonical_dataset' not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    # Remove rows without canonical dataset name
    df["canonical_dataset"] = df["canonical_dataset"].astype(str).str.strip()
    df = df[~df["canonical_dataset"].apply(is_empty)].copy()

    # Count canonical dataset appearances
    counts = df["canonical_dataset"].value_counts()

    duplicate_names = counts[counts > 1].index.tolist()

    duplicate_rows = df[df["canonical_dataset"].isin(duplicate_names)].copy()

    # Build duplicate summary
    summary_rows = []

    for canonical_name in duplicate_names:
        group = df[df["canonical_dataset"] == canonical_name].copy()

        source_locations = []

        for _, row in group.iterrows():
            source_locations.append(
                f"{row.get('source_file', '')}: row {row.get('source_row_number', '')}"
            )

        summary_rows.append({
            "canonical_dataset": canonical_name,
            "row_count": len(group),
            "original_name_variants": join_unique(group.get("dataset_original", [])),
            "source_files": join_unique(group.get("source_file", [])),
            "source_locations": " | ".join(source_locations),
        })

    duplicate_summary = pd.DataFrame(summary_rows)

    if not duplicate_summary.empty:
        duplicate_summary = duplicate_summary.sort_values(
            by=["row_count", "canonical_dataset"],
            ascending=[False, True]
        )

    # Unique canonical dataset list
    unique_rows = []

    for canonical_name, group in df.groupby("canonical_dataset"):
        unique_rows.append({
            "canonical_dataset": canonical_name,
            "row_count": len(group),
            "original_name_variants": join_unique(group.get("dataset_original", [])),
            "source_files": join_unique(group.get("source_file", [])),
        })

    unique_df = pd.DataFrame(unique_rows).sort_values(
        by="canonical_dataset",
        key=lambda col: col.str.lower()
    )

    # Stats
    stats = pd.DataFrame([
        {
            "metric": "total_rows_after_mapping",
            "value": len(df),
        },
        {
            "metric": "unique_canonical_dataset_names",
            "value": df["canonical_dataset"].nunique(),
        },
        {
            "metric": "canonical_dataset_names_with_duplicates",
            "value": len(duplicate_names),
        },
        {
            "metric": "rows_in_duplicate_canonical_groups",
            "value": len(duplicate_rows),
        },
        {
            "metric": "canonical_dataset_names_appearing_once",
            "value": (counts == 1).sum(),
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

    unique_df.to_csv(
        UNIQUE_CANONICAL_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    stats.to_csv(
        STATS_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Total rows after mapping: {len(df)}")
    print(f"Unique canonical dataset names: {df['canonical_dataset'].nunique()}")
    print(f"Canonical dataset names with duplicates: {len(duplicate_names)}")
    print(f"Rows in duplicate canonical groups: {len(duplicate_rows)}")
    print(f"Canonical dataset names appearing once: {(counts == 1).sum()}")

    print("\nSaved:")
    print(DUPLICATE_SUMMARY_OUTPUT)
    print(DUPLICATE_ROWS_OUTPUT)
    print(UNIQUE_CANONICAL_OUTPUT)
    print(STATS_OUTPUT)

    print("\nDONE")


if __name__ == "__main__":
    main()
    