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

OUTPUT_DIR = PROCESSED_DIR / "06_draft_master_merge"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# OUTPUT FILES
# ============================================================

DRAFT_MASTER_OUTPUT = OUTPUT_DIR / "06_draft_master_table.csv"
CONFLICTS_OUTPUT = OUTPUT_DIR / "06_attribute_conflicts.csv"
SOURCE_TRACE_OUTPUT = OUTPUT_DIR / "06_source_trace_by_dataset.csv"
STATS_OUTPUT = OUTPUT_DIR / "06_draft_master_stats.csv"


# ============================================================
# CONFIG
# ============================================================

# These columns are metadata from the processing pipeline, not real dataset attributes.
PIPELINE_COLUMNS = {
    "source_file",
    "source_row_number",
    "dataset_original",
    "dataset_match_key",
    "mapping_applied",
    "dataset",
}


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


def unique_non_empty(values):
    """
    Return unique non-empty values while preserving order.
    """

    result = []

    for value in values:
        text = str(value).strip()

        if not is_empty(text) and text not in result:
            result.append(text)

    return result


def join_unique(values):
    values = unique_non_empty(values)
    return " | ".join(values)


def make_source_trace(group):
    """
    Create readable source trace for each dataset.
    """

    traces = []

    for _, row in group.iterrows():
        source_file = str(row.get("source_file", "")).strip()
        source_row = str(row.get("source_row_number", "")).strip()
        original_name = str(row.get("dataset_original", "")).strip()

        trace = f"{source_file}: row {source_row} ({original_name})"

        if trace not in traces:
            traces.append(trace)

    return " | ".join(traces)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("STEP 06: MERGE ROWS INTO DRAFT MASTER TABLE")
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

    df["canonical_dataset"] = df["canonical_dataset"].astype(str).str.strip()
    df = df[~df["canonical_dataset"].apply(is_empty)].copy()

    all_columns = list(df.columns)

    attribute_columns = [
        col for col in all_columns
        if col not in PIPELINE_COLUMNS
        and col != "canonical_dataset"
    ]

    master_rows = []
    conflict_rows = []
    source_trace_rows = []

    for canonical_dataset, group in df.groupby("canonical_dataset", sort=True):
        master_row = {
            "Dataset": canonical_dataset,
            "source_files": join_unique(group["source_file"]) if "source_file" in group.columns else "",
            "source_row_count": len(group),
            "original_name_variants": join_unique(group["dataset_original"]) if "dataset_original" in group.columns else "",
            "source_trace": make_source_trace(group),
        }

        source_trace_rows.append({
            "Dataset": canonical_dataset,
            "source_row_count": len(group),
            "source_files": master_row["source_files"],
            "original_name_variants": master_row["original_name_variants"],
            "source_trace": master_row["source_trace"],
        })

        for col in attribute_columns:
            values = unique_non_empty(group[col].tolist())

            if len(values) == 0:
                master_row[col] = ""

            elif len(values) == 1:
                master_row[col] = values[0]

            else:
                # Keep all values for now to avoid data loss.
                # Later we resolve this using the conflict file.
                master_row[col] = " | ".join(values)

                conflict_rows.append({
                    "Dataset": canonical_dataset,
                    "column": col,
                    "number_of_values": len(values),
                    "values_found": " | ".join(values),
                    "source_files": join_unique(group["source_file"]) if "source_file" in group.columns else "",
                    "source_trace": make_source_trace(group),
                })

        master_rows.append(master_row)

    draft_master = pd.DataFrame(master_rows)

    # Sort columns: important metadata first, then attributes
    front_columns = [
        "Dataset",
        "source_files",
        "source_row_count",
        "original_name_variants",
        "source_trace",
    ]

    remaining_columns = [
        col for col in draft_master.columns
        if col not in front_columns
    ]

    draft_master = draft_master[front_columns + remaining_columns]

    conflicts_df = pd.DataFrame(conflict_rows)

    if not conflicts_df.empty:
        conflicts_df = conflicts_df.sort_values(
            by=["Dataset", "column"],
            key=lambda col: col.str.lower() if col.dtype == "object" else col
        )

    source_trace_df = pd.DataFrame(source_trace_rows)

    stats = pd.DataFrame([
        {
            "metric": "input_rows",
            "value": len(df),
        },
        {
            "metric": "draft_master_rows_unique_datasets",
            "value": len(draft_master),
        },
        {
            "metric": "attribute_columns_merged",
            "value": len(attribute_columns),
        },
        {
            "metric": "datasets_with_multiple_source_rows",
            "value": (draft_master["source_row_count"].astype(int) > 1).sum(),
        },
        {
            "metric": "attribute_conflict_rows",
            "value": len(conflicts_df),
        },
        {
            "metric": "datasets_with_attribute_conflicts",
            "value": conflicts_df["Dataset"].nunique() if not conflicts_df.empty else 0,
        },
    ])

    draft_master.to_csv(
        DRAFT_MASTER_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    conflicts_df.to_csv(
        CONFLICTS_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    source_trace_df.to_csv(
        SOURCE_TRACE_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    stats.to_csv(
        STATS_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Input rows: {len(df)}")
    print(f"Draft master rows: {len(draft_master)}")
    print(f"Attribute columns merged: {len(attribute_columns)}")
    print(f"Attribute conflict rows: {len(conflicts_df)}")
    print(
        "Datasets with attribute conflicts: "
        f"{conflicts_df['Dataset'].nunique() if not conflicts_df.empty else 0}"
    )

    print("\nSaved:")
    print(DRAFT_MASTER_OUTPUT)
    print(CONFLICTS_OUTPUT)
    print(SOURCE_TRACE_OUTPUT)
    print(STATS_OUTPUT)

    print("\nDONE")


if __name__ == "__main__":
    main()