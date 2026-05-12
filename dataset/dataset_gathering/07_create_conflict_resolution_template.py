from pathlib import Path
import pandas as pd


# ============================================================
# PATH CONFIG
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = ROOT_DIR / "dataset_gathering_processed"

CONFLICT_INPUT = (
    PROCESSED_DIR
    / "06_draft_master_merge"
    / "06_attribute_conflicts.csv"
)

OUTPUT_DIR = PROCESSED_DIR / "07_conflict_resolution"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# OUTPUT FILES
# ============================================================

RESOLUTION_TEMPLATE_OUTPUT = OUTPUT_DIR / "07_attribute_conflict_resolution_template.csv"
CONFLICT_VALUES_EXPLODED_OUTPUT = OUTPUT_DIR / "07_conflict_values_exploded.csv"
STATS_OUTPUT = OUTPUT_DIR / "07_conflict_resolution_stats.csv"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def split_conflict_values(value):
    """
    Split conflict values stored as:
    value1 | value2 | value3
    """

    text = str(value).strip()

    if text == "":
        return []

    values = [v.strip() for v in text.split("|")]

    cleaned = []

    for value in values:
        if value and value not in cleaned:
            cleaned.append(value)

    return cleaned


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("STEP 07: CREATE ATTRIBUTE CONFLICT RESOLUTION TEMPLATE")
    print("=" * 80)

    if not CONFLICT_INPUT.exists():
        raise FileNotFoundError(
            f"Conflict input file not found:\n{CONFLICT_INPUT}\n\n"
            "Run 06_merge_to_draft_master.py first."
        )

    conflicts = pd.read_csv(
        CONFLICT_INPUT,
        dtype=str,
        keep_default_na=False
    )

    if conflicts.empty:
        print("No conflicts found. Nothing to resolve.")

        empty_template = pd.DataFrame(columns=[
            "Dataset",
            "column",
            "values_found",
            "resolved_value",
            "resolution_decision",
            "note",
            "source_files",
            "source_trace"
        ])

        empty_template.to_csv(
            RESOLUTION_TEMPLATE_OUTPUT,
            index=False,
            encoding="utf-8-sig"
        )

        stats = pd.DataFrame([
            {"metric": "conflict_rows", "value": 0},
            {"metric": "datasets_with_conflicts", "value": 0},
        ])

        stats.to_csv(
            STATS_OUTPUT,
            index=False,
            encoding="utf-8-sig"
        )

        print("DONE")
        return

    required_cols = [
        "Dataset",
        "column",
        "values_found",
        "source_files",
        "source_trace",
    ]

    for col in required_cols:
        if col not in conflicts.columns:
            raise ValueError(
                f"Column '{col}' not found in conflict file.\n"
                f"Available columns: {list(conflicts.columns)}"
            )

    # ------------------------------------------------------------
    # 1. Create manual resolution template
    # ------------------------------------------------------------

    template_rows = []

    for _, row in conflicts.iterrows():
        values_found = str(row["values_found"]).strip()
        possible_values = split_conflict_values(values_found)

        template_rows.append({
            "Dataset": row["Dataset"],
            "column": row["column"],
            "values_found": values_found,
            "resolved_value": "",
            "resolution_decision": "",
            "note": "",
            "source_files": row["source_files"],
            "source_trace": row["source_trace"],
            "number_of_values": len(possible_values),
        })

    resolution_template = pd.DataFrame(template_rows)

    resolution_template = resolution_template.sort_values(
        by=["Dataset", "column"],
        key=lambda col: col.str.lower()
    )

    # ------------------------------------------------------------
    # 2. Create exploded value file for easier inspection
    # ------------------------------------------------------------

    exploded_rows = []

    for _, row in conflicts.iterrows():
        possible_values = split_conflict_values(row["values_found"])

        for value in possible_values:
            exploded_rows.append({
                "Dataset": row["Dataset"],
                "column": row["column"],
                "possible_value": value,
                "source_files": row["source_files"],
                "source_trace": row["source_trace"],
            })

    exploded_df = pd.DataFrame(exploded_rows)

    if not exploded_df.empty:
        exploded_df = exploded_df.sort_values(
            by=["Dataset", "column", "possible_value"],
            key=lambda col: col.str.lower()
        )

    # ------------------------------------------------------------
    # 3. Stats
    # ------------------------------------------------------------

    stats = pd.DataFrame([
        {
            "metric": "conflict_rows",
            "value": len(conflicts),
        },
        {
            "metric": "datasets_with_conflicts",
            "value": conflicts["Dataset"].nunique(),
        },
        {
            "metric": "columns_with_conflicts",
            "value": conflicts["column"].nunique(),
        },
    ])

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    resolution_template.to_csv(
        RESOLUTION_TEMPLATE_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    exploded_df.to_csv(
        CONFLICT_VALUES_EXPLODED_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    stats.to_csv(
        STATS_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Conflict rows: {len(conflicts)}")
    print(f"Datasets with conflicts: {conflicts['Dataset'].nunique()}")
    print(f"Columns with conflicts: {conflicts['column'].nunique()}")

    print("\nSaved:")
    print(RESOLUTION_TEMPLATE_OUTPUT)
    print(CONFLICT_VALUES_EXPLODED_OUTPUT)
    print(STATS_OUTPUT)

    print("\nDONE")


if __name__ == "__main__":
    main()