from pathlib import Path
import pandas as pd


# ============================================================
# PATH CONFIG
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = ROOT_DIR / "dataset_gathering_processed"

INPUT_FILE = (
    PROCESSED_DIR
    / "02_duplicate_detection"
    / "02_unique_dataset_names.csv"
)

OUTPUT_DIR = PROCESSED_DIR / "03_manual_similarity_mapping"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# OUTPUT FILES
# ============================================================

UNIQUE_NAMES_OUTPUT = OUTPUT_DIR / "03_unique_dataset_names_for_manual_review.csv"
MANUAL_MAPPING_OUTPUT = OUTPUT_DIR / "03_manual_similar_dataset_mapping.csv"
EXAMPLE_OUTPUT = OUTPUT_DIR / "03_manual_mapping_examples.csv"


# ============================================================
# MAIN
# ============================================================

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}\n\n"
            "Run 02_detect_exact_duplicates.py first."
        )

    print("=" * 80)
    print("STEP 03: CREATE MANUAL SIMILAR-DATASET MAPPING TEMPLATE")
    print("=" * 80)

    df = pd.read_csv(
        INPUT_FILE,
        dtype=str,
        keep_default_na=False
    )

    if "recommended_display_name" not in df.columns:
        raise ValueError(
            "Column 'recommended_display_name' not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    unique_names = (
        df["recommended_display_name"]
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .sort_values(key=lambda x: x.str.lower())
        .tolist()
    )

    # File 1: reference list of all unique dataset names
    unique_names_df = pd.DataFrame({
        "dataset_name": unique_names,
        "manual_note": ""
    })

    unique_names_df.to_csv(
        UNIQUE_NAMES_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    # File 2: empty manual mapping file to fill yourself
    manual_mapping_df = pd.DataFrame(columns=[
        "raw_name",
        "canonical_name",
        "merge_decision",
        "note"
    ])

    manual_mapping_df.to_csv(
        MANUAL_MAPPING_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    # File 3: examples
    example_df = pd.DataFrame([
        {
            "raw_name": "apollo",
            "canonical_name": "ApolloScape",
            "merge_decision": "merge",
            "note": "manual decision"
        },
        {
            "raw_name": "Apollo Open Platform",
            "canonical_name": "ApolloScape",
            "merge_decision": "merge",
            "note": "manual decision"
        },
        {
            "raw_name": "comma ld",
            "canonical_name": "Comma2k19",
            "merge_decision": "merge",
            "note": "manual decision"
        },
        {
            "raw_name": "comma2k19ld",
            "canonical_name": "Comma2k19",
            "merge_decision": "merge",
            "note": "manual decision"
        },
        {
            "raw_name": "v2x-sim",
            "canonical_name": "V2X-Sim 1.0",
            "merge_decision": "merge",
            "note": "manual decision"
        },
    ])

    example_df.to_csv(
        EXAMPLE_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Unique dataset names found: {len(unique_names)}")
    print()
    print("Saved:")
    print(UNIQUE_NAMES_OUTPUT)
    print(MANUAL_MAPPING_OUTPUT)
    print(EXAMPLE_OUTPUT)
    print()
    print("DONE")


if __name__ == "__main__":
    main()