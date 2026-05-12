from pathlib import Path
import pandas as pd
import re


# ============================================================
# PATH CONFIG
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

RAW_DIR = ROOT_DIR / "dataset_gathering_raw"
PROCESSED_DIR = ROOT_DIR / "dataset_gathering_processed"

OUTPUT_DIR = PROCESSED_DIR / "01_combined_raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# OUTPUT FILES
# ============================================================

COMBINED_OUTPUT = OUTPUT_DIR / "01_combined_raw_long_table.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "01_combined_raw_file_summary.csv"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_column_name(column_name):
    """
    Make column names consistent:
    Example:
    'Dataset Name' -> 'dataset_name'
    'Public Availability' -> 'public_availability'
    """

    column_name = str(column_name).strip()
    column_name = column_name.replace("\ufeff", "")
    column_name = column_name.lower()
    column_name = re.sub(r"[^\w]+", "_", column_name)
    column_name = re.sub(r"_+", "_", column_name)
    column_name = column_name.strip("_")

    return column_name


def read_csv_safely(csv_path):
    """
    Try common CSV encodings and separators.
    Everything is read as text.
    """

    attempts = [
        {"encoding": "utf-8-sig", "sep": ","},
        {"encoding": "utf-8", "sep": ","},
        {"encoding": "latin1", "sep": ","},

        {"encoding": "utf-8-sig", "sep": ";"},
        {"encoding": "utf-8", "sep": ";"},
        {"encoding": "latin1", "sep": ";"},

        {"encoding": "utf-8-sig", "sep": "\t"},
        {"encoding": "utf-8", "sep": "\t"},
        {"encoding": "latin1", "sep": "\t"},
    ]

    last_error = None

    for attempt in attempts:
        try:
            df = pd.read_csv(
                csv_path,
                dtype=str,
                keep_default_na=False,
                **attempt
            )

            return df, attempt

        except Exception as error:
            last_error = error

    raise last_error


# ============================================================
# MAIN
# ============================================================

def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw folder not found: {RAW_DIR}")

    csv_files = sorted(RAW_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {RAW_DIR}")

    print("=" * 80)
    print("STEP 01: COMBINE RAW CSV FILES")
    print("=" * 80)
    print(f"Raw folder: {RAW_DIR}")
    print(f"CSV files found: {len(csv_files)}")

    all_dataframes = []
    summary_rows = []

    for csv_file in csv_files:
        print(f"\nReading: {csv_file.name}")

        try:
            df, read_settings = read_csv_safely(csv_file)

            original_columns = list(df.columns)

            # Clean column names
            df.columns = [clean_column_name(col) for col in df.columns]

            # Add tracking columns
            df.insert(0, "source_file", csv_file.name)
            df.insert(1, "source_row_number", range(1, len(df) + 1))

            all_dataframes.append(df)

            summary_rows.append({
                "file": csv_file.name,
                "status": "OK",
                "rows": len(df),
                "columns_after_cleaning": len(df.columns),
                "encoding_used": read_settings["encoding"],
                "separator_used": repr(read_settings["sep"]),
                "original_columns": " | ".join(original_columns),
                "cleaned_columns": " | ".join(df.columns),
                "error": "",
            })

            print(f"  Status: OK")
            print(f"  Rows: {len(df)}")
            print(f"  Columns after cleaning: {len(df.columns)}")

        except Exception as error:
            summary_rows.append({
                "file": csv_file.name,
                "status": "ERROR",
                "rows": "",
                "columns_after_cleaning": "",
                "encoding_used": "",
                "separator_used": "",
                "original_columns": "",
                "cleaned_columns": "",
                "error": str(error),
            })

            print(f"  Status: ERROR")
            print(f"  Error: {error}")

    if not all_dataframes:
        raise RuntimeError("No CSV files could be read successfully.")

    # Combine vertically.
    # Different columns are allowed.
    combined_df = pd.concat(
        all_dataframes,
        ignore_index=True,
        sort=False
    )

    # Save combined table
    combined_df.to_csv(
        COMBINED_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    # Save file summary
    pd.DataFrame(summary_rows).to_csv(
        SUMMARY_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Combined raw table saved to:")
    print(COMBINED_OUTPUT)
    print()
    print(f"Summary saved to:")
    print(SUMMARY_OUTPUT)
    print()
    print(f"Total combined rows: {len(combined_df)}")
    print(f"Total combined columns: {len(combined_df.columns)}")


if __name__ == "__main__":
    main()