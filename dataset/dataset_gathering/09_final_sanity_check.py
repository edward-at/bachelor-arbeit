import pandas as pd
from pathlib import Path

FINAL_DIR = Path("dataset_gathering_final")

master_file = FINAL_DIR / "dataset_master_final.csv"
report_file = FINAL_DIR / "final_sanity_report.txt"
missing_file = FINAL_DIR / "rows_with_missing_dataset_info.csv"
duplicate_file = FINAL_DIR / "duplicate_dataset_ids_final.csv"

df = pd.read_csv(master_file, dtype=str)

lines = []

lines.append("=" * 80)
lines.append("FINAL SANITY CHECK REPORT")
lines.append("=" * 80)

lines.append(f"\nRows: {len(df)}")
lines.append(f"Columns: {len(df.columns)}")

# -------------------------------------------------
# 1. Required column check
# -------------------------------------------------

required_cols = ["Dataset_ID", "Dataset"]

lines.append("\nRequired column check:")
for col in required_cols:
    if col in df.columns:
        lines.append(f"  ✅ {col} exists")
    else:
        lines.append(f"  ❌ {col} missing")

# -------------------------------------------------
# 2. Duplicate Dataset_ID check
# -------------------------------------------------

if "Dataset_ID" in df.columns:
    duplicate_ids = df[df["Dataset_ID"].duplicated(keep=False)].copy()

    lines.append(f"\nDuplicate Dataset_ID rows: {len(duplicate_ids)}")

    if len(duplicate_ids) > 0:
        duplicate_ids.to_csv(duplicate_file, index=False)
        lines.append(f"  ⚠️ Saved duplicate rows to: {duplicate_file}")
    else:
        lines.append("  ✅ No duplicate Dataset_IDs found")

# -------------------------------------------------
# 3. Missing Dataset_ID / Dataset check
# -------------------------------------------------

missing_info = df[
    df["Dataset_ID"].isna()
    | df["Dataset"].isna()
    | (df["Dataset_ID"].astype(str).str.strip() == "")
    | (df["Dataset"].astype(str).str.strip() == "")
].copy()

lines.append(f"\nRows with missing Dataset_ID or Dataset: {len(missing_info)}")

if len(missing_info) > 0:
    missing_info.to_csv(missing_file, index=False)
    lines.append(f"  ⚠️ Saved missing-info rows to: {missing_file}")
else:
    lines.append("  ✅ No missing Dataset_ID/Dataset rows")

# -------------------------------------------------
# 4. Missingness per column
# -------------------------------------------------

lines.append("\nMissing values per column:")

missing_summary = []

for col in df.columns:
    missing_count = df[col].isna().sum() + (df[col].astype(str).str.strip().isin(["", "nan", "None", "N/A", "na", "-"])).sum()
    missing_summary.append((col, missing_count, round(missing_count / len(df) * 100, 1)))

missing_summary = sorted(missing_summary, key=lambda x: x[1], reverse=True)

for col, count, pct in missing_summary:
    lines.append(f"  {col}: {count} missing ({pct}%)")

# -------------------------------------------------
# 5. Source table coverage
# -------------------------------------------------

if "Source_Tables" in df.columns:
    lines.append("\nSource table coverage:")
    source_counts = {}

    for sources in df["Source_Tables"].dropna():
        for s in str(sources).split(";"):
            s = s.strip()
            if s:
                source_counts[s] = source_counts.get(s, 0) + 1

    for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  {source}: {count} datasets")

# -------------------------------------------------
# Save report
# -------------------------------------------------

report_text = "\n".join(lines)

print(report_text)

with open(report_file, "w", encoding="utf-8") as f:
    f.write(report_text)

print("\n✅ Saved sanity report:")
print(report_file)