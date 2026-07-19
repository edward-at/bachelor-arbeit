"""
rerun_dataset.py — re-run the standardization for named dataset(s) only.

Usage:
    python rerun_dataset.py 4seasons
    python rerun_dataset.py durlar boreas_rt

Place next to standardize_inertial_data.py. Prints every receipt (purged /
BLOCKLISTED / dropped clip / v4 gate / ERROR) for just the named datasets,
in seconds instead of the hours a full run costs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import standardize_inertial_data as std
from tqdm import tqdm

# process_dataset reads module globals that only main() initialises — set them
# the same way main() does, via the module's own root finder.
std.BASE_DIR    = std._find_dataset_root()
std.PARQUET_DIR = std.BASE_DIR / "parquet_data"
std.OUTPUT_DIR  = std.BASE_DIR / "inertial_data"

names = [n.lower() for n in sys.argv[1:]]
if not names:
    sys.exit("usage: python rerun_dataset.py <dataset> [<dataset> ...]")

print(f"Input : {std.PARQUET_DIR}")
print(f"Output: {std.OUTPUT_DIR}")
print(f"Pipeline version: {std.PIPELINE_VERSION}")

for ds in names:
    print(f"\n════════ {ds} ════════")
    bar = tqdm(total=0, desc=ds, leave=False)
    try:
        saved, n_err = std.process_dataset(ds, bar)
    finally:
        bar.close()
    print(f"{ds}: saved {saved} clips, {n_err} file-level errors")