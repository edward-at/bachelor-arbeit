#!/usr/bin/env python3
"""
inspect_clip_mapping.py — short diagnostic. For every dataset, prints the
evidence needed to check the std clip_id -> inertial filename mapping:

  * sample std_data clip_id values (+ dtype, unique count)
  * sample inertial file stems (+ file count)
  * the columns inside the first inertial file
  * the first value of any id-like column found inside that file
  * an estimated resolve rate using the current (raw + numeric-normalized)
    resolver on a 200-clip sample

Output goes to the console only. No files are written.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.append(str(Path(__file__).resolve().parents[1]))
from thesis_paths import DATA_ROOT, REPO_ROOT  # noqa: E402
INERTIAL_DIR = DATA_ROOT / "inertial_data"
STD_DATA_DIR = DATA_ROOT / "std_data"

ID_CANDIDATES = ["clip_id", "orig_clip_id", "source_clip", "clip",
                 "clip_name", "scene_id", "scene_token", "sequence_id",
                 "sequence", "seq_id", "recording_id", "recording",
                 "source_file", "orig_file", "drive_id", "log_id"]

_NUM_RE = re.compile(r"(?:clip[_-]?)?0*(\d+)$")


def norm_key(s) -> str | None:
    s = str(s)
    if s.endswith(".parquet"):
        s = s[:-8]
    m = _NUM_RE.fullmatch(s)
    return f"#num:{int(m.group(1))}" if m else None


def main():
    datasets = sorted(p.stem for p in STD_DATA_DIR.glob("*.parquet"))
    print(f"[inspect] datasets: {datasets}\n")
    for ds in datasets:
        print(f"=== {ds} " + "=" * (60 - len(ds)))
        std = pd.read_parquet(STD_DATA_DIR / f"{ds}.parquet",
                              columns=["clip_id"])
        uniq = std["clip_id"].astype(str).unique()
        print(f"  std clip_id : dtype={std['clip_id'].dtype}, "
              f"n_unique={len(uniq):,}")
        print(f"    first: {list(uniq[:3])}")
        print(f"    last : {list(uniq[-2:])}")

        ds_dir = INERTIAL_DIR / ds
        if not ds_dir.exists():
            print(f"  inertial dir MISSING: {ds_dir}\n")
            continue
        files = sorted(ds_dir.glob("*.parquet"))
        stems = [f.stem for f in files]
        print(f"  files       : n={len(files):,}")
        print(f"    first: {stems[:3]}")
        print(f"    last : {stems[-2:] if len(stems) >= 2 else stems}")
        print(f"    n_files == n_unique_clips ? "
              f"{'YES' if len(files) == len(uniq) else 'NO'}")

        if files:
            try:
                names = pq.ParquetFile(files[0]).schema_arrow.names
            except Exception as e:
                names = []
                print(f"  could not read first file schema: {e}")
            if names:
                shown = names[:24]
                more = " ..." if len(names) > len(shown) else ""
                print(f"  first-file columns ({len(names)}): {shown}{more}")
                for cand in ID_CANDIDATES:
                    if cand in names:
                        try:
                            val = pq.read_table(
                                files[0], columns=[cand]).column(0)[0].as_py()
                            print(f"    id-like column '{cand}' "
                                  f"first value: {val!r}")
                        except Exception as e:
                            print(f"    id-like column '{cand}': "
                                  f"unreadable ({e})")

            # estimated resolve rate with the numeric-normalizing resolver
            index = set(stems)
            nindex = {norm_key(s) for s in stems} - {None}
            sample = uniq[:200]
            hits = 0
            for cid in sample:
                if (cid in index or f"clip_{cid}" in index
                        or norm_key(cid) in nindex):
                    hits += 1
            print(f"  v2-resolver estimated hit rate on {len(sample)}-sample: "
                  f"{hits}/{len(sample)}")
        print()


if __name__ == "__main__":
    main()