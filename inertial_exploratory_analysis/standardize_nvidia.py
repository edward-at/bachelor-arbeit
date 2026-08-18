"""
Rerun nvidia only — adapted to v4.4's 3-tuple return contract.
Run from inertial_exploratory_analysis:
    caffeinate -i python Test.py
"""

import inspect
import re
import traceback
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent.resolve()
MAIN = SCRIPT_DIR / "standardize_inertial_data.py"
ROOT = Path.home() / "Downloads" / "dataset"
DS   = "nvidia"

assert MAIN.exists(), f"pipeline not found: {MAIN}"
assert (ROOT / "parquet_data" / DS).is_dir(), f"parquet_data/{DS} not found"

src = MAIN.read_text()
src = re.sub(r"BASE_DIR\s*=\s*_find_dataset_root\(\)",
             f'BASE_DIR = Path(r"{ROOT}")', src)
m = types.ModuleType("s14")
m.__file__ = str(MAIN)
exec(compile(src, str(MAIN), "exec"), m.__dict__)
m.PARQUET_DIR = ROOT / "parquet_data"
m.OUTPUT_DIR  = ROOT / "inertial_data"

out_dir = m.OUTPUT_DIR / DS
out_dir.mkdir(parents=True, exist_ok=True)
files = sorted((m.PARQUET_DIR / DS).rglob("*.parquet"))

print("=" * 55)
print(f" Pipeline : {MAIN.name} (v4.4)")
print(f" Files    : {len(files)}")
print(f" Output   : {out_dir}")
print("=" * 55)

saved = n_err = 0
used_native = False

# ── Path A: v4.4's own per-dataset entry point ────────────────────────────────
if hasattr(m, "process_dataset"):
    sig = inspect.signature(m.process_dataset)
    print(f"\nUsing v4.4's own process_dataset{sig}")
    try:
        with tqdm(total=len(files), desc="Overall", unit="file",
                  dynamic_ncols=True) as bar:
            if len(sig.parameters) >= 2:
                result = m.process_dataset(DS, bar)
            else:
                result = m.process_dataset(DS)
        if isinstance(result, tuple) and len(result) >= 2:
            saved, n_err = result[0], result[1]
        else:
            saved = result if isinstance(result, int) else 0
        used_native = True
    except Exception:
        print("\nprocess_dataset failed — falling back to manual loop:")
        print(traceback.format_exc()[:800])

# ── Path B: manual loop, 3-tuple aware ────────────────────────────────────────
if not used_native:
    save_params = list(inspect.signature(m.save_clip).parameters)
    n_data = len(save_params) - 1          # last param assumed to be out_path
    print(f"\nManual mode. save_clip params: {save_params} → passing {n_data} data args")

    import os
    MAX_WORKERS = min(os.cpu_count() or 4, 8)
    errors  = []
    results = [None] * len(files)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        fut = {pool.submit(m._process_one_file, (f, m.LOADERS[DS])): i
               for i, f in enumerate(files)}
        with tqdm(total=len(files), unit="file", dynamic_ncols=True) as bar:
            for future in as_completed(fut):
                idx = fut[future]
                ok, errs = [], []
                for item in future.result():
                    if isinstance(item, str) and item.startswith("ERROR::"):
                        errs.append(item)
                    else:
                        ok.append(item)
                results[idx] = ok
                errors.extend(errs)
                bar.update(1)

    for err in errors[:10]:
        parts = err.split("::", 2)
        print(f"\n[ERROR] {parts[1] if len(parts) > 1 else '?'}:\n"
              f"{parts[2][:400] if len(parts) > 2 else err}")
    n_err = len(errors)

    clip_index = 1
    for clip_list in results:
        if not clip_list:
            continue
        for item in clip_list:
            if not (isinstance(item, tuple) and len(item) >= 2):
                continue
            df = item[0]
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            args = list(item[:n_data])
            while len(args) < n_data:      # pad if save_clip wants more dicts
                args.append({})
            m.save_clip(*args, out_dir / f"clip_{clip_index:04d}.parquet")
            clip_index += 1
            saved += 1

print(f"\n{'='*55}")
print(f" Done. {saved} clips saved → {out_dir}")
if n_err:
    print(f" Errors: {n_err} file(s) failed")
print("=" * 55)