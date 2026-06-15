"""
clip_duration_check.py
======================
Quick per-dataset clip-duration profile — use it to choose WINDOW_S for the
trajectory-fan / density figures.

Reads ONLY the t_rel_s column from each clip (so it's fast), and prints per
dataset: number of clips, min / median / p90 / max duration in seconds, and the
percentage of clips shorter than a few candidate window sizes. If most clips are
shorter than your chosen WINDOW_S, that window is too big — it just returns each
short clip whole instead of splitting anything.
"""

import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    from tqdm import tqdm
except Exception:                       # graceful no-op if tqdm isn't installed
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

# ── config ───────────────────────────────────────────────────────────────────
INERTIAL_ROOT     = Path("/Users/edward/Downloads/dataset/inertial_data")
DATASETS          = None            # None -> every folder under inertial_data/
TIME_COL          = "t_rel_s"
N_WORKERS         = 8
WINDOW_CANDIDATES = [10, 15, 19, 20] # report % of clips shorter than each


def discover():
    if not INERTIAL_ROOT.exists():
        return []
    return [d.name for d in sorted(INERTIAL_ROOT.iterdir())
            if d.is_dir() and any(d.glob("clip_*.parquet"))]


def clip_duration(fp):
    """Duration (s) of one clip from its t_rel_s column, or None on failure."""
    try:
        t = pd.read_parquet(fp, columns=[TIME_COL])[TIME_COL].to_numpy(dtype=float)
        t = t[np.isfinite(t)]
        return float(t.max() - t.min()) if t.size else None
    except Exception:
        return None


def main():
    datasets = DATASETS if DATASETS else discover()
    if not datasets:
        print(f"No dataset folders with clip_*.parquet under {INERTIAL_ROOT}")
        return

    cand = "".join(f"{('<'+str(w)+'s'):>7}" for w in WINDOW_CANDIDATES)
    hdr = f"  {'dataset':<12}{'clips':>8}{'min':>8}{'median':>9}{'p90':>8}{'max':>9}    {cand}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for ds in datasets:
        clips = sorted((INERTIAL_ROOT / ds).glob("clip_*.parquet"))
        if not clips:
            continue
        durs = []
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = [ex.submit(clip_duration, fp) for fp in clips]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"  {ds:<12}", leave=False):
                d = fut.result()
                if d is not None:
                    durs.append(d)
        if not durs:
            print(f"  {ds:<12}{0:>8}   (no readable t_rel_s)")
            continue
        durs = np.array(durs)
        pcts = "".join(f"{100*np.mean(durs < w):>6.0f}%" for w in WINDOW_CANDIDATES)
        print(f"  {ds:<12}{len(durs):>8}{durs.min():>8.1f}{np.median(durs):>9.1f}"
              f"{np.percentile(durs,90):>8.1f}{durs.max():>9.1f}    {pcts}")


if __name__ == "__main__":
    main()