#!/usr/bin/env python3
"""curation_stale_audit.py v2 — per-chapter anchors, no false alarms.

v1 compared every figure against cluster_assignments.parquet. That anchor
is wrong for two groups and v1 therefore over-reported:
  * Chapter 8 reads the std_data / summary layers, never the regime
    assignments — its correct anchor is the newest std_data parquet.
  * The dataset-survey figures (camera counts, DSS scores, maps, ...)
    depend on no kinematic layer at all — frame-independent, exempt.

Verdict per figure: FRESH (newer than its own anchor), STALE (older), or
EXEMPT. Read-only; nothing is modified.

Usage:  python curation_stale_audit.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from curation_common import PROJECT_ROOT, EVAL_DIR

FRAME = PROJECT_ROOT / "dataset" / "regime_data" / "cluster_assignments.parquet"
STD_DIR = PROJECT_ROOT / "dataset" / "std_data"
SUMMARIES = PROJECT_ROOT / "dataset" / "subset_data" / "window_summaries.parquet"
FIG_ROOT = PROJECT_ROOT / "bachelor-arbeit"

# order matters: first hit wins
CHAPTER_HINTS = [
    ("13 curation",            ["figures/curation"],                 "frame"),
    ("12 sampling evaluation", ["subset_eval"],                      "frame"),
    ("10 bias quantification", ["bias", "tier2"],                    "frame"),
    ("9  regime discovery",    ["regime_discovery"],                 "frame"),
    ("8  exploratory",         ["inertial_exploratory_analysis"],    "std"),
    ("5/6 dataset survey",     ["dataset_analysis", "dataset_survey",
                                "survey", "selection"],              "exempt"),
]
# frame-independent figures recognisable by filename even if the folder
# name gives no hint (the survey scripts 00..16)
EXEMPT_NAME_HINTS = ["camera", "lidar", "radar", "imu", "gps", "can_bus",
                     "weather", "time_of_day", "road_type", "dss", "qs_",
                     "ds_score", "citation", "country", "availability",
                     "originality", "vehicle_info", "sensors", "cumulative",
                     "dataset_to_year", "task_support", "pie", "avg_sensors"]


def stamp(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def newest(pattern_dir: Path, glob: str) -> float:
    times = [stamp(p) for p in pattern_dir.glob(glob)] if pattern_dir.exists() else []
    return max(times) if times else 0.0


def fmt(t: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t)) if t else "n/a"


def main() -> None:
    anchors = {
        "frame": stamp(FRAME),
        "std": max(newest(STD_DIR, "*.parquet"), stamp(SUMMARIES)),
    }
    print(f"anchor 'frame' (cluster_assignments): {fmt(anchors['frame'])}")
    print(f"anchor 'std'   (newest std_data/summaries): {fmt(anchors['std'])}")
    print("Ch.9/10/12/13 measured against 'frame'; Ch.8 against 'std'; "
          "survey figures exempt.\n")

    rows = []
    for p in sorted(FIG_ROOT.rglob("*.png")):
        rel = str(p.relative_to(FIG_ROOT))
        if ".venv" in rel or "site-packages" in rel:
            continue
        low = rel.lower()
        chapter, anchor = "?  unclassified", "frame"
        for ch, hints, a in CHAPTER_HINTS:
            if any(hh in low for hh in hints):
                chapter, anchor = ch, a
                break
        if chapter == "?  unclassified" and any(
                hh in low for hh in EXEMPT_NAME_HINTS):
            chapter, anchor = "5/6 dataset survey", "exempt"
        t = stamp(p)
        if anchor == "exempt":
            verdict = "EXEMPT"
        else:
            verdict = "STALE" if t < anchors[anchor] else "FRESH"
        rows.append({"chapter": chapter, "file": rel, "mtime": t,
                     "verdict": verdict})

    df = pd.DataFrame(rows)
    summary = (df.pivot_table(index="chapter", columns="verdict",
                              values="file", aggfunc="count", fill_value=0)
               .sort_index())
    print("=== figures by chapter ===")
    print(summary.to_string())

    stale = df[df["verdict"] == "STALE"].sort_values(["chapter", "file"])
    if len(stale):
        by_ch = stale.groupby("chapter")["file"].apply(list)
        print(f"\n=== {len(stale)} STALE figures ===")
        for ch, files in by_ch.items():
            print(f"\n[{ch}] {len(files)} stale:")
            for fp in files[:25]:
                print(f"    {fp}")
            if len(files) > 25:
                print(f"    ... and {len(files) - 25} more")
        print("\nRe-run the producing scripts before quoting these chapters' "
              "numbers in prose.")
    else:
        print("\nNo stale figures under the per-chapter anchors.")

    unc = df[df["chapter"].str.startswith("?")]
    if len(unc):
        print(f"\n{len(unc)} unclassified figures (verify their dependency "
              f"manually):")
        for fp in unc["file"].head(15):
            print(f"    {fp}")

    print("\n=== eval-cache artifacts (anchor: frame) ===")
    for f in sorted(EVAL_DIR.glob("*")):
        t = stamp(f)
        flag = "STALE" if t < anchors["frame"] else "fresh"
        print(f"  {flag:>5}  {f.name}  {fmt(t)}")


if __name__ == "__main__":
    main()