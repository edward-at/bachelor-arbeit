#!/usr/bin/env python3
"""
run_pipeline.py — ordered re-run of the Chapter 7 + 8 pipeline
(inertial_exploratory_analysis: standardization, audits, and the full
exploratory-figure set).

The stages below are a DEPENDENCY ORDER, not a preference: every stage reads
artifacts written by an earlier one, so running them out of order silently
produces figures built on stale clips, or a coverage audit that describes
data nobody re-derived. That is the exact failure this runner exists to
prevent. Stage names follow the thesis's own section numbers so a failure
message points straight at the methodology text that explains it.

    0  standardize     pipeline_regression_test.py   synthetic gate — no real
                                                       data needed, run FIRST
                                                       and cheaply, before an
                                                       hours-long real pass
                        standardize_inertial_data.py  dataset/parquet_data/<ds>
                                                       -> dataset/inertial_data/<ds>
                                                       (§7.2 — the standardized,
                                                       43-column, 10 Hz clips)

    1  coverage        coverage.py                    dataset/inertial_data
                        coverage_baseline.py           -> audit_baseline.json
                                                       (§7.3 signal-availability
                                                       audit; this is the ONLY
                                                       check that would have
                                                       caught the NVIDIA
                                                       emptied-directory
                                                       incident — see note below)

    2  compute_std     compute_std.py                 dataset/inertial_data
                                                       -> dataset/std_data/<ds>.parquet
                                                       (§7.4 — the windowed
                                                       variability + feature layer;
                                                       every window-level figure
                                                       below reads only this)

    3  thresholds      regime_classification.py       std_data -> locked tier
                                                       thresholds (t1,t2,t3) +
                                                       §8.7 tier stacked bars.
                                                       ecdf_analysis.py and
                                                       time_series_analysis.py
                                                       both import these
                                                       thresholds, so this must
                                                       run before either.

    4  frame_figs      distributional_analysis.py     §8.1 histograms (84 figs)
       (order-free)    trajectory_fan.py               §8.4 path fans (28 figs)
                        phase_potrait.py                §8.5 phase portraits (28)
                        — all three read ONLY dataset/inertial_data (frame
                        level); none needs std_data or the tier thresholds.

    5  window_figs     ridgeline.py                    §8.2 (3 figures)
       (order-free)    density_band_analysis.py        §8.3 (14 figures)
                        boxplot_comparison.py           §8.6 (3 figures)
                        ecdf_analysis.py                §8.7 ECDF overlays (3) —
                        NEEDS the locked thresholds from stage 3.

    6  time_series      time_series_analysis.py        §8.8 — needs std_data
                                                        (window selection) AND
                                                        inertial_data (the
                                                        traces themselves) AND
                                                        the stage-3 thresholds.

    7  summary          heatmap_summary.py              §8.9 SYNTHESIS — collapses
                                                        every facet above into one
                                                        scorecard; must run last.

Test.py is a scratch file with no declared inputs/outputs in the thesis text
and is deliberately not listed as a pipeline step. tum_colors.py and
regime_classification.py's threshold constants are shared imports, not
separate runnable stages beyond the one explicit step above.

USAGE
    python run_pipeline.py                    # stages 0-7
    python run_pipeline.py --dry-run          # print the plan, run nothing
    python run_pipeline.py --from compute_std # resume at a stage
    python run_pipeline.py --only coverage    # one stage
    python run_pipeline.py --skip frame_figs  # drop a stage
    python run_pipeline.py --keep-going       # do not stop at the first failure

Fail-fast is the default: a stage that fails leaves every later stage unrun,
because a downstream artifact built on a failed upstream one is worse than a
missing one — consistent with "no cosmetic fix to unproven data".
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# canonical constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path("/Users/edward/Downloads")
PARQUET_ROOT = PROJECT_ROOT / "dataset" / "parquet_data"   # raw pipeline input
INERTIAL_ROOT = PROJECT_ROOT / "dataset" / "inertial_data"  # standardized clips
STD_ROOT = PROJECT_ROOT / "dataset" / "std_data"             # windowed layer
FIG_ROOT = PROJECT_ROOT / "bachelor-arbeit" / "inertial_exploratory_analysis" / "figures"
AUDIT_BASELINE = SCRIPT_DIR / "audit_baseline.json"

# The 14-dataset working set (folder names as used throughout the codebase).
DATASETS = [
    "4seasons", "a2d2", "hdd", "boreas_rt", "durlar", "ford", "kitti",
    "kitti360", "nuscene", "nvidia", "oxford", "waymo", "zod", "kitscenes",
]

FRAME_FIGURES = [
    "distributional_analysis.py",
    "trajectory_fan.py",
    "phase_potrait.py",
]
WINDOW_FIGURES = [
    "ridgeline.py",
    "density_band_analysis.py",
    "boxplot_comparison.py",
    "ecdf_analysis.py",
]


def stages() -> list:
    """(stage_name, [(label, argv), ...], [artifact specs]) in run order.

    An artifact spec is either:
      - a Path             -> checked with Path.exists()
      - (dir, glob, min_n) -> checked with len(list(dir.glob(glob))) >= min_n

    LIMITATION (learned 2026-08-29): existence-only checks cannot detect a
    stage whose producer was deleted — the stale file keeps passing. Pair
    every Path artifact with a script that rewrites it unconditionally.
    """
    return [
        ("standardize", [
            ("pipeline_regression_test.py", ["pipeline_regression_test.py"]),
            ("standardize_inertial_data.py", ["standardize_inertial_data.py"]),
        ], [(INERTIAL_ROOT / ds, "*.parquet", 1) for ds in DATASETS]),

        ("coverage", [
            ("coverage.py", ["coverage.py"]),
            # coverage.py stopped writing audit_baseline.json at some point and
            # the existence-only artifact check below could not notice (the
            # stale v4_10hz file kept satisfying it). coverage_baseline.py is
            # the restored producer and rewrites the file unconditionally, so
            # the existence check is honest again.
            ("coverage_baseline.py", ["coverage_baseline.py"]),
        ], [AUDIT_BASELINE]),

        ("compute_std", [
            ("compute_std.py", ["compute_std.py"]),
        ], [STD_ROOT / f"{ds}.parquet" for ds in DATASETS]),

        ("thresholds", [
            ("regime_classification.py", ["regime_classification.py"]),
        ], [(FIG_ROOT / "regime", "*.png", 3)]),

        ("frame_figs", [(f, [f]) for f in FRAME_FIGURES],
         [(FIG_ROOT / "distribution", "*.png", 1),
          (FIG_ROOT / "trajectory_fans", "*.png", 1),
          (FIG_ROOT / "phase_portrait", "*.png", 1)]),

        ("window_figs", [(f, [f]) for f in WINDOW_FIGURES],
         [(FIG_ROOT / "ridgeline", "*.png", 1),
          (FIG_ROOT / "density_band", "*.png", 1),
          (FIG_ROOT / "boxplot", "*.png", 1),
          (FIG_ROOT / "ecdf", "*.png", 1)]),

        ("time_series", [
            ("time_series_analysis.py", ["time_series_analysis.py"]),
        ], [(FIG_ROOT / "time_series", "*.png", 1)]),

        ("summary", [
            ("heatmap_summary.py", ["heatmap_summary.py"]),
        ], [(FIG_ROOT / "summary", "*.png", 1)]),
    ]


DEFAULT_STAGES = ["standardize", "coverage", "compute_std", "thresholds",
                   "frame_figs", "window_figs", "time_series", "summary"]


# ---------------------------------------------------------------------------
def hr(msg: str = "", char: str = "=") -> None:
    print(f"\n{char * 78}")
    if msg:
        print(f"  {msg}")
        print(char * 78)


def run_one(label: str, argv: list, dry: bool) -> tuple:
    """Run one script. Returns (ok, seconds)."""
    script = SCRIPT_DIR / argv[0]
    if not script.exists():
        print(f"  [MISSING] {script} — skipped")
        return False, 0.0
    cmd = [sys.executable, str(script)] + argv[1:]
    print(f"\n  ---- {label}")
    print(f"       {' '.join(cmd[1:])}")
    if dry:
        return True, 0.0
    t0 = time.time()
    res = subprocess.run(cmd, cwd=SCRIPT_DIR)
    dt = time.time() - t0
    ok = res.returncode == 0
    print(f"       {'OK' if ok else f'FAILED (exit {res.returncode})'} "
          f"in {dt:.1f}s")
    return ok, dt


def verify(specs: list, dry: bool) -> bool:
    """Confirm a stage actually wrote what the next stage will read.

    Plain Path specs are checked with .exists(); (dir, glob, min_n) specs are
    checked by counting matches — this catches an EMPTIED directory that
    still exists, which a bare .exists() check would miss (the exact shape
    of the NVIDIA inertial_data incident: the folder was present but empty).
    """
    if dry or not specs:
        return True
    ok = True
    for spec in specs:
        if isinstance(spec, tuple):
            directory, pattern, min_n = spec
            n = len(list(directory.glob(pattern))) if directory.exists() else 0
            if n < min_n:
                print(f"  [ARTIFACT MISSING] {directory}/{pattern} "
                      f"— found {n}, expected >= {min_n}")
                ok = False
        else:
            if not spec.exists():
                print(f"  [ARTIFACT MISSING] {spec}")
                ok = False
    if ok:
        print(f"  [artifacts OK] {len(specs)} check(s) passed")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Ordered re-run of the Chapter 7 + 8 pipeline.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan without running anything")
    ap.add_argument("--from", dest="from_stage", default=None,
                    help="resume at this stage")
    ap.add_argument("--only", default=None, help="run this stage alone")
    ap.add_argument("--skip", action="append", default=[],
                    help="skip a stage (repeatable)")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue after a failing script")
    args = ap.parse_args()

    all_stages = stages()
    names = [s[0] for s in all_stages]

    wanted = DEFAULT_STAGES[:]
    if args.only:
        wanted = [args.only]
    elif args.from_stage:
        if args.from_stage not in names:
            print(f"unknown stage {args.from_stage!r}; known: {names}")
            return 2
        i = names.index(args.from_stage)
        wanted = [n for n in names[i:] if n in wanted or n == args.from_stage]
    wanted = [n for n in wanted if n not in args.skip]

    hr(f"Chapter 7 + 8 pipeline"
       f"{'  [DRY RUN]' if args.dry_run else ''}")
    print(f"  scripts    : {SCRIPT_DIR}")
    print(f"  parquet_in : {PARQUET_ROOT}")
    print(f"  inertial   : {INERTIAL_ROOT}")
    print(f"  std_data   : {STD_ROOT}")
    print(f"  figures    : {FIG_ROOT}")
    print(f"  stages     : {' -> '.join(wanted)}")

    results, failures, total = [], [], 0.0
    for name, steps, artifacts in all_stages:
        if name not in wanted:
            continue
        hr(f"stage: {name}", "-")
        stage_ok = True
        for label, argv in steps:
            ok, dt = run_one(label, argv, args.dry_run)
            total += dt
            results.append((name, label, ok, dt))
            if not ok:
                stage_ok = False
                failures.append(label)
                if not args.keep_going:
                    break
        if stage_ok and not verify(artifacts, args.dry_run):
            stage_ok = False
            failures.append(f"{name}: expected artifacts missing")
        if not stage_ok and not args.keep_going:
            hr("STOPPED — a stage failed; later stages were not run", "!")
            break

    hr("summary")
    for name, label, ok, dt in results:
        print(f"  {'ok  ' if ok else 'FAIL'}  {name:<12} {label:<32} {dt:6.1f}s")
    print(f"\n  total: {total:.1f}s")
    if failures:
        print(f"  failures ({len(failures)}):")
        for f in failures:
            print(f"    - {f}")
        return 1
    if not args.dry_run:
        print("\n  Post-run checks worth doing by hand:")
        print("    * standardize stage: the (dir, glob, min_n>=1) check per")
        print("      dataset is the one that would have caught the NVIDIA")
        print("      inertial_data/ emptied-directory incident — a bare")
        print("      .exists() on the folder would NOT have. If this pipeline")
        print("      ever reports nvidia with 0 files, stop and diagnose")
        print("      before letting compute_std run on a partial corpus.")
        print("    * coverage stage: confirm audit_baseline.json's coverage")
        print("      figures for a2d2 / 4seasons / hdd reflect the POST-")
        print("      cleaning values, not the pre-cleaning ones quoted in the")
        print("      §7.3 TODO paragraph — replace that paragraph once this")
        print("      run confirms the qualifying numbers.")
        print("    * thresholds stage: t1/t2/t3 for speed and yaw were")
        print("      calibrated on PRE-v3 std values (§8.7) — re-validate")
        print("      against this run's ECDFs before treating figures as final.")
        print("    * heatmap_summary.py: verify whether the frame-pass flag")
        print("      (full pass vs. sampled) was set as intended for the")
        print("      largest datasets (NVIDIA, 298k+ clips) — this changes")
        print("      the stationary-frame-share and occupancy-entropy columns.")
    return 0


if __name__ == "__main__":
    sys.exit(main())