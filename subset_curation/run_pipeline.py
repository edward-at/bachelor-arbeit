#!/usr/bin/env python3
"""
run_pipeline.py — ordered re-run of the Chapter 11 and Chapter 12 pipeline.

The stages below are a DEPENDENCY ORDER, not a preference: every stage reads
artifacts written by an earlier one, so running them out of order silently
produces figures that disagree with the tables beside them. That is the exact
failure this runner exists to prevent.

    0  summaries   window_summary.py            frame layer  -> window_summaries
                   (DEFAULT and idempotent: per dataset it compares the cached
                    partial key set against CURRENT std_data and rebuilds only
                    on mismatch. Near-free when the cache is current,
                    self-healing when std_data was regenerated.)

    1  sampling    run_sampling.py              capability tier (default N)
                   run_sampling.py --n BUDGET_N budget tier (headline)
                                                -> M01..M11 subsets, supply_table,
                                                   allocation_table, manifest

    2  harness     subset_eval_harness.py       subsets -> master tables,
                                                   metrics_long, dist_cache,
                                                   tautology_map, srs_envelope

    3  framepass   subset_frame_pass.py         frame layer -> frame_hist_fine,
                                                   frame_stats, frame_portraits,
                                                   fan_frames, fan_index
                                                (12-bit mask: frame + M01..M11)

    4  figures     every subset_*.py figure script (order within the stage is
                   free; each reads only caches written above)

Method discovery is automatic downstream: nothing in stage 4 carries a method
list, so M11 appears in every figure as soon as stage 2 has written its row.
Skip stage 2 and the figures render ten methods without erroring; skip stage 3
and the frame-grain families (histograms, phase portraits, fans, timeseries)
stay at eleven groups.

USAGE
    python run_pipeline.py                    # stages 1-4
    python run_pipeline.py --dry-run          # print the plan, run nothing
    python run_pipeline.py --from harness     # resume at a stage
    python run_pipeline.py --only figures     # one stage
    python run_pipeline.py --skip framepass   # drop a stage
    python run_pipeline.py --budget-n 71390   # override the budget-tier size
    python run_pipeline.py --keep-going       # do not stop at the first failure

Fail-fast is the default: a stage that fails leaves every later stage unrun,
because a downstream artifact built on a failed upstream one is worse than a
missing one.
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
SUBSET_DIR = PROJECT_ROOT / "dataset" / "subset_data"
EVAL_DIR = SUBSET_DIR / "eval"

# Budget-tier size = one rarest-regime supply (Section 11.3). Must match
# BUDGET_N in subset_eval_common.py and subset_frame_pass.py, and the tier
# directory run_sampling.py derives from --n.
BUDGET_N = 71_390

# Figure scripts, grouped by which cache they read. Order inside a group is
# free; the groups are listed eval-first only so a broken eval cache surfaces
# before the slow frame-grain figures start.
FIGURES_EVAL = [
    "subset_stacked_bar.py",
    "subset_regime_grid.py",
    "subset_boxplot.py",
    "subset_ridgeline.py",
    "subset_wasserstein_heatmap.py",
    "subset_source_composition.py",
    "subset_event_retention.py",
    "subset_bias_axes.py",
    "subset_mean_shift.py",
    "subset_ecdf_grid.py",
    "subset_density_grid.py",
    "subset_pca_grid.py",
    "subset_fingerprint.py",          # last: assembles every displayed scalar
]
FIGURES_FRAME = [
    "subset_histogram.py",
    "subset_phase_potrait.py",
    "subset_fans.py",
    "subset_time_series.py",
]


def stages(budget_n: int) -> list:
    """(stage_name, [(label, argv), ...], [expected artifacts]) in run order."""
    return [
        ("summaries", [
            ("window_summary.py", ["window_summary.py"]),
        ], [SUBSET_DIR / "window_summaries.parquet"]),

        ("sampling", [
            ("run_sampling.py (capability tier)", ["run_sampling.py"]),
            (f"run_sampling.py (budget tier, N = {budget_n:,})",
             ["run_sampling.py", "--n", str(budget_n)]),
        ], [SUBSET_DIR / "manifest.json",
            SUBSET_DIR / f"tier_n{budget_n}" / "manifest.json",
            SUBSET_DIR / f"tier_n{budget_n}" / "M11_neyman_allocation.parquet"]),

        ("harness", [
            ("subset_eval_harness.py", ["subset_eval_harness.py"]),
        ], [EVAL_DIR / "master_table_budget.csv",
            EVAL_DIR / "metrics_long.parquet",
            EVAL_DIR / "dist_cache_budget.parquet"]),

        ("framepass", [
            ("subset_frame_pass.py", ["subset_frame_pass.py"]),
        ], [EVAL_DIR / "frame_hist_fine.parquet",
            EVAL_DIR / "frame_portraits.npz",
            EVAL_DIR / "fan_frames.parquet"]),

        ("figures", [(f, [f]) for f in FIGURES_EVAL + FIGURES_FRAME], []),
    ]


DEFAULT_STAGES = ["summaries", "sampling", "harness", "framepass", "figures"]


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


def verify(paths: list, dry: bool) -> bool:
    """Confirm a stage actually wrote what the next stage will read."""
    if dry or not paths:
        return True
    missing = [p for p in paths if not p.exists()]
    for p in missing:
        print(f"  [ARTIFACT MISSING] {p}")
    if not missing:
        print(f"  [artifacts OK] {len(paths)} expected file(s) present")
    return not missing


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Ordered re-run of the Chapter 11 + 12 pipeline.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan without running anything")
    ap.add_argument("--from", dest="from_stage", default=None,
                    help="resume at this stage")
    ap.add_argument("--only", default=None, help="run this stage alone")
    ap.add_argument("--skip", action="append", default=[],
                    help="skip a stage (repeatable)")
    ap.add_argument("--budget-n", type=int, default=BUDGET_N,
                    help=f"budget-tier size (default {BUDGET_N})")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue after a failing script")
    args = ap.parse_args()

    all_stages = stages(args.budget_n)
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

    hr(f"Chapter 11 + 12 pipeline — budget tier N = {args.budget_n:,}"
       f"{'  [DRY RUN]' if args.dry_run else ''}")
    print(f"  scripts   : {SCRIPT_DIR}")
    print(f"  artifacts : {SUBSET_DIR}")
    print(f"  stages    : {' -> '.join(wanted)}")
    if args.budget_n != BUDGET_N:
        print(f"  NOTE: --budget-n differs from this file's BUDGET_N "
              f"({BUDGET_N:,}). Keep subset_eval_common.py and "
              f"subset_frame_pass.py in step, or the harness will look in the "
              f"wrong tier directory.")

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
        print(f"  {'ok  ' if ok else 'FAIL'}  {name:<10} {label:<48} {dt:6.1f}s")
    print(f"\n  total: {total:.1f}s")
    if failures:
        print(f"  failures ({len(failures)}):")
        for f in failures:
            print(f"    - {f}")
        return 1
    if not args.dry_run:
        print("\n  Post-run checks worth doing by hand:")
        print("    * run_sampling log: 'fidelity guard set: ... (bias_quant "
              "(imported))' — a local fallback means the import path is wrong")
        print("    * harness canaries: frame_n 402,421 / guarded_n 401,230 / "
              "regime supplies — a mismatch means the recomputed constants "
              "need correcting, not the pipeline")
        print("    * harness warning about the decel unit anchor: refresh "
              "EXPECTED['trigger_guarded'] and ['predicate_union'] in "
              "subset_eval_harness.py from the new manifest.json")
        print("    * M11 present in master_table_budget.csv and in every "
              "figure that lists methods")
    return 0


if __name__ == "__main__":
    sys.exit(main())