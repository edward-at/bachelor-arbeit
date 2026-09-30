#!/usr/bin/env python3
"""
bias_run_pipeline.py: ordered re-run of the bias-quantification pipeline
(thesis Section 4.8, results 5.5).

The stages below are a DEPENDENCY ORDER, not a preference: every stage reads
artifacts written by an earlier one, so running them out of order silently
produces figures that disagree with the tables beside them (or crash on a
missing column). That is the exact failure this runner exists to prevent.

    0  compute     bias_quant.py                 cluster_assignments + std_data
                                                  -> bias_per_dataset, bias_corpus_
                                                     profiles, bias_corpus_scalars,
                                                     bias_sparse_regimes
                   (Tier-1. Everything below reads this stage's output, either
                    directly or via bias_tier2_common.cross_check_tier1().)

    1  tier1_figs  bias_stacked_bar.py            Tier-1 artifacts -> figures
                   bias_regime_grid.py            (order within the stage is
                   bias_share_heatmap.py           free; each reads only what
                   bias_balance_metrics.py         stage 0 wrote, no cross-
                   bias_wasserstein.py             dependency between them)

    2  tier2       bias_diversity.py              cluster_assignments.parquet,
                   bias_volume.py                 via bias_tier2_common.py,
                   bias_bootstrap.py              cross-checked against stage 0
                   bias_gmm_entropy.py            -> bias_diversity, bias_volume,
                   bias_shape_metrics.py             bias_bootstrap_ci, bias_gmm_
                   bias_regime_supply.py             entropy, bias_shape,
                                                      bias_regime_supply
                   (order within the stage is free; each is independent)

    3  fingerprint bias_fingerprint.py            SYNTHESIS. build_bias_metric_
                                                  table() merges stage 0 + every
                                                  stage-2 artifact, so this must
                                                  run strictly last.

bias_tier2_common.py, regime_style.py, and tum_colors.py are shared MODULES,
never entry points. They are imported by the stages above and are never
listed as a runnable step here.

bias_regime_supply.py runs in stage 2 because bias_regime_supply.parquet is
an input to the RQ3 supplier table. Its PNG is a supporting figure.

USAGE
    python bias_run_pipeline.py                    # stages 0-3
    python bias_run_pipeline.py --dry-run          # print the plan, run nothing
    python bias_run_pipeline.py --from tier2       # resume at a stage
    python bias_run_pipeline.py --only tier1_figs  # one stage
    python bias_run_pipeline.py --skip tier2       # drop a stage
    python bias_run_pipeline.py --keep-going       # do not stop at the first failure

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
sys.path.append(str(Path(__file__).resolve().parents[1]))
from thesis_paths import DATA_ROOT, REPO_ROOT  # noqa: E402
ARTIFACT_DIR = DATA_ROOT / "regime_data"
FIG_ROOT = REPO_ROOT / "06_bias_quantification" / "figures"

TIER1_FIGURES = [
    "bias_stacked_bar.py",
    "bias_regime_grid.py",
    "bias_share_heatmap.py",
    "bias_balance_metrics.py",
    "bias_wasserstein.py",
]
TIER2_INSTRUMENTS = [
    "bias_diversity.py",
    "bias_volume.py",
    "bias_bootstrap.py",
    "bias_gmm_entropy.py",
    "bias_shape_metrics.py",
    "bias_regime_supply.py",
]


def stages() -> list:
    """(stage_name, [(label, argv), ...], [expected artifacts]) in run order."""
    return [
        ("compute", [
            ("bias_quant.py", ["bias_quant.py"]),
        ], [ARTIFACT_DIR / "bias_per_dataset.parquet",
            ARTIFACT_DIR / "bias_corpus_profiles.parquet",
            ARTIFACT_DIR / "bias_corpus_scalars.parquet",
            ARTIFACT_DIR / "bias_sparse_regimes.parquet"]),

        ("tier1_figs", [(f, [f]) for f in TIER1_FIGURES],
         [FIG_ROOT / "bias_stacked_bar.png",
          FIG_ROOT / "bias_regime_grid.png",
          FIG_ROOT / "bias_share_heatmap.png",
          FIG_ROOT / "bias_balance_metrics.png",
          FIG_ROOT / "bias_wasserstein.png"]),

        ("tier2", [(f, [f]) for f in TIER2_INSTRUMENTS],
         [ARTIFACT_DIR / "bias_diversity.parquet",
          ARTIFACT_DIR / "bias_volume.parquet",
          ARTIFACT_DIR / "bias_bootstrap_ci.parquet",
          ARTIFACT_DIR / "bias_gmm_entropy.parquet",
          ARTIFACT_DIR / "bias_shape.parquet",
          ARTIFACT_DIR / "bias_regime_supply.parquet"]),

        ("fingerprint", [
            ("bias_fingerprint.py", ["bias_fingerprint.py"]),
        ], [ARTIFACT_DIR / "bias_fingerprint.parquet",
            ARTIFACT_DIR / "bias_metric_agreement.parquet",
            FIG_ROOT / "bias_fingerprint.png"]),
    ]


DEFAULT_STAGES = ["compute", "tier1_figs", "tier2", "fingerprint"]


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
        description="Ordered re-run of the Chapter 10 bias-quantification pipeline.")
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

    hr(f"Chapter 10 bias-quantification pipeline"
       f"{'  [DRY RUN]' if args.dry_run else ''}")
    print(f"  scripts   : {SCRIPT_DIR}")
    print(f"  artifacts : {ARTIFACT_DIR}")
    print(f"  figures   : {FIG_ROOT}")
    print(f"  stages    : {' -> '.join(wanted)}")

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
        print("    * bias_diversity.py / bias_gmm_entropy.py console output:")
        print("      cross_check_tier1() must report agreement to ~1e-12")
        print("      against bias_per_dataset.parquet — a mismatch means the")
        print("      Tier-1 recompute has drifted, not that Tier-2 is wrong")
        print("    * KITScenes colour: confirm DATASET_COLORS['kitscenes'] is")
        print("      the SAME value in bias_tier2_common.py and in every")
        print("      Tier-1 script's local dict before trusting hill_profiles.png")
        print("      or psi_thresholds.png")
        print("    * bias_fingerprint.png: 9 columns present, row order by")
        print("      low_dynamic_share descending, KITScenes rendering black")
        print("    * bias_regime_supply.png / bias_pca_map are NOT thesis")
        print("      figures (DC15) — only bias_regime_supply.parquet feeds")
        print("      the RQ3 supplier table; do not re-add either PNG to the")
        print("      chapter without reopening that decision")
    return 0


if __name__ == "__main__":
    sys.exit(main())