#!/usr/bin/env python3
"""
run_regime_pipeline.py — ordered re-run of the Chapter 9 (thesis 4.6) pipeline.

The stages below are a DEPENDENCY ORDER, not a preference: every stage reads
parquet artifacts written by an earlier one, so running them out of order
silently produces figures that disagree with the tables beside them. k-means is
stochastic and its regime indices are arbitrary, so a figure drawn against a
stale partition can disagree with its neighbour about which regime is which.
That is the exact failure this runner exists to prevent.

    0  gate       feature_gate.py        std_data -> feature_gate_vector
                  Screens the six candidate features (Pearson / Spearman / VIF)
                  and LOCKS the clustered vector. Everything downstream reads
                  that vector, so this runs first or not at all.

    1  cluster    clustering.py          locked vector -> cluster_assignments,
                                            cluster_profile, kselect_curves
                  The one stochastic fit. Run ONCE; every later stage reads its
                  saved partition and never re-fits. k is selected here and read
                  from kselect_curves by every stage-2 script.

    2  models     cap_sensitivity.py     robustness of the partition to the cap
                  validity.py            internal indices + bootstrap + consensus
                  gmm.py                 soft/elliptical cross-check
                  hmm.py                 temporal structure (SLOW: frame layer)
                  Order inside this stage is free — all four read stage 1 and
                  none reads another. hmm.py is listed last only because it
                  walks the frame layer and takes by far the longest.

    3  figures    every thin figure reader (order within the stage is free;
                  each reads only the caches written above)

NOT RUN, by design:
    tum_colors.py    palette module — imported, never executed
    regime_style.py  regime label/colour single source of truth — imported
    Test.py          scratch/verification file

Label consistency note: the figures take regime names from regime_style.py at
render time, so changing a label there and re-running stage 3 alone is enough —
no recompute needed. Any figure that hardcodes its own label list will silently
disagree; keep them importing regime_label/regime_color.

USAGE
    python run_regime_pipeline.py                  # all stages
    python run_regime_pipeline.py --dry-run        # print the plan, run nothing
    python run_regime_pipeline.py --from models    # resume at a stage
    python run_regime_pipeline.py --only figures   # one stage (after a label edit)
    python run_regime_pipeline.py --skip models    # drop a stage
    python run_regime_pipeline.py --keep-going     # do not stop at first failure

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
STD_DATA_DIR = PROJECT_ROOT / "dataset" / "std_data"
ARTIFACT_DIR = PROJECT_ROOT / "dataset" / "regime_data"
FIGURE_DIR = PROJECT_ROOT / "bachelor-arbeit" / "regime_discovery" / "figures"

# Modules that are imported, never executed, plus scratch files.
NEVER_RUN = ["tum_colors.py", "regime_style.py", "Test.py"]

# Figure scripts grouped by the cache they read. A missing FIGURE script is a
# skip (some are optional/renamed); a missing COMPUTE script is a failure.
FIGURES_MODEL_SELECTION = [
    "k_select_curve.py",              # kselect_curves
]
FIGURES_PARTITION = [
    "pca_scatter_preclustering.py",   # cluster_assignments (pc1, pc2)
    "pca_scatter_preclustering_3D.py",
    "pca_scatter_labeled.py",         # cluster_assignments (pc1, pc2, regime)
    "pca_scatter_labeled_3D.py",      # needs pc3
    "pca_scatter_3D.py",              # earlier name of the labelled 3-D view
    "regime_heatmap.py",              # cluster_profile
    "regime_heatmap_sweep.py",
]
FIGURES_VALIDITY = [
    "validity_indices.py",            # validity_indices + kselect_curves
    "regime_stability.py",            # bootstrap_jaccard
    "consensus_matrix.py",            # consensus_matrix + consensus_rows
    "reseed_stability.py",
    "resample_stability.py",
]
FIGURES_GMM = [
    "gmm_selection.py",               # gmm_selection
    "membership_entropy.py",          # gmm_assignments
    "covariance_overlay.py",          # gmm_components_2d + gmm_assignments
    "covariance_contours.py",         # later name of the same figure
    "gmm_kmeans_correspondence.py",
]
FIGURES_HMM = [
    "transition_matrix.py",           # hmm_transitions
    "state_duration.py",              # hmm_durations
    "state_timeline.py",              # hmm_states + hmm_transitions
]

ALL_FIGURES = (FIGURES_MODEL_SELECTION + FIGURES_PARTITION + FIGURES_VALIDITY
               + FIGURES_GMM + FIGURES_HMM)


def stages() -> list:
    """(stage, [(label, argv, required)], [expected artifacts]) in run order."""
    A = ARTIFACT_DIR
    return [
        ("gate", [
            ("feature_gate.py  (locks the clustered vector)",
             ["feature_gate.py"], True),
        ], [A / "feature_gate_vector.parquet",
            A / "feature_gate_stats.parquet"]),

        ("cluster", [
            ("clustering.py  (the one stochastic fit — run once)",
             ["clustering.py"], True),
        ], [A / "cluster_assignments.parquet",
            A / "cluster_profile.parquet",
            A / "kselect_curves.parquet"]),

        ("models", [
            ("cap_sensitivity.py  (cap robustness)", ["cap_sensitivity.py"], True),
            ("validity.py  (indices + bootstrap + consensus)", ["validity.py"], True),
            ("gmm.py  (probabilistic cross-check)", ["gmm.py"], True),
            ("hmm.py  (temporal structure — SLOW, walks the frame layer)",
             ["hmm.py"], True),
        ], [A / "cap_sensitivity_caps.parquet",
            A / "cap_sensitivity_drift.parquet",
            A / "validity_indices.parquet",
            A / "bootstrap_jaccard.parquet",
            A / "consensus_matrix.parquet",
            A / "consensus_rows.parquet",
            A / "gmm_selection.parquet",
            A / "gmm_assignments.parquet",
            A / "gmm_components_2d.parquet",
            A / "hmm_transitions.parquet",
            A / "hmm_durations.parquet",
            A / "hmm_states.parquet"]),

        ("figures", [(f, [f], False) for f in ALL_FIGURES], []),
    ]


DEFAULT_STAGES = ["gate", "cluster", "models", "figures"]


# ---------------------------------------------------------------------------
def hr(msg: str = "", char: str = "=") -> None:
    print(f"\n{char * 78}")
    if msg:
        print(f"  {msg}")
        print(char * 78)


def run_one(label: str, argv: list, required: bool, dry: bool) -> tuple:
    """Run one script. Returns (ok, seconds, skipped)."""
    script = SCRIPT_DIR / argv[0]
    if not script.exists():
        if required:
            print(f"\n  ---- {label}")
            print(f"       [MISSING — REQUIRED] {script}")
            return False, 0.0, False
        print(f"  [skip] {argv[0]} not present")
        return True, 0.0, True
    cmd = [sys.executable, str(script)] + argv[1:]
    print(f"\n  ---- {label}")
    print(f"       {' '.join(cmd[1:])}")
    if dry:
        return True, 0.0, False
    t0 = time.time()
    res = subprocess.run(cmd, cwd=SCRIPT_DIR)
    dt = time.time() - t0
    ok = res.returncode == 0
    print(f"       {'OK' if ok else f'FAILED (exit {res.returncode})'} in {dt:.1f}s")
    return ok, dt, False


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


def preflight() -> bool:
    """Check the input layers exist before burning time on a doomed run."""
    ok = True
    if not STD_DATA_DIR.exists():
        print(f"  [ERROR] std_data not found: {STD_DATA_DIR}")
        ok = False
    else:
        n = len(list(STD_DATA_DIR.glob("*.parquet")))
        print(f"  std_data  : {n} dataset parquet(s)")
        if n == 0:
            print("  [ERROR] std_data contains no parquet files")
            ok = False
    stray = [f for f in NEVER_RUN if (SCRIPT_DIR / f).exists()]
    if stray:
        print(f"  not run   : {', '.join(stray)}  (imported modules / scratch)")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Ordered re-run of the Chapter 9 regime-discovery pipeline.")
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
        wanted = names[i:]
    wanted = [n for n in wanted if n not in args.skip]

    hr(f"Chapter 9 regime-discovery pipeline"
       f"{'  [DRY RUN]' if args.dry_run else ''}")
    print(f"  scripts   : {SCRIPT_DIR}")
    print(f"  artifacts : {ARTIFACT_DIR}")
    print(f"  figures   : {FIGURE_DIR}")
    print(f"  stages    : {' -> '.join(wanted)}")
    if not preflight() and not args.dry_run:
        return 2

    results, failures, total = [], [], 0.0
    for name, steps, artifacts in all_stages:
        if name not in wanted:
            continue
        hr(f"stage: {name}", "-")
        stage_ok = True
        for label, argv, required in steps:
            ok, dt, skipped = run_one(label, argv, required, args.dry_run)
            total += dt
            if not skipped:
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
        print(f"  {'ok  ' if ok else 'FAIL'}  {name:<9} {label:<56} {dt:6.1f}s")
    print(f"\n  total: {total:.1f}s")
    if failures:
        print(f"  failures ({len(failures)}):")
        for f in failures:
            print(f"    - {f}")
        return 1
    if not args.dry_run:
        print("\n  Post-run checks worth doing by hand:")
        print("    * clustering log: 'Selected k = 5 (max silhouette)' and the")
        print("      PCA line 'cumulative % [38.2 59.8 78.7]' — a change here")
        print("      means the cited variance figures in 4.6.3 need updating")
        print("    * validity log: DB argmin and CH argmax should both be k = 5;")
        print("      the gap argmax is expected to over-select (reported, not fixed)")
        print("    * gmm log: BIC keeps descending to the range edge — expected,")
        print("      the count rests on the separation criteria, not on BIC")
        print("    * hmm log: EM delta decaying to ~0 and every emission mean")
        print("      physical — a positive log-likelihood means variance collapse")
        print("    * grep -rn 'Calm Driving' *.py  — must return nothing; every")
        print("      figure takes its labels from regime_style.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())