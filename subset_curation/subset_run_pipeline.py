#!/usr/bin/env python3
"""subset_run_pipeline.py — run every Chapter 12 subset_* figure script in order.

Figure scripts only, by design. The Chapter 12 note fixes the contract:
figures read the eval caches and recompute nothing, so this runner renders
and never evaluates. Cache regeneration is deliberately NOT here —
subset_eval_harness.py run standalone would fail the stale frame canaries
of the local copy AND drop the C-rows from the caches; the one correct
cache path is curation_eval.py (canaries re-anchored, C-rows discovered).
This runner verifies the caches exist and are newer than the frame, then
renders.

Excluded on purpose:
  subset_eval_common.py           imported module, not a stage
  subset_eval_harness.py          superseded by curation_eval.py (above)
  run_sampling.py                 Chapter 11 selection, compute-once
  window_summary.py               Section 11.4 summary pass, compute-once
  inspect_clip_mapping.py         diagnostic scratch

Note on C-rows: the frame caches now carry 13 groups, so the frame-grain
dossier scripts (histogram / phase / fans / time series) will also emit
C01 pages into figures/subset_eval/. Chapter 12's comparative figures are
unaffected (they iterate the M-slate constant); delete or ignore the C01
singles in the subset_eval dossier folders — the canonical C01 dossier
lives in figures/curation/.

Usage:
  python subset_run_pipeline.py                 # render everything
  python subset_run_pipeline.py --frame-pass    # re-stream frame caches first
  python subset_run_pipeline.py --from phase    # resume at a script (substring)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from curation_common import EVAL_DIR, PROJECT_ROOT  # noqa: E402

FRAME = PROJECT_ROOT / "dataset" / "regime_data" / "cluster_assignments.parquet"

STAGES: list[tuple[str, list[str]]] = [
    ("cache figures", [
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
        "subset_fingerprint.py",
    ]),
    ("frame-grain dossier", [
        "subset_histogram.py",
        "subset_phase_potrait.py",
        "subset_fans.py",
        "subset_time_series.py",
    ]),
    ("finalist overlays", [
        "subset_ecdf_finalist.py",
    ]),
]
REQUIRED_CACHES = ["metrics_long.parquet", "master_table_budget.csv",
                   "srs_envelope_budget.parquet", "frame_hist_fine.parquet",
                   "frame_portraits.npz", "fan_frames.parquet"]


def banner(msg: str) -> None:
    print("\n" + "=" * 78 + f"\n  {msg}\n" + "=" * 78)


def cache_guard() -> None:
    t_frame = FRAME.stat().st_mtime if FRAME.exists() else 0.0
    missing, stale = [], []
    for name in REQUIRED_CACHES:
        p = EVAL_DIR / name
        if not p.exists():
            missing.append(name)
        elif p.stat().st_mtime < t_frame:
            stale.append(name)
    if missing or stale:
        raise SystemExit(
            f"[guard] eval caches not ready (missing={missing}, "
            f"stale={stale}).\nRegenerate with:  python curation_eval.py"
            f"   (NOT subset_eval_harness.py — stale canaries, no C-rows)"
            + ("\nand:  python curation_render_winner.py --frame-pass"
               if any('frame' in s or 'fan' in s
                      for s in missing + stale) else ""))
    print(f"[guard] {len(REQUIRED_CACHES)} caches present and newer than "
          f"the frame — rendering only.")


def run(script: str) -> float:
    t0 = time.time()
    r = subprocess.run([sys.executable, str(HERE / script)], cwd=HERE)
    dt = time.time() - t0
    if r.returncode != 0:
        raise SystemExit(f"\nFAIL  {script}  (exit {r.returncode}) after "
                         f"{dt:.1f}s — later scripts not run.")
    return dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-pass", action="store_true",
                    help="re-stream the frame layer first (expensive)")
    ap.add_argument("--from", dest="resume", default=None,
                    help="skip until the first script containing this "
                         "substring")
    args = ap.parse_args()

    banner("Chapter 12 subset figure pipeline")
    cache_guard()

    plan = [(st, s) for st, scripts in STAGES for s in scripts]
    if args.frame_pass:
        plan.insert(0, ("frame pass", "subset_frame_pass.py"))
    if args.resume:
        idx = next((i for i, (_, s) in enumerate(plan)
                    if args.resume in s), None)
        if idx is None:
            raise SystemExit(f"[resume] no script matches '{args.resume}'")
        plan = plan[idx:]

    results = []
    for stage, script in plan:
        print(f"\n---- [{stage}] {script}")
        results.append((stage, script, run(script)))

    banner("summary")
    for stage, script, dt in results:
        print(f"  ok  {stage:<20} {script:<32} {dt:7.1f}s")
    print(f"\n  total: {sum(r[2] for r in results):.1f}s")
    print("  reminder: C01 singles in the subset_eval dossier folders are "
          "surplus — canonical copies live in figures/curation/.")


if __name__ == "__main__":
    main()