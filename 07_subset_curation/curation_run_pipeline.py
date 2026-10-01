#!/usr/bin/env python3
"""curation_run_pipeline.py — the curation stage end to end, in the protocol order.

One command reproduces the curation results (thesis 4.11, 4.12) from the
frozen inputs: audit -> diagnostics -> build -> evaluate -> gates ->
feasibility -> replicate -> reveal -> render -> dossier -> pca. Every stage keeps its own guards (exact-N canaries,
reserve assertion, replication-before-reveal), so the runner adds ordering
and bookkeeping, never permission.

Two semantics worth stating once:
  * Re-running gates and reveal is RE-APPLICATION of frozen rules that were
    declared before any curation subset was evaluated: deterministic
    reprints, not re-decisions. The original gate declaration and the
    seal-opening remain the recorded events.
  * A full re-run after upstream refreshes (e.g. a refit of the regime
    discovery, thesis 4.7) is the reproduction proof: if any frozen number
    drifted, the re-anchored canaries in curation_eval.py abort loudly.

Stage list (curation_common is a shared module, not a stage):

  audit        curation_stale_audit.py        read-only freshness report
  diagnostics  curation_event_audit.py        reserve + evasive counts
  build        curation_build.py              C01/C02/C03, both tiers
  eval         curation_eval.py               harness + C-rows + canaries
  gates        curation_gates.py              selection rule applied mechanically
  feasibility  curation_feasibility.py        pooling-justification figure
  replicate    curation_replicate.py          R=20 winner replication
  reveal       curation_reveal.py             sealed shelf + ex-confounded
  render       curation_render_winner.py      winner figures (flags below)
  dossier      curation_winner_dossier.py     seven comparative winner
                                              figures from the caches
  pca          curation_winner_pca.py         winner occupancy of the
                                              frozen PC space, 3D, two panels

Usage:
  python curation_run_pipeline.py                       # all stages
  python curation_run_pipeline.py --from replicate      # resume
  python curation_run_pipeline.py --skip audit,reveal   # omit stages
  python curation_run_pipeline.py --frame-pass --dossier   # full render
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

STAGES: list[tuple[str, str, list[str]]] = [
    ("audit",       "curation_stale_audit.py",   []),
    ("diagnostics", "curation_event_audit.py",   []),
    ("build",       "curation_build.py",         []),
    ("eval",        "curation_eval.py",          []),
    ("gates",       "curation_gates.py",         []),
    ("feasibility", "curation_feasibility.py",   []),
    ("replicate",   "curation_replicate.py",     []),
    ("reveal",      "curation_reveal.py",        []),
    ("render",      "curation_render_winner.py", []),   # flags appended below
    ("dossier",     "curation_winner_dossier.py", []),
    ("pca",         "curation_winner_pca.py",     []),
]


def banner(msg: str) -> None:
    print("\n" + "=" * 78 + f"\n  {msg}\n" + "=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="resume", default=None,
                    help="start at this stage name")
    ap.add_argument("--skip", default="",
                    help="comma-separated stage names to skip")
    ap.add_argument("--frame-pass", action="store_true",
                    help="render stage: re-stream frame caches (slow)")
    ap.add_argument("--dossier", action="store_true",
                    help="render stage: full Ch.8-grain dossier")
    args = ap.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    plan = list(STAGES)
    if args.resume:
        names = [s for s, _, _ in plan]
        if args.resume not in names:
            raise SystemExit(f"[resume] unknown stage '{args.resume}'; "
                             f"stages: {names}")
        plan = plan[names.index(args.resume):]

    render_flags = ((["--frame-pass"] if args.frame_pass else [])
                    + (["--dossier"] if args.dossier else []))

    banner("Chapter 13 curation pipeline — protocol order enforced")
    results, t_all = [], time.time()
    for name, script, extra in plan:
        if name in skip:
            print(f"\n---- [{name}] SKIPPED by request")
            results.append((name, script, 0.0, "skip"))
            continue
        cmd = [sys.executable, str(HERE / script)]
        cmd += render_flags if name == "render" else extra
        print(f"\n---- [{name}] {' '.join(Path(c).name if i < 2 else c for i, c in enumerate(cmd))}")
        t0 = time.time()
        r = subprocess.run(cmd, cwd=HERE)
        dt = time.time() - t0
        if r.returncode != 0:
            results.append((name, script, dt, "FAIL"))
            banner("STOPPED — a stage failed; later stages were not run")
            for n, s, d, st in results:
                print(f"  {st:>4}  {n:<12} {s:<28} {d:7.1f}s")
            raise SystemExit(1)
        results.append((name, script, dt, "ok"))

    banner("summary")
    for n, s, d, st in results:
        print(f"  {st:>4}  {n:<12} {s:<28} {d:7.1f}s")
    print(f"\n  total: {time.time() - t_all:.1f}s")
    print("  Register events (D4 execution, seal opening) remain dated "
          "2026-08-20;\n  re-runs are reproductions, not new decisions.")


if __name__ == "__main__":
    main()