#!/usr/bin/env python3
"""curation_run_pipeline.py — Chapter 13 end to end, in the protocol order.

One command reproduces the chapter from the frozen inputs: audit ->
diagnostics -> build -> evaluate -> gates -> feasibility -> replicate ->
reveal -> render. Every stage keeps its own guards (exact-N canaries,
reserve assertion, replication-before-reveal), so the runner adds ordering
and bookkeeping, never permission.

Two semantics worth stating once:
  * Re-running gates and reveal is RE-APPLICATION of frozen, dated rules —
    deterministic reprints, not re-decisions. The D4 declaration and the
    seal-opening remain the events recorded in the register on 2026-08-20.
  * A full re-run after upstream refreshes (e.g. the Chapter 9 refit) is
    the chapter's reproduction proof: if any frozen number drifted, the
    re-anchored canaries in curation_eval.py abort loudly.

Stage list (curation_common and curation_show_eval are a module and an
interactive tool, not stages):

  audit        curation_stale_audit.py        read-only freshness report
  diagnostics  curation_event_audit.py        reserve + evasive counts
  build        curation_build.py              C01/C02/C03, both tiers
  eval         curation_eval.py               harness + C-rows + canaries
  gates        curation_gates.py              D4 applied mechanically
  feasibility  curation_feasibility.py        pooling-justification figure
  replicate    curation_replicate.py          R=20 winner replication
  reveal       curation_reveal.py             sealed shelf + ex-confounded
  render       curation_render_winner.py      winner figures (flags below)

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