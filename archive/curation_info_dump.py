#!/usr/bin/env python3
"""curation_info_dump.py — one paste-back for everything still owed (13.x).

Gathers, read-only, into a single console dump:

  [1] Both curation_manifest.json files, full — the D23 per-regime cap
      disclosure (realised top shares, forced counts, breach flags).
  [2] Replication + run metadata one-liners, for the record.
  [3] Tier-threshold numeric check (replaces the eyeball): the fresh
      pooled ECDF value at every t1/t2/t3 in regime_classification.py's
      own THRESHOLDS, plus the pooled tier shares and the per-dataset
      spread of the High-Dynamic share. Constants are imported from the
      module, never retyped.
  [4] The S7.3 coverage numbers from audit_baseline.json — full blocks
      for a2d2 / 4seasons / hdd and a compact table for all 14 — the
      post-cleaning values the TODO paragraph needs.
  [5] D5 best effort: searches the usual folders for the Dingus / Perez /
      100-Car / SHRP PDFs and, if pypdf is available, extracts the
      passages around "0.6", "0.75", "lateral acceleration",
      "deceleration" and "trigger" so the threshold verification can be
      settled from pasted text. If nothing is found the verification
      stays a manual read — the script gathers, it does not judge.

Sections run independently: one failure prints and the rest continue.
Nothing is written or modified anywhere.

Usage:  python curation_info_dump.py
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np

from curation_common import PROJECT_ROOT, BUDGET_N, EVAL_DIR

SUBSET_DIR = PROJECT_ROOT / "dataset" / "subset_data"
AUDIT_BASELINE = None  # resolved in section 4 (two known locations)
PDF_KEYWORDS = ["dingus", "perez", "100car", "100-car", "shrp",
                "naturalistic"]
PDF_TERMS = ["0.6", "0.75", "lateral acceleration", "deceleration",
             "trigger", "longitudinal"]


def section(title):
    def deco(fn):
        def run():
            print("\n" + "=" * 74 + f"\n[{title}]\n" + "=" * 74)
            try:
                fn()
            except Exception:
                print("SECTION FAILED (continuing):")
                traceback.print_exc()
        return run
    return deco


@section("1  curation manifests — D23 per-regime cap disclosure")
def s1():
    for tier, d in [("budget", SUBSET_DIR / f"tier_n{BUDGET_N}"),
                    ("capability", SUBSET_DIR)]:
        p = d / "curation_manifest.json"
        print(f"\n----- {tier}: {p}")
        if not p.exists():
            print("MISSING — run curation_build.py")
            continue
        print(json.dumps(json.load(open(p)), indent=1))


@section("2  replication + run metadata")
def s2():
    for name in ["curation_replication_meta.json", "run_meta.json"]:
        p = EVAL_DIR / name
        if p.exists():
            print(f"\n----- {name}")
            print(json.dumps(json.load(open(p)), indent=1)[:2000])
        else:
            print(f"{name}: missing")


@section("3  tier thresholds vs fresh distributions (numeric eyeball)")
def s3():
    import sys as _sys
    _sys.path.insert(0, str(PROJECT_ROOT / "bachelor-arbeit"
                            / "04_exploratory_analysis"))
    import regime_classification as rc
    data = rc.load_std_data()
    for col, slug, title in rc.SIGNALS:
        t1, t2, t3 = rc.THRESHOLDS[col]
        pooled = np.concatenate([
            df[col].dropna().to_numpy()
            for df in data.values() if col in df.columns])
        pooled = pooled[np.isfinite(pooled)]
        q = [float((pooled <= t).mean()) for t in (t1, t2, t3)]
        shares = rc.tier_fractions(pooled, t1, t2, t3)
        hi = {}
        for ds, df in data.items():
            if col in df.columns:
                v = df[col].dropna().to_numpy()
                v = v[np.isfinite(v)]
                if len(v):
                    hi[ds] = rc.tier_fractions(v, t1, t2, t3)[-1]
        lo_ds = min(hi, key=hi.get); hi_ds = max(hi, key=hi.get)
        print(f"\n{title}  [{col}]  thresholds = {t1}/{t2}/{t3}")
        print(f"  pooled ECDF at t1/t2/t3: "
              f"{q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f}")
        print(f"  pooled tier shares (NS/LD/MD/HD): "
              + " / ".join(f"{s:.3f}" for s in shares))
        print(f"  High-Dynamic share range: {lo_ds} {hi[lo_ds]:.3f}  ..  "
              f"{hi_ds} {hi[hi_ds]:.3f}")
    print("\nReading rule: thresholds are healthy when the pooled ECDF "
          "values are\nstrictly increasing, none is ~0.000 or ~1.000, and "
          "every tier share is\nnon-negligible. Quote these numbers, not "
          "the eyeball.")


@section("4  S7.3 coverage numbers (audit_baseline.json, post-cleaning)")
def s4():
    candidates = [
        PROJECT_ROOT / "dataset" / "audit_baseline.json",
        PROJECT_ROOT / "bachelor-arbeit" / "04_exploratory_analysis"
        / "audit_baseline.json",
    ]
    p = next((c for c in candidates if c.exists()), None)
    if p is None:
        hits = list(PROJECT_ROOT.glob("**/audit_baseline.json"))
        p = hits[0] if hits else None
    if p is None:
        print("audit_baseline.json not found")
        return
    base = json.load(open(p))
    print(f"source: {p}\n")
    for ds in ["a2d2", "4seasons", "hdd"]:
        if ds in base:
            print(f"----- {ds} (full block, for the S7.3 paragraph)")
            print(json.dumps(base[ds], indent=1))
    keys = ["clips", "frames", "gated_pct", "speed_src", "yaw_rate_src",
            "n_spd_over", "n_acc_over", "mean_speed", "max_kmh"]
    print("\n----- all datasets, compact")
    hdr = "dataset".ljust(11) + " ".join(k.rjust(11) for k in keys)
    print(hdr)
    for ds in sorted(base):
        row = base[ds]
        cells = []
        for k in keys:
            v = row.get(k, "")
            if isinstance(v, float):
                v = "NaN" if v != v else f"{v:.2f}"
            cells.append(str(v)[:11].rjust(11))
        print(ds.ljust(11) + " ".join(cells))


@section("5  D5 best effort — trigger-threshold passages from local PDFs")
def s5():
    roots = [PROJECT_ROOT, PROJECT_ROOT / "bachelor-arbeit",
             PROJECT_ROOT / "literature"]
    found = []
    for r in roots:
        if r.exists():
            for p in r.rglob("*.pdf"):
                if any(k in p.name.lower() for k in PDF_KEYWORDS):
                    found.append(p)
    found = sorted(set(found))[:4]
    if not found:
        print("No candidate PDFs (Dingus / Perez / 100-Car / SHRP) found "
              "under the\nusual folders — D5 stays a manual read of the "
              "papers.")
        return
    try:
        from pypdf import PdfReader
    except ImportError:
        print(f"candidates found: {[p.name for p in found]}")
        print("pypdf not installed — run `pip install pypdf` and re-run "
              "this script\nto extract the threshold passages.")
        return
    for p in found:
        print(f"\n----- {p.name}")
        try:
            reader = PdfReader(str(p))
            shown = 0
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                for line in text.splitlines():
                    low = line.lower()
                    if any(t in low for t in PDF_TERMS) and "g" in low:
                        print(f"  p{i + 1}: {line.strip()[:110]}")
                        shown += 1
                        if shown >= 25:
                            break
                if shown >= 25:
                    print("  ... (capped at 25 lines)")
                    break
            if shown == 0:
                print("  no threshold-like passages matched; read manually.")
        except Exception as e:
            print(f"  extraction failed: {e}")


def main() -> None:
    print("curation_info_dump — read-only; paste the whole output back.")
    for fn in (s1, s2, s3, s4, s5):
        fn()


if __name__ == "__main__":
    main()