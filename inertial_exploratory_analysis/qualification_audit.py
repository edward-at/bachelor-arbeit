"""
qualification_audit.py
────────────────────────────────────────────────────────────────────────────
Pipeline-version-independent dataset qualification audit.

Background: the original exclusion criterion ("% non-zero values") was a
proxy that only worked under the v1 pipeline. The v3 standardization breaks
it in two ways: (a) zero-phase filtering converts genuine exact zeros
(stationary frames) into numerical epsilons, and (b) interpolation +
band-limiting smears quantized staircase signals (e.g. a coarse heading
channel) into continuous, plausible-LOOKING derivatives. Both raise
"non-zero coverage" without making the data more usable.

This audit replaces the proxy with criteria smoothing cannot fake:

  1. PHYSICAL PLAUSIBILITY — per dataset x signal, the median and p95 of the
     within-window stds (post bounds masking, from std_data) must lie inside
     a physically credible envelope. A smoothed-garbage channel passes
     continuity checks but still produces impossible statistics
     (e.g. median sigma(a_lat) = 38 m/s^2 ~ 4 g sustained).
  2. SAMPLE SIZE — minimum number of valid windows per signal to support
     distributional claims.
  3. RAW QUANTIZATION PROBE (--probe <parquet_dir> <column>) — measures the
     quantization step and staircase fraction of a raw channel, to attribute
     a plausibility failure to its root cause (e.g. coarse heading encoder).

Usage:
    python qualification_audit.py                      # audit all std_data
    python qualification_audit.py --probe /path/to/parquet_data/hdd yaw
────────────────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

STD_DATA_DIR = Path("/Users/edward/Downloads/dataset/std_data")

SIGNALS = ["std_speed_kmh", "std_a_long", "std_a_lat",
           "std_jerk_long", "std_jerk_lat", "std_yaw_rate"]

# Plausibility envelope for 20 s within-window stds of normal road driving.
# (median_max, p95_max). yaw rate in deg/s (converted below).
# Rationale: a 20 s window mixing cruise and a hard brake reaches
# sigma(a_long) ~ 1.5-2 m/s^2; sustained medians far above that, or p95
# beyond the physical bounds themselves, indicate a measurement chain
# producing impossible values, not driving.
ENVELOPE = {
    "std_speed_kmh": (25.0,  45.0),
    "std_a_long":    (2.0,    4.0),
    "std_a_lat":     (2.0,    4.0),
    "std_jerk_long": (4.0,   10.0),
    "std_jerk_lat":  (4.0,   10.0),
    "std_yaw_rate":  (20.0,  50.0),   # deg/s
}

MIN_WINDOWS = 400   # minimum valid windows per signal for distribution claims


def audit_std_data():
    files = sorted(STD_DATA_DIR.glob("*.parquet"))
    if not files:
        print(f"[ERROR] no std_data under {STD_DATA_DIR}")
        sys.exit(1)

    print(f"Auditing {len(files)} dataset(s) from {STD_DATA_DIR}")
    print(f"Criteria: median/p95 inside plausibility envelope; "
          f">= {MIN_WINDOWS} valid windows per signal\n")

    hdr = f"{'dataset':<12}{'signal':<16}{'n_valid':>9}{'median':>9}{'p95':>9}  verdict"
    print(hdr); print("-" * len(hdr))

    summary = {}
    for fp in files:
        ds = fp.stem
        df = pd.read_parquet(fp)
        if "std_yaw_rate" in df.columns:
            df = df.copy()
            df["std_yaw_rate"] = df["std_yaw_rate"] * (180.0 / np.pi)

        flags = []
        for sig in SIGNALS:
            if sig not in df.columns:
                continue
            v = df[sig].dropna().values
            n = len(v)
            med_max, p95_max = ENVELOPE[sig]
            if n == 0:
                verdict = "NO DATA"
                med = p95 = float("nan")
            else:
                med = float(np.median(v))
                p95 = float(np.percentile(v, 95))
                problems = []
                if n < MIN_WINDOWS:
                    problems.append(f"n<{MIN_WINDOWS}")
                if med > med_max:
                    problems.append(f"median>{med_max:g}")
                if p95 > p95_max:
                    problems.append(f"p95>{p95_max:g}")
                verdict = "FLAG: " + ", ".join(problems) if problems else "ok"
            if verdict != "ok":
                flags.append((sig, verdict))
            print(f"{ds:<12}{sig:<16}{n:>9,}{med:>9.2f}{p95:>9.2f}  {verdict}")
        summary[ds] = flags
        print()

    print("=" * 64)
    print(" SUMMARY")
    print("=" * 64)
    for ds, flags in summary.items():
        if not flags:
            print(f" {ds:<12} QUALIFIED — all signals plausible and sufficient")
        else:
            worst = "; ".join(f"{s}: {v}" for s, v in flags)
            print(f" {ds:<12} REVIEW    — {worst}")
    print("\nInterpretation guide:")
    print("  median outside envelope  → measurement chain produces impossible")
    print("    values at the TYPICAL level → exclude that signal family (or the")
    print("    dataset, if core signals are affected); attribute the root cause")
    print("    with the --probe mode before writing the exclusion reason.")
    print("  only p95 flagged         → heavy tail; inspect top windows via the")
    print("    (clip_id, window_start_s) back-reference before deciding.")
    print("  only n flagged           → dataset is usable but too small for")
    print("    distributional claims; exclusion reason = sample size.")


def probe_quantization(raw_dir: Path, column: str, max_files: int = 5):
    """Measure quantization of a raw channel: step size and staircase fraction."""
    files = sorted(raw_dir.rglob("*.parquet"))[:max_files]
    if not files:
        print(f"[ERROR] no parquet files under {raw_dir}")
        sys.exit(1)
    print(f"Probing raw column '{column}' in {len(files)} file(s) from {raw_dir}\n")
    print(f"{'file':<28}{'n':>10}{'uniq%':>8}{'step':>12}{'flat%':>8}")

    steps, flats = [], []
    for fp in files:
        try:
            df = pd.read_parquet(fp, columns=[column])
        except Exception as e:
            print(f"  [WARN] {fp.name}: {e} (check column name)")
            continue
        x = df[column].dropna().values.astype(float)
        if len(x) < 100:
            continue
        d = np.diff(x)
        nz = np.abs(d[d != 0])
        uniq_frac = len(np.unique(np.round(x, 12))) / len(x)
        step = float(np.median(nz)) if len(nz) else float("nan")
        flat = float(np.mean(d == 0))            # consecutive-identical fraction
        steps.append(step); flats.append(flat)
        print(f"{fp.name:<28}{len(x):>10,}{100*uniq_frac:>7.1f}%{step:>12.6f}{100*flat:>7.1f}%")

    if steps:
        print(f"\n median quantization step = {np.median(steps):.6f}")
        print(f" median flat (zero-diff) fraction = {np.median(flats):.1%}")
        print("\n Reading: a large step combined with a high flat fraction means the")
        print(" channel is a quantized staircase. Its time derivative at the native")
        print(" rate is zero between steps and impulsive at steps; band-limiting")
        print(" converts this into smooth but FICTITIOUS dynamics. Signals derived")
        print(" from this channel should be excluded with quantization as the")
        print(" documented root cause.")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--probe":
        probe_quantization(Path(sys.argv[2]), sys.argv[3])
    else:
        audit_std_data()