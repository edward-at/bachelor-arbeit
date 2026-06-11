"""
diagnose_durlar_along.py
────────────────────────────────────────────────────────────────────────────
Settles whether DurLAR's elevated sigma(a_long) reflects genuine driving
dynamics or residual GPS-position noise.

Idea: DurLAR's a_long is derived from GPS positions (position → speed →
SG derivative), but the dataset also carries a raw IMU accelerometer (ax),
which is an INDEPENDENT measurement of the same physical quantity. Both are
band-limited to 3 Hz and live on the same 10 Hz grid in the standardized
clips, so they are directly comparable after removing the IMU's slow
gravity/bias drift (rolling-median detrend, 5 s window).

  ratio  = std(a_long) / std(ax_detrended)   per clip
  corr   = corr(a_long, ax_detrended)        per clip

  ratio ~ 1 and corr high  → the GPS-derived signal tracks the IMU:
                             the dynamics are REAL.
  ratio >> 1 and corr low  → the GPS-derived signal contains energy the
                             IMU does not see: residual GPS NOISE.

Usage:
    python diagnose_durlar_along.py [/path/to/inertial_data/durlar]
Default: /Users/edward/Downloads/dataset/inertial_data/durlar
────────────────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLIP_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path("/Users/edward/Downloads/dataset/inertial_data/durlar")

DETREND_S = 5.0     # rolling-median window for IMU gravity/bias removal
GRID_HZ   = 10
MIN_VALID = 200     # require at least 20 s of overlapping valid frames


def analyse_clip(fp: Path) -> dict | None:
    try:
        df = pd.read_parquet(fp, columns=["t_rel_s", "a_long", "ax", "speed"])
    except Exception as e:
        print(f"  [WARN] {fp.name}: {e}")
        return None

    a_gps = df["a_long"].values.astype(float)
    a_imu = df["ax"].values.astype(float)
    speed = df["speed"].values.astype(float)

    # Remove slow gravity/bias drift from the IMU channel
    k = int(DETREND_S * GRID_HZ) | 1
    trend = pd.Series(a_imu).rolling(k, center=True, min_periods=1).median().values
    a_imu_d = a_imu - trend
    # Same detrend on the GPS-derived channel for a fair band comparison
    trend_g = pd.Series(a_gps).rolling(k, center=True, min_periods=1).median().values
    a_gps_d = a_gps - trend_g

    m = np.isfinite(a_gps_d) & np.isfinite(a_imu_d) & np.isfinite(speed) & (speed > 1.0)
    if m.sum() < MIN_VALID:
        return None

    sd_gps = float(np.std(a_gps_d[m]))
    sd_imu = float(np.std(a_imu_d[m]))
    if sd_imu < 1e-4:
        return None
    corr = float(np.corrcoef(a_gps_d[m], a_imu_d[m])[0, 1])

    return {
        "file":    fp.name,
        "n":       int(m.sum()),
        "sd_gps":  sd_gps,
        "sd_imu":  sd_imu,
        "ratio":   sd_gps / sd_imu,
        "corr":    corr,
    }


def main():
    files = sorted(CLIP_DIR.glob("clip_*.parquet"))
    if not files:
        print(f"[ERROR] no clips under {CLIP_DIR}")
        sys.exit(1)
    print(f"Analysing {len(files)} DurLAR clip(s) from {CLIP_DIR}\n")

    results = [r for fp in files if (r := analyse_clip(fp)) is not None]
    if not results:
        print("[ERROR] no usable clips (need a_long, ax, speed with ≥20 s overlap)")
        sys.exit(1)

    print(f"{'file':<24}{'n':>8}{'σ(a_long)':>11}{'σ(ax)':>8}{'ratio':>8}{'corr':>7}")
    for r in results:
        print(f"{r['file']:<24}{r['n']:>8,}{r['sd_gps']:>11.3f}"
              f"{r['sd_imu']:>8.3f}{r['ratio']:>8.2f}{r['corr']:>7.2f}")

    med_ratio = float(np.median([r["ratio"] for r in results]))
    med_corr  = float(np.median([r["corr"]  for r in results]))

    print("\n" + "=" * 60)
    print(f" median ratio σ(a_long)/σ(ax) = {med_ratio:.2f}   median corr = {med_corr:.2f}")
    if med_ratio <= 1.6 and med_corr >= 0.6:
        print(" VERDICT: GPS-derived a_long tracks the independent IMU —")
        print("          DurLAR's elevated dynamics are REAL driving behavior.")
        print(" ACTION : keep DurLAR as-is; report it as a genuinely dynamic")
        print("          (urban) dataset.")
    elif med_ratio >= 2.0 and med_corr < 0.6:
        print(" VERDICT: a_long carries energy the IMU does not see —")
        print("          residual GPS-position NOISE dominates the signal.")
        print(" ACTION : DurLAR a_long/jerk_long stds reflect the measurement")
        print("          chain, not driving. Options: tighten the GPS cleaning")
        print("          threshold, or flag DurLAR longitudinal metrics as")
        print("          measurement-limited in the thesis (do NOT compare its")
        print("          a_long/jerk_long medians against CAN/IMU datasets).")
    else:
        print(" VERDICT: MIXED — partial correlation with inflated amplitude.")
        print("          Real dynamics plus a noise floor. Report DurLAR with a")
        print("          measurement-chain caveat; inspect 2-3 clips manually")
        print("          (overlay a_long vs detrended ax).")
    print("=" * 60)


if __name__ == "__main__":
    main()