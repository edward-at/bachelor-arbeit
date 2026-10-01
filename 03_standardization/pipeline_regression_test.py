"""
pipeline_regression_test.py — synthetic regression test for the
standardization pipeline (standardize_inertial_data.build_standard_frame).

Plants every targeted failure mode into generated clips and asserts the
output behaviour (Section 4.4.6, "Pipeline validation"):

  T1a planted hard-braking event (-5.0 m/s^2, rectangular acceleration
      pulse) survives the full pipeline within the acceptance tolerance;
      the measured peak is printed. (A rectangular pulse has discontinuous edges, so the
      3 Hz zero-phase band-limit produces a slight peak OVERSHOOT, not
      attenuation.)
  T1b planted smooth braking event (raised-cosine acceleration pulse,
      peak -5.0 m/s^2, 2 s) is preserved with minimal attenuation.
  T2  planted single-frame speed spikes (+80 m/s) are removed entirely.
  T3  timestamp gaps (> 0.5 s clock jump) remain NaN on the grid
      (no fabricated bridges).
  T4  NaN-run dropouts (> 0.5 s of missing sensor values while the clock
      keeps ticking) remain NaN on the grid.
  T5  wrapped yaw is artifact-free across +/-pi crossings (all values in
      [-pi, pi], no fabricated intermediate headings).
  T6  deceleration preserves NaN wherever a_long is NaN (never 0-filled).
  T7  derivative columns are edge-masked (first/last EDGE_MASK grid frames).
  T8  a clip shorter than twice the edge window is fully masked.
  T9  position-only input reproduces a known constant speed within tolerance.
  T10 non-monotonic and near-zero-dt frames are dropped and counted in
      processing_meta.

Run:  python3 pipeline_regression_test.py
Exit code 0 and "ALL TESTS PASSED" on success; any assertion failure aborts.
Re-run after every pipeline modification.
"""

import numpy as np

from standardize_inertial_data import (
    build_standard_frame,
    EDGE_MASK,
    GAP_MAX_S,
    GRID_HZ,
)

# ─────────────────────────────────────────────────────────────────────────────
# Test parameters (all top-level constants)
# ─────────────────────────────────────────────────────────────────────────────

SEED            = 42
NATIVE_HZ       = 100          # synthetic native rate (exercises decimation)
CLIP_S          = 60.0         # main clip duration
CRUISE_MPS      = 15.0         # baseline speed
BRAKE_ACC       = -5.0         # planted braking event (m/s^2)
BRAKE_T0        = 30.0         # brake onset (s)
BRAKE_DUR       = 2.0          # brake duration (s)
SPIKE_MPS       = 80.0         # planted single-frame spike (m/s)
SPIKE_TIMES     = (10.0, 45.0) # spike locations (s)
NOISE_STD       = 0.05         # m/s Gaussian speed noise (realistic sensor)
BRAKE_TOL_FRAC  = 0.20         # accept up to 20 % attenuation of the peak
SPIKE_CEIL_MPS  = 25.0         # post-pipeline speed must stay below this
GAP_JUMP_S      = 2.0          # planted timestamp gap (s)
GAP_T0          = 20.0         # gap location (s)
NANRUN_S        = 1.5          # planted NaN dropout length (s)
NANRUN_T0       = 50.0         # dropout location (s)
YAW_RATE_RPS    = 0.5          # rad/s constant turn for the yaw test
POS_SPEED_MPS   = 12.0         # constant speed for the position-only test
POS_SPEED_TOL   = 0.05         # m/s tolerance on recovered speed


def _ts_ns(t_s):
    return (np.asarray(t_s) * 1e9).round().astype(np.int64)


def _grid_time(df):
    return df["t_rel_s"].to_numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Clip A: cruise + brake + spikes + timestamp gap + NaN run  (speed-sourced)
# ─────────────────────────────────────────────────────────────────────────────

def build_clip_a():
    rng = np.random.default_rng(SEED)
    t = np.arange(0.0, CLIP_S, 1.0 / NATIVE_HZ)

    # speed profile: cruise, then constant -5 m/s^2 for BRAKE_DUR, then cruise low
    speed = np.full_like(t, CRUISE_MPS)
    in_brake = (t >= BRAKE_T0) & (t < BRAKE_T0 + BRAKE_DUR)
    speed[in_brake] = CRUISE_MPS + BRAKE_ACC * (t[in_brake] - BRAKE_T0)
    speed[t >= BRAKE_T0 + BRAKE_DUR] = CRUISE_MPS + BRAKE_ACC * BRAKE_DUR
    speed += rng.normal(0.0, NOISE_STD, speed.shape)

    # single-frame spikes
    for ts in SPIKE_TIMES:
        speed[np.argmin(np.abs(t - ts))] = SPIKE_MPS

    # NaN run: sensor silent while clock ticks
    speed[(t >= NANRUN_T0) & (t < NANRUN_T0 + NANRUN_S)] = np.nan

    # timestamp gap: clock jumps by GAP_JUMP_S after GAP_T0
    t_stamped = t.copy()
    t_stamped[t >= GAP_T0] += GAP_JUMP_S

    return _ts_ns(t_stamped), speed


def test_clip_a():
    ts_ns, speed = build_clip_a()
    df, src, meta = build_standard_frame(timestamp_ns=ts_ns, speed_raw=speed)
    assert df is not None, "clip A: pipeline returned None"
    tg = _grid_time(df)
    a_long = df["a_long"].to_numpy()
    spd    = df["speed"].to_numpy()
    dec    = df["deceleration"].to_numpy()

    # T1a — rectangular brake survives (slight overshoot expected)
    in_ev = (tg >= BRAKE_T0 + GAP_JUMP_S) & (tg <= BRAKE_T0 + GAP_JUMP_S + BRAKE_DUR)
    peak = np.nanmin(a_long[in_ev])
    print(f"T1a planted {BRAKE_ACC:+.1f} m/s^2 rectangular brake -> measured "
          f"peak {peak:+.3f} m/s^2 (deviation {(abs(peak)/abs(BRAKE_ACC)-1)*100:+.1f} %)")
    assert peak <= BRAKE_ACC * (1.0 - BRAKE_TOL_FRAC), (
        f"T1a FAIL: brake peak {peak:.2f} attenuated beyond "
        f"{BRAKE_TOL_FRAC*100:.0f} % tolerance")
    assert peak >= BRAKE_ACC * (1.0 + BRAKE_TOL_FRAC), (
        f"T1a FAIL: brake peak {peak:.2f} overshoots beyond "
        f"{BRAKE_TOL_FRAC*100:.0f} % tolerance")

    # T2 — spikes removed
    mx = np.nanmax(spd)
    print(f"T2  planted +{SPIKE_MPS:.0f} m/s spikes -> post-pipeline max speed "
          f"{mx:.2f} m/s")
    assert mx < SPIKE_CEIL_MPS, f"T2 FAIL: spike survived (max {mx:.1f} m/s)"

    # T3 — timestamp gap masked, no fabricated bridge
    in_gap = (tg > GAP_T0 + 0.2) & (tg < GAP_T0 + GAP_JUMP_S - 0.2)
    assert in_gap.any(), "T3: no grid frames inside the planted gap"
    assert np.all(np.isnan(spd[in_gap])), "T3 FAIL: timestamp gap was bridged"
    print(f"T3  {in_gap.sum()} grid frames inside {GAP_JUMP_S:.0f} s timestamp "
          f"gap: all NaN")

    # T4 — NaN run masked
    n0 = NANRUN_T0 + GAP_JUMP_S
    in_run = (tg > n0 + 0.2) & (tg < n0 + NANRUN_S - 0.2)
    assert in_run.any(), "T4: no grid frames inside the planted NaN run"
    assert np.all(np.isnan(spd[in_run])), "T4 FAIL: NaN run was interpolated"
    print(f"T4  {in_run.sum()} grid frames inside {NANRUN_S:.1f} s NaN run: all NaN")

    # T6 — deceleration NaN-preserving
    nan_al = np.isnan(a_long)
    assert nan_al.any(), "T6: expected some NaN a_long frames"
    assert np.all(np.isnan(dec[nan_al])), "T6 FAIL: deceleration 0-filled a NaN"
    valid = ~nan_al
    assert np.allclose(dec[valid], np.maximum(-a_long[valid], 0.0)), \
        "T6 FAIL: deceleration != max(-a_long, 0) on valid frames"
    print(f"T6  deceleration preserves NaN on {nan_al.sum()} masked frames")

    # T7 — edge masking of derivatives
    for col in ("a_long", "jerk_long"):
        v = df[col].to_numpy()
        assert np.all(np.isnan(v[:EDGE_MASK])) and np.all(np.isnan(v[-EDGE_MASK:])), \
            f"T7 FAIL: {col} not edge-masked"
    print(f"T7  first/last {EDGE_MASK} frames of derivative columns are NaN")


# ─────────────────────────────────────────────────────────────────────────────
# Clip A2: smooth (raised-cosine) braking pulse  → minimal attenuation
# ─────────────────────────────────────────────────────────────────────────────

def test_clip_a2():
    rng = np.random.default_rng(SEED)
    t = np.arange(0.0, CLIP_S, 1.0 / NATIVE_HZ)
    acc = np.zeros_like(t)
    in_ev = (t >= BRAKE_T0) & (t < BRAKE_T0 + BRAKE_DUR)
    acc[in_ev] = BRAKE_ACC * 0.5 * (
        1.0 - np.cos(2.0 * np.pi * (t[in_ev] - BRAKE_T0) / BRAKE_DUR))
    speed = CRUISE_MPS + np.cumsum(acc) / NATIVE_HZ
    speed += rng.normal(0.0, NOISE_STD, speed.shape)

    df, _, _ = build_standard_frame(timestamp_ns=_ts_ns(t), speed_raw=speed)
    assert df is not None, "clip A2: pipeline returned None"
    tg = _grid_time(df)
    a_long = df["a_long"].to_numpy()
    m = (tg >= BRAKE_T0 - 0.5) & (tg <= BRAKE_T0 + BRAKE_DUR + 0.5)
    peak = np.nanmin(a_long[m])
    att = 1.0 - abs(peak) / abs(BRAKE_ACC)
    print(f"T1b planted {BRAKE_ACC:+.1f} m/s^2 raised-cosine brake -> measured "
          f"peak {peak:+.3f} m/s^2 (attenuation {att*100:.1f} %)")
    assert abs(att) < BRAKE_TOL_FRAC, (
        f"T1b FAIL: smooth brake peak {peak:.2f} outside "
        f"{BRAKE_TOL_FRAC*100:.0f} % tolerance")


# ─────────────────────────────────────────────────────────────────────────────
# Clip B: constant turn through +/-pi  (yaw-sourced)
# ─────────────────────────────────────────────────────────────────────────────

def test_clip_b():
    t = np.arange(0.0, 20.0, 1.0 / NATIVE_HZ)
    yaw_true = -np.pi + 0.1 + YAW_RATE_RPS * t          # crosses +pi
    yaw_wrapped = np.arctan2(np.sin(yaw_true), np.cos(yaw_true))
    speed = np.full_like(t, CRUISE_MPS)

    df, _, _ = build_standard_frame(
        timestamp_ns=_ts_ns(t), speed_raw=speed, yaw_raw=yaw_wrapped)
    assert df is not None, "clip B: pipeline returned None"

    yaw_g = df["yaw"].to_numpy()
    yr_g  = df["yaw_rate"].to_numpy()

    # T5 — wrapped yaw bounded and consistent with a constant turn
    v = yaw_g[~np.isnan(yaw_g)]
    assert np.all(np.abs(v) <= np.pi + 1e-9), "T5 FAIL: wrapped yaw out of range"
    mid = ~np.isnan(yr_g)
    err = np.nanmax(np.abs(yr_g[mid] - YAW_RATE_RPS))
    assert err < 0.05, f"T5 FAIL: yaw rate corrupted at wrap (max err {err:.3f})"
    print(f"T5  yaw wrap: |yaw| <= pi everywhere, yaw-rate max error "
          f"{err:.4f} rad/s across the +/-pi crossing")


# ─────────────────────────────────────────────────────────────────────────────
# Clip C: too-short clip  → fully masked derivatives
# ─────────────────────────────────────────────────────────────────────────────

def test_clip_c():
    short_s = (2 * EDGE_MASK - 1) / GRID_HZ        # just under 2x edge window
    t = np.arange(0.0, short_s, 1.0 / NATIVE_HZ)
    df, _, _ = build_standard_frame(
        timestamp_ns=_ts_ns(t), speed_raw=np.full_like(t, CRUISE_MPS))
    assert df is not None, "clip C: pipeline returned None"
    for col in ("a_long", "jerk_long", "jerk_lat"):
        assert df[col].isna().all(), f"T8 FAIL: {col} not fully masked on short clip"
    print(f"T8  {short_s:.1f} s clip (< 2 x edge window): derivatives fully NaN")


# ─────────────────────────────────────────────────────────────────────────────
# Clip D: position-only input  → known constant speed
# ─────────────────────────────────────────────────────────────────────────────

def test_clip_d():
    t = np.arange(0.0, 30.0, 1.0 / NATIVE_HZ)
    x = POS_SPEED_MPS * t
    y = np.zeros_like(t)
    df, src, _ = build_standard_frame(timestamp_ns=_ts_ns(t), x=x, y=y)
    assert df is not None, "clip D: pipeline returned None"
    spd = df["speed"].to_numpy()
    tg  = _grid_time(df)
    interior = (tg > 2.0) & (tg < tg[-1] - 2.0)
    err = np.nanmax(np.abs(spd[interior] - POS_SPEED_MPS))
    assert err < POS_SPEED_TOL, \
        f"T9 FAIL: position-derived speed off by {err:.3f} m/s"
    print(f"T9  position-only input: recovered {POS_SPEED_MPS:.0f} m/s within "
          f"{err:.4f} m/s")


# ─────────────────────────────────────────────────────────────────────────────
# Clip E: non-monotonic + duplicate timestamps  → dropped and counted
# ─────────────────────────────────────────────────────────────────────────────

def test_clip_e():
    t = np.arange(0.0, 10.0, 1.0 / NATIVE_HZ)
    ts = _ts_ns(t)
    ts[300] = ts[299]          # duplicate (non-strictly-monotonic)
    ts[500] = ts[498]          # goes backwards
    ts[700] = ts[699] + 1_000  # near-zero dt (1 us << 10 % of median dt)
    df, _, meta = build_standard_frame(
        timestamp_ns=ts, speed_raw=np.full_like(t, CRUISE_MPS))
    assert df is not None, "clip E: pipeline returned None"
    dropped = meta["n_frames_dropped_monotonic"] + meta["n_frames_dropped_dt"]
    assert meta["n_frames_dropped_monotonic"] >= 2, \
        "T10 FAIL: non-monotonic frames not dropped"
    assert meta["n_frames_dropped_dt"] >= 1, \
        "T10 FAIL: near-zero-dt frame not dropped"
    dt = df["dt_s"].to_numpy()[1:]
    assert np.allclose(dt, 1.0 / GRID_HZ), "T10 FAIL: grid dt not uniform"
    print(f"T10 monotonic/dt filters dropped {dropped} planted bad frames; "
          f"grid dt uniform at {1.0/GRID_HZ:.1f} s")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    test_clip_a()
    test_clip_a2()
    test_clip_b()
    test_clip_c()
    test_clip_d()
    test_clip_e()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
