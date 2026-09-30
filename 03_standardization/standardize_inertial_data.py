"""
standardize_inertial_data.py: Standardize Ego-Motion / Inertial Data
====================================================================
Bachelor Thesis: "Identifying Dynamic Biases in Large-Scale Public Datasets
via Distributional Analysis"

Input  : parquet_data/<dataset>/<any_subfolder>/**/*.parquet
Output : inertial_data/<dataset>/clip_0001.parquet
                                  clip_0002.parquet  ...

─── PIPELINE (per clip) ────────────────────────────────────────────────────
  1. Timestamp cleanup  — monotonic filter + Bucket B near-zero-dt drop
  2. GPS cleaning       — DurLAR/NVIDIA sustained-offset interpolation
  3. Velocity derivation— from LOW-PASSED positions (central differences)
                          when no raw velocity exists
  4. Hampel filter      — point-outlier removal on speed (preserves tails)
  5. Gap policy         — interpolate gaps ≤ 0.5 s; mask longer gaps.
                          BOTH gap kinds masked: timestamp gaps (dt > 0.5 s)
                          AND NaN-runs (sensor dropout with ticking clock)
  6. Anti-alias LP      — zero-phase 4th-order Butterworth at 3 Hz (native rate)
  7. Resample → 10 Hz   — common uniform grid; masked gaps stay NaN in ALL
                          level signals (no fabricated linear bridges)
  8. SG derivatives     — savgol deriv=1/2 per contiguous valid segment
  9. Derivative masking — NaN inside dilated gap regions + clip edge frames
 10. Curvature          — speed_min raised to 2.0 m/s

─── DESIGN NOTES ───────────────────────────────────────────────────────────
  • LOADERS registers one loader per dataset folder in parquet_data.
  • Gap masking is applied to ALL gridded level signals, not only the
    derivative columns. Plain np.interp would fabricate linear-bridge values
    for speed/position/yaw inside every long timestamp gap.
  • NaN-run gaps (values missing while timestamps keep ticking) are detected
    per channel and masked, so long sensor dropouts are not bridged linearly.
  • The wrapped `yaw` column is re-wrapped from the interpolated unwrapped yaw
    (arctan2(sin, cos)). Interpolating wrapped yaw across ±π jumps would
    produce impossible mid-jump headings.
  • Position-derived velocity is computed from LOW-PASSED positions via
    central differences (np.gradient), not raw backward differences.
    Matters most for NVIDIA (position-only, 60M rows).
  • All windows are specified in seconds. No frame-based windows are used.
  • If a clip is shorter than 2× the edge window, its derivative columns are
    fully masked (NaN) rather than keeping filter edge artifacts.
  • The position-derived velocity fallback on the grid is gap/edge-masked
    like every other derivative.

─── COLUMN ORDER (43 columns) ─────────────────────────────────────────────
  timestamp_ns, timestamp_s, t_rel_ns, t_rel_s, dt_ns, dt_s,
  x, y, z, lat, lon,
  x_rel, y_rel, x_aligned, y_aligned, heading_initial,
  vx, vy, vz, speed, speed_kmh,
  ax, ay, az, a_long, a_lat, deceleration, acc_mag,
  jerk_long, jerk_lat, jerk_mag, abs_jerk_long, abs_jerk_mag,
  roll, pitch, yaw, yaw_unwrapped, yaw_rate,
  heading_change, abs_heading_change,
  curvature, abs_curvature, curvature_valid
"""

import json
import math
import os
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import butter, savgol_filter, sosfiltfilt
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from thesis_paths import DATA_ROOT, REPO_ROOT  # noqa: E402

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL PARAMETERS  (all time-based values in seconds — no frame-based windows)
# ─────────────────────────────────────────────────────────────────────────────

GRID_HZ         = 10        # common output sampling rate
DT_GRID         = 0.1       # s — 1 / GRID_HZ
F_CUT_HZ        = 3.0       # Butterworth low-pass cutoff (vehicle dynamics < 3 Hz)
BUTTER_ORDER    = 4         # zero-phase via sosfiltfilt → effective order 8
HAMPEL_WINDOW_S = 0.5       # s — Hampel filter window in seconds
HAMPEL_T0       = 3.0       # MAD multiplier threshold
GAP_MAX_S       = 0.5       # s — gaps longer than this are masked, not interpolated
SG_POLY         = 3         # Savitzky-Golay polynomial order
W_D1            = 7         # frames @ 10 Hz, 1st-derivative window (0.7 s)
W_D2            = 11        # frames @ 10 Hz, 2nd-derivative window (1.1 s)
CURV_SPEED_MIN  = 2.0       # m/s — minimum speed for curvature validity
DT_MIN_FRAC     = 0.1       # Bucket B: drop dt < 10% of median dt
EDGE_MASK       = math.ceil(W_D2 / 2)        # frames NaN'd at clip edges (derivs)
GAP_DILATION    = math.ceil(W_D2 / 2)        # frames NaN'd around gaps (derivs)
GPS_SPEED_MAX   = 50.0      # m/s — impossible-speed threshold for GPS cleaning

# ── PHYSICAL-VALIDITY GATES ──────────────────────────────────────────────────
# The bounds are enforced here, upstream of every stage that consumes them.
# Otherwise impossible readings would propagate through the low-pass and the SG
# derivatives into a_long / a_lat / acc_mag and into std_data before anything
# checked them. The bounds are the physical-validity bounds of Section 4.3.4 /
# 4.3.6.
#
# LIMITATION (applies to BOTH gates): a gate is a metal detector. It catches
# an artifact of 138,345 km/h; it does NOT catch the same artifact at 150 km/h,
# nor an 8 m/s^2 artifact that is indistinguishable from hard braking. These
# gates are a disclosed safety net, NOT a substitute for root-cause diagnosis.
SPEED_MAX_MS       = 200.0 / 3.6        # 55.56 m/s = 200 km/h
A_MAX_MS2          = 15.0               # m/s^2  (~1.53 g)
J_MAX_MS3          = 30.0               # m/s^3
YAW_RATE_MAX_RADS  = math.radians(90.0) # 1.571 rad/s = 90 deg/s

# Gate-mask closing. Gating alone is NOT enough: NaN'ing only the frames above
# the bound leaves the *sub-threshold* garbage of the same corrupted episode in
# place, and prep_channel then LINEARLY INTERPOLATES the gated slits from those
# surviving garbage neighbours (gaps <= GAP_MAX_S are bridged). That is
# interpolation anchored on garbage — the exact failure mode diagnosed inside
# _clean_gps_position. Closing merges bad frames separated by less than
# GATE_CLOSE_S into ONE run, so a corrupted episode becomes a single long NaN
# run (> GAP_MAX_S -> masked, never bridged), while a genuinely isolated
# one-frame glitch still gets interpolated.
GATE_CLOSE_S       = 2.0                # s — merge bad runs closer than this

# Clip-drop guard. _clean_gps_position and the gap policy both REPAIR by linear
# interpolation. On a clip that is mostly garbage, "repair" fabricates a smooth,
# straight, near-zero-acceleration trajectory that enters the statistics as real
# CALM driving — biasing RQ1 in the opposite direction, invisibly. Below this
# threshold we drop the clip instead of inventing it.
#   denominator: grid frames with finite speed AFTER gating / total grid frames
MIN_VALID_FRAC_CLIP = 0.60

# Contamination-based drop. The valid-fraction guard above asks "how much
# survived?"; this asks "was the sensor ever proven broken?" A GPS that reported
# 12,000 km/h is not a sensor to trust at 150 km/h either — and its sub-bound
# garbage (108-200 km/h in central Oxford) is invisible to the gate by
# construction, so no closing window can remove it (the sweep asymptoted at 3
# residual frames even with CLOSE_S = 120 s). Empirically the rule is insensitive:
# Oxford's two broken clips sit at 18.8% / 35.1% gated while the worst of the
# other 68 sits at 0.04% — a 470x separation. Their suspect frames also run
# mean |a_long| = 4.0 m/s^2 vs 0.29 on legitimate frames (13.8x more violent):
# GPS garbage, not fast driving.
GATE_FRAC_DROP = 0.15   # drop a clip if >15% of its finite speed frames were gated

# Channels that are measurement-limited for a given dataset. NaN'd after
# construction, recorded in col_sources as SRC_EXCLUDED, and disclosed in the
# thesis — never a silent drop.
MEASUREMENT_LIMITED = {
    # 4Seasons carries NO measured speed (loader passes position + IMU only), so
    # speed is 100% position-derived. Its raw pose oscillates along-track at
    # ~2 Hz with ~+/-1.3 m amplitude. Predicted velocity error A*2*pi*f
    # = 1.3 * 2pi * 2 ~= 16 m/s ~= 59 km/h; observed swings are +/-35 km/h.
    # 2 Hz is BELOW F_CUT_HZ (3.0) — the noise is IN-BAND, i.e. inside the
    # vehicle-dynamics passband, and therefore indistinguishable in frequency
    # from the signal. No filter can remove it. Raw IMU is independent: retained.
    #
    # ── DECISION: exclusion DISABLED, 4Seasons retained ──
    # in full, WITH the contamination above, disclosed in Section 4.4. The
    # diagnosis remains valid: Gate 2 masks only the supra-threshold artifacts
    # (|a| > 15 m/s^2); sub-threshold garbage indistinguishable from genuine
    # maneuvering remains in every speed-derived channel, so all 4Seasons
    # speed/acceleration statistics read as noise-inflated upper bounds.
    # Re-enable the exclusion by uncommenting the entry below.
    # "4seasons": [
    #     "speed", "speed_kmh", "a_long", "abs_a_long", "jerk_long",
    #     "abs_jerk_long", "a_lat", "abs_a_lat", "jerk_lat", "acc_mag",
    #     "abs_acc_mag", "jerk_mag", "abs_jerk_mag", "deceleration",
    #     "curvature", "abs_curvature", "curvature_valid",
    # ],
    # NOTE: DurLAR's longitudinal channel is diagnosed as unrecoverable (sigma
    # ratio ~117x vs IMU, near-zero correlation), but it is NOT excluded here,
    # so DurLAR's a_long/jerk_long are PRESENT in the data. Exclusion entry,
    # left disabled:
    # "durlar": ["a_long", "abs_a_long", "jerk_long", "abs_jerk_long"],
}

# ── NAMED EXCLUSION OF INDIVIDUALLY-DIAGNOSED BROKEN SOURCE FILES ────────────
# Why a blocklist and not a smarter rule: raw oxford_clip_0039 gates at 8.4% of
# its frames while DurLAR's LEGITIMATELY RETAINED clips gate at 10.7-11.7% — the
# bad Oxford file sits BELOW the good DurLAR files on every per-clip gate
# statistic, so no threshold can separate them. These three files are excluded
# by name, with the diagnostic evidence recorded here:
#   * oxford_clip_0013: max raw speed    636 km/h;   883/1,439 frames > 200 (61%)
#   * oxford_clip_0039: max raw speed 11,559 km/h; 1,237 frames > 200 + 137 in-band
#   * oxford_clip_0051: max raw speed 138,439 km/h; 3,908 frames > 200 + 948 in-band
#   * context: Oxford is a ~10 km central-urban loop; every OTHER clip's max is
#     90.9 km/h, and the suspect frames run mean |a_long| = 4.0 m/s^2 vs 0.29 on
#     legitimate frames (13.8x) — GPS/INS failure, not fast driving.
# Keyed on the RAW SOURCE FILENAME (stable), never on output clip indices
# (which shift whenever a clip is dropped).
KNOWN_BAD_SOURCES = {
    "oxford": {
        "oxford_clip_0013.parquet",
        "oxford_clip_0039.parquet",
        "oxford_clip_0051.parquet",
    },
}

MAX_WORKERS = min(os.cpu_count() or 4, 8)

PIPELINE_VERSION = "v4_4_10hz_common_grid_validity_gated"  # 4Seasons exclusion disabled (decision in MEASUREMENT_LIMITED)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

def _find_dataset_root() -> Path:
    import sys
    candidates = []
    if len(sys.argv) > 1:
        candidates.append(Path(sys.argv[1]).expanduser().resolve())
    script_dir = Path(__file__).parent.resolve()
    candidates += [
        script_dir,
        Path.cwd().resolve(),
        script_dir.parent,
        Path.cwd().resolve() / "dataset",
        script_dir / "dataset",
        DATA_ROOT,
    ]
    for root in candidates:
        if (root / "parquet_data").is_dir():
            return root
    checked = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not find 'parquet_data/' in any of:\n  " + checked +
        "\nFix: python standardize_inertial_data.py ~/Downloads/dataset"
    )

# Resolved lazily in main() so the module can be imported for testing.
BASE_DIR: Path = None
PARQUET_DIR: Path = None
OUTPUT_DIR: Path = None

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_COLUMNS = [
    "timestamp_ns", "timestamp_s", "t_rel_ns", "t_rel_s", "dt_ns", "dt_s",
    "x", "y", "z", "lat", "lon",
    "x_rel", "y_rel", "x_aligned", "y_aligned", "heading_initial",
    "vx", "vy", "vz", "speed", "speed_kmh",
    "ax", "ay", "az", "a_long", "a_lat", "deceleration", "acc_mag",
    "jerk_long", "jerk_lat", "jerk_mag", "abs_jerk_long", "abs_jerk_mag",
    "roll", "pitch", "yaw", "yaw_unwrapped", "yaw_rate",
    "heading_change", "abs_heading_change",
    "curvature", "abs_curvature", "curvature_valid",
]

SRC_RAW      = "RAW"
SRC_DERIV    = "DERIVED"
SRC_NA       = "NOT_AVAILABLE"
SRC_FLAG     = "FLAG"
SRC_EXCLUDED = "EXCLUDED_MEASUREMENT_LIMITED"   # see MEASUREMENT_LIMITED

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def quat_to_euler(qw, qx, qy, qz):
    """Unit quaternion arrays → (roll, pitch, yaw) in radians."""
    qw = np.asarray(qw, dtype=float)
    qx = np.asarray(qx, dtype=float)
    qy = np.asarray(qy, dtype=float)
    qz = np.asarray(qz, dtype=float)
    sinr = 2.0 * (qw * qx + qy * qz)
    cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll  = np.arctan2(sinr, cosr)
    sinp  = np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny  = 2.0 * (qw * qz + qx * qy)
    cosy  = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw   = np.arctan2(siny, cosy)
    return roll, pitch, yaw


def latlon_to_xy(lat, lon):
    """GPS degrees → local metric (x, y) in metres via flat-Earth projection."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat0, lon0 = lat[0], lon[0]
    R = 6_371_000.0
    x = R * np.radians(lon - lon0) * np.cos(np.radians(lat0))
    y = R * np.radians(lat - lat0)
    return x, y


def hampel(x, native_dt, window_s=HAMPEL_WINDOW_S, t0=HAMPEL_T0):
    """
    Hampel identifier: replaces only statistical outliers (> t0 * 1.4826 * MAD
    from rolling median). Preserves genuine extreme events (hard braking,
    swerves) — unlike a blanket rolling median, which flattens them.
    Window is specified in seconds and converted to frames using native_dt.
    """
    x = np.asarray(x, dtype=float)
    k = max(5, int(round(window_s / native_dt)))
    k = k if k % 2 == 1 else k + 1   # must be odd
    s = pd.Series(x)
    med = s.rolling(k, center=True, min_periods=1).median()
    mad = 1.4826 * (s - med).abs().rolling(k, center=True, min_periods=1).median()
    outlier = ((s - med).abs() > t0 * mad) & mad.notna() & (mad > 0)
    y = x.copy()
    y[outlier.values] = med.values[outlier.values]
    return y


def _close_bad_runs(bad, max_gap_frames):
    """
    Morphological closing of a boolean gate mask.

    Merge bad-flag runs separated by <= max_gap_frames GOOD frames into one run.

    Why this exists: gating only the frames ABOVE the bound leaves the
    sub-threshold garbage of the same corrupted episode in place, and
    prep_channel then linearly interpolates the gated slits FROM those surviving
    garbage neighbours (any gap <= GAP_MAX_S is bridged). Closing turns a
    corrupted episode into a single long NaN run, which exceeds GAP_MAX_S and is
    therefore MASKED rather than bridged — while a genuinely isolated one-frame
    glitch, with no other bad frame within max_gap_frames, is left alone and
    still gets interpolated.
    """
    bad = np.asarray(bad, dtype=bool)
    if max_gap_frames < 1 or not bad.any():
        return bad
    idx = np.flatnonzero(bad)
    if idx.size < 2:
        return bad
    out = bad.copy()
    d = np.diff(idx)
    # a gap of d-1 good frames sits between idx[i] and idx[i+1]
    for i in np.flatnonzero((d > 1) & (d - 1 <= max_gap_frames)):
        out[idx[i]:idx[i + 1]] = True
    return out


def lowpass_segments(x, fs_native, fc=F_CUT_HZ, order=BUTTER_ORDER, label=""):
    """
    Zero-phase Butterworth low-pass, applied per contiguous non-NaN segment.
    Segments shorter than 3 × filter length are left unfiltered. NaN frames
    stay NaN.

    Uses second-order sections (sos) instead of transfer-function (ba) form.
    At 4Seasons' 2000 Hz native rate the normalised cutoff is 3/1000 = 0.003 and
    the ba-form poles sit at |z| = 0.9964, and sos is far better conditioned there.
    A numerically failing filter leaves a segment UNFILTERED, so a warning is
    issued instead of the anti-alias step silently not happening.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4:
        return x.copy()

    nyq = fs_native / 2.0
    if fc >= nyq:
        return x.copy()   # cutoff at/above Nyquist — nothing to remove

    sos = butter(order, fc / nyq, btype="low", output="sos")
    min_len = 3 * (sos.shape[0] * 2 + 1)

    result = x.copy()
    nan_mask = np.isnan(x)
    changes = np.diff(nan_mask.astype(np.int8), prepend=1, append=1)
    starts  = np.where(changes == -1)[0]
    ends    = np.where(changes ==  1)[0]

    for s, e in zip(starts, ends):
        seg = x[s:e]
        if len(seg) >= min_len:
            try:
                result[s:e] = sosfiltfilt(sos, seg)
            except Exception as exc:   # never fail silently
                warnings.warn(
                    f"[lowpass] {label or 'signal'}: sosfiltfilt failed on a "
                    f"{len(seg)}-sample segment at fs={fs_native:.1f} Hz "
                    f"({exc}). SEGMENT LEFT UNFILTERED — anti-aliasing did not "
                    f"run for these samples.",
                    RuntimeWarning,
                )

    return result


def sg_deriv(x, order, window, dt=DT_GRID, poly=SG_POLY):
    """
    Derivative via Savitzky-Golay polynomial differentiation — the derivative
    is read off the local polynomial fit instead of finite-differencing,
    which avoids the smooth-then-diff noise re-amplification cascade.
    Operates per contiguous non-NaN segment; falls back to np.gradient
    (central differences) on segments too short for the SG window.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    result = np.full(n, np.nan)

    nan_mask = np.isnan(x)
    changes  = np.diff(nan_mask.astype(np.int8), prepend=1, append=1)
    starts   = np.where(changes == -1)[0]
    ends     = np.where(changes ==  1)[0]

    for s, e in zip(starts, ends):
        seg = x[s:e]
        seg_len = len(seg)
        if seg_len < window:
            if order == 1 and seg_len >= 2:
                result[s:e] = np.gradient(seg, dt)
            # 2nd derivative on a segment shorter than the window: leave NaN
            # (too little data for a trustworthy estimate)
            continue
        try:
            result[s:e] = savgol_filter(seg, window, poly, deriv=order, delta=dt)
        except Exception:
            if order == 1 and seg_len >= 2:
                result[s:e] = np.gradient(seg, dt)

    return result


def mask_derivative(deriv_arr, level_arr, edge_frames=EDGE_MASK,
                    dilation=GAP_DILATION):
    """
    NaN-out a derivative:
      • wherever the underlying LEVEL signal is NaN, dilated by `dilation`
        frames on each side (filter/SG edge effects leak across segment
        boundaries),
      • at the first/last `edge_frames` frames of the clip,
      • entirely, if the clip is shorter than 2 × edge_frames (NaN
        beats keeping edge artifacts).
    """
    out = np.asarray(deriv_arr, dtype=float).copy()
    n = len(out)
    if n == 0:
        return out

    if n <= 2 * edge_frames:
        out[:] = np.nan
        return out

    nanpos = np.isnan(np.asarray(level_arr, dtype=float))
    if nanpos.any():
        kernel = np.ones(2 * dilation + 1)
        dilated = np.convolve(nanpos.astype(float), kernel, mode="same") > 0
        out[dilated] = np.nan

    out[:edge_frames]  = np.nan
    out[-edge_frames:] = np.nan
    return out


def _clean_gps_position(x, y, ts_ns, speed_max_ms=GPS_SPEED_MAX):
    """
    DurLAR/NVIDIA sustained GPS offset correction.
    Detects multi-frame position offsets producing impossible speeds,
    then linearly interpolates x/y over the flagged region. (A Hampel/median
    filter cannot cover 60+ frame sustained offsets — this can.)
    """
    x  = np.array(x,  dtype=float)
    y  = np.array(y,  dtype=float)
    ts = np.array(ts_ns, dtype=np.int64)
    n  = len(x)

    dt = np.diff(ts) / 1e9
    dx = np.diff(x)
    dy = np.diff(y)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, np.sqrt(dx**2 + dy**2) / dt, np.nan)

    bad = np.zeros(n, dtype=bool)
    bad_step = speed > speed_max_ms
    bad[:-1] |= bad_step
    bad[1:]  |= bad_step

    if not bad.any():
        return x, y

    good_idx = np.where(~bad)[0]
    if len(good_idx) < 2:
        return x, y

    all_idx = np.arange(n)
    x_clean = np.interp(all_idx, good_idx, x[good_idx])
    y_clean = np.interp(all_idx, good_idx, y[good_idx])
    return x_clean, y_clean


# ─────────────────────────────────────────────────────────────────────────────
# CORE: build the 43-column standardized DataFrame for ONE clip
# ─────────────────────────────────────────────────────────────────────────────

def build_standard_frame(
    timestamp_ns,
    x=None, y=None, z=None,
    lat=None, lon=None,
    vx=None, vy=None, vz=None,
    speed_raw=None,
    ax_raw=None, ay_raw=None, az_raw=None,
    roll_raw=None, pitch_raw=None, yaw_raw=None,
    yaw_rate_raw=None,
    clean_gps=False,
):
    col_sources = {}
    n_raw = len(timestamp_ns)
    if n_raw < 2:
        return None, {}, {}

    # ── STEP 1: TIMESTAMP CLEANUP ─────────────────────────────────────────────
    ts_ns = np.asarray(timestamp_ns, dtype=np.int64)

    # Vectorised strictly-monotonic filter
    run_max = np.maximum.accumulate(ts_ns)
    keep_mono = np.concatenate(([True], ts_ns[1:] > run_max[:-1]))
    n_dropped_mono = int((~keep_mono).sum())
    ts_mono = ts_ns[keep_mono]
    if len(ts_mono) < 2:
        return None, {}, {}

    # Bucket B: drop near-zero dt gaps (duplicate packets explode derivatives)
    dt_mono = np.diff(ts_mono) / 1e9
    median_dt_s = float(np.median(dt_mono))
    if not np.isfinite(median_dt_s) or median_dt_s <= 0:
        return None, {}, {}
    native_hz = round(1.0 / median_dt_s)
    dt_min_s  = max(1e-9, median_dt_s * DT_MIN_FRAC)
    keep_dt = np.concatenate(([True], dt_mono >= dt_min_s))
    n_dropped_dt = int((~keep_dt).sum())

    native_indices  = np.where(keep_mono)[0][keep_dt]
    final_keep_full = np.zeros(n_raw, dtype=bool)
    final_keep_full[native_indices] = True

    ts_native   = ts_ns[final_keep_full]
    n_native    = len(ts_native)
    if n_native < 5:
        return None, {}, {}
    t_rel_s_nat = (ts_native - ts_native[0]) / 1e9
    fs_native   = 1.0 / median_dt_s
    gap_limit_frames = max(1, int(GAP_MAX_S / median_dt_s))

    def arr_k(v):
        if v is None:
            return np.full(n_native, np.nan)
        a_full = np.asarray(v, dtype=float)
        if len(a_full) < len(final_keep_full):
            tmp = np.full(len(final_keep_full), np.nan)
            tmp[:len(a_full)] = a_full
            a_full = tmp
        return a_full[final_keep_full]

    # ── GRID DEFINITION + GAP MASKS ──────────────────────────────────────────
    t_grid = np.arange(t_rel_s_nat[0], t_rel_s_nat[-1] + DT_GRID * 0.5, DT_GRID)
    n_grid = len(t_grid)
    if n_grid < 2:
        return None, {}, {}

    # Timestamp-gap mask: native dt > GAP_MAX_S (applied to level signals too)
    dt_nat_arr = np.diff(t_rel_s_nat)
    ts_gap_mask_grid = np.zeros(n_grid, dtype=bool)
    for gi in np.where(dt_nat_arr > GAP_MAX_S)[0]:
        ts_gap_mask_grid |= (t_grid > t_rel_s_nat[gi]) & (t_grid < t_rel_s_nat[gi + 1])

    gap_fraction = float(dt_nat_arr[dt_nat_arr > GAP_MAX_S].sum() /
                         t_rel_s_nat[-1]) if t_rel_s_nat[-1] > 0 else 0.0

    def nan_run_mask_grid(x_nat_raw):
        """Grid mask covering NaN runs longer than GAP_MAX_S
        (sensor dropout while timestamps keep ticking)."""
        isn = np.isnan(np.asarray(x_nat_raw, dtype=float))
        mask = np.zeros(n_grid, dtype=bool)
        if not isn.any():
            return mask
        d = np.diff(isn.astype(np.int8), prepend=0, append=0)
        starts = np.where(d == 1)[0]
        ends   = np.where(d == -1)[0]          # exclusive
        for s, e in zip(starts, ends):
            t0 = t_rel_s_nat[s]
            t1 = t_rel_s_nat[min(e - 1, n_native - 1)]
            if t1 - t0 > GAP_MAX_S:
                mask |= (t_grid >= t0) & (t_grid <= t1)
        return mask

    def prep_channel(x_nat_raw, do_lp=True):
        """
        Full level-signal path: interpolate short NaN gaps → low-pass at native
        rate → resample onto the 10 Hz grid → mask timestamp gaps AND this
        channel's long NaN runs (no fabricated bridges).
        """
        x_nat_raw = np.asarray(x_nat_raw, dtype=float)
        if np.all(np.isnan(x_nat_raw)):
            return np.full(n_grid, np.nan)
        ch_mask = nan_run_mask_grid(x_nat_raw)
        filled = pd.Series(x_nat_raw).interpolate(
            method="linear", limit=gap_limit_frames,
            limit_direction="both").values
        sig = lowpass_segments(filled, fs_native) if do_lp else filled
        valid = ~np.isnan(sig)
        if valid.sum() < 2:
            return np.full(n_grid, np.nan)
        g = np.interp(t_grid, t_rel_s_nat[valid], sig[valid],
                      left=np.nan, right=np.nan)
        g[ts_gap_mask_grid | ch_mask] = np.nan
        return g

    # ── STEP 2: POSITION (native rate) ───────────────────────────────────────
    x_nat   = arr_k(x)
    y_nat   = arr_k(y)
    z_nat   = arr_k(z)
    lat_nat = arr_k(lat)
    lon_nat = arr_k(lon)

    col_sources["x"]   = SRC_RAW if x   is not None else SRC_NA
    col_sources["y"]   = SRC_RAW if y   is not None else SRC_NA
    col_sources["z"]   = SRC_RAW if z   is not None else SRC_NA
    col_sources["lat"] = SRC_RAW if lat is not None else SRC_NA
    col_sources["lon"] = SRC_RAW if lon is not None else SRC_NA

    if np.all(np.isnan(x_nat)) and not np.all(np.isnan(lat_nat)):
        x_nat, y_nat = latlon_to_xy(lat_nat, lon_nat)
        col_sources["x"] = SRC_DERIV
        col_sources["y"] = SRC_DERIV

    if clean_gps and not np.all(np.isnan(x_nat)):
        x_nat, y_nat = _clean_gps_position(x_nat, y_nat, ts_native)

    # ── STEP 3: VELOCITY (native rate) ───────────────────────────────────────
    vx_nat = arr_k(vx)
    vy_nat = arr_k(vy)
    vz_nat = arr_k(vz)

    col_sources["vx"] = SRC_RAW if vx is not None else SRC_NA
    col_sources["vy"] = SRC_RAW if vy is not None else SRC_NA
    col_sources["vz"] = SRC_RAW if vz is not None else SRC_NA

    def _lp_fill(sig):
        filled = pd.Series(sig).interpolate(
            method="linear", limit=gap_limit_frames,
            limit_direction="both").values
        return lowpass_segments(filled, fs_native)

    # Derive velocity from LOW-PASSED positions via central
    # differences (np.gradient handles non-uniform t), not raw backward diffs.
    if np.all(np.isnan(vx_nat)) and not np.all(np.isnan(x_nat)):
        x_lp_nat = _lp_fill(x_nat)
        y_lp_nat = _lp_fill(y_nat)
        with np.errstate(invalid="ignore"):
            vx_nat = np.gradient(x_lp_nat, t_rel_s_nat)
            vy_nat = np.gradient(y_lp_nat, t_rel_s_nat)
        col_sources["vx"] = SRC_DERIV
        col_sources["vy"] = SRC_DERIV

    if np.all(np.isnan(vz_nat)) and not np.all(np.isnan(z_nat)):
        z_lp_nat = _lp_fill(z_nat)
        with np.errstate(invalid="ignore"):
            vz_nat = np.gradient(z_lp_nat, t_rel_s_nat)
        col_sources["vz"] = SRC_DERIV

    if speed_raw is not None:
        speed_nat = arr_k(speed_raw)
        col_sources["speed"] = SRC_RAW
    else:
        speed_nat = np.sqrt(vx_nat**2 + vy_nat**2)
        col_sources["speed"] = SRC_DERIV if not np.all(np.isnan(speed_nat)) else SRC_NA

    # ── STEP 3b: PHYSICAL-VALIDITY GATE ON SPEED ─────────────────────────────
    # Both a lower and an upper bound. Without the upper bound, impossible values
    # such as Oxford (138,345 km/h), DurLAR (5,374) and NVIDIA (336) would flow on
    # into the low-pass, the SG derivatives, a_lat = speed * yaw_rate, acc_mag,
    # and therefore std_data.
    #
    # PLACEMENT IS DELIBERATE — this must run BEFORE prep_channel, because
    # lowpass_segments uses a ZERO-PHASE (non-causal) filter that smears a spike
    # both forwards and backwards in time. Gate after the low-pass and the poison
    # has already leaked into frames that were clean.
    #
    # (It is not a purity guarantee for POSITION-derived speed: there, _lp_fill
    # has already interpolated and filtered the positions before a velocity even
    # exists, so some smearing precedes the gate. Closing mitigates exactly that.)
    #
    # An impossible reading is a dropout that forgot to be NaN. Handing it to the
    # gap policy (STEP 5) reuses machinery that already exists.
    n_speed_finite_pre = int(np.isfinite(speed_nat).sum())
    bad_speed = (speed_nat < 0.0) | (speed_nat > SPEED_MAX_MS)
    bad_speed = _close_bad_runs(bad_speed, max(1, int(round(GATE_CLOSE_S / median_dt_s))))
    n_speed_gated = int((bad_speed & np.isfinite(speed_nat)).sum())
    speed_nat = np.where(bad_speed, np.nan, speed_nat)

    # ── STEP 4: HAMPEL point-outlier removal on speed ────────────────────────
    # NOTE: Hampel flags a point only if it deviates from the ROLLING MEDIAN by
    # > HAMPEL_T0 * MAD. Inside a SUSTAINED garbage run the rolling median is
    # itself garbage and the MAD is small, so nothing is flagged. Hampel kills
    # impulses, not sustained excursions — which is why STEP 3b above, not this
    # step, is what actually removes the Oxford / DurLAR corruption.
    if not np.all(np.isnan(speed_nat)) and n_native > 5:
        speed_nat = hampel(speed_nat, median_dt_s)

    # ── ORIENTATION (native rate) ────────────────────────────────────────────
    roll_nat   = arr_k(roll_raw)
    pitch_nat  = arr_k(pitch_raw)
    yaw_nat    = arr_k(yaw_raw)
    yaw_uw_nat = np.unwrap(yaw_nat)

    col_sources["roll"]          = SRC_RAW if roll_raw  is not None else SRC_NA
    col_sources["pitch"]         = SRC_RAW if pitch_raw is not None else SRC_NA
    col_sources["yaw"]           = SRC_RAW if yaw_raw   is not None else SRC_NA
    col_sources["yaw_unwrapped"] = SRC_DERIV

    ax_nat = arr_k(ax_raw)
    ay_nat = arr_k(ay_raw)
    az_nat = arr_k(az_raw)
    col_sources["ax"] = SRC_RAW if ax_raw is not None else SRC_NA
    col_sources["ay"] = SRC_RAW if ay_raw is not None else SRC_NA
    col_sources["az"] = SRC_RAW if az_raw is not None else SRC_NA

    # ── STEPS 5–7: GAP POLICY + LOW-PASS + RESAMPLE (per channel) ────────────
    speed_g   = prep_channel(speed_nat)
    x_g       = prep_channel(x_nat)
    y_g       = prep_channel(y_nat)
    z_g       = prep_channel(z_nat)
    yaw_uw_g  = prep_channel(yaw_uw_nat)
    vx_g      = prep_channel(vx_nat)
    vy_g      = prep_channel(vy_nat)
    vz_g      = prep_channel(vz_nat)
    roll_g    = prep_channel(roll_nat)
    pitch_g   = prep_channel(pitch_nat)
    ax_g      = prep_channel(ax_nat)
    ay_g      = prep_channel(ay_nat)
    az_g      = prep_channel(az_nat)
    # lat/lon: reference coordinates, never differentiated — no low-pass,
    # but gap-masked like everything else (no fabricated GPS tracks).
    lat_g     = prep_channel(lat_nat, do_lp=False)
    lon_g     = prep_channel(lon_nat, do_lp=False)

    # Wrapped yaw re-derived from clean unwrapped yaw. Interpolating
    # the wrapped signal across ±π jumps fabricates impossible headings.
    yaw_g = np.arctan2(np.sin(yaw_uw_g), np.cos(yaw_uw_g))

    if yaw_rate_raw is not None:
        yaw_rate_g = prep_channel(arr_k(yaw_rate_raw))
        col_sources["yaw_rate"] = SRC_RAW
    else:
        yaw_rate_g = None   # derived below from yaw_uw_g

    # ── STEP 8: DERIVATIVES VIA SG POLYNOMIAL DIFFERENTIATION ────────────────
    a_long_g    = sg_deriv(speed_g, order=1, window=W_D1)
    jerk_long_g = sg_deriv(speed_g, order=2, window=W_D2)

    if yaw_rate_g is None:
        yaw_rate_g = sg_deriv(yaw_uw_g, order=1, window=W_D1)
        yaw_rate_g = mask_derivative(yaw_rate_g, yaw_uw_g)
        col_sources["yaw_rate"] = (SRC_DERIV if not np.all(np.isnan(yaw_rate_g))
                                   else SRC_NA)

    # ── STEP 8a: GATE YAW RATE BEFORE IT MULTIPLIES INTO a_lat ───────────────
    # a_lat is speed * yaw_rate — a DIRECT multiplication, no derivative and no
    # filter in between. A bad yaw rate therefore poisons a_lat, acc_mag and
    # std_acc_mag just as effectively as a bad speed does. Several datasets take
    # yaw from the same INS attitude solution that produced their velocities
    # (Oxford), or store it in the wrong unit (the ZOD deg/rad precedent), so
    # this channel gets the same published bound as the others rather than an
    # assumption that it is clean.
    n_yaw_gated = int((np.isfinite(yaw_rate_g)
                       & (np.abs(yaw_rate_g) > YAW_RATE_MAX_RADS)).sum())
    yaw_rate_g = np.where(np.abs(yaw_rate_g) <= YAW_RATE_MAX_RADS,
                          yaw_rate_g, np.nan)

    a_lat_g    = speed_g * yaw_rate_g
    jerk_lat_g = sg_deriv(a_lat_g, order=1, window=W_D1)

    # Velocity fallback from gridded positions (rare: only when native-rate
    # derivation found nothing). Masked like every other derivative.
    if np.all(np.isnan(vx_g)) and not np.all(np.isnan(x_g)):
        vx_g = mask_derivative(sg_deriv(x_g, order=1, window=W_D1), x_g)
        col_sources["vx"] = SRC_DERIV
    if np.all(np.isnan(vy_g)) and not np.all(np.isnan(y_g)):
        vy_g = mask_derivative(sg_deriv(y_g, order=1, window=W_D1), y_g)
        col_sources["vy"] = SRC_DERIV
    if np.all(np.isnan(vz_g)) and not np.all(np.isnan(z_g)):
        vz_g = mask_derivative(sg_deriv(z_g, order=1, window=W_D1), z_g)
        col_sources["vz"] = SRC_DERIV

    # ── STEP 9: DERIVATIVE GAP + EDGE MASKING ────────────────────────────────
    a_long_g    = mask_derivative(a_long_g,    speed_g)
    jerk_long_g = mask_derivative(jerk_long_g, speed_g)
    a_lat_g     = mask_derivative(a_lat_g,     a_lat_g)
    jerk_lat_g  = mask_derivative(jerk_lat_g,  a_lat_g)

    # ── STEP 9b: VALIDITY GATE ON THE DERIVED CHANNELS ───────────────────────
    # Applied AFTER masking and BEFORE acc_mag / jerk_mag, so the magnitudes are
    # built from gated components rather than inheriting garbage through a sqrt.
    #
    # LIMITATION: this gate is a safety net, not a root-cause fix. It catches KITTI's 21 m/s^2 and
    # KITScenes' 36 m/s^2 — but it CANNOT catch the same artifact at 8 m/s^2,
    # which is indistinguishable from hard braking. If a dataset's derivative
    # stage manufactures artifacts, the ones this gate removes are the tip; the
    # sub-threshold ones remain, invisibly, inside acc_mag and std_acc_mag.
    # The KITTI and KITScenes artifacts are not root-caused, and this gate does
    # not resolve them.
    _pre = [a_long_g, a_lat_g, jerk_long_g, jerk_lat_g]
    _lim = [A_MAX_MS2, A_MAX_MS2, J_MAX_MS3, J_MAX_MS3]
    n_deriv_gated = int(sum((np.isfinite(v) & (np.abs(v) > lim)).sum()
                            for v, lim in zip(_pre, _lim)))
    a_long_g    = np.where(np.abs(a_long_g)    <= A_MAX_MS2, a_long_g,    np.nan)
    a_lat_g     = np.where(np.abs(a_lat_g)     <= A_MAX_MS2, a_lat_g,     np.nan)
    jerk_long_g = np.where(np.abs(jerk_long_g) <= J_MAX_MS3, jerk_long_g, np.nan)
    jerk_lat_g  = np.where(np.abs(jerk_lat_g)  <= J_MAX_MS3, jerk_lat_g,  np.nan)

    # ── DERIVED SIGNALS (after masking + gating, NaN propagates) ─────────────
    jerk_mag_g      = np.sqrt(jerk_long_g**2 + jerk_lat_g**2)
    abs_jerk_long_g = np.abs(jerk_long_g)
    abs_jerk_mag_g  = jerk_mag_g.copy()
    acc_mag_g       = np.sqrt(a_long_g**2 + a_lat_g**2)

    # NaN-preserving deceleration (NaN must NOT become a fake 0.0)
    deceleration_g = np.where(
        np.isnan(a_long_g), np.nan,
        np.where(a_long_g < 0, -a_long_g, 0.0)
    )

    col_sources["a_long"]       = SRC_DERIV if not np.all(np.isnan(a_long_g)) else SRC_NA
    col_sources["a_lat"]        = SRC_DERIV if not np.all(np.isnan(a_lat_g))  else SRC_NA
    col_sources["deceleration"] = col_sources["a_long"]
    col_sources["acc_mag"]      = SRC_DERIV
    jerk_src = SRC_DERIV if not np.all(np.isnan(jerk_long_g)) else SRC_NA
    for c in ["jerk_long", "jerk_lat", "jerk_mag", "abs_jerk_long", "abs_jerk_mag"]:
        col_sources[c] = jerk_src

    # ── HEADING CHANGE (per 0.1 s grid frame) ────────────────────────────────
    heading_change_g     = np.concatenate(([np.nan], np.diff(yaw_uw_g)))
    abs_heading_change_g = np.abs(heading_change_g)
    hc_src = SRC_DERIV if not np.all(np.isnan(heading_change_g)) else SRC_NA
    col_sources["heading_change"]     = hc_src
    col_sources["abs_heading_change"] = hc_src

    # ── TRAJECTORY ALIGNMENT ──────────────────────────────────────────────────
    def _first_finite(a, default=np.nan):
        f = a[np.isfinite(a)]
        return float(f[0]) if len(f) else default

    x0 = _first_finite(x_g)
    y0 = _first_finite(y_g)
    x_rel_g = x_g - x0 if np.isfinite(x0) else x_g.copy()
    y_rel_g = y_g - y0 if np.isfinite(y0) else y_g.copy()

    heading_initial = _first_finite(yaw_uw_g)
    if not np.isfinite(heading_initial):
        k = min(5, n_grid - 1)
        dx0 = x_g[k] - x_g[0]
        dy0 = y_g[k] - y_g[0]
        heading_initial = float(np.arctan2(dy0, dx0)) \
            if np.isfinite(dx0) and np.isfinite(dy0) else 0.0

    rot = math.pi / 2.0 - heading_initial
    x_aligned_g = x_rel_g * np.cos(rot) - y_rel_g * np.sin(rot)
    y_aligned_g = x_rel_g * np.sin(rot) + y_rel_g * np.cos(rot)

    for c in ["x_rel", "y_rel", "x_aligned", "y_aligned", "heading_initial"]:
        col_sources[c] = SRC_DERIV

    # ── STEP 10: CURVATURE ────────────────────────────────────────────────────
    curv_valid_g = np.where(np.isnan(speed_g), False, speed_g > CURV_SPEED_MIN)
    with np.errstate(divide="ignore", invalid="ignore"):
        curvature_g = np.where(curv_valid_g, yaw_rate_g / speed_g, np.nan)
    abs_curvature_g = np.abs(curvature_g)
    curv_src = SRC_DERIV if not np.all(np.isnan(curvature_g)) else SRC_NA
    col_sources["curvature"]       = curv_src
    col_sources["abs_curvature"]   = curv_src
    col_sources["curvature_valid"] = SRC_FLAG

    # ── GRID TIMESTAMPS ───────────────────────────────────────────────────────
    ts_ns_g    = (ts_native[0] + (t_grid * 1e9)).round().astype(np.int64)
    ts_s_g     = ts_ns_g / 1e9
    t_rel_ns_g = ts_ns_g - ts_ns_g[0]
    t_rel_s_g  = t_rel_ns_g / 1e9

    dt_ns_raw = np.empty(n_grid, dtype=object)
    dt_ns_raw[0] = pd.NA
    dt_ns_raw[1:] = np.diff(ts_ns_g).astype(np.int64)
    dt_ns_arr = pd.array(dt_ns_raw, dtype="Int64")

    dt_s_g = np.full(n_grid, np.nan)
    dt_s_g[1:] = np.diff(ts_ns_g) / 1e9

    for c in ["timestamp_ns", "timestamp_s", "t_rel_ns", "t_rel_s", "dt_ns", "dt_s"]:
        col_sources[c] = SRC_DERIV

    speed_kmh_g = speed_g * 3.6
    col_sources["speed_kmh"] = SRC_DERIV

    # ── ASSEMBLE ──────────────────────────────────────────────────────────────
    df = pd.DataFrame({
        "timestamp_ns":       pd.array(ts_ns_g,    dtype="Int64"),
        "timestamp_s":        ts_s_g,
        "t_rel_ns":           pd.array(t_rel_ns_g, dtype="Int64"),
        "t_rel_s":            t_rel_s_g,
        "dt_ns":              dt_ns_arr,
        "dt_s":               dt_s_g,
        "x":                  x_g,
        "y":                  y_g,
        "z":                  z_g,
        "lat":                lat_g,
        "lon":                lon_g,
        "x_rel":              x_rel_g,
        "y_rel":              y_rel_g,
        "x_aligned":          x_aligned_g,
        "y_aligned":          y_aligned_g,
        "heading_initial":    np.full(n_grid, heading_initial),
        "vx":                 vx_g,
        "vy":                 vy_g,
        "vz":                 vz_g,
        "speed":              speed_g,
        "speed_kmh":          speed_kmh_g,
        "ax":                 ax_g,
        "ay":                 ay_g,
        "az":                 az_g,
        "a_long":             a_long_g,
        "a_lat":              a_lat_g,
        "deceleration":       deceleration_g,
        "acc_mag":            acc_mag_g,
        "jerk_long":          jerk_long_g,
        "jerk_lat":           jerk_lat_g,
        "jerk_mag":           jerk_mag_g,
        "abs_jerk_long":      abs_jerk_long_g,
        "abs_jerk_mag":       abs_jerk_mag_g,
        "roll":               roll_g,
        "pitch":              pitch_g,
        "yaw":                yaw_g,
        "yaw_unwrapped":      yaw_uw_g,
        "yaw_rate":           yaw_rate_g,
        "heading_change":     heading_change_g,
        "abs_heading_change": abs_heading_change_g,
        "curvature":          curvature_g,
        "abs_curvature":      abs_curvature_g,
        "curvature_valid":    curv_valid_g.astype(bool),
    })

    # ── CLIP-DROP GUARD ──────────────────────────────────────────────────────
    # Denominator (pinned): finite gridded speed frames / total grid frames.
    # Below MIN_VALID_FRAC_CLIP the clip is DROPPED, not repaired — because both
    # _clean_gps_position and the gap policy repair by linear interpolation, and
    # interpolating a mostly-garbage clip fabricates a smooth, straight,
    # near-zero-acceleration trajectory that would enter the statistics as real
    # CALM driving. That biases RQ1 in the opposite direction, with no visible
    # trace in any figure. Dropping the clip is the safer choice.
    n_grid = int(len(t_grid))
    valid_frac_speed = (float(np.isfinite(speed_g).sum()) / n_grid) if n_grid else 0.0
    # Contamination fraction at native rate (post-closing count / pre-gate finite)
    gated_frac_speed = (float(n_speed_gated) / n_speed_finite_pre
                        if n_speed_finite_pre else 0.0)

    processing_meta = {
        "native_hz":                  int(native_hz),
        "native_median_dt_s":         float(median_dt_s),
        "n_frames_raw":               int(n_raw),
        "n_frames_dropped_monotonic": int(n_dropped_mono),
        "n_frames_dropped_dt":        int(n_dropped_dt),
        "gap_fraction":               float(gap_fraction),
        "grid_hz":                    int(GRID_HZ),
        "f_cut_hz":                   float(F_CUT_HZ),
        "butter_order":               int(BUTTER_ORDER),
        "hampel_window_s":            float(HAMPEL_WINDOW_S),
        "hampel_t0":                  float(HAMPEL_T0),
        "gap_max_s":                  float(GAP_MAX_S),
        "sg_poly":                    int(SG_POLY),
        "w_d1":                       int(W_D1),
        "w_d2":                       int(W_D2),
        "edge_mask_frames":           int(EDGE_MASK),
        "pipeline_version":           PIPELINE_VERSION,
        # ── gate telemetry (feeds the gated-mass report) ─────────────────────
        "speed_max_kmh_gate":         float(SPEED_MAX_MS * 3.6),
        "a_max_ms2_gate":             float(A_MAX_MS2),
        "j_max_ms3_gate":             float(J_MAX_MS3),
        "yaw_rate_max_degs_gate":     float(math.degrees(YAW_RATE_MAX_RADS)),
        "gate_close_s":               float(GATE_CLOSE_S),
        "n_speed_finite_pre_gate":    int(n_speed_finite_pre),
        "n_speed_gated":              int(n_speed_gated),
        "n_yaw_rate_gated":           int(n_yaw_gated),
        "n_deriv_gated":              int(n_deriv_gated),
        "valid_frac_speed":           float(valid_frac_speed),
        "min_valid_frac_clip":        float(MIN_VALID_FRAC_CLIP),
        "gated_frac_speed":           float(gated_frac_speed),
        "gate_frac_drop":             float(GATE_FRAC_DROP),
        "dropped_reason":             None,
    }

    drop_reasons = []
    if valid_frac_speed < MIN_VALID_FRAC_CLIP:
        drop_reasons.append(
            f"valid_frac_speed={valid_frac_speed:.3f} < {MIN_VALID_FRAC_CLIP:.2f} "
            f"after the physical-validity gate — dropped rather than interpolated")
    if gated_frac_speed > GATE_FRAC_DROP:
        drop_reasons.append(
            f"gated_frac_speed={gated_frac_speed:.3f} > {GATE_FRAC_DROP:.2f} — "
            f"GPS proven broken on this clip; sub-bound garbage is invisible to "
            f"the gate, so the clip is dropped, not trusted")
    if drop_reasons:
        processing_meta["dropped_reason"] = "; ".join(drop_reasons)
        return None, col_sources, processing_meta

    return df[OUTPUT_COLUMNS], col_sources, processing_meta


# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────

def apply_measurement_limits(df, col_sources, dataset_name):
    """
    NaN the channels listed in MEASUREMENT_LIMITED for this dataset and
    record the exclusion in col_sources (which is persisted to the parquet
    schema metadata). This is a DISCLOSED exclusion, not a silent drop: the
    columns remain present with the correct dtype so every downstream script
    keeps its schema, but they carry no values and no derived statistic.
    """
    cols = MEASUREMENT_LIMITED.get(dataset_name, [])
    if not cols or df is None or df.empty:
        return df, col_sources, 0
    n = 0
    for c in cols:
        if c not in df.columns:
            continue
        if pd.api.types.is_bool_dtype(df[c]):
            df[c] = False          # e.g. curvature_valid — cannot hold NaN
        else:
            df[c] = np.nan
        col_sources[c] = SRC_EXCLUDED
        n += 1
    return df, col_sources, n


def save_clip(df, col_sources, processing_meta, out_path: Path):
    table = pa.Table.from_pandas(df, preserve_index=False)
    meta  = table.schema.metadata or {}
    meta[b"col_sources"]     = json.dumps(col_sources).encode()
    meta[b"processing_meta"] = json.dumps(processing_meta).encode()
    table = table.replace_schema_metadata(meta)
    pq.write_table(table, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def to_timestamp_ns(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.astype(np.int64)
    try:
        vals = series.astype(float)
    except (ValueError, TypeError):
        try:
            return pd.to_datetime(series, format="ISO8601").astype(np.int64)
        except Exception:
            return pd.to_datetime(series, format="mixed").astype(np.int64)
    median = float(np.nanmedian(vals))
    if median > 1e17:
        return vals.round().astype(np.int64)
    elif median > 1e14:
        return (vals * 1_000).round().astype(np.int64)
    elif median > 1e11:
        return (vals * 1_000_000).round().astype(np.int64)
    else:
        return (vals * 1_000_000_000).round().astype(np.int64)


def load_4seasons(f):
    raw = pd.read_parquet(f)
    ts_ns = to_timestamp_ns(raw["timestamp"])
    roll_r, pitch_r, yaw_r = quat_to_euler(
        raw["qw"], raw["qx"], raw["qy"], raw["qz"])
    return build_standard_frame(
        timestamp_ns = ts_ns.values,
        x            = raw["x"].values   if "x"  in raw.columns else None,
        y            = raw["y"].values   if "y"  in raw.columns else None,
        z            = raw["z"].values   if "z"  in raw.columns else None,
        ax_raw       = raw["ax"].values,
        ay_raw       = raw["ay"].values,
        az_raw       = raw["az"].values  if "az" in raw.columns else None,
        roll_raw     = roll_r,
        pitch_raw    = pitch_r,
        yaw_raw      = yaw_r,
        yaw_rate_raw = raw["wz"].values  if "wz" in raw.columns else None,
    )


def load_a2d2(f):
    raw = pd.read_parquet(f)
    ts_ns = (raw["timestamp"].astype(np.int64) * 1000)
    return build_standard_frame(
        timestamp_ns = ts_ns.values,
        # vehicle_speed is ALREADY m/s. Evidence from A2D2's own data: the raw
        # IMU acceleration_x involves no unit factor, and dividing speed by 3.6
        # gives corr(ax, a_long) = 0.977 with std(ax)/std(a_long) = 3.73 ~= 3.6,
        # i.e. a_long shrunk by exactly that factor. With the division, max
        # speed is 20.4 km/h (implausible) and max|a_long| 1.11 m/s^2. Read as
        # m/s: 73.3 km/h and 4.0 m/s^2, matching the IMU's independent 4.43.
        speed_raw    = raw["vehicle_speed"].values,

        ax_raw       = raw["acceleration_x"].values,
        ay_raw       = raw["acceleration_y"].values,
        az_raw       = raw["acceleration_z"].values if "acceleration_z" in raw.columns else None,
        roll_raw     = np.radians(raw["roll_angle"].values),
        pitch_raw    = np.radians(raw["pitch_angle"].values),
        yaw_rate_raw = raw["angular_velocity_omega_z"].values,
    )


def load_boreas(f):
    raw = pd.read_parquet(f)
    ts_col = "GPSTime" if "GPSTime" in raw.columns else "ROSTime"
    ts_ns  = (raw[ts_col].astype(float) * 1e9).round().astype(np.int64)
    return build_standard_frame(
        timestamp_ns = ts_ns.values,
        x            = raw["easting"].values   if "easting"   in raw.columns else None,
        y            = raw["northing"].values  if "northing"  in raw.columns else None,
        z            = raw["altitude"].values  if "altitude"  in raw.columns else None,
        lat          = raw["latitude"].values  if "latitude"  in raw.columns else None,
        lon          = raw["longitude"].values if "longitude" in raw.columns else None,
        vx           = raw["vel_east"].values  if "vel_east"  in raw.columns else None,
        vy           = raw["vel_north"].values if "vel_north" in raw.columns else None,
        vz           = raw["vel_up"].values    if "vel_up"    in raw.columns else None,
        speed_raw    = raw["speed"].values     if "speed"     in raw.columns else None,
        ax_raw       = raw["accelx"].values    if "accelx"    in raw.columns else None,
        ay_raw       = raw["accely"].values    if "accely"    in raw.columns else None,
        az_raw       = raw["accelz"].values    if "accelz"    in raw.columns else None,
        roll_raw     = np.radians(raw["roll"].values)    if "roll"    in raw.columns else None,
        pitch_raw    = np.radians(raw["pitch"].values)   if "pitch"   in raw.columns else None,
        yaw_raw      = np.radians(raw["heading"].values) if "heading" in raw.columns else None,
        yaw_rate_raw = raw["angvel_z"].values  if "angvel_z"  in raw.columns else None,
    )


def load_durlar(f):
    raw = pd.read_parquet(f)
    ts_ns = raw["timestamp"].astype(np.int64)
    roll_r, pitch_r, yaw_r = quat_to_euler(
        raw["field.orientation.w"], raw["field.orientation.x"],
        raw["field.orientation.y"], raw["field.orientation.z"])
    return build_standard_frame(
        timestamp_ns = ts_ns.values,
        lat          = raw["field.latitude"].values,
        lon          = raw["field.longitude"].values,
        z            = raw["field.altitude"].values,
        ax_raw       = raw["field.linear_acceleration.x"].values,
        ay_raw       = raw["field.linear_acceleration.y"].values,
        az_raw       = raw["field.linear_acceleration.z"].values,
        roll_raw     = roll_r,
        pitch_raw    = pitch_r,
        yaw_raw      = yaw_r,
        yaw_rate_raw = raw["field.angular_velocity.z"].values,
        clean_gps    = True,
    )


def load_ford(f):
    raw = pd.read_parquet(f)
    ts_ns = to_timestamp_ns(raw["timestamp"])
    roll_r, pitch_r, yaw_r = quat_to_euler(
        raw["qw"], raw["qx"], raw["qy"], raw["qz"])
    return build_standard_frame(
        timestamp_ns = ts_ns.values,
        x            = raw["x"].values        if "x"         in raw.columns else None,
        y            = raw["y"].values        if "y"         in raw.columns else None,
        z            = raw["z"].values        if "z"         in raw.columns else None,
        lat          = raw["latitude"].values  if "latitude"  in raw.columns else None,
        lon          = raw["longitude"].values if "longitude" in raw.columns else None,
        vx           = raw["vx"].values        if "vx"        in raw.columns else None,
        vy           = raw["vy"].values        if "vy"        in raw.columns else None,
        vz           = raw["vz"].values        if "vz"        in raw.columns else None,
        speed_raw    = raw["speed"].values     if "speed"     in raw.columns else None,
        ax_raw       = raw["ax"].values        if "ax"        in raw.columns else None,
        ay_raw       = raw["ay"].values        if "ay"        in raw.columns else None,
        az_raw       = raw["az"].values        if "az"        in raw.columns else None,
        roll_raw     = roll_r,
        pitch_raw    = pitch_r,
        yaw_raw      = yaw_r,
        yaw_rate_raw = raw["wz"].values        if "wz"        in raw.columns else None,
    )


def load_hdd(f):
    raw = pd.read_parquet(f)
    ts_ns = to_timestamp_ns(raw["timestamp"])
    # HDD has swapped column names: 'lat' contains longitude values (-122°)
    # and 'lng' contains latitude values (37°). Confirmed: 'lat' violates
    # the ±90° latitude constraint. Swap corrected here before projection.
    has_coords = "lat" in raw.columns and "lng" in raw.columns
    if has_coords:
        true_lat = raw["lng"].values   # mislabeled as 'lng', actually latitude
        true_lon = raw["lat"].values   # mislabeled as 'lat', actually longitude
        x_arr, y_arr = latlon_to_xy(true_lat, true_lon)
    else:
        true_lat, true_lon, x_arr, y_arr = None, None, None, None
    yaw_rad = np.radians(raw["yaw"].values) if "yaw" in raw.columns else None
    return build_standard_frame(
        timestamp_ns = ts_ns.values,
        x            = x_arr,
        y            = y_arr,
        lat          = true_lat,
        lon          = true_lon,
        speed_raw    = raw["vel"].values / 3.6 if "vel" in raw.columns else None,  # vel is in km/h
        yaw_raw      = yaw_rad,
    )


def load_kitti_generic(f):
    raw = pd.read_parquet(f)
    ts_ns = to_timestamp_ns(raw["timestamp"])
    return build_standard_frame(
        timestamp_ns = ts_ns.values,
        z            = raw["alt"].values   if "alt"   in raw.columns else None,
        lat          = raw["lat"].values   if "lat"   in raw.columns else None,
        lon          = raw["lon"].values   if "lon"   in raw.columns else None,
        vx           = raw["vf"].values    if "vf"    in raw.columns else None,
        vy           = raw["vl"].values    if "vl"    in raw.columns else None,
        vz           = raw["vu"].values    if "vu"    in raw.columns else None,
        ax_raw       = raw["ax"].values    if "ax"    in raw.columns else None,
        ay_raw       = raw["ay"].values    if "ay"    in raw.columns else None,
        az_raw       = raw["az"].values    if "az"    in raw.columns else None,
        roll_raw     = raw["roll"].values  if "roll"  in raw.columns else None,
        pitch_raw    = raw["pitch"].values if "pitch" in raw.columns else None,
        yaw_raw      = raw["yaw"].values   if "yaw"   in raw.columns else None,
        yaw_rate_raw = raw["wz"].values    if "wz"    in raw.columns else None,
    )


def load_nuscene(f):
    raw = pd.read_parquet(f)
    ts_ns = (raw["utime"].astype(np.int64) * 1000)
    roll_r, pitch_r, yaw_r = quat_to_euler(
        raw["qw"], raw["qx"], raw["qy"], raw["qz"])
    return build_standard_frame(
        timestamp_ns = ts_ns.values,
        x            = raw["x"].values     if "x"     in raw.columns else None,
        y            = raw["y"].values     if "y"     in raw.columns else None,
        z            = raw["z"].values     if "z"     in raw.columns else None,
        vx           = raw["vel_x"].values if "vel_x" in raw.columns else None,
        vy           = raw["vel_y"].values if "vel_y" in raw.columns else None,
        vz           = raw["vel_z"].values if "vel_z" in raw.columns else None,
        speed_raw    = raw["speed"].values if "speed" in raw.columns else None,
        ax_raw       = raw["ax"].values    if "ax"    in raw.columns else None,
        ay_raw       = raw["ay"].values    if "ay"    in raw.columns else None,
        az_raw       = raw["az"].values    if "az"    in raw.columns else None,
        roll_raw     = roll_r,
        pitch_raw    = pitch_r,
        yaw_raw      = yaw_r,
        yaw_rate_raw = raw["wz"].values    if "wz"    in raw.columns else None,
    )


def load_nvidia(f):
    raw = pd.read_parquet(f)
    # Drop first frame — bad initial timestamp gap (8–57ms vs normal 100ms)
    raw = raw.iloc[1:].reset_index(drop=True)
    ts_ns = (raw["timestamp"].astype(np.int64) * 1_000)
    roll_r, pitch_r, yaw_r = quat_to_euler(
        raw["qw"], raw["qx"], raw["qy"], raw["qz"])
    return build_standard_frame(
        timestamp_ns = ts_ns.values,
        x            = raw["x"].values  if "x"  in raw.columns else None,
        y            = raw["y"].values  if "y"  in raw.columns else None,
        z            = raw["z"].values  if "z"  in raw.columns else None,
        vx           = raw["vx"].values if "vx" in raw.columns else None,
        vy           = raw["vy"].values if "vy" in raw.columns else None,
        vz           = raw["vz"].values if "vz" in raw.columns else None,
        ax_raw       = raw["ax"].values if "ax" in raw.columns else None,
        ay_raw       = raw["ay"].values if "ay" in raw.columns else None,
        az_raw       = raw["az"].values if "az" in raw.columns else None,
        roll_raw     = roll_r,
        pitch_raw    = pitch_r,
        yaw_raw      = yaw_r,
        clean_gps    = True,
    )


def load_kitscenes(f):
    raw = pd.read_parquet(f)
    # KitScenes timestamp is already in seconds.
    ts_ns = (raw["timestamp"].astype(float) * 1e9).round().astype(np.int64)
    roll_r, pitch_r, yaw_r = quat_to_euler(
        raw["qw"], raw["qx"], raw["qy"], raw["qz"])
    return build_standard_frame(
        timestamp_ns = ts_ns.values,
        x            = raw["x"].values if "x" in raw.columns else None,
        y            = raw["y"].values if "y" in raw.columns else None,
        z            = raw["z"].values if "z" in raw.columns else None,
        roll_raw     = roll_r,
        pitch_raw    = pitch_r,
        yaw_raw      = yaw_r,
        clean_gps    = True,
    )


def load_oxford(f):
    # The standard Hampel + 3 Hz low-pass path alone does not handle Oxford:
    # Hampel kills impulses, and Oxford clip_0051 carries a ~8-minute SUSTAINED
    # garbage run (4,847 frames > 108 km/h), for which the rolling median is
    # itself garbage and the MAD is small.
    #
    # clean_gps is NOT what cleans Oxford's speed. This loader passes RAW INS
    # velocities (velocity_east/north), so speed = sqrt(vx^2+vy^2) never touches
    # position, while _clean_gps_position repairs x/y ONLY. Enabling it here is
    # a TRAJECTORY fix (x/y/easting/northing). Oxford's SPEED is cleaned solely
    # by the STEP 3b gate + mask closing. (For the same reason clean_gps does not
    # remove the DurLAR and NVIDIA speed artifacts.)
    raw = pd.read_parquet(f)
    ts_ns = to_timestamp_ns(raw["timestamp"])
    return build_standard_frame(
        timestamp_ns = ts_ns.values,
        x            = raw["easting"].values        if "easting"        in raw.columns else None,
        y            = raw["northing"].values       if "northing"       in raw.columns else None,
        z            = raw["altitude"].values       if "altitude"       in raw.columns else None,
        lat          = raw["latitude"].values       if "latitude"       in raw.columns else None,
        lon          = raw["longitude"].values      if "longitude"      in raw.columns else None,
        vx           = raw["velocity_east"].values  if "velocity_east"  in raw.columns else None,
        vy           = raw["velocity_north"].values if "velocity_north" in raw.columns else None,
        vz           = raw["velocity_down"].values  if "velocity_down"  in raw.columns else None,
        roll_raw     = raw["roll"].values  if "roll"  in raw.columns else None,
        pitch_raw    = raw["pitch"].values if "pitch" in raw.columns else None,
        yaw_raw      = raw["yaw"].values   if "yaw"   in raw.columns else None,
        clean_gps    = True,   # trajectory columns only; see note above
    )


def load_waymo(f):
    raw = pd.read_parquet(f)

    if "scenario_id" not in raw.columns:
        ts_ns = (raw["timestamp_s"].astype(float) * 1e9).round().astype(np.int64)
        return [(build_standard_frame(
            timestamp_ns = ts_ns.values,
            x            = raw["x"].values          if "x"          in raw.columns else None,
            y            = raw["y"].values          if "y"          in raw.columns else None,
            z            = raw["z"].values          if "z"          in raw.columns else None,
            vx           = raw["velocity_x"].values if "velocity_x" in raw.columns else None,
            vy           = raw["velocity_y"].values if "velocity_y" in raw.columns else None,
            speed_raw    = raw["speed"].values      if "speed"      in raw.columns else None,
            yaw_raw      = raw["heading"].values    if "heading"    in raw.columns else None,
        ), None)]

    clips = []
    for scen_id, grp in raw.groupby("scenario_id"):
        grp   = grp.sort_values("timestamp_s").reset_index(drop=True)
        ts_ns = (grp["timestamp_s"].astype(float) * 1e9).round().astype(np.int64)
        result = build_standard_frame(
            timestamp_ns = ts_ns.values,
            x            = grp["x"].values          if "x"          in grp.columns else None,
            y            = grp["y"].values          if "y"          in grp.columns else None,
            z            = grp["z"].values          if "z"          in grp.columns else None,
            vx           = grp["velocity_x"].values if "velocity_x" in grp.columns else None,
            vy           = grp["velocity_y"].values if "velocity_y" in grp.columns else None,
            speed_raw    = grp["speed"].values      if "speed"      in grp.columns else None,
            yaw_raw      = grp["heading"].values    if "heading"    in grp.columns else None,
        )
        clips.append((result, str(scen_id)))
    return clips


def _zod_yaw_rate(raw):
    """ZOD UNIT FIX: angularRateZ is stored in °/s,
    not rad/s. Evidence: heading-derivative ratio test gave a median
    |gyro| / |d(heading)/dt| of 57.26 ~= 180/pi across all sampled clips;
    IMU cross-check |v*w| p95 = 1.39 m/s^2 (deg reading) vs 79.4 m/s^2
    (rad reading) against measured |ay| p95 = 1.55 m/s^2.
    Conversion is UNCONDITIONAL: per-file auto-detection would leave calm
    clips unconverted and create mixed units across clips. A sanity warning
    fires if the upstream convention ever changes."""
    if "angularRateZ" not in raw.columns:
        return None
    gyro_dps = raw["angularRateZ"].values.astype(float)
    gyro = np.radians(gyro_dps)
    p999 = np.nanpercentile(np.abs(gyro), 99.9) if np.isfinite(gyro).any() else 0.0
    if p999 > 4.0:   # > 229 deg/s sustained is physically impossible
        print("  [WARN] ZOD: yaw_rate p99.9 = "
              f"{p999:.1f} rad/s AFTER deg->rad conversion - upstream unit "
              "convention may have changed; re-run diagnose_zod_yawrate.py")
    return gyro


def load_zod(f):
    raw = pd.read_parquet(f)
    ts_ns = to_timestamp_ns(raw["timestamp"])

    heading_col = raw["heading"].values if "heading" in raw.columns else None
    if heading_col is not None and np.nanmax(np.abs(heading_col)) > 4.0:
        heading_col = np.radians(heading_col)

    roll_col  = raw["roll"].values  if "roll"  in raw.columns else None
    pitch_col = raw["pitch"].values if "pitch" in raw.columns else None
    if roll_col  is not None and np.nanmax(np.abs(roll_col))  > 4.0:
        roll_col  = np.radians(roll_col)
    if pitch_col is not None and np.nanmax(np.abs(pitch_col)) > 4.0:
        pitch_col = np.radians(pitch_col)

    return build_standard_frame(
        timestamp_ns = ts_ns.values,
        lat          = raw["posLat"].values if "posLat" in raw.columns else None,
        lon          = raw["posLon"].values if "posLon" in raw.columns else None,
        z            = raw["posAlt"].values if "posAlt" in raw.columns else None,
        vx           = raw["velEast"].values    if "velEast"    in raw.columns else None,
        vy           = raw["velNorth"].values   if "velNorth"   in raw.columns else None,
        vz           = raw["velDown"].values    if "velDown"    in raw.columns else None,
        speed_raw    = raw["speed"].values      if "speed"      in raw.columns else None,
        ax_raw       = raw["accelerationX"].values if "accelerationX" in raw.columns else None,
        ay_raw       = raw["accelerationY"].values if "accelerationY" in raw.columns else None,
        az_raw       = raw["accelerationZ"].values if "accelerationZ" in raw.columns else None,
        roll_raw     = roll_col,
        pitch_raw    = pitch_col,
        yaw_raw      = heading_col,
        yaw_rate_raw = _zod_yaw_rate(raw),
    )


# ─────────────────────────────────────────────────────────────────────────────
# LOADER REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

LOADERS = {
    "4seasons":  load_4seasons,
    "a2d2":      load_a2d2,
    "boreas_rt": load_boreas,
    "durlar":    load_durlar,
    "ford":      load_ford,
    "hdd":       load_hdd,
    "kitti":     load_kitti_generic,
    "kitti360":  load_kitti_generic,
    "nuscene":   load_nuscene,
    "nuscenes":  load_nuscene,
    "kitscenes": load_kitscenes,
    "nvidia":    load_nvidia,
    "oxford":    load_oxford,
    "waymo":     load_waymo,
    "zod":       load_zod,
}


# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def _process_one_file(args):
    f, loader = args
    try:
        result = loader(f)
        if isinstance(result, list):
            items = []
            for item in result:
                if isinstance(item, tuple) and len(item) == 2:
                    # Waymo returns (build_result_3tuple, scen_id) — unwrap
                    inner, _ = item
                    if isinstance(inner, tuple) and len(inner) == 3:
                        items.append(inner)
                elif isinstance(item, tuple) and len(item) == 3:
                    items.append(item)
        elif isinstance(result, tuple) and len(result) == 3:
            items = [result]
        else:
            items = []
        # Stamp the raw source filename into every clip's metadata.
        # Output clip indices RENUMBER whenever anything is dropped, so they can
        # never identify a source. This makes the raw<->output mapping a lookup.
        for _, _, pm in items:
            if isinstance(pm, dict):
                pm["source_file"] = f.name
        return items
    except Exception:
        return [f"ERROR::{f.name}::{traceback.format_exc()}"]


def process_dataset(dataset_name: str, global_bar: tqdm):
    src_dir = PARQUET_DIR / dataset_name
    out_dir = OUTPUT_DIR  / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(src_dir.rglob("*.parquet"))
    if not parquet_files:
        global_bar.write(f"  [{dataset_name}] SKIP – no parquet files found")
        return 0, 0

    loader_key = dataset_name.lower()
    if loader_key not in LOADERS:
        global_bar.write(f"  [{dataset_name}] SKIP – no loader registered")
        return 0, 0

    loader  = LOADERS[loader_key]

    # ── apply the named source blocklist (see KNOWN_BAD_SOURCES) ─────────────
    blocked = KNOWN_BAD_SOURCES.get(loader_key, set())
    if blocked:
        skipped = [f.name for f in parquet_files if f.name in blocked]
        parquet_files = [f for f in parquet_files if f.name not in blocked]
        for name in skipped:
            global_bar.write(f"  [{dataset_name}] BLOCKLISTED source skipped: {name}")

    # ── purge stale outputs before writing ───────────────────────────────────
    # Because clips can be dropped, a run may write fewer clips than a previous
    # run, which would leave stale clip_00NN files at the high indices and mix
    # fresh and stale outputs. Purge first so the directory contains only what
    # THIS run writes.
    stale = sorted(out_dir.glob("clip_*.parquet"))
    for f in stale:
        f.unlink()
    if stale:
        global_bar.write(f"  [{dataset_name}] purged {len(stale)} existing output "
                         f"clip files before writing")

    n_files = len(parquet_files)
    errors  = []
    results = [None] * n_files

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_idx = {
            pool.submit(_process_one_file, (f, loader)): i
            for i, f in enumerate(parquet_files)
        }
        with tqdm(
            total=n_files, desc=f"  {dataset_name:<12}", unit="file",
            leave=False, dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} files [{elapsed}<{remaining}]",
        ) as pbar:
            for future in as_completed(future_to_idx):
                idx       = future_to_idx[future]
                clip_list = future.result()
                ok, errs  = [], []
                for item in clip_list:
                    if isinstance(item, str) and item.startswith("ERROR::"):
                        errs.append(item)
                    else:
                        ok.append(item)
                results[idx] = ok
                errors.extend(errs)
                pbar.update(1)

    for err in errors:
        parts  = err.split("::", 2)
        fname  = parts[1] if len(parts) > 1 else "?"
        detail = parts[2] if len(parts) > 2 else err
        global_bar.write(f"  [{dataset_name}] ERROR in {fname}:\n{detail}")

    clip_index = 1
    saved = 0
    # ── gate telemetry, aggregated per dataset ───────────────────────────────
    dropped_clips   = []
    n_speed_gated   = 0
    n_yaw_gated     = 0
    n_deriv_gated   = 0
    n_speed_pre     = 0
    n_limited_cols  = 0

    for clip_list in results:
        if not clip_list:
            continue
        for item in clip_list:
            if not isinstance(item, tuple) or len(item) != 3:
                continue
            df, col_sources, processing_meta = item

            n_speed_gated += int(processing_meta.get("n_speed_gated", 0) or 0)
            n_yaw_gated   += int(processing_meta.get("n_yaw_rate_gated", 0) or 0)
            n_deriv_gated += int(processing_meta.get("n_deriv_gated", 0) or 0)
            n_speed_pre   += int(processing_meta.get("n_speed_finite_pre_gate", 0) or 0)

            # A clip below MIN_VALID_FRAC_CLIP is dropped, never fabricated.
            if df is None or df.empty:
                reason = processing_meta.get("dropped_reason")
                if reason:
                    src = processing_meta.get("source_file", "?")
                    dropped_clips.append(f"{src} — {reason}")
                continue

            # Disclosed per-dataset channel exclusions (MEASUREMENT_LIMITED)
            df, col_sources, n_lim = apply_measurement_limits(
                df, col_sources, dataset_name)
            n_limited_cols = max(n_limited_cols, n_lim)

            out_name = f"clip_{clip_index:04d}.parquet"
            save_clip(df, col_sources, processing_meta, out_dir / out_name)
            clip_index += 1
            saved += 1

    # ── report the gate, never let it act silently ───────────────────────────
    if n_speed_gated or n_yaw_gated or n_deriv_gated or dropped_clips or n_limited_cols:
        pct = (100.0 * n_speed_gated / n_speed_pre) if n_speed_pre else 0.0
        global_bar.write(
            f"  [{dataset_name}] v4 gate: "
            f"speed {n_speed_gated:,} ({pct:.3f}%) | "
            f"yaw {n_yaw_gated:,} | deriv {n_deriv_gated:,} | "
            f"clips dropped {len(dropped_clips)} | "
            f"channels excluded {n_limited_cols}"
        )
        for reason in dropped_clips[:5]:
            global_bar.write(f"      dropped clip: {reason}")

    global_bar.update(n_files)
    return saved, len(errors)


def main():
    global BASE_DIR, PARQUET_DIR, OUTPUT_DIR
    BASE_DIR    = _find_dataset_root()
    PARQUET_DIR = BASE_DIR / "parquet_data"
    OUTPUT_DIR  = BASE_DIR / "inertial_data"

    print("=" * 60)
    print(" Script 14 v3 – Ego-Motion Standardization")
    print(f" Grid      : {GRID_HZ} Hz  |  LP cutoff: {F_CUT_HZ} Hz  |  SG: W_D1={W_D1} W_D2={W_D2}")
    print(f" Hampel    : window={HAMPEL_WINDOW_S}s  t0={HAMPEL_T0}")
    print(f" Gap policy: interpolate ≤ {GAP_MAX_S}s, mask longer (timestamp AND NaN-run gaps)")
    print(f" Workers   : {MAX_WORKERS} threads")
    print(f" Input     : {PARQUET_DIR}")
    print(f" Output    : {OUTPUT_DIR}")
    print("=" * 60)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets    = sorted(d.name for d in PARQUET_DIR.iterdir() if d.is_dir())
    all_files   = [f for ds in datasets
                   for f in sorted((PARQUET_DIR / ds).rglob("*.parquet"))]
    total_files = len(all_files)

    print(f" Datasets : {len(datasets)}")
    print(f" Files    : {total_files} total")
    print()

    total_saved = total_errors = 0

    with tqdm(
        total=total_files, desc="Overall", unit="file",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} files [{elapsed}<{remaining}, {rate_fmt}]",
    ) as master_bar:
        for ds in datasets:
            master_bar.set_postfix_str(ds)
            saved, errs   = process_dataset(ds, master_bar)
            total_saved  += saved
            total_errors += errs
            master_bar.write(
                f"  [{ds}] done — {saved} clip(s) saved"
                + (f", {errs} error(s)" if errs else "")
            )

    print()
    print("=" * 60)
    print(f" Done.  {total_saved} clips written to {OUTPUT_DIR}")
    if total_errors:
        print(f" Errors : {total_errors} file(s) failed")
    print("=" * 60)


if __name__ == "__main__":
    main()