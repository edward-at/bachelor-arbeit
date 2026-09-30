"""
hmm.py  —  Section 4.6.9, temporal regime structure (hidden Markov model)
────────────────────────────────────────────────────────────────────────────
Adds the time axis the static analyses discard, across ALL 14 datasets.

The 20 s analysis window is too coarse for temporal modelling: 10 of the 14
datasets are single ~20 s perception scenes, which collapse to one window and so
admit no within-clip transition. But a 20 s scene is not temporally empty — at
the 10 Hz frame grid it contains ~200 frames of braking, turning and
acceleration. This section therefore re-derives features at a finer resolution
FOR THE HMM ONLY: short sub-windows (HMM_WIN_S, HMM_STEP_S) computed within each
clip from the frame layer, so every dataset — including the ~20 s perception
sets — contributes ordered sequences. Sub-windows are never formed across clip
boundaries, so no transition is fabricated between two independent recordings.

A Gaussian HMM models the sub-window SEQUENCE — each regime a latent state
emitting the feature vector, the states evolving by a first-order transition
matrix. Three products absent from the static view: the transition matrix (its
diagonal = persistence), per-regime dwell time, and decoded regime timelines.

DATASET BALANCE: at the sub-window scale the largest datasets would again
dominate (NVIDIA's ~298 k scenes alone are millions of sub-windows). Each dataset
is therefore capped to CAP_SUBWIN_PER_DATASET sub-windows, accumulated as whole
clips, so the transition structure is a property of driving in general rather than
of the largest dataset — the same principle as the clustering fit-population cap.

REGIME ALIGNMENT: states are placed in the SAME regime space as the clustering by
standardising the sub-window features with the 20 s fit-population scaler, then
aligned to the k-means regimes by nearest emission mean. The features are the same
five; four are scale-stable means, but std_speed_kmh over a short sub-window is
necessarily smaller than over 20 s, so it is the least comparable dimension and
the alignment is dominated by the scale-stable features. The recovered emission
means are printed against the k-means centroids in physical units so the
alignment can be checked directly.

BOUNDARY: the transition structure here is the POOLED corpus's — how regimes
succeed one another in driving in general. The SAME construction per dataset, and
the comparison of one dataset's transitions against another's, is the bias
measurement of Section 4.7, not this section.

Requires hmmlearn. k is read from the saved k-sweep curve.

Reads:   dataset/regime_data/kselect_curves.parquet
         dataset/std_data/<dataset>.parquet                 (20 s regime reference)
         dataset/inertial_data/<dataset>/clip_*.parquet     (frame layer, sub-windows)
Writes:  dataset/regime_data/hmm_transitions.parquet
         dataset/regime_data/hmm_durations.parquet
         dataset/regime_data/hmm_states.parquet   (decoded example timelines)
         dataset/regime_data/hmm_sequences.parquet   (per dataset: clips used, sub-windows; block 5D)
         dataset/regime_data/hmm_alignment.parquet   (emission means vs k-means centroids, physical; block 5D)
         dataset/regime_data/hmm_run_summary.parquet (sequences, sub-windows, EM convergence; block 5D)
────────────────────────────────────────────────────────────────────────────
"""

import warnings
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from hmmlearn.hmm import GaussianHMM

# shared steps — single source of truth
from clustering import (_discover, _load, _build_fit, _order_regimes,
                        FEATURES, N_INIT, SEED, CAP, ARTIFACT_DIR, PROJECT_ROOT)

warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k):
        return x

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

INERTIAL_DIR           = PROJECT_ROOT / "dataset" / "inertial_data"
HMM_WIN_S              = 2.0      # sub-window width (s) — finer than the 20 s analysis window
HMM_STEP_S             = 1.0      # sub-window step (s) — 50 % overlap, within-clip only
GRID_HZ                = 10.0     # frame grid
MIN_VALID_FRAC         = 0.5      # drop a sub-window with fewer valid frames than this
CAP_SUBWIN_PER_DATASET = 30000    # per-dataset sub-window budget (dataset balance)
READ_CHUNK             = 512      # clips read per parallel batch
HMM_N_ITER             = 50       # EM iterations (raise toward 200 for the final run if not converged)
HMM_COV_TYPE           = "diag"   # emission covariance. The crash was a SINGULAR covariance on
                                  # hard-zero features (mean_acc_pos/neg are exactly 0 in steady
                                  # sub-windows); HMM_MIN_COVAR floors it and fixes that for 'full'.
                                  # FALLBACK: if a regime comes back unpopulated (0 runs, or two
                                  # states aligning to one regime in the alignment check), switch to
                                  # 'diag' — fewer parameters, more stable, slightly less expressive.
HMM_MIN_COVAR          = 1e-2     # covariance floor (standardised space) — prevents singular/NaN covariances
N_TIMELINE             = 3        # longest collected clips decoded for the timeline figure

WLEN = int(round(HMM_WIN_S * GRID_HZ))    # frames per sub-window
STEP = int(round(HMM_STEP_S * GRID_HZ))   # frames per step
FRAME_COLS = ["t_rel_s", "speed_kmh", "a_long", "yaw_rate"]

# Physical-validity bounds — same as BOUNDS_FRAME in the std/viz layer. Frames
# outside these are masked to NaN before the sub-window statistics are computed.
# yaw_rate is bounded in rad/s here (the frame layer stores rad/s); ±90°/s.
BOUNDS_SPEED = (0.0, 200.0)
BOUNDS_ALONG = (-15.0, 15.0)
BOUNDS_YAW   = (-np.radians(90.0), np.radians(90.0))


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _runs(seq: np.ndarray):
    """Yield (state, run_length) for each maximal constant run in a sequence."""
    if len(seq) == 0:
        return
    start = 0
    for i in range(1, len(seq)):
        if seq[i] != seq[i - 1]:
            yield int(seq[start]), i - start
            start = i
    yield int(seq[start]), len(seq) - start


def _subwindow_features(df: pd.DataFrame):
    """Frame-level → ordered sub-window feature rows (same five features as the
    20 s layer), plus each sub-window's start time. Within one clip only.

    Physical-bounds masking is applied first, identical to the std_data / viz
    layer: frames outside BOUNDS_FRAME are set NaN and excluded from the window
    statistics (stored data is never modified). The 20 s layer is clean because
    compute_std applies this; the sub-window features, read fresh from the raw
    frame layer (whose Hampel filter deliberately preserves tails), must apply it
    too, or sustained glitch frames (e.g. 10^5 km/h speed spikes) poison the
    means and produce non-physical regime states.
    """
    sp = df["speed_kmh"].to_numpy(float)
    al = df["a_long"].to_numpy(float)
    yr = df["yaw_rate"].to_numpy(float)
    tt = df["t_rel_s"].to_numpy(float)
    # out-of-bounds → NaN (excluded from nanmean/nanstd below)
    sp = np.where((sp >= BOUNDS_SPEED[0]) & (sp <= BOUNDS_SPEED[1]), sp, np.nan)
    al = np.where((al >= BOUNDS_ALONG[0]) & (al <= BOUNDS_ALONG[1]), al, np.nan)
    yr = np.where((yr >= BOUNDS_YAW[0])   & (yr <= BOUNDS_YAW[1]),   yr, np.nan)
    n = len(sp)
    if n < WLEN:
        return None
    rows, starts = [], []
    for s in range(0, n - WLEN + 1, STEP):
        e = s + WLEN
        spw = sp[s:e]
        if np.isfinite(spw).mean() < MIN_VALID_FRAC:
            continue
        alw, yrw = al[s:e], yr[s:e]
        pos = alw[(alw > 0) & np.isfinite(alw)]
        neg = alw[(alw < 0) & np.isfinite(alw)]
        fd = {"mean_speed_kmh":    np.nanmean(spw),
              "std_speed_kmh":     np.nanstd(spw),
              "mean_acc_pos":      float(pos.mean()) if pos.size else 0.0,
              # magnitude of the negative part, on the polarity of the window layer's
              # mean_acc_neg (block 5D fix: the signed conditional mean carried the
              # opposite sign and swapped the Braking / Accelerating alignment)
              "mean_acc_neg":      float(-neg.mean()) if neg.size else 0.0,
              "mean_abs_yaw_rate": np.nanmean(np.abs(yrw))}
        row = [fd[f] for f in FEATURES]
        if not np.all(np.isfinite(row)):
            continue
        rows.append(row)
        starts.append(float(tt[s]))
    if not rows:
        return None
    return np.asarray(rows, float), np.asarray(starts, float)


def _read_clip(fp):
    cid = int(fp.stem.split("_")[1])
    try:
        df = pd.read_parquet(fp, columns=FRAME_COLS)
    except Exception:
        return (fp.parent.name, cid, None, None)
    out = _subwindow_features(df)
    if out is None:
        return (fp.parent.name, cid, None, None)
    return (fp.parent.name, cid, out[0], out[1])


def _collect_dataset(ds: str):
    """Accumulate whole-clip sub-window sequences for one dataset up to the
    per-dataset budget; clips read in seeded-shuffled order, in parallel batches."""
    files = sorted((INERTIAL_DIR / ds).glob("clip_*.parquet"))
    if not files:
        return [], 0
    order = np.random.RandomState(SEED).permutation(len(files))
    files = [files[i] for i in order]

    seqs, total, i = [], 0, 0
    while i < len(files) and total < CAP_SUBWIN_PER_DATASET:
        chunk = files[i:i + READ_CHUNK]
        with ThreadPoolExecutor(max_workers=8) as ex:
            for ds_, cid, Xs, st in ex.map(_read_clip, chunk):
                if Xs is None:
                    continue
                seqs.append((ds_, cid, Xs, st))
                total += len(Xs)
                if total >= CAP_SUBWIN_PER_DATASET:
                    break
        i += READ_CHUNK
    return seqs, total


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    curves = pd.read_parquet(ARTIFACT_DIR / "kselect_curves.parquet")
    k = int(curves.loc[curves["silhouette"].idxmax(), "k"])

    print(f"\n{'='*78}")
    print(f"  hmm.py  ·  {_now()}")
    print(f"  states fixed at k={k}   ·   seed {SEED}   ·   sub-window {HMM_WIN_S:.0f}s/"
          f"{HMM_STEP_S:.0f}s   ·   cap {CAP_SUBWIN_PER_DATASET:,}/dataset")
    print(f"{'='*78}")

    # ── 20 s regime reference: fit-population scaler + k-means centroids ───────
    pop = _discover()
    with ThreadPoolExecutor(max_workers=min(8, len(pop))) as ex:
        frames = list(tqdm(ex.map(_load, pop), total=len(pop), desc="reading 20s"))
    full20 = pd.concat([f for f in frames if f is not None], ignore_index=True)
    fit    = _build_fit(full20, CAP, SEED)
    scaler = StandardScaler().fit(fit[FEATURES])
    km     = KMeans(k, n_init=N_INIT, random_state=SEED).fit(scaler.transform(fit[FEATURES]))
    klmap  = _order_regimes(km.cluster_centers_)
    kcent  = np.zeros_like(km.cluster_centers_)                 # ordered, scaled space
    for old, new in klmap.items():
        kcent[new] = km.cluster_centers_[old]
    kcent_phys = kcent * scaler.scale_ + scaler.mean_          # ordered, physical units

    # ── build balanced sub-window sequences from the frame layer ──────────────
    print(f"\n  Building {HMM_WIN_S:.0f}s sub-window sequences (all 14 datasets):")
    print(f"    {'dataset':<12}{'clips used':>12}{'sub-windows':>14}")
    X_list, starts_list, seq_meta = [], [], []
    seq_rows = []                                   # per-dataset collection (block 5D)
    for ds in tqdm(pop, desc="frame layer"):
        seqs, total = _collect_dataset(ds)
        used = 0
        for ds_, cid, Xs, st in seqs:
            X_list.append(Xs); starts_list.append(st)
            seq_meta.append((ds_, cid, len(Xs)))
            used += 1
        seq_rows.append({"dataset": ds, "clips_used": int(used), "sub_windows": int(total)})
        print(f"    {ds:<12}{used:>12,}{total:>14,}")

    X_sub    = np.vstack(X_list)
    starts   = np.concatenate(starts_list)
    lengths  = np.array([m[2] for m in seq_meta])
    print(f"\n  Collected: {len(seq_meta):,} sequences · {len(X_sub):,} sub-windows "
          f"(median sequence {int(np.median(lengths))} sub-windows, "
          f"longest {int(lengths.max())})")

    # ── fit the HMM in the 20 s regime space ──────────────────────────────────
    Xc = scaler.transform(X_sub)
    print(f"  Fitting Gaussian HMM (per-iteration log-likelihood below) …")
    hmm = GaussianHMM(n_components=k, covariance_type=HMM_COV_TYPE,
                      n_iter=HMM_N_ITER, random_state=SEED, verbose=True,
                      min_covar=HMM_MIN_COVAR, init_params="stmc", params="stmc")
    hmm.fit(Xc, lengths)
    states = hmm.predict(Xc, lengths)

    # align states -> k-means regimes by nearest emission mean (scaled space)
    ri, ci = linear_sum_assignment(cdist(hmm.means_, kcent))
    state2reg = {int(r): int(c) for r, c in zip(ri, ci)}
    regimes = np.array([state2reg[int(s)] for s in states])

    # alignment diagnostic — emission means vs k-means centroids in physical units
    means_phys = hmm.means_ * scaler.scale_ + scaler.mean_
    si = FEATURES.index("mean_speed_kmh")
    print(f"\n  Alignment check (physical mean speed, km/h):")
    print(f"    {'regime':>7}{'k-means centroid':>18}{'HMM emission':>15}")
    for st, rg in sorted(state2reg.items(), key=lambda kv: kv[1]):
        print(f"    {rg:>7}{kcent_phys[rg, si]:>18.1f}{means_phys[st, si]:>15.1f}")
    align_rows = []                                 # all five features, physical units (block 5D)
    for st, rg in sorted(state2reg.items(), key=lambda kv: kv[1]):
        row = {"regime": int(rg), "state": int(st)}
        for fi, f in enumerate(FEATURES):
            row[f"emission_{f}"] = float(means_phys[st, fi])
            row[f"centroid_{f}"] = float(kcent_phys[rg, fi])
        align_rows.append(row)

    # ── reordered transition matrix (rows/cols = regimes) ─────────────────────
    T = hmm.transmat_
    Tre = np.zeros_like(T)
    for a in range(k):
        for b in range(k):
            Tre[state2reg[a], state2reg[b]] = T[a, b]
    print(f"\n  Transition matrix (rows = from regime, cols = to regime):")
    print("        " + "".join(f"{j:>8}" for j in range(k)))
    for i in range(k):
        print(f"    {i:>3} " + "".join(f"{Tre[i, j]:>8.3f}" for j in range(k)))
    print(f"  Self-transition (persistence): {np.round(np.diag(Tre), 3).tolist()}")

    # ── per-regime dwell-time distributions (seconds) ─────────────────────────
    # a run of L sub-windows advances ≈ L · HMM_STEP_S seconds of drive.
    dwell = {r: [] for r in range(k)}
    off = 0
    for L in lengths:
        for st, run in _runs(states[off:off + L]):
            dwell[state2reg[st]].append(run * HMM_STEP_S)
        off += L
    print(f"\n  Per-regime dwell time (seconds, step = {HMM_STEP_S:.0f}s):")
    print(f"    {'regime':>7}{'n_runs':>9}{'mean':>9}{'median':>9}{'p90':>9}{'max':>9}")
    dur_rows = []
    for r in range(k):
        v = np.asarray(dwell[r], float)
        if len(v) == 0:
            v = np.array([0.0])
        dur_rows.append({"regime": r, "n_runs": int(len(dwell[r])),
                         "mean_s": round(float(v.mean()), 1),
                         "median_s": round(float(np.median(v)), 1),
                         "p25_s": round(float(np.percentile(v, 25)), 1),
                         "p75_s": round(float(np.percentile(v, 75)), 1),
                         "p90_s": round(float(np.percentile(v, 90)), 1),
                         "max_s": round(float(v.max()), 1)})
        print(f"    {r:>7}{len(dwell[r]):>9,}{v.mean():>9.1f}{np.median(v):>9.1f}"
              f"{np.percentile(v, 90):>9.1f}{v.max():>9.1f}")

    # ── decoded timelines for the longest collected clips ─────────────────────
    top_idx = np.argsort(lengths)[::-1][:N_TIMELINE]
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    tl_rows = []
    for idx in top_idx:
        ds_, cid, L = seq_meta[idx]
        a, b = offsets[idx], offsets[idx] + L
        for t, rg in zip(starts[a:b], regimes[a:b]):
            tl_rows.append({"dataset": ds_, "clip_id": cid,
                            "window_start_s": float(t), "regime": int(rg)})
    tl = pd.DataFrame(tl_rows)
    print(f"\n  Timeline example clips (longest {N_TIMELINE}): "
          f"{[f'{seq_meta[i][0]}/{seq_meta[i][1]}' for i in top_idx]}")

    # ── persist (parquet only) ────────────────────────────────────────────────
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(Tre, columns=[str(j) for j in range(k)]).to_parquet(
        ARTIFACT_DIR / "hmm_transitions.parquet", index=False)
    pd.DataFrame(dur_rows).to_parquet(ARTIFACT_DIR / "hmm_durations.parquet", index=False)
    tl.to_parquet(ARTIFACT_DIR / "hmm_states.parquet", index=False)

    # ── persisted for Chapter 5 (appended 2026-09-26, block 5D): the collection per
    #    dataset, the alignment check, and the run summary the console printed.
    #    No existing output changes.
    pd.DataFrame(seq_rows).to_parquet(ARTIFACT_DIR / "hmm_sequences.parquet", index=False)
    pd.DataFrame(align_rows).to_parquet(ARTIFACT_DIR / "hmm_alignment.parquet", index=False)
    hist = list(getattr(hmm.monitor_, "history", []))
    pd.DataFrame([{"n_sequences": int(len(seq_meta)),
                   "n_subwindows": int(len(X_sub)),
                   "median_seq_len": int(np.median(lengths)),
                   "longest_seq_len": int(lengths.max()),
                   "n_iter_max": int(HMM_N_ITER),
                   "n_iter_done": int(getattr(hmm.monitor_, "iter", len(hist))),
                   "converged": bool(getattr(hmm.monitor_, "converged", False)),
                   "final_loglik": float(hist[-1]) if hist else float("nan"),
                   "timeline_clips": ", ".join(f"{seq_meta[i][0]}/{seq_meta[i][1]}" for i in top_idx)}]
                 ).to_parquet(ARTIFACT_DIR / "hmm_run_summary.parquet", index=False)

    print(f"\n  Saved → {ARTIFACT_DIR}")
    print(f"          (hmm_transitions · hmm_durations · hmm_states)")
    print(f"          (hmm_sequences · hmm_alignment · hmm_run_summary)   (block 5D)")
    print(f"  Reproduce: rerun under seed {SEED}.")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()