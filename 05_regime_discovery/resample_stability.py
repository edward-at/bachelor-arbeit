"""
resample_stability.py  —  Section 4.6.6 resampling-stability control
────────────────────────────────────────────────────────────────────────────
Extends the single resampling check of the clustering (one redraw of the
stratified fit population, compared to the reference partition) to a
distribution: the stratified draw is repeated N_RESAMPLES times under different
seeds, the clustering refitted on each redraw with k and the k-means seed held
fixed, the full corpus relabelled by each, and every relabelling compared to the
reference partition by the adjusted Rand index. Reporting the distribution of
those indices — rather than a single value — shows that the regimes do not
depend on the particular windows drawn into the fit population.

DESIGN (locked):
  • Redraw-only: only the stratified sample changes between replicates; the
    k-means initialisation seed is held fixed, so this isolates sampling luck
    from initialisation luck (the latter is reseed_stability.py).
  • Each replicate is compared to the REFERENCE partition (the production
    seed-CAP-SEED clustering), not pairwise among replicates — the direct
    N-fold generalisation of the clustering's single resampling number, so the
    index remains interpretable as agreement with the production partition.
  • Permutation-invariant throughout (ARI), so cluster numbering never matters.
  • FIGURE: one point per replicate on a truncated ARI axis with the mean as a
    horizontal line — every replicate individually visible. The truncation and
    the 0.85 reference level are stated in the caption, not drawn in-axis.
  • REUSE_SAVED: when the persisted distribution already exists, the script
    replots from it without recomputing (set False to force the full rerun).

Every shared step (loading, the stratified fit build, the scaler+k-means fit,
the regime ordering) is imported from clustering.py so the two cannot diverge.
k is read from the saved k-sweep curve (argmax silhouette); no model file.

Reads:   dataset/regime_data/kselect_curves.parquet     (selected k)
         dataset/std_data/<dataset>.parquet             (via clustering._load)
Writes:  dataset/regime_data/resample_stability.parquet (the ARI distribution)
Output:  figures/cap_sensitivity/resample_stability.png
────────────────────────────────────────────────────────────────────────────
"""

import warnings
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

# shared steps — single source of truth (same imports the other §4.6.6 scripts use)
from clustering import (_discover, _load, _build_fit, _order_regimes,
                        FEATURES, N_INIT, SEED, CAP, ARTIFACT_DIR, PROJECT_ROOT)
from tum_colors import TUM_PRIMARY, TUM_SECONDARY

warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k):
        return x

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

N_RESAMPLES = 20          # number of independent stratified redraws
RESEED_BASE = SEED + 200  # redraw seeds: RESEED_BASE + 0 .. N_RESAMPLES-1
                          # (offset well clear of the clustering's SEED+100 single redraw)
REUSE_SAVED = True        # replot from the saved parquet if it exists (no refitting)
OUT_DIR     = PROJECT_ROOT / "bachelor-arbeit" / "05_regime_discovery" / "figures" / "cap_sensitivity"
DPI         = 300


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _fit_reference(full: pd.DataFrame, k: int):
    """The production partition: stratified fit at CAP under SEED, ordered labels
    over the full corpus. Identical construction to clustering.py."""
    fit    = _build_fit(full, CAP, SEED)
    scaler = StandardScaler().fit(fit[FEATURES])
    km     = KMeans(k, n_init=N_INIT, random_state=SEED).fit(scaler.transform(fit[FEATURES]))
    lmap   = _order_regimes(km.cluster_centers_)
    ref    = np.vectorize(lmap.get)(km.predict(scaler.transform(full[FEATURES])))
    return ref


def _fit_redraw(full: pd.DataFrame, k: int, draw_seed: int) -> np.ndarray:
    """One resampling replicate: redraw the stratified fit population under
    draw_seed, refit scaler + k-means (k-means seed held at SEED), relabel the
    full corpus. Only the DRAW seed varies between replicates."""
    fit    = _build_fit(full, CAP, draw_seed)                 # ← only this changes
    scaler = StandardScaler().fit(fit[FEATURES])
    km     = KMeans(k, n_init=N_INIT, random_state=SEED).fit(scaler.transform(fit[FEATURES]))
    lmap   = _order_regimes(km.cluster_centers_)
    return np.vectorize(lmap.get)(km.predict(scaler.transform(full[FEATURES])))


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE  (same construction as reseed_stability.py, for visual parallelism)
# ─────────────────────────────────────────────────────────────────────────────

def _plot(aris: np.ndarray, out_fp: Path):
    n = len(aris)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    x = np.arange(1, n + 1)
    ax.scatter(x, aris, color=TUM_SECONDARY["blue2"], edgecolor="black",
               linewidth=0.5, label="redraw vs reference")
    ax.axhline(float(aris.mean()), color=TUM_PRIMARY["tum_blue"], linewidth=1.4,
               label=f"mean = {aris.mean():.4f}")

    # zoom to the data; the axis truncation and the 0.85 reference level are
    # stated in the caption rather than drawn in-axis.
    lo = float(aris.min()) - 0.005
    hi = min(1.0005, float(aris.max()) + 0.005)
    if hi - lo < 0.004:
        lo, hi = lo - 0.002, hi + 0.002
    ax.set_ylim(lo, hi)
    ax.set_xlim(0.5, n + 0.5)
    ax.set_xticks(x)

    ax.set_xlabel("stratified redraw")
    ax.set_ylabel("Adjusted Rand index vs reference partition")
    ax.set_title("Stability under resampling: agreement of independent\n"
                 "fit-population redraws with the reference partition")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    fig.tight_layout()

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fp, dpi=DPI)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    pq = ARTIFACT_DIR / "resample_stability.parquet"

    print(f"\n{'='*78}")
    print(f"  resample_stability.py  ·  {_now()}")

    if REUSE_SAVED and pq.exists():
        aris = pd.read_parquet(pq)["ari_to_reference"].to_numpy(dtype=float)
        print(f"  Loaded {len(aris)} saved replicates from {pq.name} — replot only")
        print(f"  (set REUSE_SAVED = False to force the full recompute)")
        print(f"{'='*78}")
    else:
        curves = pd.read_parquet(ARTIFACT_DIR / "kselect_curves.parquet")
        k = int(curves.loc[curves["silhouette"].idxmax(), "k"])
        print(f"  k held at {k}  ·  k-means seed {SEED}  ·  cap {CAP:,}  ·  "
              f"{N_RESAMPLES} redraws (seeds {RESEED_BASE}..{RESEED_BASE + N_RESAMPLES - 1})")
        print(f"{'='*78}")

        pop = _discover()
        with ThreadPoolExecutor(max_workers=min(8, len(pop))) as ex:
            frames = list(tqdm(ex.map(_load, pop), total=len(pop), desc="reading"))
        full = pd.concat([f for f in frames if f is not None], ignore_index=True)

        ref = _fit_reference(full, k)
        print(f"\n  Reference partition: {len(ref):,} windows labelled")

        aris = np.empty(N_RESAMPLES, dtype=float)
        for i in tqdm(range(N_RESAMPLES), desc="  redraws"):
            lab = _fit_redraw(full, k, RESEED_BASE + i)
            aris[i] = adjusted_rand_score(ref, lab)

        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"redraw_seed": [RESEED_BASE + i for i in range(N_RESAMPLES)],
                      "ari_to_reference": np.round(aris, 6)}
                     ).to_parquet(pq, index=False)

    print(f"\n  ARI vs reference over {len(aris)} redraws:")
    print(f"    mean   {aris.mean():.4f}")
    print(f"    min    {aris.min():.4f}")
    print(f"    p05    {np.percentile(aris, 5):.4f}")
    print(f"    median {np.median(aris):.4f}")
    print(f"    max    {aris.max():.4f}")

    out = OUT_DIR / "resample_stability.png"
    _plot(aris, out)

    print(f"\n  Figure → {out}")
    print(f"  Saved  → {pq}")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()