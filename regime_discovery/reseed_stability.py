"""
reseed_stability.py  —  Section 4.6.6 reseeding control
────────────────────────────────────────────────────────────────────────────
Figure + persisted distribution for the reproducibility-under-reseeding check:
k-means is refitted N_RESEEDS times on the FIXED fit population, changing only
the initialisation seed, and every refit is compared to the reference partition
by the adjusted Rand index. A high distribution — in particular a high minimum,
so that even the worst start agrees — shows the regimes are a property of the
data rather than of any particular initialisation.

DESIGN (locked):
  • Reseed-only: the fit population is built ONCE (CAP under SEED) and held
    fixed; only KMeans(random_state=...) varies. This isolates initialisation
    luck from sampling luck (the latter is resample_stability.py).
  • Comparison on the fit population itself (labels_ vs fit_predict): every
    reseed clusters the SAME windows, so the labellings compare directly — no
    full-corpus projection is needed. This matches the in-clustering seed-ARI
    check, and with seeds SEED+1..SEED+N the first five replicate exactly the
    values clustering.py prints.
  • Permutation-invariant throughout (ARI), so cluster numbering never matters.
  • FIGURE: one point per reseed on a truncated ARI axis with the mean as a
    horizontal line — every reseed individually visible. The truncation and
    the 0.85 reference level are stated in the caption, not drawn in-axis.
  • REUSE_SAVED: when the persisted distribution already exists, the script
    replots from it without recomputing (set False to force the full rerun).

Shared steps are imported from clustering.py so the two cannot diverge.
k is read from the saved k-sweep curve (argmax silhouette); no model file.

Reads:   dataset/regime_data/kselect_curves.parquet   (selected k)
         dataset/std_data/<dataset>.parquet           (via clustering._load)
Writes:  dataset/regime_data/reseed_stability.parquet (the ARI distribution)
Output:  figures/cap_sensitivity/reseed_stability.png
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
from clustering import (_discover, _load, _build_fit,
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

N_RESEEDS   = 20          # reseeds SEED+1 .. SEED+N_RESEEDS
                          # (first five = the seeds of clustering.py's printed check)
REUSE_SAVED = True        # replot from the saved parquet if it exists (no refitting)
OUT_DIR     = PROJECT_ROOT / "bachelor-arbeit" / "regime_discovery" / "figures" / "cap_sensitivity"
DPI         = 300


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE  (same construction as resample_stability.py, for visual parallelism)
# ─────────────────────────────────────────────────────────────────────────────

def _plot(aris: np.ndarray, out_fp: Path):
    n = len(aris)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    x = np.arange(1, n + 1)
    ax.scatter(x, aris, color=TUM_SECONDARY["blue2"], edgecolor="black",
               linewidth=0.5, label="reseed vs reference")
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

    ax.set_xlabel("k-means reseed")
    ax.set_ylabel("Adjusted Rand index vs reference partition")
    ax.set_title("Reproducibility under reseeding: agreement of independent\n"
                 "k-means initialisations with the reference partition")
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
    pq = ARTIFACT_DIR / "reseed_stability.parquet"

    print(f"\n{'='*78}")
    print(f"  reseed_stability.py  ·  {_now()}")

    if REUSE_SAVED and pq.exists():
        aris = pd.read_parquet(pq)["ari_to_reference"].to_numpy(dtype=float)
        print(f"  Loaded {len(aris)} saved reseeds from {pq.name} — replot only")
        print(f"  (set REUSE_SAVED = False to force the full recompute)")
        print(f"{'='*78}")
    else:
        curves = pd.read_parquet(ARTIFACT_DIR / "kselect_curves.parquet")
        k = int(curves.loc[curves["silhouette"].idxmax(), "k"])
        print(f"  k held at {k}  ·  fit population fixed (cap {CAP:,}, seed {SEED})  ·  "
              f"{N_RESEEDS} reseeds (seeds {SEED + 1}..{SEED + N_RESEEDS})")
        print(f"{'='*78}")

        pop = _discover()
        with ThreadPoolExecutor(max_workers=min(8, len(pop))) as ex:
            frames = list(tqdm(ex.map(_load, pop), total=len(pop), desc="reading"))
        full = pd.concat([f for f in frames if f is not None], ignore_index=True)

        # the fit population and reference clustering — built once, held fixed
        fit    = _build_fit(full, CAP, SEED)
        scaler = StandardScaler().fit(fit[FEATURES])
        Xfit   = scaler.transform(fit[FEATURES])
        ref    = KMeans(k, n_init=N_INIT, random_state=SEED).fit(Xfit)
        print(f"\n  Fit population: {len(fit):,} windows  ·  reference fitted")

        aris = np.empty(N_RESEEDS, dtype=float)
        for i in tqdm(range(N_RESEEDS), desc="  reseeds"):
            lab = KMeans(k, n_init=N_INIT, random_state=SEED + 1 + i).fit_predict(Xfit)
            aris[i] = adjusted_rand_score(ref.labels_, lab)      # ← only the seed varied

        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"reseed_seed": [SEED + 1 + i for i in range(N_RESEEDS)],
                      "ari_to_reference": np.round(aris, 6)}
                     ).to_parquet(pq, index=False)

    print(f"\n  ARI vs reference over {len(aris)} reseeds:")
    print(f"    mean   {aris.mean():.4f}")
    print(f"    min    {aris.min():.4f}")
    print(f"    p05    {np.percentile(aris, 5):.4f}")
    print(f"    median {np.median(aris):.4f}")
    print(f"    max    {aris.max():.4f}")

    out = OUT_DIR / "reseed_stability.png"
    _plot(aris, out)

    print(f"\n  Figure → {out}")
    print(f"  Saved  → {pq}")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()