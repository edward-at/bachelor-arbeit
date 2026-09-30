"""
validity.py  —  Sections 4.7.8–4.7.9, internal validity indices and cluster stability
────────────────────────────────────────────────────────────────────────────
Hardens the k-means partition of Section 4.7.5 with evidence beyond the
silhouette and the global ARIs of Section 4.7.7, all on the same FIT population
in the same full standardised feature space:

  • Internal indices over the k-sweep — the gap statistic (departure of the
    within-cluster dispersion from a no-structure null), Davies–Bouldin (lower =
    compact, separated), Calinski–Harabasz (higher = the same). Three criteria
    built on different principles; their agreement on a k corroborates it.
  • Per-regime bootstrap-Jaccard — resample the fit population, refit, match each
    reference regime to its most-overlapping refit regime by Jaccard. Localises
    robustness to the single regime (Hennig: ≥0.85 highly stable, ≥0.75 stable,
    <0.60 not trustworthy). The rare change-regimes are expected least stable.
  • Consensus matrix — over many resamples, the fraction of runs each window pair
    is co-assigned. Block-diagonal once ordered by regime if the structure holds.

k is read from the saved k-sweep curve (argmax silhouette); no model file is
needed. Every shared step is imported from clustering.py so the two cannot
diverge. All outputs parquet.

Reads:   dataset/regime_data/kselect_curves.parquet   (selected k)
         dataset/regime_data/feature_gate_vector.parquet (via clustering)
         dataset/std_data/<dataset>.parquet
Writes:  dataset/regime_data/validity_indices.parquet
         dataset/regime_data/bootstrap_jaccard.parquet
         dataset/regime_data/consensus_matrix.parquet
         dataset/regime_data/consensus_rows.parquet
────────────────────────────────────────────────────────────────────────────
"""

import warnings
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

# shared steps — single source of truth
from clustering import (_discover, _load, _build_fit, _order_regimes,
                        FEATURES, N_INIT, SEED, K_RANGE, CAP, ARTIFACT_DIR)

warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k):
        return x

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

B_GAP        = 10        # uniform reference datasets per k for the gap statistic
B_BOOT       = 50        # bootstrap resamples for per-regime Jaccard
B_CONS       = 50        # resamples for the consensus matrix
CONSENSUS_N  = 800       # window subsample the consensus matrix is built on
JACCARD_OK   = 0.75      # Hennig: ≥0.75 stable
JACCARD_HIGH = 0.85      # Hennig: ≥0.85 highly stable


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _within_dispersion(X: np.ndarray, labels: np.ndarray) -> float:
    """Pooled within-cluster sum of squares (Tibshirani's W_k)."""
    return float(sum(((X[labels == c] - X[labels == c].mean(0)) ** 2).sum()
                     for c in np.unique(labels)))


def _gap(X: np.ndarray, k: int, b: int, seed: int) -> tuple[float, float]:
    """Gap statistic at k against b uniform references over the data's box."""
    km = KMeans(k, n_init=N_INIT, random_state=seed).fit(X)
    logW = np.log(_within_dispersion(X, km.labels_) + 1e-12)
    lo, hi = X.min(0), X.max(0)
    rng = np.random.RandomState(seed)
    logWb = np.empty(b)
    for i in range(b):
        Xb = rng.uniform(lo, hi, size=X.shape)
        kmb = KMeans(k, n_init=1, random_state=seed).fit(Xb)
        logWb[i] = np.log(_within_dispersion(Xb, kmb.labels_) + 1e-12)
    gap = float(logWb.mean() - logW)
    sd  = float(logWb.std() * np.sqrt(1.0 + 1.0 / b))
    return gap, sd


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def _bootstrap_jaccard(X: np.ndarray, ref: np.ndarray, k: int, b: int, seed: int) -> dict:
    """Per-regime stability: resample, refit, classify all originals, take each
    reference regime's best Jaccard against a refit regime."""
    rng = np.random.RandomState(seed)
    n = len(X)
    per = {r: [] for r in range(k)}
    for _ in tqdm(range(b), desc="  bootstrap"):
        idx = rng.randint(0, n, n)
        kmb = KMeans(k, n_init=1, random_state=seed).fit(X[idx])
        pred = kmb.predict(X)
        for r in range(k):
            per[r].append(max(_jaccard(ref == r, pred == s) for s in range(k)))
    return {r: np.asarray(v) for r, v in per.items()}


def _consensus(X: np.ndarray, ref: np.ndarray, k: int, n_sub: int, b: int, seed: int):
    """Co-assignment frequency over resamples on a fixed window subsample,
    ordered by reference regime so a genuine partition is block-diagonal."""
    rng = np.random.RandomState(seed)
    n = len(X)
    sub = rng.choice(n, min(n_sub, n), replace=False)
    M = np.zeros((len(sub), len(sub)))
    for _ in tqdm(range(b), desc="  consensus"):
        idx = rng.randint(0, n, n)
        lab = KMeans(k, n_init=1, random_state=seed).fit(X[idx]).predict(X[sub])
        M += (lab[:, None] == lab[None, :]).astype(float)
    M /= b
    order = np.argsort(ref[sub])
    return M[np.ix_(order, order)], ref[sub][order]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    curves = pd.read_parquet(ARTIFACT_DIR / "kselect_curves.parquet")
    k = int(curves.loc[curves["silhouette"].idxmax(), "k"])

    print(f"\n{'='*78}")
    print(f"  validity.py  ·  {_now()}")
    print(f"  k held at {k}   ·   seed {SEED}   ·   gap B={B_GAP} · boot B={B_BOOT} · consensus B={B_CONS}")
    print(f"{'='*78}")

    pop = _discover()
    with ThreadPoolExecutor(max_workers=min(8, len(pop))) as ex:
        frames = list(tqdm(ex.map(_load, pop), total=len(pop), desc="reading"))
    full = pd.concat([f for f in frames if f is not None], ignore_index=True)

    fit    = _build_fit(full, CAP, SEED)
    scaler = StandardScaler().fit(fit[FEATURES])
    Xfit   = scaler.transform(fit[FEATURES])

    # reference partition (ordered 0 = calmest), identical to clustering.py
    km   = KMeans(k, n_init=N_INIT, random_state=SEED).fit(Xfit)
    lmap = _order_regimes(km.cluster_centers_)
    ref  = np.vectorize(lmap.get)(km.labels_)
    print(f"\n  Fit set: {len(Xfit):,} windows")

    # ── internal indices over the k-sweep ─────────────────────────────────────
    print(f"\n  Internal validity indices over k:")
    print(f"    {'k':>3}{'gap':>9}{'gap_sd':>9}{'davies_bouldin':>16}{'calinski_harabasz':>19}")
    rows = []
    for kk in tqdm(list(K_RANGE), desc="  indices"):
        lab = KMeans(kk, n_init=N_INIT, random_state=SEED).fit_predict(Xfit)
        gap, sd = _gap(Xfit, kk, B_GAP, SEED)
        db = float(davies_bouldin_score(Xfit, lab))
        ch = float(calinski_harabasz_score(Xfit, lab))
        rows.append({"k": kk, "gap": gap, "gap_sd": sd,
                     "davies_bouldin": db, "calinski_harabasz": ch})
    idx_df = pd.DataFrame(rows)
    for _, r in idx_df.iterrows():
        print(f"    {int(r['k']):>3}{r['gap']:>9.3f}{r['gap_sd']:>9.3f}"
              f"{r['davies_bouldin']:>16.3f}{r['calinski_harabasz']:>19.0f}")
    print(f"    gap argmax k={int(idx_df.loc[idx_df['gap'].idxmax(),'k'])} · "
          f"DB argmin k={int(idx_df.loc[idx_df['davies_bouldin'].idxmin(),'k'])} · "
          f"CH argmax k={int(idx_df.loc[idx_df['calinski_harabasz'].idxmax(),'k'])} "
          f"(silhouette selected k={k})")

    # ── per-regime bootstrap-Jaccard ──────────────────────────────────────────
    print(f"\n  Per-regime bootstrap-Jaccard stability:")
    bj = _bootstrap_jaccard(Xfit, ref, k, B_BOOT, SEED)
    jac_rows = []
    print(f"    {'regime':>7}{'share %':>10}{'mean J':>10}{'min J':>9}  verdict")
    for r in range(k):
        m, lo, sd = float(bj[r].mean()), float(bj[r].min()), float(bj[r].std())
        share = 100.0 * np.mean(ref == r)
        verdict = ("highly stable" if m >= JACCARD_HIGH else
                   "stable" if m >= JACCARD_OK else "TENTATIVE")
        jac_rows.append({"regime": r, "share_pct": round(share, 2),
                         "mean_jaccard": round(m, 4), "min_jaccard": round(lo, 4),
                         "std_jaccard": round(sd, 4)})
        print(f"    {r:>7}{share:>10.2f}{m:>10.3f}{lo:>9.3f}  {verdict}")

    # ── consensus matrix ──────────────────────────────────────────────────────
    print(f"\n  Consensus matrix on {min(CONSENSUS_N, len(Xfit))} windows:")
    C, rowreg = _consensus(Xfit, ref, k, CONSENSUS_N, B_CONS, SEED)
    offdiag = (C.sum() - np.trace(C)) / (C.size - len(C))
    print(f"    diagonal mean {np.diag(C).mean():.3f} · off-diagonal mean {offdiag:.3f}")

    # ── persist (parquet only) ────────────────────────────────────────────────
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    idx_df.to_parquet(ARTIFACT_DIR / "validity_indices.parquet", index=False)
    pd.DataFrame(jac_rows).to_parquet(ARTIFACT_DIR / "bootstrap_jaccard.parquet", index=False)
    pd.DataFrame(C, columns=[str(i) for i in range(C.shape[1])]).to_parquet(
        ARTIFACT_DIR / "consensus_matrix.parquet", index=False)
    pd.DataFrame({"regime": rowreg.astype(int)}).to_parquet(
        ARTIFACT_DIR / "consensus_rows.parquet", index=False)

    print(f"\n  Saved → {ARTIFACT_DIR}")
    print(f"          (validity_indices · bootstrap_jaccard · consensus_matrix · consensus_rows)")
    print(f"  Reproduce: rerun under seed {SEED}.")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()