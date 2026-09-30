"""
gmm.py  —  Section 4.6.8, probabilistic regime model (Gaussian mixture)
────────────────────────────────────────────────────────────────────────────
Cross-checks the k-means partition with a Gaussian mixture, which relaxes the
three k-means assumptions (hard membership, spherical regimes, equal spread) and
of which k-means is the hard-isotropic special case. Two products:

  • a BIC/AIC sweep over the component count — a generative route to the number
    of regimes, independent of the silhouette;
  • soft membership — each window's posterior over regimes, whose entropy marks
    the boundary windows where the regimes interpenetrate, plus the component
    covariances for the ellipse overlay; and the ARI of the GMM hard labels
    against k-means, quantifying that the two recover the same regimes.

Fitted within the SAME fit/assign firewall as the clustering: the mixture is
estimated on the capped FIT population and applied to ALL windows, so the
probabilistic regime definition is not the largest dataset's either. Components
are ordered by the same intensity ordering as the k-means regimes.

BOUNDARY: this characterises the POOLED corpus (where regimes overlap). Which
DATASETS concentrate in the ambiguous boundary regions is a per-dataset
measurement and belongs to Section 4.7, not here.

k is read from the saved k-sweep curve (argmax silhouette); no model file.

Reads:   dataset/regime_data/kselect_curves.parquet
         dataset/regime_data/cluster_assignments.parquet   (k-means regime, pc1, pc2)
         dataset/std_data/<dataset>.parquet
Writes:  dataset/regime_data/gmm_selection.parquet
         dataset/regime_data/gmm_assignments.parquet
         dataset/regime_data/gmm_components_2d.parquet
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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

# shared steps — single source of truth
from clustering import (_discover, _load, _build_fit, _order_regimes, _pin_pca_signs,
                        FEATURES, N_INIT, SEED, K_RANGE, CAP, BACKREF, ARTIFACT_DIR)

warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k):
        return x

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

N_RANGE     = K_RANGE     # component-count sweep — same range as the k-sweep
N_INIT_GMM  = 5           # GMM restarts
COV_TYPE    = "full"      # elliptical, unequal-covariance components


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _order_map_to_array(lmap: dict, k: int) -> np.ndarray:
    """Vector mapping raw component index -> ordered regime index."""
    a = np.empty(k, dtype=int)
    for old, new in lmap.items():
        a[old] = new
    return a


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    curves = pd.read_parquet(ARTIFACT_DIR / "kselect_curves.parquet")
    k = int(curves.loc[curves["silhouette"].idxmax(), "k"])

    print(f"\n{'='*78}")
    print(f"  gmm.py  ·  {_now()}")
    print(f"  k held at {k}   ·   covariance '{COV_TYPE}'   ·   seed {SEED}")
    print(f"{'='*78}")

    pop = _discover()
    with ThreadPoolExecutor(max_workers=min(8, len(pop))) as ex:
        frames = list(tqdm(ex.map(_load, pop), total=len(pop), desc="reading"))
    full = pd.concat([f for f in frames if f is not None], ignore_index=True)

    fit    = _build_fit(full, CAP, SEED)
    scaler = StandardScaler().fit(fit[FEATURES])
    Xfit   = scaler.transform(fit[FEATURES])
    Xall   = scaler.transform(full[FEATURES])
    pca    = PCA(n_components=3, random_state=SEED).fit(Xfit)   # viz only (same as clustering)
    pca    = _pin_pca_signs(pca)                                # same orientation pin as clustering
    W2     = pca.components_[:2]                                # (2,5) loading for the plane

    # k-means reference centroids (ordered), to align GMM components and compare
    km    = KMeans(k, n_init=N_INIT, random_state=SEED).fit(Xfit)
    klmap = _order_regimes(km.cluster_centers_)
    kcent = np.zeros_like(km.cluster_centers_)
    for old, new in klmap.items():
        kcent[new] = km.cluster_centers_[old]

    # ── component-count selection (BIC / AIC) ─────────────────────────────────
    print(f"\n  Component-count sweep (BIC / AIC):")
    print(f"    {'n':>3}{'BIC':>14}{'AIC':>14}{'loglik':>14}")
    rows = []
    for n in tqdm(list(N_RANGE), desc="  gmm sweep"):
        g = GaussianMixture(n, covariance_type=COV_TYPE, n_init=N_INIT_GMM,
                            random_state=SEED).fit(Xfit)
        rows.append({"n": n, "bic": float(g.bic(Xfit)), "aic": float(g.aic(Xfit)),
                     "loglik": float(g.score(Xfit) * len(Xfit))})
    sel = pd.DataFrame(rows)
    n_bic = int(sel.loc[sel["bic"].idxmin(), "n"])
    for _, r in sel.iterrows():
        print(f"    {int(r['n']):>3}{r['bic']:>14.0f}{r['aic']:>14.0f}{r['loglik']:>14.0f}")
    print(f"    BIC selects n={n_bic}  (silhouette selected k={k})")

    # ── final mixture at k, applied to ALL windows ────────────────────────────
    gmm = GaussianMixture(k, covariance_type=COV_TYPE, n_init=N_INIT_GMM,
                          random_state=SEED).fit(Xfit)
    # order components to the k-means intensity ordering, then align to k-means id
    glmap = _order_regimes(gmm.means_)
    g2ord = _order_map_to_array(glmap, k)
    means_ord = np.zeros_like(gmm.means_)
    cov_ord   = np.zeros_like(gmm.covariances_)
    for old in range(k):
        means_ord[g2ord[old]] = gmm.means_[old]
        cov_ord[g2ord[old]]   = gmm.covariances_[old]
    # final identity match to k-means centroids (guards against an intensity tie)
    ri, ci = linear_sum_assignment(cdist(means_ord, kcent))
    perm = np.empty(k, dtype=int); perm[ri] = ci      # ordered-GMM -> k-means regime
    means_ord = means_ord[np.argsort(perm)]
    cov_ord   = cov_ord[np.argsort(perm)]

    proba_raw = gmm.predict_proba(Xall)               # (n, k) in raw component order
    proba = proba_raw[:, np.argsort(g2ord)]           # to intensity order
    proba = proba[:, np.argsort(perm)]                # to k-means regime id
    hard = proba.argmax(1)
    ent  = -(proba * np.log(proba + 1e-12)).sum(1) / np.log(k)

    # compare to k-means labels for the same windows (merge on the back-reference)
    ca = pd.read_parquet(ARTIFACT_DIR / "cluster_assignments.parquet",
                         columns=BACKREF + ["regime", "pc1", "pc2"])
    asg = full[BACKREF].copy()
    asg["gmm_regime"] = hard
    asg["max_posterior"] = proba.max(1)
    asg["entropy"] = ent
    asg = asg.merge(ca, on=BACKREF, how="left")
    ari = adjusted_rand_score(asg["regime"].to_numpy(), asg["gmm_regime"].to_numpy())
    print(f"\n  Soft assignment: membership entropy in [{ent.min():.3f}, {ent.max():.3f}]"
          f" · mean {ent.mean():.3f}")
    print(f"  Correspondence to k-means: ARI = {ari:.3f}")

    # ── component means + covariances projected to the PC1/PC2 plane ───────────
    comp_rows = []
    for r in range(k):
        m2 = pca.transform(means_ord[r:r+1])[0, :2]
        c2 = W2 @ cov_ord[r] @ W2.T
        comp_rows.append({"regime": r, "mean_pc1": float(m2[0]), "mean_pc2": float(m2[1]),
                          "cov00": float(c2[0, 0]), "cov01": float(c2[0, 1]),
                          "cov11": float(c2[1, 1])})

    # ── persist (parquet only) ────────────────────────────────────────────────
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    sel.assign(ari_to_kmeans=round(float(ari), 4), selected_k=k, bic_selected_n=n_bic
               ).to_parquet(ARTIFACT_DIR / "gmm_selection.parquet", index=False)
    asg[BACKREF + ["gmm_regime", "max_posterior", "entropy", "pc1", "pc2"]].to_parquet(
        ARTIFACT_DIR / "gmm_assignments.parquet", index=False)
    pd.DataFrame(comp_rows).to_parquet(ARTIFACT_DIR / "gmm_components_2d.parquet", index=False)

    print(f"\n  Saved → {ARTIFACT_DIR}")
    print(f"          (gmm_selection · gmm_assignments · gmm_components_2d)")
    print(f"  Reproduce: rerun under seed {SEED}.")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()