"""
clustering.py  —  Sections 4.6.3–4.6.5, dynamic regime discovery via clustering
────────────────────────────────────────────────────────────────────────────
Recovers the dynamic regimes by k-means on the locked feature vector, split into
two populations:

  FIT    — every dataset capped to a common ceiling by STRATIFIED sampling, so
           no single dataset dominates the regime definition. The scaler, the
           PCA projection (visualisation only), and the cluster centroids are
           learned here.
  ASSIGN — all non-overlapping windows. Every window is labelled by the fixed
           centroids; this is the population Chapter 5 measures.

Capping the FIT population makes the regimes a property of driving in general
rather than of the largest dataset, and is the precondition for the unbiased
bias measurement of Chapter 5. The cap NEVER discards data from the measurement.

k is selected by maximum silhouette, each silhouette compared against a null
baseline (each feature independently shuffled). Clustering runs on the FULL
standardised vector; PCA is for the scatter only.

Run ONCE; downstream figures and the bias measurement of Chapter 5 read the
saved parquet layer, never re-fit. Reproducibility rests on the fixed seed:
rerunning under the same seed regenerates the identical partition — no model
file is persisted, and none is needed.

Reads:   dataset/regime_data/feature_gate_vector.parquet   (locked vector)
         dataset/std_data/<dataset>.parquet                 (non-overlap windows)
Writes:  dataset/regime_data/cluster_assignments.parquet
         dataset/regime_data/cluster_profile.parquet
         dataset/regime_data/kselect_curves.parquet
────────────────────────────────────────────────────────────────────────────
"""

import warnings
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k):
        return x

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS  (edit here only)
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path("/Users/edward/Downloads")
STD_DATA_DIR = PROJECT_ROOT / "dataset" / "std_data"
ARTIFACT_DIR = PROJECT_ROOT / "dataset" / "regime_data"

WINDOW_S          = 20.0
CAP               = 5000          # per-dataset ceiling on the FIT population
N_STRATA          = 10            # intensity strata for the stratified cap
N_INIT            = 10            # k-means restarts
SEED              = 42
K_RANGE           = range(2, 11)  # k sweep 2..10
SIL_SAMPLE        = 10000         # subsample for silhouette (O(n^2) otherwise)
N_STABILITY_SEEDS = 5             # reseeds for the seed-ARI stability check

BACKREF = ["dataset", "clip_id", "window_start_s", "window_end_s"]

# The locked vector is read from the screen artefact (parquet) — never hard-coded.
FEATURES = pd.read_parquet(ARTIFACT_DIR / "feature_gate_vector.parquet")["feature"].tolist()
# Intensity = how much is happening, excluding how fast: all change features.
INTENSITY_FEATURES = [f for f in FEATURES if f != "mean_speed_kmh"]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS  (imported by cap_sensitivity.py)
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _discover() -> list[str]:
    return sorted(p.stem for p in STD_DATA_DIR.glob("*.parquet"))


def _load(name: str) -> pd.DataFrame | None:
    """One dataset's non-overlapping, complete-case windows with back-reference."""
    fp = STD_DATA_DIR / f"{name}.parquet"
    if not fp.exists():
        print(f"  [WARN] {name}: not found"); return None
    df = pd.read_parquet(fp, columns=["clip_id", "window_start_s", "window_end_s"] + FEATURES)
    t0 = df.groupby("clip_id")["window_start_s"].transform("min")
    k  = (df["window_start_s"] - t0) / WINDOW_S
    df = df[(k - k.round()).abs() < 1e-6].dropna(subset=FEATURES).copy()
    df["dataset"] = name
    return df


def _intensity_score(df: pd.DataFrame) -> pd.Series:
    """Per-window dynamic intensity: mean of within-dataset percentile ranks of
    the change features (mean speed excluded)."""
    return df[INTENSITY_FEATURES].rank(pct=True).mean(axis=1)


def _stratified_cap(df: pd.DataFrame, cap: int, n_strata: int, seed: int) -> pd.DataFrame:
    """Cap a dataset to `cap` rows by drawing evenly across intensity strata, so
    rare high-dynamic windows survive at the same rate as abundant calm ones."""
    if len(df) <= cap:
        return df
    strata = pd.qcut(_intensity_score(df), n_strata, labels=False, duplicates="drop")
    rng = np.random.RandomState(seed)
    per = max(1, cap // n_strata)
    picks = [g.sample(min(len(g), per), random_state=rng) for _, g in df.groupby(strata)]
    out = pd.concat(picks)
    if len(out) < cap:
        rem = df.drop(out.index)
        extra = min(len(rem), cap - len(out))
        if extra > 0:
            out = pd.concat([out, rem.sample(extra, random_state=rng)])
    return out


def _build_fit(full: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    return pd.concat([_stratified_cap(g, cap, N_STRATA, seed)
                      for _, g in full.groupby("dataset")], ignore_index=True)


def _order_regimes(centroids_scaled: np.ndarray) -> dict:
    """Relabel regimes deterministically by ascending dynamic intensity, so
    regime 0 is the calmest. Intensity compared in scaled (z-score) space."""
    idx = [FEATURES.index(f) for f in INTENSITY_FEATURES]
    intensity = centroids_scaled[:, idx].mean(axis=1)
    order = list(np.argsort(intensity))
    return {int(old): new for new, old in enumerate(order)}


def _sil(X: np.ndarray, labels: np.ndarray, seed: int) -> float:
    return float(silhouette_score(X, labels, sample_size=min(SIL_SAMPLE, len(X)),
                                  random_state=seed))


def _null_silhouette(X: np.ndarray, k: int, seed: int) -> float:
    """Silhouette of k-means on column-shuffled data: structure destroyed,
    marginals preserved. The baseline a real silhouette must beat."""
    rng = np.random.RandomState(seed)
    Xn = X.copy()
    for c in range(Xn.shape[1]):
        Xn[:, c] = Xn[rng.permutation(len(Xn)), c]
    lab = KMeans(k, n_init=N_INIT, random_state=seed).fit_predict(Xn)
    return _sil(Xn, lab, seed)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    pop = _discover()
    print(f"\n{'='*78}")
    print(f"  clustering.py  ·  {_now()}")
    print(f"  Features ({len(FEATURES)}): {FEATURES}")
    print(f"  Cap: {CAP}/dataset · {N_STRATA} intensity strata · seed {SEED}")
    print(f"{'='*78}")

    with ThreadPoolExecutor(max_workers=min(8, len(pop))) as ex:
        frames = list(tqdm(ex.map(_load, pop), total=len(pop), desc="reading"))
    full = pd.concat([f for f in frames if f is not None], ignore_index=True)

    fit = _build_fit(full, CAP, SEED)
    print(f"\n  Assign set: {len(full):,} windows   ·   Fit set (capped): {len(fit):,} windows")
    print(f"  {'dataset':<12}{'assign':>10}{'fit':>9}")
    print(f"  {'-'*31}")
    fit_counts = fit["dataset"].value_counts()
    for name, g in full.groupby("dataset"):
        print(f"  {name:<12}{len(g):>10,}{int(fit_counts.get(name, 0)):>9,}")

    # ── fit scaler + PCA (viz) on the FIT population ──────────────────────────
    scaler = StandardScaler().fit(fit[FEATURES])
    Xfit   = scaler.transform(fit[FEATURES])
    pca    = PCA(n_components=2, random_state=SEED).fit(Xfit)
    evr    = float(pca.explained_variance_ratio_[:2].sum())

    # ── k sweep with null-baseline silhouette gap ─────────────────────────────
    rows = []
    for k in K_RANGE:
        km   = KMeans(k, n_init=N_INIT, random_state=SEED).fit(Xfit)
        sil  = _sil(Xfit, km.labels_, SEED)
        null = _null_silhouette(Xfit, k, SEED)
        rows.append({"k": k, "inertia": float(km.inertia_),
                     "silhouette": sil, "null": null, "gap": sil - null})
    curves = pd.DataFrame(rows)
    k_sel  = int(curves.loc[curves["silhouette"].idxmax(), "k"])

    print(f"\n  k sweep (PC1+PC2 explain {evr*100:.1f}% of variance):")
    print(f"    {'k':>3}{'inertia':>12}{'silhouette':>13}{'null':>9}{'gap':>8}")
    for _, r in curves.iterrows():
        print(f"    {int(r['k']):>3}{r['inertia']:>12,.0f}{r['silhouette']:>13.3f}"
              f"{r['null']:>9.3f}{r['gap']:>8.3f}")
    print(f"\n  Selected k = {k_sel}  (max silhouette)")

    # ── final centroids + stability ───────────────────────────────────────────
    km = KMeans(k_sel, n_init=N_INIT, random_state=SEED).fit(Xfit)

    seed_aris = [adjusted_rand_score(
        km.labels_, KMeans(k_sel, n_init=N_INIT, random_state=SEED + s).fit_predict(Xfit))
        for s in range(1, N_STABILITY_SEEDS + 1)]

    fit2 = _build_fit(full, CAP, SEED + 100)               # redraw the cap
    sc2  = StandardScaler().fit(fit2[FEATURES])
    km2  = KMeans(k_sel, n_init=N_INIT, random_state=SEED).fit(sc2.transform(fit2[FEATURES]))
    cap_ari = adjusted_rand_score(km.predict(scaler.transform(full[FEATURES])),
                                  km2.predict(sc2.transform(full[FEATURES])))
    print(f"  Stability: seed-ARI mean {np.mean(seed_aris):.3f} (min {np.min(seed_aris):.3f})"
          f" · cap-reseed ARI {cap_ari:.3f}")

    # ── assign ALL windows + deterministic regime ordering ────────────────────
    label_map = _order_regimes(km.cluster_centers_)
    Xall = scaler.transform(full[FEATURES])
    raw  = km.predict(Xall)
    full["regime"] = np.vectorize(label_map.get)(raw)
    pcs = pca.transform(Xall)
    full["pc1"], full["pc2"] = pcs[:, 0], pcs[:, 1]

    # ── per-regime profile (physical units, ASSIGN population) ────────────────
    prof = (full.groupby("regime")[FEATURES].mean()
            .assign(n_windows=full.groupby("regime").size())
            .assign(share_pct=lambda d: (d["n_windows"] / len(full) * 100).round(2))
            .sort_index())
    print(f"\n  Regimes (0 = calmest), n={len(full):,}:")
    print(prof.round(3).to_string())

    # ── persist (parquet only) ────────────────────────────────────────────────
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    full[BACKREF + ["regime", "pc1", "pc2"]].to_parquet(
        ARTIFACT_DIR / "cluster_assignments.parquet", index=False)
    prof.reset_index().to_parquet(ARTIFACT_DIR / "cluster_profile.parquet", index=False)
    curves.to_parquet(ARTIFACT_DIR / "kselect_curves.parquet", index=False)

    print(f"\n  Saved → {ARTIFACT_DIR}  (cluster_assignments · cluster_profile · kselect_curves)")
    print(f"  Reproduce: rerun under seed {SEED}.")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()