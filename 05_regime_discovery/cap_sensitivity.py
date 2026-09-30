"""
cap_sensitivity.py  —  Section 4.7.7 robustness control
────────────────────────────────────────────────────────────────────────────
Confirms the recovered structure does not depend on the fit ceiling. Re-runs the
clustering at three caps with k and the seed held fixed, and reports only
permutation-invariant evidence:

  • cross-cap ARI            — do windows keep their regime across caps?
  • per-dataset ranking ρ    — is the bias ranking of the 14 datasets preserved?
  • per-regime centroid drift — in pooled-σ units, regimes matched across caps by
                                centroid (linear_sum_assignment).

The cap=5000 run is the reference (the production cap). k is read from the
saved k-sweep curve (argmax silhouette); no model file is needed. Imports every
shared step from clustering.py so the two cannot diverge. All outputs parquet.

Reads:   dataset/regime_data/kselect_curves.parquet   (selected k)
         dataset/std_data/<dataset>.parquet
Writes:  dataset/regime_data/cap_sensitivity_caps.parquet
         dataset/regime_data/cap_sensitivity_rank.parquet
         dataset/regime_data/cap_sensitivity_drift.parquet
         figures/cap_sensitivity/cap_sensitivity_ari.png
────────────────────────────────────────────────────────────────────────────
"""

import warnings
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

# shared steps — single source of truth
from clustering import (_discover, _load, _build_fit, _order_regimes, _sil,
                        FEATURES, N_INIT, SEED, ARTIFACT_DIR, DATA_ROOT, REPO_ROOT)

warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k):
        return x

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

CAPS       = [2500, 5000, 10000]
REF_CAP    = 5000                    # reference cap (the production setting)
ARI_OK     = 0.85
RANK_OK    = 0.90
RARE_SHARE = 1.0                     # regimes below this % reported separately
ARI_VMIN   = 0.85                    # heatmap colour floor — the concern level;
                                     # values are annotated, truncation stated in caption
OUT_DIR    = REPO_ROOT / "05_regime_discovery" / "figures" / "cap_sensitivity"
DPI        = 300


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _norm_entropy(frac: np.ndarray) -> float:
    """Normalised Shannon entropy of a regime-fraction vector (0..1)."""
    p = frac[frac > 0]
    if len(p) <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / np.log(len(frac)))


def _fit_one_cap(full: pd.DataFrame, cap: int, k: int) -> dict:
    """Cluster at one cap; return ordered full-population labels, ordered physical
    centroids, regime shares, and the fit silhouette."""
    fit    = _build_fit(full, cap, SEED)
    scaler = StandardScaler().fit(fit[FEATURES])
    km     = KMeans(k, n_init=N_INIT, random_state=SEED).fit(scaler.transform(fit[FEATURES]))

    lmap = _order_regimes(km.cluster_centers_)
    reg  = np.vectorize(lmap.get)(km.predict(scaler.transform(full[FEATURES])))

    cent_old  = km.cluster_centers_ * scaler.scale_ + scaler.mean_      # physical
    cent_phys = np.zeros_like(cent_old)
    for old, new in lmap.items():
        cent_phys[new] = cent_old[old]

    shares = {r: round(100.0 * np.mean(reg == r), 1) for r in range(k)}
    sil    = _sil(scaler.transform(fit[FEATURES]), km.labels_, SEED)
    return {"fit_n": len(fit), "labels": reg, "centroids_phys": cent_phys,
            "shares": shares, "silhouette": sil}


def _plot_ari(ari: np.ndarray, caps: list[int], out_fp: Path):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    # colour floor at ARI_VMIN: on a 0-1 scale every value above 0.9 renders as
    # the same dark blue. Cells are annotated, so the truncation costs nothing
    # and is stated in the caption.
    im = ax.imshow(ari, cmap="Blues", vmin=ARI_VMIN, vmax=1.0)
    labels = [f"{c:,}" for c in caps]           # exact caps ("2,500", not "2k")
    ax.set_xticks(range(len(caps))); ax.set_yticks(range(len(caps)))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("fit cap (windows per dataset)"); ax.set_ylabel("fit cap (windows per dataset)")
    ax.set_title("Cap robustness: cross-cap assignment agreement (ARI)")
    thresh = ARI_VMIN + 0.55 * (1.0 - ARI_VMIN)
    for i in range(len(caps)):
        for j in range(len(caps)):
            ax.text(j, i, f"{ari[i, j]:.3f}", ha="center", va="center",
                    color=("white" if ari[i, j] > thresh else "black"))
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches="tight": the equal-aspect image plus the colorbar leaves the
    # y label outside the canvas under tight_layout alone
    fig.savefig(out_fp, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # selected k from the saved k-sweep curve (argmax silhouette) — no model file
    curves = pd.read_parquet(ARTIFACT_DIR / "kselect_curves.parquet")
    k = int(curves.loc[curves["silhouette"].idxmax(), "k"])

    print(f"\n{'='*78}")
    print(f"  cap_sensitivity.py  ·  {_now()}")
    print(f"  Caps: {CAPS}   ·   k held at {k}   ·   seed held at {SEED}")
    print(f"{'='*78}")

    pop = _discover()
    with ThreadPoolExecutor(max_workers=min(8, len(pop))) as ex:
        frames = list(tqdm(ex.map(_load, pop), total=len(pop), desc="reading"))
    full = pd.concat([f for f in frames if f is not None], ignore_index=True)
    pooled_sigma = full[FEATURES].std().to_numpy()
    datasets = sorted(full["dataset"].unique())

    res = {}
    for cap in CAPS:
        res[cap] = _fit_one_cap(full, cap, k)
        print(f"\n  cap={cap:>6}  fit={res[cap]['fit_n']:>7,}  silhouette={res[cap]['silhouette']:.3f}"
              f"  regime shares %: {res[cap]['shares']}")

    # ── cross-cap ARI (permutation-invariant) ─────────────────────────────────
    ari = np.ones((len(CAPS), len(CAPS)))
    for a in range(len(CAPS)):
        for b in range(a + 1, len(CAPS)):
            v = adjusted_rand_score(res[CAPS[a]]["labels"], res[CAPS[b]]["labels"])
            ari[a, b] = ari[b, a] = v
    print(f"\n  Cross-cap ARI (windows keeping their regime):")
    print("           " + "".join((f"{c//1000}k" if c >= 1000 else str(c)).rjust(9) for c in CAPS))
    for a, ca in enumerate(CAPS):
        lab = f"{ca//1000}k" if ca >= 1000 else str(ca)
        print(f"    {lab:>5}" + "".join(f"{ari[a, b]:>9.3f}" for b in range(len(CAPS))))

    # ── per-dataset bias-ranking Spearman between caps ────────────────────────
    ent = {}
    for cap in CAPS:
        dfreg = pd.DataFrame({"dataset": full["dataset"].to_numpy(), "regime": res[cap]["labels"]})
        ent[cap] = {ds: _norm_entropy(g["regime"].value_counts(normalize=True)
                                      .reindex(range(k), fill_value=0).to_numpy())
                    for ds, g in dfreg.groupby("dataset")}
    print(f"\n  Per-dataset bias-ranking Spearman between caps:")
    rank_rhos, rank_rows = [], []
    for a in range(len(CAPS)):
        for b in range(a + 1, len(CAPS)):
            va = [ent[CAPS[a]][d] for d in datasets]
            vb = [ent[CAPS[b]][d] for d in datasets]
            rho = float(spearmanr(va, vb).correlation)
            rank_rhos.append(rho)
            # persisted so the thesis table can be regenerated from the artefact
            rank_rows.append({"cap_a": CAPS[a], "cap_b": CAPS[b],
                              "ari": round(float(ari[a, b]), 4),
                              "spearman_rho": round(rho, 4)})
            print(f"    {CAPS[a]:>6} vs {CAPS[b]:>6}:  ρ = {rho:+.3f}")

    # ── per-regime centroid drift (matched to REF_CAP) ────────────────────────
    ref = res[REF_CAP]
    drift = {r: 0.0 for r in range(k)}
    for cap in CAPS:
        if cap == REF_CAP:
            continue
        ri, ci = linear_sum_assignment(cdist(res[cap]["centroids_phys"], ref["centroids_phys"]))
        for a, b in zip(ri, ci):
            d = float(np.max(np.abs(res[cap]["centroids_phys"][a] - ref["centroids_phys"][b]) / pooled_sigma))
            drift[b] = max(drift[b], d)
    print(f"\n  Per-regime profile drift (pooled-σ units, regimes matched by centroid):")
    print(f"   {'regime':>7}{'ref share %':>14}{'max drift σ':>14}")
    for r in range(k):
        print(f"   {r:>7}{ref['shares'][r]:>14.2f}{drift[r]:>14.3f}")

    # ── verdict ───────────────────────────────────────────────────────────────
    min_ari  = float(ari[np.triu_indices(len(CAPS), 1)].min())
    min_rank = float(min(rank_rhos))
    dom_drift  = max((drift[r] for r in range(k) if ref["shares"][r] >= RARE_SHARE), default=0.0)
    rare_drift = max((drift[r] for r in range(k) if ref["shares"][r] <  RARE_SHARE), default=0.0)
    ok = (min_ari >= ARI_OK) and (min_rank >= RANK_OK)
    print(f"\n  {'-'*64}")
    print(f"  VERDICT (permutation-invariant metrics):")
    print(f"    min cross-cap ARI      {min_ari:.3f}   ({'≥' if min_ari>=ARI_OK else '<'} {ARI_OK} "
          f"{'✓' if min_ari>=ARI_OK else '✗'})")
    print(f"    min ranking Spearman  {min_rank:+.3f}   ({'≥' if min_rank>=RANK_OK else '<'} {RANK_OK} "
          f"{'✓' if min_rank>=RANK_OK else '✗'})")
    print(f"  → dominant regimes + bias ranking are {'INVARIANT' if ok else 'NOT invariant'} to the cap")
    print(f"    diagnostic: dominant-regime drift {dom_drift:.2f} σ · rare-regime drift {rare_drift:.2f} σ (disclosed)")
    print(f"  {'-'*64}")

    # ── persist (parquet only) ────────────────────────────────────────────────
    _plot_ari(ari, CAPS, OUT_DIR / "cap_sensitivity_ari.png")

    ref_idx = CAPS.index(REF_CAP)
    cap_rows = []
    for i, cap in enumerate(CAPS):
        row = {"cap": cap, "fit_n": res[cap]["fit_n"],
               "silhouette": round(res[cap]["silhouette"], 4),
               "ari_to_ref": round(float(ari[i, ref_idx]), 4)}
        for r in range(k):
            row[f"share_{r}"] = res[cap]["shares"][r]
        cap_rows.append(row)
    pd.DataFrame(cap_rows).to_parquet(ARTIFACT_DIR / "cap_sensitivity_caps.parquet", index=False)

    pd.DataFrame(rank_rows).to_parquet(ARTIFACT_DIR / "cap_sensitivity_rank.parquet", index=False)

    pd.DataFrame([{"regime": r, "ref_share_pct": ref["shares"][r],
                   "max_drift_sigma": round(drift[r], 4)} for r in range(k)]
                 ).to_parquet(ARTIFACT_DIR / "cap_sensitivity_drift.parquet", index=False)

    print(f"\n  Figure → {OUT_DIR / 'cap_sensitivity_ari.png'}")
    print(f"  Saved  → {ARTIFACT_DIR / 'cap_sensitivity_caps.parquet'}")
    print(f"           {ARTIFACT_DIR / 'cap_sensitivity_rank.parquet'}")
    print(f"           {ARTIFACT_DIR / 'cap_sensitivity_drift.parquet'}")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()