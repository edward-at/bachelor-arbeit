"""
explore_k_profiles.py  —  THROWAWAY exploration, not part of the pipeline.

Shows what the regime profile heatmap looks like at every k in the sweep, so the
qualitative story behind the index curves is visible: which behaviour splits off
from which as k grows.

WRITES NOTHING. No parquet, no PNG — the figure is only displayed. Delete this
file afterwards; it is not a Section 4.6 artefact and nothing downstream reads it.

Reuses the production fit population, scaler and regime ordering from
clustering.py so the profiles are directly comparable to the k=5 production run.
"""

import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")          # interactive window; comment out if it errors
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from clustering import (_discover, _load, _build_fit, _order_regimes,
                        FEATURES, N_INIT, SEED, CAP, K_RANGE)

warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k):
        return x

LABELS = {"mean_speed_kmh": "mean speed", "std_speed_kmh": "σ speed",
          "mean_acc_pos": "acc⁺", "mean_acc_neg": "acc⁻",
          "mean_abs_yaw_rate": "|yaw rate|"}
YAW_TO_DEG = True                # show yaw in °/s (more readable than rad/s)


def _fmt(v: float) -> str:
    return f"{v:.3f}" if abs(v) < 1 else f"{v:.1f}"


def main():
    pop = _discover()
    with ThreadPoolExecutor(max_workers=min(8, len(pop))) as ex:
        frames = list(tqdm(ex.map(_load, pop), total=len(pop), desc="reading"))
    full = pd.concat([f for f in frames if f is not None], ignore_index=True)

    fit    = _build_fit(full, CAP, SEED)
    scaler = StandardScaler().fit(fit[FEATURES])
    Xfit   = scaler.transform(fit[FEATURES])
    Xall   = scaler.transform(full[FEATURES])

    ks = list(K_RANGE)
    ncol = 3
    nrow = int(np.ceil(len(ks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.4 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, k in zip(axes, ks):
        km   = KMeans(k, n_init=N_INIT, random_state=SEED).fit(Xfit)
        lmap = _order_regimes(km.cluster_centers_)
        reg  = np.vectorize(lmap.get)(km.predict(Xall))

        prof = full.groupby(reg)[FEATURES].mean()
        share = np.array([100.0 * np.mean(reg == r) for r in range(k)])

        M = prof[FEATURES].to_numpy(float, copy=True)
        if YAW_TO_DEG and "mean_abs_yaw_rate" in FEATURES:
            M[:, FEATURES.index("mean_abs_yaw_rate")] *= 180.0 / np.pi

        sd = M.std(0); sd[sd == 0] = 1.0
        Z = (M - M.mean(0)) / sd                      # standardise per feature

        im = ax.imshow(Z, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(FEATURES)))
        ax.set_xticklabels([LABELS.get(f, f) for f in FEATURES], rotation=30,
                           ha="right", fontsize=7)
        ax.set_yticks(range(k))
        ax.set_yticklabels([f"{r} ({share[r]:.0f}%)" for r in range(k)], fontsize=7)
        ax.set_title(f"k = {k}", fontsize=10)
        for i in range(k):
            for j in range(len(FEATURES)):
                ax.text(j, i, _fmt(M[i, j]), ha="center", va="center", fontsize=6)

    for ax in axes[len(ks):]:
        ax.axis("off")

    yaw_note = "yaw in °/s" if YAW_TO_DEG else "yaw in rad/s"
    fig.suptitle(f"Regime feature profiles across the k sweep "
                 f"(cell = mean in physical units, {yaw_note}; "
                 f"colour = standardised across regimes; row label = regime (share))")
    fig.tight_layout()
    plt.show()          # display only — nothing is written to disk


if __name__ == "__main__":
    main()