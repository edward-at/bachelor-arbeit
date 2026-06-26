"""
feature_gate.py  (A2 — feature redundancy gate for the regime clustering)
────────────────────────────────────────────────────────────────────────────
Closes the design gate before any k-means is run. Reads the per-window feature
table (std_data, v4), screens the seven candidate features for redundancy
(Pearson + Spearman + VIF), and prunes to the final feature vector. Persists the
decision and the correlation figure.

Why screen at all: under a Euclidean distance in standardised space, two
near-duplicate features double the weight one underlying behaviour exerts on the
clustering. Lateral acceleration and yaw rate are physically coupled
(a_lat = v·ω) and jerk is the derivative of acceleration, so the candidate set is
deliberately over-specified and must be reduced transparently.

Population: ALL 14 qualified datasets (Section 7.3), auto-discovered from the
std_data directory. No dataset is excluded — HDD, A2D2 and 4Seasons were
re-qualified after cleaning, KITScenes is included, and DurLAR's GPS multipath is
corrected at the position level (Section 7.2), so it too is fully included.

Grain: clustering operates on NON-OVERLAPPING 20 s windows. The std_data layer is
50 % overlap (for Chapter 8); the non-overlapping subset is exactly the
grid-aligned windows, recovered here by a start-time filter — no recomputation.

Correlations and VIF are scale-invariant, so no standardisation is applied here;
the pooled StandardScaler is a clustering-time step (B1).

Outputs:
  dataset/regime_data/feature_gate.json                         (locked vector + stats)
  bachelor-arbeit/regime_discovery/figures/feature_gate/
      feature_correlation_gate.png                              (Pearson | Spearman)
────────────────────────────────────────────────────────────────────────────
"""

import json
import warnings
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

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
ARTIFACT_DIR = PROJECT_ROOT / "dataset" / "regime_data"                       # WP4 cached artifacts
OUT_DIR      = PROJECT_ROOT / "bachelor-arbeit" / "regime_discovery" / "figures" / "feature_gate"

WINDOW_S = 20.0
DPI      = 300
CMAP     = "Blues"

# All 14 datasets are qualified (Section 7.3); none excluded. Auto-discovered
# from std_data. EXCLUDE is kept only as an escape hatch and is empty by default.
EXCLUDE = set()

# Candidate feature vector (Section 9.1). std_speed_kmh is reused from the
# variability channels; the other six are the v4 means.
FEATURES = ["mean_speed_kmh", "std_speed_kmh", "mean_acc_pos", "mean_acc_neg",
            "mean_abs_a_lat", "mean_abs_yaw_rate", "mean_abs_jerk"]
LABELS   = {"mean_speed_kmh": "speed", "std_speed_kmh": "σ speed",
            "mean_acc_pos": "acc+", "mean_acc_neg": "acc−",
            "mean_abs_a_lat": "|aLat|", "mean_abs_yaw_rate": "|yaw|",
            "mean_abs_jerk": "|jerk|"}

CORR_THRESHOLD = 0.85   # |Pearson| or |Spearman| above this flags a redundant pair
VIF_THRESHOLD  = 5.0    # VIF above this flags multicollinearity


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _discover() -> list[str]:
    """All qualified datasets, discovered from the std_data directory."""
    return sorted(fp.stem for fp in STD_DATA_DIR.glob("*.parquet")
                  if fp.stem not in EXCLUDE)


def _load(name: str) -> pd.DataFrame | None:
    """Read one dataset, keep the non-overlapping (grid-aligned 20 s) subset."""
    fp = STD_DATA_DIR / f"{name}.parquet"
    if not fp.exists():
        print(f"  [WARN] {name}: {fp.name} not found")
        return None
    df = pd.read_parquet(fp, columns=["clip_id", "window_start_s"] + FEATURES)
    # Non-overlap = windows whose start, measured from the clip's first window,
    # is an integer multiple of WINDOW_S. Robust to skipped windows (uses the
    # actual start time, not the window index).
    t0 = df.groupby("clip_id")["window_start_s"].transform("min")
    k  = (df["window_start_s"] - t0) / WINDOW_S
    df = df[(k - k.round()).abs() < 1e-6].copy()
    df["dataset"] = name
    return df


def _vif(X: np.ndarray) -> list[float]:
    """Variance inflation factor per column: VIF_i = 1 / (1 - R²_i)."""
    n, p = X.shape
    out = []
    for i in range(p):
        y = X[:, i]
        Z = np.column_stack([np.ones(n), np.delete(X, i, axis=1)])
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        resid = y - Z @ beta
        ss_res = float((resid ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        out.append(float(1.0 / (1.0 - r2)) if r2 < 1.0 else float("inf"))
    return out


def _spearman_matrix(X: np.ndarray) -> np.ndarray:
    """Spearman correlation matrix (handles the 2-column scalar return)."""
    if X.shape[1] == 2:
        r = spearmanr(X[:, 0], X[:, 1]).correlation
        return np.array([[1.0, r], [r, 1.0]])
    return np.asarray(spearmanr(X).correlation)


def _prune(df_feat: pd.DataFrame) -> tuple[list[str], list[dict]]:
    """
    Greedy redundancy prune: while any |Pearson| or |Spearman| pair exceeds
    CORR_THRESHOLD or any VIF exceeds VIF_THRESHOLD, drop the most-redundant
    feature (the higher-VIF member of the worst pair, or the global max-VIF
    feature). Re-screens after every drop.
    """
    keep, log = list(df_feat.columns), []
    while len(keep) > 2:
        X = df_feat[keep].to_numpy(dtype=float)
        P = np.corrcoef(X, rowvar=False)
        S = _spearman_matrix(X)
        V = _vif(X)

        absmax = np.maximum(np.abs(P), np.abs(S))
        np.fill_diagonal(absmax, 0.0)
        i, j = map(int, np.unravel_index(absmax.argmax(), absmax.shape))
        worst_pair, worst_vif = float(absmax[i, j]), float(np.nanmax(V))

        if worst_pair <= CORR_THRESHOLD and worst_vif <= VIF_THRESHOLD:
            break

        if worst_pair > CORR_THRESHOLD:
            drop_idx = i if V[i] >= V[j] else j
            partner  = keep[j] if drop_idx == i else keep[i]
            reason   = f"|r|={worst_pair:.2f} with {LABELS.get(partner, partner)}, VIF={V[drop_idx]:.1f}"
        else:
            drop_idx = int(np.nanargmax(V))
            reason   = f"VIF={V[drop_idx]:.1f}"

        log.append({"dropped": keep.pop(drop_idx), "reason": reason})
    return keep, log


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING  (pure default matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

def _plot_correlation(P: np.ndarray, S: np.ndarray, labels: list[str], out_fp: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, M, name in [(axes[0], P, "Pearson"), (axes[1], S, "Spearman")]:
        im = ax.imshow(M, cmap=CMAP, vmin=-1.0, vmax=1.0)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_title(f"{name} correlation")
        # documented heatmap exception: a single small fontsize for in-cell values
        for a in range(M.shape[0]):
            for b in range(M.shape[1]):
                ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fp, dpi=DPI)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    pop = _discover()

    print(f"\n{'='*78}")
    print(f"  feature_gate.py  (A2)  ·  {_now()}")
    print(f"  Population ({len(pop)} datasets, all qualified): {pop}")
    print(f"  Thresholds: |r| > {CORR_THRESHOLD}  ·  VIF > {VIF_THRESHOLD}")
    print(f"{'='*78}")

    with ThreadPoolExecutor(max_workers=min(8, len(pop))) as ex:
        frames = [f for f in tqdm(ex.map(_load, pop), total=len(pop), desc="reading") if f is not None]
    if not frames:
        print("[ERROR] no population data loaded."); return

    full  = pd.concat(frames, ignore_index=True)
    n_raw = len(full)
    full  = full.dropna(subset=FEATURES).reset_index(drop=True)
    n_cc  = len(full)

    # ── population shares (imbalance check for B1) ───────────────────────────
    print(f"\n  Non-overlap complete-case windows: {n_cc:,} of {n_raw:,}")
    print(f"  {'dataset':<12}{'windows':>10}{'share':>9}")
    print(f"  {'-'*31}")
    shares = {}
    for name, g in full.groupby("dataset"):
        sh = 100.0 * len(g) / n_cc
        shares[name] = {"windows": int(len(g)), "share_pct": round(sh, 1)}
        print(f"  {name:<12}{len(g):>10,}{sh:>8.1f}%")

    # ── screen ───────────────────────────────────────────────────────────────
    Xdf = full[FEATURES]
    X   = Xdf.to_numpy(dtype=float)
    P   = np.corrcoef(X, rowvar=False)
    S   = _spearman_matrix(X)
    V   = _vif(X)

    print(f"\n  VIF (full candidate set):")
    for f, v in zip(FEATURES, V):
        flag = "  <-- > %.0f" % VIF_THRESHOLD if v > VIF_THRESHOLD else ""
        print(f"    {LABELS[f]:<8} {v:6.2f}{flag}")

    if {"mean_abs_a_lat", "mean_abs_yaw_rate"} <= set(FEATURES):
        print(f"\n  a_lat–yaw Pearson per dataset (coupling should hold everywhere):")
        for name, g in full.groupby("dataset"):
            r = g["mean_abs_a_lat"].corr(g["mean_abs_yaw_rate"])
            print(f"    {name:<12} r = {r:+.2f}")

    # ── prune ────────────────────────────────────────────────────────────────
    keep, log = _prune(Xdf)
    print(f"\n  {'-'*60}")
    if log:
        print("  Pruned:")
        for d in log:
            print(f"    drop {LABELS.get(d['dropped'], d['dropped']):<8} ({d['reason']})")
    else:
        print("  Pruned: nothing — all features below the thresholds.")
    print(f"\n  LOCKED FEATURE VECTOR ({len(keep)}): {[LABELS[f] for f in keep]}")
    print(f"    columns: {keep}")
    print(f"  {'-'*60}")

    # ── persist ──────────────────────────────────────────────────────────────
    fig_fp = OUT_DIR / "feature_correlation_gate.png"
    _plot_correlation(P, S, [LABELS[f] for f in FEATURES], fig_fp)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    art = {
        "generated":          _now(),
        "population":         pop,
        "n_windows":          n_cc,
        "dataset_shares":     shares,
        "candidate_features": FEATURES,
        "thresholds":         {"corr": CORR_THRESHOLD, "vif": VIF_THRESHOLD},
        "pearson":            {f: dict(zip(FEATURES, row)) for f, row in zip(FEATURES, P.tolist())},
        "spearman":           {f: dict(zip(FEATURES, row)) for f, row in zip(FEATURES, S.tolist())},
        "vif":                dict(zip(FEATURES, V)),
        "prune_log":          log,
        "locked_vector":      keep,
    }
    with open(ARTIFACT_DIR / "feature_gate.json", "w") as fh:
        json.dump(art, fh, indent=2)

    print(f"\n  Figure   → {fig_fp}")
    print(f"  Artifact → {ARTIFACT_DIR / 'feature_gate.json'}")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()