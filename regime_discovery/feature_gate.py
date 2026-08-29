"""
feature_gate.py  —  Section 4.6.2, the feature redundancy screen
────────────────────────────────────────────────────────────────────────────
Locks the clustering feature vector before any k-means is run. Reads the
per-window feature table (std_data, non-overlapping since 2026-08-29), keeps the windows of
all fourteen qualified datasets, screens the candidate features for redundancy
(Pearson + Spearman + VIF), prunes the redundant ones, and persists the locked
vector together with the screen statistics — all as parquet, no JSON.

Why screen: under the Euclidean distance k-means minimises, two near-duplicate
features grant the single behaviour they share a double vote in the partition.
The candidate set is deliberately over-specified, so the redundancy must be
removed transparently and the decision recorded.

Candidate set (six): mean speed, speed variability, the positive and negative
parts of longitudinal acceleration, lateral acceleration intensity, and yaw-rate
intensity. Jerk is NOT a candidate — excluded a priori on signal-quality grounds
(a twice-differentiated GPS position, noise-dominated even in the mean for the
position-derived datasets; see Section 4.6.1).

Decision rule: a pair is redundant when |Pearson| or |Spearman| exceeds
CORR_THRESHOLD, or a feature is multicollinear when its VIF exceeds
VIF_THRESHOLD. The more redundant feature is removed — normally the higher-VIF
member, but when a pair's VIFs are within VIF_TIE_MARGIN the tie is broken on
interpretability, retaining the directly-sensed feature (TIEBREAK_KEEP) over a
partner derived by differentiation. This resolves the a_lat / yaw-rate coupling
(a_lat = v·ω) toward yaw rate.

Grain: clustering uses NON-OVERLAPPING 20 s windows. Since 2026-08-29 the
std_data layer is itself non-overlapping (STEP_S == WINDOW_S), so the
start-time filter below passes every window and is retained only as a cheap
grid-alignment invariant. Correlations
and VIF are scale-invariant, so no standardisation is applied here.

Outputs (all parquet):
  dataset/regime_data/feature_gate_vector.parquet   — the locked vector (ordered)
  dataset/regime_data/feature_gate_stats.parquet    — per-feature VIF (before/after) + decision
  figures/feature_gate/feature_correlation_gate.png — Pearson | Spearman (annotated)
────────────────────────────────────────────────────────────────────────────
"""

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
ARTIFACT_DIR = PROJECT_ROOT / "dataset" / "regime_data"
OUT_DIR      = PROJECT_ROOT / "bachelor-arbeit" / "regime_discovery" / "figures" / "feature_gate"

WINDOW_S = 20.0
DPI      = 300

FEATURES = ["mean_speed_kmh", "std_speed_kmh", "mean_acc_pos", "mean_acc_neg",
            "mean_abs_a_lat", "mean_abs_yaw_rate"]
LABELS   = {"mean_speed_kmh": "speed", "std_speed_kmh": "σ speed",
            "mean_acc_pos": "acc+", "mean_acc_neg": "acc−",
            "mean_abs_a_lat": "|aLat|", "mean_abs_yaw_rate": "|yaw|"}

CORR_THRESHOLD = 0.85    # |Pearson| or |Spearman| above this flags a redundant pair
VIF_THRESHOLD  = 5.0     # VIF above this flags multicollinearity
VIF_TIE_MARGIN = 0.5     # a redundant pair within this VIF gap is treated as a tie
TIEBREAK_KEEP  = ["mean_abs_yaw_rate"]   # on a tie, keep the directly-sensed feature


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _discover() -> list[str]:
    return sorted(p.stem for p in STD_DATA_DIR.glob("*.parquet"))


def _load_population(name: str) -> pd.DataFrame | None:
    """Read one dataset; keep grid-aligned windows.

    Under the non-overlapping layer (2026-08-29) the filter is an identity
    and acts as an alignment invariant, not a selection."""
    fp = STD_DATA_DIR / f"{name}.parquet"
    if not fp.exists():
        print(f"  [WARN] {name}: {fp.name} not found")
        return None
    df = pd.read_parquet(fp, columns=["clip_id", "window_start_s"] + FEATURES)
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
    if X.shape[1] == 2:
        r = spearmanr(X[:, 0], X[:, 1]).correlation
        return np.array([[1.0, r], [r, 1.0]])
    return np.asarray(spearmanr(X).correlation)


def _prune(df_feat: pd.DataFrame) -> tuple[list[str], list[dict]]:
    """Greedy redundancy prune with the interpretability tie-break (see docstring)."""
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
            vi, vj = V[i], V[j]
            tie = abs(vi - vj) < VIF_TIE_MARGIN
            keep_i, keep_j = keep[i] in TIEBREAK_KEEP, keep[j] in TIEBREAK_KEEP
            if tie and (keep_i != keep_j):
                drop_idx = j if keep_i else i
                note = "VIF tie broken toward directly-sensed feature"
            else:
                drop_idx = i if vi >= vj else j
                note = "higher VIF"
            partner = keep[j] if drop_idx == i else keep[i]
            reason = (f"|r|={worst_pair:.2f} with {LABELS.get(partner, partner)} "
                      f"(VIF {vi:.1f}/{vj:.1f}; {note})")
        else:
            drop_idx = int(np.nanargmax(V))
            reason = f"VIF={V[drop_idx]:.1f}"

        log.append({"dropped": keep[drop_idx], "reason": reason})
        keep.pop(drop_idx)
    return keep, log


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING  (pure default matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

def _plot_correlation(P: np.ndarray, S: np.ndarray, labels: list[str], out_fp: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, M, name in [(axes[0], P, "Pearson"), (axes[1], S, "Spearman")]:
        im = ax.imshow(M, cmap="Blues", vmin=-1.0, vmax=1.0)
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
    print(f"  feature_gate.py  ·  {_now()}")
    print(f"  Population ({len(pop)} datasets, all qualified): {pop}")
    print(f"  Candidates ({len(FEATURES)}): {[LABELS[f] for f in FEATURES]}   (jerk excluded a priori)")
    print(f"  Thresholds: |r| > {CORR_THRESHOLD}  ·  VIF > {VIF_THRESHOLD}  ·  tie margin {VIF_TIE_MARGIN}")
    print(f"{'='*78}")

    with ThreadPoolExecutor(max_workers=min(8, len(pop))) as ex:
        frames = list(tqdm(ex.map(_load_population, pop), total=len(pop), desc="reading"))
    frames = [f for f in frames if f is not None]
    if not frames:
        print("[ERROR] no population data loaded."); return

    full = pd.concat(frames, ignore_index=True)
    n_raw = len(full)
    full = full.dropna(subset=FEATURES).reset_index(drop=True)
    n_cc  = len(full)

    print(f"\n  Non-overlap complete-case windows: {n_cc:,} of {n_raw:,}")
    print(f"  {'dataset':<12}{'windows':>10}{'share':>9}")
    print(f"  {'-'*31}")
    for name, g in full.groupby("dataset"):
        print(f"  {name:<12}{len(g):>10,}{100.0*len(g)/n_cc:>8.1f}%")

    # ── screen statistics on the full candidate set ───────────────────────────
    X = full[FEATURES].to_numpy(dtype=float)
    P = np.corrcoef(X, rowvar=False)
    S = _spearman_matrix(X)
    V = _vif(X)   # VIF "before" — full six-candidate set

    if {"mean_abs_a_lat", "mean_abs_yaw_rate"} <= set(FEATURES):
        print(f"\n  a_lat–yaw Pearson per dataset (the coupling should hold within each):")
        for name, g in full.groupby("dataset"):
            print(f"    {name:<12} r = {g['mean_abs_a_lat'].corr(g['mean_abs_yaw_rate']):+.2f}")

    # ── prune ─────────────────────────────────────────────────────────────────
    keep, log = _prune(full[FEATURES])

    # VIF "after" — recomputed on the locked set, reported beside the "before" values
    V_after = dict(zip(keep, _vif(full[keep].to_numpy(dtype=float))))
    print(f"\n  VIF  before -> after  (dropped features have no 'after' value):")
    print(f"    {'feature':<9}{'before':>9}{'after':>10}")
    for f, vb in zip(FEATURES, V):
        va = f"{V_after[f]:.2f}" if f in keep else "dropped"
        flag = "  <-- > %.0f" % VIF_THRESHOLD if vb > VIF_THRESHOLD else ""
        print(f"    {LABELS[f]:<9}{vb:>9.2f}{va:>10}{flag}")

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

    # ── persist (parquet only) ────────────────────────────────────────────────
    fig_fp = OUT_DIR / "feature_correlation_gate.png"
    _plot_correlation(P, S, [LABELS[f] for f in FEATURES], fig_fp)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": keep}).to_parquet(
        ARTIFACT_DIR / "feature_gate_vector.parquet", index=False)

    drop_reason = {d["dropped"]: d["reason"] for d in log}
    pd.DataFrame({
        "feature":     FEATURES,
        "vif_before":  [round(v, 4) for v in V],
        "vif_after":   [round(V_after[f], 4) if f in keep else None for f in FEATURES],
        "kept":        [f in keep for f in FEATURES],
        "drop_reason": [drop_reason.get(f, "") for f in FEATURES],
    }).to_parquet(ARTIFACT_DIR / "feature_gate_stats.parquet", index=False)

    print(f"\n  Figure → {fig_fp}")
    print(f"  Saved  → {ARTIFACT_DIR / 'feature_gate_vector.parquet'}")
    print(f"           {ARTIFACT_DIR / 'feature_gate_stats.parquet'}")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()