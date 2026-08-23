"""
thesis_audit.py
────────────────────────────────────────────────────────────────────────────
READ-ONLY diagnostic. Reads the frozen thesis artifacts and prints every
number still open in the Chapter 4 / Chapter 5 ledger.

It writes exactly one new file, PROJECT_ROOT/thesis_audit_report.txt, and
modifies nothing else. No parquet is rewritten, no figure is regenerated.

Run:   python thesis_audit.py
       python thesis_audit.py --root /some/other/root

Then paste thesis_audit_report.txt back into the chat.
────────────────────────────────────────────────────────────────────────────
Answers:
  A  frame size arbitration      402,975 vs 402,421 vs BUDGET_N 71,215/71,390
  B  regime supplies + tiers     capability / budget N, per-regime counts
  C  tier threshold calibration  channel percentiles + populated-tier check
  D  VERIFY-8                    complete-case window count, dataset shares
  E  VERIFY-9                    volume_gap_js, js_to_balanced
  F  corpus state                per-dataset windows/clips, NVIDIA presence
  G  subset artifacts            which tier dir exists, manifest constants
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:                                    # tqdm is optional here
    def tqdm(x, **k):
        return x

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS (mirrored from the pipeline; edited here only for this audit)
# ─────────────────────────────────────────────────────────────────────────────

WINDOW_S = 20.0
CONFOUNDED_ACCEL = {"a2d2", "4seasons", "durlar"}
VAR_CHANNELS = ["std_speed_kmh", "std_acc_mag", "std_yaw_rate"]
FEATURE_SIX = ["mean_speed_kmh", "std_speed_kmh", "mean_acc_pos",
               "mean_acc_neg", "mean_abs_a_lat", "mean_abs_yaw_rate"]

# Candidate tier triples to score against the data.
# CURRENT  = what ecdf_analysis.py / regime_classification.py hold today.
# ANCHORED = physics-anchored proposal (one maneuver in a quiet 20 s window).
CANDIDATES = {
    "CURRENT": {
        "std_speed_kmh": (1.0, 5.0, 15.0),
        "std_acc_mag":   (0.2, 0.5, 1.5),
        "std_yaw_rate":  (1.0, 5.0, 15.0),
    },
    "ANCHORED": {
        "std_speed_kmh": (1.0, 5.0, 15.0),
        "std_acc_mag":   (0.2, 0.6, 1.3),
        "std_yaw_rate":  (1.0, 3.0, 8.0),
    },
}
TIER_NAMES = ["near-stationary", "low-dynamic", "medium-dynamic", "high-dynamic"]

REPORT: list[str] = []


def say(line: str = "") -> None:
    print(line)
    REPORT.append(line)


def rule(title: str) -> None:
    say("")
    say("=" * 78)
    say(title)
    say("=" * 78)


def safe(fn, label: str):
    """Run a section, never let one missing artifact kill the whole audit."""
    try:
        fn()
    except FileNotFoundError as exc:
        say(f"  !! NOT FOUND: {exc}")
    except Exception as exc:                            # noqa: BLE001
        say(f"  !! {label} FAILED: {type(exc).__name__}: {exc}")


def fmt(n) -> str:
    return f"{n:,}" if isinstance(n, (int, np.integer)) else f"{n}"


# ─────────────────────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def read_parquet_cols(path: Path, cols=None) -> pd.DataFrame:
    """Read only the columns that actually exist, so a schema change cannot abort."""
    import pyarrow.parquet as pq
    schema = pq.ParquetFile(path).schema_arrow
    have = list(schema.names)
    if cols is None:
        return pd.read_parquet(path)
    seen, use = set(), []
    for c in cols:                       # de-duplicate, preserve order:
        if c in have and c not in seen:  # a repeated name yields a duplicated
            seen.add(c)                  # column and silently doubles len()
            use.append(c)
    return pd.read_parquet(path, columns=use)


def load_std_data(std_dir: Path) -> dict[str, pd.DataFrame]:
    files = sorted(std_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet in {std_dir}")
    want = dict.fromkeys(                       # std_speed_kmh is in both lists
        ["clip_id", "window_idx", "window_start_s", "window_end_s", "n_frames"]
        + VAR_CHANNELS + FEATURE_SIX + ["mean_abs_jerk"])
    want = list(want)

    def one(fp: Path):
        return fp.stem, read_parquet_cols(fp, want)

    out = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for name, df in tqdm(ex.map(one, files), total=len(files),
                             desc="std_data", disable=None):
            out[name] = df
    return out


def non_overlapping_mask(df: pd.DataFrame) -> np.ndarray:
    """Grid-aligned windows: starts on integer multiples of the window length."""
    if "window_start_s" not in df.columns:
        return np.ones(len(df), dtype=bool)
    m = np.asarray(df["window_start_s"], dtype=float) % WINDOW_S
    return np.isclose(m, 0.0, atol=1e-6) | np.isclose(m, WINDOW_S, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# SECTIONS
# ─────────────────────────────────────────────────────────────────────────────

def section_A_frame(paths: dict) -> None:
    rule("A · FRAME SIZE ARBITRATION  (402,975 vs 402,421 · BUDGET_N 71,215 vs 71,390)")
    fp = paths["regime"] / "cluster_assignments.parquet"
    df = read_parquet_cols(fp, ["dataset", "clip_id", "window_start_s",
                                "window_end_s", "regime"])
    say(f"  file            : {fp}")
    say(f"  mtime           : {pd.Timestamp(fp.stat().st_mtime, unit='s')}")
    say(f"  FRAME N (rows)  : {fmt(len(df))}   <<< AUTHORITATIVE")
    say(f"  columns         : {list(df.columns)}")
    say(f"  datasets        : {df['dataset'].nunique()}")
    say(f"  unique clips    : {fmt(df['clip_id'].nunique())}")

    dup = df.duplicated(subset=[c for c in ["dataset", "clip_id", "window_start_s"]
                                if c in df.columns]).sum()
    say(f"  duplicate keys  : {fmt(int(dup))}  (must be 0)")

    nov = non_overlapping_mask(df)
    say(f"  grid-aligned    : {fmt(int(nov.sum()))} of {fmt(len(df))} "
        f"({100 * nov.mean():.2f} %)  -> frame is "
        f"{'NON-overlapping' if nov.mean() > 0.99 else 'OVERLAPPING (!)'}")

    guarded = df[~df["dataset"].str.lower().isin(CONFOUNDED_ACCEL)]
    say(f"  GUARDED N       : {fmt(len(guarded))}  (frame minus "
        f"{sorted(CONFOUNDED_ACCEL)})")
    say(f"  windows/clip    : {len(df) / max(df['clip_id'].nunique(), 1):.3f}")
    paths["_frame"] = df


def section_B_supplies(paths: dict) -> None:
    rule("B · REGIME SUPPLIES AND TIER SIZES")
    df = paths.get("_frame")
    if df is None:
        say("  (frame not loaded, skipped)")
        return
    counts = df["regime"].value_counts().sort_index()
    total = int(counts.sum())
    say("  regime   count        share")
    for r, c in counts.items():
        say(f"    {r}    {fmt(int(c)):>10}    {100 * c / total:6.2f} %")
    rarest = int(counts.min())
    rarest_id = int(counts.idxmin())
    say("")
    say(f"  rarest regime          : {rarest_id}  ({fmt(rarest)} windows)")
    say(f"  BUDGET tier   N        : {fmt(rarest)}          -> dir tier_n{rarest}")
    say(f"  CAPABILITY tier N      : {fmt(5 * rarest)}      (5 x rarest)")
    say(f"  capability retains     : {100 * 5 * rarest / total:.2f} % of frame")

    gd = df[~df["dataset"].str.lower().isin(CONFOUNDED_ACCEL)]
    gcounts = gd["regime"].value_counts().sort_index()
    say(f"  guarded rarest supply  : {fmt(int(gcounts.min()))} "
        f"(regime {int(gcounts.idxmin())})")

    say("")
    say("  per-dataset x regime (rows = dataset, cols = regime):")
    piv = pd.crosstab(df["dataset"], df["regime"])
    for line in piv.to_string().splitlines():
        say("    " + line)
    empty = [(d, r) for d in piv.index for r in piv.columns if piv.loc[d, r] == 0]
    say(f"  structurally empty cells: {len(empty)}  {empty[:12]}")


def section_C_tiers(paths: dict, std: dict) -> None:
    rule("C · TIER THRESHOLD CALIBRATION EVIDENCE")
    pooled = {}
    for ch in VAR_CHANNELS:
        parts = []
        for d in std.values():
            if ch not in d.columns:
                continue
            c = d[ch]
            if isinstance(c, pd.DataFrame):
                c = c.iloc[:, 0]
            parts.append(c.to_numpy(dtype=float))
        vals = np.concatenate(parts) if parts else np.array([])
        vals = vals[np.isfinite(vals)]
        if ch == "std_yaw_rate":
            p99_raw = np.percentile(vals, 99)
            unit = "rad/s" if p99_raw < 1.5 else "deg/s"
            say(f"  [unit check] std_yaw_rate p99 = {p99_raw:.4f} -> stored in {unit}")
            if unit == "rad/s":
                vals = vals * (180.0 / np.pi)
                say("               converted to deg/s for the tier check below")
        pooled[ch] = vals

    say("")
    say("  pooled percentiles (diagnostic only, NOT a calibration source):")
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    say("    channel            " + "".join(f"{('p%d' % q):>9}" for q in qs))
    for ch, v in pooled.items():
        say(f"    {ch:<18}" + "".join(f"{np.percentile(v, q):9.3f}" for q in qs))
        say(f"      {'':<16}n={fmt(len(v))}  mean={v.mean():.3f}  max={v.max():.3f}")

    for name, triples in CANDIDATES.items():
        say("")
        say(f"  --- POOLED tier shares under {name} thresholds ---")
        for ch, (t1, t2, t3) in triples.items():
            v = pooled[ch]
            sh = [
                float((v < t1).mean()),
                float(((v >= t1) & (v < t2)).mean()),
                float(((v >= t2) & (v < t3)).mean()),
                float((v >= t3).mean()),
            ]
            say(f"    {ch:<18} t=({t1}, {t2}, {t3})   " +
                "  ".join(f"{TIER_NAMES[i][:4]}={100 * sh[i]:5.2f}%" for i in range(4)))
            if sh[3] < 0.005:
                say(f"      !! top tier under 0.5 % pooled -> threshold likely too high")

    say("")
    say("  --- PER-DATASET high-dynamic share, CURRENT vs ANCHORED (speed / yaw) ---")
    for ch in ["std_speed_kmh", "std_yaw_rate"]:
        say(f"    channel {ch}")
        for name, df in sorted(std.items()):
            if ch not in df.columns:
                continue
            col = df[ch]
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            v = col.to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if ch == "std_yaw_rate" and np.percentile(pooled[ch], 99) > 1.5:
                if np.percentile(v, 99) < 1.5:
                    v = v * (180.0 / np.pi)
            if len(v) == 0:
                continue
            cur = CANDIDATES["CURRENT"][ch][2]
            anc = CANDIDATES["ANCHORED"][ch][2]
            say(f"      {name:<12} n={fmt(len(v)):>9}  "
                f">=t3_cur({cur}) {100 * (v >= cur).mean():6.2f}%   "
                f">=t3_anc({anc}) {100 * (v >= anc).mean():6.2f}%")


def section_D_verify8(std: dict) -> None:
    rule("D · VERIFY-8  COMPLETE-CASE COUNT AND DATASET SHARES")
    rows = []
    for name, df in sorted(std.items()):
        nov = non_overlapping_mask(df)
        sub = df.loc[nov]
        sub = sub.loc[:, ~sub.columns.duplicated()]
        have = [c for c in FEATURE_SIX if c in sub.columns]
        missing = [c for c in FEATURE_SIX if c not in sub.columns]
        cc = int(sub[have].notna().all(axis=1).sum()) if have else 0
        rows.append({"dataset": name, "windows_all": len(df),
                     "non_overlapping": int(nov.sum()), "complete_case": cc,
                     "missing_cols": ",".join(missing) or "-"})
    t = pd.DataFrame(rows).sort_values("complete_case", ascending=False)
    tot = int(t["complete_case"].sum())
    t["share_%"] = (100 * t["complete_case"] / max(tot, 1)).round(2)
    for line in t.to_string(index=False).splitlines():
        say("    " + line)
    say("")
    say(f"  COMPLETE-CASE non-overlapping windows (6 candidates): {fmt(tot)}")
    if len(t):
        say(f"  largest dataset share : {t['share_%'].iloc[0]:.2f} %  "
            f"({t['dataset'].iloc[0]})")
        say(f"  two largest together  : {t['share_%'].iloc[:2].sum():.2f} %")


def section_E_verify9(paths: dict) -> None:
    rule("E · VERIFY-9  CORPUS SCALARS AND PER-DATASET BIAS TABLE")
    rd = paths["regime"]
    for fname in ["bias_corpus_scalars.parquet", "bias_corpus_profiles.parquet"]:
        fp = rd / fname
        if not fp.exists():
            say(f"  {fname}: NOT FOUND")
            continue
        df = pd.read_parquet(fp)
        say(f"  --- {fname}  (mtime {pd.Timestamp(fp.stat().st_mtime, unit='s')}) ---")
        for line in df.to_string().splitlines():
            say("    " + line)

    fp = rd / "bias_per_dataset.parquet"
    if fp.exists():
        df = pd.read_parquet(fp)
        say("")
        say(f"  --- bias_per_dataset.parquet  columns ---")
        say(f"    {list(df.columns)}")
        keep = [c for c in ["low_dyn_share", "entropy_norm", "hhi_norm",
                            "js_to_pooled", "js_to_balanced", "psi_to_pooled",
                            "n_windows", "confounded_accel"] if c in df.columns]
        if keep:
            for line in df[keep].round(4).to_string().splitlines():
                say("    " + line)
    else:
        say("  bias_per_dataset.parquet: NOT FOUND")


def section_F_corpus(paths: dict, std: dict) -> None:
    rule("F · CORPUS STATE  (NVIDIA restoration check)")
    frame = paths.get("_frame")
    rows = []
    for name, df in sorted(std.items()):
        rows.append({
            "dataset": name,
            "std_windows": len(df),
            "clips": int(df["clip_id"].nunique()) if "clip_id" in df else -1,
            "in_assignments": (int((frame["dataset"] == name).sum())
                               if frame is not None else -1),
        })
    t = pd.DataFrame(rows)
    t["delta"] = t["in_assignments"] - t["std_windows"]
    for line in t.to_string(index=False).splitlines():
        say("    " + line)
    say("")
    say(f"  datasets in std_data      : {len(std)}  (expect 14)")
    nv = [k for k in std if "nvidia" in k.lower()]
    if nv:
        n = len(std[nv[0]])
        say(f"  NVIDIA windows present    : {fmt(n)}")
        say(f"  NVIDIA share of std_data  : "
            f"{100 * n / sum(len(d) for d in std.values()):.2f} %  "
            f"(expect ~74 % if fully restored)")
    else:
        say("  !! NVIDIA ABSENT from std_data -> counts below are partial-corpus")


def section_G_subsets(paths: dict) -> None:
    rule("G · SUBSET ARTIFACTS AND MANIFESTS")
    sd = paths["subset"]
    if not sd.exists():
        say(f"  {sd}: NOT FOUND")
        return
    tier_dirs = sorted([p for p in sd.iterdir() if p.is_dir()
                        and p.name.startswith("tier_")])
    say(f"  tier directories present : {[p.name for p in tier_dirs] or 'none'}")
    for d in [sd] + tier_dirs:
        mf = d / "manifest.json"
        say("")
        say(f"  --- {d.name or d} ---")
        pq_files = sorted(p.name for p in d.glob("*.parquet"))
        say(f"    parquet files ({len(pq_files)}): {pq_files}")
        if not mf.exists():
            say("    manifest.json: NOT FOUND")
            continue
        m = json.loads(mf.read_text())
        say(f"    manifest mtime: {pd.Timestamp(mf.stat().st_mtime, unit='s')}")
        flat = {}

        def walk(obj, prefix=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    walk(v, f"{prefix}{k}.")
            elif isinstance(obj, (str, int, float, bool)) or obj is None:
                flat[prefix.rstrip(".")] = obj

        walk(m)
        interesting = [k for k in flat
                       if any(s in k.lower() for s in
                              ("n", "seed", "guard", "trigger", "predicate",
                               "supply", "count", "size", "tier", "q", "g"))]
        for k in sorted(interesting)[:80]:
            say(f"      {k} = {flat[k]}")

    for name in ["supply_table.csv", "allocation_table.csv"]:
        fp = sd / name
        if fp.exists():
            say("")
            say(f"  --- {name} ---")
            for line in pd.read_csv(fp).to_string(index=False).splitlines()[:25]:
                say("    " + line)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Users/edward/Downloads")
    args = ap.parse_args()

    root = Path(args.root)
    paths = {
        "root": root,
        "std": root / "dataset" / "std_data",
        "regime": root / "dataset" / "regime_data",
        "subset": root / "dataset" / "subset_data",
    }

    say("THESIS AUDIT REPORT")
    say(f"generated : {pd.Timestamp.now()}")
    say(f"root      : {root}")
    say(f"python    : {sys.version.split()[0]}  pandas {pd.__version__}")
    for k in ("std", "regime", "subset"):
        say(f"{k:<10}: {paths[k]}  {'OK' if paths[k].exists() else 'MISSING'}")

    std: dict[str, pd.DataFrame] = {}
    if paths["std"].exists():
        try:
            std = load_std_data(paths["std"])
        except Exception as exc:                        # noqa: BLE001
            say(f"!! std_data load failed: {exc}")

    if paths["regime"].exists():
        safe(lambda: section_A_frame(paths), "A")
        safe(lambda: section_B_supplies(paths), "B")
    if std:
        safe(lambda: section_C_tiers(paths, std), "C")
        safe(lambda: section_D_verify8(std), "D")
    if paths["regime"].exists():
        safe(lambda: section_E_verify9(paths), "E")
    if std:
        safe(lambda: section_F_corpus(paths, std), "F")
    safe(lambda: section_G_subsets(paths), "G")

    rule("END OF REPORT")
    out = root / "thesis_audit_report.txt"
    out.write_text("\n".join(REPORT), encoding="utf-8")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())