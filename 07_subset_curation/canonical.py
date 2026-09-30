#!/usr/bin/env python3
"""canonical.py — one derivation of every canonical corpus quantity.

WHY THIS EXISTS
---------------
Canonical counts restated as literals in several files drift apart silently
whenever the corpus is regenerated. Re-anchoring a stale literal hides the
drift but does not remove it.

Every quantity below is DERIVED from the artifacts on disk at import time.
Nothing here is typed by hand. Downstream scripts import from this module
instead of restating a number, so a re-run propagates automatically and a
stale literal cannot survive anywhere.

REPLACING THE CANARIES
----------------------
A canary that asserts a literal breaks on every legitimate re-run, which
invites passing --force, which defeats the canary. Instead:

    python canonical.py --freeze     # after an accepted run, writes a
                                     # snapshot (with its date) next to
                                     # the artifacts
    python canonical.py --check      # compares live values to the snapshot,
                                     # prints every drift old -> new, exits 1

So the guard answers "did anything change since the corpus was last
accepted" rather than "does the corpus match a hand-typed number".

USAGE
-----
    from canonical import C
    print(C.frame_n, C.budget_n, C.regime_supplies)
    subsets = C.budget_dir            # .../subset_data/tier_n<budget_n>

    python canonical.py              # print the full table
    python canonical.py --json       # machine-readable
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import date
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# module discovery
# ---------------------------------------------------------------------------
# The pipeline modules this file reads its constants from live in sibling
# directories (07_subset_curation, 06_bias_quantification, and so on), so a single
# path entry cannot cover them. Search this file's directory and its immediate
# subtrees for the modules by name and add wherever they are found. An explicit
# CANONICAL_MODULE_DIR (colon-separated) always wins.
_MARKER_MODULES = ["run_sampling.py", "bias_quant.py", "curation_common.py",
                   "subset_eval_common.py", "regime_style.py"]
_SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules",
              ".ipynb_checkpoints", "figures", "dataset"}
_module_dirs: list[str] = []


def _discover_module_dirs(root: Path, max_depth: int = 3) -> list[str]:
    found: list[str] = []
    roots = [root, root.parent]
    seen: set = set()
    for base in roots:
        if not base.is_dir():
            continue
        stack = [(base, 0)]
        while stack:
            d, depth = stack.pop()
            if d in seen or depth > max_depth:
                continue
            seen.add(d)
            try:
                entries = list(d.iterdir())
            except (PermissionError, OSError):
                continue
            if any(e.name in _MARKER_MODULES for e in entries if e.is_file()):
                found.append(str(d))
            for e in entries:
                if e.is_dir() and e.name not in _SKIP_DIRS \
                        and not e.name.startswith("."):
                    stack.append((e, depth + 1))
    return found


_here = Path(__file__).resolve().parent
for _d in os.environ.get("CANONICAL_MODULE_DIR", "").split(os.pathsep):
    if _d:
        _module_dirs.append(_d)
_module_dirs.append(str(_here))
_module_dirs.extend(_discover_module_dirs(_here))
for _d in dict.fromkeys(_module_dirs):          # de-dup, preserve order
    if _d not in sys.path:
        sys.path.insert(0, _d)


# ---------------------------------------------------------------------------
# paths — same convention and env override as subset_eval_common.py
# ---------------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[1]))
from thesis_paths import DATA_ROOT, REPO_ROOT  # noqa: E402
DATASET_DIR = DATA_ROOT
REGIME_DIR = DATASET_DIR / "regime_data"
SUBSET_DIR = DATASET_DIR / "subset_data"
STD_DATA_DIR = DATASET_DIR / "std_data"

REGIME_ASSIGN = REGIME_DIR / "cluster_assignments.parquet"
WINDOW_SUMMARIES = SUBSET_DIR / "window_summaries.parquet"
SNAPSHOT = DATASET_DIR / "canonical_snapshot.json"

# ---------------------------------------------------------------------------
# imported single sources of truth — never redefined here
# ---------------------------------------------------------------------------
try:
    from regime_style import REGIME_LABELS, N_REGIMES
except ImportError:                                    # pragma: no cover
    REGIME_LABELS = ["Calm / Stationary", "Highway Cruising", "Turning",
                     "Braking", "Accelerating"]
    N_REGIMES = len(REGIME_LABELS)

# Imports below are LAZY on purpose. Downstream modules (subset_eval_common,
# curation_common, subset_eval_harness) will import canonical, so importing
# them at module level here would create a cycle that the try/except would
# swallow into a silent literal fallback — the exact failure this module
# exists to prevent. Resolving them inside functions keeps the graph acyclic.

def _confounded():
    """The fidelity flag (thesis 4.4.5, 4.8.2), from its owning module."""
    try:
        from bias_quant import CONFOUNDED_ACCEL
        return set(CONFOUNDED_ACCEL), "bias_quant.CONFOUNDED_ACCEL"
    except Exception:
        pass
    try:
        from subset_eval_common import CONFOUNDED
        return set(CONFOUNDED), "subset_eval_common.CONFOUNDED"
    except Exception:
        return {"a2d2", "4seasons", "durlar"}, "fallback literal"


def _trigger():
    """Longitudinal trigger, from run_sampling, which defines the event set.

    Unresolved inconsistency in the tree: run_sampling.py uses G_MS2 = 9.81
    and curation_common.py uses 9.80665, so 0.6 g is 5.886 in one module and
    5.884 in the other. Harmless at present tolerances; should be unified.
    """
    try:
        from run_sampling import DECEL_TRIGGER_G, G_MS2
        return float(DECEL_TRIGGER_G), float(G_MS2), "run_sampling"
    except Exception:
        return 0.6, 9.81, "fallback literal"


def _evasive_consts():
    """Evasive-rule constants from curation_common, which owns the rule.

    Note the g it returns is curation_common's 9.80665, NOT run_sampling's
    9.81. Each constant is taken from the module that defines the rule using
    it: mixing them shifts the implied-lateral threshold by 0.034% and moves
    the qualifying count by a handful of windows.
    """
    try:
        from curation_common import EVASIVE_LAT_G, YAW_PHYS_BOUND_DEGS, G_MS2
        return float(EVASIVE_LAT_G), float(YAW_PHYS_BOUND_DEGS), float(G_MS2)
    except Exception:
        return 0.75, 90.0, 9.80665


WSTART_CANDIDATES = ["window_start_s", "start_s", "win_start_s"]
WIDX_CANDIDATES = ["window_idx", "win_idx", "widx", "window_index"]
CLIP_CANDIDATES = ["clip_id", "clip", "clip_name", "source_file"]


def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _norm_keys(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Copy of df with float key columns rounded, so a merge on a window
    start time cannot miss by a floating-point ulp."""
    out = df[keys].copy()
    for k in keys:
        if pd.api.types.is_float_dtype(out[k]):
            out[k] = out[k].round(3)
    return out


def _is_unique(df: pd.DataFrame, keys: list[str]) -> bool:
    return not _norm_keys(df, keys).duplicated().any()


def _hill_q2(shares: np.ndarray) -> float:
    """Effective number of contributors, Hill order 2 (inverse Simpson)."""
    s = np.asarray(shares, dtype=float)
    s = s[s > 0]
    return float(1.0 / np.sum(s ** 2)) if s.size else 0.0


# ---------------------------------------------------------------------------
@dataclass
class Canonical:
    """Every canonical quantity, derived once and cached."""

    assign_path: Path = REGIME_ASSIGN
    summaries_path: Path = WINDOW_SUMMARIES
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    join_route: str | None = None
    join_coverage: float | None = None
    confounded_source: str | None = None
    trigger_source: str | None = None

    # ---------------- source frames ----------------
    @cached_property
    def _assign(self) -> pd.DataFrame:
        if not self.assign_path.exists():
            raise SystemExit(
                f"[canonical] missing {self.assign_path}\n"
                f"            run the clustering stage before importing this."
            )
        df = pd.read_parquet(self.assign_path)
        for col in ("dataset", "regime"):
            if col not in df.columns:
                raise SystemExit(
                    f"[canonical] {self.assign_path.name} has no '{col}' column; "
                    f"found {list(df.columns)}"
                )
        return df

    @cached_property
    def _summaries(self) -> pd.DataFrame | None:
        if not self.summaries_path.exists():
            self.warnings.append(
                f"{self.summaries_path.name} absent — event_reserve_n and "
                f"evasive_n unavailable"
            )
            return None
        return pd.read_parquet(self.summaries_path)

    # ---------------- frame ----------------
    @cached_property
    def frame_n(self) -> int:
        return int(len(self._assign))

    @cached_property
    def confounded(self) -> set:
        s, src = _confounded()
        self.confounded_source = src
        if src == "fallback literal":
            self.warnings.append(
                "CONFOUNDED_ACCEL could not be imported, using a literal copy. "
                "The single-source guarantee is broken. Searched: "
                + ", ".join(dict.fromkeys(_module_dirs))
                + ". Set CANONICAL_MODULE_DIR to the directory holding "
                  "bias_quant.py."
            )
        return s

    @cached_property
    def guarded_mask(self) -> pd.Series:
        return ~self._assign["dataset"].isin(self.confounded)

    @cached_property
    def guarded_n(self) -> int:
        return int(self.guarded_mask.sum())

    @cached_property
    def datasets(self) -> list[str]:
        return sorted(self._assign["dataset"].unique().tolist())

    @cached_property
    def n_datasets(self) -> int:
        return len(self.datasets)

    # ---------------- regimes ----------------
    @cached_property
    def regime_supplies(self) -> list[int]:
        counts = (self._assign["regime"].value_counts()
                  .reindex(range(N_REGIMES), fill_value=0)
                  .sort_index())
        return [int(v) for v in counts.values]

    @cached_property
    def rarest_regime(self) -> int:
        return int(np.argmin(self.regime_supplies))

    @cached_property
    def rarest_regime_label(self) -> str:
        return REGIME_LABELS[self.rarest_regime]

    # ---------------- tiers ----------------
    @cached_property
    def budget_n(self) -> int:
        """One rarest-regime supply: the largest uniformly allocable budget."""
        return int(min(self.regime_supplies))

    @cached_property
    def capability_n(self) -> int:
        return int(N_REGIMES * self.budget_n)

    @cached_property
    def capability_frac(self) -> float:
        return self.capability_n / self.frame_n

    @cached_property
    def frame_to_budget(self) -> float:
        """Retention ceiling for any census construction."""
        return self.frame_n / self.budget_n

    @cached_property
    def regime_quota(self) -> int:
        """Equal allocation quota per regime at the budget tier."""
        return self.budget_n // N_REGIMES

    @cached_property
    def budget_dir(self) -> Path:
        return SUBSET_DIR / f"tier_n{self.budget_n}"

    # ---------------- source concentration ----------------
    @cached_property
    def source_shares(self) -> pd.Series:
        return (self._assign["dataset"].value_counts(normalize=True)
                .sort_values(ascending=False))

    @cached_property
    def top_source(self) -> str:
        return str(self.source_shares.index[0])

    @cached_property
    def top_source_share(self) -> float:
        return float(self.source_shares.iloc[0])

    @cached_property
    def top_two_share(self) -> float:
        return float(self.source_shares.iloc[:2].sum())

    @cached_property
    def d2_sources(self) -> float:
        """Effective number of contributing datasets, Hill order 2."""
        return _hill_q2(self.source_shares.values)

    @cached_property
    def structural_zeros(self) -> list[tuple[str, int]]:
        """(dataset, regime) cells with no windows at all."""
        pivot = (self._assign.groupby(["dataset", "regime"]).size()
                 .unstack(fill_value=0)
                 .reindex(columns=range(N_REGIMES), fill_value=0))
        return [(str(ds), int(r))
                for ds in pivot.index for r in pivot.columns
                if pivot.loc[ds, r] == 0]

    # ---------------- event populations ----------------
    @cached_property
    def join_keys(self) -> list[str] | None:
        """The first candidate key set present and unique on BOTH frames.

        A merge on a non-unique key silently multiplies rows and inflates
        every count derived from it, so uniqueness is verified before the
        merge rather than asserted during it.
        """
        s = self._summaries
        if s is None:
            return None
        a = self._assign

        clip_a, clip_s = (_first_present(a, CLIP_CANDIDATES),
                          _first_present(s, CLIP_CANDIDATES))
        ws_a, ws_s = (_first_present(a, WSTART_CANDIDATES),
                      _first_present(s, WSTART_CANDIDATES))
        wi_a, wi_s = (_first_present(a, WIDX_CANDIDATES),
                      _first_present(s, WIDX_CANDIDATES))
        has_ds = "dataset" in a.columns and "dataset" in s.columns

        candidates: list[list[str]] = []
        clip = clip_a if clip_a and clip_a == clip_s else None
        ws = ws_a if ws_a and ws_a == ws_s else None
        wi = wi_a if wi_a and wi_a == wi_s else None

        if clip and ws:
            candidates.append((["dataset"] if has_ds else []) + [clip, ws])
        if clip and wi:
            candidates.append((["dataset"] if has_ds else []) + [clip, wi])
        if clip and ws:
            candidates.append([clip, ws])
        if clip and wi:
            candidates.append([clip, wi])

        for keys in candidates:
            if not (set(keys) <= set(a.columns) and set(keys) <= set(s.columns)):
                continue
            if _is_unique(a, keys) and _is_unique(s, keys):
                return keys

        tried = " | ".join(",".join(k) for k in candidates) or "none shared"
        self.notes.append(
            "no unique key shared by assignments and window_summaries "
            f"(tried: {tried}); using the std_data bridge, which is the "
            "documented route for this pipeline"
        )
        return None

    @cached_property
    def _bridge(self) -> pd.DataFrame | None:
        """std_data carries BOTH window keys by pipeline contract, so it
        bridges the assignment grid (start-time key) to window_summaries
        (index key). This is the same bridge subset_eval_harness.py uses.
        """
        if not STD_DATA_DIR.exists():
            self.warnings.append(f"{STD_DATA_DIR} absent — cannot bridge keys")
            return None
        frames = []
        for p in sorted(STD_DATA_DIR.glob("*.parquet")):
            try:
                d = pd.read_parquet(p, columns=["clip_id", "window_idx",
                                                "window_start_s"])
            except Exception as exc:                    # pragma: no cover
                self.warnings.append(f"bridge skipped {p.name}: {exc}")
                continue
            d["dataset"] = p.stem
            frames.append(d)
        if not frames:
            self.warnings.append(f"no readable parquet in {STD_DATA_DIR}")
            return None
        b = pd.concat(frames, ignore_index=True)
        b["window_start_s"] = b["window_start_s"].round(3)
        return b

    @cached_property
    def _joined(self) -> pd.DataFrame | None:
        """Assignments joined to window summaries, directly where a unique
        shared key exists, otherwise through the std_data bridge."""
        s = self._summaries
        if s is None:
            return None
        a = self._assign

        keys = self.join_keys
        if keys is not None:
            left, right = a.copy(), s.copy()
            for k in keys:
                if pd.api.types.is_float_dtype(left[k]):
                    left[k] = left[k].round(3)
                if pd.api.types.is_float_dtype(right[k]):
                    right[k] = right[k].round(3)
            extra = [c for c in right.columns
                     if c not in left.columns and c not in keys]
            j = left.merge(right[keys + extra], on=keys, how="left",
                           validate="one_to_one")
            self.join_route = f"direct on {keys}"
            return self._check_coverage(j, keys)

        # --- bridged route -------------------------------------------------
        b = self._bridge
        if b is None:
            return None
        k1 = ["dataset", "clip_id", "window_start_s"]
        k2 = ["dataset", "clip_id", "window_idx"]
        if not (set(k1) <= set(a.columns) and set(k2) <= set(s.columns)):
            self.warnings.append(
                f"bridge needs {k1} on assignments and {k2} on summaries; "
                f"assignments have {list(a.columns)}, "
                f"summaries have {list(s.columns)}"
            )
            return None
        if not (_is_unique(b, k1) and _is_unique(b, k2)):
            self.warnings.append(
                "std_data bridge is not unique on both key sets; "
                "the window layer may hold duplicate rows"
            )
            return None
        if not (_is_unique(a, k1) and _is_unique(s, k2)):
            self.warnings.append(
                "assignments or summaries are not unique on their own window "
                "key; event counts unavailable"
            )
            return None

        left = a.copy()
        left["window_start_s"] = left["window_start_s"].round(3)
        step1 = left.merge(b[k1 + ["window_idx"]], on=k1, how="left",
                           validate="one_to_one")
        missing = int(step1["window_idx"].isna().sum())
        if missing:
            self.warnings.append(
                f"{missing:,} assignment windows ({missing / len(step1):.2%}) "
                f"are absent from the std_data bridge"
            )
        step1 = step1.dropna(subset=["window_idx"])
        step1["window_idx"] = step1["window_idx"].astype(int)
        extra = [c for c in s.columns if c not in step1.columns and c not in k2]
        j = step1.merge(s[k2 + extra], on=k2, how="left", validate="one_to_one")
        self.join_route = f"bridged via std_data, {k1} -> {k2}"
        return self._check_coverage(j, k2)

    def _check_coverage(self, j: pd.DataFrame, keys: list[str]):
        if "max_decel_ms2" in j.columns:
            cov = float(j["max_decel_ms2"].notna().mean())
            self.join_coverage = cov
            if cov < 0.95:
                self.warnings.append(
                    f"window_summaries join coverage {cov:.1%} is below 0.95 "
                    f"on keys {keys}; event counts are unreliable"
                )
        return j

    @cached_property
    def decel_trigger_ms2(self) -> float:
        g_frac, g, src = _trigger()
        self.trigger_source = f"{src}  {g_frac} g"
        if src == "fallback literal":
            self.warnings.append(
                "DECEL_TRIGGER_G could not be imported from run_sampling, "
                "using a literal copy. Change the constant there and this "
                "module will not follow. Searched: "
                + ", ".join(dict.fromkeys(_module_dirs))
                + ". Set CANONICAL_MODULE_DIR to the directory holding "
                  "run_sampling.py."
            )
        return g_frac * g

    @cached_property
    def event_reserve_n(self) -> int | None:
        """Guarded windows reaching the longitudinal trigger. The event-reserve census (thesis 4.11)."""
        j = self._joined
        if j is None or "max_decel_ms2" not in j.columns:
            return None
        m = (~j["dataset"].isin(self.confounded)) & \
            (j["max_decel_ms2"] >= self.decel_trigger_ms2)
        return int(m.sum())

    @cached_property
    def event_reserve_frac(self) -> float | None:
        n = self.event_reserve_n
        return None if n is None else n / self.guarded_n

    @cached_property
    def evasive_n(self) -> int | None:
        """Evasive held-out population under the implied-lateral rule."""
        j = self._joined
        if j is None:
            return None
        need = {"mean_speed_kmh", "max_abs_yaw_degs"}
        if not need.issubset(j.columns):
            self.warnings.append(
                f"evasive_n needs {sorted(need)}; not all present"
            )
            return None
        lat_g, yaw_bound, g = _evasive_consts()
        v = j["mean_speed_kmh"].to_numpy(dtype=float) / 3.6
        with np.errstate(divide="ignore", invalid="ignore"):
            thr = np.degrees(lat_g * g / np.maximum(v, 1e-9))
        thr = np.where(v <= 0, np.inf, thr)
        ok = thr <= yaw_bound
        yaw = j["max_abs_yaw_degs"].to_numpy(dtype=float)
        return int(np.sum(ok & (yaw >= thr)))

    # ---------------- export ----------------
    def to_dict(self) -> dict:
        return {
            "frame_n": self.frame_n,
            "guarded_n": self.guarded_n,
            "n_datasets": self.n_datasets,
            "n_regimes": N_REGIMES,
            "regime_supplies": self.regime_supplies,
            "rarest_regime": self.rarest_regime,
            "rarest_regime_label": self.rarest_regime_label,
            "budget_n": self.budget_n,
            "regime_quota": self.regime_quota,
            "capability_n": self.capability_n,
            "capability_frac": round(self.capability_frac, 6),
            "frame_to_budget": round(self.frame_to_budget, 6),
            "top_source": self.top_source,
            "top_source_share": round(self.top_source_share, 6),
            "top_two_share": round(self.top_two_share, 6),
            "d2_sources": round(self.d2_sources, 6),
            "n_structural_zeros": len(self.structural_zeros),
            "structural_zeros": self.structural_zeros,
            "decel_trigger_ms2": round(self.decel_trigger_ms2, 6),
            "event_reserve_n": self.event_reserve_n,
            "event_reserve_frac": (None if self.event_reserve_frac is None
                                   else round(self.event_reserve_frac, 6)),
            "evasive_n": self.evasive_n,
            "budget_dir": str(self.budget_dir),
            "join_route": self.join_route,
        }


C = Canonical()


# ---------------------------------------------------------------------------
# snapshot: freeze / check, the canary replacement
# ---------------------------------------------------------------------------
def freeze(path: Path = SNAPSHOT) -> dict:
    payload = {"frozen_on": date.today().isoformat(), "values": C.to_dict()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[canonical] snapshot written {path}  ({payload['frozen_on']})")
    return payload


def check(path: Path = SNAPSHOT, verbose: bool = True) -> list[tuple]:
    """Return [(key, old, new)] for every value that moved since the freeze."""
    if not path.exists():
        if verbose:
            print(f"[canonical] no snapshot at {path} — run --freeze once you "
                  f"accept the current corpus")
        return []
    snap = json.loads(path.read_text(encoding="utf-8"))
    old, new = snap["values"], C.to_dict()

    def norm(v):
        """JSON has no tuples, so round-tripping turns them into lists.
        Normalize both sides or every check reports false drift."""
        if isinstance(v, (list, tuple)):
            return [norm(x) for x in v]
        return v

    ignore = {"join_route"}          # provenance, not a canonical quantity
    drift = [(k, old.get(k), new[k]) for k in new
             if k in old and k not in ignore and norm(old[k]) != norm(new[k])]
    if verbose:
        print(f"[canonical] snapshot of {snap['frozen_on']}")
        if not drift:
            print("[canonical] no drift, every canonical value matches")
        else:
            print(f"[canonical] {len(drift)} value(s) moved:")
            for k, o, n in drift:
                print(f"    {k:<22} {o}  ->  {n}")
    return drift


# ---------------------------------------------------------------------------
def _print_table() -> None:
    d = C.to_dict()
    _ = C.confounded, C.decel_trigger_ms2      # resolve provenance strings
    print("=" * 68)
    print("  canonical corpus quantities, derived from artifacts")
    print("=" * 68)
    print(f"  confounded set from : {C.confounded_source}  "
          f"({sorted(C.confounded)})")
    print(f"  trigger constant    : {C.trigger_source} "
          f"= {d['decel_trigger_ms2']} m/s2")
    print("-" * 68)
    rows = [
        ("frame windows", f"{d['frame_n']:,}"),
        ("guarded windows", f"{d['guarded_n']:,}"),
        ("datasets", d["n_datasets"]),
        ("regime supplies", ", ".join(f"{v:,}" for v in d["regime_supplies"])),
        ("rarest regime", f"{d['rarest_regime']} {d['rarest_regime_label']}"),
        ("budget N", f"{d['budget_n']:,}"),
        ("quota per regime", f"{d['regime_quota']:,}"),
        ("capability N", f"{d['capability_n']:,} "
                         f"({d['capability_frac']:.2%} of frame)"),
        ("frame-to-budget", f"{d['frame_to_budget']:.3f}"),
        ("top source", f"{d['top_source']} at {d['top_source_share']:.4f}"),
        ("top two share", f"{d['top_two_share']:.4f}"),
        ("effective sources", f"{d['d2_sources']:.2f} of {d['n_datasets']}"),
        ("structural zeros", d["n_structural_zeros"]),
        ("event reserve", "unavailable" if d["event_reserve_n"] is None
         else f"{d['event_reserve_n']:,} "
              f"({d['event_reserve_frac']:.3%} of guarded)"),
        ("evasive population", "unavailable" if d["evasive_n"] is None
         else f"{d['evasive_n']:,}"),
        ("join route", d["join_route"] or "none"),
        ("budget directory", d["budget_dir"]),
    ]
    for k, v in rows:
        print(f"  {k:<20} {v}")
    if d["structural_zeros"]:
        print("-" * 68)
        for ds, r in d["structural_zeros"]:
            print(f"  structural zero: {ds} x regime {r} "
                  f"({REGIME_LABELS[r]})")
    if C.notes:
        print("-" * 68)
        for n in C.notes:
            print(f"  note: {n}")
    if C.warnings:
        print("-" * 68)
        for w in C.warnings:
            print(f"  WARNING: {w}")
    print("=" * 68)


def diagnose() -> int:
    """Report both artifact schemas and why a join key does or does not work."""
    print("=" * 68)
    print("  join diagnosis")
    print("=" * 68)
    a = C._assign
    print(f"  cluster_assignments : {len(a):,} rows")
    print(f"    columns: {list(a.columns)}")
    for c in a.columns:
        print(f"      {c:<22} {str(a[c].dtype):<12} "
              f"nunique={a[c].nunique(dropna=True):,}")

    s = C._summaries
    if s is None:
        print("\n  window_summaries    : ABSENT")
        print("=" * 68)
        return 1
    print(f"\n  window_summaries    : {len(s):,} rows")
    print(f"    columns: {list(s.columns)}")
    for c in s.columns:
        print(f"      {c:<22} {str(s[c].dtype):<12} "
              f"nunique={s[c].nunique(dropna=True):,}")

    _ = C._joined                      # force the join so the route resolves
    print("\n  candidate key sets:")
    shared = [c for c in a.columns if c in s.columns]
    tests = []
    clip = _first_present(a, CLIP_CANDIDATES)
    ws = _first_present(a, WSTART_CANDIDATES)
    wi = _first_present(a, WIDX_CANDIDATES)
    has_ds = "dataset" in shared
    for combo in ([clip], [clip, ws], [clip, wi],
                  ["dataset", clip], ["dataset", clip, ws],
                  ["dataset", clip, wi]):
        keys = [k for k in combo if k]
        if not keys or (keys[0] == "dataset" and not has_ds):
            continue
        if not (set(keys) <= set(shared)):
            tests.append((keys, "not shared by both", None, None))
            continue
        ua, us = _is_unique(a, keys), _is_unique(s, keys)
        da = int(_norm_keys(a, keys).duplicated().sum())
        ds_ = int(_norm_keys(s, keys).duplicated().sum())
        verdict = "USABLE" if (ua and us) else "not unique"
        tests.append((keys, verdict, da, ds_))
    for keys, verdict, da, ds_ in tests:
        dup = "" if da is None else f"  dup rows: assign={da:,} summaries={ds_:,}"
        print(f"    {','.join(keys):<44} {verdict}{dup}")

    print(f"\n  chosen: {C.join_keys}")
    for n in C.notes:
        print(f"\n  note: {n}")
    for w in C.warnings:
        print(f"\n  WARNING: {w}")
    print(f"\n  join route: {C.join_route or 'not resolved'}")
    print("=" * 68)
    return 0 if C._joined is not None else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--freeze", action="store_true",
                    help="write a dated snapshot of the current values")
    ap.add_argument("--check", action="store_true",
                    help="compare current values to the snapshot, exit 1 on drift")
    ap.add_argument("--diagnose", action="store_true",
                    help="report artifact schemas and join-key uniqueness")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    if args.diagnose:
        return diagnose()
    if args.freeze:
        freeze()
        return 0
    if args.check:
        return 1 if check() else 0
    if args.json:
        print(json.dumps(C.to_dict(), indent=2))
        return 0
    _print_table()
    return 0


if __name__ == "__main__":
    sys.exit(main())