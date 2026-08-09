#!/usr/bin/env python3
"""Per-method trajectory & speed-change fans (Section 8.4 grammar).

Two figures per group (X-Y trajectory fan, speed-change fan) reconstructed
from the seeded FAN_N-per-method window sequences cached by
subset_frame_pass.py — the mixed-source adaptation the plan called the one
real coding task. Dead-reckoning, colouring, ordering, robust framing, shared
norms and shared extents are verbatim from the per-dataset originals; the only
change is the unit of grouping (sampling method, not dataset), so every fan
page is directly comparable across methods and, at Section 12.3, against the
curation rows.
Reads ONLY the fan caches. Outputs: figures/subset_eval/fans/…
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd

from subset_eval_common import DPI, EVAL_DIR, ROW_LABELS, ensure_outdir

WINDOW_S = 20.0
ALPHA, LINEWIDTH = 0.16, 0.8
EXTENT_PCT, SPEED_PCT = 99.0, 99.0
COLORMAP = "viridis"
SINGLE_IN_XY, SINGLE_IN_SPEED = (8.0, 5.0), (8.0, 4.5)
GROUPS = ["R0_frame"] + [f"M{i:02d}" for i in range(1, 12)]


def _norm(vals):
    a = np.asarray(vals, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return Normalize(0, 1)
    lo, hi = np.percentile(a, [2, 98])
    return Normalize(lo, hi if hi > lo else lo + 1e-6)


def reconstruct(seq):
    """One window's frames -> dead-reckoned x,y,t,speed + summary scalars."""
    t = seq["t"].to_numpy(float)
    order = np.argsort(t)
    t = t[order]
    v = seq["speed_kmh"].to_numpy(float)[order] / 3.6
    yaw = seq["yaw_rate"].to_numpy(float)[order]      # rad/s
    acc = seq["acc_mag"].to_numpy(float)[order]
    m = np.isfinite(v) & np.isfinite(yaw) & np.isfinite(t)
    if m.sum() < 5:
        return None
    v, yaw, t = v[m], yaw[m], t[m]
    acc = acc[m]
    dt = np.gradient(t)
    tl = t - t[0]
    theta = np.cumsum(yaw * dt); theta -= theta[0]
    x = np.cumsum(v * np.cos(theta) * dt); x -= x[0]
    y = np.cumsum(v * np.sin(theta) * dt); y -= y[0]
    if not np.isfinite(acc).any():
        acc = np.gradient(v, tl) if len(tl) > 2 else np.zeros_like(v)
    acc = np.nan_to_num(acc)
    return dict(x=x.astype(np.float32), y=y.astype(np.float32),
                t=tl.astype(np.float32), speed_kmh=(v * 3.6).astype(np.float32),
                heading_change=float(np.sum(np.abs(yaw) * dt)),
                peak_acc=float(np.max(np.abs(acc))) if acc.size else 0.0)


def build(frames, idx):
    per = {g: [] for g in GROUPS}
    for kid, seq in frames.groupby("key_id"):
        rec = reconstruct(seq)
        if rec is None:
            continue
        mask = int(idx.loc[kid, "sample_mask"])
        for b, g in enumerate(GROUPS):
            if (mask >> b) & 1:
                per[g].append(rec)
    return {g: r for g, r in per.items() if r}


def xy_fan(per, out_dir):
    cmap = plt.get_cmap(COLORMAP)
    allr = [r for rs in per.values() for r in rs]
    norm = _norm([r["heading_change"] for r in allr])
    xmax = max(np.percentile([float(r["x"].max()) for r in allr],
                             EXTENT_PCT), 1.0)
    ymax = max(np.percentile([np.abs(r["y"]).max() for r in allr],
                             EXTENT_PCT), 1.0)
    xlim, ylim = (-0.05 * xmax, xmax * 1.05), (-ymax * 1.05, ymax * 1.05)
    for g, recs in per.items():
        fig, ax = plt.subplots(figsize=SINGLE_IN_XY)
        for r in sorted(recs, key=lambda r: r["heading_change"], reverse=True):
            ax.plot(r["x"], r["y"], color=cmap(norm(r["heading_change"])),
                    linewidth=LINEWIDTH, alpha=ALPHA)
        ax.set_xlabel("X position (m)")
        ax.set_ylabel("Y position (m)")
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.grid(True); ax.set_axisbelow(True)
        sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02).set_label(
            "Absolute total heading change (rad)")
        ax.set_title(f"Trajectory distribution — {ROW_LABELS[g]} "
                     f"(window = {WINDOW_S:g}s)")
        rid = "frame" if g == "R0_frame" else g
        fig.savefig(out_dir / f"trajectory_fan_{rid}.png", dpi=DPI,
                    bbox_inches="tight")
        plt.close(fig)


def speed_fan(per, out_dir):
    cmap = plt.get_cmap(COLORMAP)
    allr = [r for rs in per.values() for r in rs]
    norm = _norm([r["peak_acc"] for r in allr])
    dvals = [float(np.abs(r["speed_kmh"] - r["speed_kmh"][0]).max())
             for r in allr if r["speed_kmh"].size]
    vmax = max(float(np.percentile(dvals, SPEED_PCT)), 1.0)
    for g, recs in per.items():
        fig, ax = plt.subplots(figsize=SINGLE_IN_SPEED)
        for r in sorted(recs, key=lambda r: r["peak_acc"]):
            ax.plot(r["t"], r["speed_kmh"] - r["speed_kmh"][0],
                    color=cmap(norm(r["peak_acc"])), linewidth=LINEWIDTH,
                    alpha=ALPHA)
        ax.axhline(0, color="0.5", linewidth=0.8, zorder=1)
        ax.set_xlabel("Time within window (s)")
        ax.set_ylabel("Speed change from start (km/h)")
        ax.set_xlim(0, WINDOW_S); ax.set_ylim(-vmax * 1.05, vmax * 1.05)
        ax.grid(True); ax.set_axisbelow(True)
        sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02).set_label(
            "Peak acceleration magnitude (m/s\u00b2)")
        ax.set_title(f"Speed change per {WINDOW_S:g}s window ({ROW_LABELS[g]})")
        rid = "frame" if g == "R0_frame" else g
        fig.savefig(out_dir / f"speed_change_fan_{rid}.png", dpi=DPI,
                    bbox_inches="tight")
        plt.close(fig)


def main():
    out_dir = ensure_outdir() / "fans"
    out_dir.mkdir(exist_ok=True)
    frames = pd.read_parquet(EVAL_DIR / "fan_frames.parquet")
    idx = pd.read_parquet(EVAL_DIR / "fan_index.parquet").set_index("key_id")
    if frames.empty:
        raise SystemExit("fan_frames empty — run subset_frame_pass.py first")
    per = build(frames, idx)
    xy_fan(per, out_dir)
    speed_fan(per, out_dir)
    print(f"  [SAVED] fans for {len(per)} groups")


if __name__ == "__main__":
    main()