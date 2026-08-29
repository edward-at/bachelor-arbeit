#!/usr/bin/env python3
"""curation_winner_dossier.py — the winner's comparative dossier (13.3).

Six figures, every one the same visual argument: the winner in the
curation colour over the frame as a light-gray ghost, so the reader sees
in one glance where C01 sits relative to the biased corpus without any
method or dataset being reprinted (duplication rule of Section 13.2).
The vs-methods evidence is carried by the finalist ECDF overlay, which
already exists. Everything here reads the frozen caches and recomputes
nothing.

  curation_winner_hist_grid.png        six signal densities, log y
  curation_winner_ecdf_grid.png        six signal ECDFs
  curation_winner_trajectory_fan.png   XY fans, frame | C01
  curation_winner_speed_fan.png        20 s speed-change fans, frame | C01
  curation_winner_phase_portraits.png  speed x a_long and speed x yaw
  curation_winner_w1_nullband.png      held-out W1 vs the K=100 SRS band

Requires the frame caches to include the winner group (run
curation_render_winner.py --frame-pass once if the assert fires).

Usage:  python curation_winner_dossier.py
"""

from __future__ import annotations

import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import curation_common as cc

GHOST = cc.GHOST
BLUE = cc.CURATION_COLOR
DT = 0.1

# Fan styling mirrored verbatim from trajectory_fan.py (Chapter 8 grammar):
# viridis coloured by the window's dynamics, 2-98 percentile normalisation,
# curvature-sorted draw order, robust P99 framing. All 5,000 cached fan
# windows per group are drawn, as in Chapter 8.
FAN_ALPHA = 0.16
FAN_LW = 0.8
FAN_CMAP = "viridis"
EXTENT_PCT = 99.0
SPEED_PCT = 99.0
XY_LABEL = "Absolute total heading change (rad)"
ST_LABEL = "Peak acceleration magnitude (m/s\u00b2)"

# the dist cache carries exactly these channels; 'regime' is excluded on
# purpose (categorical — its story is the regime-profile bar figure)
CACHE_CHANNELS = ["pc1", "pc2", "score",
                  "std_acc_mag", "std_speed_kmh", "std_yaw_deg"]
LABEL = {"pc1": "PC1", "pc2": "PC2", "score": "Dynamic score",
         "std_acc_mag": "\u03c3(acc_mag) [m/s\u00b2] *",
         "std_speed_kmh": "\u03c3(speed) [km/h]",
         "std_yaw_deg": "\u03c3(yaw) [\u00b0/s]"}
W1_METRICS = ["w1_mean_speed", "w1_std_speed", "w1_std_acc",
              "w1_std_yaw", "w1_peak_abs_yaw", "w1_peak_decel"]


def winner() -> str:
    if not cc.WINNER:
        raise SystemExit("[dossier] curation_common.WINNER unset")
    return cc.WINNER


def channel_values(dist: pd.DataFrame, method: str, ch: str) -> np.ndarray:
    v = dist[(dist["method"] == method)
             & (dist["channel"] == ch)]["value"].to_numpy(float)
    v = v[np.isfinite(v)]
    if ch == "yaw_rate":
        v = np.degrees(v)                # stored rad/s, displayed deg/s
    return v


def fig_hist_ecdf(dist: pd.DataFrame, win: str, outdir) -> None:
    chans = [c for c in CACHE_CHANNELS if c in set(dist["channel"])]
    for kind in ("hist", "ecdf"):
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
        for ax, ch in zip(axes.ravel(), chans):
            f = channel_values(dist, "R0_frame", ch)
            c = channel_values(dist, win, ch)
            lo = min(np.quantile(f, 0.001), np.quantile(c, 0.001))
            hi = max(np.quantile(f, 0.999), np.quantile(c, 0.999))
            if kind == "hist":
                bins = np.linspace(lo, hi, 80)
                ax.hist(f, bins=bins, density=True, color=GHOST,
                        edgecolor="black", linewidth=0.3, label="Frame")
                ax.hist(c, bins=bins, density=True, histtype="step",
                        color=BLUE, linewidth=1.4, label=win)
                ax.set_yscale("log")
                ax.set_ylabel("density (log)")
            else:
                for v, col, lw, lab in ((f, cc.FRAME_LINE, 2.6, "Frame"),
                                        (c, BLUE, 1.4, win)):
                    xs = np.sort(v)
                    ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs),
                            color=col, linewidth=lw, label=lab)
                ax.set_xlim(lo, hi)
                ax.set_ylabel("ECDF")
            ax.set_xlabel(LABEL.get(ch, ch))
            ax.set_axisbelow(True)
            ax.grid(True, alpha=0.4)
        axes[0, 0].legend()
        for ax in axes.ravel()[len(chans):]:
            ax.axis("off")
        fig.suptitle(f"{win} vs frame: per-channel "
                     f"{'densities' if kind == 'hist' else 'ECDFs'}")
        fig.tight_layout()
        out = outdir / f"curation_winner_{'hist' if kind == 'hist' else 'ecdf'}_grid.png"
        fig.savefig(out, dpi=cc.DPI)
        plt.close(fig)
        print(f"[dossier] wrote {out.name}")


def _norm(values):
    # verbatim from trajectory_fan.py
    from matplotlib.colors import Normalize
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return Normalize(0, 1)
    lo, hi = np.percentile(arr, [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    return Normalize(vmin=lo, vmax=hi)


def load_fan_records(win: str) -> dict:
    """Per group: one record per cached fan window, Chapter 8 fields."""
    idx = pd.read_parquet(cc.EVAL_DIR / "fan_index.parquet")
    frames = pd.read_parquet(cc.EVAL_DIR / "fan_frames.parquet")
    meta = json.load(open(cc.EVAL_DIR / "frame_pass_meta.json"))
    groups = meta.get("groups") or meta.get("methods")
    out = {}
    for g in ("R0_frame", win):
        if g not in groups:
            raise SystemExit(f"[dossier] group {g} missing from the frame "
                             f"caches. Run curation_render_winner.py "
                             f"--frame-pass once, then retry.")
        bit = groups.index(g)
        keys = idx[(idx["sample_mask"].to_numpy() >> bit) & 1 == 1]["key_id"]
        sub = frames[frames["key_id"].isin(keys.to_numpy())]
        recs, n_skip = [], 0
        for _, w in sub.groupby("key_id"):
            spd = w["speed_kmh"].to_numpy(float)
            if not np.isfinite(spd).any():
                n_skip += 1
                continue
            yaw = np.nan_to_num(w["yaw_rate"].to_numpy(float))
            acc = w["acc_mag"].to_numpy(float)
            t = (w["t"].to_numpy(float) if "t" in w.columns
                 else np.arange(len(spd)) * DT)
            t = t - t[0]
            v = spd / 3.6
            psi = np.cumsum(yaw * DT)
            recs.append({
                "t": t, "speed_kmh": spd,
                "x": np.cumsum(v * np.cos(psi) * DT),
                "y": np.cumsum(v * np.sin(psi) * DT),
                "heading_change": float(np.sum(np.abs(yaw)) * DT),
                "peak_acc": float(np.nanmax(acc)) if len(acc) else 0.0,
            })
        if n_skip:
            print(f"[dossier] {g}: skipped {n_skip} all-NaN fan windows")
        out[g] = recs
    return out


def fig_fans(win: str, outdir) -> None:
    from matplotlib.cm import ScalarMappable
    per = load_fan_records(win)
    cmap = plt.get_cmap(FAN_CMAP)

    # X-Y trajectory fan, frame | winner, shared norm and shared frame
    norm = _norm([r["heading_change"] for recs in per.values()
                  for r in recs])
    def fmax(a):
        f = np.abs(a)
        f = f[np.isfinite(f)]
        return float(f.max()) if f.size else None
    xs = [m for recs in per.values() for r in recs
          if (m := fmax(r["x"])) is not None]
    ys = [m for recs in per.values() for r in recs
          if (m := fmax(r["y"])) is not None]
    xext = np.percentile(xs, EXTENT_PCT) if xs else 100.0
    yext = np.percentile(ys, EXTENT_PCT) if ys else 100.0
    xlim, ylim = (-xext * 0.15, xext), (-yext, yext)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharey=True)
    for ax, (g, recs) in zip(axes, per.items()):
        # curvature_straight_top: curved drawn first, straight on top
        for r in sorted(recs, key=lambda r: r["heading_change"],
                        reverse=True):
            ax.plot(r["x"], r["y"], color=cmap(norm(r["heading_change"])),
                    linewidth=FAN_LW, alpha=FAN_ALPHA)
        ax.set_xlabel("X position (m)")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.set_title("Frame" if g == "R0_frame" else g)
    axes[0].set_ylabel("Y position (m)")
    fig.suptitle("20 s trajectory fans, frame and C01 "
                 "(identical axes and color scale)")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.tolist(), fraction=0.02,
                 pad=0.02).set_label(XY_LABEL)
    out = outdir / "curation_winner_trajectory_fan.png"
    fig.savefig(out, dpi=cc.DPI)
    plt.close(fig)
    print(f"[dossier] wrote {out.name}")

    # speed-change fan, frame | winner, coloured by peak |a|
    norm = _norm([r["peak_acc"] for recs in per.values() for r in recs])
    def base(a):
        f = a[np.isfinite(a)]
        return f[0] if f.size else 0.0
    dmax = [float(np.nanmax(np.abs(r["speed_kmh"] - base(r["speed_kmh"]))))
            for recs in per.values() for r in recs if r["speed_kmh"].size]
    dmax = [v for v in dmax if np.isfinite(v)]
    vmax = max(float(np.percentile(dmax, SPEED_PCT)), 1.0)
    tmax = max(float(r["t"][-1]) for recs in per.values() for r in recs)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)
    for ax, (g, recs) in zip(axes, per.items()):
        # scalar_extreme_top: hard-accel lines drawn last so they pop
        for r in sorted(recs, key=lambda r: r["peak_acc"]):
            ax.plot(r["t"], r["speed_kmh"] - base(r["speed_kmh"]),
                    color=cmap(norm(r["peak_acc"])),
                    linewidth=FAN_LW, alpha=FAN_ALPHA)
        ax.axhline(0, color=cc.FRAME_LINE, linewidth=1.4, zorder=1)
        ax.set_xlabel("Time within window (s)")
        ax.set_xlim(0, tmax)
        ax.set_ylim(-vmax * 1.05, vmax * 1.05)
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.set_title("Frame" if g == "R0_frame" else g)
    axes[0].set_ylabel("Speed change from start (km/h)")
    fig.suptitle("20 s speed-change fans, frame and C01 "
                 "(identical axes and color scale)")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.tolist(), fraction=0.02,
                 pad=0.02).set_label(ST_LABEL)
    out = outdir / "curation_winner_speed_fan.png"
    fig.savefig(out, dpi=cc.DPI)
    plt.close(fig)
    print(f"[dossier] wrote {out.name}")


def fig_portraits(win: str, outdir) -> None:
    z = np.load(cc.EVAL_DIR / "frame_portraits.npz", allow_pickle=True)
    methods = [str(m) for m in z["methods"]]
    for g in ("R0_frame", win):
        if g not in methods:
            raise SystemExit(f"[dossier] {g} missing from portraits — run "
                             f"curation_render_winner.py --frame-pass")
    # frame pass stores speed edges in m/s and yaw edges ALREADY in deg/s
    # (subset_frame_pass.py line 290); the first render mislabelled speed
    # and double-converted yaw to a +-3000 axis. Fixed here.
    se = z["speed_edges"] * 3.6                       # m/s -> km/h
    ae, ye = z["along_edges"], z["yaw_edges"]         # native units
    panels = [("Ha", ae, "a_long (m/s^2)"),
              ("Hy", ye, "yaw rate (deg/s)")]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2))
    for r, (key, edges, ylab) in enumerate(panels):
        H = z[key]
        for c, g in enumerate(("R0_frame", win)):
            h = H[methods.index(g)]
            ax = axes[r, c]
            im = ax.imshow(np.log10(h.T + 1), origin="lower",
                           aspect="auto", cmap="viridis",  # Ch.8 grammar
                           extent=[se[0], se[-1], edges[0], edges[-1]])
            ax.set_title(("Frame" if g == "R0_frame" else g)
                         if r == 0 else "")
            ax.set_xlabel("speed (km/h)" if r == 1 else "")
            ax.set_ylabel(ylab if c == 0 else "")
            fig.colorbar(im, ax=ax, label="log10(frames + 1)",
                         fraction=0.046)
    fig.suptitle(f"Frame-grain phase portraits, frame vs {win} "
                 f"(identical axes)")
    fig.tight_layout()
    out = outdir / "curation_winner_phase_portraits.png"
    fig.savefig(out, dpi=cc.DPI)
    plt.close(fig)
    print(f"[dossier] wrote {out.name}")


def fig_fidelity(win: str, outdir) -> None:
    """THE held-out no-damage evidence: per-regime |shift of mean
    sigma(speed)| against the 0.5 km/h P5 limit, with the R = 20
    replication spread. This, not the W1-to-frame distances, is the
    quantity the D4 tie-break is defined on."""
    d = cc.load_long()
    d = d[(d["tier"] == "budget") & (d["variant"] == "main")
          & (~d["sealed"]) & (d["method"] == win)]
    mets = sorted(m for m in set(d["metric"])
                  if re.match(r"shift_r\d_speed$", m))
    vals = [abs(float(d[d["metric"] == m]["value"].iloc[0])) for m in mets]
    x = np.arange(len(mets))
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.bar(x, vals, 0.55, color=BLUE, edgecolor="black", linewidth=0.5,
           label=f"{win} realised draw")
    rep_p = cc.EVAL_DIR / "curation_replication_budget.parquet"
    if rep_p.exists():
        rep = pd.read_parquet(rep_p)
        mu = [rep[m].abs().mean() for m in mets]
        lo = [rep[m].abs().min() for m in mets]
        hi = [rep[m].abs().max() for m in mets]
        ax.errorbar(x, mu, yerr=[np.subtract(mu, lo), np.subtract(hi, mu)],
                    fmt="o", color="black", markersize=4, capsize=4,
                    linewidth=1.2, label="replication mean (R=20, min-max)")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2,
               label="P5 limit (0.5 km/h)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"regime {i}" for i in range(len(mets))])
    ax.set_ylabel("|shift of mean \u03c3(speed)| (km/h)")
    ax.set_ylim(0, 0.62)
    ax.set_title(f"{win}: within-regime fidelity on the held-out channel")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = outdir / "curation_winner_fidelity.png"
    fig.savefig(out, dpi=cc.DPI)
    plt.close(fig)
    print(f"[dossier] wrote {out.name}")


def fig_w1_departure(win: str, outdir) -> None:
    """Departure PRICING, not a pass-fail test: any regime-rebalancing
    method must leave the SRS band on distances TO THE FRAME, because the
    frame is the biased reference and SRS reproduces it. The band is the
    do-nothing reference, and the bars price the intended departure in
    physical units."""
    d = cc.load_long()
    d = d[(d["tier"] == "budget") & (d["variant"] == "main")
          & (~d["sealed"])]
    mets = [m for m in W1_METRICS if m in set(d["metric"])]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6))
    for ax, met in zip(axes.ravel(), mets):
        band = cc.envelope_band(met)
        row = d[(d["method"] == win) & (d["metric"] == met)]["value"]
        if row.empty:
            ax.axis("off")
            continue
        v = float(row.iloc[0])
        if band:
            ax.axhspan(band[0], band[1], color=GHOST,
                       label="SRS band (do nothing)")
        ax.bar([0], [v], width=0.28, color=BLUE, edgecolor="black",
               linewidth=0.5, label=win)
        ax.set_xlim(-0.8, 0.8)
        ax.set_xticks([])
        top = max(v, band[1] if band else 0)
        ax.set_ylim(0, top * 1.18)
        ax.set_title(met.replace("w1_", "W1 "), fontsize=10)
        ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.4)
    axes[0, 0].legend(fontsize=8)
    for ax in axes.ravel()[len(mets):]:
        ax.axis("off")
    fig.suptitle(f"{win}: distance to the biased frame on held-out "
                 f"summaries, in physical units\n(the SRS band is the "
                 f"do-nothing reference, leaving it is the intended "
                 f"rebalancing, priced here)")
    fig.tight_layout()
    out = outdir / "curation_winner_w1_departure.png"
    fig.savefig(out, dpi=cc.DPI)
    plt.close(fig)
    print(f"[dossier] wrote {out.name}")


def main() -> None:
    win = winner()
    outdir = cc.ensure_curation_outdir()
    dist = pd.read_parquet(cc.EVAL_DIR / "dist_cache_budget.parquet")
    fig_hist_ecdf(dist, win, outdir)
    fig_fans(win, outdir)
    fig_portraits(win, outdir)
    fig_fidelity(win, outdir)
    fig_w1_departure(win, outdir)
    print("[dossier] done — seven comparative figures in figures/curation/. "
          "The vs-methods figure remains subset_ecdf_finalists.png.")


if __name__ == "__main__":
    main()