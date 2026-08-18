#!/usr/bin/env python3
"""F9 — the method fingerprint scorecard (budget tier).

Layout mechanics mirror bias_fingerprint.py exactly: imshow Blues, per-column
min-max shading, real values printed with per-column formats, facet-group
separators, the group-label band INSIDE the axes above row 0 (ylim headroom +
framing hlines — the construction that prevents the title/label collision),
and a slim colorbar. One deliberate difference from Chapter 10, stated:
shading here is UNORIENTED (no darker-equals-worse) because orienting columns
before the decision rule exists (D4) would pre-judge it; the printed value
and the tautology mark carry the truth. No composite index, deliberately.
Rows: Frame and guard-cost context rows above a rule, then the fixed slate.
Marks: small triangle = construction-determined, small circle = influenced, blank =
held-out (tautology_map.csv). The asterisk stays reserved for the
confounded-accel convention, which is why it is not used here.
Reads ONLY the eval cache. Output: figures/subset_eval/subset_fingerprint.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from subset_eval_common import (DPI, ROW_LABELS, TIER, BUDGET_N,
                                ensure_outdir, load_cls, load_master,
                                present_methods)

# tautology marks (local by design: the only consumer of these symbols).
# small triangle = construction-determined, small circle = influenced.
# The asterisk stays reserved for the confounded-accel convention.
MARKS = {"T": "\u25b4", "P": "\u2022", "O": "", "N": ""}

CELL_FS = 7    # documented heatmap exception: in-cell + group-label font
# Cell text is one string: value, one space, mark — a single text object has
# one fontsize, so the mark matches the numbers; raise CELL_FS to lift both.

# key, on-figure label, facet group, format
COLUMNS = [
    ("low_dyn_share",   "LowDyn",        "Regime balance", "%.3f"),
    ("entropy_norm",    "H (norm)",      "Regime balance", "%.3f"),
    ("d2_regimes",      "$D_2$ reg",     "Regime balance", "%.2f"),
    ("js_to_frame",     "JS frame",      "Regime balance", "%.4f"),
    ("js_to_uniform",   "JS unif",       "Regime balance", "%.4f"),
    ("w1_std_speed",    "\u03c3 Spd",    "W1 to frame",    "%.2f"),
    ("w1_std_acc",      "\u03c3 Acc*",   "W1 to frame",    "%.3f"),
    ("w1_std_yaw",      "\u03c3 Yaw",    "W1 to frame",    "%.2f"),
    ("w1_mean_speed",   "MeanSpd",       "W1 to frame",    "%.2f"),
    ("w1_peak_decel",   "PkDec*",        "W1 to frame",    "%.3f"),
    ("w1_peak_abs_yaw", "Pk|Yaw|",       "W1 to frame",    "%.2f"),
    ("mean_score_ratio","Score\u00d7",   "Dynamic",        "%.2f"),
    ("gini_sig_speed",  "Gini \u03c3Spd","Dynamic",        "%.3f"),
    ("volume_gini",     "Vol Gini",      "Source",         "%.2f"),
    ("d2_sources",      "$D_2$ src",     "Source",         "%.2f"),
    ("top_source_share","TopSrc",        "Source",         "%.3f"),
    ("worst_regime_top_source", "WorstReg", "Source",      "%.3f"),
    ("ret_hard_decel",  "Decel*",        "Retention",      "%.2f"),
    ("ret_high_yaw",    "Yaw",           "Retention",      "%.2f"),
    ("ret_high_speed",  "Speed",         "Retention",      "%.2f"),
    ("ret_stop_go",     "StopGo",        "Retention",      "%.2f"),
    ("ret_near_stat",   "Stat",          "Retention",      "%.2f"),
    ("ret_top_score",   "TopSc*",        "Retention",      "%.2f"),
    ("pc_coverage_share","PC cells",     "Coverage",       "%.3f"),
    ("cell_coverage_share","Reg\u00d7Src","Coverage",      "%.3f"),
]


def main() -> None:
    out_dir = ensure_outdir()
    mt = load_master(TIER)
    cls = load_cls()
    rows = ["R0_frame", "R1_guarded"] + present_methods(mt)

    cols = [c for c in COLUMNS if c[0] in mt.columns]
    keys = [c[0] for c in cols]
    raw = mt.loc[rows, keys].to_numpy(dtype=float)

    inten = np.full_like(raw, np.nan)
    for j in range(raw.shape[1]):
        col = raw[:, j]
        finite = np.isfinite(col)
        if not finite.any():
            continue
        lo, hi = np.nanmin(col), np.nanmax(col)
        t = (col - lo) / (hi - lo) if hi > lo else np.zeros_like(col)
        t[~finite] = np.nan
        inten[:, j] = t

    n_r, n_c = raw.shape
    fig_w = max(12.0, 0.82 * n_c + 3.0)
    fig_h = max(6.5, 0.46 * n_r + 2.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(np.nan_to_num(inten, nan=0.0), cmap="Blues",
                   vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(n_c))
    ax.set_xticklabels([c[1] for c in cols], rotation=0, fontsize=CELL_FS + 1)
    ax.set_yticks(np.arange(n_r))
    ax.set_yticklabels([ROW_LABELS[r] for r in rows])
    ax.set_ylim(n_r - 0.5, -1.5)          # headroom band for group labels

    for i, r in enumerate(rows):
        for j, (key, _, _, fmt) in enumerate(cols):
            v = raw[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "\u2013", ha="center", va="center",
                        fontsize=CELL_FS)
                continue
            mark = MARKS.get(cls.get((key, r), ""), "")
            shade = inten[i, j]
            tc = "white" if np.isfinite(shade) and shade > 0.55 else "black"
            txt = (fmt % v) + (" " + mark if mark else "")
            ax.text(j, i, txt, ha="center", va="center",
                    color=tc, fontsize=CELL_FS)

    spans, start = [], 0
    for j in range(1, n_c + 1):
        if j == n_c or cols[j][2] != cols[start][2]:
            spans.append((cols[start][2], start, j - 1))
            start = j
    for _, a, _ in spans[1:]:
        ax.axvline(a - 0.5, color="black", linewidth=1)
    ax.axhline(-1.5, color="black", linewidth=1)
    ax.axhline(-0.5, color="black", linewidth=1)
    for name, a, b in spans:
        ax.text((a + b) / 2.0, -0.95, name, ha="center", va="center",
                fontsize=CELL_FS + 1)
    ax.axhline(1.5, color="black", linewidth=1.2)   # context rows | methods

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.015)
    cbar.set_label("column min\u2013max (unoriented)")

    ax.set_title(f"Method fingerprint "
                 f"(\u25b4=construction-determined),  (\u2022=influenced),  "
                 f"(blank=held-out), (no composite index)")
    fig.tight_layout()
    out = out_dir / "subset_fingerprint.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  [SAVED] {out}")


if __name__ == "__main__":
    main()