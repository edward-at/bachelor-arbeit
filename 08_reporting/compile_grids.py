#!/usr/bin/env python3
"""compile_grids.py: the per-method dossier of thesis Appendix F as five family grids
(Figures F.15 to F.19).

Tiles the existing per-method dossier PNGs of the sampling-method evaluation
UNCHANGED (no number is redrawn, every tile keeps its native pixels) into one
comparison grid per figure family, the frame first and then the slate M01 to
M11 in slate order:

  dossier_trajectory_fans.png       trajectory_fan_{frame,M01..M11}.png    4 rows x 3 columns
  dossier_speed_change_fans.png     speed_change_fan_{frame,M01..M11}.png  4 rows x 3 columns
  dossier_portraits_speed_along.png {frame,M01..M11}_speed_along.png       4 rows x 3 columns
  dossier_portraits_speed_yaw.png   {frame,M01..M11}_speed_yaw.png         4 rows x 3 columns
  dossier_exemplar_traces.png       timeseries_{frame,M01..M11}.png        6 rows x 2 columns

The C01 singles in the same folders are not tiled here (their dossier is the
winner family under 07_subset_curation/figures/curation/, Section 5.8).

Paths: tiles are read from the per-method singles in
07_subset_curation/figures/subset_eval/{fans,phase,timeseries}/ and written to
08_reporting/compiled_figures/. DOSSIER_SRC overrides the tile folder (flat
folders are searched as well) and DOSSIER_OUT overrides the output folder.
DATA_ROOT comes from thesis_paths.py and the environment variable
THESIS_DATA_ROOT.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
from thesis_paths import DATA_ROOT, REPO_ROOT  # noqa: E402
BA = REPO_ROOT
SRC = Path(os.environ.get("DOSSIER_SRC", BA / "07_subset_curation" / "figures" / "subset_eval"))
OUT = Path(os.environ.get("DOSSIER_OUT", BA / "08_reporting" / "compiled_figures"))
SUBDIRS = ["fans", "phase", "timeseries", "hist", ""]

GROUPS = ["frame"] + [f"M{i:02d}" for i in range(1, 12)]
GUTTER = 40          # white pixels between tiles
MARGIN = 40          # white border

FAMILIES = [
    ("dossier_trajectory_fans.png",       "trajectory_fan_{g}.png",  4, 3),
    ("dossier_speed_change_fans.png",     "speed_change_fan_{g}.png", 4, 3),
    ("dossier_portraits_speed_along.png", "{g}_speed_along.png",     4, 3),
    ("dossier_portraits_speed_yaw.png",   "{g}_speed_yaw.png",       4, 3),
    ("dossier_exemplar_traces.png",       "timeseries_{g}.png",      6, 2),
]


def find(name: str) -> Path:
    for sub in SUBDIRS:
        p = SRC / sub / name if sub else SRC / name
        if p.exists():
            return p
    raise SystemExit(f"[compile_5F_dossier] tile not found under {SRC}: {name}")


def compile_family(out_name: str, pattern: str, nrows: int, ncols: int) -> Path:
    tiles = [Image.open(find(pattern.format(g=g))).convert("RGB") for g in GROUPS]
    if len(tiles) != nrows * ncols:
        raise SystemExit(f"[compile_5F_dossier] {out_name}: {len(tiles)} tiles for a "
                         f"{nrows} x {ncols} grid")
    w = max(t.width for t in tiles)
    h = max(t.height for t in tiles)
    canvas = Image.new("RGB", (2 * MARGIN + ncols * w + (ncols - 1) * GUTTER,
                               2 * MARGIN + nrows * h + (nrows - 1) * GUTTER), "white")
    for k, t in enumerate(tiles):
        r, c = divmod(k, ncols)
        x = MARGIN + c * (w + GUTTER) + (w - t.width) // 2
        y = MARGIN + r * (h + GUTTER) + (h - t.height) // 2
        canvas.paste(t, (x, y))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / out_name
    canvas.save(out, dpi=(300, 300))
    print(f"[compile_5F_dossier] {out}  ({nrows} x {ncols} tiles of {w} x {h} px, "
          f"canvas {canvas.width} x {canvas.height} px)")
    return out


def main() -> None:
    if not SRC.exists():
        print(f"[compile_5F_dossier] source folder missing: {SRC}", file=sys.stderr)
        sys.exit(1)
    for spec in FAMILIES:
        compile_family(*spec)
    print("[compile_5F_dossier] done: 5 compiled figures from 60 tiles, nothing redrawn")


if __name__ == "__main__":
    main()