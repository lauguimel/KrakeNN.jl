#!/usr/bin/env python3
r"""Reproduce the Kraken.jl example-11 Zalesak-disk VOF field figure.

Reads ``zalesak.csv`` (tidy columns: x, y, C) sitting next to this script,
pivots it back to a 2-D lattice and draws the volume-fraction heatmap (the final
``C`` field after one full rigid-body rotation), on the dark #1b1b1f Vitepress
page background.

    python zalesak.py               # -> zalesak_repro.png
    python zalesak.py -o /tmp/z.png # custom output path

Styling: uses the shared ``krakendark`` layer if it can be found (via
``$KRAKENDARK_DIR``, a repo ``viz/``, or the kraken-doc skill ``assets/``);
otherwise falls back to a clean self-contained dark matplotlib style, so the
script is portable for anyone who downloads it.

Env (Kraken docs): conda run -n kraken-v0-3-figures python zalesak.py
"""
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
DARK = "#1b1b1f"


def _try_krakendark():
    """Locate and import krakendark; return the module or None."""
    candidates = []
    env = os.environ.get("KRAKENDARK_DIR")
    if env:
        candidates.append(pathlib.Path(env))
    for d in [HERE, *HERE.parents]:
        candidates += [d, d / "viz", d / "assets"]
    candidates.append(pathlib.Path.home() / ".claude" / "skills" / "kraken-doc" / "assets")
    for c in candidates:
        if (c / "krakendark.py").exists():
            sys.path.insert(0, str(c))
            try:
                import krakendark as kd  # noqa
                return kd
            except Exception:
                return None
    return None


def _apply_fallback_style():
    """Clean dark style if krakendark is unavailable."""
    plt.rcParams.update({
        "figure.facecolor": DARK,
        "axes.facecolor": DARK,
        "savefig.facecolor": DARK,
        "text.color": "0.92",
        "axes.labelcolor": "0.92",
        "axes.edgecolor": "0.55",
        "xtick.color": "0.85",
        "ytick.color": "0.85",
        "axes.titlecolor": "0.92",
        "font.family": "serif",
        "figure.dpi": 130,
    })


def load_field(path: pathlib.Path):
    """Pivot the tidy (x, y, C) CSV into a Nx x Ny array + axis vectors."""
    xs, ys, vs = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            xs.append(int(float(row["x"])))
            ys.append(int(float(row["y"])))
            vs.append(float(row["C"]))
    xs = np.array(xs); ys = np.array(ys); vs = np.array(vs)
    nx, ny = xs.max(), ys.max()
    field = np.full((ny, nx), np.nan)
    field[ys - 1, xs - 1] = vs  # rows = y, cols = x
    return np.arange(1, nx + 1), np.arange(1, ny + 1), field


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--csv", default=str(HERE / "zalesak.csv"),
                    help="input CSV (default: zalesak.csv next to this script)")
    ap.add_argument("-o", "--out", default=str(HERE / "zalesak_repro.png"),
                    help="output PNG path")
    args = ap.parse_args()

    x, y, field = load_field(pathlib.Path(args.csv))

    kd = _try_krakendark()
    if kd is not None:
        kd.apply(field=True)
    else:
        _apply_fallback_style()

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    pcm = ax.pcolormesh(x, y, field, cmap="viridis", shading="auto",
                        vmin=0.0, vmax=1.0)
    ax.contour(x, y, field, levels=[0.5], colors="#ff6b6b", linewidths=1.2)
    cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(r"$C$")
    if kd is not None:
        kd.dark_colorbar(cbar)
    else:
        cbar.ax.tick_params(color="0.7", labelsize=9)
        cbar.outline.set_edgecolor("0.55")

    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title("Zalesak disk — VOF field after one rotation")

    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
