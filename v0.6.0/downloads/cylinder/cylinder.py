#!/usr/bin/env python3
r"""Reproduce the Kraken.jl example-06 cylinder velocity-magnitude field figure.

Reads ``cylinder.csv`` (tidy columns: x, y, umag) sitting next to this script,
pivots it back to a 2-D lattice and draws the |u| heatmap (magma, the SAME field
the committed ``cylinder_umag.png`` plots), on the dark #1b1b1f Vitepress page
background. Re = 20.

    python cylinder.py               # -> cylinder_repro.png
    python cylinder.py -o /tmp/c.png # custom output path

Styling: uses the shared ``krakendark`` layer if it can be found (via
``$KRAKENDARK_DIR``, a repo ``viz/``, or the kraken-doc skill ``assets/``);
otherwise falls back to a clean self-contained dark matplotlib style, so the
script is portable for anyone who downloads it.

Env (Kraken docs): conda run -n kraken-v0-3-figures python cylinder.py
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
U_IN = 0.04  # inlet speed -> colour range capped at 1.5 * u_in (matches figure)


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
    """Pivot the tidy (x, y, umag) CSV into a Nx x Ny array + axis vectors."""
    xs, ys, vs = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            xs.append(int(float(row["x"])))
            ys.append(int(float(row["y"])))
            vs.append(float(row["umag"]))
    xs = np.array(xs); ys = np.array(ys); vs = np.array(vs)
    nx, ny = xs.max(), ys.max()
    field = np.full((ny, nx), np.nan)
    field[ys - 1, xs - 1] = vs  # rows = y, cols = x
    return np.arange(1, nx + 1), np.arange(1, ny + 1), field


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--csv", default=str(HERE / "cylinder.csv"),
                    help="input CSV (default: cylinder.csv next to this script)")
    ap.add_argument("-o", "--out", default=str(HERE / "cylinder_repro.png"),
                    help="output PNG path")
    args = ap.parse_args()

    x, y, field = load_field(pathlib.Path(args.csv))

    kd = _try_krakendark()
    if kd is not None:
        kd.apply(field=True)
    else:
        _apply_fallback_style()

    fig, ax = plt.subplots(figsize=(8.0, 3.5))
    pcm = ax.pcolormesh(x, y, field, cmap="magma", shading="auto",
                        vmin=0.0, vmax=1.5 * U_IN)
    cbar = fig.colorbar(pcm, ax=ax, fraction=0.018, pad=0.02)
    cbar.set_label(r"$|u|$")
    if kd is not None:
        kd.dark_colorbar(cbar)
    else:
        cbar.ax.tick_params(color="0.7", labelsize=9)
        cbar.outline.set_edgecolor("0.55")

    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title("Velocity magnitude — $Re = 20$")

    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
