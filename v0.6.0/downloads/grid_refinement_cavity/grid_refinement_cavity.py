#!/usr/bin/env python3
r"""Reproduce the Kraken.jl example-20 grid-refinement cavity centreline figure.

Reads ``grid_refinement_cavity.csv`` (columns: y, ux_centerline) sitting next to
this script and plots the horizontal velocity along the vertical centreline of
the uniform 64x64 reference lid-driven cavity (the profile the refined run is
validated against), on the dark #1b1b1f Vitepress page background.

    python grid_refinement_cavity.py               # -> grid_refinement_cavity_repro.png
    python grid_refinement_cavity.py -o /tmp/g.png # custom output path

Styling: uses the shared ``krakendark`` layer if it can be found (via
``$KRAKENDARK_DIR``, a repo ``viz/``, or the kraken-doc skill ``assets/``);
otherwise falls back to a clean self-contained dark matplotlib style, so the
script is portable for anyone who downloads it.

Env (Kraken docs): conda run -n kraken-v0-3-figures python grid_refinement_cavity.py
"""
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
DARK = "#1b1b1f"
COOL = "#4ea1d3"


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


def load_csv(path: pathlib.Path):
    y, ux = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            y.append(float(row["y"]))
            ux.append(float(row["ux_centerline"]))
    return y, ux


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--csv", default=str(HERE / "grid_refinement_cavity.csv"),
                    help="input CSV (default: grid_refinement_cavity.csv next to this script)")
    ap.add_argument("-o", "--out", default=str(HERE / "grid_refinement_cavity_repro.png"),
                    help="output PNG path")
    args = ap.parse_args()

    y, ux = load_csv(pathlib.Path(args.csv))

    kd = _try_krakendark()
    if kd is not None:
        kd.apply()
    else:
        _apply_fallback_style()

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    if kd is not None:
        kd.grid(ax)
    else:
        ax.grid(True, color="0.45", alpha=0.4, lw=0.6)

    ax.plot(ux, y, "-o", color=COOL, mec="0.92", mew=0.8, ms=5, lw=2.0,
            label="Uniform $64^2$ reference")

    ax.set_xlabel(r"$u_x / u_\mathrm{lid}$")
    ax.set_ylabel(r"$y / L$")
    ax.set_title("Lid-driven cavity — vertical centreline")

    if kd is not None:
        kd.dark_legend(ax, loc="upper left")
    else:
        ax.legend(loc="upper left", facecolor=DARK, edgecolor="0.5",
                  labelcolor="0.9", framealpha=0.85)

    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
