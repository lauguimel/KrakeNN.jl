#!/usr/bin/env python3
r"""Reproduce the Kraken.jl example-07 heat-conduction temperature-profile figure.

Reads ``heat_conduction.csv`` (columns: y_over_H, T_analytic, T_kraken) sitting
next to this script and plots the analytical linear conduction profile (line)
against the Kraken thermal-LBM solution (markers), on the dark #1b1b1f Vitepress
page background.

    python heat_conduction.py               # -> heat_conduction_repro.png
    python heat_conduction.py -o /tmp/h.png # custom output path

Styling: uses the shared ``krakendark`` layer if it can be found (via
``$KRAKENDARK_DIR``, a repo ``viz/``, or the kraken-doc skill ``assets/``);
otherwise falls back to a clean self-contained dark matplotlib style, so the
script is portable for anyone who downloads it.

Env (Kraken docs): conda run -n kraken-v0-3-figures python heat_conduction.py
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
ACCENT = "#ff6b6b"
COOL = "#4ea1d3"


def _try_krakendark():
    """Locate and import krakendark; return the module or None."""
    candidates = []
    env = os.environ.get("KRAKENDARK_DIR")
    if env:
        candidates.append(pathlib.Path(env))
    # Walk up from this file looking for a viz/ or assets/ carrying krakendark.py.
    for d in [HERE, *HERE.parents]:
        candidates += [d, d / "viz", d / "assets"]
    # The kraken-doc skill assets/ (developer machine fallback).
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
    y, t_ana, t_num = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            y.append(float(row["y_over_H"]))
            t_ana.append(float(row["T_analytic"]))
            t_num.append(float(row["T_kraken"]))
    return y, t_ana, t_num


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--csv", default=str(HERE / "heat_conduction.csv"),
                    help="input CSV (default: heat_conduction.csv next to this script)")
    ap.add_argument("-o", "--out", default=str(HERE / "heat_conduction_repro.png"),
                    help="output PNG path")
    args = ap.parse_args()

    y, t_ana, t_num = load_csv(pathlib.Path(args.csv))

    kd = _try_krakendark()
    if kd is not None:
        kd.apply()
        line_c, mark_c = COOL, ACCENT
    else:
        _apply_fallback_style()
        line_c, mark_c = COOL, ACCENT

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    if kd is not None:
        kd.grid(ax)
    else:
        ax.grid(True, color="0.45", alpha=0.4, lw=0.6)

    ax.plot(t_ana, y, "-", color=line_c, lw=2.0, label="Analytical (linear)", zorder=1)
    ax.plot(t_num, y, "o", color=mark_c, mec="0.92", mew=1.0, ms=7,
            ls="none", label="Kraken", zorder=2)

    ax.set_xlabel(r"Temperature")
    ax.set_ylabel(r"$y / H$")
    ax.set_title("Heat conduction — $Ra = 100$ (sub-critical)")

    if kd is not None:
        kd.dark_legend(ax, loc="upper right")
    else:
        ax.legend(loc="upper right", facecolor=DARK, edgecolor="0.5",
                  labelcolor="0.9", framealpha=0.85)

    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
