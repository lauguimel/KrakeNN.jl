#!/usr/bin/env python3
r"""Dark docs figure — IN-FLOW (coupled Poiseuille) trace(C) for the 4 models.

Reads ``ve_poiseuille_profiles.csv`` next to this script (the coupled
planar-Poiseuille y-profiles produced by ``make_poiseuille_csv.jl`` with the
REAL Kraken FVFD log-conformation drivers) and renders
``ve-constitutive-models-poiseuille.png`` on the live Vitepress dark page
colour (#1b1b1f) via the ``krakendark`` layer.

Two-panel story (channel half-width, wall -> centre):
  (left)  trace(C)(y) = (C_xx + C_yy + C_zz)(y). All four collapse to tr C = 3
          at the low-shear centre, then separate in the near-wall high-shear
          band: Oldroyd-B stretches highest (unbounded), FENE-P / Giesekus /
          PTT are thinned / bounded below it.
  (right) the coupled velocity profile u(y) — slightly fuller (higher peak) for
          the shear-thinning closures, since the thinned near-wall polymer
          stress lets the solvent shear move faster.

Reproduce (env ``kraken-v0-3-figures``):
    conda run -n kraken-v0-3-figures python plot_poiseuille.py

  input  : ve_poiseuille_profiles.csv          (next to this script)
  output : docs/src/users/benchmarks/ve-constitutive-models-poiseuille.png
           (written directly into the docs page; committed PNG, do not rename)
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Find krakendark: the docs skill assets, or a repo viz/ copy.
for cand in (
    os.path.expanduser("~/.claude/skills/kraken-doc/assets"),
    os.path.join(HERE, "..", "..", "..", "viz"),
    os.path.join(HERE, "..", "..", "viz"),
):
    if os.path.isfile(os.path.join(cand, "krakendark.py")):
        sys.path.insert(0, cand)
        break
import krakendark as kd  # noqa: E402

# Model display order, label and marker (colour from the bright palette).
MODELS = [
    ("oldroydb", "Oldroyd-B"),
    ("fenep",    r"FENE-P  ($L^2=50$)"),
    ("giesekus", r"Giesekus  ($\alpha=0.2$)"),
    ("ptt",      r"PTT  ($\varepsilon=0.25$)"),
]
MARKERS = {"oldroydb": "o", "fenep": "s", "giesekus": "^", "ptt": "D"}


def load(name):
    """Return {model: (y[], u[], trC[]) on the wall->centre half-channel}."""
    path = os.path.join(HERE, name)
    raw = {}
    with open(path, newline="") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
        for r in csv.DictReader(lines):
            raw.setdefault(r["model"], []).append(
                (int(r["j"]), float(r["u"]), float(r["trC"])))
    out = {}
    for model, rows in raw.items():
        rows.sort(key=lambda t: t[0])
        arr = np.array(rows, dtype=float)
        j = arr[:, 0]
        ny = int(j.max())
        # Half-channel: keep wall (j=1) .. centre (j=ny/2); y in wall-distance.
        half = j <= ny // 2
        # y = wall-normal distance in cell units (0 at the wall plane).
        y = j[half] - 0.5
        out[model] = (y, arr[half, 1], arr[half, 2])
    return out


def main():
    usetex = kd.apply()
    pal = kd.palette(4)
    colours = {m[0]: pal[i] for i, m in enumerate(MODELS)}

    data = load("ve_poiseuille_profiles.csv")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 5.6),
                                   constrained_layout=True)

    # --- left: trace(C) vs wall distance ------------------------------------
    kd.grid(axL)
    for name, _label in MODELS:
        y, _u, trc = data[name]
        c = colours[name]
        axL.plot(y, trc, "-", color=kd.CONNECT, lw=1.6, zorder=2)
        axL.plot(y, trc, **kd.kraken_marker(c, marker=MARKERS[name], ms=8),
                 zorder=4)
    axL.axhline(3.0, color="0.5", ls=":", lw=1.4, zorder=1)
    axL.annotate(r"equilibrium  $\mathrm{tr}\,C = 3$",
                 (data["oldroydb"][0].max(), 3.0), textcoords="offset points",
                 xytext=(-6, 6), ha="right", fontsize=11, color="0.7")
    axL.set_xlabel(r"wall-normal distance  $y$  (lattice units, $0$ = wall)")
    axL.set_ylabel(r"polymer stretch  $\mathrm{tr}\,C = C_{xx}+C_{yy}+C_{zz}$")
    axL.set_title("Conformation across the channel half-width")

    # --- right: coupled velocity profile u(y) -------------------------------
    kd.grid(axR)
    for name, _label in MODELS:
        y, u, _trc = data[name]
        c = colours[name]
        axR.plot(y, u, "-", color=kd.CONNECT, lw=1.6, zorder=2)
        axR.plot(y, u, **kd.kraken_marker(c, marker=MARKERS[name], ms=8),
                 zorder=4)
    axR.set_xlabel(r"wall-normal distance  $y$  (lattice units, $0$ = wall)")
    axR.set_ylabel(r"streamwise velocity  $u(y)$")
    axR.set_title("Coupled velocity profile")

    # --- shared legend ------------------------------------------------------
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=colours[m[0]], lw=2.0, marker=MARKERS[m[0]],
               mfc=colours[m[0]], mec="0.92", ms=8, label=m[1])
        for m in MODELS
    ]
    kd.dark_legend(axL, handles=handles, loc="upper right", title="Model")

    fig.suptitle(
        kd.tex("Kraken viscoelastic models in a coupled Poiseuille channel "
               "(Wi_wall ~ 1, beta = 0.5) — near-wall stretch separation"),
        fontsize=14.5, fontweight="bold")

    # Write straight into the docs page location (repo_root = HERE/../../../..).
    repo_root = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
    docs_dir = os.path.join(repo_root, "docs", "src", "users", "benchmarks")
    out = (os.path.join(docs_dir, "ve-constitutive-models-poiseuille.png")
           if os.path.isdir(docs_dir)
           else os.path.join(HERE, "ve-constitutive-models-poiseuille.png"))
    fig.savefig(out, dpi=150)
    print(f"wrote {out} (usetex={usetex})")


if __name__ == "__main__":
    main()
