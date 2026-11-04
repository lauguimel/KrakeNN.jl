#!/usr/bin/env python3
r"""Dark docs figure — PLANAR STAGNATION-POINT (cross-slot strand) trace(C).

Reads ``ve_crossslot_fields.csv`` next to this script (the z-mid trace(C)
fields produced by ``make_crossslot_csv.jl`` with the REAL Kraken FVFD
log-conformation solver in the imposed planar-stagnation field) and renders
``ve-constitutive-models-crossslot.png`` on the live Vitepress dark page
colour (#1b1b1f) via the ``krakendark`` layer.

Layout:
  (top, 2x2)  the trace(C)(x,y) z-mid field for the 4 models on a SHARED
              colour scale + a single colourbar. The imposed strain
              u=(edot*x,-edot*y,0) is uniform, yet conformation advection
              builds a high-stretch STRAND along the outflow (x) axis and
              keeps the inflow (y) axis near equilibrium — Oldroyd-B the
              longest / most intense, the bounded closures shorter / weaker.
  (bottom)    trace(C) ALONG the outflow (x) axis through the stagnation
              point for the 4 models — the quantitative strand-length /
              intensity differentiation.

Reproduce (env ``kraken-v0-3-figures``):
    conda run -n kraken-v0-3-figures python plot_crossslot.py

  input  : ve_crossslot_fields.csv                  (next to this script)
  output : docs/src/users/benchmarks/ve-constitutive-models-crossslot.png
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

MODELS = [
    ("oldroydb", "Oldroyd-B"),
    ("fenep",    r"FENE-P  ($L^2=50$)"),
    ("giesekus", r"Giesekus  ($\alpha=0.2$)"),
    ("ptt",      r"PTT  ($\varepsilon=0.25$)"),
]
MARKERS = {"oldroydb": "o", "fenep": "s", "giesekus": "^", "ptt": "D"}


def load(name):
    """Return {model: (X, Y, trC field as (ny,nx) grid)} from the long CSV."""
    path = os.path.join(HERE, name)
    by_model = {}
    with open(path, newline="") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
        for r in csv.DictReader(lines):
            by_model.setdefault(r["model"], []).append(
                (int(r["i"]), int(r["j"]), float(r["x"]),
                 float(r["y"]), float(r["trC"])))
    out = {}
    for m, rows in by_model.items():
        ii = np.array([t[0] for t in rows])
        jj = np.array([t[1] for t in rows])
        nx, ny = ii.max(), jj.max()
        xg = np.full((ny, nx), np.nan)
        yg = np.full((ny, nx), np.nan)
        fg = np.full((ny, nx), np.nan)
        for (i, j, x, y, tc) in rows:
            xg[j - 1, i - 1] = x
            yg[j - 1, i - 1] = y
            fg[j - 1, i - 1] = tc
        out[m] = (xg, yg, fg)
    return out


def main():
    usetex = kd.apply()
    pal = kd.palette(4)
    colours = {m[0]: pal[i] for i, m in enumerate(MODELS)}

    data = load("ve_crossslot_fields.csv")

    # Shared colour scale across all 4 fields (positive scalar → 0..vmax).
    vmax = max(np.nanmax(data[m[0]][2]) for m in MODELS)
    vmin = 3.0  # equilibrium trace(C) = 3

    fig = plt.figure(figsize=(11.0, 11.6), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.85])

    pcm = None
    for idx, (name, label) in enumerate(MODELS):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        X, Y, F = data[name]
        pcm = ax.pcolormesh(X, Y, F, cmap="inferno", shading="auto",
                            vmin=vmin, vmax=vmax)
        ax.set_aspect("equal")
        ax.set_title(kd.tex(label), fontsize=12.5)
        # Mark the stagnation point + axis sense on the OB panel only.
        ax.plot(0, 0, "o", mfc="none", mec="0.85", ms=7, mew=1.3)
        if idx == 0:
            ax.annotate("", xy=(X.max() * 0.92, 0), xytext=(X.max() * 0.45, 0),
                        arrowprops=dict(arrowstyle="->", color="0.9", lw=1.6))
            ax.annotate("", xy=(0, Y.min() * 0.45), xytext=(0, Y.min() * 0.92),
                        arrowprops=dict(arrowstyle="->", color="0.6", lw=1.6))
            ax.text(X.max() * 0.6, X.max() * 0.10, "outflow (strand)",
                    color="0.92", fontsize=9.5, ha="left")
            ax.text(0.04 * X.max(), Y.min() * 0.75, "inflow",
                    color="0.65", fontsize=9.5, ha="left", rotation=90)
        if idx % 2 == 0:
            ax.set_ylabel(r"$y$  (inflow axis,  LU)")
        if idx // 2 == 1:
            ax.set_xlabel(r"$x$  (outflow axis,  LU)")

    cbar = fig.colorbar(pcm, ax=fig.axes[:4], fraction=0.04, pad=0.02,
                        location="right", shrink=0.9)
    kd.dark_colorbar(cbar)
    cbar.set_label(r"$\mathrm{tr}\,\mathbf{C}=C_{xx}+C_{yy}+C_{zz}$",
                   color="0.9")

    # --- bottom: trace(C) along the outflow (x) axis ----------------------
    axL = fig.add_subplot(gs[2, :])
    kd.grid(axL)
    for name, label in MODELS:
        X, Y, F = data[name]
        ny, nx = F.shape
        jc = ny // 2          # y = 0 row (stagnation point)
        x_line = X[jc, :]
        f_line = F[jc, :]
        c = colours[name]
        axL.plot(x_line, f_line, "-", color=kd.CONNECT, lw=1.4, zorder=2)
        axL.plot(x_line, f_line,
                 **kd.kraken_marker(c, marker=MARKERS[name], ms=7), zorder=4,
                 label=kd.tex(label))
    axL.axvline(0, color="0.5", ls=":", lw=1.0)
    axL.set_xlabel(r"$x$ along the outflow axis  (LU,  $y=0$)")
    axL.set_ylabel(r"$\mathrm{tr}\,\mathbf{C}$ on the strand")
    axL.set_title("Strand profile along the outflow axis")
    kd.dark_legend(axL, loc="upper center", ncol=4, title=None)

    fig.suptitle(
        kd.tex("Kraken viscoelastic models — planar stagnation-point "
               "(cross-slot) strand in $\\mathrm{tr}\\,\\mathbf{C}$"),
        fontsize=14.5, fontweight="bold")

    repo_root = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
    docs_dir = os.path.join(repo_root, "docs", "src", "users", "benchmarks")
    out = (os.path.join(docs_dir, "ve-constitutive-models-crossslot.png")
           if os.path.isdir(docs_dir)
           else os.path.join(HERE, "ve-constitutive-models-crossslot.png"))
    fig.savefig(out, dpi=140)
    print(f"wrote {out} (usetex={usetex})")


if __name__ == "__main__":
    main()
