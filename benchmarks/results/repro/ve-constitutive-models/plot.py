#!/usr/bin/env python3
r"""Dark docs figure — the 4 Kraken viscoelastic constitutive models.

Reads ``ve_constitutive_models.csv`` next to this script (the single-cell
sweep produced by ``make_csv.jl`` with the REAL Kraken log-conformation
constitutive solver) and renders ``ve-constitutive-models.png`` on the live
Vitepress dark page colour (#1b1b1f) via the ``krakendark`` layer.

Two-panel story (Oldroyd-B vs the 3 bounded / shear-thinning closures):
  (left)  steady SIMPLE SHEAR — first normal-stress difference
          N1 = G(C_xx - C_yy) vs Wi = lambda*gammadot, log-log. Oldroyd-B is
          the unbounded quadratic N1 = 2 Wi^2 (constant viscosity); FENE-P,
          Giesekus and PTT shear-thin and fall below it, each with a distinct
          slope.
  (right) steady PLANAR EXTENSION — C_xx vs lambda*edot in [0, 0.49].
          Oldroyd-B diverges at the coil-stretch pole lambda*edot = 0.5
          (C_xx = 1/(1-2 lambda edot)); FENE-P, Giesekus and PTT saturate at
          distinct finite plateaus (finite extensibility / bounded stretch).

Reproduce (env ``kraken-v0-3-figures``):
    conda run -n kraken-v0-3-figures python plot.py

  input  : ve_constitutive_models.csv            (next to this script)
  output : docs/src/users/benchmarks/ve-constitutive-models.png
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

# Model display order, label and a stable colour from the bright palette.
MODELS = [
    ("oldroydb", "Oldroyd-B"),
    ("fenep",    r"FENE-P  ($L^2=50$)"),
    ("giesekus", r"Giesekus  ($\alpha=0.2$)"),
    ("ptt",      r"PTT  ($\varepsilon=0.25$)"),
]


def load(name):
    """Return {(flow, model): (control[], Cxx[], Cyy[], N1[])} sorted by control."""
    path = os.path.join(HERE, name)
    data = {}
    with open(path, newline="") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
        for r in csv.DictReader(lines):
            key = (r["flow"], r["model"])
            data.setdefault(key, []).append(
                (float(r["control"]), float(r["Cxx"]),
                 float(r["Cyy"]), float(r["N1"])))
    out = {}
    for key, rows in data.items():
        rows.sort(key=lambda t: t[0])
        arr = np.array(rows)
        out[key] = (arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3])
    return out


def main():
    usetex = kd.apply()
    pal = kd.palette(4)
    colours = {m[0]: pal[i] for i, m in enumerate(MODELS)}
    markers = {"oldroydb": "o", "fenep": "s", "giesekus": "^", "ptt": "D"}

    data = load("ve_constitutive_models.csv")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 5.6),
                                   constrained_layout=True)

    # --- left: simple shear, N1 vs Wi (log-log) -----------------------------
    kd.grid(axL)
    axL.set_xscale("log")
    axL.set_yscale("log")
    for name, _label in MODELS:
        wi, _cxx, _cyy, n1 = data[("shear", name)]
        # drop Wi=0 (log axis); N1(0)=0.
        m = wi > 0
        c = colours[name]
        axL.plot(wi[m], n1[m], "-", color=kd.CONNECT, lw=1.6, zorder=2)
        axL.plot(wi[m], n1[m], **kd.kraken_marker(c, marker=markers[name], ms=9),
                 zorder=4)
    # Oldroyd-B quadratic guide line N1 = 2 Wi^2.
    wig = np.array([0.05, 10.0])
    axL.plot(wig, 2 * wig**2, ":", color="0.55", lw=1.4, zorder=1)
    axL.annotate(r"$N_1 = 2\,\mathrm{Wi}^2$ (Oldroyd-B)",
                 (6.0, 2 * 6.0**2), textcoords="offset points",
                 xytext=(-4, 10), ha="right", fontsize=11, color="0.7")
    axL.set_xlabel(r"Weissenberg number  $\mathrm{Wi}=\lambda\dot\gamma$")
    axL.set_ylabel(r"first normal-stress diff.  $N_1=G\,(C_{xx}-C_{yy})$")
    axL.set_title("Steady simple shear")

    # --- right: planar extension, C_xx vs lambda*edot -----------------------
    kd.grid(axR)
    for name, _label in MODELS:
        le, cxx, _cyy, _n1 = data[("extension", name)]
        c = colours[name]
        axR.plot(le, cxx, "-", color=kd.CONNECT, lw=1.6, zorder=2)
        axR.plot(le, cxx, **kd.kraken_marker(c, marker=markers[name], ms=9),
                 zorder=4)
    # Oldroyd-B coil-stretch pole at lambda*edot = 0.5.
    axR.axvline(0.5, color=kd.ACCENT, ls="--", lw=1.8, zorder=1)
    axR.annotate(r"coil-stretch pole  $\lambda\dot\varepsilon=0.5$",
                 (0.5, 38.0), textcoords="offset points",
                 xytext=(-8, 0), ha="right", rotation=90, fontsize=11,
                 color=kd.ACCENT)
    axR.set_xlim(0.0, 0.54)
    axR.set_ylim(0.5, 55.0)
    axR.set_xlabel(r"extension number  $\lambda\dot\varepsilon$")
    axR.set_ylabel(r"streamwise conformation  $C_{xx}$")
    axR.set_title("Steady planar extension")

    # --- shared legend ------------------------------------------------------
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=colours[m[0]], lw=2.0, marker=markers[m[0]],
               mfc=colours[m[0]], mec="0.92", ms=9, label=m[1])
        for m in MODELS
    ]
    kd.dark_legend(axL, handles=handles, loc="upper left", title="Model")

    fig.suptitle(
        kd.tex("Kraken viscoelastic constitutive models — "
               "Oldroyd-B unbounded vs FENE-P / Giesekus / PTT bounded"),
        fontsize=14.5, fontweight="bold")

    # Write straight into the docs page location (repo_root = HERE/../../../..).
    repo_root = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
    docs_dir = os.path.join(repo_root, "docs", "src", "users", "benchmarks")
    out = (os.path.join(docs_dir, "ve-constitutive-models.png")
           if os.path.isdir(docs_dir)
           else os.path.join(HERE, "ve-constitutive-models.png"))
    fig.savefig(out, dpi=150)
    print(f"wrote {out} (usetex={usetex})")


if __name__ == "__main__":
    main()
