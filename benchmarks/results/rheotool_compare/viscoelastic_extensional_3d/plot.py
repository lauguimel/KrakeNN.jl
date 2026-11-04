#!/usr/bin/env python3
r"""Reproduce the Oldroyd-B planar-extension RheoTool-vs-Kraken-vs-analytic figure.

Self-contained: reads ``rheotool_extensional_transient.csv`` next to this script
(the rheoTestFoam transient C-tensor curve) and regenerates
``viscoelastic_extensional_3d_compare.png``.

Physics (mandate §3.2 extensional module). A uniform planar-extension box,
U = (eps_dot*x, -eps_dot*y, 0), advances the Oldroyd-B conformation to the
analytic fixed point (2*lambda*eps_dot < 1):

    C_xx = 1/(1 - 2*lambda*eps_dot),  C_yy = 1/(1 + 2*lambda*eps_dot),  C_zz = 1.

At lambda=50, eps_dot=0.005 (lambda*eps_dot = 0.25): C_xx=2, C_yy=2/3, C_zz=1.

RheoTool's ``rheoTestFoam`` realises exactly this uniform deformation in a single
cell (gradU = diag(1,-1,0)) -- no geometry mismatch, no stagnation point, no
residence-time effect (unlike a cross-slot). Its total extra-stress is inverted
to the conformation tensor and shown relaxing to the analytic fixed point, which
it reaches to machine precision at steady state. Kraken's 3D FVFD log-conformation
canary (run_viscoelastic_fvfd_extensional_3d, 1000-step horizon) is overlaid as a
filled marker; it sits 0.0023 % from the fixed point on C_xx (0.39 % before the
periodic-z advection fix of #54, which removed a first-order fallback).

Dark Documenter theme (DARK = #1f2424), matching the other rheotool_compare
figures. LaTeX used for all text when a system ``latex`` is available.

Usage:
    python plot.py            # writes viscoelastic_extensional_3d_compare.png
Run under the documentation figure environment:
    conda run -n kraken-v0-3-figures python plot.py
"""
import csv
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
_USETEX = shutil.which("latex") is not None
DARK = "#1f2424"  # Documenter dark theme background

# --- operating point ---
LAMBDA, EPS_DOT = 50.0, 0.005
CXX_A = 1.0 / (1.0 - 2.0 * LAMBDA * EPS_DOT)   # 2.0
CYY_A = 1.0 / (1.0 + 2.0 * LAMBDA * EPS_DOT)   # 2/3
CZZ_A = 1.0
# Kraken FVFD log-conf canary (1000 steps, CPU/CUDA F64), center cell:
KR_CXX, KR_CYY, KR_CZZ = 1.9999546, 0.6666667, 1.0
KR_STEPS = 1000


def load(name, *cols):
    path = os.path.join(HERE, name)
    out = {c: [] for c in cols}
    with open(path, newline="") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
        for r in csv.DictReader(lines):
            for c in cols:
                out[c].append(float(r[c]))
    return out


def main():
    d = load("rheotool_extensional_transient.csv", "t", "C_xx", "C_yy", "C_zz")
    t = [x / LAMBDA for x in d["t"]]   # time in relaxation units

    plt.rcParams.update({
        "text.usetex": _USETEX,
        "font.family": "serif",
        "font.size": 12,
        "figure.facecolor": DARK,
        "axes.facecolor": DARK,
        "savefig.facecolor": DARK,
        "text.color": "white",
        "axes.labelcolor": "white",
        "axes.edgecolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
    })
    if _USETEX:
        plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    cxx_c, cyy_c, czz_c = "#ff6b6b", "#4dd0e1", "#b0bec5"

    # RheoTool transient relaxation curves
    ax.plot(t, d["C_xx"], color=cxx_c, lw=2.0, label=r"RheoTool $C_{xx}$")
    ax.plot(t, d["C_yy"], color=cyy_c, lw=2.0, label=r"RheoTool $C_{yy}$")
    ax.plot(t, d["C_zz"], color=czz_c, lw=1.6, label=r"RheoTool $C_{zz}$")

    # Analytic fixed-point lines
    for cval, col in ((CXX_A, cxx_c), (CYY_A, cyy_c), (CZZ_A, czz_c)):
        ax.axhline(cval, color=col, ls="--", lw=1.0, alpha=0.7)

    # Kraken FVFD canary markers (at its 1000-step horizon, placed at the
    # transient end for visual comparison)
    tk = t[-1]
    ax.scatter([tk], [KR_CXX], color=cxx_c, marker="o", s=70, ec="white",
               zorder=5, label=r"Kraken FVFD (1000 steps)")
    ax.scatter([tk], [KR_CYY], color=cyy_c, marker="o", s=70, ec="white", zorder=5)
    ax.scatter([tk], [KR_CZZ], color=czz_c, marker="o", s=70, ec="white", zorder=5)

    ax.annotate(r"analytic OB: $C_{xx}=2,\ C_{yy}=\tfrac{2}{3},\ C_{zz}=1$",
                xy=(0.02, 0.5), xycoords="axes fraction",
                fontsize=10.5, color="white", alpha=0.85)

    ax.set_xlabel(r"$t/\lambda$")
    ax.set_ylabel(r"conformation $C_{ij}$")
    ax.set_title(r"Oldroyd-B planar extension ($\lambda\dot\varepsilon=0.25$, $\beta=0.5$):"
                 "\nRheoTool $\\equiv$ Kraken $\\equiv$ analytic")
    ax.set_xlim(0, max(t))
    ax.set_ylim(0.5, 2.15)
    ax.grid(True, color="white", alpha=0.12)
    leg = ax.legend(loc="center right", frameon=True, framealpha=0.15, fontsize=10)
    for txt in leg.get_texts():
        txt.set_color("white")

    fig.tight_layout()
    out = os.path.join(HERE, "viscoelastic_extensional_3d_compare.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
