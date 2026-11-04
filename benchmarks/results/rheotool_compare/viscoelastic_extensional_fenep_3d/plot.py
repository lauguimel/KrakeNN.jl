#!/usr/bin/env python3
r"""Reproduce the FENE-P planar-extension RheoTool-vs-Kraken cross-validation figure.

Self-contained: reads ``rheotool_fenep_extensional_transient.csv`` next to this
script (the rheoTestFoam transient C-tensor curve, reconstructed in Kraken's
conformation convention) and regenerates
``viscoelastic_extensional_fenep_3d_compare.png``.

Physics (mandate §3.2 extensional module). A uniform planar-extension box,
U = (eps_dot*x, -eps_dot*y, 0), advances the FENE-P conformation to its steady
state. FENE-P has NO closed-form fixed point (transcendental in trC), so this is
a RheoTool-vs-Kraken CROSS-VALIDATION at matched parameters (both numerical):

    lambda=50, eps_dot=0.005 (2*lambda*eps_dot=0.5), beta=0.5, L2=50.

RheoTool's ``rheoTestFoam`` realises the uniform deformation in a single cell
(gradU = diag(1,-1,0)). Its total extra-stress is inverted to the conformation
tensor. IMPORTANT: RheoTool and Kraken implement DIFFERENT FENE-P Peterlin
closures -- RheoTool transports A with equilibrium a*I and varf=1/(1-trA/L2);
Kraken transports C with equilibrium I and f=(L2-3)/(L2-trC). The two coincide
only as L2->inf (Oldroyd-B): RheoTool at L2=1e5 returns C_xx=1.99985 (0.007% vs
the OB value 2), which validates the reconstruction pipeline. At the finite
L2=50 a genuine ~11% closure gap remains (RheoTool C_xx=1.737 vs Kraken 1.950).

Dark Documenter theme (#1f2424), matching the other rheotool_compare figures.

Usage:
    python plot.py            # writes viscoelastic_extensional_fenep_3d_compare.png
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
LAMBDA, EPS_DOT, L2 = 50.0, 0.005, 50.0
# Kraken FENE-P (its own closure): canary center + steady transcendental.
KR_CXX, KR_CYY, KR_CZZ = 1.949729, 0.661, 0.987   # canary C_xx, transcendental C_yy/C_zz
KR_CXX_T, KR_CYY_T, KR_CZZ_T = 1.949745, 0.660988, 0.987276  # Kraken transcendental
# RheoTool FENE-P steady (its own closure, reconstructed to Kraken convention).
RT_CXX, RT_CYY, RT_CZZ = 1.737435, 0.634695, 0.929747
# OB limit (L2->inf), both codes agree:
OB_CXX, OB_CYY = 2.0, 2.0 / 3.0


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
    d = load("rheotool_fenep_extensional_transient.csv", "t", "C_xx", "C_yy", "C_zz")
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

    # RheoTool FENE-P transient relaxation curves
    ax.plot(t, d["C_xx"], color=cxx_c, lw=2.0, label=r"RheoTool $C_{xx}$")
    ax.plot(t, d["C_yy"], color=cyy_c, lw=2.0, label=r"RheoTool $C_{yy}$")
    ax.plot(t, d["C_zz"], color=czz_c, lw=1.6, label=r"RheoTool $C_{zz}$")

    # Kraken FENE-P steady-state lines (its own closure)
    for cval, col in ((KR_CXX_T, cxx_c), (KR_CYY_T, cyy_c), (KR_CZZ_T, czz_c)):
        ax.axhline(cval, color=col, ls="--", lw=1.0, alpha=0.7)

    # Kraken FVFD canary marker (C_xx at its 1000-step horizon)
    tk = t[-1]
    ax.scatter([tk], [KR_CXX], color=cxx_c, marker="o", s=70, ec="white",
               zorder=5, label=r"Kraken FENE-P $C_{xx}$ (1000 steps)")

    # OB-limit reference (both codes agree as L2->inf)
    ax.axhline(OB_CXX, color="white", ls=":", lw=0.9, alpha=0.5)
    ax.annotate(r"OB limit ($L^2\!\to\!\infty$): $C_{xx}=2$",
                xy=(0.30, OB_CXX), xycoords=("axes fraction", "data"),
                fontsize=9.5, color="white", alpha=0.7, va="bottom")

    ax.annotate(r"finite $L^2=50$: $\mathrm{tr}\,C<L^2$ (bounded)",
                xy=(0.04, 0.10), xycoords="axes fraction",
                fontsize=10, color="white", alpha=0.85)

    ax.set_xlabel(r"$t/\lambda$")
    ax.set_ylabel(r"conformation $C_{ij}$ (Kraken convention)")
    ax.set_title(r"FENE-P planar extension ($2\lambda\dot\varepsilon=0.5$, $\beta=0.5$, $L^2=50$):"
                 "\nRheoTool vs Kraken cross-validation (dashed = Kraken steady)")
    ax.set_xlim(0, max(t))
    ax.set_ylim(0.5, 2.15)
    ax.grid(True, color="white", alpha=0.12)
    leg = ax.legend(loc="center right", frameon=True, framealpha=0.15, fontsize=10)
    for txt in leg.get_texts():
        txt.set_color("white")

    fig.tight_layout()
    out = os.path.join(HERE, "viscoelastic_extensional_fenep_3d_compare.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
