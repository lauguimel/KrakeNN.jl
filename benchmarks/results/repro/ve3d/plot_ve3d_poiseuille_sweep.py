#!/usr/bin/env python3
r"""Dark docs figure for the FVFD-3D viscoelastic Poiseuille convergence sweep.

Reads ``ve3d_poiseuille_sweep.csv`` next to this script (the Aqua H100 / CUDA F64
sweep: ny = 32 / 64 / 128) and renders ``ve3d-poiseuille-convergence.png`` on the
live Vitepress dark page colour (#1b1b1f) via the ``krakendark`` layer.

Two-panel story (the #2B-cure demonstration):
  (left)  near-wall C_xy error vs Ny, log-y. FVFD log-conformation path stays
          machine-exact (<=1.9e-7) at every resolution, while the diffusive
          LBM-CDE path sits at 25.9 % near-wall on the SAME case (constant ref
          marker, resolution-independent because the kappa*d2C/dy2 over-smoothing
          is a modelling term, not a discretisation one).
  (right) peak velocity ratio u_peak / u_parabola: FVFD ~ 1.0 vs LBM ~ 1.13.

The LBM reference numbers (25.9 % near-wall, u_ratio 1.13) are the headline values
of the M4 payoff canary ``test/test_fvfd_poiseuille_payoff_3d.jl`` on the same
Ny = 32 case; they are shown as a flat reference band, not a converging curve.

Reproduce (env ``kraken-v0-3-figures``):
    conda run -n kraken-v0-3-figures python plot_ve3d_poiseuille_sweep.py

  input  : ve3d_poiseuille_sweep.csv          (next to this script)
  output : docs/src/users/benchmarks/ve3d-poiseuille-convergence.png
           (written directly into the docs page; committed PNG, do not rename)
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))

# Find krakendark: the docs skill assets, or a repo viz/ copy.
for cand in (
    os.path.expanduser("~/.claude/skills/kraken-doc/assets"),
    os.path.join(HERE, "..", "..", "viz"),
):
    if os.path.isfile(os.path.join(cand, "krakendark.py")):
        sys.path.insert(0, cand)
        break
import krakendark as kd  # noqa: E402

# Headline LBM-CDE reference values from the M4 payoff canary (same Ny=32 case).
LBM_NEAR_WALL_ERR = 0.259   # 25.9 % near-wall C_xy error (diffusive kappa term)
LBM_U_RATIO = 1.13          # peak velocity over-shoot vs parabola


def load(name):
    path = os.path.join(HERE, name)
    rows = []
    with open(path, newline="") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
        for r in csv.DictReader(lines):
            rows.append(r)
    ny = [int(float(r["ny"])) for r in rows]
    err = [float(r["near_wall_Cxy_err_abs"]) for r in rows]
    uratio = [float(r["u_ratio"]) for r in rows]
    return ny, err, uratio


def main():
    usetex = kd.apply()
    pct = r"\%" if usetex else "%"
    pal = kd.palette(4)
    c_fv, c_lbm = pal[0], pal[3]  # blue (Kraken FVFD), red (LBM-CDE reference)

    ny, err, uratio = load("ve3d_poiseuille_sweep.csv")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 5.6),
                                   constrained_layout=True)

    # --- left: near-wall C_xy error vs Ny (log-y) ---------------------------
    kd.grid(axL)
    axL.set_yscale("log")
    # FVFD curve: machine-exact, climbs slightly with Ny but stays <= 1.9e-7.
    axL.plot(ny, err, "-", color=kd.CONNECT, lw=2.0, zorder=2)
    axL.plot(ny, err, **kd.kraken_marker(c_fv, marker="o", ms=12), zorder=4)
    # LBM-CDE reference: flat band at 25.9 % (resolution-independent modelling term).
    axL.axhline(LBM_NEAR_WALL_ERR, color=c_lbm, ls="--", lw=2.2, zorder=1)
    axL.plot([ny[0]], [LBM_NEAR_WALL_ERR],
             **kd.reference_marker(c_lbm, marker="s", ms=13), zorder=3)
    axL.annotate(fr"LBM-CDE  $25.9\,{pct}$",
                 (ny[1], LBM_NEAR_WALL_ERR), textcoords="offset points",
                 xytext=(0, -22), ha="center", fontsize=12, color=c_lbm)
    axL.annotate(r"FVFD log-conf  $\leq 1.9\times10^{-7}$",
                 (ny[1], err[1]), textcoords="offset points",
                 xytext=(0, 18), ha="center", fontsize=12, color="0.9")
    axL.set_xticks(ny)
    axL.set_xlabel(r"wall-normal resolution  $N_y$")
    axL.set_ylabel(kd.tex(r"near-wall  $|C_{xy}-C_{xy}^{\rm ref}|$  (abs.)"))
    axL.set_title(kd.tex("Near-wall conformation error"))
    axL.set_ylim(1e-15, 1.0)

    h_err = [
        Line2D([0], [0], color=c_fv, lw=2.0, marker="o", mfc=c_fv, mec="0.92",
               ms=11, label="Kraken FVFD log-conf (CUDA F64)"),
        Line2D([0], [0], color=c_lbm, lw=2.2, ls="--", marker="s", mfc="none",
               mec=c_lbm, ms=11, mew=2.0, label="LBM-CDE diffusive (same case)"),
    ]
    kd.dark_legend(axL, handles=h_err, loc="lower right")

    # --- right: peak velocity ratio vs parabola -----------------------------
    kd.grid(axR)
    axR.axhline(1.0, color="0.6", ls=":", lw=1.4, zorder=0)
    axR.plot(ny, uratio, "-", color=kd.CONNECT, lw=2.0, zorder=2)
    axR.plot(ny, uratio, **kd.kraken_marker(c_fv, marker="o", ms=12), zorder=4)
    axR.axhline(LBM_U_RATIO, color=c_lbm, ls="--", lw=2.2, zorder=1)
    axR.plot([ny[0]], [LBM_U_RATIO],
             **kd.reference_marker(c_lbm, marker="s", ms=13), zorder=3)
    axR.annotate(fr"LBM-CDE  $\approx 1.13$ $(+13\,{pct})$",
                 (ny[1], LBM_U_RATIO), textcoords="offset points",
                 xytext=(0, 10), ha="center", fontsize=12, color=c_lbm)
    axR.annotate(r"FVFD  $\approx 1.000$",
                 (ny[-1], uratio[-1]), textcoords="offset points",
                 xytext=(-6, -20), ha="right", fontsize=12, color="0.9")
    axR.set_xticks(ny)
    axR.set_xlabel(r"wall-normal resolution  $N_y$")
    axR.set_ylabel(r"peak velocity ratio  $u_{\rm peak}/u_{\rm parabola}$")
    axR.set_title(kd.tex("Velocity over-shoot"))
    axR.set_ylim(0.98, 1.16)

    fig.suptitle(
        kd.tex("FVFD log-conformation cures CDE over-diffusion "
               "(Oldroyd-B 3D Poiseuille, ") +
        r"$\mathrm{Wi}_{\rm wall}=0.5$" + kd.tex(")"),
        fontsize=15, fontweight="bold")

    # Write straight into the docs page location (repo_root = HERE/../../../..).
    repo_root = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
    docs_dir = os.path.join(repo_root, "docs", "src", "users", "benchmarks")
    out = (os.path.join(docs_dir, "ve3d-poiseuille-convergence.png")
           if os.path.isdir(docs_dir)
           else os.path.join(HERE, "ve3d-poiseuille-convergence.png"))
    fig.savefig(out, dpi=150)
    print(f"wrote {out} (usetex={usetex})")


if __name__ == "__main__":
    main()
