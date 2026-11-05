#!/usr/bin/env python3
r"""Reproduce the GPU-certification figure from the shipped CSV.

Self-contained: reads ``certification_a100.csv`` next to this script and
regenerates ``comparison.png``.

Two panels:

    left  -- measured sustained MLUPS vs domain size $N$ (BGK D2Q9, CUDA F64,
             1000 steps on an NVIDIA A100-40GB). Colour encodes $N$ (crest).
    right -- the best sustained Kraken throughput against the memory-bandwidth
             roofline ceilings and a published single-GPU F64 LBM code, as a
             horizontal bar chart. The reference ceilings / published number
             are literature constants (see README), not fitted data.

LaTeX is used for all text when a system ``latex`` is available, otherwise the
matplotlib mathtext engine with the Computer-Modern font set.
Dependencies: csv + matplotlib + seaborn.

Usage:
    python plot.py            # writes comparison.png next to this script
"""
import csv
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
_USETEX = shutil.which("latex") is not None
DARK = "#1f2424"  # Documenter dark theme background

# Published / derived reference ceilings (see README + the docs page).
# A100-40GB peak HBM 1.555 TB/s, 144 bytes/update F64 D2Q9 -> 10799 MLUPS;
# A100-80GB 2.039 TB/s -> 14160 MLUPS.
REFERENCES = [
    ("A100-40GB roofline", 10799, "ceiling"),
    ("A100-80GB roofline", 14160, "ceiling"),
    ("Palabos F64 D3Q19 TRT", 4656, "code"),
]


def _load():
    path = os.path.join(HERE, "certification_a100.csv")
    with open(path, newline="") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
        return list(csv.DictReader(lines))


def main():
    sns.set_theme(style="ticks", context="talk", font="serif")
    plt.rcParams.update({
        "text.usetex": _USETEX, "font.family": "serif", "mathtext.fontset": "cm",
        "figure.facecolor": DARK, "axes.facecolor": DARK, "savefig.facecolor": DARK,
        "text.color": "0.92", "axes.labelcolor": "0.92", "axes.titlecolor": "0.96",
        "axes.edgecolor": "0.55", "xtick.color": "0.85", "ytick.color": "0.85",
    })

    rows = sorted(_load(), key=lambda r: int(r["N"]))
    Ns = [int(r["N"]) for r in rows]
    mlups = [float(r["MLUPS"]) for r in rows]
    best = max(mlups)

    fig, (axn, axb) = plt.subplots(1, 2, figsize=(14, 6.2), constrained_layout=True)

    # --- left: MLUPS vs N -------------------------------------------------
    axn.grid(True, color="0.45", alpha=0.4, lw=0.6)
    palette = sns.color_palette("bright", len(Ns))  # vivid; pops on dark
    axn.plot(Ns, mlups, "-", color="0.8", lw=2.0, zorder=1)
    for color, n, m in zip(palette, Ns, mlups):
        axn.plot(n, m, "o", color=color, ms=13, mec="0.92", mew=1.0, zorder=3)
        axn.annotate(fr"${m:.0f}$", (n, m), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=11)
    axn.set_xscale("log", base=2)
    axn.set_xticks(Ns)
    axn.set_xticklabels([fr"${n}$" for n in Ns])
    axn.set(xlabel=r"domain size $N$ ($N\times N$ periodic)",
            ylabel=r"sustained MLUPS",
            title=r"Throughput vs $N$ --- BGK D2Q9, CUDA F64")
    axn.set_ylim(0, best * 1.18)

    # --- right: Kraken best vs roofline / published bars ------------------
    axb.grid(True, axis="x", color="0.45", alpha=0.4, lw=0.6)
    bars = [("Kraken (best)", best, "kraken")] + REFERENCES
    labels = [b[0] for b in bars]
    vals = [b[1] for b in bars]
    kinds = [b[2] for b in bars]
    bar_palette = sns.color_palette("bright", len(bars))  # vivid; pops on dark
    colors = []
    for k, c in zip(kinds, bar_palette):
        if k == "kraken":
            colors.append("#ff6b6b")
        elif k == "ceiling":
            colors.append("0.6")
        else:
            colors.append(c)
    ypos = range(len(bars))
    axb.barh(list(ypos), vals, color=colors, edgecolor="0.85", height=0.62)
    for y, v in zip(ypos, vals):
        ratio = v / best if v != best else None
        txt = fr"${v:.0f}$"
        axb.text(v + best * 0.02, y, txt, va="center", fontsize=11)
    axb.set_yticks(list(ypos))
    axb.set_yticklabels([fr"{lab}" for lab in labels], fontsize=11)
    axb.invert_yaxis()
    axb.set(xlabel=r"MLUPS (F64)",
            title=r"Kraken vs roofline \& published code"
                  if _USETEX else r"Kraken vs roofline & published code")
    axb.set_xlim(0, max(vals) * 1.18)

    fig.suptitle("GPU efficiency certification --- BGK D2Q9 CUDA F64 "
                 "(NVIDIA A100-40GB)", fontsize=15, fontweight="bold")
    out = os.path.join(HERE, "comparison.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out} (usetex={_USETEX})  best={best:.1f} MLUPS")


if __name__ == "__main__":
    main()
