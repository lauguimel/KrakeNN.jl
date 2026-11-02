#!/usr/bin/env python3
"""Generate the E1 GPU efficiency certification bar chart.

Kraken best sustained F64 D2Q9 BGK throughput (3461 MLUPS, A100, N=2048)
plotted against memory-bandwidth rooflines and the Palabos single-GPU F64
reference (D3Q19, a different lattice). The rooflines use the D2Q9 traffic,
2 * 9 * sizeof(Float64) = 144 bytes per update. A dashed line marks the
0.5x-roofline pass gate against the A100-40GB ceiling the run used.

Run:
    python3 bench/certification/plot_certification.py
or:
    conda run -n kraken-v0-3-figures python bench/certification/plot_certification.py
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "docs", "src", "users", "benchmarks", "gpu-certification.png",
)

# Measured + reference numbers (MLUPS). Do not edit without new measurements.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV = os.path.join(REPO, "benchmarks", "results", "certification_a100.csv")
with open(CSV) as fh:
    KRAKEN_BEST = max(float(r["MLUPS"]) for r in csv.DictReader(fh))  # A100, F64, N=2048

# Bandwidth ceilings: peak_BW / 144 bytes per D2Q9 F64 update (2 * 9 * 8).
BYTES_PER_UPDATE = 2 * 9 * 8
ceiling = lambda tb_s: tb_s * 1e12 / BYTES_PER_UPDATE / 1e6
A100_40, A100_80, H100 = ceiling(1.555), ceiling(2.039), ceiling(3.35)

labels = [
    "Kraken\n(A100 F64)",
    "Palabos D3Q19\n(A100 F64)",
    "A100-40GB\nroofline",
    "A100-80GB\nroofline",
    "H100\nroofline",
]
values = [KRAKEN_BEST, 4656.0, A100_40, A100_80, H100]
colors = ["#2b6cb0", "#718096", "#a0aec0", "#a0aec0", "#cbd5e0"]

# 0.5x gate against the confirmed run GPU (A100-40GB, gpu0n009).
GATE = 0.5 * A100_40  # = 5400 MLUPS

fig, ax = plt.subplots(figsize=(7.0, 4.4))

bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6, zorder=3)

# Highlight the Kraken bar.
bars[0].set_color("#2b6cb0")
bars[0].set_edgecolor("black")

for rect, val in zip(bars, values):
    ax.text(
        rect.get_x() + rect.get_width() / 2.0,
        val + 300,
        f"{val:,.0f}",
        ha="center",
        va="bottom",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.8),
        zorder=4,
    )

# 0.5x-roofline pass gate (vs the A100-40GB ceiling).
ax.axhline(GATE, color="#c53030", linestyle="--", linewidth=1.3, zorder=2)
ax.text(
    len(labels) - 0.5,
    GATE + 250,
    f"0.5x gate (A100-40GB) = {GATE:,.0f}",
    ha="right",
    va="bottom",
    fontsize=8.5,
    color="#c53030",
)

ax.set_ylabel("Sustained throughput (MLUPS, F64)")
ax.set_title("Single-GPU certification — BGK D2Q9 CUDA Float64")
ax.set_ylim(0, 26000)
ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
