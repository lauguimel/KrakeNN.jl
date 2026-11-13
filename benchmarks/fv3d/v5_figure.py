#!/usr/bin/env python3
"""Two-panel figure of the V5 case (3D duct, law recovery from plane PIV).

Reads the figure files written by stages d and e to <FV5_OUT>/figure/ (see the
README, section "V5 figure data"); needs numpy, matplotlib, Python >= 3.11
(tomllib).

Left: eta/eta0 against lambda*gamma_dot (true eta0 and lambda): true law,
Carreau fit (stage d), MLP fit (stage e, if present), 95 % band of the Carreau
fit over the observed shear range (noise only, and noise + model error), and
the histogram of the shear rate seen by the fitted planes (FV state at the fit)
in the background.
Right: one PIV plane: measured and fitted in-plane/stereo velocity magnitude
over the bulk velocity, and the residual of one component over sigma_u.

    python3 benchmarks/fv3d/v5_figure.py <FV5_OUT> [--data noisy] [--plane z_mid]
            [--fit d|e] [--component x] [--output path.pdf]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    sys.exit("v5_figure.py needs Python >= 3.11 (tomllib)")

# Okabe-Ito colours (colour-blind safe)
C_TRUE = "#000000"
C_CARREAU = "#0072B2"
C_MLP = "#D55E00"
C_HIST = "#D9D9D9"

STYLE = {
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "lines.linewidth": 1.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


def load_toml(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def read_csv(path):
    return np.genfromtxt(path, delimiter=",", names=True)


def read_bin(out, rel, shape):
    return np.fromfile(os.path.join(out, rel), dtype="<f8").reshape(shape, order="F")


def plane_entry(index, name):
    for p in index["planes"]:
        if p["name"] == name:
            return p
    sys.exit(f"plane {name!r} not in the index (have {[p['name'] for p in index['planes']]})")


def panel_law(ax, out, d_index, e_index):
    case = d_index["case"]
    law = read_csv(os.path.join(out, d_index["law_csv"]))
    hist = read_csv(os.path.join(out, d_index["gamma_hist_csv"]))
    lg = law["lambda_gamma"]
    inr = law["in_observed_range"] > 0

    # shear-rate histogram in the background (own y axis, lowest third of the panel)
    axh = ax.twinx()
    cnt = hist["count_footprint_fv"]
    if cnt.max() > 0:
        lo, hi = hist["lambda_gamma_lo"], hist["lambda_gamma_hi"]
        axh.bar(lo, cnt / cnt.max(), width=hi - lo, align="edge", color=C_HIST, edgecolor="none", zorder=0)
    axh.set_ylim(0, 3.0)
    axh.set_yticks([])
    axh.spines["right"].set_visible(False)
    ax.set_zorder(axh.get_zorder() + 1)
    ax.patch.set_visible(False)

    # bands of the Carreau fit, observed range only
    if np.all(np.isfinite(law["total_lo"][inr])):
        ax.fill_between(lg, law["total_lo"], law["total_hi"], where=inr, color=C_CARREAU, alpha=0.16,
                        lw=0, label="95 % band, noise + model error")
        ax.fill_between(lg, law["noise_lo"], law["noise_hi"], where=inr, color=C_CARREAU, alpha=0.40,
                        lw=0, label="95 % band, noise only")
    ax.plot(lg, law["eta_true_over_eta0"], color=C_TRUE, lw=1.2, label="true law")
    ax.plot(lg, law["eta_fit_over_eta0"], color=C_CARREAU, lw=1.2, ls="--", label="Carreau fit")
    if e_index is not None:
        mlp = read_csv(os.path.join(out, e_index["law_csv"]))
        ax.plot(mlp["lambda_gamma"], mlp["eta_mlp_over_eta0"], color=C_MLP, lw=1.2, ls="-.", label="MLP fit")

    g0, g1 = d_index["gamma_range"]
    for g in (g0, g1):
        ax.axvline(case["lambda"] * g, color="0.45", lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lg.min(), lg.max())
    ax.set_xlabel(r"$\lambda\dot\gamma$")
    ax.set_ylabel(r"$\eta/\eta_0$")
    ax.text(np.sqrt(g0 * g1) * case["lambda"], 0.97, "observed\n5–95 %", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=6.5, color="0.35")
    # The upper right corner is the only region free of curves and histogram.
    # The legend sits there, below the "observed" note, on an opaque white box
    # that hides the dotted range marker passing behind it.
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.88), handlelength=2.2, frameon=True,
              facecolor="white", edgecolor="none", framealpha=1.0, borderpad=0.3)


# Figure layout, in inches. The PIV maps keep their true aspect ratio, so the
# width of panel (b) follows from the height left for its three stacked maps,
# and panel (a) takes the remaining width. Fixed margins (rather than an
# automatic layout) keep each colourbar against its map and leave no dead
# space next to the equal-aspect maps.
FIG_W, FIG_H = 7.2, 3.4
TOP = 0.22      # panel labels and title of the first map
BOTTOM = 0.42   # x tick labels and x label
MAP_GAP = 0.24  # between two stacked maps: room for the title of the lower one
LEFT_A = 0.50   # y tick labels and y label of panel (a)
GAP_AB = 0.28   # between panel (a) and the y label of the maps
DECOR_B = 0.40  # y tick labels and y label of the maps
RIGHT = 0.38    # colourbar, its tick labels and the right margin
CBAR_PAD, CBAR_W = 0.05, 0.08  # gap map-colourbar and colourbar width


def make_layout(fig, map_aspect):
    """Create the axes: panel (a), and the three maps of panel (b).

    map_aspect is the width/height ratio of one map in data units.
    Returns (ax_law, [map axes], x of the two panel labels in figure fraction).
    """
    h_map = (FIG_H - TOP - BOTTOM - 2 * MAP_GAP) / 3
    w_map = map_aspect * h_map
    x_map = FIG_W - RIGHT - w_map  # left edge of the maps
    x_a_right = x_map - DECOR_B - GAP_AB
    gs_a = GridSpec(1, 1, figure=fig, left=LEFT_A / FIG_W, right=x_a_right / FIG_W,
                    bottom=BOTTOM / FIG_H, top=1 - TOP / FIG_H)
    gs_b = GridSpec(3, 1, figure=fig, left=x_map / FIG_W, right=(x_map + w_map) / FIG_W,
                    bottom=BOTTOM / FIG_H, top=1 - TOP / FIG_H, hspace=MAP_GAP / h_map)
    ax_law = fig.add_subplot(gs_a[0, 0])
    maps = [fig.add_subplot(gs_b[k, 0]) for k in range(3)]
    label_x = (0.04 / FIG_W, (x_map - DECOR_B) / FIG_W)
    return ax_law, maps, label_x


def cell_extent(c):
    """Length covered by cells centred at c (uniform spacing assumed at the ends)."""
    c = np.asarray(c)
    if c.size < 2:
        return 1.0
    return c[-1] - c[0] + 0.5 * ((c[1] - c[0]) + (c[-1] - c[-2]))


def load_plane(out, fit_index, plane, component):
    """Read one PIV plane: measured and fitted |u|/U_b and the residual over sigma_u."""
    case = fit_index["case"]
    p = plane_entry(fit_index, plane)
    shape = tuple(p["shape"])
    mask = read_bin(out, p["mask"], shape) > 0
    comps = p["components"]
    data = [read_bin(out, f, shape) for f in p["data"]]
    model = [read_bin(out, f, shape) for f in p["model"]]
    res = [read_bin(out, f, shape) for f in p["residual"]]
    ub = case["U_bulk"]
    mag_d = np.sqrt(sum(c ** 2 for c in data)) / ub
    mag_m = np.sqrt(sum(c ** 2 for c in model)) / ub
    ci = comps.index(component) if component in comps else 0
    r = res[ci] / case["sigma_u"]
    for a in (mag_d, mag_m, r):
        a[~mask] = np.nan
    H = case["Ly"]
    ca = np.asarray(p["centres_a"]) / H
    cb = np.asarray(p["centres_b"]) / H
    na, nb = p["axes"]
    kind = "stereo" if p["stereo"] else "2C"
    vmax = np.nanmax([np.nanmax(mag_d), np.nanmax(mag_m)])
    rms = np.sqrt(np.nanmean(r ** 2))
    return {"ca": ca, "cb": cb, "na": na, "nb": nb, "kind": kind, "comp": comps[ci],
            "mag_d": mag_d, "mag_m": mag_m, "r": r, "vmax": vmax, "rms": rms,
            "aspect": cell_extent(ca) / cell_extent(cb)}


def panel_plane(fig, axes, P):
    """Draw the three maps of one PIV plane P (from load_plane) on the given axes."""
    vel = plt.get_cmap("viridis").copy()
    vel.set_bad("0.82")
    div = plt.get_cmap("RdBu_r").copy()
    div.set_bad("0.82")
    c = P["comp"]
    rows = [(P["mag_d"], vel, 0, P["vmax"], f"measured $|\\mathbf{{u}}|/U_b$ ({P['kind']})"),
            (P["mag_m"], vel, 0, P["vmax"], "fitted $|\\mathbf{u}|/U_b$"),
            (P["r"], div, -3, 3, f"$(u_{{{c},\\mathrm{{fit}}}}-u_{{{c},\\mathrm{{meas}}}})/\\sigma_u$, "
                                 f"rms {P['rms']:.2f}")]
    for k, (ax, (A, cmap, lo, hi, title)) in enumerate(zip(axes, rows)):
        im = ax.pcolormesh(P["ca"], P["cb"], A.T, cmap=cmap, vmin=lo, vmax=hi, shading="nearest", rasterized=True)
        ax.set_aspect("equal")
        ax.set_title(title, loc="left", pad=2)
        ax.set_ylabel(f"${P['nb']}/H$")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=2, steps=[1, 2, 5, 10]))
        if k < len(rows) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(f"${P['na']}/H$")
        # Colourbar as a child of the map, so it stays against the drawn map.
        w_in = ax.get_position().width * fig.get_figwidth()
        cax = ax.inset_axes([1 + CBAR_PAD / w_in, 0, CBAR_W / w_in, 1])
        fig.colorbar(im, cax=cax)
        cax.tick_params(labelsize=6)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="FV5_OUT directory of the run")
    ap.add_argument("--data", default="noisy", help="data variant of the fits (noisy | clean | twin | twin_noisy)")
    ap.add_argument("--plane", default="z_mid", help="PIV plane of the right panel")
    ap.add_argument("--fit", default="d", choices=["d", "e"], help="model of the right panel: d Carreau, e MLP")
    ap.add_argument("--component", default="x", help="velocity component of the residual map")
    ap.add_argument("--output", default=None, help="output file (default <out>/figure/v5_figure_<data>.pdf, plus .png)")
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    fdir = os.path.join(out, "figure")
    d_path = os.path.join(fdir, f"d_{a.data}.toml")
    if not os.path.isfile(d_path):
        sys.exit(f"missing {d_path}: run stage d with FV5_DATA={a.data}")
    d_index = load_toml(d_path)
    e_path = os.path.join(fdir, f"e_{a.data}.toml")
    e_index = load_toml(e_path) if os.path.isfile(e_path) else None
    if e_index is None:
        print(f"note: {e_path} not found, MLP curve omitted")
    fit_index = d_index if a.fit == "d" or e_index is None else e_index

    plt.rcParams.update(STYLE)
    plane = load_plane(out, fit_index, a.plane, a.component)
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    axl, maps, label_x = make_layout(fig, plane["aspect"])
    panel_law(axl, out, d_index, e_index)
    panel_plane(fig, maps, plane)
    # Panel labels on one line, in the top margin, at the left of each panel.
    label_y = 1 - 0.03 / FIG_H
    for x, lab in zip(label_x, ("(a)", "(b)")):
        fig.text(x, label_y, lab, fontsize=9, fontweight="bold", ha="left", va="top")

    path = a.output or os.path.join(fdir, f"v5_figure_{a.data}.pdf")
    fig.savefig(path)
    root, ext = os.path.splitext(path)
    if ext.lower() != ".png":
        fig.savefig(root + ".png")
    print("written", path)


if __name__ == "__main__":
    main()
