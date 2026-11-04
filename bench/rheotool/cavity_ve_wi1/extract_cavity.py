"""Extract the cavity comparison quantities from a solved RheoTool run.

Reads the cell-centre fields written by rheoFoam (``<t>/U[.gz]``,
``<t>/tau[.gz]``, every written time t > 0) and the ``kinEner.txt`` log of the
case, and writes three plain CSV files (one header line each):

- ``<prefix>_energy.csv``   t, kinetic_energy_mean, elastic_energy_mean: every
  row of ``kinEner.txt`` (t = 0 included), as logged by the coded function
  object every 20 time steps;
- ``<prefix>_profiles.csv`` long format, one row per (t, line, coord): u_x, u_y
  and, for a viscoelastic run, the xx, xy, yy components of tau_p and of
  C = I + (lambda/eta_p) tau_p (empty for a run without ``tau``), along
  x = 0.5 and y = 0.5, 0.75, 0.9, 0.95; coord is the cell-centre coordinate
  along the line (y for x = 0.5, x for the others). lambda and eta_p are the
  command-line values (defaults: those of the case template), not read from
  the case;
- ``<prefix>_vortex.csv``   t, x_centre, y_centre, psi_min,
  max_abs_column_integral_ux: centre and value of the minimum of the stream
  function (the primary vortex), and max over the cell columns of
  |int_0^1 u_x dy|, a consistency check of the sampled field.

All three reductions work on the cell-centre values only, with rules simple
enough for a Kraken field sampled at the same cell centres to be reduced
identically:

- line values: linear interpolation between the two cell-centre rows (or
  columns) around the line (``interp_rows``);
- stream function: psi(x, y) = int_0^y u_x dy' from the bottom wall, midpoint
  rule over the cells below plus the lower half of the current cell
  (``stream_function``);
- vortex centre: local 4x4 tensor-product cubic interpolation of psi around its
  discrete minimum, stationary point by Newton iteration, psi_min = the
  interpolant there; the full rule, stencil choice and fallbacks included, is
  in the docstring of ``vortex_centre``. A fallback (discrete minimum written
  unrefined) is reported on stderr.

The mesh is the tutorial's single hex block (N x N x 1), whose cells are
ordered x fastest, then y; the cell centres are ((i + 1/2)/N, (j + 1/2)/N).

Usage:
    python extract_cavity.py <case_dir> <N> <out_prefix> [--lambda 1 --etaP 0.5]
"""

import argparse
import gzip
import os
import re
import sys

import numpy as np

HORIZONTAL_LINES = (0.5, 0.75, 0.9, 0.95)
VERTICAL_LINES = (0.5,)


def read_internal_field(path, ncomp):
    """Return the internalField of an ascii volField as an (ncells, ncomp) array."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        text = fh.read()
    body = text[text.index("internalField"):]
    if body.startswith("internalField   uniform") or re.match(r"internalField\s+uniform", body):
        raise ValueError(f"{path}: uniform internalField, no solved data")
    m = re.match(r"internalField\s+nonuniform\s+List<\w+>\s*(\d+)\s*\(", body)
    if m is None:
        raise ValueError(f"{path}: cannot parse internalField header")
    n = int(m.group(1))
    start = m.end()
    end = body.index("\n)", start)
    values = np.array(body[start:end].replace("(", " ").replace(")", " ").split(), dtype=float)
    if values.size != n * ncomp:
        raise ValueError(f"{path}: expected {n * ncomp} values, read {values.size}")
    return values.reshape(n, ncomp)


def field_path(case, tdir, name):
    for candidate in (name + ".gz", name):
        p = os.path.join(case, tdir, candidate)
        if os.path.exists(p):
            return p
    return None


def time_dirs(case):
    out = []
    for d in os.listdir(case):
        try:
            t = float(d)
        except ValueError:
            continue
        if t > 0 and field_path(case, d, "U") is not None:
            out.append((t, d))
    return sorted(out)


def interp_rows(field, coord, n):
    """Linear interpolation across the row index at physical coordinate `coord`.

    `field` is (N_row, ...) indexed by the cell index normal to the line.
    """
    s = coord * n - 0.5
    j0 = int(np.floor(s))
    w = s - j0
    return (1.0 - w) * field[j0] + w * field[j0 + 1]


def stream_function(ux, n):
    """psi(x, y) = int_0^y u_x dy' on the cell centres, integrated from the bottom wall.

    psi[j, i] = h (u[0, i] + ... + u[j-1, i] + u[j, i] / 2), h = 1/n: the midpoint
    rule over the j cells below, plus the lower half of cell j with its centre value.
    """
    dy = 1.0 / n
    return np.cumsum(ux, axis=0) * dy - 0.5 * ux * dy


NEWTON_TOL = 1e-12   # step size (cell units) below which Newton has converged
NEWTON_MAX_ITER = 20  # Newton iteration cap per block


def lagrange4(s):
    """Cubic Lagrange basis on the nodes 0, 1, 2, 3 and its first two derivatives at s.

    Returns three length-4 arrays (L, dL/ds, d2L/ds2), entry a for node a.
    """
    L = np.array([-(s - 1) * (s - 2) * (s - 3) / 6,
                  s * (s - 2) * (s - 3) / 2,
                  -s * (s - 1) * (s - 3) / 2,
                  s * (s - 1) * (s - 2) / 6])
    dL = np.array([-(3 * s * s - 12 * s + 11) / 6,
                   (3 * s * s - 10 * s + 6) / 2,
                   -(3 * s * s - 8 * s + 3) / 2,
                   (3 * s * s - 6 * s + 2) / 6])
    d2L = np.array([2 - s, 3 * s - 5, 4 - 3 * s, s - 1])
    return L, dL, d2L


def _newton_on_block(P, s, r):
    """Newton iteration for the stationary point of the tensor cubic through the 4x4 block P.

    P[b, a] is the value at local node (s, r) = (a, b); (s, r) is the start point.
    Returns ((s, r, iterations), "") on convergence, or (None, reason) when an iterate
    has a Hessian that is not positive definite, leaves [0, 3]^2, or the cap is reached
    (step 4 of `vortex_centre`).
    """
    for it in range(1, NEWTON_MAX_ITER + 1):
        Ls, dLs, d2Ls = lagrange4(s)
        Lr, dLr, d2Lr = lagrange4(r)
        gs = Lr @ P @ dLs
        gr = dLr @ P @ Ls
        hss = Lr @ P @ d2Ls
        hrr = d2Lr @ P @ Ls
        hsr = dLr @ P @ dLs
        det = hss * hrr - hsr * hsr
        if not (hss > 0 and det > 0):
            return None, f"Hessian not positive definite at iteration {it}"
        ds = -(hrr * gs - hsr * gr) / det
        dr = -(hss * gr - hsr * gs) / det
        s, r = s + ds, r + dr
        if not (0.0 <= s <= 3.0 and 0.0 <= r <= 3.0):
            return None, f"iterate left the 4x4 block at iteration {it}"
        if max(abs(ds), abs(dr)) <= NEWTON_TOL:
            return (s, r, it), ""
    return None, f"no convergence in {NEWTON_MAX_ITER} iterations"


def vortex_centre(psi, n, info=None):
    """Location and value of the minimum of psi, from a local tensor-product cubic.

    Algorithm (stated completely so that another code, e.g. Kraken in Julia, can
    reduce its own cell-centre field with the identical rule). Grid indices below
    are 0-based; the local coordinates s, r and the node labels 0..3 are the same in
    any language. 1-based equivalents (I = i + 1, J = j + 1, for an array stored
    psi[I, J]): X = x n + 1/2; refine only if 2 <= I*, J* <= n - 1; block inside the
    grid if 1 <= I0 and I0 + 3 <= n (same for J0); P[b+1, a+1] = psi[I0+a, J0+b] -
    psi[I*, J*]; x = (I0 + s - 1/2)/n, y = (J0 + r - 1/2)/n; fallback
    ((I* - 1/2)/n, (J* - 1/2)/n, psi[I*, J*]).

    Grid. psi[j, i] is the value at the cell centre ((i + 1/2)/n, (j + 1/2)/n),
    j the row (y), i the column (x). All arithmetic in double precision: convert
    a single-precision u_x to double before computing psi.

    1. Discrete minimum. (j*, i*) = the smallest psi[j, i]; on a tie, the first in
       the order j ascending, then i ascending (numpy argmin on psi[j, i]; Julia
       argmin on an array stored as psi[i, j]).
    2. Block. A 4x4 block of cell centres, columns i0..i0+3 and rows j0..j0+3; its
       central square is [i0+1, i0+2] x [j0+1, j0+2] in index units. The side is
       taken from the neighbours of the discrete minimum, so that the central
       square spans the discrete minimum and its neighbour with the smaller psi:
         i0 = i* - 1 if psi[j*, i*+1] < psi[j*, i*-1], else i0 = i* - 2 (a tie
         goes to i* - 2); j0 = j* - 1 if psi[j*+1, i*] < psi[j*-1, i*], else j* - 2.
    3. Interpolant. Local coordinates s = X - i0, r = Y - j0 (X, Y = cell index
       coordinates, X = x n - 1/2), so the block nodes are s, r in {0, 1, 2, 3}.
       With P[b, a] = psi[j0+b, i0+a] - psi[j*, i*],
         p(s, r) = sum_{a,b = 0..3} L_b(r) P[b, a] L_a(s),
       where L_a is the cubic Lagrange basis on the nodes 0, 1, 2, 3:
         L_0 = -(s-1)(s-2)(s-3)/6,  L_1 = s(s-2)(s-3)/2,
         L_2 = -s(s-1)(s-3)/2,      L_3 = s(s-1)(s-2)/6
       (derivatives in `lagrange4`). Subtracting psi[j*, i*] does not change the
       interpolant (the basis sums to one); it keeps the rounding error of the
       gradient at the size of the local variation of psi, not of psi itself.
    4. Newton on grad p = 0. Start at the discrete minimum, (s, r) = (i* - i0,
       j* - j0), which is 1 or 2 in each direction. At each iterate: gradient
       (p_s, p_r) and Hessian (p_ss, p_sr, p_rr) of p, i.e. the same double sum
       with L_a(s) and/or L_b(r) replaced by their exact first or second
       derivatives (``lagrange4``); if p_ss <= 0 or
       det = p_ss p_rr - p_sr^2 <= 0 the attempt fails; otherwise
         ds = -(p_rr p_s - p_sr p_r)/det,  dr = -(p_ss p_r - p_sr p_s)/det,
       (s, r) += (ds, dr). The attempt fails if the new iterate leaves the block
       [0, 3]^2. Converged when max(|ds|, |dr|) <= 1e-12 (cell units), the updated
       iterate being the result; the attempt fails after 20 iterations.
    5. Acceptance. A converged point inside the closed central square [1, 2]^2 is
       the result. Outside it, the block is shifted once, by one cell towards the
       point in each direction where it lies outside (i0 += 1 if s > 2, i0 -= 1 if
       s < 1, same for j0 with r), and step 4 is repeated on the new block,
       starting from the converged point in the new local coordinates
       (s - di, r - dj). The second result is accepted if it lies in the new
       central square enlarged by half a cell, [0.5, 2.5]^2 (its nearest cell
       centre is one of the four central nodes). The tolerance covers a
       stationary point on the node line shared by the two central squares,
       which the two blocks, differing by their interpolation error, can each
       place just outside their own square. There is no second shift.
    6. Result. x = (i0 + s + 1/2)/n, y = (j0 + r + 1/2)/n, and psi_min = the
       interpolant at the located point, psi[j*, i*] + p(s, r).
    7. Fallback: the discrete minimum, ((i* + 1/2)/n, (j* + 1/2)/n, psi[j*, i*]),
       unrefined. It is used when the discrete minimum is on the first or last
       row or column (no neighbour on one side), when a block (first or shifted)
       does not lie inside the grid (0 <= i0, i0 + 3 <= n - 1, same for j0; this
       happens only when the minimum is within two cells of a wall), when an
       attempt fails in step 4, or
       when the second result is rejected in step 5. A fallback centre sits
       exactly on a cell centre, so it can be recognised in the output.

    Accuracy (for a smooth psi whose minimum is non-degenerate and spans many
    cells): the interpolant is fourth-order accurate in
    value but its gradient is only third-order, so the located position carries
    an O(h^3) error and psi_min an O(h^4) error, h = 1/n. The leading O(h^3)
    term depends on where the minimum sits inside its cell: it vanishes at the
    centre of the central square (s = r = 3/2) and is largest near its corners.
    One physical minimum followed over several n can therefore show erratic
    observed orders; test a port against analytic minima at many sub-cell
    positions, or against this function on identical input. Status "central" or
    "shifted" only means that Newton converged on a positive-definite Hessian; it
    does not certify accuracy: a flat or under-resolved minimum can be off by a
    fraction of a cell, or by several cells when the minimum is narrower than a
    cell (the discrete minimum itself is then misplaced), without a fallback.

    `info`, if a dict, receives "status" ("central", "shifted" or "fallback: <reason>"),
    "iterations" and "block" (i0, j0 of the block used).
    """
    j, i = np.unravel_index(np.argmin(psi), psi.shape)
    j, i = int(j), int(i)
    fallback = ((i + 0.5) / n, (j + 0.5) / n, float(psi[j, i]))

    def give_up(reason):
        if info is not None:
            info.update(status="fallback: " + reason, iterations=0, block=None)
        return fallback

    if not (1 <= i <= n - 2 and 1 <= j <= n - 2):
        return give_up("discrete minimum on the first or last row or column")
    i0 = i - 1 if psi[j, i + 1] < psi[j, i - 1] else i - 2
    j0 = j - 1 if psi[j + 1, i] < psi[j - 1, i] else j - 2
    ref = float(psi[j, i])
    start = (float(i - i0), float(j - j0))
    total_it = 0
    for attempt in (1, 2):
        if not (0 <= i0 and i0 + 3 <= n - 1 and 0 <= j0 and j0 + 3 <= n - 1):
            return give_up("4x4 block not inside the grid (within two cells of a wall)")
        P = np.asarray(psi[j0:j0 + 4, i0:i0 + 4], dtype=float) - ref
        res, reason = _newton_on_block(P, *start)
        if res is None:
            return give_up(f"block ({i0}, {j0}): {reason}")
        s, r, it = res
        total_it += it
        lo, hi = (1.0, 2.0) if attempt == 1 else (0.5, 2.5)
        if lo <= s <= hi and lo <= r <= hi:
            Ls = lagrange4(s)[0]
            Lr = lagrange4(r)[0]
            if info is not None:
                info.update(status="central" if attempt == 1 else "shifted",
                            iterations=total_it, block=(i0, j0))
            return (i0 + s + 0.5) / n, (j0 + r + 0.5) / n, ref + float(Lr @ P @ Ls)
        if attempt == 2:
            return give_up(f"shifted block ({i0}, {j0}): point at (s, r) = ({s:.4f}, {r:.4f}) "
                           "outside [0.5, 2.5]^2")
        di = 1 if s > 2.0 else (-1 if s < 1.0 else 0)
        dj = 1 if r > 2.0 else (-1 if r < 1.0 else 0)
        i0, j0 = i0 + di, j0 + dj
        start = (s - di, r - dj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("N", type=int)
    ap.add_argument("prefix")
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0)
    ap.add_argument("--etaP", type=float, default=0.5)
    args = ap.parse_args()
    n = args.N

    kin = np.loadtxt(os.path.join(args.case, "kinEner.txt"), ndmin=2)
    np.savetxt(args.prefix + "_energy.csv", kin[:, :3], delimiter=",",
               header="t,kinetic_energy_mean,elastic_energy_mean", comments="", fmt="%.12g")

    prof_rows, vort_rows = [], []
    for t, d in time_dirs(args.case):
        U = read_internal_field(field_path(args.case, d, "U"), 3).reshape(n, n, 3)  # [j, i, c]
        tau_path = field_path(args.case, d, "tau")
        tau = None
        if tau_path is not None:
            tau = read_internal_field(tau_path, 6).reshape(n, n, 6)  # xx xy xz yy yz zz
        centres = (np.arange(n) + 0.5) / n

        def emit(line, coord_vals, uvals, tvals):
            for k, c in enumerate(coord_vals):
                row = [t, line, c, uvals[k, 0], uvals[k, 1]]
                if tvals is not None:
                    txx, txy, tyy = tvals[k, 0], tvals[k, 1], tvals[k, 3]
                    f = args.lam / args.etaP
                    row += [txx, txy, tyy, 1.0 + f * txx, f * txy, 1.0 + f * tyy]
                else:
                    row += [""] * 6
                prof_rows.append(row)

        for xl in VERTICAL_LINES:
            emit(f"x={xl}", centres, interp_rows(np.swapaxes(U, 0, 1), xl, n),
                 None if tau is None else interp_rows(np.swapaxes(tau, 0, 1), xl, n))
        for yl in HORIZONTAL_LINES:
            emit(f"y={yl}", centres, interp_rows(U, yl, n),
                 None if tau is None else interp_rows(tau, yl, n))

        psi = stream_function(U[:, :, 0], n)
        info = {}
        xc, yc, pmin = vortex_centre(psi, n, info)
        if info["status"].startswith("fallback"):
            print(f"extract_cavity.py: t = {t:g}: vortex centre {info['status']}; "
                  "the discrete minimum is written", file=sys.stderr)
        # max over columns of |int_0^1 u_x dy| from cell-centre u (midpoint rule):
        # a consistency check of the sampled field, not mass conservation
        column_integral = np.max(np.abs(psi[-1] + 0.5 * U[-1, :, 0] / n))
        vort_rows.append([t, xc, yc, pmin, column_integral])

    header = "t,line,coord,ux,uy,tau_xx,tau_xy,tau_yy,C_xx,C_xy,C_yy"
    with open(args.prefix + "_profiles.csv", "w") as fh:
        fh.write(header + "\n")
        for r in prof_rows:
            fh.write(",".join(r_ if isinstance(r_, str) else f"{r_:.12g}" for r_ in r) + "\n")
    np.savetxt(args.prefix + "_vortex.csv", np.array(vort_rows), delimiter=",",
               header="t,x_centre,y_centre,psi_min,max_abs_column_integral_ux", comments="", fmt="%.12g")


if __name__ == "__main__":
    main()
