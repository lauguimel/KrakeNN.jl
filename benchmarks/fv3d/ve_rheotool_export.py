"""Export RheoTool square-duct velocities for the KrakeNN calibration driver (plan step F6.3).

Input: quadrant npz files (lower-left quarter x, y <= 0.5 of the unit section, float32,
keys U (3, ny/2, nx/2) = (Ux, Uy, Uz) at the cell centres x, y of a graded mesh, param_*).
Output: one CSV per file, same stem, with the velocity at the centres (k - 1/2)/100,
k = 1..50, of a fixed 50 x 50 grid on that quarter, in the axes of the KrakeNN slice
(axial x, section (y, z)):  y_K = x_RT, z_K = y_RT, u_K = Uz, v_K = Ux, w_K = Uy.

Interpolation: bilinear (separable linear) on the graded grid, extended by the no-slip
wall value 0 at x = 0 and y = 0 and by the mirror value across the symmetry lines
x = 0.5 and y = 0.5 (parities: under x -> 1 - x, Ux is odd; under y -> 1 - y, Uy is odd).

    python -I ve_rheotool_export.py <out_dir> <De0.1.npz> [<De1.0.npz> ...]

numpy only.
"""
import os
import sys

import numpy as np

NS = 50
PTS = (np.arange(1, NS + 1) - 0.5) / (2 * NS)          # 0.005, 0.015, ..., 0.495
PARITY_X = (-1.0, 1.0, 1.0)                            # (Ux, Uy, Uz) under x -> 1 - x
PARITY_Y = (1.0, -1.0, 1.0)                            # under y -> 1 - y


def _extend(coords, F, axis, parity):
    """Prepend the wall (coordinate 0, value 0) and append the mirror of the last centre."""
    first = np.zeros_like(np.take(F, [0], axis=axis))
    last = parity * np.take(F, [-1], axis=axis)
    c = np.concatenate([[0.0], coords, [1.0 - coords[-1]]])
    return c, np.concatenate([first, F, last], axis=axis)


def _interp(coords, F, axis):
    return np.apply_along_axis(lambda a: np.interp(PTS, coords, a), axis, F)


def sample(d):
    """(u_K, v_K, w_K), each indexed [i_y_K, i_z_K] = [i_x_RT, i_y_RT]."""
    x = np.asarray(d["x"], float)
    y = np.asarray(d["y"], float)
    out = []
    for c in range(3):
        F = np.asarray(d["U"][c], float)               # [iy_RT, ix_RT]
        xe, Fe = _extend(x, F, 1, PARITY_X[c])
        ye, Fe = _extend(y, Fe, 0, PARITY_Y[c])
        G = _interp(ye, _interp(xe, Fe, 1), 0)          # [iy_RT, ix_RT] at PTS x PTS
        out.append(G.T)                                 # [ix_RT, iy_RT] = [iy_K, iz_K]
    ux, uy, uz = out
    return uz, ux, uy


def _param(d, key):
    return float(d[key]) if key in d.files else float("nan")


def export(path, out_dir):
    d = np.load(path)
    u, v, w = sample(d)
    stem = os.path.splitext(os.path.basename(path))[0]
    dest = os.path.join(out_dir, stem + ".csv")
    with open(dest, "w") as f:
        f.write(f"# source {os.path.basename(path)}\n")
        for k in ("lambda", "alpha", "etaS", "etaP", "rho"):
            f.write(f"# {k} {_param(d, 'param_' + k):.17g}\n")
        f.write(f"# U_mean {_param(d, 'U_mean'):.17g}\n")
        f.write("# axes: y_K = x_RT, z_K = y_RT, u_K = Uz, v_K = Ux, w_K = Uy\n")
        f.write("y,z,u,v,w\n")
        for i in range(NS):
            for j in range(NS):
                f.write(f"{PTS[i]:.6f},{PTS[j]:.6f},{u[i, j]:.9e},{v[i, j]:.9e},{w[i, j]:.9e}\n")
    print(f"{dest}: max|u| {np.abs(u).max():.6f}  max|(v, w)| {np.hypot(v, w).max():.4e}")
    return dest


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    out_dir = argv[1]
    os.makedirs(out_dir, exist_ok=True)
    for p in argv[2:]:
        export(p, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
