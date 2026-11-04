# RheoTool reference: viscoelastic lid-driven cavity, Wi = 1 (N = 64 gate data)

Reference values for issue #51: the Newtonian entry gate and the future CPU test.

Produced by Guillaume Maitrejean on 2026-09-24/25 (UTC+10) on the QUT Aqua cluster (serial
CPU jobs) with `rheoFoam`, model `Oldroyd-BLog`, on OpenFOAM-9 build `9-b456138dc4bc`; rheoTool
source tree at commit `81f3e8c` (binaries assumed built from it, not rebuilt). Case:
Oldroyd-B, beta = 0.5, Wi = 1 (lambda = 1), Re = 1 (0.01 in the two `ve_re001` rows;
Newtonian rows eta = 1), uniform N x N mesh, deltaT 2e-4 (1e-4 in the `dt1e-4` rows).
Tables built by `bench/rheotool/cavity_ve_wi1/build_tables.py`. Full provenance, case,
run list and column definitions:
[`benchmarks/results/rheotool_compare/viscoelastic_cavity/README.md`](../../../benchmarks/results/rheotool_compare/viscoelastic_cavity/README.md).
For the analysis behind the statements below, see the analysis comment on issue #51.

## Files

- `summary_t10.csv`: one row per finished run (12 runs). Columns `tag, model, N, Re,
  deltaT, t, x_centre, y_centre, psi_min, energy_t, kinetic_energy_mean,
  elastic_energy_mean`. `t` is the time of the written field used for the vortex values
  (10); `energy_t` is the time of the logged energy row nearest to t = 10 (9.99999999999
  for the three `deltaT = 1e-4` runs).
- `ve_re1_n64_profiles_t10.csv`: run `ve_re1_n64` (Oldroyd-B, N = 64, Re = 1) at t = 10.
  Columns `line, coord, ux, uy, tau_xx, tau_xy, tau_yy, C_xx, C_xy, C_yy`, 64 points on
  each of x = 0.5, y = 0.5, 0.75, 0.9, 0.95.
- `newt_re1_n64_profiles_t10.csv`: run `newt_re1_n64` (Newtonian, eta = 1, N = 64,
  Re = 1) at t = 10. Columns `line, coord, ux, uy`, same lines.

Every value is at t = 10 after a start from rest (zero velocity and polymer stress) with
the lid `u_x = 8 (1 + tanh(8 (t - 0.5))) x^2 (1 - x)^2`, including its start-up ramp.
For the viscoelastic runs t = 10 is a point on a transient, not a steady state: a Kraken
result must reproduce this start-up and be read at t = 10. Drift rates at t = 10 at
N = 64, per time unit, as central differences from the run `ve_re1_n64_t30` (vortex
rows at t = 9 and 11, energy rows at t = 9.904 and 10.096 of `energy.csv`): `x_centre`
-2.25e-4, `y_centre` +1.51e-4, `psi_min` +5.7e-5, `kinetic_energy_mean` -1.64e-5,
`elastic_energy_mean` +1.10e-2. A read-out
0.1 away from t = 10 thus shifts these scalars by 0.4 % (elastic energy) to 2.1 %
(`y_centre`) of their 64 -> 128 step. Profile lines can be about twice as sensitive
(inferred, assuming the same decay time: their t = 10 -> 30 drift reaches 1.01 of the
64 -> 128 difference for `uy` on x = 0.5 and 0.98 for `ux` on y = 0.9, against 0.51 for
`y_centre`).

`x_centre, y_centre, psi_min` come from the 4 x 4 tensor-product cubic + Newton locator of
`extract_cavity.py` (full rule in the docstring of `vortex_centre`), applied to
`psi = int_0^y u_x dy'` integrated from the bottom wall by the midpoint rule. Profile
values are linear interpolations between the two cell-centre rows or columns around the
line; `C = I + (lambda/etaP) tau`. A Kraken result must be reduced with the same script,
`bench/rheotool/cavity_ve_wi1/extract_cavity.py`, on the same N x N cell centres.

## Input expected by `extract_cavity.py`

`python extract_cavity.py <case_dir> <N> <out_prefix> [--lambda 1 --etaP 0.5]`, with in
`<case_dir>`:

- one directory per written time, named by the time (`1`, ..., `10`), holding `U` and,
  for a viscoelastic run, `tau` (plain or `.gz`) in OpenFOAM ASCII volField format: an
  `internalField nonuniform List<vector>` (or `List<symmTensor>`) of N^2 values, cells
  ordered x fastest, then y (cell `i + N j` has centre `((i + 1/2)/N, (j + 1/2)/N)`),
  `tau` components in the order `xx xy xz yy yz zz`;
- `kinEner.txt`: whitespace-separated columns `t`, `(0.5/N^2) sum |U|^2`,
  `(0.5/N^2)(lambda/etaP) sum tr(tau)`. The script copies these energies and does not
  recompute them, so a Kraken run must log them itself with these formulas, including a
  row at the time nearest to t = 10.

## What can be gated at N = 64

Newtonian rows (entry gate). The flow is steady at t = 10 and the N = 64 / 128 / 256
series converges at second order (observed 2.08 on `x_centre`, 2.00 on `y_centre`, 2.02
on `psi_min` and on `kinetic_energy_mean`). Distance of the N = 64 values from the
values extrapolated with the observed order (Richardson): 4.22e-4 on `y_centre`, 1.66e-4
on `psi_min`, 7.91e-5 on `kinetic_energy_mean`, 1.20e-5 on `x_centre`. These are
RheoTool's own N = 64 errors; a correct Kraken result at N = 64 carries its own, which is
not known and can be larger or of the other sign: these numbers inform the tolerance,
they do not fix it. `y_centre` only with the bottom-wall `psi` construction: at N = 64
other constructions move it by up to 5.0e-4 (bicubic-spline minimum), more than
RheoTool's own N = 64 error; extrapolated with their observed orders, their limits agree
to 3e-7. Velocity profiles, L2 ratio (log2 of the relative L2 difference 64 -> 128 over
that of 128 -> 256): 1.99 to 2.03 on x = 0.5, y = 0.5 and y = 0.75, an observed order.
On y = 0.9 and y = 0.95 the linear sampling weight changes with N and the L2 ratio is
0.25 to 2.6, not an order (1.9 to 2.06 with 4-point cubic sampling across the rows, the
finer profile still interpolated linearly onto the coarser points); at fixed N this does not
matter if both codes use the same rule.

Viscoelastic rows (`ve_re1_n64`). The mesh series is not in an asymptotic range, so a
tolerance can only come from measured differences, not from a formal order.

- Gate candidates: `x_centre` (64 -> 128 step 1.265e-3; steps grow), `y_centre`
  (7.28e-4), `psi_min` (5.02e-4), `kinetic_energy_mean` (1.73e-4). A tolerance of one
  64 -> 128 step tests agreement with RheoTool's N = 64 discretisation error, not
  accuracy: RheoTool itself moves by more than that from N = 64 to 256 (`x_centre`
  2.73e-3, 2.16 steps; `y_centre` 1.10e-3, 1.51; `psi_min` 8.40e-4, 1.68;
  `kinetic_energy_mean` 2.85e-4, 1.65). If the series stays monotone beyond N = 256 (not
  measured), RheoTool's N = 64 error is at least that large, and a more accurate code
  can differ from it by as much. The tolerance is left to PR 4; one option is a
  tolerance of at least |f256 - f64|. A gate on the direction of Kraken's own steps
  would reject a correct code: its error can have the other sign, so it approaches the
  limit from the other side, and for `x_centre` the RheoTool steps grow, so there is no
  RheoTool limit to move towards. `y_centre` only with the identical `psi`
  construction: five constructions spread it over 7.65e-4 at N = 64 (located with a
  bicubic-spline minimum), while the locator itself differs from a bicubic-spline
  minimum by 2.0e-5. Velocity profiles: the difference between meshes shrinks slowly on
  most lines (L2 ratio 0.21 to 0.75) and grows for `uy` on x = 0.5, and for `ux` on
  y = 0.95 under the linear sampling rule only (with 4-point cubic sampling across the
  rows it shrinks, ratio 0.36).
- Monitor, with at least the 64 -> 128 -> 256 spread as tolerance: stress profiles and
  interior stress extrema. Their L2 ratio is below 1, or their differences grow, on
  x = 0.5, y = 0.5, 0.9 and 0.95 (`tau_xy` on y = 0.95 is borderline, 0.99; the ratio
  depends on the construction, defined in the analysis comment); on y = 0.75, 4 of the 6 stress extrema change the sign
  of their step. Example: the largest `tau_yy` sample on y = 0.95 is 47.0 / 51.9 / 45.2
  for N = 64 / 128 / 256.
- Not safe to gate at any single N:
  - `elastic_energy_mean` (2.015 / 2.304 / 2.612). Not for its growing steps alone (those
    of `x_centre` grow faster, step ratio 1.16 against 1.07), but because its steps are
    14 % of its value and come from the band y > 15/16 under the lid (+0.30, then +0.32;
    the rest of the cavity changes by -0.014, then -0.008), where the value follows the
    lid-row resolution. `x_centre` is an interior quantity whose steps are 0.3 % of its
    value;
  - stresses in the cell row under the lid, e.g. `tau_xx` on the last x = 0.5 sample,
    h/2 below the lid: 57.2 / 153.2 / 378.1; and the domain maxima of C (not in these
    files): `C_xx` 128.8 / 332.3 / 800.0 in the cell row under the lid (x = 0.37 / 0.39 /
    0.41), `C_yy` 147.0 / 204.3 / 276.2 in the same row at x = 0.98;
  - wall-adjacent extrema, taken on the sample h/2 from a wall, which move with h by
    construction: maximum of `ux` on x = 0.5, maximum of `tau_yy` on y = 0.75, minimum of
    `tau_xx` on y = 0.95.

Run at Re = 1 (`rho = 1`): at N = 128, Re 0.01 moves the vortex centre by 1.41e-3,
0.93 times the 128 -> 256 mesh displacement. Halving the time step changes `y_centre`
by 2.6e-6 at N = 64, 2.4e-7 at N = 128 and 9.0e-8 at N = 256.
