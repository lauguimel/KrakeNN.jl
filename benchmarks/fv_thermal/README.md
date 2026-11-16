# FV thermal benchmarks

Scripts for the steady FV energy equation, its thermal coupling and its adjoint
(documented in `docs/src/users/fv-thermal.md`).

## Scripts

- `nahme_reference.py` writes the independent continuum reference of the Nahme
  channel (viscosity `nu(T) = nu0 exp(-beta (T - Tw))` with viscous heating,
  Poiseuille and Couette, plus the fold branch) to
  `test/reference/fv_thermal_nahme/`. It uses SciPy collocation only, no KrakeNN
  code. Regenerate from the repository root:
  `conda run -n kraken-v0-3-figures python benchmarks/fv_thermal/nahme_reference.py`.
  The provenance and the frozen error bounds are in that directory's `README.md`.
- `e5_compare.jl` compares the FV solver with the LBM code on the E5
  six-cylinder periodic cell, same voxel mask and same lattice coefficients,
  in three stages: (1) flow with the LBM temperature imposed, (2) heat only with
  the LBM face velocities and heat source imposed, (3) full coupling by
  continuation in the heating coefficient, adjoint of the mean temperature rise
  `m` against the LBM gradient `dm/dtheta` (8 law parameters), and central finite
  differences on `eta0` and `beta`. It ends with a `[thresholds]` table.

- `die_predict.jl` predicts flow and temperature in the tapered extrusion die
  of the LBM open-die driver (branch `feat/gnf-open-steady`) with the steady
  FV Newton solver and a given law, on the same cells as the LBM truth.

## Running `e5_compare.jl`

Input: an export directory of the LBM case with `case.toml`, the binary fields
(`ux`, `uy`, `uf`, `vf`, `T`, `is_solid`, `nu_from_gradient_base`,
`I1_from_gradient_base`; `nu` and `I1` at 600x200) and optionally `gradient.toml`
(LBM `dm/dtheta`, needed for the stage 3 gradient comparison). The default is
`tmp/e5_export/<grid>`.

Environment variables:

| Variable | Values | Default |
|---|---|---|
| `KRAKEN_E5_BACKEND` | `cpu`, `cuda` (needs CUDA.jl and CUDSS.jl) | `cpu` |
| `KRAKEN_E5_GRID` | `300x100`, `600x200` | `300x100` |
| `KRAKEN_E5_EXPORT` | export directory | `tmp/e5_export/<grid>` |
| `KRAKEN_E5_OUT` | output directory | `tmp/e5_compare_out/<grid>_<backend>` |
| `KRAKEN_E5_STAGES` | comma list of `1,2,3` | `1,2,3` |
| `KRAKEN_E5_FD` | `1`, `0` (stage 3 finite differences) | `1` |
| `KRAKEN_E5_INIT` | `zero`, `lbm` (initial flow of stages 1 and 3) | `zero` |

```sh
KRAKEN_E5_BACKEND=cuda KRAKEN_E5_GRID=600x200 julia --project=. benchmarks/fv_thermal/e5_compare.jl
```

Use a GPU for real runs: stage 3 at 600x200 has 480 000 unknowns. On CPU run one
stage at a time (`KRAKEN_E5_STAGES=2` is the cheap one).

Outputs, in the output directory: `summary.toml` (export consistency checks,
one table per stage with errors against the LBM fields, Newton and continuation
statistics, round-off floor diagnostics, timings, GPU memory, adjoint and
finite-difference results, `[thresholds]`) and `stage<k>_fields.csv` (cell
fields, FV and LBM side by side).

## Thresholds

`[thresholds]` is evaluated on stage 3 and applies to grid 600x200 only (decided
2026-11-08 from the job measurement of that day, with margin):

- `|m_FV / m_LBM − 1| <= 2 %`;
- `|flow-rate ratio − 1| <= 2 %`;
- `||dm/dtheta_FV − dm/dtheta_LBM|| / ||dm/dtheta_LBM|| <= 3 %`, each component
  within 6 %;
- the 8 signs agree.

Each item has `limit`, `applies`, and, once stage 3 has run, `value` and (for
600x200) `pass`; `all_pass` summarises. For other grids, or when stage 3 was not
run, `applies = false` or `evaluated = false` and no `pass` is written. The
constants are `THRESHOLD_*` in `e5_compare.jl`; do not loosen them to make a run
pass.

## Test

`test/analytical/incns_fv_e5_compare_smoke_test.jl` runs the three stages and the
thresholds logic on a small synthetic cell on CPU (a few seconds).

## Running `die_predict.jl`

Geometry and data are those of `run_gnf_open_steady`: inlet gap `G` cells
(`H = G/2`, length `Nx = 7H`), linear taper 2H → H/2 from 1.5H to 4.5H (or
`tanh`), Poiseuille inlet of mean `U = Re nu_ref / H` at `theta_in`, walls and
die voxels at `theta_wall`, outlet at `p = 0` with zero normal gradients,
`alpha` and `sigma` from `Pe` and `Na` as in `gnf_open_thermal_groups`. The
solve is a `sigma = 0` Newton solve (retried with pseudo-transient
continuation `dtau0 = 1` if it fails), then a heating ladder up to the nominal
`sigma` with step halving.

| Variable | Values | Default |
|---|---|---|
| `KRAKEN_DIE_BACKEND` | `cpu`, `cuda` (needs CUDA.jl and CUDSS.jl) | `cpu` |
| `KRAKEN_DIE_G` | even inlet gap in cells | `64` |
| `KRAKEN_DIE_PROFILE` | `linear`, `tanh` | `linear` |
| `KRAKEN_DIE_PE`, `KRAKEN_DIE_NA`, `KRAKEN_DIE_RE`, `KRAKEN_DIE_NUREF` | groups | `30`, `0.05`, `0.02`, `0.3` |
| `KRAKEN_DIE_THETA_IN`, `KRAKEN_DIE_THETA_WALL` | temperatures | `1.0`, `0.0` |
| `KRAKEN_DIE_EPS` | shear-rate regularisation of the GNF law | `0.0` |
| `KRAKEN_DIE_LAW_FILE` | Julia file defining `die_viscosity(c)` | Carreau–WLF law of the truth |
| `KRAKEN_DIE_DTAU0` | pseudo-transient start | none, then `1.0` |
| `KRAKEN_DIE_TRUTH` | directory with `ux.bin`, `uy.bin`, `theta.bin` (Nx × Ny Float64), `truth.toml` (`dp`, `mass_in`) | none |
| `KRAKEN_DIE_OUT` | output directory | `tmp/die_predict/G<G>_<profile>_<backend>` |

```sh
KRAKEN_DIE_BACKEND=cuda julia --project=. benchmarks/fv_thermal/die_predict.jl
```

Outputs: `summary.toml` (case, Newton record per heating step, `dp`,
`dp_tilde = dp / dp_scale`, mass in/out, mixing-cup outlet `theta`, energy
balance terms and relative imbalance, timings, GPU memory, optional
comparison) and `ux.bin`, `uy.bin`, `p.bin`, `theta.bin`, `is_solid.bin` (raw
Float64, Nx × Ny, column-major, the layout of the LBM `r.ux`, `r.theta`) and
`fields.csv`. `dp` is the difference of the mean pressure over the fluid cells
of columns 1 and Nx, the same columns as the LBM `dp`. A smoke run (G = 8, CPU)
is in `test/analytical/incns_fv_energy_open_bc_test.jl`.

Known differences with the LBM truth: the LBM outlet imposes a density and
zero transverse velocity on the outlet nodes, the FV outlet imposes `p = 0` on
the face and zero normal gradient of `u` and `v`; the LBM inlet node sits at
`x = 1/2`, the FV inlet face at `x = 0`; the LBM temperature is advected by
projected node velocities, the FV temperature by the Rhie–Chow flux; the FV
viscosity is evaluated on faces (face temperature = mean of the two cells),
the LBM viscosity on nodes.
