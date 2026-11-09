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
