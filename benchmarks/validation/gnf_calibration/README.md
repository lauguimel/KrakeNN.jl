# GNF Carreau calibration and adjoint cost

Two scripts on the periodic obstacle channel of `test/ad/gnf_adjoint_test.jl`
and `benchmarks/validation/gnf_adjoint/` (domain 8 × 4 units, grid 8m × 4m,
periodic in x, half-way walls in y, a half-disk of radius 1.5 on each wall,
uniform body force along x). Run from the repository root. Float64 by default.
No package beyond the repository's `Project.toml` is used; the minimiser is
written in `bfgs.jl`, shared helpers are in `common.jl`.

Parameters: the code's Carreau–Yasuda law
`CarreauYasuda(eta_0, eta_inf, lambda, a, n)`,
ν = η∞ + (η0 − η∞)(1 + (λγ̇)^a)^((n−1)/a), with η0 = ν0 = 0.17 (lattice units)
fixed, in reduced form θ = (ln η∞*, ln k*, n, a), η∞* = η∞/η0, k* = λ ν0/m².

## `calibrate_carreau.jl` — calibration (noise-free, noise)

What it does:

- Calibrated parameters: (ln η∞*, ln k*, n), with a = 2 fixed (Carreau law).
- True values: η∞* = 0.02, k* = 5, n = 0.7, the reference point of the adjoint
  check, where forward and gradient were validated.
- Data: steady velocity fields (both components at every fluid node) computed
  from rest at the true values for S = 3 body forces, 0.1, 0.2 and 0.3 times
  the reduced force G* = 5 (0.3 is the force of the adjoint check). On 32×16
  the Mach numbers are 0.026, 0.063 and 0.108.
- Misfit: J(θ) = Σ_s ½ Σ ((u(θ) − u_data)/u_ref,s)², u_ref,s the maximum fluid
  speed of the noise-free data of forcing s. RMS residual = √(2J/M) over the M
  observations.
- Gradient: `gnf_steady_gradient` (steady discrete adjoint), one adjoint solve
  per forcing, tolerance 1e-8, bordered GMRES by default (`--solver=gmres`; the
  K-step solver with `--solver=kstep`). Forwards to a distance estimate below
  1e-10, warm-started from the last accepted iterate.
- Minimiser: BFGS with a dense inverse-Hessian update and an Armijo
  backtracking line search (`bfgs.jl`). A step is limited to 0.25 in ln η∞*,
  0.25 in ln k* and 0.1 in n; a point with n outside (0.05, 2) or whose forward
  does not converge counts as a failed trial and shortens the step.
- Stopping rule: noise-free runs stop as soon as the RMS residual is ≤ 1e-6.
  Every run also stops when ‖∇J‖∞ has fallen by 1e8 from the start, when an
  accepted step is ≤ 1e-9 in every component, when J decreases by ≤ 1e-12 J over
  an iteration, when the line search finds no decrease after 30 reductions, or
  after 100 iterations. The stop reason is in the CSV.
- Five noise-free starts, fixed in the script (`CAL_STARTS`): factors on
  (η∞*, k*, n) of (1.2, 1.2, 1.2), (0.8, 0.8, 0.8), (1.5, 0.6, 1.3),
  (0.5, 1.5, 1.3) and (1.4, 0.5, 0.7), each component 20 to 50 % away.
- Noise: Gaussian, standard deviation 0.1 %, 1 % and 5 % of u_ref,s on each
  component at each fluid node; ten seeds fixed in the script
  (`CAL_NOISE_SEEDS`), the same draws for the three levels; noisy runs start
  from start 1.
- Pass criterion: every noise-free run reaches the RMS target. Parameter errors
  are reported without a threshold; noise runs have no pass/fail criterion.

Commands:

```bash
# CPU smoke run (16x8, start 1, three iterations; passes when the misfit decreases)
julia --project=. benchmarks/validation/gnf_calibration/calibrate_carreau.jl \
      --backend=cpu --grid=16x8 --mode=smoke --out=output/gnf_calibration

# GPU campaign: 5 noise-free runs and 30 noise runs
export JULIA_CUDA_USE_COMPAT=false   # on clusters whose nodes need the system driver
julia --project=. benchmarks/validation/gnf_calibration/calibrate_carreau.jl \
      --backend=cuda --grid=32x16 --mode=all
```

Options (`--key=value`): `backend` (`cuda`, default, or `cpu`), `grid`
(`32x16`; any 8m × 4m), `precision` (`Float64`), `solver` (`gmres` or
`kstep`), `mode` (`all`, `noisefree`, `noise`, `smoke`), `starts` (`1:5`),
`levels` (`1:3`), `draws` (`1:10`), `maxiter` (100; 3 in smoke mode), `out`
(`benchmarks/validation/gnf_calibration/results`), `allow-unknown-commit`
(`no`). Exit status 0 when the pass criterion holds, 1 otherwise. With
`--precision=Float32` the forward does not reach the 1e-10 distance and the run
stops at data generation with a message (seen on 16×8, CPU); the scripts are
meant for Float64.

Outputs, stem `carreau_calibration_<backend>_<grid>_<mode>_<local time>`:

- `.csv`: one row per run (start or noise draw): kind, start, noise level and
  seed, initial and final θ, final (η∞*, k*, n) and their signed relative
  errors, initial and final J and RMS, the RMS against the noise-free data,
  stop reason, iterations, objective and gradient evaluations, forward solves
  (and failed ones), adjoint solves (and unconverged ones), adjoint iterations
  and operator products, largest adjoint residual, forward steps, wall times
  (total, forwards, adjoints).
- `_history.csv`: one row per accepted iterate (J, RMS, ‖∇J‖∞, step, line-search
  factor and reductions, θ).
- `.log`: everything printed, starting with the provenance header.

## `cost_bench.jl` — cost

Times, separately and on each grid (default 32×16, 64×32, 128×64, 256×128),
for each forcing:

- one forward solve from rest to a distance estimate below 1e-10;
- one adjoint gradient (`gnf_steady_gradient`, tolerance 1e-8) per solver
  (`kstep` and `gmres` by default), with its linearisation, solve and
  parameter-gradient parts, iterations and operator products;
- the 2p forward solves of a central finite-difference gradient (step
  1e-4 |θ_p|), warm-started from the base state.

Two points: `interior`, the four parameters at a = 2.5 (p = 4, 8 forwards per
forcing), and `calibration`, the three parameters of the calibration at a = 2
(p = 3, 6 forwards per forcing). The observation is that of the adjoint check
(targets 0.8 u). One warm-up repeat on the first grid and forcing, marked
`warmup=true`, is excluded from the medians. The finite-difference gradient is
compared with the adjoint one in informational rows. No threshold.

Commands:

```bash
# CPU, one grid, one repeat
julia --project=. benchmarks/validation/gnf_calibration/cost_bench.jl \
      --backend=cpu --grids=32x16 --repeats=1 --out=output/gnf_cost_bench

# GPU, the four grids, three repeats
julia --project=. benchmarks/validation/gnf_calibration/cost_bench.jl --backend=cuda
```

Options: `backend` (`cuda`, default), `grids`, `precision` (`Float64`),
`solvers` (`kstep,gmres`), `points` (`interior,calibration`), `forcings`
(`1:3`), `repeats` (3), `fd-step` (`1e-4`), `out`, `allow-unknown-commit`.
On the CPU backend the timings depend on the thread counts in the log header.

Outputs, stem `gnf_cost_bench_<backend>_<local time>`: `.csv`, one row per timed
task (`forward_rest`, `adjoint`, `fd_forward`, and `fd_vs_adjoint`
comparisons); `_summary.csv`, one row per grid, point, forcing and solver with
the medians and the ratios adjoint / 2p forwards and adjoint / one forward;
`.log`.

## Provenance

Every log starts with the script, the git commit (`git rev-parse HEAD`), its
source, the branch and the modified tracked files, then Julia, OS and CPU,
thread counts, the Kraken, KernelAbstractions, CUDA.jl and ForwardDiff
versions, backend, precision, device, CUDA runtime and driver, and the options.
In a copy without `.git`, set `KRAKEN_GIT_COMMIT` to the commit the code was
copied from; otherwise the scripts stop with a message, unless
`--allow-unknown-commit=yes` (the log then says `git_commit=unknown`). Logs
print paths relative to the working directory.

## GPU backend

`--backend=cuda` loads CUDA before the run starts and stops with a message if
`CUDA.functional()` is false. The forward, the linearised operator products and
the GMRES vectors run on the device; the linearisation blocks, the parameter
gradient sum, the misfit and the minimiser run on the host.

## Test

`test/ad/gnf_calibration_test.jl` (in the test suite, CPU only, no CUDA):
the minimiser on an ill-conditioned quadratic, with an infeasible region and a
step cap, the command-line helpers, the 16×8 synthetic data and noise, and a
three-iteration calibration whose misfit decreases.

## Status

- 2026-11-04, CPU (macOS arm64, Julia 1.12.5), the scripts of this
  directory on top of commit 8d25bd318: the smoke run and a full noise-free calibration from
  start 1 on 16×8 met the stated criteria (16×8 is the smoke grid; the
  campaign grid is 32×16). The 32×16 CPU cost bench ran with one repeat.
- No GPU run yet; results are planned for v0.7.
