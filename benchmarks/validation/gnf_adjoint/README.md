# GNF adjoint GPU validation

Repeats the periodic obstacle-channel case of `test/ad/gnf_adjoint_test.jl`
on the selected backend, and adds the accuracy checks (the adjoint
gradient against finite differences). Run from the repository root. No Enzyme is
needed.

The test point is θ = (ln η∞*, ln k*, n, a) = (log(0.02), log(5), 0.7, a) with
`--a=2.5` by default, an interior value of a: at a = 2, the reference point of
the test, a central difference in a crosses a < 2. The committed GPU results
below were produced at a = 2.0 with the four decade steps only; reproduce them
with `--a=2.0 --hs=decades`.

## Checks and frozen gates

Checks run in the order P, F, A, accuracy, C, S, K. Gate sources for P, F and A are the
reference test's testsets G, T5, and `case 32×16 (forward)`; the accuracy gates are
written as constants at the top of the script. No
threshold is tuned here.

- **P — parity (G):** the same CPU steady state and Float64 local blocks feed
  CPU and device operators. JVP and VJP relative errors must each be ≤ `1e-14`;
  adjoint solutions at tolerance `1e-10` must differ relatively by ≤ `1e-8`.
  This check is skipped with `--backend=cpu`.
- **F — forward:** Float64 forward LBM on the selected backend, with the
  reference chain's nominal distances `1e-10`, `1e-12`, `1e-14`. Only the first
  level uses the distance estimator; subsequent levels each take the same K
  additional steps derived from its contraction. Steps, contraction, estimated
  distance and host Newton distance are recorded. The reference forward gates
  require level 1 convergence, Newton distance ≤ `1e-9`, a finest distance no
  larger than level 1, Mach < `0.15`, and viscosity ratio > `2`.
- **A — gradient (T5):** finest state, adjoint tolerance `1e-8` (bordered
  GMRES, `gnf_adjoint_solve`), parameters `(ln η∞, ln k, n, a)` at the test
  point. Central differences use the steps of `--hs` (default: the sweep below,
  which contains the four decade steps exactly), with perturbation
  `h*abs(θ[p])` and chains on the selected backend starting from the level-1
  state. For each parameter choose the best of `h = (1e-2, 1e-3, 1e-4, 1e-5)`
  and require:

  ```math
  |g_p - \mathrm{FD}_p| \le 10^{-3}\|g\|_2/2 + 10^{-3}|\mathrm{FD}_p|.
  ```
- **accuracy — gated.** With the budget
  b_p = 1e-6 |g_p| + 1e-10 max(1, ‖g‖∞) per component:
  - *accuracy:* |g_p − FD_p| ≤ b_p at two neighbouring steps (finest forward
    level);
  - *sweep* (`--hs=sweep`, skipped with `--hs=decades`): steps
    h = 1e-2 … 1e-5, three per decade (10 steps); the least-squares slope of
    log|g_p − FD_p| against log h on a window of four consecutive steps (one
    decade) lies in [1.7, 2.3] for at least one window; the CSV lists the window
    used and the slopes of all windows;
  - *solves:* the gradient at adjoint tolerance 1e-8 against 1e-10 (finest
    state), and the gradient from the level-1 forward (target 1e-10) against the
    level-3 forward (target 1e-14), both at adjoint tolerance 1e-10, differ by
    ≤ b_p/10; the Newton distances of the two levels are recorded;
  - *branch:* for every perturbation (4 parameters × steps × 2 signs), the
    number of fluid nodes with I1 < 1e-12 (the zero-shear branch, as counted by
    `gnf_linearise`) at the finest perturbed state equals the base count; any
    change fails. Carreau–Yasuda has no viscosity bound, so no node can cross
    one (one informational row).
- **C — GPU against CPU:** compare the Float64 gradient with the same procedure
  using the CPU operator, on the same finest state, observation, blocks and
  right-hand side. With `--backend=cuda` each component must agree within b_p
  (accuracy budget); with `--backend=cpu` both operators are CPU operators and the rows are
  informational.
- **S — informational, optional:** Float32 operator products on the same
  Float64 linearisation; attempt adjoint tolerance `1e-5`, record convergence
  and the achieved explicit relative residual, and compare the gradient with
  Float64. A missed tolerance is reported, not gated. No Float32 forward
  finite-difference check is run: the Float32 forward floor makes it meaningless.
- **K — informational:** the gradient on the same finest state and observation
  through `gnf_steady_gradient` with its default `:kstep` solver (tolerance
  1e-8), against A.

With CUDA selected, forward LBM, JVP/VJP kernels and GMRES vectors and vector
operations run on the device. GMRES small dense algebra stays on the host.
Local linearisation blocks are built on the host in Float64, then copied to
the device. The parameter gradient sum, observations and Newton-distance
diagnostic stay on the host. The forward populations and relaxation times
stay on their backend between levels; diagnostic host copies are explicit.
S retains Float64 GMRES vectors and gauge; only operator products use Float32.

## Prerequisites and commands

Prepare a Julia 1.11 or 1.12 environment instantiated from this repository's
`Project.toml` before submission. CUDA must be available in that environment
and `CUDA.functional()` must be true for a GPU run. The PBS job performs no
package installation or updates. It uses PBS Pro and one A100 GPU, and sets
`JULIA_CUDA_USE_COMPAT=false` before launching Julia.

Local CPU smoke run (including the informational Float32 operator):

```bash
julia --project=. benchmarks/validation/gnf_adjoint/run_gnf_adjoint_gpu.jl --backend=cpu --out=output/gnf_adjoint_cpu_smoke
```

PBS submission, from the prepared repository root:

```bash
qsub -P <project-code> benchmarks/validation/gnf_adjoint/run_gnf_adjoint_gpu.pbs
qsub -P <project-code> -v JL=/path/to/julia,PROJECT=.,OUT=benchmarks/validation/gnf_adjoint/results benchmarks/validation/gnf_adjoint/run_gnf_adjoint_gpu.pbs
```

The script accepts `--backend=cuda|cpu`, `--out=DIR`, `--float32=yes|no`,
`--a=A` (test value of a), `--grid=NXxNY` (8m × 4m, e.g. `32x16`, `64x32`,
`128x64`, `256x128`), `--hs=sweep|decades` and `--allow-unknown-commit=yes|no`;
defaults are `cuda`, `benchmarks/validation/gnf_adjoint/results`, `yes`, `2.5`,
`32x16`, `sweep` and `no`. On the larger grids of the cost bench, use four steps:
`--hs=decades`. The job script forwards
extra options through the `EXTRA` variable, e.g.
`qsub -v EXTRA="--grid=64x32 --hs=decades" ...`.
CUDA is checked before any output file is opened; failure exits immediately with
code 1 and a message. Any failed gated check also exits with code 1. S and K are
informational, and so is C on the CPU backend.

## Output and provenance

CSV and log files share the stem
`gnf_adjoint_gpu_<backend>_<Nx>x<Ny>_a<a>_<yyyymmdd_HHMMSS>` in `--out` (before
2026-11-04: `gnf_adjoint_gpu_<backend>_<yyyymmdd_HHMMSS>`); the timestamp is UTC. CSV columns are `check, parameter, value,
reference, abs_err, rel_err, bound, pass`. Parameter rows contain signed
gradient values; vector comparisons contain norms and norm errors. P bounds
apply to relative error, A bounds to absolute error; F predicates are named.
Informational and skipped rows are explicitly labelled, with no gate bound.
The best h for each A parameter is recorded in its own informational row;
`accuracy FD data` rows hold every finite difference at the finest level.
The log mirrors stdout/stderr and records the commit, its source, the branch and
the modified tracked files, the Julia, Kraken, KernelAbstractions, CUDA.jl and
ForwardDiff versions, backend/device, CUDA runtime and driver, precision, the
options, start/end times and elapsed time per check. The commit comes from
`git rev-parse HEAD`; in a copy without `.git`, set `KRAKEN_GIT_COMMIT` to the
commit the code was copied from (`qsub -v KRAKEN_GIT_COMMIT=...`); otherwise the
script stops with a message, unless `--allow-unknown-commit=yes` (the log then
says `git_commit=unknown`). Preserve both files together when citing a run.

## GPU evidence

Files: `results/gnf_adjoint_gpu.csv` and `results/gnf_adjoint_gpu.log`, copied
from the Aqua output directory (originally
`gnf_adjoint_gpu_cuda_20261102_173836.{csv,log}`, renamed to stable names). The
only edit is on the last line of the log (`CSV=... LOG=...`), where the absolute
cluster home path was replaced by `~`. The scheduler log of the job is not
archived (it holds the job environment).

| Item | Value |
| --- | --- |
| Code | branch `feat/gnf-adjoint`, commit `12fd5d3ba`, kept reachable by the tag `archive/gnf-adjoint-gpu-check`. The branch was later rebased onto `dev/platform` 660b8e871; the only source changes that brought in (the `.krk` `Backend` directive, in `src/io/krk/directives.jl`, `src/io/krk/parser.jl` and `src/simulation_runner.jl`) are not used by this script. The cluster copy is not a git checkout (the log says `git_commit=unknown`); a checksum comparison (`rsync -n -rc`) of `src/`, `ext/`, `test/` and `benchmarks/` against the commit, excluding `results/` and `out/`, listed no differing file |
| Scheduler job | `27544976.aqua`, exit status 0, node `gpu0n009`, 4 CPUs, 32 GB, 1 GPU |
| Run | started 2026-11-02 17:38:36 UTC, 76.4 s in the script; log ends with `Gated checks: PASS` |
| GPU | NVIDIA A100-SXM4-40GB (sm_80), 39.5 GiB available |
| Driver / CUDA | driver 610.57.4 (CUDA 13.3); CUDA runtime 13.2.0 |
| Julia / packages | Julia 1.12.5 (Linux x86_64), CUDA.jl 5.11.3, `JULIA_CUDA_USE_COMPAT=false` |
| Precision | Float64 for all gated checks; Float32 operator for S only |

What the CSV shows (case 32x16, as in `test/ad/gnf_adjoint_test.jl`):

- **P, gated.** JVP relative difference between CPU and device operators
  1.23e-16, VJP 8.66e-17 (bound 1e-14). Adjoint solves at tolerance `1e-10`
  (214 iterations on both backends) differ relatively by 6.25e-14 (bound 1e-8).
- **F, gated.** Forward chain 1880 / 383 / 383 steps, K = 383; Newton distances
  3.7e-10 / 1.3e-10 / 4.9e-11; Mach 0.1077 (< 0.15); viscosity ratio 2.55 (> 2).
- **A, gated.** Adjoint gradient (193 iterations, relative residual 8.26e-9)
  against central finite differences, best h = 1e-5 for all four parameters:
  absolute differences 9.44e-10 (ln η∞), 3.70e-11 (ln k), 7.10e-08 (n),
  1.96e-09 (a), against bounds 2.11e-2, 2.60e-2, 6.20e-2, 2.09e-2.
- **C, informational.** Device gradient against the CPU operator on the same
  state: relative differences 5.9e-13 to 9.1e-13 per parameter, 7.4e-13 on the
  gradient norm.
- **S, informational.** Float32 operator products reached GMRES tolerance `1e-5`
  (152 iterations, relative residual 9.09e-6); the gradient differs from Float64
  by a relative 1.47e-7 to 1.72e-7 per parameter, 1.63e-7 on the norm.

Not shown by this run: one problem size only (32x16); one run on one A100
(no H100, no repeat); no timing claim (the per-check wall times in the log
include compilation and setup and are not a performance measurement); Float32 is
informational, with no gate and no Float32 finite-difference check; the
gradient solve was the bordered GMRES (`gnf_adjoint_solve`, the `:gmres` option
of `gnf_steady_gradient`), so the `:kstep` default and `:fixed_point` were not run
on the GPU. This run predates the accuracy checks: test point a = 2.0, four decade
steps, script version of commit `12fd5d3ba`.

Command run by the job script, from the code directory:

```bash
export JULIA_CUDA_USE_COMPAT=false
julia --project=. benchmarks/validation/gnf_adjoint/run_gnf_adjoint_gpu.jl --backend=cuda --out=<output directory>
```

## Status

GPU run done on 2026-11-02 (job `27544976.aqua`, A100, exit 0, all gated checks
passed); see "GPU evidence".
CPU smoke run done on 2026-11-02 (macOS arm64, Julia 1.12.5): exit 0,
all nine executed gated checks passed; P skipped. The four A gradients,
best h values and bounds match the T5 criterion lines at their printed
precision. C relative difference was zero, as expected on CPU, where both operators are
CPU operators. Informational S converged at
`1e-5`, with achieved relative residual about `9.18e-6` and gradient relative
difference about `1.43e-7`.
CPU run of the extended script done on 2026-11-04 (macOS arm64, Julia 1.12.5,
commit 8d25bd318 plus the uncommitted script changes, default options: a = 2.5,
32x16, sweep): exit 0, all gated checks passed, P skipped. Smallest accuracy
margin: component a, 4.0e-10 against a budget of 8.3e-8; sweep slopes 2.00 on
h = 1e-2 … 1e-3 for all four components; solve differences at least 300 times
below b/10 (smallest margin: component a, adjoint tolerance 2.5e-11 against
8.3e-9); no perturbation changed the zero-shear count (0 at the base).
No GPU run of the extended script yet.
