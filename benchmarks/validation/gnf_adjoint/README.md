# GNF adjoint GPU validation

Repeats the periodic obstacle-channel case of `test/ad/gnf_adjoint_test.jl`
on the selected backend. Run from the repository root. No Enzyme is needed.

## Checks and frozen gates

Checks run in the order P, F, A, C, S. Gate sources are the reference test's
testsets G, T5, and `case 32×16 (forward)`; no thresholds are tuned here.

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
- **A — gradient (T5):** finest state, adjoint tolerance `1e-8`, parameters
  `(ln η∞, ln k, n, a)` at `(log(0.02), log(5), 0.7, 2)`. Central differences
  use `h = (1e-2, 1e-3, 1e-4, 1e-5)`, with perturbation `h*abs(θ[p])` and
  chains on the selected backend starting from the level-1 state. For each
  parameter choose the best h and require:

  ```math
  |g_p - \mathrm{FD}_p| \le 10^{-3}\|g\|_2/2 + 10^{-3}|\mathrm{FD}_p|.
  ```
- **C — informational:** compare the Float64 gradient with the same procedure
  using the CPU operator, on the same finest state, observation, blocks and
  right-hand side. There is no frozen gate for this comparison.
- **S — informational, optional:** Float32 operator products on the same
  Float64 linearisation; attempt adjoint tolerance `1e-5`, record convergence
  and the achieved explicit relative residual, and compare the gradient with
  Float64. A missed tolerance is reported, not gated. No Float32 forward
  finite-difference check is run: the Float32 forward floor makes it meaningless.

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

The script accepts `--backend=cuda|cpu`, `--out=DIR`, `--float32=yes|no`;
defaults are `cuda`, `benchmarks/validation/gnf_adjoint/results`, and `yes`.
CUDA is checked before any validation; failure exits immediately with code 1.
Any failed gated check also exits with code 1. C and S are informational.

## Output and provenance

CSV and log files share the stem `gnf_adjoint_gpu_<backend>_<yyyymmdd_HHMMSS>`
in `--out`; the timestamp is UTC. CSV columns are `check, parameter, value,
reference, abs_err, rel_err, bound, pass`. Parameter rows contain signed
gradient values; vector comparisons contain norms and norm errors. P bounds
apply to relative error, A bounds to absolute error; F predicates are named.
Informational and skipped rows are explicitly labelled, with no gate bound.
The best h for each A parameter is recorded in its own informational row.
The log mirrors stdout/stderr and records commit, dirty-status line count,
hostname, Julia version, backend/device, CUDA runtime, precision, start/end
times and elapsed time per check. Preserve both files together when citing a run.

## Status

GPU run not yet performed.
CPU smoke run done on 2026-11-02 (macOS arm64, Julia 1.12.5): exit 0,
all nine executed gated checks passed; P skipped. The four A gradients,
best h values and bounds match the T5 criterion lines at their printed
precision. C relative difference was zero, as expected on CPU, where both operators are
CPU operators. Informational S converged at
`1e-5`, with achieved relative residual about `9.18e-6` and gradient relative
difference about `1.43e-7`.
