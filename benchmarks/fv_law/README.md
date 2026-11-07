# Learned viscosity: prepared GPU validation and CPU parameter cost

See [validation.md](validation.md) for the focused test counts, derivative
errors, compilation breakdown, allocation diagnosis, and local GPU guard.

`law_gpu.jl` is prepared, **not GPU validated or submitted**. It uses a
512 × 512 periodic Stokes channel in Float64 by default. It compares converged
Newton solutions and every residual-history entry for `ClassicalLaw` and
`FVGNFViscosity`, then solves a monotone two-input `LearnedViscosity` with a
spatially varying prescribed temperature field. It checks the complete steady
gradient against central differences in two seeded random directions at two
fixed steps. Temperature is fixed when differentiating network parameters.

The fixed gates are: relative solution difference ≤ 1e-12; residual-history
agreement with relative 1e-13 and absolute 1e-14 (the final arithmetic floor);
relative directional derivative difference ≤ 1e-6 for derivatives above 1e-10.
Device adaptation must produce an isbits kernel argument with device parameter
and field arrays. Warm Newton and gradient calls must allocate zero device
bytes. Host allocations are reported separately. First calls include compilation;
warm calls reuse the solver and adjoint workspace. Output CSVs contain classical
and learned residual histories, directional derivatives, and timing/allocation
receipts. These are proposed gates, not measured GPU results.

From an environment already containing CUDA, CUDSS, and this Kraken checkout:

```sh
KRAKEN_REQUIRE_GPU=1 julia --startup-file=no --project=env benchmarks/fv_law/law_gpu.jl
```

`KRAKEN_LAW_N` overrides the size and `KRAKEN_LAW_OUTPUT` selects the output
directory. Without `KRAKEN_REQUIRE_GPU=1`, unavailable CUDA or cuDSS prints a
skip and starts no CPU simulation. With it, missing GPU support is an error.
The local macOS guard was checked: `CUDA.functional()` returned false; the
optional path exited 0 and the required path exited 1. No packages were installed.

For Aqua, copy the checkout as `$WORKDIR/src_law`, then run `setup_law.sh`
manually on the login node. It prepares `$WORKDIR/env` and `$WORKDIR/depot`.
`law_gpu.pbs` requests one A100, 64 GB host memory and two hours, with
`JULIA_CUDA_USE_COMPAT=false`. Set `WORKDIR` explicitly or use
`$HOME/kraken_law`; supply the allocation externally with `qsub -P`.
Neither setup nor submission was executed here.

## Isolated CPU parameter-dual cost

```sh
julia --startup-file=no --project=. benchmarks/fv_law/parameter_sweep_cost.jl
# Optional positional argument: destination CSV path.
```

`parameter_sweep_cost.csv` records a 64 × 64 CPU Float64 measurement on Apple
M3, Julia 1.12.5, one thread. Each model has two inputs, one hidden layer,
monotone stress, and hidden width 1, 8, or 35. Their actual parameter counts are
10, 59, and 248; the requested approximate counts are not padded with inactive
parameters. Each dual residual evaluation carries four parameter derivatives,
requiring 3, 15, and 62 sweeps respectively.

The measured operation is the existing `Kraken._fv_param_cotangent!`:
`-lambda' * dR/dtheta`, at a seeded arbitrary velocity/pressure state and seeded
fixed `lambda`. It includes parameter seeds, residual evaluations, reduction,
and the host result. It excludes Newton iterations, Jacobian assembly,
factorization, the transposed solve, and objective derivatives. Consequently
these numbers are **not full gradient or simulation timings**. Per-sweep columns
are whole-contraction measurements divided by the sweep count, not separately
timed kernels. Cold elapsed and Julia-reported compilation times are separate;
warm elapsed is the median of three calls and allocation bytes their minimum.
Solver construction is timed separately. All three returned contractions must
match the first result exactly and contain finite values.

Measured on 2026-11-07; full details and timestamps are in the CSV:

| Hidden width | Parameters | Sweeps | Warm contraction (ms) | Warm host bytes | Cold compilation (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 10 | 3 | 5.405 | 4,656 | 714.580 |
| 8 | 59 | 15 | 94.099 | 55,248 | 382.414 |
| 35 | 248 | 62 | 1,469.918 | 704,400 | 389.812 |

At 248 parameters, this part of one gradient takes about 1.47 seconds on this
64 × 64 CPU case, before any adjoint solve or objective work. The amortized
per-sweep times are 1.802, 6.273, and 23.708 ms. The cost grows with both the
network width and the number of parameter sweeps; no per-face reverse
contraction is implemented or measured here.

The CSV is the measurement receipt; runtime and allocation figures depend on
the current implementation and machine. Compilation and package precompilation
must not be confused: the CSV excludes process startup and package loading.
