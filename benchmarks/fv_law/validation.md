# FV scalar-law integration: validation receipts

The first sections preserve the L2 receipt. L3 measurements are appended below.
Date: 2026-11-07. Machine: Apple M3, macOS arm64, Julia 1.12.5.
Backend: KernelAbstractions CPU. Numerical integration checks: Float64;
scalar inference checks also exercise Float32 and mixed numeric metadata.
This receipt covers focused tests and the CPU parameter-cost measurement,
not the full repository suite or GPU numerical execution.

## Unchanged-test comparison

The six existing files below were run unchanged, in the same order, each
inside its own module and top-level testset. The before process loaded KrakeNN
before source modifications. Test Summary counts and rounded times are copied
verbatim from that runner's output. Both processes exited 0; no failures,
errors, or broken tests were reported. The before/after test counts match.

| Test file | Before Pass / Total | Before Time | After Pass / Total | After Time |
| --- | ---: | ---: | ---: | ---: |
| `test/platform/law_closure_test.jl` | 2421 / 2421 | 8.8s | 2421 / 2421 | 9.5s |
| `test/analytical/incns_fv_gnf.jl` | 731 / 731 | 52.8s | 731 / 731 | 57.4s |
| `test/analytical/incns_fv_adjoint.jl` | 1319 / 1319 | 36.4s | 1319 / 1319 | 37.7s |
| `test/analytical/incns_fv_immersed.jl` | 1005 / 1005 | 32.2s | 1005 / 1005 | 32.4s |
| `test/analytical/incns_fv_immersed_objectives.jl` | 149 / 149 | 16.1s | 149 / 149 | 17.1s |
| `test/analytical/incns_fv_immersed_m3_adjoint.jl` | 360 / 360 | 14.0s | 360 / 360 | 14.7s |

Before and after totals: 5985 passing tests each. The times include compilation and must not
be read as warmed simulation timings. These files contain several distinct
small test cases; no production-size CPU simulation was added.

To reproduce the runner, execute this from the repository root with
`julia --startup-file=no --project=.`:

```julia
using Test
paths = [
    "test/platform/law_closure_test.jl",
    "test/analytical/incns_fv_gnf.jl",
    "test/analytical/incns_fv_adjoint.jl",
    "test/analytical/incns_fv_immersed.jl",
    "test/analytical/incns_fv_immersed_objectives.jl",
    "test/analytical/incns_fv_immersed_m3_adjoint.jl",
]
for path in paths
    m = Module(gensym(:FVRegression))
    Core.eval(m, :(include(path) = Base.include($m, path)))
    @testset "$path" begin
        Base.include(m, abspath(path))
    end
end
```

## Added scalar-law tests

The independent run of `test/analytical/incns_fv_law_test.jl` reported:

| Test Summary | Pass | Total | Time |
| --- | ---: | ---: | ---: |
| FV law: field convention and validation | 63 | 63 | 2.1s |
| FV law: classical equivalence and unchanged structure | 77 | 77 | 6.6s |
| FV law: concrete scalar and kernel inference | 26 | 26 | 2.1s |
| FV learned law: Newton, full objective gradients and parameter reuse | 74 | 74 | 16.3s |

Total: 240 passing tests, zero failures, errors, or broken tests. The largest
relative error against the dense ForwardDiff reference was
`1.9436760419320343e-14`; the largest relative central-difference error was
`4.514116300018368e-8`. Both maxima occurred for the monotone learned law
and the residual-flux force objective. The monotone law has full gradient
checks for a state probe, residual-flux force, and surface force; the
unconstrained law has one full probe-gradient check. Every objective uses
the complete dense reference and two random finite-difference directions.
Temperature remains fixed while network parameters vary. Scalar/cell type
inference covers all six real/dual and monotone/free combinations; kernel
reflection checks the monotone parameter-dual and state-dual cases.

The final independent check used a fresh Julia process, loaded KrakeNN, then
timed the complete file inclusion. Total elapsed was **27.751042417 s,
compilation included** (`27.334223335 s` of compilation). A separate fresh
process measured `28.511797834 s`, also including compilation. Both complete
inclusions satisfy the 30-second gate; no compilation time was subtracted.
Package startup and the initial `using KrakeNN` precede the timer. The timing
refinement removed duplicate free-law force checks and redundant kernel
reflection; no numerical threshold or required objective check was weakened.

```sh
julia --startup-file=no --project=. test/analytical/incns_fv_law_test.jl
julia --startup-file=no --project=. -e 'using KrakeNN; t=@timed include("test/analytical/incns_fv_law_test.jl"); println((elapsed_seconds=t.time,compile_seconds=t.compile_time));'
```

## Allocation diagnosis and corrected implementation

Recursive numerical-type discovery through `TinyMLP` metadata inside
`LearnedViscosity` did not infer a concrete type in the FV wrapper. The resulting
scalar boxing propagated through `fv_cell_residual`. A 4 × 4 allocation profile
recorded 899 allocations / 15,225 bytes, dominated by boxed `Float64` arithmetic
in that cell-residual function. Explicit numeric-metadata type methods for
`TinyMLP` and `LearnedViscosity` removed this inference loss; constitutive
formulas did not change.

The diagnostic below measures one warmed `fv_residual!` call at 64 × 64,
without a Newton solve. The learned model has two inputs and hidden width 1.
`Dual4` carries four parameter derivatives; each time is the median of three
calls and each allocation figure their minimum host bytes.

| Residual model and parameter type | Before bytes | After bytes | Before ms | After ms |
| --- | ---: | ---: | ---: | ---: |
| GNF, Float64 | 864 | 864 | 0.804083 | 0.883500 |
| GNF, Dual4 | 1632 | 1632 | 1.596833 | 1.590459 |
| ClassicalLaw, Float64 | 864 | 864 | 0.814791 | 0.869333 |
| ClassicalLaw, Dual4 | 864 | 864 | 1.579916 | 1.610125 |
| LearnedViscosity, Float64 | 4326464 | 1056 | 9.997875 | 1.427542 |
| LearnedViscosity, Dual4 | 10593376 | 1056 | 11.722083 | 1.817833 |

The complete parameter contractions retained bit-identical norms before and
after this correction. Their final measured costs, architecture, compilation
time, and allocations are recorded in [parameter_sweep_cost.csv](parameter_sweep_cost.csv)
and explained in [README.md](README.md). These isolate `-lambda' * dR/dtheta`;
they do not time a full gradient or solve.

## Prepared GPU artifact and local guard

`law_gpu.jl` was parsed and run through its availability guard locally:
`CUDA.functional()` returned false. The optional path exited 0 with an
explicit skip; `KRAKEN_REQUIRE_GPU=1` exited 1 as intended. No numerical GPU
test, GPU timing, setup script, or HPC submission was executed. In particular,
the GPU solution, gradient, device-allocation, and device-isbits assertions
remain unexecuted gates.

Both shell files passed `bash -n`; both Julia scripts parsed successfully;
`git diff --check` passed. No dependencies were installed for these checks.

## L3 local contraction validation

Date: 2026-11-07. Apple M3, macOS arm64, Julia 1.12.5, CPU Float64.
The new `test/analytical/incns_fv_law_contraction_test.jl` passed **99/99**
checks. A fresh process, with KrakeNN loaded before the timer, took
`27.715442833 s` for the complete file inclusion, including
`27.482308501 s` of compilation. The test therefore meets the 30-second
unit-test gate without subtracting compilation. Reproduction:

```sh
julia --startup-file=no --project=. -e 'using KrakeNN; t=@timed include("test/analytical/incns_fv_law_contraction_test.jl"); println((elapsed_seconds=t.time,compile_seconds=t.compile_time));'
```

Every law (ClassicalLaw/Carreau--Yasuda, monotone LearnedViscosity, and free
LearnedViscosity) is checked in a periodic channel and around an immersed
circle. Learned laws have one fixed, spatially varying prescribed input field.
Full gradients agree between `law_gradient=:contraction` (the default) and
`:duals` for the channel probe and both immersed force definitions
(`:residual_flux`, `:surface`). The maximum relative gradient difference is
`1.2162686446611086e-13`, below the frozen `1e-12` gate.

Two seeded central-difference directions are checked for each of the six
law/geometry combinations. Immersed finite differences use residual-flux force
for the classical and monotone laws, surface force for the free law. The maximum
relative difference is `3.963468746555463e-7`, below the frozen `1e-6` gate.
Additional fixed-state comparisons cover inlet/outlet boundaries with
convection enabled, random nonzero velocity and pressure, random adjoint values,
and supplied parameters differing from those in the solver problem. Invalid
gradient-mode keywords are rejected.

The unchanged-test runner shown in the L2 receipt was rerun with the following
six paths. The before process loaded KrakeNN before any L3 source edits, and the
after process loaded the finished implementation. Both exited 0. Counts and
rounded test-summary times follow; times include compilation and are not warmed
flow timings.

| Test file | Before Pass / Total | Before time | After Pass / Total | After time |
| --- | ---: | ---: | ---: | ---: |
| `test/platform/law_closure_test.jl` | 2421 / 2421 | 7.3 s | 2421 / 2421 | 7.0 s |
| `test/analytical/incns_fv_law_test.jl` | 240 / 240 | 26.4 s | 240 / 240 | 27.4 s |
| `test/analytical/incns_fv_gnf.jl` | 731 / 731 | 50.3 s | 731 / 731 | 51.8 s |
| `test/analytical/incns_fv_adjoint.jl` | 1319 / 1319 | 35.8 s | 1319 / 1319 | 36.5 s |
| `test/analytical/incns_fv_immersed.jl` | 1005 / 1005 | 30.6 s | 1005 / 1005 | 30.5 s |
| `test/analytical/incns_fv_immersed_objectives.jl` | 149 / 149 | 16.4 s | 149 / 149 | 16.3 s |

Totals: 5865 passing checks before and after; 5964 after including the new
99-check file. No failures, errors, or broken tests occurred. This is focused
validation, not the complete repository suite.

## L3 parameter cost at fixed state

`parameter_sweep_cost.csv` retains all original L2 values in its three rows;
16 appended `l3_` columns record the new same-session comparison. Measured
operations include parameter upload, local derivative work, reduction, and the
returned host vector. Both paths use identical states, cotangents, fields,
parameters, and 64² grids. No Newton solve, Jacobian assembly, factorization,
adjoint solve, or explicit objective derivative is timed. Warm times are medians
of three calls and host bytes are minima; no other validation process ran during
this measurement. Cold and compilation times remain available in the CSV.

| Parameters | Contraction (ms) | Duals (ms) | Contraction host bytes | Dual host bytes | Relative difference |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 3.723041 | 5.655083 | 640 | 4656 | 2.887956189761536e-16 |
| 59 | 16.708417 | 90.659500 | 1024 | 55248 | 3.238959237284326e-16 |
| 248 | 66.662125 | 1455.281042 | 2560 | 704400 | 1.905802552307277e-16 |

At 248 parameters, this part of one gradient decreases from about 1.46 seconds
to 0.067 seconds on the measured 64² CPU case. The result does not establish a
speedup for a complete gradient or simulation.

The explicit objective derivative retains host parameter duals with its existing
chunk size of 8. Its cost was measured separately at 64² with the same
248-parameter learned law, temperature field, and state/adjoint seeds, but with
a periodic channel and an immersed M1 circle centered at `(0.511, 0.489)` with
radius `0.184` and 92 wall faces. The state remains arbitrary and fixed; no
Newton or adjoint solve is performed. Reproduce with:

```sh
julia --startup-file=no --project=. benchmarks/fv_law/objective_explicit_cost.jl
```

Three warm calls of `_fv_objective_grad_theta` give:

| Operation on the immersed case | Median time (ms) | Minimum host bytes |
| --- | ---: | ---: |
| Residual parameter contraction | 60.827917 | 2592 |
| Explicit residual-flux force derivative | 11.851833 | 5853872 |
| Explicit surface-force derivative | 11.849833 | 5169392 |

Each explicit term costs about 19.5% of the residual contraction on this case,
so it is not negligible. The measured cost supports retaining the existing objective path here, while
the allocations and repeated parameter sweeps remain a limitation. The complete
gradient is not independent of parameter-count sweeps, and no larger-grid or
GPU claim follows from this host objective measurement.

## GPU execution on Aqua (job 27746753)

Date: 2026-11-08. Aqua node `gpu1n014`, one NVIDIA H100 80GB HBM3, Julia 1.12.5,
CUDA.jl 6.4.1, CUDSS.jl 1.0.0, Float64, code at commit `2b6dfeadf`. The job
installed its environment on the compute node (`setup_law.sh`, 179 s of
precompilation), then ran the full validation mode of `law_gpu.jl` at 512 × 512
followed by the contraction-only mode. Exit status 0, 8 min 33 s of walltime.
Receipts (user paths and project identifiers removed) are in
[results/aqua_27746753/](results/aqua_27746753/).

| Gate | Result |
| --- | --- |
| `ClassicalLaw` vs `FVGNFViscosity` Newton, 512² | 6/6; residual histories differ by at most 2.7e-19 |
| `LearnedViscosity` (17 parameters, prescribed temperature field) Newton, 512² | converged in 2 iterations, residual 9.8e-5 → 1.3e-7 → 3.1e-13 |
| Complete steady gradient vs central differences, 2 directions × 2 steps | 8/8; relative error ≤ 2.0e-9 |
| Contraction vs parameter duals and GPU vs CPU, 12², three laws, with and without immersed circle | 24/24; max relative difference 2.8e-15 |

Timings at 512² (first call includes compilation): classical first Newton
10.0 s, learned first Newton 10.7 s, learned warm Newton 0.54 s, first gradient
3.8 s, warm gradient 0.16 s. Warm Newton and warm gradient allocate zero device
bytes; host allocations are listed in `law_gpu_timings.csv`.

Parameter contraction with 248 parameters: warm 36.5 ms at 512². At 128² the
first call takes 0.034 s with contraction and 2.96 s with parameter duals (62
residual sweeps); warm calls take 34.3 ms and 36.6 ms. At this size the warm
GPU calls are not limited by the number of sweeps; the 512² dual-path timing
was not run in this job. These figures exclude Newton, factorisation,
transposed solve and objective derivatives.

An earlier set of A100 measurements, taken from an uncommitted tree by a job
that was not authorised, has been discarded and is not reported here.
