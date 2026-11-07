# FV scalar-law integration: local validation receipt

Date: 2026-11-07. Machine: Apple M3, macOS arm64, Julia 1.12.5.
Backend: KernelAbstractions CPU. Numerical integration checks: Float64;
scalar inference checks also exercise Float32 and mixed numeric metadata.
This receipt covers focused tests and the CPU parameter-cost measurement,
not the full repository suite or GPU numerical execution.

## Unchanged-test comparison

The six existing files below were run unchanged, in the same order, each
inside its own module and top-level testset. The before process loaded Kraken
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

The final independent check used a fresh Julia process, loaded Kraken, then
timed the complete file inclusion. Total elapsed was **27.751042417 s,
compilation included** (`27.334223335 s` of compilation). A separate fresh
process measured `28.511797834 s`, also including compilation. Both complete
inclusions satisfy the 30-second gate; no compilation time was subtracted.
Package startup and the initial `using Kraken` precede the timer. The timing
refinement removed duplicate free-law force checks and redundant kernel
reflection; no numerical threshold or required objective check was weakened.

```sh
julia --startup-file=no --project=. test/analytical/incns_fv_law_test.jl
julia --startup-file=no --project=. -e 'using Kraken; t=@timed include("test/analytical/incns_fv_law_test.jl"); println((elapsed_seconds=t.time,compile_seconds=t.compile_time));'
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
