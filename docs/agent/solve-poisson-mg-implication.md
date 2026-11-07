---
module: solve-poisson-mg
path: src/solve/poisson_mg.jl
owner_concern: elliptic-solve
status: implemented
last_verified: 2026-06-11
depends_on:
  - solve-linear
---

# solve-poisson-mg — module implication map

Matrix-free geometric multigrid (V-cycle) Poisson/Helmholtz solver, GPU-native
via KernelAbstractions — the **GPU-performance** elliptic path (issue #8; cuDSS
under-uses the GPU at ~9% occupancy, the RBGS-smoothed V-cycle saturates it).
Same source CPU + CUDA; O(N) per V-cycle with an N-independent cycle count.
Operator conventions (Dirichlet GHOST-0 "+1/h²", Neumann mirror "-1/h²") match
`src/solve/poisson.jl` EXACTLY — that parity is test-pinned. **Registered in
`src/KrakeNN.jl`** (`solve_poisson_mg` and `solve_poisson_mgcg` exported).

## Public surface

- `solve_poisson_mg(f, N; bc=:dirichlet|:neumann, backend, backend_ka, atype,
  tol, maxcycles, nu1, nu2, ncoarse, smoother=:rbgs|:jacobi, min_size, sigma,
  u0, fixed_cycles, hier, mixed_precision, hier_f32, verbose)
  -> (u, ncycles, resid_history)` — the driver. Key seams:
  `sigma` (Helmholtz shift `(σI - ∇²)`, used by the cavity momentum predictor;
  σ>0 de-singularizes Neumann), `u0` (warm start), `fixed_cycles` (run exactly
  n V-cycles, ZERO reductions/host syncs, static launch sequence —
  CUDA-graph-capturable; Neumann projections skipped, answer differs by a
  gauge constant), `hier`/`hier_f32` (pre-built hierarchy reuse across repeated
  solves), `mixed_precision` (F64 defect correction wrapping an all-F32 inner
  V-cycle; returned solution stays F64).
- `MGHierarchy` + `build_mg_hierarchy(N, atype; min_size=4)` — pre-allocated
  level stack (u, f, r, scratch per level), allocate ONCE and reuse.
- `vcycle!(hier, level, bc, kab; nu1, nu2, ncoarse, smoother, neumann_pin,
  sigma)` — the recursive V-cycle (used directly by power users/tests).
- `solve_poisson_mgcg(f, N; ...)` — bonus MG-preconditioned CG (Dirichlet/SPD
  only).
- `_mg_eltype_variant(atype, T)` — derive `CuArray{Float32}` from
  `CuArray{Float64}` etc. (de-facto public: cavity_mg uses it for `hier_f32`).
- BC tags `MG_BC_DIRICHLET`/`MG_BC_NEUMANN` (distinct from the FVFD `UInt8`
  consts).

## Reads from

`solve-linear` (`src/solve/linear_solve.jl`, guarded include): ONLY the backend
tags `CPUBackendTag`/`CUDABackendTag` accepted at the API boundary and
translated to a KA backend by `_mg_ka_backend` (the `CUDABackendTag` method
errors by design — a GPU job passes the live `backend_ka=CUDABackend()`
instead). Otherwise KernelAbstractions + `LinearAlgebra.norm` only. RHS `f` may
be a host Function (sampled into a staging Matrix, then `copyto!`) or an `N x N`
array.

## Writes to

Mutates ONLY caller-visible state it owns or was handed: the returned `u` IS
`hier.u[1]` (aliased — copy it before reusing the hierarchy), and a reused
`hier`/`hier_f32` has every level fully re-initialized per call (reuse is
value-identical). No globals, no I/O. Tolerance mode allocates the residual
history; fixed-cycles mode returns an empty history.

## Backend constraints

- Every elementwise op is a `@kernel` launched on `backend_ka`; reductions go
  through device `sum`/`norm`. NO per-launch host synchronize (KA CPU kernels
  are synchronous; CUDA launches are stream-ordered) — host syncs happen only
  where a host scalar is read (`norm`, `sum` in tolerance mode / zero-mean
  projection).
- Grid scalars are converted to the FIELD eltype before launch: identity for
  F64, and keeps ALL kernel arithmetic in F32 for the mixed-precision hierarchy
  (no silent F64 promotion — that would forfeit the F32 bandwidth gain).
- `N` must halve cleanly to `min_size` (power-of-two multiple), or the
  hierarchy bottoms out early on an odd size (coarsest grid then larger —
  correct but slower).
- Measured (Aqua A100, `benchmarks/krk/inc_ns/poisson_mg_gpu_bench.{jl,pbs}`,
  `benchmarks/results/poisson_mg_gpu_aqua_a100.md`): V-cycles flat at 10–13
  from 16k to 16M DOF; MG-GPU ~43x MG-CPU at 16M DOF, peak 99% device util;
  GPU↔CPU parity ‖Δ‖∞ ≤ 1.3e-13. For moderate-N 2D steady, factorize-once
  cuDSS still wins per-solve — different regimes.

## Failure modes

- **Operator-convention drift vs poisson.jl** breaks the CHOLMOD parity gate in
  `test/analytical/poisson_mg_mms.jl` — change conventions in BOTH files or not
  at all (cavity_mg.jl's wall/lid sources also assume GHOST-0).
- **Mixed precision is iterative refinement**: it converges while
  `cond(A)·eps(F32) < 1` (true at N=512, cond ~1e5). Very large N or strong σ
  asymmetries can stall the F64 residual contraction — fall back to F64 cycles.
- **fixed_cycles + :neumann** assumes a discretely zero-mean RHS (true for a
  conservative face divergence) and SKIPS all projections: the answer carries
  an arbitrary additive constant the CALLER must gauge away (cavity_mg does the
  zero-mean gauge on `pcorr`). Feeding a non-zero-mean RHS diverges slowly.
- **Hierarchy aliasing**: the returned `u` aliases `hier.u[1]`; a second solve
  with the same `hier` overwrites the previous answer. `copyto!` out first
  (cavity_mg's phase functions do).
- **`_mg_ka_backend(CUDABackendTag())` errors on purpose** — this file stays
  CUDA-free; pass `backend_ka=CUDABackend()` in GPU jobs. Don't "fix" the error
  by importing CUDA here.
- Jacobi smoother needs the scratch ping-pong buffer and converges slower
  (smoothing factor ~0.6 vs ~0.25 RBGS); `:rbgs` is the validated default.

## Touch order

1. `src/solve/poisson_mg.jl` — kernels (apply/residual/smoothers/transfers),
   the V-cycle, the driver's mode logic (tolerance vs fixed vs mixed).
2. `test/analytical/poisson_mg_mms.jl` — MMS order, flat-cycle hallmark,
   CHOLMOD parity, Neumann+pin; run after ANY edit here.
3. `src/solve/poisson.jl` — the convention reference if parity breaks.
4. `src/methods/inc_ns/cavity_mg.jl` — the main consumer (σ-shifted momentum +
   Neumann pressure; gauge handling) if cavity results drift but MMS passes.
5. `benchmarks/krk/inc_ns/poisson_mg_gpu_bench.jl` +
   `test/scratch/poisson_mg_driver.jl` — perf/parity drivers.
