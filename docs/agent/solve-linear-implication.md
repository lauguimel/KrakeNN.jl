---
module: solve-linear
path: src/solve/linear_solve.jl; src/solve/linear_solve_cuda.jl; src/solve/linear_solve_frontend.jl; ext/KrakenLinearSolveExt.jl; ext/KrakenCUDSSExt.jl
owner_concern: backend-dispatch
status: implemented
last_verified: 2026-07-22
depends_on:
  - solve-poisson
---

# solve-linear — module implication map

The backend-parametric **factorize-once** linear-solve seam (issues #7/#8). The
pressure-Poisson operator on a fixed grid is geometry-only and constant across
every SIMPLE outer iteration (~3000 of them in the cavity), so it is factorized
ONCE and the factorization reused per RHS. Two functions are the ENTIRE
contract; dispatch is keyed on a backend singleton tag. The CPU file is
CUDA-free; the CUDA methods live in the companion `linear_solve_cuda.jl`,
loaded ONLY inside a job that already did `using CUDA, CUDSS`. **Registered in
`src/Kraken.jl` THROUGH the tail-include in `solve/poisson.jl`** (the file has
NO self-guard — never include it directly in `Kraken.jl`); seam names exported
(`lin_factorize`, `lin_solve!`, `LinearSolveCache`, backend tags).
`linear_solve_cuda.jl` stays manual-load (bench scripts); package users get the
SAME cuDSS seam through the `KrakenCUDSSExt` package extension instead.

Activation requirement (#61): `KrakenCUDSSExt` loads only when `CUDSS` is loaded
in the same session as `Kraken` (`using CUDA, CUDSS, Kraken`, in any order), AND the
`CUDSS` weakdep UUID in `Project.toml` equals the registered one
(`45b445bb-4962-46a0-9369-b4df9d0f772e`). From v0.3.0 until the fix the UUID was
wrong, so the extension could not activate. Check with
`Base.get_extension(Kraken, :KrakenCUDSSExt) !== nothing`; `test/platform/weakdeps_metadata_test.jl`
guards the UUIDs offline and `test/analytical/poisson_cudss_gpu.jl` checks activation on GPU.

Issue #8 adds two [weakdeps] package-extension backends behind the seam
(ADR rule: no solver-backend dependency under plain `using Kraken`; precedent
KrakenOptimExt): `KrakenLinearSolveExt` (trigger `LinearSolve`) backs the new
`PoissonLinearSolve` tag, and `KrakenCUDSSExt` (triggers `CUDSS` + `CUDA`;
CUDA is a strong dep listed as an extra trigger so the ext may `using CUDA`)
backs `CUDABackendTag` for package users. The exts are INDEPENDENT — cuDSS is
reached through Kraken's own seam, NOT through LinearSolve's CUDSS bindings,
so either loads without the other. `src/solve/linear_solve_frontend.jl` holds
the CPU-only tag + ext-indirection stubs (`_ls_factorize`/`_ls_solve!`/
`_cudss_factorize`, the `_fit_lbfgs` stub-error pattern — no method
overwriting) and the assembled-direct driver `solve_poisson_direct`, the
drop-in alternative to `solve_poisson_mg` on the SAME poisson.jl
discretization.

## Public surface

- `lin_factorize(A; backend=CPUBackendTag(), spd=true, pin_k0=0) ->
  LinearSolveCache` — factorize once. `spd=true` Cholesky; `spd=false` LDLᵀ
  with LU fallback; `pin_k0>0` pins reference DOF `k0` (singular all-Neumann /
  periodic operator) at factorize time via `pin_reference_dof`.
- `lin_solve!(cache, b) -> x` — reuse the factors per RHS; NEVER re-factorizes;
  applies the consistent RHS pinning when `pin_k0>0`.
- `LinearSolveCache{B,F,M}` — backend tag, factor object, `A` (as factorized,
  pinned if applicable), `A_unpinned` (for RHS pinning), `pin_k0`, `spd`.
- Tags: `LinearSolveBackend` (abstract), `CPUBackendTag`, `CUDABackendTag`.
  New backend = new tag + the two methods.
- `PoissonLinearSolve(; alg=nothing, kwargs...) <: LinearSolveBackend` —
  seam tag for the LinearSolve.jl route (ext-backed). `alg=nothing` picks
  `CHOLMODFactorization()` (`spd=true`) / `UMFPACKFactorization()`
  (`spd=false`), mirroring the CPU seam; `kwargs` forward to
  `LinearSolve.init`. The ext caches a `LinearSolve.init` problem in
  `LinearSolveCache.factor`; factorization happens at first `lin_solve!` and
  is reused per RHS (same factorize-once contract).
- `solve_poisson_direct(f, N; bc=:dirichlet, method=PoissonLinearSolve(),
  k0=1) -> Matrix{Float64}` — assembled direct alternative to
  `solve_poisson_mg` on the SAME 5-point discretization (poisson.jl
  assemblers). `bc=:neumann` pins DOF `k0` to 0 (zero-anchored gauge —
  zero-mean both fields before MMS comparison). `method` accepts any
  `LinearSolveBackend`: `PoissonLinearSolve()` (needs `using LinearSolve`),
  `CPUBackendTag()` (built-in CHOLMOD), `CUDABackendTag()` (needs
  `using CUDA, CUDSS`; result gathered back to host).
- `KrakenCUDSSExt` additions: host-matrix `lin_factorize(A_csc;
  backend=CUDABackendTag(), pin_k0)` (pins on CPU, uploads BOTH matrices,
  cuDSS-factorizes) and a `CuSparseMatrixCSR` generic keyword entry — plus the
  same `CUDABackendTag` seam methods as the manual-load file (which attaches
  to the INCLUDING module, so no method clash).
- CUDA companion (`linear_solve_cuda.jl`): the `CUDABackendTag` methods for
  `CuSparseMatrixCSR{Float64,Int32}` via CUDSS.jl (`cholesky/ldlt/lu(A)` once,
  `ldiv!` per RHS). The CUDA `lin_factorize` takes the ALREADY-PINNED matrix
  plus an `A_unpinned` keyword (pinning is a CPU CSC routine); `lin_solve!`
  returns a `CuVector{Float64}` and pins the RHS on device
  (`@allowscalar bdev[k0] = 0`).

## Reads from

`solve-poisson` (`src/solve/poisson.jl`): `pin_reference_dof` — included under
an `isdefined` guard (mutual standalone-include with poisson.jl; poisson.jl
includes THIS file at its end). CPU file: LinearAlgebra + SparseArrays only
(CHOLMOD/UMFPACK). CUDA file additionally: CUDA, CUDA.CUSPARSE, CUDSS — which is
why it must never be loaded in the main project (CUDSS is not in Project.toml;
it lives in the Aqua GPU job's environment).

## Writes to

No globals, no I/O. `lin_factorize` allocates the factorization (CHOLMOD
workspace on CPU; cuDSS device factors on GPU — ~15 GB resident at 16M DOF, see
the MG results note) and returns a fresh cache. `lin_solve!` allocates the
result vector per call; despite the `!` it does NOT mutate `b` or the cache
(the name reserves the in-place contract).

## Backend constraints

- The seam IS the backend boundary: call sites never branch on device — they
  hold a cache and call `lin_solve!`. A GPU run swaps the tag + matrix type
  with NO call-site change (`simple.jl` documents exactly this drop-in).
- CPU file must stay include-able on a CPU-only box: NO `using CUDA` here,
  ever. The reverse include guard (`isdefined(:CUDABackendTag)`) in the CUDA
  file exists so it can be included standalone in a bench script.
- Per-solve cost is the back-substitution only (amortized 4.7 ms at 1M DOF on
  A100); the cuDSS factorize-once pressure solve measured ~30x CPU CHOLMOD at
  1M DOF (`benchmarks/krk/inc_ns/poisson_gpu_bench.{jl,pbs}`). cuDSS
  under-utilizes the GPU (~9% occupancy, sequential triangular solves) — the
  throughput path is `solve-poisson-mg`, a different regime, not a replacement.

## Failure modes

- **Re-factorizing in the loop** is the exact waste this module exists to
  prevent — if profiles show `cholesky` per outer iteration, a call site is
  bypassing the cache (receipt: design notes in the file header; bench
  `test/scratch/linear_solve_cache_driver.jl`).
- **Pinning lives in the cache, not the call site**: pin at factorize time
  (`pin_k0`), never hand-pin the RHS at call sites — `lin_solve!` does it
  consistently from `A_unpinned`. Double-pinning gives a wrong (still
  plausible-looking) pressure field.
- **CUDA pinning asymmetry**: on GPU the caller pins on CPU and uploads BOTH
  matrices (`A` pinned, `A_unpinned` kwarg). Passing the unpinned matrix as `A`
  makes cuDSS factorize a singular operator (garbage or failure).
- **`spd=true` on a singular operator throws** (CHOLMOD `check=true`); the
  all-Neumann pressure operator needs `pin_k0>0` first.
- The CUDA file at include time asserts nothing about CUDSS versions; the cache
  `factor` field is `Any`-typed via the type parameter, so an API rename in
  CUDSS surfaces at `ldiv!`, not at include.
- **Ext not loaded**: `PoissonLinearSolve` seam calls error with a
  `using LinearSolve` hint; `lin_factorize(A_csc; backend=CUDABackendTag())`
  errors with a `using CUDA, CUDSS` hint (stub fallbacks in
  `linear_solve_frontend.jl`). A `MethodError` instead of these hints means a
  call bypassed the documented entries.
- **Weakdeps gotcha**: after adding/altering [weakdeps], run `Pkg.resolve()`
  before `Pkg.precompile()` (same as the KrakenOptimExt rollout) or the ext
  will silently not load.

## Touch order

1. `src/solve/linear_solve.jl` — the contract, tags, CPU methods, pinning
   logic (start here for any wrong-solution or perf regression).
2. `src/solve/linear_solve_cuda.jl` — GPU-only failures (matrix type, device
   pinning, CUDSS API).
3. `src/solve/poisson.jl` — `pin_reference_dof` semantics if pinned solves
   drift.
4. Consumers if the seam is fine: `src/methods/inc_ns/simple.jl`
   (`_incns_factorise`/`_incns_solve!`), `benchmarks/krk/inc_ns/poisson_gpu_bench.jl`.
5. `src/solve/linear_solve_frontend.jl` — tag/stub/driver layer
   (`PoissonLinearSolve`, `solve_poisson_direct`, ext indirection).
6. `ext/KrakenLinearSolveExt.jl` / `ext/KrakenCUDSSExt.jl` — ext-backend
   failures (LinearSolve API drift, cuDSS upload/pinning). Receipts:
   `test/analytical/poisson_linearsolve_mms.jl` (CPU MMS + parity vs CHOLMOD
   and MG + cache reuse), `test/analytical/poisson_cudss_gpu.jl` (GPU parity,
   gated on `CUDA.functional()`).
7. `test/scratch/linear_solve_cache_driver.jl` — cache-reuse parity driver.

## GPU validation + Julia 1.12 packaging landmine (2026-07-22)

- KrakenCUDSSExt validated on RTX A6000 (driver 535, CUDA 12.2 pin): Poisson MMS parity
  Linf 3.9e-14 (Dirichlet) / 1.6e-12 (Neumann pinned), 3/3 tests.
  **Correction (2026-10, #61):** the loading path of this run was not recorded. With the
  wrong CUDSS UUID declared until 0.5.1, plain `using CUDSS` could not activate the
  extension, so these numbers may come from a manual include. Extension-path evidence:
  Aqua job 26559033 (reported 2026-10-10), H100, 41/41 GPU assertions, no CPU fallback
  (see the 0.5.1 changelog entry).
- **Landmine (Julia 1.12.6, empirically bisected)**: KrakenCUDSSExt fails to precompile with
  "Package CUDSS … does not seem to be installed" whenever a SIBLING declared weakdep
  (Enzyme/CairoMakie/Optim) is NOT installed in the environment. KrakenLinearSolveExt is immune.
  Workaround: co-install the sibling weakdeps (installing Enzyme fixed it); do NOT list CUDA
  (strong dep) in [weakdeps] — extensions may use parent strong deps directly since Julia 1.11.
