---
module: incns-fv-newton
path: src/methods/inc_ns/fv_newton.jl
owner_concern: steady-newton
status: implemented
last_verified: 2026-11-04
depends_on:
  - incns-fv-steady-residual
  - solve-linear
  - incns-platform
  - platform
---

# incns-fv-newton — module implication map

Step C2 of the FV Newton roadmap: the exact sparse Jacobian `∂R/∂U` of the C1
steady FV residual, assembled by coloured forward-mode duals through the C1
KernelAbstractions kernel itself (no hand-written derivative, no CPU mirror),
and a Newton solver whose linear solves go through the linear-solve seam
(`lin_factorize` once, then `lin_refactorize!` + in-place `lin_solve!`). C3
(GNF viscosity) widens the pattern; C4 (discrete adjoint) reuses the finalized
factors for `Jᵀ λ = g`.

## Public surface

- Set-up (CPU, once per geometry): `fv_jacobian_pattern(prob) ->
  SparseMatrixCSC{Bool}` built from the stencil (not traced: 44 s at 1024² in
  J0), Newtonian only; `fv_jacobian_coloring(S) -> (colors, ncolors)`
  (SparseMatrixColorings, greedy LargestFirst column colouring);
  `fv_jacobian_assembler(S, colors, ncolors; to, order = :csc | :csr, chunk = 4)
  -> FVJacobianAssembler`.
- `fv_assemble_jacobian!(nzval, asm, U, prob; sync = true)` — values in the
  storage order of the target matrix (`nonzeros(J)` CSC on CPU, `J.nzVal` of
  the cuDSS CSR matrix); `ceil(ncolors/chunk)` passes of seed → C1 residual on
  `Dual{FVJacobianTag,Float64,chunk}` → decompress.
- `fv_newton_solver(prob; to = Array, linsolve = CPUBackendTag(), chunk = 4)
  -> FVNewtonSolver` (fields `prob, asm, J, nzval, linsolve, cache, R, dU, Ut,
  norms, n_factorize, n_refactorize, t_pattern, t_coloring`).
- `fv_jacobian!(s, U; shift = 0)` (assemble + optional momentum-diagonal shift);
  `fv_newton!(s, U; atol, rtol, maxit, linesearch, armijo, alpha_min, dtau0,
  dtau_max, verbose) -> NamedTuple` (history, split timings, counters);
  `fv_newton_finalize!(s, U)` (unshifted J at U, refactorized: the C4 entry,
  `lin_solve!(λ, s.cache, g; transpose = true)`).
- Driver `solve_incns_fv_newton(; problem, U0, to, linsolve, chunk, newton...)`;
  platform `IncNS(:fv_newton)`, the only IncNS driver declaring
  `SteadyResidual` (decision 2026-11-04: capabilities are driver-dependent).

## Reads from

- `fv_residual!`, `FVSteadyProblem2D`, `fv_adapt`, `_fv_face_kind` access
  paths (`fv_steady_residual.jl`): the pattern mirrors the kernel's reads.
- `lin_factorize`, `lin_refactorize!`, `lin_solve!` (`src/solve/linear*`,
  CUDSS extension). Device CSR matrix through the hook
  `_fv_device_csr(tag, rowptr, colval, n)`, implemented in
  `ext/KrakenCUDSSExt.jl` for `CUDABackendTag`.
- ForwardDiff (duals), SparseMatrixColorings (colouring), KernelAbstractions.

## Writes to

The solver's own buffers (`nzval`, `R`, `dU`, `Ut`, norm partials) and `U` in
place; `s.cache` (created on the first factorization, then refactorized).
No globals.

## Backend constraints

- Same generic KA kernels on CPU and GPU; CPU assembly equals the dense
  Jacobian bit for bit and the GPU one equals the CPU one to 1e-12.
- GPU hot path (assembly, refactorize, solve, residual, norms, whole Newton
  iterations once the cache exists) allocates 0 device bytes; norms use
  preallocated device partials + host buffer (no `mapreduce` allocation).
- Pattern rows: ≤ 23 entries (u/v rows 23, continuity 15 in the interior);
  colours ≤ 45 (43 on a doubly periodic grid). Chunk 4 per J0.
- Only the first factorization runs the symbolic analysis (UMFPACK pointer and
  cuDSS solver reused); `n_factorize == 1` is tested.

## Failure modes

- **GNF (C3)**: the pattern builder assumes ν independent of U. A
  shear-dependent ν widens the support (J0: 35/row); a pattern too narrow
  silently drops entries (assembly writes only listed positions). Rebuild the
  pattern rule before C3.
- **Doubly periodic**: uniform (u, v) is a null mode; Newton needs a
  mean-velocity constraint (not provided).
- **PTC shift**: with `dtau0` set the shift is dropped only once
  `Δτ ≥ dtau_max` (default 1e8); `s.cache` then holds shifted factors unless
  `fv_newton_finalize!` is called.
- **Pressure pin row** (`p_P = 0`, coefficient 1): no effect on UMFPACK
  solutions (scale 1e-6…1e6 give identical δ, `benchmarks/fv_c2/results/`);
  cuDSS checked in the A100 job.
- **Wall pressure order** (inherited from C1, kept by decision 2026-11-04):
  first order at walls with variable ν; revisit before C4.

## Touch order

1. `src/methods/inc_ns/fv_newton.jl` — pattern rule (must follow any C1 stencil
   change), assembler, Newton.
2. `ext/KrakenCUDSSExt.jl` — `_fv_device_csr`.
3. `src/methods/inc_ns/method.jl` — `:fv_newton` driver, capabilities.
4. `test/analytical/incns_fv_newton.jl` (+ `fv_newton_helpers.jl`) — pattern
   superset/tightness, dense / C1-helper parity, quadratic convergence,
   SIMPLE and Ghia agreement, analysis reuse.
5. `test/analytical/incns_fv_newton_gpu.jl` — GPU parity, cuDSS Newton,
   allocations (`KRAKEN_REQUIRE_GPU=1`).
6. `benchmarks/fv_c2/` — A100 receipts.
