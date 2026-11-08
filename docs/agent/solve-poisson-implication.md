---
module: solve-poisson
path: src/solve/poisson.jl
owner_concern: elliptic-solve
status: implemented
last_verified: 2026-06-11
depends_on:
  - solve-linear
---

# solve-poisson — module implication map

The assembled CPU **reference** Poisson service of the IncNS solver stack (issue #7):
5-point `-∇²` on the unit square, cell-centred `N x N` grid, sparse assembly +
direct solve. It is the convention anchor — `poisson_mg.jl` (matrix-free GPU path)
and the embedded variants must match its Dirichlet/Neumann boundary conventions
exactly. **Registered in `src/KrakeNN.jl`** — included FIRST in the IncNS solve
block (its tail-include loads `linear_solve.jl`); `solve_poisson_dirichlet`,
`solve_poisson_neumann` and `pin_reference_dof` exported. The `isdefined`
guards keep every `src/solve/` file standalone-include-able.

## Public surface

- `solve_poisson(A, b, N) -> Matrix` — solve an assembled system through the
  factorize-once seam (`lin_factorize`/`lin_solve!`, CPU CHOLMOD, `spd=true`);
  bit-identical to `cholesky(Symmetric(A)) \ b`.
- `solve_poisson_dirichlet(N, f)` / `solve_poisson_neumann(N, f, u_exact; k0=1)`
  — assemble-and-solve drivers (Neumann pins DOF `k0` to the exact value).
- Assembly: `assemble_poisson_dirichlet`, `assemble_poisson_neumann_unpinned`,
  `assemble_poisson_neumann_pinned`, `pin_reference_dof(A, b, k0, value)`.
- Grid/indexing helpers used by every sibling: `cell_center`, `linear_index`
  (`k = i + (j-1)N`), `cell_ij`, `cell_coordinates`.
- MMS utilities: `exact_field`, `l2_error`.

## Reads from

`solve-linear` (`src/solve/linear_solve.jl`): `lin_factorize`, `lin_solve!`,
`CPUBackendTag` — pulled in by an `include` at the END of this file (guarded by
`isdefined(:lin_factorize)`); it must come last because `linear_solve.jl` needs
`pin_reference_dof` from here (deliberate mutual standalone-include pattern).
Otherwise only LinearAlgebra + SparseArrays. No Kraken module, no `.krk` hook.

## Writes to

Nothing persistent: every function returns fresh arrays (`sparse(I,J,V)`, new
`b`, reshaped solution). `pin_reference_dof` does NOT mutate its inputs — it
rebuilds the matrix without row/col `k0` and returns a pinned copy. No globals,
no I/O, no device state.

## Backend constraints

CPU-only by design (host loops over `1:N`, `SparseMatrixCSC`, CHOLMOD). This is
the reference/parity path; GPU routes are `solve_poisson_mg` (matrix-free KA)
and the `CUDABackendTag` seam methods (cuDSS) — both validated against THIS
module's output. Assembly is O(N²) host work each call; do not put it inside an
outer iteration loop (factorize once, reuse the cache).

## Failure modes

- **Singular all-Neumann operator**: `assemble_poisson_neumann_unpinned` has
  zero row sums; `solve_poisson` on it throws (Cholesky `check=true`). Pin first.
  Receipt: `test/analytical/poisson_mms.jl` ("Unpinned Neumann singularity").
- **Boundary-convention drift**: Dirichlet here is GHOST-0, "+1/h² per missing
  face" (ghost value at the ghost CELL CENTRE), NOT the "+2/h²" half-spacing
  form. `poisson_mg.jl` and `cavity_mg.jl` replicate this; changing one side
  silently breaks the MG↔CHOLMOD parity test (`test/analytical/poisson_mg_mms.jl`).
- **Include-order trap**: moving the trailing `include("linear_solve.jl")`
  before `pin_reference_dof` breaks standalone inclusion of `linear_solve.jl`
  (it needs the symbol at load time). The guards make every file in `src/solve/`
  include-able in any order — keep them.
- Single-RHS `solve_poisson` rebuilds the factorization per call (cache built
  then consumed); fine for MMS, wasteful in loops — the SIMPLE solvers hold a
  `LinearSolveCache` instead.

## Touch order

1. `src/solve/poisson.jl` — assembly conventions, indexing helpers, the
   `solve_poisson` seam routing (look here for any operator/BC question).
2. `src/solve/linear_solve.jl` — if the failure is in factorize/solve (cache,
   pinning at solve time, backend tag).
3. `test/analytical/poisson_mms.jl` — MMS gates (2nd order Dirichlet/Neumann,
   singularity assert); first thing to run after any edit.
4. `test/analytical/poisson_mg_mms.jl` — cross-module parity (MG must match
   this module bit-for-bit on the same RHS).
5. `test/scratch/poisson_mms_driver.jl` — quick manual driver.
