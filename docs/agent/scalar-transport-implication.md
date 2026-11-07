---
module: scalar-transport
path: src/methods/scalar_transport/thermal_transport.jl
owner_concern: method
status: implemented
last_verified: 2026-06-11
depends_on:
  - fvfd
  - linear-solve
  - inc_ns
---

# scalar-transport — module implication map

The scalar-transport module is the **single owner** of the DECOUPLED STEADY
scalar advection–diffusion ("thermal transport") brick: given a FROZEN
face-normal velocity field `(uf, vf)`, it solves `∇·(u T) − DT ∇²T = 0` to
steady state. The assembled sparse CSC matrix is ALWAYS the stable first-order
UPWIND advection operator plus the 5-point diffusion stencil. The default
`advection=:linear_upwind` applies second-order advection by Picard deferred
correction on the RHS only,
`A*T(k+1) = b + F_low(T(k)) - F_high(T(k))`, reusing the SAME factorization.
`advection=:upwind` preserves the legacy single-solve path. Validated by
`test/analytical/scalar_transport_heated_channel.jl` and
`test/analytical/scalar_transport_advection_order.jl`: pure conduction recovered
to machine precision (l2_rel ≈ 5e-14), second-order diffusion convergence
(order ≈ 2.00), constant-flux parallel-plate developed Nusselt number
`Nu = 8.2357` (analytic 8.235, rel error 8.5e-5), and advection order ≈1.0 for
`:upwind` vs ≈2.0 for `:linear_upwind` at Pe_cell > 5.

## Public surface

The brick is registered in `src/KrakeNN.jl` and its driver exported
(`solve_scalar_transport`), mirroring the `inc_ns/simple.jl` and
`inc_ns/cavity.jl` pattern (include guards keep it standalone-include-able):

- `solve_scalar_transport(; nx, ny, dx, dy, uf, vf, DT, is_solid=falses(nx,ny), bc, source=nothing, backend=nothing, advection=:linear_upwind, deferred_passes=4, deferred_tol=1e-8) -> NamedTuple` — the single entry point. Returns `(; T, residual_history, iters, converged, dx, dy, ycenters, nx, ny, DT, Pe_cell, advection, deferred_passes_used, deferred_converged, deferred_rel_change)`. `T` is the steady `nx×ny` field; `residual_history` stores direct linear residuals for each RHS solve; `iters == 1` for `:upwind` and `1 + deferred_passes_used` for `:linear_upwind`; `Pe_cell = max|u|·min(dx,dy)/DT`.
- `bc` is a NamedTuple keyed `west/east/south/north`, each `(kind::Symbol, value)` with `kind ∈ {:dirichlet, :flux, :outflow}`. `:dirichlet` imposes `T = value`; `:flux` imposes a Neumann wall heat flux `q = value` injected as a SOURCE in `b` (not in `A`); `:outflow` is zero-gradient (ghost = interior, advection carries the interior upwind value).
- `source` (optional) is a cell-centred volumetric source field `S[i,j]`; the balance becomes `div(uT) − DT·lap(T) = S` (used by the manufactured second-order convergence test). `nothing` = no source.

Internal (file-private) assembly helper: `_st_assemble_system(nx, ny, dx, dy, uf, vf, DT, is_solid; bc, source) -> (A::SparseMatrixCSC, b::Vector)`. Deferred correction helpers compute a conservative interior-face vector only; domain boundaries and fluid-solid faces fall back to first-order, so the global energy balance remains controlled by the matrix residual.

## Reads from

- `fvfd` (collocated cell-centred FVFD geometry) — the brick mirrors the
  `inc_ns/simple.jl` sparse assembly directly: the linear index `k = i + (j−1)*nx`,
  the 5-point Laplacian coefficients `±DT/h²`, and the `uf[i,j]=east face` /
  `vf[i,j]=north face` velocity layout. It reimplements NONE of the matrix-free KA
  operators — it transcribes their stencil into CSC coefficients.
- `inc_ns` — the UPWIND advection logic is transcribed from `_cavity_convection!`
  (`src/methods/inc_ns/cavity.jl`): first-order donor/acceptor on the frozen face
  fluxes `Fe=uf[i,j]`, `Fw=uf[i-1,j]`, `Fn=vf[i,j]`, `Fs=vf[i,j-1]`, turned into
  matrix coefficients (donor → `+F/h` on the diagonal, acceptor → `−F/h`
  off-diagonal). The Dirichlet wall-value source convention (`+2·DT/h²·T_wall` to
  `b`) mirrors `_cavity_dirichlet_rhs!`.
- `linear-solve` — the factorize-once seam (`src/solve/linear_solve.jl`):
  `lin_factorize(A; backend, spd=false)` + `lin_solve!(cache, b)`. The brick
  factorizes the first-order matrix once and reuses the cache for each deferred
  correction RHS.

## Writes to

- **Returns a fresh `T::Matrix{Float64}` (nx×ny)** and the NamedTuple above; it
  mutates none of its inputs (`uf`, `vf`, `is_solid`, `bc`, `source` are read-only).
- **Allocates per call**: the sparse triplet vectors `I/J/V`, the assembled
  `A::SparseMatrixCSC`, the RHS `b`, the LU factorization, the solution vector,
  and, for `:linear_upwind`, one correction/RHS vector pair reused across Picard
  passes. The matrix and factorization are not rebuilt during deferred
  correction.
- No files written, no global registry mutated, no `using KrakeNN` side effects.

## Backend constraints

- **CPU-first, KA + stdlib only.** Default `backend=nothing` routes to
  `CPUBackendTag()`. The solve goes through the seam's
  `lin_factorize(A; backend, spd=false)` + `lin_solve!`; since the upwind
  advection makes `A` NON-symmetric, the seam's `issymmetric(A)` gate selects a
  sparse LU (UMFPACK) on CPU.
- A non-CPU `backend` tag is forwarded to the same `lin_factorize` call; the GPU
  sparse-LU path (cuDSS) is the seam's responsibility, untested here.
- The validated path uses the REGULAR (non-embedded) stencil with
  `is_solid=falses`; cut-cell support is a signature placeholder only.

## Failure modes

- **Seam `spd=false` must keep its non-symmetry gate.** The CPU `spd=false`
  branch HISTORICALLY did `try ldlt(Symmetric(A)) catch lu(A)`, which on a
  genuinely non-symmetric `A` silently factorized the SYMMETRIZED matrix and
  returned garbage (residual ~1e2 vs ~1e-13). FIXED in 7dd95b03e: the branch now
  gates on `issymmetric(A)` and uses LU for truly non-symmetric operators, so the
  brick routes through `lin_factorize(A; spd=false)` directly. If that gate ever
  regresses, the Nusselt rung breaks (Nu reads ~12 instead of 8.24) and
  `converged` flips false.
- **Outlet contamination of the developed Nu.** The `:outflow` zero-gradient BC
  biases the last ~5% of streamwise cells (Nu drifts to ~8.7); the test averages
  `Nu` over 50%–95% of the channel to stay in the developed region. Averaging
  including the very last cells inflates the reported Nu.
- **Deferred correction is interior-face only.** `:linear_upwind` uses an
  unlimited upwind-side one-sided gradient in the donor cell and falls back to
  first order next to domain boundaries and solids. This matches the current
  orthogonal-grid boundary machinery and keeps internal corrections conservative,
  but sharp voxel staircases can still set an accuracy floor.
- **First-order upwind numerical diffusion.** `advection=:upwind` still adds
  artificial diffusion at larger cell Péclet and is now mainly a legacy/parity
  path. The default `:linear_upwind` removes the leading interior advection error
  while retaining the same M-matrix solve.
- **Wall-flux sign convention.** `:flux value=q` injects `+q/h` into `b` on the
  wall row (conductive flux INTO the domain). A sign flip silently inverts the
  near-wall gradient; the conduction rung (machine-precision linear profile) is the
  guard.
- **`source` units.** `source[i,j]` is the RHS of `div(uT) − DT·lap(T) = S`, NOT a
  temperature; passing a temperature field corrupts the manufactured-solution
  convergence order.

## Touch order

For a scalar-transport bug (wrong steady field, non-machine residual, wrong Nu,
bad convergence order), inspect in this order:

1. `src/methods/scalar_transport/thermal_transport.jl` — `_st_assemble_system`:
   the per-face diffusion `±DT/h²` and upwind advection `±F/h` coefficients, the
   per-wall BC branches (`:dirichlet` diagonal `+2DT/h²` + `b` source, `:flux`
   `b += q/h`, `:outflow` interior-upwind diagonal). 90% of wrong-field and
   wrong-Nu bugs are a face coefficient or a BC sign here.
2. `_st_advection_deferred_correction!` — if only `:linear_upwind` regresses,
   check the conservative face pair signs, zero correction across solids/domain
   boundaries, and the near-wall fallback before touching the matrix.
3. `solve_scalar_transport` — the seam call
   (`lin_factorize(A; backend, spd=false)` + `lin_solve!`). If the residual is
   ~1e2 instead of ~1e-13, the seam's `spd=false` non-symmetry gate has
   regressed; confirm the cache holds an LU factor for this non-symmetric `A`.
4. `src/solve/linear_solve.jl` — the seam contract (`lin_factorize`,
   `lin_solve!`, `CPUBackendTag`). Check the `issymmetric(A)` gate in the CPU
   `spd=false` branch (LDLᵀ only for genuinely symmetric operators, LU
   otherwise).
5. `src/methods/inc_ns/cavity.jl` / `simple.jl` — the SOURCE patterns this brick
   mirrors (upwind donor/acceptor, Dirichlet wall-value source, 5-point
   Laplacian assembly). Cross-check here when porting a new BC kind or a
   cut-cell stencil.
6. `test/analytical/scalar_transport_heated_channel.jl` and
   `test/analytical/scalar_transport_advection_order.jl` — validation rungs for
   conduction, diffusion order, Nu, segment energy, solid exclusion, legacy
   upwind parity, and second-order advection.
