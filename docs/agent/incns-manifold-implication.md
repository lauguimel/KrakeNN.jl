---
module: incns-manifold
path: src/methods/inc_ns/manifold_flow.jl
owner_concern: localized-inlet-outlet-simple
status: implemented
last_verified: 2026-06-12
depends_on:
  - solve-poisson
  - solve-linear
  - fvfd-operators-2d-grad-div-laplacian
  - scalar-transport
---

# incns-manifold — module implication map

The second black-box staging rung for steady incompressible SIMPLE: localized
west/east velocity inlets, localized pressure-reference outlets, and full-cell
immersed solid plates. It keeps the cavity convention for collocated fields and
returns face velocities for scalar-transport handoff. **Registered in
`src/KrakeNN.jl`** and exported (`solve_incns_manifold`,
`manifold_full_cell_mask`); also reachable as
`solve(params, IncNS(:manifold))` (`src/methods/inc_ns/method.jl`).

## Public surface

- `solve_incns_manifold(; nx, ny, Lx, Ly, Re, U_in, is_solid=nothing, inlet,
  outlet, mu=nothing, relax=(u=0.7,p=0.3), scheme=:simplec, tol=1e-7,
  maxiter=4000, backend=CPU(), verbose=false,
  momentum_advection=:linear_upwind) -> NamedTuple`. `scheme=:simple` keeps the
  legacy pressure-correction coefficient path; `scheme=:simplec` uses the
  SIMPLE-consistent correction denominator for the projection path.
  `momentum_advection=:linear_upwind` is the default implicit-upwind plus
  deferred-correction momentum path; `:upwind` preserves the legacy donor-cell
  path.
  `inlet` is
  `(; side::Symbol, j0::Int, j1::Int, u::Float64)`; `outlet` is
  `(; side::Symbol, j0::Int, j1::Int)`. Current localized spans are west/east.
- Returns `u, v, p, uf, vf, is_solid, dx, dy, nx, ny, xcenters, ycenters,
  residual_history, iters, converged, vel_change, mass_imbalance, dp, Re, mu,
  U_in, Lx, Ly, checkerboard, scheme, momentum_advection`.
- `uf[i,j]` is the east face of cell `(i,j)` and `vf[i,j]` is the north face,
  matching `cavity.jl` and the `solve_scalar_transport` consumer contract.
- `manifold_full_cell_mask(nx, ny, Lx, Ly, plates)` builds axis-aligned
  full-cell plate masks from grid-line-aligned coordinate ranges.

## Reads from

- `src/solve/poisson.jl`: `pin_reference_dof`, indirectly through the linear
  seam include guard.
- `src/solve/linear_solve.jl`: `lin_factorize` / `lin_solve!` /
  `CPUBackendTag`. The legacy `:upwind` momentum and pressure systems use
  cached `spd=true` Cholesky. The `:linear_upwind` momentum matrices are
  non-symmetric and use per-outer-iteration `spd=false` UMFPACK LU; pressure
  remains `spd=true` with `pin_k0=0`.
- `src/fvfd/operators_2d_grad_div_laplacian.jl`: guarded include for the
  existing FVFD operator context/constants. The manifold solver mirrors the
  regular full-cell `is_solid` semantics but assembles its own SIMPLE matrices.
- `src/methods/scalar_transport/thermal_transport.jl` consumes the returned
  `uf`, `vf`, `is_solid`, `dx`, `dy`, `nx`, and `ny` in the smoke handoff.

## Writes to

Nothing global. The solver allocates fields and factorization caches inside the
call, mutates them across SIMPLE iterations, and returns a NamedTuple. Scratch
driver output is printed by `test/scratch/incns_manifold_driver.jl`; no files
are written.

## Backend constraints

CPU Float64 path only. Sparse systems are assembled as `SparseMatrixCSC`.
`momentum_advection=:upwind` keeps the factorize-once CHOLMOD path.
`momentum_advection=:linear_upwind` rebuilds the convection-diffusion momentum
matrices and pressure-correction operator each SIMPLE outer iteration because
the face fluxes and SIMPLEC coefficients are lagged from the previous iterate.
`backend=CPU()` is kept for API consistency with the existing standalone
bricks; GPU lowering is deferred to a future seam-compatible implementation.

## Failure modes

- Outlet Dirichlet is the pressure reference. Do **not** additionally pin the
  pressure-correction matrix; `pin_k0` must stay `0`.
- Inlet faces must stay homogeneous Neumann on the pressure-correction side and
  must be skipped by face correction. Correcting the inlet changes the imposed
  inlet flux and breaks the global mass balance.
- Full-cell geometry is an assumption, not a cut-cell approximation. Plate
  coordinates must lie on grid lines; a mid-cell plate edge silently loses the
  partial face physics and must be rejected.
- Fluid-solid pressure faces are Neumann drops; fluid-solid momentum faces are
  no-slip Dirichlet `+2μ/h²` contributions. Reusing the pressure stencil for
  momentum would create slip along plates.
- `momentum_advection=:linear_upwind` uses first-order donor-cell convection
  implicitly in the momentum matrix and adds the lagged
  `(linear_upwind - donor_cell)` correction to the RHS. The deferred term is
  evaluated on the previous iterate's velocities and face fluxes.
- `momentum_advection=:upwind` is the bit-exact legacy donor-cell mode and is
  covered by solution-quantity fingerprints in
  `test/analytical/incns_manifold.jl`. Do not pin residual-like quantities.
- `scheme=:simplec` changes the pressure-correction and velocity-correction
  response coefficients. In the `:linear_upwind` path, the SIMPLE/SIMPLEC
  `d` coefficients use the convection-augmented momentum diagonals, and the
  pressure-correction operator is refactorized after those coefficients update.
  If a lagged convection field makes a local SIMPLEC denominator nonpositive,
  that cell falls back to the positive SIMPLE coefficient for the current
  iteration.
- West-boundary inlet flux is part of the internal projection but is not stored
  in `uf` because the cavity face layout only stores east faces. Consumers that
  need a west boundary advective flux must impose it through their own boundary
  condition.

## Touch order

1. `src/methods/inc_ns/manifold_flow.jl` — boundary masks, matrix assembly,
   Rhie-Chow faces, momentum advection, projection, diagnostics.
2. `test/analytical/incns_manifold.jl` — Poiseuille profile/Δp/order, plate
   sanity, scalar-transport handoff.
3. `test/analytical/incns_momentum_advection_order.jl` — isolated
   donor-cell vs linear-upwind momentum-convection order receipt.
4. `test/scratch/incns_manifold_driver.jl` — battery-manifold OpenFOAM
   comparison and per-gap split printout.
5. `src/methods/inc_ns/cavity.jl` — sibling convention reference only; do not
   edit for this rung.
6. `src/solve/linear_solve.jl` — solve seam reference only; both manifold
   systems should remain SPD.
