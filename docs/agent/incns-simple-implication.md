---
module: incns-simple
path: src/methods/inc_ns/simple.jl
owner_concern: pressure-velocity-coupling
status: implemented
last_verified: 2026-06-12
depends_on:
  - solve-poisson
  - solve-linear
---

# incns-simple — module implication map

The FIRST rung of the steady SIMPLE incompressible solver (issue #7): collocated
cell-centred grid, body-force-driven periodic plane Poiseuille. Assembled sparse
operators + the factorize-once CHOLMOD seam; host loops (NOT KA kernels) for the
per-iteration physics. It establishes the SIMPLE structure (direct momentum
predictor -> Rhie-Chow faces -> pressure correction -> compact-gradient
correction) that `incns-cavity-mg` re-implements backend-parametrically.
**Registered in `src/KrakeNN.jl`** and exported (`solve_incns_simple`); also
reachable through the platform contract as `solve(params, IncNS(:simple))`
(`src/methods/inc_ns/method.jl`). Include guards keep the file
standalone-include-able.

## Public surface

- `solve_incns_simple(; nx, ny, H, mu, G, relax=(u=0.7,p=0.3),
  scheme=:simplec, momentum_advection=:linear_upwind, tol=1e-10, maxiter=200,
  Lx=H, backend=CPU()) -> NamedTuple` — the only entry point. `scheme=:simple`
  keeps the legacy pressure-correction coefficient path; `scheme=:simplec` uses
  the SIMPLE-consistent correction denominator for the pressure-correction path.
  `momentum_advection` is accepted and validated for API parity with the
  manifold/cavity steady solvers, but this fully-developed body-force Poiseuille
  rung has no nonlinear momentum-convection term.
  Returns `(u, v, p, residual_history, iters, converged, vel_change, dx, dy,
  ycenters, H, mu, G, Lx, nx, ny, scheme, momentum_advection)`. `backend` is
  currently cosmetic (host loops; the KA path is cavity_mg).
- De-facto public internals tests poke: `_incns_assemble_neg_laplacian(nx, ny,
  dx, dy; bc_x, bc_y)` (`:periodic`/`:dirichlet0` "+2/h²" half-spacing ghost /
  `:neumann`), `_incns_rhie_chow_faces!`, `_incns_face_divergence!`,
  `_incns_compact_gradient!` (the discrete transpose of the face divergence),
  `_incns_factorise`/`_incns_solve!` (thin wrappers over the seam).

## Reads from

- `solve-poisson` (`src/solve/poisson.jl`): `pin_reference_dof` (via the seam)
  — guarded include.
- `solve-linear` (`src/solve/linear_solve.jl`): `lin_factorize`/`lin_solve!`/
  `CPUBackendTag` — the momentum operator (SPD, factorized once) and the pinned
  pressure operator (`pin_k0=1`) each hold ONE `LinearSolveCache` across all
  outer iterations.
- `src/fvfd/operators_2d_grad_div_laplacian.jl` (guarded include): the
  `FVFD_BC_*` consts (and the gdl operators are available, though the hot loop
  uses its own compact host stencils). No fvfd implication map exists yet —
  the operators are covered by their docstrings.

## Writes to

Nothing global; allocates its fields once and mutates them across the outer
loop, returning them in the NamedTuple. No I/O, no device state. The two
factorization caches live for the duration of the call.

## Backend constraints

CPU-only in practice (host `@inbounds` loops over fields, CHOLMOD factorizations,
`Vector` RHS). This is deliberate: rung-1 favours exactness (direct solves) over
throughput. GPU = `incns-cavity-mg` (same SIMPLE skeleton, KA kernels + MG).
Cost profile: two momentum back-substitutions + at most one pressure
back-substitution per outer iteration — factorize-once makes the loop cheap.

## Failure modes

- **Continuity alone is NOT convergence**: any divergence-free `u(y)` —
  including an under-scaled one — satisfies it. The solver requires BOTH
  `res < tol` AND `vel_change < tol`. Removing the velocity-settle gate
  re-opens the under-scaled-profile trap (documented in the loop body).
- **Divergence floor**: when `div(u*)` is at machine noise, inverting it only
  injects a spurious pressure mode (collocated SIMPLE is sensitive); the
  pressure correction is skipped below `1e-10·ref`. Removing the gate breaks
  late-stage convergence on this case.
- **Transpose consistency**: the SIMPLE velocity correction must use the SAME
  compact gradient (`_incns_compact_gradient!`) whose transpose is the face
  divergence — using the wide operator gradient de-couples `div(d·grad p')`
  from the assembled pressure Laplacian and the projection stops being
  idempotent.
- **Rhie-Chow is under-stressed here** (fully-developed Poiseuille has uniform
  pressure, so the correction ~0): a checkerboard bug can pass this case and
  only surface in the cavity. Validate cavity too after touching face
  interpolation.
- **Momentum is solved DIRECTLY, not point-relaxed** (`αu` only damps the
  pressure coupling via `d = αu/a_p`): "adding" momentum under-relaxation
  iterations here converges at Jacobi rate — that's a regression, not a fix.
- `scheme=:simplec` changes the pressure-correction response coefficient only;
  the Rhie-Chow face model stays on the legacy coefficient so the converged
  finite-grid face model is not changed by the acceleration path.
- `momentum_advection=:upwind` and `:linear_upwind` are numerically identical in
  this rung because nonlinear convection is absent. Keep the kwarg so platform
  callers can pass the same scheme selector to `IncNS(:simple)` and
  `IncNS(:manifold)`; do not add a fake convection term to this analytic case.
- Receipt: `test/analytical/incns_poiseuille.jl` — analytic parabola at 0.033%
  L2 error; manual driver `test/scratch/incns_poiseuille_driver.jl`.

## Touch order

1. `src/methods/inc_ns/simple.jl` — everything lives here (assembly, faces,
   gradient, loop); read the section comments first, they encode the design
   decisions above.
2. `test/analytical/incns_poiseuille.jl` — the 0.033% gate; run after any edit.
3. `src/solve/linear_solve.jl` — if solves are wrong/slow (cache misuse,
   pinning).
4. `src/methods/inc_ns/cavity.jl` / `cavity_mg.jl` — the siblings that fully
   stress the pressure-velocity coupling this file under-stresses.
5. `test/scratch/incns_poiseuille_driver.jl` — manual driver.
