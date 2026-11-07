---
module: incns-projection
path: src/methods/inc_ns/projection.jl
owner_concern: pressure-velocity-coupling
status: implemented
last_verified: 2026-06-11
depends_on:
  - solve-poisson
  - solve-linear
---

# incns-projection — module implication map

The UNSTEADY rung of the incompressible solver family: incremental
pressure-correction projection (fractional step) on the same collocated
cell-centred grid and conventions as `simple.jl`/`cavity.jl`. Per step: AB2
explicit conservative-central advection (Euler on the first step), implicit
θ-diffusion (Crank-Nicolson default, backward-Euler fallback), ONE pressure
Poisson for the increment `phi = p^{n+1} - p^n`. Per-axis `:periodic`/`:wall`
BCs. Both implicit operators are constant, so they are factorized ONCE before
the time loop via the `lin_factorize`/`lin_solve!` seam — 2 factorizations per
run regardless of step count. **Registered in `src/KrakeNN.jl`** and exported
(`solve_incns_projection`); also reachable through the platform contract as
`solve(params, IncNS(:projection))` (`src/methods/inc_ns/method.jl`). Include
guards keep the file standalone-include-able.

## Public surface

- `solve_incns_projection(; nx, ny, Lx, Ly, nu, dt, nsteps, bc_x=:periodic,
  bc_y=:periodic, fx=0.0, fy=0.0, u0=nothing, v0=nothing, p0=nothing,
  scheme=:cn, rhie_chow=:increment, backend=CPU(), callback=nothing) ->
  NamedTuple` — the only entry point. Returns `(u, v, p, uf, vf, t_final,
  ke_history, max_div_inf, div_inf_final, nfactorizations, nlinsolves, dx, dy,
  xcenters, ycenters, nx, ny, nu, dt, nsteps, scheme, rhie_chow)`. `u0/v0/p0`
  accept `nothing`, an `(x,y)->value` function, or an `(nx,ny)` matrix;
  `fx/fy` are constant body forces (e.g. `fx=G` drives a periodic channel).
  `callback(step, t, u, v, p)` runs per step with the LIVE arrays — copy what
  you keep. `backend` is currently cosmetic (host loops; KA seam for later).
- De-facto public internals tests/siblings poke:
  `_proj_assemble_neg_laplacian(nx, ny, dx, dy; bc_x, bc_y)`
  (`:periodic`/`:dirichlet0` "+2/h²"/`:neumann`, local copy of the simple.jl
  assembler), `_proj_advect!` (central fluxes advected by the stored
  divergence-free FACE velocities), `_proj_faces_from_cells!` (momentum
  interpolation, `dcoef` selects plain averaging vs Rhie-Chow deviation),
  `_proj_face_divergence!`, `_proj_compact_gradient!` (the discrete transpose
  of the face divergence), `_proj_correct_faces!` (exact face projection of
  `phi`), `_proj_init_field`.

## Reads from

- `solve-poisson` (`src/solve/poisson.jl`): `pin_reference_dof` (via the seam)
  — guarded include.
- `solve-linear` (`src/solve/linear_solve.jl`): `lin_factorize`/`lin_solve!`/
  `CPUBackendTag` — the momentum Helmholtz `I/dt + nu*theta*(-Lap)` (SPD,
  velocity walls `:dirichlet0`) and the singular pressure `(-Lap)` (pressure
  walls `:neumann`, `pin_k0=1`) each hold ONE `LinearSolveCache` across the
  whole time loop: `nfactorizations == 2` for ANY `nsteps`,
  `nlinsolves == 3*nsteps` (2 momentum + 1 Poisson per step).

## Writes to

Nothing global; allocates all fields once before the loop and mutates them in
place across steps, returning them in the NamedTuple (`ke_history` is pushed
per step). The callback is the only external write surface — it receives the
live arrays. No I/O, no device state. The two factorization caches live for
the duration of the call.

## Backend constraints

CPU-only in practice (host `@inbounds` loops, CHOLMOD factorizations, `Vector`
RHS); the `backend` kwarg is the future KA seam, mirroring `simple.jl`. Cost
profile: three back-substitutions + a handful of O(n) host sweeps per step —
factorize-once makes the loop back-substitution-cheap. GPU path = swap the
seam backend tag (cuDSS) once a device variant exists; until then send long
runs to Aqua per the GPU policy.

## Failure modes

- **`rhie_chow=:full` collapses temporal order**: the classical `d = dt`
  Rhie-Chow deviation against the FULL pressure injects an O(dt·h²) error into
  the advecting face flux — measured temporal order degrades to ~1.1–1.6 (the
  time-step-dependent momentum-interpolation defect, Choi IJNMF 1999). The
  default `:increment` (faces = `avg(u*)`; the compact pressure coupling rides
  on the EXACT face projection of the increment `phi`) restores clean order 2
  (measured 2.008/2.019). Keep `:full` only for steady-dominated or strongly
  pressure-coupled runs — never for time-accuracy claims.
- **Sign convention vs cavity.jl**: the assembled pressure operator is the
  POSITIVE-definite `(-Lap)`; here the minus sign lives in the RHS
  (`b = -div(uf*)/dt`) so `phi` IS the physical increment and all updates use
  plain signs. cavity.jl solves `Ap*pcorr = +div(u*)` so its `pcorr` is MINUS
  the correction. Mixing the two conventions diverges within a few steps — do
  not "fix" one side without the other (header SIGN NOTE).
- **Exact face projection is structural**: the face divergence of
  `grad_face(phi)` is EXACTLY `-Ap*phi`, so one Poisson + `_proj_correct_faces!`
  annihilates the face divergence in one shot (measured 7e-15). Cells must use
  the COMPACT gradient (the discrete transpose) — substituting a wide stencil
  on either side breaks the exact projection and re-opens checkerboarding.
- **Central, not upwind, advection**: upwinding the advected face value clamps
  the spatial order to 1 (measured on Taylor-Green). Wall faces carry zero
  flux identically (no-slip), so the wall term vanishes — "adding" a wall flux
  treatment is a regression.
- **AB2 bootstrap and CN explicit half**: step 1 is Euler; `convo_*` must be
  copied AFTER the step (reordering silently corrupts the AB2 history). The CN
  explicit diffusion half is a `Lmom` matvec on `u^n` (wall ghosts `-u_c`);
  for `:be` it is skipped via `nu_expl = 0`, not by re-assembly.
- Receipts: `test/analytical/incns_unsteady_taylor_green.jl` — spatial order
  1.995/1.999, temporal CN self-convergence order 2.008/2.019 (gate
  [1.8, 2.4]), BE fallback 1.05/1.10 (gate ≥0.9), max face divergence 7e-15,
  KE-decay error 6.4e-5, factorize-once receipts asserted;
  `test/analytical/incns_unsteady_startup_channel.jl` — transient profile L2
  0.03–0.11% vs the Fourier series (gate 0.5%), `max|v|` 5e-16, `p` at
  roundoff (wall-row checkerboard guard).

## Touch order

1. `src/methods/inc_ns/projection.jl` — everything lives here (assembly,
   advection, faces, projection, loop); the header encodes the time scheme,
   the `:increment` vs `:full` trade-off and the sign convention.
2. `test/analytical/incns_unsteady_taylor_green.jl` — the order/divergence/KE
   gates; run after any edit.
3. `test/analytical/incns_unsteady_startup_channel.jl` — wall BCs, transient
   profile, zero cross-flow and the pressure-roundoff checkerboard guard.
4. `src/solve/linear_solve.jl` — if a solve is wrong (pinning, `spd`, cache
   reuse).
5. `src/methods/inc_ns/simple.jl` / `cavity.jl` — the steady siblings sharing
   the grid/operator conventions; the pressure-sign trap lives between this
   file and cavity.jl.
6. `test/scratch/incns_projection_driver.jl` — manual driver (gitignored).
