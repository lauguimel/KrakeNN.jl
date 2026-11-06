---
module: incns-fv-steady-residual
path: src/methods/inc_ns/fv_steady_residual.jl
owner_concern: steady-residual
status: implemented
last_verified: 2026-11-05
depends_on:
  - incns-platform
  - platform
---

# incns-fv-steady-residual — module implication map

Step C1 of the FV Newton roadmap: the steady 2D colocated finite-volume
residual R(u, v, p) of incompressible Navier–Stokes (or Stokes), as ONE
KernelAbstractions kernel, generic in the element type. It is the single
source of truth that C2 (Jacobian by coloured duals + Newton on cuDSS), C3
(GNF viscosity) and C4 (discrete adjoint) differentiate; there is no CPU
mirror — the CPU path is the same kernel on the KA CPU backend. Discretisation
reproduces the J0 prototype (`benchmarks/fv_newton_j0/FVJ0.jl` on
`feat/fv-newton-j0`) bit for bit (receipt `benchmarks/fv_c1/results/`).

## Public surface

- `FVGrid2D{T}` (isbits, passed by value to the kernel): `Nx, Ny, h, conv`
  (1 = NS, 0 = Stokes) and side codes `bcE, bcW, bcN, bcS` ∈
  `FV_BC_WALL | FV_BC_INLET | FV_BC_OUTLET | FV_BC_PERIODIC`.
- `FVSteadyProblem2D(grid, tag, nu, bvel, src[, visc])`: `tag::Matrix{UInt8}` (`Nx×Ny`,
  `FV_CELL_FLUID | FV_CELL_SOLID | FV_CELL_PINNED`), `nu` (`Nx×Ny` cell
  kinematic viscosity), `bvel` (`2×max(Nx,Ny)×4`, (u,v) of each boundary face,
  side order E, W, N, S, read on wall/inlet sides), `src` (`3NxNy`, sources in
  unknown ordering: momentum rows subtract `h²s`, continuity `h²s_mass`). Each
  array may have its own eltype (Dual parameter arrays are legal: C4 path).
  `visc` is the viscosity model: `FVNewtonianViscosity()` (default, the
  5-argument constructor; reads `nu`) or `FVGNFViscosity(law, eps)` (C3, map
  `incns-fv-gnf-implication.md`; `nu` is then not read). Code that rebuilds a
  problem from its fields must pass `p.visc` on, or the GNF law is dropped.
- `fv_residual!(R, U, prob; workgroup = 64, sync = true)` — the hot path;
  `fv_residual(U, prob)` allocating (promoted eltype).
- `fv_cell_residual(U, tag, nu, bvel, src, grid, i, j[, visc]) -> (Ru, Rv, Rc)` — the
  per-cell function the kernel calls (for a C2 kernel that fuses seeding /
  decompression around it).
- Setup (CPU, once): `fv_steady_problem(Nx, Ny; h, nu, bc, boundary_velocity,
  solid, source, conv, T, rheology, gnf_eps)`, `fv_pressure_pins!(tag, grid)`, `fv_adapt(to, prob)`,
  `fv_state(prob, u, v, p)`, `fv_fields(prob, U)`, `fv_index(grid, i, j, comp)`,
  `fv_nunknowns`.
- Platform: `residual(prob::FVSteadyProblem2D, IncNS(:fv_newton), U[, p])`;
  only `IncNS(:fv_newton)` declares `SteadyResidual` (decision 2026-11-04:
  capabilities are driver-dependent); other drivers raise `ArgumentError`.
  Consumer: C2 Newton, map `incns-fv-newton-implication.md`.
- Unknown ordering `k = 3(i-1 + (j-1)Nx) + {1:u, 2:v, 3:p}` (interleaved, as J0).

## Reads from

- `IncNS`, `residual`, `SteadyResidual` from `src/methods/inc_ns/method.jl`
  and `src/platform/` (included after `method.jl` in `src/Kraken.jl`; the
  generic `residual` fallback in `platform/residual.jl` is less specific).
- KernelAbstractions only; no FVFD operator is reused: the residual is a fused
  per-cell stencil, so that one kernel launch evaluates every equation in any
  element type (the `src/fvfd` operators are per-field kernels on Float arrays).

## Writes to

Only `R` (length `3NxNy`). No globals, no caches. `fv_pressure_pins!` rewrites
the pin marks of a host `tag` in place.

## Backend constraints

- All arrays of `prob` and `U`, `R` on one backend; `fv_adapt(CuArray, prob)`.
- GPU hot path allocation-free (`CUDA.@allocated == 0`, free memory stable over
  10 evaluations): `test/analytical/incns_fv_steady_residual_gpu.jl`, opt-in
  with `KRAKEN_REQUIRE_GPU=1`.
- Branches depend only on integers (tags, side codes), never on values: the
  upwind switch is `abs`-based. Keeps duals exact through every branch (H6')
  and keeps the kernel traceable by global sparsity detectors (C2 pattern).
- Viscosity hooks `_fv_face_nu(visc, nuP, nuN, ux, uy, vx, vy)` and
  `_fv_wall_nu(visc, nuP, ux, uy, vx, vy)`: the face velocity gradient is
  computed for every model (dead code for the Newtonian one, whose result is
  unchanged bit for bit).
- Dependency support of a cell row: cells within the 5×5 box around it (the
  Rhie–Chow mean-gradient term reaches p_{i±2}, cross terms reach diagonals).
  The test helper `test/analytical/fv_steady_helpers.jl` colours by
  `(i mod 5, j mod 5)` and reproduces the dense Jacobian exactly; C2 still
  computes its own pattern / colouring per grid (plan F21).

## Failure modes

- **Periodic size and box colouring**: the 5×5 test colouring is wrong when a
  periodic side is not divisible by 5 (the helper throws).
- **Doubly periodic box**: uniform (u, v) is a null mode of the Jacobian (J0
  D-RC study); a Newton solve needs a mean-velocity constraint there.
- **Pressure accuracy at walls**: the half-cell wall closure (`(u_w - u_P)·2/h`,
  `ν_P`, `p_f = p_P`) makes the pressure first order in the wall cells with
  variable ν (velocity stays order 2): MMS measured 2026-11-04. Kept as is
  by decision 2026-11-04 (Guillaume); revisit before C4 (adjoint objectives
  on wall pressure / wall shear would inherit the first-order error).
- **Pins**: a closed pocket without a pin makes the Jacobian singular; always
  build through `fv_steady_problem` or call `fv_pressure_pins!` after editing
  `tag`. Pinned row is `p_P = 0` (not `p_P − p_ref`).
- **Not the SIMPLE discretisation**: the SIMPLE drivers' converged state depends
  on relaxation (F16); expect agreement at discretisation level only.

## Touch order

1. `src/methods/inc_ns/fv_steady_residual.jl` — discretisation, setup helpers.
2. `src/Kraken.jl` — include after `methods/inc_ns/method.jl`, export block.
3. `test/analytical/incns_fv_steady_residual.jl` (+ `fv_steady_helpers.jl`) —
   pins, exact discrete solutions, MMS orders, duals vs FD, SIMPLE agreement.
4. `test/analytical/incns_fv_steady_residual_gpu.jl` — GPU parity, allocations.
5. `benchmarks/fv_c1/j0_parity.jl` — rerun against FVJ0 after any change to the
   discretisation (a deliberate change must say so and drop the parity claim).
6. `src/methods/inc_ns/fv_newton.jl` — the Jacobian pattern mirrors the
   kernel's reads: any stencil change must update `fv_jacobian_pattern`
   (the C2 dense-parity test catches a missing entry).
7. `docs/agent/incns-platform-implication.md` — capability list.
