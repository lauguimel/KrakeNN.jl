---
module: incns-fv-adjoint
path: src/methods/inc_ns/fv_adjoint.jl
owner_concern: steady-adjoint
status: implemented
last_verified: 2026-11-06
depends_on:
  - incns-fv-newton
  - incns-fv-steady-residual
  - incns-fv-gnf
  - solve-linear
  - platform
---

# incns-fv-adjoint — module implication map

Step C4 of the FV Newton roadmap: steady discrete adjoint of the C1 residual
on the C2 Newton factors. The assembly is differentiated, not the solver:
`fv_newton_finalize!` gives the unshifted J at the converged U, refactorized
(no new analysis); `Jᵀ λ = ∂J/∂U` reuses those factors; the parameter term
`λᵀ ∂R/∂θ` comes from parameter duals through the same C1 kernel. API aligned
with the LBM GNF adjoint (`gnf_steady_gradient`, `gnf_steady_adjoint ->
(s, stats)`).

## Public surface

- Parameter maps (`FVParameterMap`): `FVUniformViscosity()`,
  `FVViscosityField()` (θ = ν per cell, coloured cotangent, θ = `nothing`
  means the current field), `FVSideVelocity(prob, side; shape)` (one amplitude
  per side profile, e.g. lid speed, inlet amplitude), `FVUniformSource(comp)`,
  `FVProblemMap(f, n)` (`f(prob, θ)` returns a problem; generic eltype; the C3
  GNF entry). Every built-in map preserves `prob.visc`. For GNF law parameters,
  use `FVProblemMap(fv_with_gnf_parameters, length(fv_gnf_parameters(prob)))`.
  `fv_nparams`, `fv_param_problem`, `fv_set_params!(solver, map, θ)` (copies
  nu/bvel/src in place and replaces the viscosity model only with the same
  concrete type; otherwise throws before writing solver arrays). C5 preserves
  `prob.immersed` too: cut fractions, wall velocities, surface data and M3 mass
  data are copied only after the grid, cell tags, closure and topology checks.
- C5 from `feat/fv-c5`: `FVShapeMap(builder, base_problem, nparams)` builds the
  existing `FVProblemMap` protocol with frozen cell/cut/wall/M3 connectivity.
  `FVWallForce(...; kind = :residual_flux|:surface, wall = 0)` and
  `FVWallTorque(center; kind, region, wall)` distinguish the complete discrete
  momentum flux (including convection) from physical-cut traction integrated
  with analytic surface weights. The default force kind remains `:legacy`;
  pressure overrides apply only to that kind.
- Objectives (`FVObjective`): `FVStateObs` (½Σ w(v − t)², from indices or a
  `LineProfile`), `fv_obs_values`, `FVLinearFunctional` (+ `fv_flow_rate`,
  `fv_pressure_drop`), `FVWallForce(dir; region = :solid|:E|:W|:N|:S,
  pressure = :cell|:linear)` (the residual's Dirichlet-face viscous stress;
  Newtonian cell ν or GNF law at the wall-face strain rate, including explicit
  law-parameter dependence). Recommended drag objective:
  `FVWallForce(1; region = :solid, pressure = :linear)`; the API default stays
  `:cell`. `fv_objective`, `fv_objective_support`, `fv_objective_explicit`.
- `fv_adjoint_workspace(solver, map; chunk = 4)`;
  `fv_steady_adjoint(solver, U, gU; params, θ, ws, finalize) -> (s, stats)`;
  `fv_steady_gradient(solver, U, θ, obj; params, ws, finalize) -> NamedTuple`
  (`J, grad, gU, explicit, s, λ, residual_inf`, split timings);
  `fv_steady_tangent(solver, U, θ, δθ; ...) -> dU`.
- Platform: `IncNS(:fv_newton)` declares `SteadyAdjoint` and the new
  `AssembledJacobian`.

## Reads from

- `FVNewtonSolver` fields `prob, asm, J, cache, R` and `fv_newton_finalize!`
  (`fv_newton.jl`); `lin_solve!(...; transpose = true)` and
  `lin_supports_transpose` (`src/solve/linear*`, CUDSS extension).
- C1 kernel `fv_cell_residual` through `fv_residual!` on Dual parameter
  arrays (`fv_steady_residual.jl`), `_fv_with`, `fv_index`, `LineProfile`.
- C3 `prob.visc`, `_fv_wall_nu` and `fv_with_gnf_parameters`; wall forces use
  the residual's one-sided normal and cell tangential gradients on C1/domain
  walls; C5 immersed walls use the shared cut-wall closure.
- C5 immersed closure, surface reconstruction and mass compatibility helpers
  (`fv_immersed*.jl`); force, torque and shape derivatives share the residual's
  cut-wall data and include M3 mass unknowns where required.
- ForwardDiff (tags `FVParamTag`, objective tag); its own distance-2
  ν colouring (`fv_nu_coloring`, 7 colours on a cavity, 8 on the
  masked channel).

## Writes to

Workspace buffers (λ, g, Dual residual, colour seeds, partials) and, through
`fv_newton_finalize!`, `s.J`, `s.nzval`, `s.cache` (refactorized, symbolic
analysis kept). `fv_set_params!` writes the solver problem's nu/bvel/src,
updates a same-type viscosity model and invalidates the numerical factors.
Immersed parameter updates also copy cut, wall, surface and M3 arrays after
validation. No globals.

## Backend constraints

- Same KA kernels on CPU and GPU; objectives are evaluated on the host
  (support-restricted ForwardDiff for ∂J/∂U, ∂J/∂θ).
- GPU transposed solve needs CUDSS.jl ≥ 1.0 (0.6.5 transposes wrongly, J0);
  `lin_supports_transpose` is checked and the call throws otherwise.
- With a workspace the GPU path is designed to allocate 0 device bytes
  (tested by `incns_fv_adjoint_gpu.jl`, not yet run on a GPU).
- `n_factorize` stays 1 through any number of gradients (tested on CPU).
- CPU inference guards inspect GNF parameter residuals and adjoint kernels
  with `KernelAbstractions.CPU()`; the opt-in GPU test includes a small GNF
  law-parameter gradient parity case. CPU inference is not GPU validation.
- Integration checks use CPU Float64, Carreau–Yasuda and power law with
  `eps = 1e-2`: dense derivatives, converged central finite differences,
  summed boundary force versus momentum residual, and bitwise Newtonian
  wall-force limits on domain sides and cell-aligned solids.

## Failure modes

- **Unconverged U**: the adjoint is exact for R(U) = 0 only; `residual_inf`
  is returned, check it.
- **Shifted/stale factors**: `finalize = false` requires the cache to hold the
  unshifted J at U (shift + ΣU, ΣU² recorded in `fac_sig`); otherwise it throws an `ArgumentError`.
- **State-dependent ν (C3)**: built-in maps keep the GNF model, so changing
  `nu` with `FVUniformViscosity` or `FVViscosityField` has zero sensitivity:
  GNF reads the law instead. Change law parameters through `FVProblemMap`.
  Replacing the model type requires a new solver with the appropriate C3
  Jacobian pattern; `fv_set_params!` rejects such a replacement.
- **Wall objectives**: the residual wall closure is kept (maintainer decision
  2026-11-05). Recommend `pressure = :linear` for drag; this extrapolates only
  objective pressure, falling back to `p_P` when the second cell is not fluid.
  Both pressure options use the same GNF viscous stress. The Newtonian obstacle
  study (`benchmarks/fv_c4/results/wall_closure_cpu.txt`) found orders ≈ 1.4
  for cell pressure and 1.57 for linear pressure; these are not GNF accuracy
  claims.
- **Explicit-ν objectives** (`FVWallForce`): the explicit ∂J/∂θ for the field
  map is restricted to the objective support; a new objective with explicit
  ν dependence must declare its support correctly.

- **C5 geometry and mass**: shape derivatives require frozen connectivity;
  crossing a cell, cut-link, wall-identity or M3 body/region boundary is rejected.
  The historical M1 closed-mass path retains 15 expected failures; M3 uses one
  affine compatibility equation per solid-body/fluid-region pair. These are
  separate from opt-in GPU skips. A physical-surface force or torque needs
  analytic surface metadata; its quadrature does not exactly annihilate
  constant stress on irregular cuts.

## Touch order

1. `src/methods/inc_ns/fv_adjoint.jl` — maps, objectives, kernels, drivers.
2. `src/KrakeNN.jl` (exports), `src/platform/contract.jl` (`AssembledJacobian`),
   `src/methods/inc_ns/method.jl` (capabilities).
3. `test/analytical/incns_fv_adjoint.jl` — dense-dual parity, Taylor FD,
   tangent, factor reuse, Stokes identities, GNF preservation and gradients,
   GNF wall force and bitwise Newtonian limits.
4. `test/analytical/incns_fv_adjoint_gpu.jl` — GPU parity, allocations
   (`KRAKEN_REQUIRE_GPU=1`).
5. `benchmarks/fv_c4/` — wall-closure study, A100 cost job.
6. `test/analytical/incns_fv_adjoint_compiler.jl` — CPU inference guards,
   including GNF law-parameter residuals and adjoint kernels.
7. `src/methods/inc_ns/fv_immersed_objectives.jl`,
   `src/methods/inc_ns/fv_immersed_shape.jl`,
   `test/analytical/incns_fv_immersed*.jl` — C5 force/torque, shape, M1/M3
   contracts and opt-in GPU checks, ported from `feat/fv-c5`.
