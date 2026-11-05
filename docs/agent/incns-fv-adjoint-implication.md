---
module: incns-fv-adjoint
path: src/methods/inc_ns/fv_adjoint.jl
owner_concern: steady-adjoint
status: implemented
last_verified: 2026-11-05
depends_on:
  - incns-fv-newton
  - incns-fv-steady-residual
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
  GNF entry). `fv_nparams`, `fv_param_problem`, `fv_set_params!(solver, map, θ)`
  (copies nu/bvel/src in place).
- Objectives (`FVObjective`): `FVStateObs` (½Σ w(v − t)², from indices or a
  `LineProfile`), `fv_obs_values`, `FVLinearFunctional` (+ `fv_flow_rate`,
  `fv_pressure_drop`), `FVWallForce(dir; region = :solid|:E|:W|:N|:S,
  pressure = :cell|:linear)` (the residual's own Dirichlet-face flux; explicit
  ν dependence). `fv_objective`, `fv_objective_support`, `fv_objective_explicit`.
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
- ForwardDiff (tags `FVParamTag`, objective tag); its own distance-2
  ν colouring (`fv_nu_coloring`, 7 colours on a cavity, 8 on the
  masked channel).

## Writes to

Workspace buffers (λ, g, Dual residual, colour seeds, partials) and, through
`fv_newton_finalize!`, `s.J`, `s.nzval`, `s.cache` (refactorized, symbolic
analysis kept). `fv_set_params!` writes the solver problem's nu/bvel/src. No
globals.

## Backend constraints

- Same KA kernels on CPU and GPU; objectives are evaluated on the host
  (support-restricted ForwardDiff for ∂J/∂U, ∂J/∂θ).
- GPU transposed solve needs CUDSS.jl ≥ 1.0 (0.6.5 transposes wrongly, J0);
  `lin_supports_transpose` is checked and the call throws otherwise.
- With a workspace the GPU path is designed to allocate 0 device bytes
  (tested by `incns_fv_adjoint_gpu.jl`, not yet run on a GPU).
- `n_factorize` stays 1 through any number of gradients (tested on CPU).

## Failure modes

- **Unconverged U**: the adjoint is exact for R(U) = 0 only; `residual_inf`
  is returned, check it.
- **Shifted factors**: `finalize = false` assumes the cache holds the
  unshifted J at U (true right after a finalize); otherwise the gradient is
  wrong without error.
- **State-dependent ν (C3)**: parameter maps change only nu/bvel/src; a GNF
  law whose ν depends on U must enter the residual (C3) and its parameters
  through `FVProblemMap` or a dedicated map. The Jacobian pattern must already
  cover the widened stencil (C3).
- **Wall objectives**: first-order wall pressure at walls; the obstacle drag
  converges at order ≈ 1.4 (`benchmarks/fv_c4/results/wall_closure_cpu.txt`);
  `pressure = :linear` is an objective-level option only.
- **Explicit-ν objectives** (`FVWallForce`): the explicit ∂J/∂θ for the field
  map is restricted to the objective support; a new objective with explicit
  ν dependence must declare its support correctly.

## Touch order

1. `src/methods/inc_ns/fv_adjoint.jl` — maps, objectives, kernels, drivers.
2. `src/Kraken.jl` (exports), `src/platform/contract.jl` (`AssembledJacobian`),
   `src/methods/inc_ns/method.jl` (capabilities).
3. `test/analytical/incns_fv_adjoint.jl` — dense-dual parity, Taylor FD,
   tangent, factor reuse, Stokes identities.
4. `test/analytical/incns_fv_adjoint_gpu.jl` — GPU parity, allocations
   (`KRAKEN_REQUIRE_GPU=1`).
5. `benchmarks/fv_c4/` — wall-closure study, A100 cost job.
