---
module: incns-fv-gnf
path: src/methods/inc_ns/fv_gnf.jl
owner_concern: steady-newton
status: implemented
last_verified: 2026-11-06
depends_on:
  - incns-fv-steady-residual
  - incns-fv-newton
  - physics-viscoelastic
---

# incns-fv-gnf — module implication map

Step C3 of the FV Newton roadmap: generalised-Newtonian (GNF) viscosity inside
the steady colocated FV residual (C1) and its Newton solver (C2). The face
viscosity is `gnf_viscosity(law, S:S + ε²/2)` = `effective_viscosity(law, γ̇_ε)`,
`γ̇_ε = √(2 S:S + ε²)`, with the full 2D strain-rate tensor of the face
gradient the viscous flux already uses (compact normal difference, mean of the
two Green–Gauss cell gradients tangentially; wall face: one-sided normal
difference and the cell gradient). The residual kernel evaluates ν itself, so
the coloured dual assembly of C2 differentiates through ν(U) with no extra
code: only the sparsity pattern changes. No second rheology hierarchy: the law
types are those of `src/rheology/models.jl`, as in the LBM GNF drivers and the
GNF adjoint.

## Public surface

- `FVGNFViscosity(law, eps)` — isbits viscosity model stored in
  `FVSteadyProblem2D.visc`; `law <: GeneralizedNewtonian` (kinematic units,
  `thermal` ignored); `eps ≥ 0`, and `eps = 0` only for `Newtonian` or
  `CarreauYasuda` with `a ≥ 2` (else `ArgumentError`).
- `fv_steady_problem(...; rheology = law, gnf_eps = ε)` builds it; `nu` is
  then not read. `FVNewtonianViscosity()` is the C1/C2 default.
- `fv_jacobian_pattern(prob)` method for GNF problems (dispatch on the
  `visc` type parameter; `fv_newton_solver` picks it up, the colouring follows).
- Parameters as a vector (C4): `fv_gnf_parameter_names(prob)` (law fields,
  `thermal` excluded), `fv_gnf_parameters(prob) -> θ`,
  `fv_with_gnf_parameters(prob, θ; eps) -> prob'` (arrays shared, law rebuilt
  in `eltype(θ)`; a θ of duals gives ∂R/∂θ through the same kernel, CPU or GPU).
  Use `FVProblemMap(fv_with_gnf_parameters, length(fv_gnf_parameters(prob)))`
  with `fv_set_params!` and `fv_steady_gradient` for law-parameter adjoints.
  Built-in C4 maps preserve `visc`; changing the unused `nu` field has zero
  sensitivity on a GNF problem. C5 from `feat/fv-c5` also preserves
  `prob.immersed` when replacing law parameters; viscosity and immersed
  geometry remain independent problem fields.

## Reads from

- `gnf_viscosity` (`src/rheology/gnf_cell.jl`), `effective_viscosity`
  (`src/rheology/viscosity.jl`), law types (`src/rheology/models.jl`).
- Hooks `_fv_face_nu` / `_fv_wall_nu` / `_fv_visc_zero` / `_fv_visc_eltype`
  declared in `fv_steady_residual.jl`; `_fv_face_kind`, `fv_index` and the
  Newtonian `fv_jacobian_pattern` (called by `invoke`).
- C5 immersed cell dependencies, cut-wall gradients and M3 mass pattern
  extension from `fv_immersed.jl` and `fv_immersed_mass.jl`; the GNF pattern
  keeps the additional cut-closure and mass-equation couplings.

## Writes to

Nothing beyond C1/C2: the residual writes `R`, the assembler the Jacobian
values. The pattern builder returns a new host matrix.
C4's `fv_set_params!` can replace the law values while preserving the concrete
model type, solver arrays and sparsity structure. A model-type change throws
`ArgumentError` before any solver-array mutation and needs a new solver.

## Backend constraints

- The law and `eps` are isbits kernel arguments; same kernel on CPU and GPU.
  GPU parity, cuDSS Newton and zero-allocation checks in
  `test/analytical/incns_fv_gnf_gpu.jl` (`KRAKEN_REQUIRE_GPU=1`). The first
  A100 run failed during residual compilation (CUDA 6.4.1, KA 0.9.45,
  ForwardDiff 1.4.7); GPU validation after the type-promotion fix is pending.
- Pattern: u and v at the 3 × 3 cell block for the momentum rows (all
  non-outlet faces) and for the continuity row (interior faces, Rhie–Chow
  coefficient); 27 entries per interior row (Newtonian: 23/23/15). Narrower
  than the radius-2 diamond of a cell-centred viscosity (J0 measured 35/row).
  Colours 33–34 on the test grids.
- Residual promotion includes the law's number type (`_fv_visc_eltype`), so
  `fv_residual(U, prob)` with a dual law returns duals. Promotion is resolved
  from the field types of the law and the type of `eps`, excluding `thermal`,
  without collecting law values inside the residual kernel. CPU inference
  guards cover all six laws, scalar and dual state gradients, mixed
  law/regularisation precisions, and dual law parameters.
- C4 integration tests compare Carreau–Yasuda and regularised power-law
  adjoints against dense ForwardDiff and central finite differences on CPU
  Float64 with `eps = 1e-2`, including wall-force objectives and a summed
  wall-force/momentum-residual balance. CPU inference guards include the GNF
  adjoint path;
  opt-in gradient CPU/GPU parity is in `incns_fv_adjoint_gpu.jl` and remains
  unverified on a GPU in this integration session.

- C5 compiler checks cover Newtonian and GNF immersed residuals with dual
  state, viscosity, cut fractions, wall velocity and law parameters. The
  registered immersed tests cover geometry, force/torque and shape derivatives,
  M1 expected failures and M3 compatibility. These CPU checks do not establish
  GPU compilation or parity.

## Failure modes

- **Dynamic viscosity promotion**: collecting a filtered tuple of law field
  values inside `_fv_visc_eltype` loses the tuple's concrete type. Then
  `_fv_visc_zero` and the cell accumulators infer as `Any`; CUDA rejects the
  resulting dynamic conversion when the kernel writes the residual. Keep
  promotion in the type domain; successful CPU arithmetic alone does not
  establish GPU compilability.
- **PowerLaw clamp**: `nu_min`/`nu_max` are applied as written (bounds in FV
  units; LBM defaults 1e-6 / 10). Where active the derivative is 0 and Newton
  stalls: pass `nu_min = 0, nu_max = Inf` unless the clamp is intended.
  Building the problem with finite bounds emits one `@warn` (`FVGNFViscosity`).
- **ε is a model change**: results are those of the regularised law;
  the power-law Poiseuille test shows convergence to the unregularised
  profile as ε → 0 (ε = 1e-4: 7.7e-4 relative at 64 cells).
- **Laws singular at zero shear with ε = 0** (power law, Bingham, HB, Cross,
  CY with a < 2): NaN Jacobian on states at rest; the constructor refuses.
- **Rebuilding a problem** with the 5-argument `FVSteadyProblem2D` constructor
  drops the GNF model silently (Newtonian default); pass `p.visc`. C4's
  `_fv_with` now preserves it for every built-in parameter map and C5
  additionally keeps the immersed geometry. Custom reconstruction must pass
  both `p.visc` and `p.immersed`.
- **Wall closure**: wall faces use the one-sided normal gradient; the wall
  pressure stays first order with variable ν (C1 decision 2026-11-04, kept).
  `FVWallForce` evaluates the same GNF law at the same wall-face strain rate
  for both pressure options, including its state and law-parameter derivatives.
  The 2026-11-05 decision recommends `pressure = :linear` for drag (objective
  pressure extrapolation only); the API default remains `:cell`.

## Touch order

1. `src/methods/inc_ns/fv_steady_residual.jl` — hooks, face gradient, `visc`
   field (any stencil change there must update the GNF pattern).
2. `src/methods/inc_ns/fv_gnf.jl` — model, pattern rule, parameter helpers.
3. `src/Kraken.jl` — include after `fv_newton.jl`, exports.
4. `test/analytical/incns_fv_gnf.jl` — dense-Jacobian parity (catches dropped
   entries), concrete viscosity/residual inference, parameters, n = 1,
   Poiseuille, cavity, LBM obstacle channel.
5. `test/analytical/incns_fv_gnf_gpu.jl`, `benchmarks/fv_c3/` — GPU receipts.
6. `src/methods/inc_ns/fv_adjoint.jl`, `test/analytical/incns_fv_adjoint*.jl`
   — parameter-map preservation, law-parameter adjoints and GNF wall stress.
7. `src/methods/inc_ns/fv_immersed*.jl`,
   `test/analytical/incns_fv_immersed*.jl` — C5 cut-wall gradient and
   parameter preservation, extended sparsity and M1/M3 contracts from
   `feat/fv-c5`.
