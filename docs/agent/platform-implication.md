---
module: platform
path: src/platform/
owner_concern: method-agnostic-contract
status: phase-2a
last_verified: 2026-06-12
depends_on: []
---

# platform — module implication map

The `platform/` module is the **stable, method-agnostic contract** every method and physics
enters behind (the "6 nouns"). **Phase 0** added the abstract types + capability introspection.
**Phase 0b** added the first concrete wrapper (`LBM`, `LBMSolution`, behaviour-preserving
`solve`/`sample` forwarding to `run_simulation`). **Phase 1** adds observables + `observe`/`predict`
— quantities comparable to data, computed **via `sample`** (the path to calibration). **Phase 2a**
adds `residual` and `adjoint_vjp` as thin CPU-Float64 AD seam delegations. Still additive, LBM
untouched. The remaining verb (`fit`) and `ParameterSpace`/closures land in later phases. Design dossier: `docs/platform/` (00-BILAN … 06-WORKFLOW); free-DOF rationale in
`docs/platform/05-DOF-LIBRES.md`.

## Public surface

Defined under `src/platform/`, re-exported by `Kraken`:

Phase 0 (`contract.jl`):
- `AbstractProblem`, `AbstractMethod`, `AbstractSolution`, `AbstractObservable`, `AbstractClosure`.
- `@enum Capability` = `ForwardSolve`, `GPUExecution`, `SteadyAdjoint`, `TransientAdjoint`,
  `FiniteDiff`, `NeuralClosure`, `SteadyResidual`.
- `capabilities(m::AbstractMethod) -> Set{Capability}` — default empty; a method opts in.

Phase 0b (`solution.jl`, `sample.jl`):
- `LBM <: AbstractMethod` — `capabilities(::LBM)` = `{ForwardSolve, GPUExecution, SteadyAdjoint, SteadyResidual}`.
- `LBMSolution{R} <: AbstractSolution` — thin wrapper over the `run_simulation` `NamedTuple`.
- `solve(problem, ::LBM; kwargs...) -> LBMSolution` — forwards verbatim to `run_simulation`.
- `sample(sol::LBMSolution, field[, query])` — pass-through query (`getproperty`, `:`, `idx::Tuple`).

Phase 1 (`observe.jl`):
- `observe(sol::AbstractSolution, o::AbstractObservable) -> Prediction` — computes `o` via `sample`
  ONLY (generic on the solution type; new observables add a method here).
- `predict(problem, ::AbstractMethod, o; kwargs...)` = `observe(solve(problem, method; kwargs...), o)`.
- `Prediction{O,V}` (observable + value); observables `FieldProbe(field, index::Tuple)`,
  `LineProfile(field, indices)`, `FieldReduction(field, reducer)`. **Integral QoIs (drag, Nusselt)
  deferred** — they need boundary integration, not pure `sample`.

Phase 2a (`residual.jl`):
- `LBMGeomParams` — parameter bundle for Newtonian CPU-Float64 AD path (q_wall, is_solid, u_profile, rho_out, s_plus, s_minus, Nx, Ny).
- `LBMThermalParams` — parameter bundle for thermal (Boussinesq) AD path (q_wall, params::ADNatconvParams, Nx, Ny).
- `LBMVEParams` — parameter bundle for viscoelastic (Oldroyd-B) AD path (g::ADVEEmbeddedGeom, q_wall, u_profile, p::ADVECoupledParams).
- `SteadyResidual` (Capability enum value) — added to `capabilities(::LBM)`.
- `residual(problem, ::LBM, u, p) -> same shape as u` — R = u - G(u,p); Enzyme-free (calls ad_step!/ad_thermal_cut_step!/ad_ve_coupled_step! directly).
- `adjoint_vjp(problem, ::LBM, u_star, p, v) -> same shape as v` — (I - dG/du)^T v; delegates to `_ad_vjp_GtT` / `_ad_thermal_vjp_GtT` / `_ad_ve_vjp_GtT`; requires Enzyme loaded.

Phase 2b-1 (`residual.jl` additive, `ad_step.jl` additive, `KrakenADExt.jl` additive):
- `LBMScalarParams` — parameter bundle with free scalar ν; geometry fields same as `LBMGeomParams`; `s_plus`/`s_minus` derived from ν at construction.
- `residual(problem, ::LBM, f, p::LBMScalarParams)` — identical body to `LBMGeomParams` dispatch; uses derived rates.
- `adjoint_vjp(problem, ::LBM, f_star, p::LBMScalarParams, v)` — identical body to `LBMGeomParams` dispatch; delegates to `_ad_vjp_GtT`.
- `_ad_pvjp_nu(f_star, lambda, p::LBMScalarParams) -> Float64` — (private) Enzyme Reverse over `ad_step_nu!` with `Active(ν)`; returns dL/dν scalar.

Phase 2b-2 (`calibration.jl` NEW):
- `ParameterSpace` — named↔flat bijection with bounds, log-scale, fixed/free masks. Methods: `to_flat`, `from_flat`, `n_free`, `project!`.
- `loss(predictions, data; weights) -> Float64` — data-misfit `(1/2)||ŷ-y||²`; Enzyme-free.
- `fit(problem, ::LBM, data, p0, pspace; observables, kwargs...) -> CalibResult` — projected gradient + Armijo backtracking; gradient via steady adjoint chain (`_ad_pvjp_nu` for ν DOF, `_ad_dqwall_terms` for radius DOF).
- `CalibResult` — (`p_opt`, `loss_final`, `loss_trace`, `grad_trace`, `n_iter`, `converged`, `message`).
- `_dJ_df_lineprofile_ux` — (private) analytic dL/df for `LineProfile(:ux)` observable; Enzyme-free.

Phase 2c-2 (`calibration.jl` extended, `ext/KrakenOptimExt.jl` NEW, `Project.toml` +Optim weakdep):
- `_reg_loss(nu_vec, alpha)` + `_reg_grad(nu_vec, alpha)` — Tikhonov smoothness
  penalty `(α/2) ‖D·ν‖²` and analytic gradient (discrete negative Laplacian); Enzyme-free.
- `_is_nufield_pspace(pspace)` + `_extract_nufield(p_named, Ny)` — routing helpers.
- `fit(...; reg_weight=0.0, method=:pgd)` — `reg_weight > 0` adds regularization to
  the loss and gradient for field-ν problems; `method=:lbfgs` routes to L-BFGS.
- `_fit_lbfgs` stub in `src/ad/ad_api.jl` + impl in `ext/KrakenOptimExt.jl` —
  L-BFGS-B via `Optim.Fminbox(LBFGS())`; first non-Enzyme `[weakdeps]` in Kraken.
- `Project.toml`: `Optim = "429524aa-..."` in `[weakdeps]`,
  `KrakenOptimExt = "Optim"` in `[extensions]`, `Optim = "2"` in `[compat]`.
- Exit gate C-2a: field twin experiment `ν(y) = 0.03 + 0.04·sin(π·(j-1)/(Ny-1))`,
  Ny=8, σ=5e-4, reg_weight=1e-4, seed=42 — rel_L2 < 10% in ≤200 L-BFGS iters.

Phase 2c-1 (`residual.jl` + `ad_step.jl` + `ad_forward.jl` + `KrakenADExt.jl`, additive):
- `LBMFieldParams` — parameter bundle with FREE per-row ν(y) field (Vector{Float64}, length Ny);
  `s_plus_field`/`s_minus_field` derived at construction via `ad_trt_rates_inline`.
- `residual(_, ::LBM, f, p::LBMFieldParams)` — calls `ad_step_nufield!`; degenerates to scalar
  path bit-exactly for uniform nu_field.
- `adjoint_vjp(_, ::LBM, f_star, p::LBMFieldParams, v)` — delegates to `_ad_vjp_GtT_nufield`
  (exact state VJP, Const nu_field, not mean-ν approximation).
- `_ad_pvjp_nufield(f_star, lambda, nu_field, ...) -> Vector{Float64}` — (private) Enzyme Reverse
  with Duplicated array; O(1) Enzyme calls for all Ny rows simultaneously.
- `_ad_vjp_GtT_nufield(f_star, v, ..., nu_field, ...) -> Array{Float64,3}` — (private) state VJP,
  Const nu_field; implemented in `ext/KrakenADExt.jl`.

Second concrete method (`src/methods/inc_ns/method.jl`, mirrors the LBM wrapper):
- `IncNS <: AbstractMethod` — `IncNS(driver)` with driver ∈
  `{:simple, :cavity, :cavity_mg, :projection, :manifold, :fv_newton}`;
  `capabilities` is driver-dependent: `{ForwardSolve, SteadyResidual}` for
  `:fv_newton`, `{ForwardSolve}` for the others (CUDA seam is manual-load).
- `IncNSSolution{R} <: AbstractSolution` — thin wrapper over the driver `NamedTuple`.
- `solve(params::NamedTuple, m::IncNS)` — splats `params` verbatim into the matching
  `solve_incns_*` driver; `sample` pass-through methods as for `LBMSolution`.
  Registration map: `docs/agent/incns-platform-implication.md`.

## Reads from

Phase 0 reads nothing. **Phase 0b**: `solve(_, ::LBM)` forwards to `run_simulation`
(`src/simulation_runner.jl`) at call time; `LBMSolution` wraps the `NamedTuple` it returns
(`ρ`, `ux`, `uy`, `setup`). **Phase 1**: `observe` reads a solution only through `sample`; `predict`
also goes through `solve`. No new external read; nothing imported beyond `Base`. The runner
reference is **lazy** (resolved at call time), so the `platform/` includes can precede the runner.

## Writes to

Nothing persistent. No mutation, no global registries, no side effects, no I/O. `capabilities`,
`solve`, `observe`, `predict` each return fresh values; `sample` returns the wrapped arrays by
reference (`sample(sol, :ux) === sol.result.ux`) or an indexed scalar. The abstract/spec types carry
no mutable state. `solve`/`predict` do not mutate `problem`.

## Backend constraints

Phase 0/1 remains backend-irrelevant by construction: type definitions + thin dispatch; nothing
there enters a kernel or allocates in a loop. Backend portability is expressed *through* the
contract (a method declares `GPUExecution in capabilities(m)`). `solve`/`sample`/`observe` add no
backend coupling — whatever backend `run_simulation` ran on is preserved untouched inside
`LBMSolution`. `observe` operates on the host arrays returned by the runner. Phase 2a residual/VJP
dispatches are explicitly CPU-Float64 AD paths; unsupported arrays fall through to the generic
platform fallback.

## Failure modes

- The default `capabilities(::AbstractMethod) = Set{Capability}()` is a deliberate **fail-safe**:
  an undeclared method appears *incapable* rather than silently wrong.
- `solve`/`sample`/`observe` are pass-throughs/derivations, inheriting `run_simulation`'s failure
  modes verbatim (the parity test pins "no behaviour change"). `sample(sol, :badfield)` raises the
  normal `getproperty` error; an out-of-range probe index raises the normal bounds error — both
  intentional, not swallowed.
- **Name-collision watch**: `solve`/`sample`/`observe`/`predict` are exported as Kraken's own
  generic functions (no dep exports them today). When SciML enters (Phase 4), switch `solve` to
  extend `CommonSolve.solve` rather than shadow it.
- `adjoint_vjp` without Enzyme loaded propagates the existing `_ad_*_vjp_GtT` stub error by design;
  `residual` remains Enzyme-free.
- The real risk stays **architectural**: over-abstraction. Guardrail (`docs/platform/06-WORKFLOW.md`
  red-team #1) — generalize the contract only when a 2nd concrete `AbstractMethod` forces it.

## Touch order

1. `src/platform/contract.jl` — the 6 types + `Capability` + `capabilities`.
2. `src/platform/solution.jl` — `LBM`, `LBMSolution`, `solve` (forwards to `run_simulation`).
3. `src/platform/sample.jl` — `sample` pass-through.
4. `src/platform/observe.jl` — `observe`/`predict`/`Prediction` + observables
   (`FieldProbe`/`LineProfile`/`FieldReduction`).
5. `src/platform/residual.jl` — Phase 2a: `residual`, `adjoint_vjp`, `LBMGeomParams`, `LBMThermalParams`, `LBMVEParams`, `SteadyResidual`.
5b. `src/platform/calibration.jl` — Phase 2b-2: `ParameterSpace`, `loss`, `fit`, `CalibResult`.
6. `src/Kraken.jl` — the `# --- Platform contract ---` include + export block (a choke file;
   edits serialized on `dev/platform`).
7. `test/platform/contract_parity_test.jl` — capabilities, bit-for-bit parity vs `run_simulation`,
   and observe/predict (Phase 1).
8. `test/platform/residual_vjp_test.jl` — Phase 2a residual construction/fixed-point checks plus Enzyme-gated VJP parity.
9. Later phases add siblings under `src/platform/` (`calibration.jl`, `closure.jl`)
   — see `docs/platform/02-PLAN-IMPLEMENTATION.md`.
