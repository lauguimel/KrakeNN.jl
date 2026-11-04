---
module: incns-platform
path: src/methods/inc_ns/method.jl
owner_concern: method-registration
status: implemented
last_verified: 2026-06-11
depends_on:
  - platform
  - incns-simple
  - incns-cavity-mg
  - incns-projection
  - incns-manifold
  - scalar-transport
  - solve-linear
  - solve-poisson
  - solve-poisson-mg
---

# incns-platform — module implication map

The REGISTRATION layer for the IncNS solver stack (mission M-REG-IMPL): the
formerly standalone `src/solve/*` services, `src/methods/inc_ns/*` drivers and
the scalar-transport brick are now included in `src/Kraken.jl` and exported,
plus a thin platform-contract wrapper (`IncNS <: AbstractMethod`) mirroring the
LBM precedent (`src/platform/solution.jl` + `sample.jl`). WIRING ONLY — no
solver math lives here; every driver call is a verbatim keyword forward.

## Public surface

Contract wrapper (`src/methods/inc_ns/method.jl`):
- `IncNS(driver::Symbol) <: AbstractMethod`, driver ∈
  `{:simple, :cavity, :cavity_mg, :projection, :manifold}` (constructor-validated).
- `capabilities(::IncNS) = Set((ForwardSolve, SteadyResidual))` — no
  GPU/adjoint over-claiming (the CUDA seam files are manual-load).
  `SteadyResidual` = the C1 steady colocated FV residual
  (`residual(prob::FVSteadyProblem2D, ::IncNS, U)`, map
  `incns-fv-steady-residual-implication.md`), not the SIMPLE drivers' residual.
- `IncNSSolution{R} <: AbstractSolution` — NamedTuple wrapper; `sample`
  pass-through (`getproperty`, `:`, `idx::Tuple`), like `LBMSolution`.
- `solve(params::NamedTuple, m::IncNS) -> IncNSSolution` — splats `params` into
  the matching `solve_incns_*` driver (bit-identical to a direct call; parity
  pinned by `test/platform/incns_contract_test.jl`).

Exports added in `src/Kraken.jl`:
- Drivers: `solve_incns_simple`, `solve_incns_cavity`, `solve_incns_cavity_mg`,
  `solve_incns_projection`, `solve_incns_manifold`, `manifold_full_cell_mask`,
  `solve_scalar_transport`.
- Linear-solve seam: `lin_factorize`, `lin_solve!`, `LinearSolveCache`,
  `LinearSolveBackend`, `CPUBackendTag`, `CUDABackendTag`.
- Elliptic: `solve_poisson_dirichlet`, `solve_poisson_neumann`,
  `pin_reference_dof`, `assemble_poisson_embedded`, `solve_poisson_embedded`,
  `assemble_poisson_embedded_from_fvfd`, `fractions_from_fvfd`,
  `solve_poisson_mg`, `solve_poisson_mgcg`.
- FVFD operators: `gdl_divergence_2d!`, `gdl_pressure_gradient_2d!`,
  `gdl_laplacian_apply_2d!` + the `_embedded_2d!` variants.

UNexported (call as `Kraken.x`): generic helpers (`l2_error`, `cell_ij`,
`cell_coordinates`, `exact_field`, `solve_poisson`, `assemble_poisson_*`
non-embedded, `tilted_half_plane_fractions`, `first_fluid_dof`,
`fluid_l2_error`, `FVFD_BC_*`, …).

## Reads from

Include order in `src/Kraken.jl` (dependency-true, after `fvfd/FVFD.jl`):
`solve/poisson.jl` FIRST — it tail-includes `solve/linear_solve.jl`, which has
NO self-guard, so `linear_solve.jl` must NEVER be included directly — then
`poisson_embedded.jl`, `poisson_embedded_fvfd.jl`, `poisson_mg.jl`, the five
`inc_ns` drivers, `scalar_transport/thermal_transport.jl`, and `method.jl`
last. `fvfd/FVFD.jl` now includes `operators_2d_grad_div_laplacian.jl` (its
`FVFD_BC_*` fallback guard no-ops against `specs.jl`, identical values).
`SparseArrays` (stdlib) was added to Project.toml `[deps]`/`[compat]` — it is
used by every `src/solve/` file.

## Writes to

Nothing persistent: `solve` returns a fresh `IncNSSolution`; no globals, no
registry, no I/O. The wrapped driver result is stored verbatim (same arrays by
reference, `sample(sol, :u) === sol.result.u`).

## Backend constraints

CPU is the registered path (drivers default `backend = CPU()` /
`backend_ka = KernelAbstractions.CPU()`). NOT included in the package:
`src/solve/linear_solve_cuda.jl` and `src/methods/inc_ns/cavity_mg_cuda.jl` —
CUDSS is not a dependency; GPU jobs `include` them manually after
`using CUDA, CUDSS` (HPC pattern unchanged by registration).

## Failure modes

- **Double-include of the seam**: adding `include("solve/linear_solve.jl")` to
  `Kraken.jl` redefines the seam (no self-guard) — keep it reachable only via
  `poisson.jl`'s guarded tail-include.
- **Capability over-claim**: declaring `GPUExecution` on `IncNS` would lie —
  the platform would dispatch GPU paths that need manual-load CUDA files.
- **Unqualified helpers in tests**: `l2_error`, `cell_ij`, `FVFD_BC_*` etc. are
  NOT exported; migrated tests must `Kraken.`-qualify them (the old file-local
  includes are skipped by their `isdefined` guards once Kraken is loaded).
- **Driver kwargs drift**: `solve(params, IncNS(...))` splats blindly — an
  unknown key throws `MethodError` at the driver, which is the intended
  fail-loud behaviour (no silent filtering).

## Touch order

1. `src/Kraken.jl` — the include block + export block (any registration issue).
2. `src/methods/inc_ns/method.jl` — wrapper types, driver dispatch table.
3. `test/platform/incns_contract_test.jl` — bit-identical wrapper parity gate.
4. `test/runtests.jl` — the "IncNS + solve services" testset (heavy cases
   gated behind `KRAKEN_TEST_HEAVY=true`).
5. `docs/agent/platform-implication.md` — the contract card (IncNS is listed
   there as the second concrete method).
