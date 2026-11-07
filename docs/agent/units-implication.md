---
module: units
path: src/units/
owner_concern: lu-nondim-conversion
status: implemented
last_verified: 2026-05-31
depends_on:
  - geometry
  - io-krk
  - lbm
---

# units — module implication map

The `KrakeNN.Units` submodule is the **single owner** of every conversion between
real units ⇄ non-dimensional numbers (Re, Wi, β, Pr, Ra, Ma) ⇄ lattice units
(τ, ν, u, λ, R_LU). It is compile-time-only and exists to terminate the chronic
hand-coded-LU rabbit-holes (M48 / M59–M61). Full design contract:
`docs/spec/units-v1.md` (KRK-UNITS-001). This map is the reference example for
the `docs/spec/llm-implication-map-v1.md` format.

## Public surface

The module exports nothing into `Base`; the bare names are reserved, so callers
use `KrakeNN.Units.<name>` (or `using KrakeNN.Units: compile`). `Kraken.jl`
re-exports only the `Units` submodule name itself.

- `Units.compile(; physics::Symbol, geometry, bc, refinement=nothing, backend=CPU(), T=Float64, strict=true, kwargs...) -> SimulationPlan{T}` — FORWARD path: non-dim numbers → validated plan. Strict by default (fatal/error issues raise).
- `Units.audit(driver_kw::NamedTuple; physics::Symbol, geometry, bc, backend=CPU(), T=Float64, strict=false, kwargs...) -> SimulationPlan{T}` — REVERSE/forensic path: raw LU driver kwargs → reconstructed + audited plan. Lenient by default (issues collected, not raised).
- `Units.driver_kwargs(plan::SimulationPlan{T}) -> DriverKwargs{T}` — lowers a plan to a splat-ready `NamedTuple` for an existing driver. Concrete only for Newtonian + Viscoelastic; other specs raise `NotImplementedError`.
- `Units.report(plan; io=stdout, format=:markdown) -> Union{Nothing,String}` — human (`:markdown`) or machine (`:jsonl`) provenance of a plan. `io=nothing` returns the string.
- Extension API (called by physics files at include time): `Units.register_physics!(sym, ::Type{<:AbstractPhysicsSpec})`, `Units.register_stability!(::Type{<:AbstractWallBC}, ::Type{<:AbstractPhysicsSpec}, pred)`, `Units.register_bc_combo!(key::NTuple{4,Symbol}, status)`.
- `.krk` binding: `Units.load_units_krk(path)`, `Units.parse_units_krk(text)`, `Units.plan_from_krk(text; name=…)` — parse a `Plan{…}` mega-block or `Define … from_nondim` / `from_plan` cross-reference into `SimulationPlan`(s).
- Public types: `AbstractPhysicsSpec` and its subtypes `NewtonianSpec{T}`, `ViscoelasticSpec{T}`, `ThermalBoussinesqSpec{T}` (concrete); `PowerLawSpec`, `MultiphaseSpec`, `MHDSpec` (raising stubs). Payload types `LBMUnits{T}`, `SimulationPlan{T}`, `GeometryDescriptor`, `BCConfig`, `DiscretizationConfig`, `Issue`. Wall-BC tags `HalfwayBB`, `BouzidiFL`, `LiBBV2`, `MeiBouzidi`. Exceptions `PlanValidationError`, `NotImplementedError`.

## Reads from

The module reads, never mutates, types/concepts owned by sibling modules. As of
M1 several sibling dirs are net-new, so Phase 1 accepts a **NamedTuple fallback**
and normalises it internally (`_normalize_geometry` / `_normalize_bc` in
`Units.jl`).

- `geometry` (`src/geometry/`) — a geometry descriptor (type, blockage, L_up,
  L_down, q_wall histogram, kappa_max, STL hash). Consumed read-only via the
  duck-typed `GeometryDescriptor`; the real `GeometryDescriptor` lives in
  `src/units/Units.jl` and is fed by `compile(geometry=…)`.
- `io-krk` (`src/io/`) — the `.krk` text / parse tree. `krk_binding.jl` reads it
  to build plans (mega-block and cross-reference syntaxes).
- `lbm` (`src/lattice/`) — lattice topology constants (cs², the `sqrt(3)` factor
  in `Ma = u_LU·√3`). Used read-only in the Mach-number math.

## Writes to

- **Returns** an immutable `SimulationPlan{T}` (and `LBMUnits{T}` inside it).
  Mutates none of its arguments.
- **Mutates module-global registries at include/extension time only**:
  `PHYSICS_REGISTRY`, `STABILITY_REGISTRY`, `WALL_BC_REGISTRY`,
  `BC_COMPATIBILITY` (via the `register_*!` helpers), and the compile-time
  memo `STL_AUDIT_CACHE` (keyed by `stl_hash`). These mutate during module load
  or when a Phase-2 physics file registers itself — never per simulation step.
- **Side effects**: `:warn`-severity issues are emitted via `Logging.@warn`
  (`emit_warning_logs`) and also stored in `plan.warnings`. `report(io=…)`
  prints to `io`. No files are written by the module itself.
- **Mutates NOTHING in the time loop**: `driver_kwargs(plan)` materialises a
  `NamedTuple` once before the loop; the module is never called per timestep.

## Backend constraints

- **Compile-time only / GPU-irrelevant.** All arithmetic runs once at plan
  construction. `SimulationPlan` is immutable; `driver_kwargs(plan)` is called
  once before the time loop. Nothing in `units/` enters a GPU kernel.
- **No per-step allocation.** The module produces plain Julia scalars/NamedTuples
  on the host; the `backend` kwarg (default `KernelAbstractions.CPU()`) is
  recorded for downstream use but the conversion math is backend-agnostic.
- **Float32 caveat is enforced, not silent**: `intrinsic_unit_issues` raises a
  fatal `:tau_float32_floor` when `T===Float32` and `tau_hydro < 0.6`. This is
  the only `T`-dependent guardrail.
- The only type-instability point is the abstract `physics_spec::AbstractPhysicsSpec`
  field of `SimulationPlan`; drivers cross it via a function barrier that
  re-dispatches on the concrete spec type (per the spec §2.3).

## Failure modes

This module exists to PREVENT the historical hand-coded-LU rabbit-holes. Each
guardrail below cites the receipt it encodes:

- **M48 silent fixture toggle** — a hand-edited `embedded_gradient=true` flip in
  one VE fixture diverged its numerics from its siblings, burning ~6 missions
  before the JSONL grep found it. Encoded: `_shared_validation_issues` fires a
  `:m48_toggle` warn when a `ViscoelasticSpec` plan has `embedded_gradient=true`.
- **M59–M61 acoustic-scaling U-shape** — at fixed Re, sweeping R_LU under
  *acoustic* scaling drifts τ out of the TRT window and produces a non-monotone
  Cd "U-shape" artifact (NOT a bug; 3 sessions lost). Encoded two ways:
  (a) `_resolve_scaling` auto-selects `:diffusive` when an R-sweep is detected
  (`_is_r_sweep`), else `:acoustic`; (b) `intrinsic_unit_issues` fires
  `:tau_above_magic` (warn) for τ>1.2 — `audit((u_mean=0.005,…); R_LU=50)`
  reconstructs τ=1.25 and retro-detects the artifact.
- **TRT magic-window τ** — `intrinsic_unit_issues` raises fatal
  `:tau_below_trt_window` / `:tau_above_trt_window` outside `0.55 ≤ τ ≤ 1.5`,
  plus the `_halfway_pred` / `_bouzidi_ve_pred` stability predicates duplicate
  the bound per wall-BC. `:tau_above_magic` warns in the high-τ audit band.
- **F32 floor** — fatal `:tau_float32_floor` if `T===Float32 && τ<0.6` (Float32
  collision-rate precision floor; see Metal F32 R-drift findings).
- **Unknown physics / kwargs**: unknown `physics` symbol → immediate
  `ArgumentError` listing registered keys; unknown kwargs → `:unknown_keyword`
  (error under strict `compile`, warn under lenient `audit`).
- **Stub physics** (`:power_law`, `:multiphase`, `:mhd`, and pre-Phase-2
  `:thermal_boussinesq`) raise `NotImplementedError` from `_build_spec` /
  `_compile_with_spec` — pointing at units-v1.md §7. (Thermal-Boussinesq is now
  concrete; the other three remain stubs.)
- **`from_plan` override hazard**: if a `.krk` `from_plan` block hand-codes a
  planner-owned key (`u_mean`, `tau`, `nu_*`, `lambda`, `max_steps`, …), the
  override is IGNORED and a `:planner_override` warn is emitted — the
  planner-owns-LU guardrail in DSL form.

## Touch order

For a units bug (wrong τ/ν/u, a missing/spurious issue, a `.krk` parse glitch),
inspect in this order:

1. `src/units/lattice_units.jl` — the core algebra (`nondim_to_lu`,
   `lu_to_nondim`, `_resolve_scaling`, `_viscosity_factor`, `_max_steps`,
   `intrinsic_unit_issues`). 80% of "wrong number" bugs are here.
2. `src/units/physics/<physics>.jl` — the physics-specific `_build_spec` /
   `_compile_with_spec` / `lu_to_nondim` (e.g. `viscoelastic.jl` for the
   nu_s/nu_p/lambda split, `thermal.jl` for alpha/beta_thermal).
3. `src/units/Units.jl` — the facade: type definitions, `compile`/`audit`
   orchestration, `_normalize_geometry`/`_normalize_bc`/`_normalize_discretization`,
   `driver_kwargs`, and the include order (a stub firing unexpectedly is usually
   an include-order / registry issue).
4. `src/units/stability_cone.jl` — wrong/missing stability issue: check the
   `STABILITY_REGISTRY` entry and the per-wall predicate (`_halfway_pred`,
   `_bouzidi_ve_pred`, `_thermal_halfway_pred`).
5. `src/units/stl_audit.jl` + `src/units/bc_consistency.jl` — for STL/BC-combo
   issues (`audit_stl`, `check_bc_consistency`, `BC_COMPATIBILITY`).
6. `src/units/krk_binding.jl` — only for `.krk`-sourced bugs (block extraction,
   `from_plan` merge, `:planner_override` warn).
7. `src/units/audit_trail.jl` + `report.jl` — `Issue` severity/sorting and
   report rendering; rarely the root cause but where issue presentation lives.
