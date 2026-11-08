---
module: physics-thermal
path: src/kernels/thermal_2d.jl
owner_concern: constitutive
status: implemented
last_verified: 2026-05-31
depends_on:
  - lbm
  - bc
  - units
---

# physics-thermal — module implication map

The thermal physics module is the **single owner** of the Boussinesq
double-distribution-function (DDF) coupling: a passive temperature scalar tracked
by a second D2Q9/D3Q19 population `g` whose macroscopic moment `T = Σ g_q` feeds a
per-node buoyancy body force `Fy = β_g·(T − T_ref)` back into the flow population
`f` via Guo forcing. It owns the thermal collision (`ω_T = 1/(3α + 0.5)`), the
anti-bounce-back Dirichlet temperature walls, and the fused single-launch natural
convection step. Validated M7: de Vahl Davis differentially-heated cavity Ra=1e3
Nu ≈ 1.118 within <10% at N=64 (`test/test_thermal.jl`). The compile-time
non-dim ⇄ LU half lives in `KrakeNN.Units` (`ThermalBoussinesqSpec`).

## Public surface

Exported into `Kraken` (driver entry points, see `src/KrakeNN.jl`):

- `run_natural_convection_2d(; N, Ra, Pr, Rc=1.0, T_hot, T_cold, max_steps, backend, FT) -> NamedTuple` — differentially-heated square cavity (hot west / cold east, adiabatic top/bottom). Uses the fused kernel; returns `(ρ, ux, uy, Temp, Nu, …)`. `Rc>1` switches to temperature-dependent viscosity.
- `run_rayleigh_benard_2d(; Nx, Ny, Ra, Pr, T_hot, T_cold, max_steps, backend, FT) -> NamedTuple` — hot-bottom / cold-top, periodic-x; uses the unfused step kernels.
- `run_natural_convection_3d(; N, Ra, Pr, T_hot, T_cold, max_steps, backend, FT) -> NamedTuple` — D3Q19 cube, Boussinesq gravity in y. Reference Fusegi et al. (1991).
- `run_natural_convection_refined_2d(; N, Ra, Pr, Rc, T_hot, T_cold, max_steps, wall_fraction, ratio, backend, FT) -> NamedTuple` — same physics with wall-refined patches; Nu read from the fine west patch.

Kernel-level surface (de-facto public, called by the drivers and by refinement code):

- `collide_thermal_2d!(g, ux, uy, ω_T)` / `collide_thermal_masked_2d!(g, ux, uy, ω_T, is_skip)` — BGK on the temperature populations (masked variant skips patch-covered cells).
- `collide_boussinesq_2d!(f, Temp, is_solid, ω, β_g, T_ref)` — flow BGK + Guo buoyancy force, constant viscosity.
- `collide_boussinesq_vt_2d!(f, Temp, is_solid, ν_ref, T_ref, A_arr, β_g)` — Arrhenius `ν(T)=ν_ref·exp(A·(1/T − 1/T_ref))`.
- `collide_boussinesq_vt_modified_2d!(f, Temp, is_solid, ν_ref, T0_visc, α_visc, β_g, T_ref_buoy)` — Frank-Kamenetskii `ν(T)=ν_ref·exp(α·(T − T0))`, `α_visc = ln(Rc)`.
- `compute_temperature_2d!(Temp, g)` — moment `T = Σ g_q`.
- `apply_fixed_temp_{south,north,west,east}_2d!(g, T_wall, …)` — anti-bounce-back Dirichlet temperature walls.
- `fused_natconv_step!` / `fused_natconv_vt_step!` (`fused_thermal_2d.jl`) — one GPU launch per step: stream + thermal BC + macroscopic + both collisions.
- 3D mirrors: `collide_thermal_3d!`, `collide_boussinesq_3d!`, `compute_temperature_3d!`, `apply_fixed_temp_{west,east}_3d!` (`thermal_3d.jl`).
- Units side: `KrakeNN.Units.ThermalBoussinesqSpec{T}(Re, Pr, Ra)` and its `nondim_to_lu` / `lu_to_nondim` (`src/units/physics/thermal.jl`); registered under physics symbol `:thermal_boussinesq` (params `Re, Pr, Ra`).

## Reads from

- `lbm` — the flow population `f`, the `feq_2d(Val(q), ρ, ux, uy, usq)` equilibrium, `compute_macroscopic_2d!`/`_3d!`, the stream kernels (`stream_periodic_x_wall_y_2d!`, `stream_3d!`, the inline `stream_pull_node`), `bounce_back_2d!`, and the `D2Q9`/`D3Q19` weights/lattice. Read-only; the buoyancy collision wraps the standard `feq`.
- `bc` — flow no-slip via `apply_bounce_back_wall_2d!(f, Nx, Ny, :side)` and `is_solid`. The thermal Dirichlet walls are owned here, but the flow-side wall BCs are consumed from the BC module (especially in the refined driver).
- `units` — at plan time only: `ThermalBoussinesqSpec` converts `(Re, Pr, Ra)` → `(ν, α=ν/Pr, β_thermal=Ra·ν·α/R³)`; the thermal stability predicate `_thermal_halfway_pred` lives in `src/units/physics/thermal.jl`. The drivers themselves currently hard-code their own `ν=0.05`, `α=ν/Pr`, `β_g=Ra·ν·α/(ΔT·H³)` rather than calling `Units.compile` (the units half is the v0.3 contract, the drivers predate it).

## Writes to

- **Mutates the temperature populations `g` in place** (collision + anti-bounce-back BC) and the temperature field `Temp[i,j]` (moment recovery). The fused kernels write into the swapped `g_out`/`f_out` and set `Temp` per node.
- **Mutates the flow populations `f` in place** through the Boussinesq collision (adds the Guo force term to every direction), and via `bounce_back_2d!` for `is_solid` cells.
- **Allocates per run** (not per step): `g_in`, `g_out`, `Temp`, `is_skip`, the CPU staging `g_cpu` and `Nu_local`/`Nu_arr` arrays. Drivers return host `Array` copies plus the scalar `Nu`.
- **Mutates the global units registry at include time**: `register_stability!(HalfwayBB, ThermalBoussinesqSpec, _thermal_halfway_pred)` and `register_bc_combo!((:velocity_parabolic, :zou_he_pressure, :temperature_dirichlet, :temperature_dirichlet), :ok)`.
- No files written; `@info` lines report Nu in the test runner.

## Backend constraints

- **GPU-clean KernelAbstractions.** Every collision/BC/moment is a `@kernel`; the
  drivers select the backend from `KernelAbstractions.get_backend(f)` and convert
  scalars to `eltype(f)` before launch. The fused kernel is the production path:
  one launch per step (stream + BC + macroscopic + both collisions inlined),
  preferred over the unfused `run_rayleigh_benard_2d` sequence (~6 launches/step).
- **No allocation inside the hot kernels** — populations are read into 9 locals,
  the `g_eq`/`Sq` terms are computed register-resident, results written back.
- **`@Const`-tagged inputs** (`ux`, `uy`, `is_solid`, `f_in`, `Temp`, `is_skip`)
  for read-only GPU aliasing safety.
- **Float32 caveat**: `collide_boussinesq_vt_2d!` uses Arrhenius `exp(A·(1/T − 1/T_ref))` —
  near `T→0` (cold wall, `T_cold=0`) `1/T` blows up, so the *modified* Frank-Kamenetskii
  form `exp(α·(T − T0))` is the safe default for `Rc>1` and is what the cavity driver wires.
- **Per-step cost**: O(Nx·Ny) (2D) / O(Nx·Ny·Nz·19) (3D), doubled vs isothermal flow because both `f` and `g` are streamed and collided.

## Failure modes

- **Refined-patch 50× velocity blowup (resolved 587f3a5, 2026-04-12)** — two
  bugs in `run_natural_convection_refined_2d`: (1) ghost buffers sized to
  `length(parent_range)+2` while `save_coarse_state!` filled fewer columns at
  domain-edge patches → trailing zeros read as ρ≈0 → explosive u from step 1;
  (2) fine-grid Boussinesq used `β_g·ratio` instead of `β_g/ratio` (acoustic
  scaling gives a_fine = a_coarse/ratio, an r²=4 error at ratio=2). Receipt:
  memory `project_refined_natconv_debug`.
- **Patch interiors never initialised from coarse (fixed aa5b471)** —
  `prolongate_f_rescaled_*` and the thermal ghost-fill kernels only filled GHOST
  cells, so patch interiors stayed at uniform `T_init=0.5` / rest and then
  polluted the coarse grid via restriction. Fix: `prolongate_f_rescaled_full_2d!`
  + `fill_thermal_full!` at init (see the driver's init loop over `domain.patches`).
- **Ghost-timing off-by-one (flow side still open)** — at `sub_step=1` the
  thermal ghost fill must read `g_prev` (state at n), not `g_in` which is already
  at n+1 after the fused base step. Fixed thermal-side; the same hazard on the
  flow side (`fill_ghost_from_coarse!` reading post-step `f_in`) is documented but
  not closed (memory `project_thermal_refinement_status`).
- **Coarse Nusselt under-resolves the wall gradient (~20%)** — after restriction
  the coarse grid carries block-averaged temperatures that smear the wall
  gradient; `run_natural_convection_refined_2d` therefore computes Nu from the
  FINE west-patch field (`thermals[1].Temp`), not the coarse `Temp`. Forget this
  and Nu reads ~20% low.
- **Equilibrium drops the u² term** — the thermal `g_eq = w_q·T·(1 + 3·c_q·u)` is
  the linearised passive-scalar form (no `O(u²)` terms, by design). Do NOT copy
  the flow `feq_2d` here; the scalar transport is only first-order accurate in
  velocity and that is intentional.
- **Anti-bounce-back wall direction is hand-indexed per wall** — each
  `apply_fixed_temp_*` hard-codes exactly the three unknown populations pointing
  INTO the domain (e.g. south wall fills q=3,6,7). A wrong opposite-index here
  silently injects/leaks heat at the wall. The fused `apply_thermal_bc_cavity`
  duplicates this logic inline for west/east; keep the two in sync.
- **Units stability gate** — `_thermal_halfway_pred` raises fatal
  `:thermal_alpha_nonpositive` / `:thermal_beta_nonpositive` and, for the scalar
  relaxation, `:thermal_tau_below_floor` (τ_thermal<0.55) / `:thermal_tau_above_ceiling`
  (>1.5). A diverging-T run with a "valid" τ_hydro is usually τ_thermal = 0.5+3α
  out of window, not a kernel bug.
- **`:thermal_boussinesq` was a raising stub** — the parameterless
  `_compile_with_spec(::ThermalBoussinesqSpec)` still throws `phase_stub_error`;
  only the full 7-arg method is concrete. Calling the wrong arity raises a
  `NotImplementedError` pointing at units-v1.md §7.

## Touch order

For a thermal bug (wrong Nu, T leaking/exploding at a wall, buoyancy too
strong/weak, refined-patch divergence), inspect in this order:

1. `src/kernels/thermal_2d.jl` — the unfused reference kernels: `collide_thermal_2d!`,
   `collide_boussinesq_2d!`, the anti-bounce-back `apply_fixed_temp_*`. 80% of
   "wrong wall temperature" and "buoyancy sign/magnitude" bugs are here. Check the
   Guo `Sq` terms and the per-wall unknown-population indices first.
2. `src/kernels/fused_thermal_2d.jl` — the PRODUCTION path. If a uniform-grid run
   misbehaves, the bug is in `apply_thermal_bc_cavity`, `macroscopic_boussinesq`
   (note `fy/2` Guo half-force), or `collide_thermal_node` — not in (1).
3. `src/drivers/thermal.jl` — parameter assembly (`β_g = Ra·ν·α/(ΔT·H³)`,
   `ω_T = 1/(3α+0.5)`, `α_visc = -log(Rc)`), initialisation profile, and the Nu
   finite-difference. Wrong-magnitude Nu with correct fields → the FD or β_g here.
4. `src/refinement/thermal_refinement.jl` + `src/kernels/refinement_exchange_2d.jl`
   — only for `run_natural_convection_refined_2d`: ghost buffer sizing, `β_g/ratio`
   rescale, `fill_thermal_full!` init, ghost timing (`g_prev` vs `g_in`).
5. `src/units/physics/thermal.jl` — for a spurious/missing stability issue or a
   bad `(Re,Pr,Ra)→(ν,α,β)` conversion: `nondim_to_lu`, `lu_to_nondim`,
   `_thermal_halfway_pred`, and the `register_stability!`/`register_bc_combo!` calls.
6. `src/kernels/thermal_3d.jl` — the D3Q19 analogues for `run_natural_convection_3d`
   (Fusegi reference); mirror the 2D checks (opposite-index pairs, gravity-in-y).
