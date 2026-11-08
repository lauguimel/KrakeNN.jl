---
module: platform-residual-vjp
path: src/platform/residual.jl
owner_concern: platform-residual-seam
status: phase-2a
last_verified: 2026-06-12
depends_on: [platform, ad-step, ad-thermal-step, ad-ve-step]
---

# platform-residual-vjp — module implication map

Phase 2a of the Kraken platform contract: exposes `residual` and `adjoint_vjp`
as thin delegations to the validated CPU-Float64 AD seams. Zero new numerics.

## Public surface

- `LBMGeomParams` — Newtonian parameter bundle (q_wall, is_solid, u_profile, rho_out, s_plus, s_minus, Nx, Ny).
- `LBMThermalParams` — Thermal (Boussinesq) parameter bundle (q_wall, params::ADNatconvParams, Nx, Ny).
- `LBMVEParams` — Viscoelastic (Oldroyd-B) parameter bundle (g::ADVEEmbeddedGeom, q_wall, u_profile, p::ADVECoupledParams).
- `SteadyResidual` — new `Capability` enum value; declared in `capabilities(::LBM)`.
- `residual(problem, ::LBM, u, p::T)` — steady residual R = u - G(u,p); three dispatches (T in {LBMGeomParams, LBMThermalParams, LBMVEParams}). Enzyme-free.
- `adjoint_vjp(problem, ::LBM, u_star, p::T, v)` — (I - dG/du)^T v; three dispatches. Requires Enzyme loaded (delegates to private `_ad_*_vjp_GtT` stubs).

## Reads from

- `src/ad/ad_step.jl`: `ad_step!` (Newtonian one-step operator G).
- `src/ad/ad_thermal_step.jl`: `ad_thermal_cut_step!` (thermal coupled G).
- `src/ad/ad_ve_step.jl`: `ad_ve_coupled_step!` (VE coupled G).
- `src/ad/ad_api.jl`: `_ad_vjp_GtT`, `_ad_thermal_vjp_GtT`, `_ad_ve_vjp_GtT` (Enzyme stubs; throw without Enzyme).
- `ext/KrakeNNADExt.jl`: provides the Enzyme implementations of the VJP stubs (loaded as a weak dependency).
- `src/platform/contract.jl`: `AbstractMethod`, `Capability`, `LBM`.

## Writes to

Nothing persistent. `residual` and `adjoint_vjp` allocate and return fresh arrays; no mutation of `problem`, `u`, or `p`. The param structs are immutable.

## Backend constraints

CPU-Float64 only at this rung. `ad_step!`, `ad_thermal_cut_step!`, and `ad_ve_coupled_step!` are plain-Julia CPU operators (Enzyme-tapeable). GPU arrays passed to these dispatches will fail at the `Array{Float64}` method narrowing; the fallback error from the generic dispatch is intentional.

## Failure modes

- `adjoint_vjp` without Enzyme loaded: the `_ad_*_vjp_GtT` stubs throw `"Load Enzyme to enable AD: using Enzyme"` — propagated up by design.
- `residual` / `adjoint_vjp` for unsupported `(method, p)` pair: falls through to the generic fallback which throws `error("residual not implemented for ...")`.
- Bit-exact parity: `adjoint_vjp` is a thin delegation; any argument ordering mismatch in the param struct would show as a nonzero `delta` in the exit criterion (plan §3 Criterion 2).

## Touch order

1. `src/platform/residual.jl` — param structs + dispatches.
2. `src/platform/contract.jl` — +1 line `SteadyResidual` to `@enum Capability`.
3. `src/platform/solution.jl` — +1 `SteadyResidual` to `capabilities(::LBM)`.
4. `src/KrakeNN.jl` — include + export.
5. `test/platform/residual_vjp_test.jl` — Enzyme-free + Enzyme-gated tests.
6. `test/runtests.jl` — +1 include.

## Phase 2b-1 — ν-channel VJP seam (M-P2b-1, 2026-06-12)

### New public symbol
- `LBMScalarParams` — parameter bundle with FREE scalar ν (geometry fields identical to `LBMGeomParams`; `s_plus`/`s_minus` derived at construction from ν via `ad_trt_rates_inline`).
- `residual(_, ::LBM, f, p::LBMScalarParams)` — same body as `LBMGeomParams` dispatch; uses p.s_plus/s_minus (derived from ν).
- `adjoint_vjp(_, ::LBM, f_star, p::LBMScalarParams, v)` — same body as `LBMGeomParams` dispatch; delegates to `_ad_vjp_GtT` unchanged.

### New private symbols
- `ad_step_nu!(out, f, q_wall, is_solid, u_profile, rho_out, ν, Nx, Ny)` — Enzyme-diffable wrapper: calls `ad_trt_rates_inline(ν)` then `ad_step!`. Target for `Active(ν)` Enzyme differentiation.
- `_ad_pvjp_nu(f_star, lambda, p::LBMScalarParams) -> Float64` — Enzyme Reverse over `ad_step_nu!` with `Active(ν)`; returns dL/dν scalar cotangent. Impl in `ext/KrakeNNADExt.jl`.

### ν chain
- ν → (s_plus, s_minus) via `ad_trt_rates_inline` (exact inline formula, no residual).
- dL/dν = Enzyme reverse of `ad_step_nu!` wrt `Active(ν)` with λ as seed on `out`.

### Files modified (additive only)
- `src/platform/residual.jl` — +LBMScalarParams struct, +2 dispatches.
- `src/ad/ad_step.jl` — +`ad_step_nu!` wrapper.
- `src/ad/ad_api.jl` — +`_ad_pvjp_nu` stub.
- `ext/KrakeNNADExt.jl` — +`_ad_pvjp_nu` impl.
- `src/KrakeNN.jl` — +export LBMScalarParams.

## Phase 2c-1 — Field ν(y) channel (M-P2c-1, 2026-06-13)

### New public symbols
- `LBMFieldParams` — parameter bundle with FREE per-row ν(y) field (Vector{Float64}, length Ny);
  `s_plus_field`/`s_minus_field` derived at construction. Construct: `LBMFieldParams(geom, nu_field)`.
- `residual(_, ::LBM, f, p::LBMFieldParams)` — calls `ad_step_nufield!`; uniform nu_field degenerates
  to `LBMScalarParams` result bit-exactly.
- `adjoint_vjp(_, ::LBM, f_star, p::LBMFieldParams, v)` — delegates to `_ad_vjp_GtT_nufield`
  (Const nu_field, exact state linearization).

### New private symbols
- `ad_bulk_nufield!(out, f, q_wall, is_solid, nu_field, Nx, Ny)` — per-row TRT bulk; rates from
  `ad_trt_rates_inline(nu_field[j])` inside the j-loop.
- `ad_apply_zou_he_rebuild_nufield!(out, f, u_profile, rho_out, nu_field, Nx, Ny)` — per-row Zou-He
  rebuild with per-row rates.
- `ad_step_nufield!(out, f, q_wall, is_solid, u_profile, rho_out, nu_field, Nx, Ny)` — composition
  of bulk + Zou-He nufield variants. Enzyme-diffable wrt `nu_field` (Duplicated array).
- `ad_forward_solve_nufield(; ...)` — INTERNAL forward solver (same convergence loop as
  `ad_forward_solve` but calls `ad_step_nufield!`). Not exported.
- `_ad_pvjp_nufield(f_star, lambda, nu_field, ...) -> Vector{Float64}` — Enzyme Reverse with
  `Duplicated(nu_field, dnu)`; returns length-Ny field cotangent in one Enzyme call (O(1) not O(Ny)).
- `_ad_vjp_GtT_nufield(f_star, v, ..., nu_field, ...) -> Array{Float64,3}` — state VJP with
  `Const(nu_field)`. Exact (not mean-ν approximation).

### ν-field chain
- nu_field[j] → `ad_trt_rates_inline(nu_field[j])` per row → `ad_step_nufield!` → dL/dnu[j] via
  Enzyme Duplicated array cotangent accumulation (one pass for all j).
- State VJP: `_ad_vjp_GtT_nufield` with Const(nu_field) — correct Jacobian ∂G/∂f at the current
  nu_field, not mean-ν approximation.

### Files modified (additive only)
- `src/platform/residual.jl` — +`LBMFieldParams` struct, +2 dispatches.
- `src/ad/ad_step.jl` — +`ad_bulk_nufield!`, +`ad_apply_zou_he_rebuild_nufield!`, +`ad_step_nufield!`.
- `src/ad/ad_forward.jl` — +`ad_forward_solve_nufield` (INTERNAL).
- `src/ad/ad_api.jl` — +`_ad_pvjp_nufield` stub, +`_ad_vjp_GtT_nufield` stub.
- `ext/KrakeNNADExt.jl` — +`_ad_pvjp_nufield` impl, +`_ad_vjp_GtT_nufield` impl, +imports.
- `src/KrakeNN.jl` — +export `LBMFieldParams`.
