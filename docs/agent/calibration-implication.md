---
module: platform/calibration
path: src/platform/calibration.jl
owner_concern: steady-calibration-stack
status: phase-2b-2
last_verified: 2026-06-12
depends_on: [platform/residual.jl, ad/ad_api.jl, ad/ad_adjoint.jl, ad/ad_forward.jl, platform/observe.jl]
---

# platform/calibration — implication map

Implements the steady calibration stack: `ParameterSpace` / `loss` / `fit` / `CalibResult`.
Enzyme-free except for the adjoint gradient, which delegates to `_ad_pvjp_nu`,
`_ad_vjp_GtT`, `_ad_dJdf`, and `_ad_dqwall_terms` stubs.

## Public surface

Defined in `src/platform/calibration.jl`, re-exported by `Kraken`:

- `ParameterSpace(free, lb, ub; scale)` — declares the free parameter names, box
  bounds, and per-parameter scale (`:natural` or `:log`); `to_flat`/`from_flat`/`project!`
  are internal companions.
- `loss(preds, data) -> Float64` — weighted sum-of-squares mismatch between
  `Prediction`s and observations.
- `fit(problem, ::LBM, data, p0, pspace; observables, reg_weight=0.0, method=:pgd, kwargs...)`
  — projected BB+Armijo gradient descent (`:pgd`, default, dep-free) or
  `method=:lbfgs` via `ext/KrakeNNOptimExt.jl` (`Optim.Fminbox(LBFGS())`).
- `CalibResult` — `p_opt`, `loss_final`, `loss_trace`, `grad_trace`, `n_iter`,
  `converged`, `message`.

## Call graph: `fit`

```
fit(problem, ::LBM, data, p0, pspace; observables)
  ├─ to_flat(pspace, p0) → x0
  ├─ iteration k:
  │   ├─ from_flat(pspace, xk, p0) → p_named
  │   ├─ ad_forward_solve(; problem..., nu=νk) → fwd             [EXISTING]
  │   ├─ _obs_lineprofile_ux_from_f(fwd.f_star, obs) → preds     [NEW local]
  │   ├─ loss(preds, data) → L                                   [NEW]
  │   ├─ _dJ_df_lineprofile_ux(...) or _ad_dJdf(...) → dL/df      [NEW / EXISTING]
  │   ├─ gmres_adjoint(apply_GtT, dL/df) → λ                     [EXISTING]
  │   │     └─ apply_GtT = v -> _ad_vjp_GtT(fwd.f_star, v, ...)   [EXISTING ext]
  │   ├─ _ad_pvjp_nu(fwd.f_star, λ, LBMScalarParams(..., νk)) → dL/dν
  │   └─ project!(pspace, xk - α∇L) → xk+1
  └─ return CalibResult(p_opt, loss_final, loss_trace, grad_trace, n_iter, converged, message)
```

## ν chain

ν → `(s_plus, s_minus)` → `G` → `L`. `_ad_pvjp_nu` differentiates the ν→G chain
in one Enzyme call over `ad_step_nu!`. The calibrated `fit` loop reconstructs
`LBMScalarParams(geom, ν)` after bound projection, so the bundle passed to
`_ad_pvjp_nu` matches the ν used by the forward solve.

## Geometry parity guarantee

For `pspace={:radius}`, `fit` calls `_ad_dqwall_terms` and
`ad_assemble_radius_terms`, the same chain used by `_steady_drag_sensitivity`.
For `FieldReduction(:Cd, identity)`, the direct Cd residual scales the explicit
q-wall and `D = 2R` terms, while λ carries the implicit state response.
Gate 4c pins the direct chain against `steady_shape_sensitivity` at rel < 1e-6.

## Optimizer design

Projected gradient descent + Armijo backtracking. No new Project.toml deps.
Upgrade path: add Optim.jl as a weak-dependency extension for L-BFGS in a later phase.

## Phase 2c-2 additions (M-P2c-2)

`_reg_loss(nu_vec, alpha)` and `_reg_grad(nu_vec, alpha)` implement Tikhonov
smoothness penalty `(α/2) ‖D·ν‖²` and its analytic gradient (discrete negative
Laplacian). Both are Enzyme-free and used by both the PGD and L-BFGS paths.

`_is_nufield_pspace(pspace)` detects whether pspace describes a ν-field (all free
names match `ν_\d+`). Used to route `forward_at`, `compute_gradient_flat`, and
`eval_at` between the scalar and field-ν paths.

`_extract_nufield(p_named, Ny)` extracts `[p_named[:ν_j] for j in 1:Ny]`.

`fit(...; reg_weight=0.0, method=:pgd)` — two new kwargs (backward-compatible
defaults). `method=:pgd` (default) is the existing BB+Armijo loop. `method=:lbfgs`
delegates to `_fit_lbfgs` in `ext/KrakeNNOptimExt.jl`; raises a documented error if
Optim is not loaded.

`ext/KrakeNNOptimExt.jl` — `_fit_lbfgs` uses `Optim.Fminbox(Optim.LBFGS())` with
box bounds in the optimizer's native (log-scale or natural) space. Cache-last-forward:
`compute_fg!` is called once per point; f and g are always consistent.

## Reads from

- `src/platform/residual.jl`: `LBMGeomParams`, `LBMScalarParams`, `LBMFieldParams`
- `src/ad/ad_api.jl`: `_ad_pvjp_nu`, `_ad_pvjp_nufield`, `_ad_vjp_GtT`, `_ad_vjp_GtT_nufield`, `_ad_dJdf`, `_ad_dqwall_terms`, `_fit_lbfgs`
- `src/ad/ad_adjoint.jl`: `gmres_adjoint`, `AD_LINEAR_RES_TOL`
- `src/ad/ad_forward.jl`: `ad_forward_solve`, `ad_forward_solve_nufield`
- `src/platform/observe.jl`: `Prediction`, `LineProfile`, `FieldReduction`
- `src/ad/ad_geometry.jl`: `dq_wall_dR_cylinder`, `ad_assemble_radius_terms`

## Writes to

Nothing persistent. `fit` allocates intermediate arrays per iteration and keeps no global state.

## Backend constraints

The whole calibration stack is CPU-Float64 by construction: every iteration calls
`ad_forward_solve`/`ad_forward_solve_nufield` and the Enzyme adjoint chain, which are
CPU-only AD paths (see `ad-implication.md`). No kernels are launched from this file;
GPU arrays are never accepted — bundles are rebuilt host-side each iteration.
`_reg_loss`/`_reg_grad` are plain-Julia dense loops, backend-irrelevant.

## Failure modes

- `method=:lbfgs` without Optim loaded raises the documented "load Optim.jl" error
  (weak-dep extension not triggered) — by design, not swallowed.
- Armijo backtracking can stall on a non-descent BB step: `fit` returns
  `converged=false` with the reason in `message` rather than erroring.
- A `pspace` whose free names match neither a known scalar/geometry name nor the
  `ν_\d+` field pattern routes to the scalar path and fails downstream in
  `from_flat` — check `_is_nufield_pspace` first when adding parameter natures.
- Enzyme adjoint stubs (`_ad_pvjp_nu` etc.) propagate their own "extension not
  loaded" errors when Enzyme is absent.

## Touch order

1. `src/platform/calibration.jl` — `ParameterSpace`/`loss`/`fit`/`CalibResult`,
   `_reg_loss`/`_reg_grad`, ν-field routing helpers.
2. `ext/KrakeNNOptimExt.jl` — `_fit_lbfgs` (only when the optimizer path changes).
3. `src/KrakeNN.jl` — export block (choke file; edits serialized on `dev/platform`).
4. `test/platform/calibration_test.jl` — twin experiments (scalar ν, sine ν(y) field)
   and Gate 4c geometry parity pin.
