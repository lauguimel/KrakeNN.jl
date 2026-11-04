---
module: ad-ve
path: src/ad/ad_ve_ops.jl; src/ad/ad_ve_step.jl; src/ad/ad_ve_forward.jl; src/ad/ad_ve_geometry.jl; ext/KrakenADExt.jl
owner_concern: lbm-operator
status: implemented
last_verified: 2026-06-05
depends_on:
  - ad
  - lbm
  - physics-viscoelastic
  - bc
  - geometry
  - io-krk
---

# ad-ve — module implication map

The `src/ad/ad_ve_*.jl` files own the steady viscoelastic shape-adjoint:
`d(Cd_polymer)/dR` for a residual-converged Oldroyd-B confined cylinder on the
coupled log-conformation FVFD + D2Q9 TRT/LI-BB path. This is the viscoelastic
analogue of the Newtonian `dCd/dR` / thermal `dNu/dL` track in
[`ad-implication.md`](ad-implication.md) and shares the same public entry point
`steady_shape_sensitivity` and the same `.krk` `Sensitivity { ... }` surface.
Core code is Enzyme-free; `ext/KrakenADExt.jl` supplies the reverse-mode seams
when the weak dependency `Enzyme` is loaded. The central maintenance contract is
that `ad_ve_ops.jl` + `ad_ve_step.jl` stay a bit-mirror of the production M8 VE
coupled step (`drivers/viscoelastic_logfv_coupled_step_2d.jl`), guarded by the
anti-drift QoI check.

## Public surface

- `steady_shape_sensitivity(; qoi=:polymer_drag, wrt=:radius, Nx, Ny, radius, cx, cy, Wi, beta=0.5, nu_p, nu_s, Fx_body, fwd_tol=1e-13, bc=:open, fd_check=false, ...)` — the VE branch of the exported AD API; returns the same NamedTuple shape as the Newtonian/thermal tracks: `gradient` (`d(Cd_polymer)/dR`), `qoi_value`, `value`, `solver`, `terms` (`explicit`, `state_response`, `gradient`, `bc`), `forward` info, and optional `fd_check`.
- `.krk` dispatch: a parsed `Sensitivity { qoi = polymer_drag, wrt = radius }` request is routed by `run_simulation(setup)` / `run_krk_sensitivity` to `steady_shape_sensitivity`; requires `Module viscoelastic`, D2Q9, and a `Rheology oldroyd_b { nu_s, nu_p, lambda }` block.
- Core implementation helpers are de-facto public as `Kraken.<name>` but not exported: `ad_ve_build_geom`, `ad_ve_build_circle_geom`, `ad_ve_build_matched_geom`, `ad_ve_build_wall_points`, `ad_ve_poiseuille_profile`, `ad_ve_initial_state`, `ad_ve_coupled_step!`, `ad_ve_forward_solve`, `ad_ve_J_fx`, `ad_ve_extract_tau`, `ad_ve_assemble_dGdR`, `ad_ve_build_dcircle_geom_dR`, `ad_ve_dJ_dR_geom_explicit`, `ad_ve_ungauged_adjoint`, `ad_ve_mass_gradient`, `ad_ve_antidrift_delta`, `ad_ve_fd_dCdpoly_dR`, plus the `ADVECoupledParams` / `ADVEGeom` / `ADVEEmbeddedGeom` / `ADVEWallPoint` structs.
- Extension seam methods imported by `KrakenADExt`: `_ad_ve_dJdw` (reverse of `ad_ve_J_fx` wrt `w`), `_ad_ve_vjp_GtT` (reverse of `ad_ve_coupled_step!`, the adjoint matvec `dG^T . v`), `_ad_ve_dGdR_jvp` (forward-JVP of `ad_ve_coupled_step!` seeded by `d(geom)/dR` + `dq_wall/dR`). In core they throw "Load Enzyme"; the extension replaces them with Enzyme passes.

## Reads from

- `ad` — the shared adjoint solver `ad_gmres_givens` / `ad_gauge_augmented_adjoint` / `ad_richardson_rhohat` and `ad_dot_arrays`, plus the `AD_LINEAR_RES_TOL` linear tolerance, reused by the VE branch.
- `lbm` / `physics-viscoelastic` — D2Q9 weights/directions/opposite pairs, the TRT rate convention (`ad_ve_trt_rates`), and the production M8 coupled algebra that `ad_ve_coupled_step!` mirrors: pull-stream + LI-BB cut links + TRT-Guo collide for `f`; embedded FVFD advection (MUSCL-Superbee, production Dirichlet edge BC), log-conformation constitutive substeps, polymer stress `tau_p = (nu_p/Wi)(C - I)`, and the embedded `div . tau_p` force for `psi`.
- `bc` — the fused west-velocity / east-pressure Zou-He rebuild algebra and the regularized TRT collide (`ad_ve_trt_collide_local`) mirrored from `src/bc/specs.jl`.
- `geometry` — `precompute_q_wall_cylinder` / `dq_wall_dR_cylinder` for the f-side cut links (node frame `(cx-0.5, cy-0.5)`) and the production `reconstruct_wall_link_value_2d` / `compute_polymeric_drag_2d` used by the QoI and the anti-drift check.
- `io-krk` — the parser-owned `Sensitivity` request plus the runner extraction of VE keywords (`cx`, `cy`, `Wi`, `beta`, `nu_p`, `nu_s`, `Fx_body`, optional `n_substeps`, `samples`, `dt`, `u_mean`, tolerances) from `Physics` / `Define` and the `Rheology oldroyd_b` block.
- `Project.toml` weakdep/extension metadata — `KrakenADExt = "Enzyme"` keeps Enzyme out of the unconditional `using Kraken` load path.

## Writes to

- Returns a fresh NamedTuple from `steady_shape_sensitivity`; it writes no files and mutates no global registries.
- Mutates local host arrays during the forward fixed-point solve (the stacked `w = (f, psi)` buffers `w_in`/`w_out`, length `12n`) and during the one-step AD products (`out`, cotangent and tangent arrays). These arrays are allocated inside the call or copied from inputs; `ad_ve_coupled_step!` heap-allocates all of its work arrays locally for Enzyme.
- `ext/KrakenADExt.jl` allocates cotangent/tangent work arrays for the reverse `dJ/dw` (psi-block only), the coupled `dG^T . v`, and the `dG/dR` forward-JVP seeded by the analytic geometry/`q_wall` shadows.
- Optional `fd_check=true` rebuilds the matched geometry at `R ± h`, re-runs two additional perturbed tight forwards, and returns their finite-difference data (`value`, `Jp`, `Jm`, `relerr`, `topo_fixed`); it does not cache results globally.

## Backend constraints

- CPU Float64 only. The public API does not accept a backend keyword, and `.krk` polymer-drag dispatch rejects non-D2Q9 / non-`viscoelastic` setups before any derivative work.
- No GPU kernel is differentiated. `ad_ve_coupled_step!` (+ `ad_ve_ops.jl`) is a plain-Julia, host-side, unfused mirror of the production fused VE path; production GPU forward runs still use the production drivers.
- Memory is O(1) in the number of forward steps because Enzyme tapes one coupled step at the converged state `w*`, not the transient history. GMRES still allocates host Krylov work arrays proportional to the `12n` state size and restart.
- Enzyme is optional at package load but mandatory at call time. Without `using Enzyme`, the core seam stubs (`_ad_ve_dJdw`, `_ad_ve_vjp_GtT`, `_ad_ve_dGdR_jvp`) throw before any derivative work starts.
- The open cylinder BC (west Zou-He velocity inlet + east Zou-He pressure outlet) pins the `rho = 1` mass mode, so `(I - dG^T)` is non-singular and the adjoint is solved UNGAUGED via `ad_ve_ungauged_adjoint` (`solver.gauge == :ungauged`). The mass-gauge path (`ad_ve_mass_gradient` + `ad_gauge_augmented_adjoint`) is reserved for the `:closed` / `:periodic` BC.
- `ρ_out` / `rho_out` is accepted for API parity but must be `1.0`: the coupled operator hardcodes the east-pressure ZouHe outlet density at production `1.0`.

## Failure modes

- **Forward not converged tightly enough** — the net `d(Cd_polymer)/dR` is a catastrophic cancellation between the explicit geometry partial and the implicit state response. The forward must reconverge to `fwd_tol=1e-13`; a looser `1e-11` floor poisons the finite-difference reference (agreement degrades from 0.63% to 53.3%, measured 2026-10-12 after the #67 TRT Guo forcing fix) even when the adjoint is exact. This is why `AD_VE_FWD_TOL = 1e-13` is the default in `ad_ve_forward_solve` and the API.
- **Finite-differenced `dG/dR`** — because of the same cancellation, `dG/dR` must be the analytic chain (`ad_ve_build_dcircle_geom_dR` + `dq_wall_dR_cylinder` through `_ad_ve_dGdR_jvp`). A central-FD `dG/dR` injects truncation noise into the cancelling terms; only the explicit `(partial J / partial R)|geom` term is allowed to be a finite difference, at a frozen state.
- **Bit-mirror drift** — any production change to the M8 VE coupled step (`drivers/viscoelastic_logfv_coupled_step_2d.jl`), the LI-BB cut-link branch, TRT rates / regularized TRT collide, the fused ZouHe rebuild, the embedded FVFD advection / wall-gradient / `div . tau_p`, or the log-conformation constitutive math must be reflected in `ad_ve_ops.jl` and `ad_ve_step.jl`. The anti-drift receipt is `ad_ve_antidrift_delta`: the inline QoI `ad_ve_J_fx` must equal the production `compute_polymeric_drag_2d` Fx on the same `(tau, q_wall)` to machine zero (<= 1e-12).
- **Differentiating the Boolean mask / invalid cut interval** — the cylinder `is_solid` mask (the LBM node mask via `ad_ve_build_matched_geom`) and the cut set are held constant; the derivative is valid only within a smooth cut-set interval. The `fd_check` `topo_fixed` flag (equal cut/solid counts across `±h`) guards this; finite differences crossing topology disagree by construction.
- **Wrong is_solid mask in the step** — `psi` is advected / differentiated on the LBM node mask, NOT the FVFD cell-fraction mask. `ad_ve_build_matched_geom` swaps in the LBM mask while keeping the FVFD fractions; using the raw FVFD mask reintroduces the original M8 forensic bug.
- **Weakdep seam not loaded** — calling the VE API after only `using Kraken` fails at the first Enzyme seam with the explicit "Load Enzyme" error.
- **Wrong gauge for the BC** — using the mass-gauged adjoint on the open cylinder (or the ungauged solve on closed/periodic) misgauges the `rho = 1` mass mode; `bc` selects the correct path (`:open` -> ungauged, `:closed`/`:periodic` -> mass-gauged).
- **Unsupported API symbols** — anything except `qoi=:polymer_drag, wrt=:radius` throws `ArgumentError`; `ρ_out != 1.0` throws; `bc` outside `(:open, :closed, :periodic)` throws. Do not document unvalidated pairs as working extension points. Validated envelope: `Wi <= 1`, `beta >= 0.5`.
- **Incomplete `.krk` setup** — `qoi=polymer_drag` dispatch requires `Module viscoelastic`, D2Q9, `cx`/`cy`, `Wi`, `Fx_body`, and an `oldroyd_b` rheology block with `nu_s`/`nu_p`; missing any of these fails before derivative work starts.

## Touch order

For a VE shape-adjoint bug, inspect in this order:

1. `src/ad/ad_api.jl` — `_steady_polymer_drag_sensitivity`: public keyword contract (`Nx, Ny, radius, cx, cy, Wi, beta, nu_p/ν_p, nu_s/ν_s, Fx_body, fwd_tol, bc, fd_check`), Enzyme-stub calls, `prefactor = nu_p/Wi`, gauge selection, gradient assembly (`explicit + state_response`), and the returned NamedTuple fields.
2. `src/ad/ad_ve_forward.jl` — `ad_ve_build_geom`, `ad_ve_build_wall_points`, `ad_ve_poiseuille_profile`, the tight `ad_ve_forward_solve` (`AD_VE_FWD_TOL = 1e-13`), the QoI `ad_ve_J_fx`, `ad_ve_dJ_dR_geom_explicit`, the `ad_ve_antidrift_delta` receipt, and the `ad_ve_fd_dCdpoly_dR` cross-check.
3. `src/ad/ad_ve_geometry.jl` — the analytic `d(geom)/dR` field derivatives (`ad_ve_d_vface_frac_dR`, `ad_ve_d_hface_frac_dR`, `ad_ve_d_cell_fraction_dR`, `ad_ve_build_dcircle_geom_dR`), `ad_ve_assemble_dGdR` (the FD-free `dG/dR` chain), `ad_ve_mass_gradient`, and the `ad_ve_ungauged_adjoint` open-BC solve.
4. `ext/KrakenADExt.jl` — the Enzyme reverse `_ad_ve_dJdw`, the coupled adjoint matvec `_ad_ve_vjp_GtT`, and the `_ad_ve_dGdR_jvp` forward-JVP seeded by the analytic shadows.
5. `src/ad/ad_ve_step.jl` — the unfused coupled `(f, psi)` bit-mirror; check the LI-BB cut links, TRT-Guo collide, the fused ZouHe rebuild, the LBM-mask advection, and the embedded `div . tau_p` against the production M8 step before changing constants.
6. `src/ad/ad_ve_ops.jl` — the shared FVFD embedded operators, log-conformation constitutive math, D2Q9 helpers, and the cut-cell circle builder (`ad_ve_build_circle_geom` / `ad_ve_build_matched_geom`); validate against the production FVFD / `logconformation_fv_2d` operators.
7. `.krk` path: `src/ad/ad_krk.jl` (`_krk_sensitivity_polymer_drag_kwargs`), `src/io/krk/directives.jl`, `src/io/krk/parser.jl`, `src/simulation_runner.jl` — only after the direct Julia API works but `Sensitivity { qoi = polymer_drag }` dispatch fails.
