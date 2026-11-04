---
module: physics-ehd
path: src/kernels/ehd_2d.jl
owner_concern: constitutive
status: implemented
last_verified: 2026-09-28
depends_on:
  - lbm
  - bc
  - platform/state
---

# physics-ehd — module implication map

The EHD physics module owns the electric potential and charge-density D2Q9
scalar populations for the Luo-Wu-Yi-Tan electroconvection model, plus the
CPU coupled canary that feeds Coulomb force density `F=qE` into the existing
Navier-Stokes BGK+Guo kernel. It implements the Jiachen standard-LBM formulas
for `phi` pseudo-time Poisson collision, electric-field recovery from the
potential population first moment, charge drift-diffusion collision, and the
coarse electroconvection onset bracket.

## Public surface

- `run_ehd_hydrostatic_2d(; Nx, Ny, C, M, Ma_E, alpha, charge_scheme, phi_scheme, injector_closure=:nee, phi_tol, field_tol, phi_max_iter, backend, FT)` — standalone hydrostatic EHD validation driver. Returns 2D fields, x-averaged profiles, analytic profiles, relative L2 errors, convergence metadata, and lattice parameters. The experimental `:static_moment` option is restricted to this zero-fluid-velocity, periodic-x, flat-plate driver with `charge_scheme=:regularized`, `phi_scheme=:direct`, `FT=Float64`, `delta_U=1` and finite positive `C`, `M`, `Ma_E`, `alpha`. It preserves the outgoing injector population aggregate and reconstructs the prescribed charge and normal moment; collector and field operators are unchanged. `:nee` remains the default. The coupled driver and `init_state(ECState)` accept only `:nee` in this static-only increment and reject `:static_moment` before allocation.
- `run_electroconvection_2d(; Nx, Ny, C, M, T, Ma_E, alpha, max_cycles, phi_tol, field_tol, phi_max_iter, phi_substeps, force_projection, backend, FT)` — coupled CPU/GPU canary. Since issue #26 this is a **one-shot wrapper** over the platform state contract (`docs/agent/state-implication.md`): `init_state(ECState; ...)` then `advance!(s, max_cycles; sample_final=true)` then `solution(s).result`. Returns flow, charge, potential, electric field, Coulomb force, velocity history, lattice mapping, and loop diagnostics including `loop_ms_per_step` measured over the coupled loop only. Same keywords and same numbers as before the split (see Touch order below), with one later change: since issue #23 the adaptive `:lbm` potential solve also stops on the field (`field_tol`, default `1e-4`), which changes the numbers on that path whenever the field check delays a stop (it does at default settings); `field_tol=Inf` reproduces the pre-#23 numbers bit for bit, and the split-parity test passes it for that reason.
- `ECState`/`ECSolution` (`src/drivers/ehd_ec_state.jl`) — the electroconvection cycle body as a client of the state contract: `init_state(ECState; ...)`, `advance!(s, n; sample_final=false)`, `solution(s)`. Not implemented for this client: `Kraken.snapshot`, `restore_state`, `Kraken.update_parameter!` — checkpointing electroconvection to disk is not available yet (verified against the source, not a placeholder omission).
- `src/drivers/ehd_phi_ddf.jl` (not exported, called as `Kraken.xxx`) — the DDF potential solve shared by both drivers. `ehd_phi_ddf_step!` (one iteration: collide, stream, plate NEE, zeroth moment; `xbc=:neumann` for the EC box, `:periodic` for the hydrostatic driver), `ehd_phi_ddf_solve!` (the adaptive loop; returns the swapped population pair and `(; iters, phi_rel, field_rel, converged)` without raising, each driver raises its own error through `_ehd_phi_ddf_require_converged`), `ehd_phi_ddf_workspace`, and the check cadences `EHD_EC_PHI_CHECK_EVERY = 8` and `EHD_HYDROSTATIC_PHI_CHECK_EVERY = 1`. The fixed-substep branch of `advance!` calls `ehd_phi_ddf_step!`. `field_tol` is a Julia keyword of `init_state(ECState)`, `run_electroconvection_2d` and `run_ehd_hydrostatic_2d`; `.krk` does not set it.
- `Kraken.Units.EHDSpec` plus `Kraken.Units.ehd_ec_lattice_params` own the electroconvection nondimensional-to-lattice mapping for `T`, `C`, `M`, `alpha`, and `Ma_E`. The coupled driver keeps its public `T` keyword but stores the spec field as `T_ehd` to avoid colliding with the units module's numeric type convention.
- `.krk` runner surface: `Module ehd` dispatches `Simulation ehd_hydrostatic ...` to the hydrostatic driver and `Simulation electroconvection_2d ...` to the coupled driver. EHD nondimensional groups and scheme selectors live in `Physics` params; `Preset electroconvection_2d` emits the validated MRT/direct-potential configuration. `Physics injector_closure = static_moment` forwards the opt-in selector to the hydrostatic driver; omission retains `nee`. Unsupported combinations, including selecting it for electroconvection, are rejected rather than ignored.
- Kernel-level surface: `collide_electric_potential_2d!`, `compute_electric_field_2d!`, `collide_electric_charge_srt_2d!`, `collide_electric_charge_regularized_2d!`, `compute_ehd_scalar_2d!`, the stopping reductions `ehd_rel_change_2d!` (potential) and `ehd_field_change_2d!` (field, #23; single work item, like the former), scalar NEE wall/box BC kernels, EHD-local non-periodic stream, free-slip sidewall port, Coulomb force helper, and Guo-corrected macro recovery.

## Reads from

- `lbm` — D2Q9 ordering and `collide_guo_field_2d!` for forced Navier-Stokes.
- `bc` — only the convention that wall-node scalar BCs are applied after streaming. EHD wall values are local non-equilibrium extrapolation kernels owned by this module.

## Writes to

- Mutates potential populations `phi_f`, charge populations `q_f`, scalar moments `phi`/`q`, and electric fields `Ex`/`Ey` in place.
- The coupled driver also mutates local NS populations, `rho/ux/uy`, and `Fx/Fy`.
- Allocates per run in the driver: two ping-pong DDF arrays for each scalar, scalar moment arrays, electric-field arrays, the previous-check field `Ex_prev`/`Ey_prev` of the potential stopping rule (in `ECState`, or in the hydrostatic driver's `ehd_phi_ddf_workspace`), and small host copies for convergence tests and returned profiles. No buffer is allocated per check of the potential solve (on the CPU backend, each check still pays the kernel-launch overhead of its three launches, about 350 B).
- Does not mutate parser state, units registries, generic BC framework, or GPU-specific paths.

## Backend constraints

- Kernels are KernelAbstractions `@kernel` functions and use `@Const` for read-only arrays.
- Hot kernels are allocation-free and unroll the D2Q9 operations.
- The hydrostatic driver is backend-generic for arrays, but convergence checks copy the small validation fields to the host each step. Each check of the adaptive potential solve costs one device-to-host copy of a length-4 vector (`phi_rel`, potential finite, `field_rel`, field finite). The coupled electroconvection driver is backend-generic for arrays; `phi_scheme=:direct` has a GPU direct-solve path when CUDSS is loaded, while diagnostics and returned fields still gather to host as before.
- No MRT/TRT charge collision or generic parser branch is included; `.krk` EHD dispatch is a thin runner over existing EHD drivers.

## Coupled Loop Conventions

- PRE/Jiachen lattice mapping: `K = Ma_E*H*cs/delta_U`, `nu = M^2*K*delta_U/T`, `tau = 0.5 + 3*nu`, `eps = (M*K)^2`, `q_inj = C*eps*delta_U/H^2`, `D = alpha*K*delta_U`, and `tau_q = 0.5 + 3*D`. The arithmetic source of truth is `src/units/physics/electromagn.jl`; `src/drivers/ehd_ec.jl` only constructs the EHD spec and delegates.
- Each outer cycle solves or substeps `phi`, computes `E`, computes charge-advection macros from the previous force, advances charge with equilibrium drift `u + K*E`, forms current `F=qE`, collides NS through `collide_guo_field_2d!`, streams, applies free-slip sidewall mirroring, then recovers current Guo-corrected macros.
- `phi_scheme=:lbm` is the faithful pseudo-time DDF path. `phi_scheme=:direct` replaces only the potential solve: it assembles the wall-node, unit-spacing 5-point operator once and solves `laplacian(phi) = -q/eps` once per outer step through the factorize-once linear-solve seam. The source sign follows `collide_electric_potential_2d!`, whose positive lattice source converges to `-q/eps` on the right-hand side.
- Direct Poisson BCs: bottom plate `phi=1`, top plate `phi=0`; hydrostatic uses periodic x because the DDF streams periodic-x; electroconvection uses mirror Neumann x sides matching the box sidewall scalar BC. Plate rows are identity rows and interior rows carry the source. The mixed identity/stencil matrix is non-symmetric as assembled, so both direct paths use `spd=false`.
- Direct Poisson setup dispatches on setup type. CPU/default setup returns `EhdPoissonSetup` and preserves the historical UMFPACK path byte-for-byte, including per-step host `q`/`phi` transfers. GPU setup returns `EhdPoissonSetupGPU`: the SAME assembled host CSC operator is factorized once through `lin_factorize(A; backend=CUDABackendTag(), spd=false, pin_k0=0)`, the RHS is filled by a KernelAbstractions kernel on device, `lin_solve!` consumes the device RHS through `KrakenCUDSSExt`, and `phi` is copied device-to-device with zero per-step host transfers.
- Missing CUDSS extension behavior is loud degradation, not a crash: GPU setup catches only the documented CUDSS load-hint error, emits one warning telling the user to `using CUDA, CUDSS`, then falls back to the CPU UMFPACK setup. Julia 1.12 GPU environments must also account for the sibling-weakdep packaging landmine documented in `docs/agent/solve-linear-implication.md`.
- Direct electric-field recovery uses `E=-grad(phi)` in `compute_electric_field_fd_2d!`: central differences in the interior, second-order one-sided differences on the plates, `Ex=0` on mirror side nodes for the EC box, and cyclic central differences for periodic-x hydrostatic runs.
- Guo convention: `collide_guo_field_2d!` receives force density and computes the internal equilibrium velocity with `+F/2`. Driver macro recovery also adds `+F/2` from the matching time level: previous force before charge advection and current force after the forced NS step.
- Side scalar BCs are EHD-local NEE extrapolation. `phi` uses fixed bottom/top potential and zero-gradient sides; `q` uses fixed bottom injection, zero-gradient top, and zero-gradient sides. Wall/side corners are owned by side BCs, matching the MATLAB plate-mask order.
- Free-slip flow sidewalls are not reused from a generic BC path. The MATLAB population mirror is ported explicitly after streaming, with `ux=0` on side columns and `uy` copied from the adjacent interior column for macro enforcement.
- Optional force projection ports Jiachen's per-row mean subtraction over fluid nodes. `:xy` subtracts both components; `:y` subtracts only vertical force; default is `:none`.
- NS coupling uses BGK+Guo rather than Jiachen's MRT NS collision. This is sufficient for the analytical canary but is not a production onset benchmark claim.
- Coupled EC host-sync cadence: normal `:lbm` outer steps do not copy diagnostics to host. Charge rel-change, charge/velocity finite checks, `velocity_stop`, and history sampling run only when `cycle % history_interval == 0` or on the final cycle, so instability detection can lag by at most `history_interval` cycles. In the adaptive `:lbm` inner potential loop, the stopping diagnostics are copied only every `EHD_EC_PHI_CHECK_EVERY=8` iterations and on `phi_max_iter`. A check accepts when `phi_rel <= phi_tol` (one-iteration change of `phi`) and `field_rel <= field_tol`, where `field_rel` is the change of `E` since the previous check, relative to `max(max|E|, E_ref)`, with `E_ref = |phi_bottom − phi_top|/(Ny − 1)` the applied field (`1/(Ny − 1)` in all public drivers). The applied field is a minimum scale: it applies while `E` is below it everywhere, as at the first checks of a cold start, where `E` starts at 0. A `floatmin` guard avoids a division by zero when the plates are at the same potential and the field is zero. Measured on 2026-10-08 at every accepting check: `max|E| / E_ref` is 1.484 to 1.495 in the default hydrostatic run and in 60x96 EC over 20 cycles (Float64 and Float32), 0.99999999994 to 1.0043 on the ES-002 capacitors; the rule relative to `max|E|` alone gave the same verdict at each of these acceptances. A final check off cadence cannot accept. The cadence is therefore part of the rule, not only a sync saving: `field_rel` measures a change over 8 iterations, and changing the cadence changes where the solve stops. The hydrostatic driver checks every iteration (`EHD_HYDROSTATIC_PHI_CHECK_EVERY=1`).
- `benchmarks/ehd/tc_sweep.jl` writes `ms_per_step` from `result.loop_ms_per_step` into each summary CSV/Markdown row and case log. With `--gpu`, the script loads `CUDA, CUDSS` before constructing `CUDA.CUDABackend()` so the direct phi GPU path activates when available.

## Failure modes

- `tau_q` is close to `0.5` at `alpha=1e-4`; the hydrostatic validation passes with SRT, while the coupled onset canary uses the regularized charge collision as in Jiachen's default validation path.
- Scalar electrodes are wall nodes at `y*=0,1`. For legacy `injector_closure=:nee`, the returned analytic diagnostics retain effective interior half-link samples `y*=(j-3/2)/(Ny-1)` and `y_mapping=:wall_nodes_interior_half_link`. For `:static_moment`, the injector is a normal-moment reconstruction and all analytic samples use actual nodes `y*=(j-1)/(Ny-1)`, with `y_mapping=:wall_nodes`; the collector is unchanged. Thus returned `err_q`/`err_E` are not directly comparable between closures: both the boundary update and comparison coordinates differ. These built-in references neglect diffusion and are diagnostics, not finite-alpha acceptance. Issue #73 uses independent finite-alpha references on common physical nodes and all-node current checks. The converged CPU regression in `test/analytical/ehd_static_moment_2d.jl` guards the integrated call order and prescribed injector charge, not complete physical validation.
- The analytic E profile is positive upward; if `Ey` changes sign, inspect the potential source sign and D2Q9 direction ordering before changing analytics.
- The hydrostatic validation assumes x-invariance and periodic x. The onset validation uses sidewalls and a sinusoidal charge perturbation.
- `test/analytical/ehd_onset_2d.jl` uses `Nx=59`, `Ny=96`, `A≈0.611`, `C=10`, `M=10`, `alpha=1e-4`, `Ma_E=0.01`, `phi_substeps=1`, and 50k cycles per branch. Setup runs measured about 21 s per branch. `T=150` decayed from `max|u|≈2.40e-3` to `5.08e-6`; `T=190` was marginal at `1.89e-4`; `T=220` grew to `1.62e-3`, over 100x the `T=150` final value. The bracket is a coarse BGK+Guo canary with shifted `T_c`, not a PRE-accurate critical-number measurement.
- Potential stopping (#23). Before the repair the adaptive solve stopped on `phi_rel` alone, and `E`, rebuilt from the first moment, relaxes on its own as `(1-1/tau_U)^n`: a cold start could exit with `phi` converged and `E` not (relative error 0.29 on the first hydrostatic solve at default settings). The field check removes that failure; it does not bound the error. Once the slow diffusive mode `sin(pi*y/H)` dominates, the relative `E` error is about `kappa*field_rel`, `kappa ≈ H^2/(m*gamma*pi^2)` (`H=Ny-1`, `m` the cadence, `gamma=nu_U`): about 5, 22 and 380 on 8x12, 16x24 and 60x96 at `gamma=0.3`, `m=8`, and about 3000 on the hydrostatic default (`Ny=96`, `m=1`). The ES-002 public-driver case at default settings keeps a 3.5e-5 `E*` error from this mode and stays `@test_broken`.
- Float32 potential solve: on grids with `H ≳ 100` the DDF iteration freezes at bit level before `E` converges. Measured on the first EC cycle at 120x192 (`T=220`): `field_rel` is exactly 0 after 824 iterations, the check accepts even with `field_tol=0`, and the frozen field (identical at 1000 and 8000 iterations) differs from the Float64 iterate at 8000 iterations by 4.5e-4 of `max|E|`, while the Float64 field still moves by 1.9e-4 of `max|E|` between those two iterations. At 60x96 Float32 does not freeze, but `field_tol=1e-9` is not met within 20000 iterations (last `field_rel` 1.2e-7, against 1.2e-9 in Float64), so the driver would raise. Use `phi_scheme=:direct` or Float64 for an accurate `E`.

## Touch order

1. `src/kernels/ehd_2d.jl` — collision formulas, E moment, EHD-specific wall extrapolation, and the
   stopping reductions `ehd_rel_change_2d!` / `ehd_field_change_2d!`.
2. `src/kernels/ehd_bc_2d.jl` — EHD-local sidewall, force, Guo-macro, and free-slip canary helpers.
3. `src/drivers/ehd_phi_ddf.jl` — the DDF potential step and adaptive solve (stopping rule, check
   cadences, workspace, shared non-convergence error); read it before either driver.
4. `src/drivers/ehd.jl` / `src/drivers/ehd_ec_state.jl` (cycle body: `ECState`, `init_state`,
   `advance!`, `solution`) / `src/drivers/ehd_ec.jl` (parameter mapping, analytic profiles, and the
   thin one-shot wrapper `run_electroconvection_2d` over the three verbs above).
5. `test/analytical/ehd_hydrostatic_2d.jl` and `test/analytical/ehd_onset_2d.jl` — CPU analytical validations.
6. `test/analytical/ES-002-STOP.jl` (capacitor fixtures through the production solve, with a `field_tol=Inf`
   negative control) and `test/analytical/ehd_phi_ddf_solve_2d.jl` (public driver with a tight `field_tol`,
   wiring guards on the defaults, function contract, Float32/Float64 bounds) — the potential stopping rule (#23).
7. `test/analytical/ehd_ec_split_parity_2d.jl` — behaviour-preservation test for the driver split
   (issue #26): compares `run_electroconvection_2d` bit for bit against the frozen pre-split copy
   `test/reference/ehd_ec_legacy.jl` (test-only, never edited, evaluated inside the `Kraken` module
   because it uses unexported internals), then checks that any split of a run into segments matches
   the continuous run bit for bit (exact `isequal`, not a tolerance). The new side of the legacy comparison
   passes `field_tol=Inf`, the pre-#23 rule, so that comparison stays bit-exact.
8. `src/Kraken.jl` — include/export registration only (`ehd_phi_ddf.jl` is included after `ehd_poisson.jl`,
   before `ehd.jl`).
