function _ehd_ec_lattice_params(Ny, C, M, T_ehd, Ma_E, alpha, delta_U, gamma; FT)
    spec = Units.EHDSpec{FT}(FT(T_ehd), FT(C), FT(M), FT(alpha), FT(Ma_E))
    return Units.ehd_ec_lattice_params(spec, Ny, delta_U, gamma; FT=FT)
end

function _fill_charge_populations_ec!(f_cpu, q_init, Ey_profile, K, Nx, Ny, FT)
    for j in 1:Ny, i in 1:Nx, qdir in 1:9
        f_cpu[i, j, qdir] = _charge_feq_host(q_init[i, j], zero(FT), K * Ey_profile[j], qdir, FT)
    end
    return f_cpu
end

function _project_coulomb_force_rows!(Fx, Fy, is_solid, mode)
    mode === :none && return nothing
    mode in (:xy, :y) || throw(ArgumentError("force_projection must be :none, :xy, or :y."))
    mode_code = mode === :xy ? 1 : 2
    Nx, Ny = size(Fx)
    project_coulomb_force_rows_2d!(Fx, Fy, is_solid, mode_code, Nx, Ny)
    return nothing
end

"""
    run_electroconvection_2d(; Nx, Ny, C, M, T, Ma_E, alpha, max_cycles, ...)

Run a CPU-oriented coupled EHD electroconvection canary. The electric potential
uses the pseudo-time DDF Poisson solve, charge uses drift equilibrium
`u + K*E`, and Navier-Stokes uses BGK + Guo forcing with force density `q*E`.
Electrical sidewalls retain EHD-local zero-gradient scalar NEE. Flow sidewalls
select `sidewall_bc=:free_slip` (default, preserving the existing mirroring) or
`:no_slip` (stationary on-node walls at i=1,Nx, separation Nx-1). Plate rows and
corners retain their existing treatment in both modes.

Potential solve (`phi_scheme = :lbm`, `phi_substeps = nothing`): each cycle
iterates the DDF until a check accepts, or raises after `phi_max_iter` iterations.
Checks run every `Kraken.EHD_EC_PHI_CHECK_EVERY = 8` iterations. A check accepts
when the potential changed by at most `phi_tol` (relative, over the last
iteration) and the field by at most `field_tol`. The field change is the change
of `E` since the previous check, relative to `max(max|E|, E_ref)`, with
`E_ref = |phi_bottom − phi_top|/(Ny − 1)` the applied field (`1/(Ny − 1)` in all
public drivers). The applied field is a minimum scale: it applies while `E` is
below it everywhere, as at the first checks of a cold start, where `E` starts at
0. A `floatmin` guard avoids a division by zero when the plates are at the same
potential and the field is zero. Measured on 60x96 over 20 cycles (2026-10-08,
Float64 and Float32): `max|E| / E_ref` is 1.484 at every accepting check, so the
floor never decides there.

- `field_tol` (default `1e-4`, independent of `phi_tol`) bounds a change between
  checks, not the error of `E`. Once the slow diffusive mode of the pseudo-time
  iteration dominates, the relative error of `E` is about `κ * field_tol`, with
  `κ ≈ H^2 / (m * gamma * π^2)`, `H = Ny - 1` and `m = 8` the cadence: about 5,
  22 and 380 on 8x12, 16x24 and 60x96 at `gamma = 0.3`. For a relative error
  `ε` on `E`, use `field_tol ≈ ε / κ`, or `phi_scheme = :direct`.
- `field_tol = Inf` means no field check (the rule before issue #23); it is
  reserved for non-regression comparisons against that rule.
- `FT = Float32`: on grids with `H ≳ 100` the iteration stops changing at bit
  level before `E` has converged, so the field change drops to 0 and the check
  accepts whatever `field_tol` is. Use `phi_scheme = :direct` or `FT = Float64`
  for an accurate `E`.
- Keep `phi_max_iter` a multiple of 8. Otherwise the last check falls off the
  cadence, and with a finite `field_tol` that check cannot accept: the solve
  raises even when both reported changes are below their tolerances. With a
  finite `field_tol`, `phi_max_iter < 8` never converges.

`field_tol` is a Julia keyword only; `.krk` files do not set it. `init_state` rejects
a negative or `NaN` `field_tol`, and on the adaptive path a `phi_max_iter` that is not
an integral value of at least 1, with an `ArgumentError` before allocating.
"""
function run_electroconvection_2d(; Nx=60, Ny=96, C=10.0, M=10.0, T=175.0,
                                    Ma_E=1e-2, alpha=1e-4, delta_U=1.0,
                                    gamma=0.3, max_cycles=2000,
                                    target_t_star=nothing,
                                    phi_tol=1e-4, field_tol=1e-4, phi_max_iter=10000,
                                    phi_substeps=nothing,
                                    phi_scheme=:lbm,
                                    charge_scheme=:regularized,
                                    injector_closure=:nee,
                                    ns_scheme=:bgk,
                                    sidewall_bc=:free_slip,
                                    perturb_amplitude=1e-4,
                                    perturb_mode=1,
                                    force_projection=:none,
                                    velocity_stop=0.2,
                                    history_interval=1,
                                    backend=KernelAbstractions.CPU(),
                                    FT=Float64)
    # Run control lives here; everything else is the state contract
    # (src/drivers/ehd_ec_state.jl): init_state -> advance! -> solution.
    s = init_state(ECState; Nx=Nx, Ny=Ny, C=C, M=M, T=T, Ma_E=Ma_E, alpha=alpha,
                   delta_U=delta_U, gamma=gamma, phi_tol=phi_tol, field_tol=field_tol,
                   phi_max_iter=phi_max_iter, phi_substeps=phi_substeps,
                   phi_scheme=phi_scheme, charge_scheme=charge_scheme,
                   injector_closure=injector_closure,
                   ns_scheme=ns_scheme, sidewall_bc=sidewall_bc,
                   perturb_amplitude=perturb_amplitude,
                   perturb_mode=perturb_mode, force_projection=force_projection,
                   velocity_stop=velocity_stop, history_interval=history_interval,
                   backend=backend, FT=FT)
    p = s.p
    if target_t_star !== nothing
        target_cycles = Int(ceil(FT(target_t_star) / p.dt_star))
        max_cycles = min(Int(max_cycles), target_cycles)
    else
        max_cycles = Int(max_cycles)
    end
    advance!(s, max_cycles; sample_final=true)
    return solution(s).result
end
