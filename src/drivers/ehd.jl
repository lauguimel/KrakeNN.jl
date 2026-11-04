"""
    ehd_hydrostatic_profiles(C, Ny; FT=Float64)

Analytic strong-injection hydrostatic EHD base state from Luo, Wu, Yi & Tan
(PRE 93, 023309, 2016), evaluated on node-on-wall coordinates
`y* = (j - 1) / (Ny - 1)`.
"""
function ehd_hydrostatic_profiles(C, Ny; FT=Float64)
    CT = FT(C)
    froot(b) = (FT(4) * CT / FT(3)) * sqrt(b) * ((one(FT) + b)^(FT(1.5)) - b^(FT(1.5))) - one(FT)
    lo = FT(1e-12)
    hi = one(FT)
    flo = froot(lo)
    fhi = froot(hi)
    while sign(flo) == sign(fhi) && hi < FT(1e12)
        hi *= FT(2)
        fhi = froot(hi)
    end
    sign(flo) == sign(fhi) && error("Unable to bracket hydrostatic EHD root for C=$(C).")
    for _ in 1:200
        mid = (lo + hi) / FT(2)
        fm = froot(mid)
        if abs(fm) <= FT(10) * eps(FT)
            lo = mid
            hi = mid
            break
        elseif sign(flo) == sign(fm)
            lo = mid
            flo = fm
        else
            hi = mid
        end
    end
    b = (lo + hi) / FT(2)
    a = FT(2) * CT * sqrt(b)
    y = [FT(j - 1) / FT(Ny - 1) for j in 1:Ny]
    q_star = [a / (FT(2) * CT * sqrt(yj + b)) for yj in y]
    E_star = [a * sqrt(yj + b) for yj in y]
    phi = [one(FT) - (FT(2) * a / FT(3)) * ((yj + b)^(FT(1.5)) - b^(FT(1.5))) for yj in y]
    return (y=y, b=b, a=a, q_star=q_star, E_star=E_star, phi=phi)
end

function _ehd_lattice_params(Ny, C, M, Ma_E, alpha, delta_U; FT)
    H = FT(Ny - 1)
    cs = inv(sqrt(FT(3)))
    K = FT(Ma_E) * H * cs / FT(delta_U)
    eps_e = (FT(M) * K)^2
    q_inj = FT(C) * eps_e * FT(delta_U) / H^2
    D = FT(alpha) * K * FT(delta_U)
    gamma = FT(0.3)
    tau_U = FT(3) * gamma + FT(0.5)
    nu_U = (tau_U - FT(0.5)) / FT(3)
    tau_q = FT(3) * D + FT(0.5)
    return (H=H, cs=cs, K=K, eps=eps_e, q_inj=q_inj, D=D,
            tau_U=tau_U, nu_U=nu_U, tau_q=tau_q,
            omega_U=inv(tau_U), omega_q=inv(tau_q))
end

function _fill_phi_populations!(f_cpu, phi, Nx, Ny, FT)
    w = (FT(4)/FT(9), FT(1)/FT(9), FT(1)/FT(9), FT(1)/FT(9), FT(1)/FT(9),
         FT(1)/FT(36), FT(1)/FT(36), FT(1)/FT(36), FT(1)/FT(36))
    for j in 1:Ny, i in 1:Nx, q in 1:9
        f_cpu[i, j, q] = w[q] * phi[j]
    end
    return f_cpu
end

function _charge_feq_host(q, ux, uy, qdir, FT)
    w = qdir == 1 ? FT(4)/FT(9) : (qdir <= 5 ? FT(1)/FT(9) : FT(1)/FT(36))
    cx = (FT(0), FT(1), FT(0), FT(-1), FT(0), FT(1), FT(-1), FT(-1), FT(1))[qdir]
    cy = (FT(0), FT(0), FT(1), FT(0), FT(-1), FT(1), FT(1), FT(-1), FT(-1))[qdir]
    cu3 = FT(3) * (cx * ux + cy * uy)
    return q * w * (one(FT) + cu3 + FT(0.5) * cu3^2 - FT(1.5) * (ux^2 + uy^2))
end

function _fill_charge_populations!(f_cpu, q_profile, E_profile, K, Nx, Ny, FT)
    for j in 1:Ny, i in 1:Nx, qdir in 1:9
        f_cpu[i, j, qdir] = _charge_feq_host(q_profile[j], zero(FT), K * E_profile[j], qdir, FT)
    end
    return f_cpu
end

function _rel_l2(a, b)
    num = sqrt(sum(abs2, a .- b))
    den = max(sqrt(sum(abs2, b)), floatmin(eltype(float.(b))))
    return num / den
end

function _ehd_check_injector_closure(closure, driver, charge_scheme, phi_scheme, FT, delta_U)
    closure in (:nee, :static_moment, :moment) ||
        throw(ArgumentError("injector_closure must be :nee, :static_moment or :moment."))
    closure === :nee && return nothing
    if closure === :moment
        driver === :ec || throw(ArgumentError(":moment is restricted to ECState."))
        (charge_scheme === :regularized && phi_scheme === :direct && FT === Float64 && delta_U == 1) ||
            throw(ArgumentError(":moment requires regularized/direct, Float64 and delta_U=1."))
        return nothing
    end
    driver === :hydrostatic || throw(ArgumentError(
        ":static_moment is experimental and restricted to run_ehd_hydrostatic_2d; ECState/T190 is not supported."))
    charge_scheme === :regularized && phi_scheme === :direct ||
        throw(ArgumentError(":static_moment requires charge_scheme=:regularized and phi_scheme=:direct."))
    FT === Float64 || throw(ArgumentError(":static_moment is qualified for testing only in Float64."))
    delta_U == 1 || throw(ArgumentError(":static_moment currently requires delta_U=1 (the driver's unit-potential plates)."))
    return nothing
end

"""
    run_ehd_hydrostatic_2d(; Nx=8, Ny=96, C=10.0, M=10.0, Ma_E=1e-2,
                            alpha=1e-4, charge_scheme=:srt, ...)

Run the uncoupled D2Q9/D2Q9 EHD hydrostatic base-state solve with `u=0`.
The electric potential uses the Jiachen pseudo-time Poisson DDF and wall-node
non-equilibrium extrapolation BCs (`phi_bottom=1`, `phi_top=0`). The charge
field uses full second-order drift equilibrium with either `:srt` or the
regularized collision. The hydrostatic validation uses SRT; regularized is kept
available because the default `alpha=1e-4` gives `tau_q - 0.5 = O(1e-4)`, where
the faithful MATLAB driver commonly uses regularization.

Steady state is declared when
`max(abs(q^{n+1}-q^n)) / max(abs(q^{n+1})) <= charge_tol` after a converged
inner phi solve. Wall values are node-on-wall; interior DDF profiles are compared
at the effective half-link samples `y*=(j-3/2)/(Ny-1)`, which is the grid where
the faithful wall-node NEE charge update matches the diffusion-free base state.

Potential solve (`phi_scheme = :lbm`): each step iterates the DDF until a check
accepts, or raises after `phi_max_iter` iterations. The check runs after every
iteration (`Kraken.EHD_HYDROSTATIC_PHI_CHECK_EVERY = 1`) and accepts when the
potential changed by at most `phi_tol` (relative, over the last iteration) and the
field by at most `field_tol`. The field change is the change of `E` since the
previous check, relative to `max(max|E|, E_ref)`, with
`E_ref = |phi_bottom − phi_top|/(Ny − 1)` the applied field (`1/(Ny − 1)` in all
public drivers). The applied field is a minimum scale: it applies while `E` is
below it everywhere, as at the first checks of a cold start, where `E` starts at 0.
A `floatmin` guard avoids a division by zero when the plates are at the same
potential and the field is zero. Measured on the default run (2026-10-08, Float64
and Float32): `max|E| / E_ref` is 1.484 to 1.495 at every accepting check, so the
floor never decides there.

- `field_tol` (default `1e-4`, independent of `phi_tol`) bounds a change between
  checks, not the error of `E`. Once the slow diffusive mode of the pseudo-time
  iteration dominates, the relative error of `E` left by one solve is about
  `κ * field_tol`, with `κ ≈ H^2 / (m * gamma * π^2)`, `H = Ny - 1`, `m = 1` the
  cadence and `gamma = 0.3` fixed here (`tau_U = 1.4`): about 3000 at the default
  `Ny = 96`. The first solve of the default run (cold start) is not yet in that
  regime: its relative error is 3.4e-3 at `field_tol = 1e-4` and 6.3e-4 at `1e-6`
  (0.29 with the rule before issue #23). Later steps start from the previous
  populations; the default run ends with `err_E = 0.76 %` (relative L2 against
  the analytic field), as before #23. For a relative error `ε` on `E` from one
  solve, use `field_tol ≈ ε / κ`, or `phi_scheme = :direct`.
- `field_tol = Inf` means no field check (the rule before issue #23); it is
  reserved for non-regression comparisons against that rule.
- `FT = Float32`: on grids with `H ≳ 100` the iteration stops changing at bit
  level before `E` has converged, so the field change drops to 0 and the check
  accepts whatever `field_tol` is. Use `phi_scheme = :direct` or `FT = Float64`
  for an accurate `E`.

`injector_closure=:nee` retains the legacy default. Experimental
`:static_moment` retains the outgoing normal charge-population aggregate at the
flat injector. It is restricted to this u=0, periodic-x driver, regularized charge,
direct potential, Float64 and delta_U=1; it does not enable ECState/T190.
The collector and field operator are unchanged. For this option only, returned
analytic comparisons use actual wall-node coordinates (no half-link shift).
Consequently the returned `err_q`/`err_E` are not directly comparable between
closures: the legacy diagnostics use different interior comparison coordinates.
Those diffusion-free comparisons are diagnostics, not finite-alpha acceptance;
issue #73 uses an independent finite-alpha reference and all-node current gates.

`field_tol` is a Julia keyword only; `.krk` files do not set it. Throws an
`ArgumentError` before allocating when `field_tol` is negative or `NaN`, or, with
`phi_scheme = :lbm`, when `phi_max_iter` is not an integral value of at least 1.
"""
function run_ehd_hydrostatic_2d(; Nx=8, Ny=96, C=10.0, M=10.0, Ma_E=1e-2,
                                  alpha=1e-4, delta_U=1.0,
                                  charge_scheme=:srt,
                                  phi_scheme=:lbm,
                                  injector_closure=:nee,
                                  max_steps=100000, charge_tol=1e-8,
                                  phi_tol=1e-4, field_tol=1e-4, phi_max_iter=10000,
                                  backend=KernelAbstractions.CPU(), FT=Float64)
    Nx < 3 && throw(ArgumentError("Nx must be at least 3."))
    Ny < 8 && throw(ArgumentError("Ny must be at least 8."))
    charge_scheme in (:srt, :regularized) ||
        throw(ArgumentError("charge_scheme must be :srt or :regularized."))
    phi_scheme in (:lbm, :direct) ||
        throw(ArgumentError("phi_scheme must be :lbm or :direct."))
    _ehd_check_injector_closure(injector_closure, :hydrostatic, charge_scheme, phi_scheme, FT, delta_U)
    if injector_closure === :static_moment
        all(x -> isfinite(x) && x > 0, (C, M, Ma_E, alpha)) ||
            throw(ArgumentError(":static_moment requires finite positive C, M, Ma_E and alpha."))
    end
    _ehd_phi_ddf_check_field_tol(field_tol)
    phi_scheme === :lbm && (phi_max_iter = _ehd_phi_ddf_max_iter(phi_max_iter))

    p = _ehd_lattice_params(Ny, C, M, Ma_E, alpha, delta_U; FT=FT)
    p.tau_q <= FT(0.5) && error("Charge relaxation time must be greater than 0.5.")
    analytic = ehd_hydrostatic_profiles(C, Ny; FT=FT)
    q_init_profile = FT(p.q_inj) .* analytic.q_star
    Ey_init_profile = FT(delta_U) .* analytic.E_star ./ FT(p.H)
    y_compare = copy(analytic.y)
    if injector_closure === :nee
        for j in 2:(Ny - 1)
            y_compare[j] = FT(j) - FT(1.5)
            y_compare[j] /= FT(p.H)
        end
    end
    q_star_compare = [analytic.a / (FT(2) * FT(C) * sqrt(yj + analytic.b)) for yj in y_compare]
    E_star_compare = [analytic.a * sqrt(yj + analytic.b) for yj in y_compare]
    phi_compare = [one(FT) - (FT(2) * analytic.a / FT(3)) *
                   ((yj + analytic.b)^(FT(1.5)) - analytic.b^(FT(1.5))) for yj in y_compare]
    q_analytic = FT(p.q_inj) .* q_star_compare
    Ey_analytic = FT(delta_U) .* E_star_compare ./ FT(p.H)

    phi_f_in = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    phi_f_out = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    q_f_in = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    q_f_out = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    phi = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    qfield = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    phi_ws = ehd_phi_ddf_workspace(phi)
    q_prev = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    diag = KernelAbstractions.zeros(backend, FT, 2)
    diag_host = Vector{FT}(undef, 2)
    Ex = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Ey = KernelAbstractions.zeros(backend, FT, Nx, Ny)

    phi_init = _fill_phi_populations!(zeros(FT, Nx, Ny, 9), analytic.phi, Nx, Ny, FT)
    q_init = _fill_charge_populations!(zeros(FT, Nx, Ny, 9), q_init_profile, Ey_init_profile, FT(p.K), Nx, Ny, FT)
    copyto!(phi_f_in, phi_init)
    copyto!(phi_f_out, phi_init)
    copyto!(q_f_in, q_init)
    copyto!(q_f_out, q_init)
    compute_ehd_scalar_2d!(phi, phi_f_in)
    compute_ehd_scalar_2d!(qfield, q_f_in)
    poisson_setup = phi_scheme === :direct ? ehd_poisson_setup(Nx, Ny, p.eps; xbc=:periodic, backend=backend) : nothing
    if phi_scheme === :direct
        ehd_poisson_solve!(phi, poisson_setup, qfield)
        compute_electric_field_fd_2d!(Ex, Ey, phi, :periodic, Nx, Ny)
    end

    phi_iters_last = 0
    phi_rel_last = Inf
    q_rel = Inf
    steps_done = 0

    for step in 1:max_steps
        steps_done = step
        copyto!(q_prev, qfield)

        if phi_scheme === :direct
            ehd_poisson_solve!(phi, poisson_setup, qfield)
            compute_electric_field_fd_2d!(Ex, Ey, phi, :periodic, Nx, Ny)
            phi_iters_last = 1
            phi_rel_last = zero(FT)
        else
            phi_f_in, phi_f_out, phi_stats = ehd_phi_ddf_solve!(
                phi_f_in, phi_f_out, phi, qfield, p, :periodic, phi_ws;
                phi_tol=phi_tol, field_tol=field_tol, max_iter=phi_max_iter,
                check_every=EHD_HYDROSTATIC_PHI_CHECK_EVERY,
                phi_bottom=one(FT), phi_top=zero(FT))
            phi_iters_last = phi_stats.iters
            phi_rel_last = phi_stats.phi_rel
            _ehd_phi_ddf_require_converged(phi_stats, phi_max_iter, phi_tol)
            compute_electric_field_2d!(Ex, Ey, phi_f_in, p.tau_U)
        end

        if charge_scheme == :srt
            collide_electric_charge_srt_2d!(q_f_in, Ex, Ey, p.tau_q, p.K)
        else
            collide_electric_charge_regularized_2d!(q_f_in, Ex, Ey, p.tau_q, p.K)
        end
        stream_periodic_x_wall_y_2d!(q_f_out, q_f_in, Nx, Ny)
        compute_ehd_scalar_2d!(qfield, q_f_out)
        if injector_closure === :nee
            apply_charge_nee_walls_2d!(q_f_out, qfield, Ex, Ey, p.q_inj, zero(FT), p.K, Nx, Ny)
        else
            _apply_charge_static_moment_walls!(q_f_out, qfield, Ex, Ey, p.q_inj, p.K, Nx, Ny)
        end
        compute_ehd_scalar_2d!(qfield, q_f_out)
        ehd_rel_change_2d!(diag, qfield, q_prev, Nx, Ny)
        copyto!(diag_host, diag)
        q_rel = diag_host[1]
        q_f_in, q_f_out = q_f_out, q_f_in
        diag_host[2] == zero(FT) && error("Charge field became non-finite at step $(step).")
        q_rel <= charge_tol && break
        step == max_steps &&
            error("EHD hydrostatic charge solve did not converge within $(max_steps) steps. Last relative change: $(q_rel).")
    end

    if phi_scheme === :lbm
        compute_electric_field_2d!(Ex, Ey, phi_f_in, p.tau_U)
        compute_ehd_scalar_2d!(phi, phi_f_in)
    end
    compute_ehd_scalar_2d!(qfield, q_f_in)
    phi_cpu = Array(phi)
    q_cpu = Array(qfield)
    Ex_cpu = Array(Ex)
    Ey_cpu = Array(Ey)
    q_profile = [sum(@view q_cpu[:, j]) / FT(Nx) for j in 1:Ny]
    phi_profile = [sum(@view phi_cpu[:, j]) / FT(Nx) for j in 1:Ny]
    Ex_profile = [sum(@view Ex_cpu[:, j]) / FT(Nx) for j in 1:Ny]
    Ey_profile = [sum(@view Ey_cpu[:, j]) / FT(Nx) for j in 1:Ny]
    interior = 2:(Ny - 1)
    err_q = _rel_l2(q_profile[interior], q_analytic[interior])
    err_E = _rel_l2(Ey_profile[interior], Ey_analytic[interior])
    xvar_q = maximum(abs.(q_cpu .- reshape(q_profile, 1, Ny)))
    xvar_phi = maximum(abs.(phi_cpu .- reshape(phi_profile, 1, Ny)))

    return (q=q_cpu, phi=phi_cpu, Ex=Ex_cpu, Ey=Ey_cpu,
            q_profile=q_profile, phi_profile=phi_profile,
            Ex_profile=Ex_profile, Ey_profile=Ey_profile,
            analytic=(y=y_compare, b=analytic.b, a=analytic.a,
                      q_star=q_star_compare, E_star=E_star_compare,
                      phi=phi_compare, q=q_analytic, Ey=Ey_analytic,
                      y_node=analytic.y, phi_node=analytic.phi),
            err_q=err_q, err_E=err_E, xvar_q=xvar_q, xvar_phi=xvar_phi,
            steps=steps_done, q_rel_change=q_rel,
            phi_iters_last=phi_iters_last, phi_rel_last=phi_rel_last,
            Nx=Nx, Ny=Ny, C=C, M=M, Ma_E=Ma_E, alpha=alpha,
            charge_scheme=charge_scheme, phi_scheme=phi_scheme,
            bc=injector_closure === :nee ? :non_equilibrium_extrapolation : :static_moment_injector,
            y_mapping=injector_closure === :nee ? :wall_nodes_interior_half_link : :wall_nodes,
            params=p)
end
