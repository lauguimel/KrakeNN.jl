# Frozen reference copy of `run_electroconvection_2d` as it stood at d531e6007,
# before the driver was split into init / advance / solution (issue #26).
#
# Test-only. It exists so the behaviour-preservation test has something to compare
# against: once the production function is rewritten on top of the state contract,
# the only way to prove "same numbers as before" is to keep the old body verbatim.
# Do not edit the body; do not call it from `src/`.
#
# The body uses unexported Kraken internals, so the file must be evaluated inside
# the package module:  Base.include(Kraken, joinpath(@__DIR__, "ehd_ec_legacy.jl"))

function _legacy_run_electroconvection_2d(; Nx=60, Ny=96, C=10.0, M=10.0, T=175.0,
                                    Ma_E=1e-2, alpha=1e-4, delta_U=1.0,
                                    gamma=0.3, max_cycles=2000,
                                    target_t_star=nothing,
                                    phi_tol=1e-4, phi_max_iter=10000,
                                    phi_substeps=nothing,
                                    phi_scheme=:lbm,
                                    charge_scheme=:regularized,
                                    ns_scheme=:bgk,
                                    perturb_amplitude=1e-4,
                                    perturb_mode=1,
                                    force_projection=:none,
                                    velocity_stop=0.2,
                                    history_interval=1,
                                    backend=KernelAbstractions.CPU(),
                                    FT=Float64)
    Nx < 4 && throw(ArgumentError("Nx must be at least 4."))
    Ny < 8 && throw(ArgumentError("Ny must be at least 8."))
    charge_scheme in (:srt, :regularized) ||
        throw(ArgumentError("charge_scheme must be :srt or :regularized."))
    ns_scheme in (:bgk, :mrt) ||
        throw(ArgumentError("ns_scheme must be :bgk or :mrt."))
    force_projection in (:none, :xy, :y) ||
        throw(ArgumentError("force_projection must be :none, :xy, or :y."))
    phi_scheme in (:lbm, :direct) ||
        throw(ArgumentError("phi_scheme must be :lbm or :direct."))
    history_interval = Int(history_interval)
    history_interval > 0 ||
        throw(ArgumentError("history_interval must be positive."))

    p = _ehd_ec_lattice_params(Ny, C, M, T, Ma_E, alpha, delta_U, gamma; FT=FT)
    p.tau <= FT(0.5) && error("NS relaxation time must be greater than 0.5.")
    p.tau_q <= FT(0.5) && error("Charge relaxation time must be greater than 0.5.")
    if target_t_star !== nothing
        target_cycles = Int(ceil(FT(target_t_star) / p.dt_star))
        max_cycles = min(Int(max_cycles), target_cycles)
    else
        max_cycles = Int(max_cycles)
    end

    analytic = ehd_hydrostatic_profiles(C, Ny; FT=FT)
    q_profile = FT(p.q_inj) .* analytic.q_star
    Ey_profile = FT(delta_U) .* analytic.E_star ./ FT(p.H)
    A = FT(Nx - 1) / FT(Ny - 1)

    phi_f_in = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    phi_f_out = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    q_f_in = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    q_f_out = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    f_in = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    f_out = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)

    phi = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    qfield = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Ex = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Ey = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    rho = KernelAbstractions.ones(backend, FT, Nx, Ny)
    ux = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    uy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fx = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fx_prev = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fy_prev = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    phi_prev = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    q_prev = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    diag = KernelAbstractions.zeros(backend, FT, 2)
    diag_host = Vector{FT}(undef, 2)
    is_solid = KernelAbstractions.zeros(backend, Bool, Nx, Ny)

    q_init = zeros(FT, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        xstar = A * FT(i - 1) / FT(Nx - 1)
        ystar = FT(j - 1) / FT(Ny - 1)
        q_init[i, j] = q_profile[j] +
                       FT(perturb_amplitude) * FT(p.q_inj) *
                       sin(FT(pi) * ystar) *
                       cos(FT(pi) * FT(perturb_mode) * xstar / A)
    end
    q_init[:, 1] .= FT(p.q_inj)

    phi_init = _fill_phi_populations!(zeros(FT, Nx, Ny, 9), analytic.phi, Nx, Ny, FT)
    q_pop_init = _fill_charge_populations_ec!(zeros(FT, Nx, Ny, 9), q_init, Ey_profile, FT(p.K), Nx, Ny, FT)
    ns_init = zeros(FT, Nx, Ny, 9)
    for qdir in 1:9
        ns_init[:, :, qdir] .= ehd_w(Val(qdir), FT)
    end
    # Matrix{Bool}, not falses(): copyto!(CuArray{Bool}, BitMatrix) falls back to scalar indexing
    solid_cpu = zeros(Bool, Nx, Ny)
    solid_cpu[:, 1] .= true
    solid_cpu[:, Ny] .= true

    copyto!(phi_f_in, phi_init)
    copyto!(phi_f_out, phi_init)
    copyto!(q_f_in, q_pop_init)
    copyto!(q_f_out, q_pop_init)
    copyto!(f_in, ns_init)
    copyto!(f_out, ns_init)
    copyto!(is_solid, solid_cpu)

    compute_ehd_scalar_2d!(phi, phi_f_in)
    compute_ehd_scalar_2d!(qfield, q_f_in)
    poisson_setup = phi_scheme === :direct ? ehd_poisson_setup(Nx, Ny, p.eps; xbc=:neumann, backend=backend) : nothing
    if phi_scheme === :direct
        ehd_poisson_solve!(phi, poisson_setup, qfield)
        compute_electric_field_fd_2d!(Ex, Ey, phi, :neumann, Nx, Ny)
    else
        compute_electric_field_2d!(Ex, Ey, phi_f_in, p.tau_U)
    end

    hist_capacity = cld(max_cycles, history_interval)
    umax_history = Vector{FT}(undef, hist_capacity)
    cycle_history = Vector{Int}(undef, hist_capacity)
    phi_iters_last = 0
    phi_rel_last = FT(Inf)
    q_rel_last = FT(Inf)
    hist_count = 0
    steps_done = 0
    phi_check_every = 8

    t0 = time_ns()
    for cycle in 1:max_cycles
        steps_done = cycle
        sample_cycle = (cycle % history_interval == 0) || (cycle == max_cycles)

        if phi_scheme === :direct
            ehd_poisson_solve!(phi, poisson_setup, qfield)
            compute_electric_field_fd_2d!(Ex, Ey, phi, :neumann, Nx, Ny)
            phi_iters_last = 1
            phi_rel_last = zero(FT)
        elseif phi_substeps === nothing
            for iter in 1:phi_max_iter
                copyto!(phi_prev, phi)
                collide_electric_potential_2d!(phi_f_in, qfield, p.eps, p.omega_U, p.nu_U)
                stream_wall_x_wall_y_2d!(phi_f_out, phi_f_in, Nx, Ny)
                compute_ehd_scalar_2d!(phi, phi_f_out)
                apply_phi_nee_box_2d!(phi_f_out, phi, one(FT), zero(FT), Nx, Ny)
                compute_ehd_scalar_2d!(phi, phi_f_out)
                phi_f_in, phi_f_out = phi_f_out, phi_f_in
                phi_iters_last = iter
                if iter % phi_check_every == 0 || iter == phi_max_iter
                    ehd_rel_change_2d!(diag, phi, phi_prev, Nx, Ny)
                    copyto!(diag_host, diag)
                    phi_rel_last = diag_host[1]
                    phi_rel_last <= phi_tol && break
                    iter == phi_max_iter &&
                        error("Electric potential solve did not converge within $(phi_max_iter) iterations. Last relative change: $(phi_rel_last).")
                end
            end
            compute_electric_field_2d!(Ex, Ey, phi_f_in, p.tau_U)
        else
            sample_cycle && copyto!(phi_prev, phi)
            for _ in 1:Int(phi_substeps)
                collide_electric_potential_2d!(phi_f_in, qfield, p.eps, p.omega_U, p.nu_U)
                stream_wall_x_wall_y_2d!(phi_f_out, phi_f_in, Nx, Ny)
                compute_ehd_scalar_2d!(phi, phi_f_out)
                apply_phi_nee_box_2d!(phi_f_out, phi, one(FT), zero(FT), Nx, Ny)
                compute_ehd_scalar_2d!(phi, phi_f_out)
                phi_f_in, phi_f_out = phi_f_out, phi_f_in
            end
            if sample_cycle
                ehd_rel_change_2d!(diag, phi, phi_prev, Nx, Ny)
                copyto!(diag_host, diag)
                phi_rel_last = diag_host[1]
            end
            phi_iters_last = Int(phi_substeps)
            compute_electric_field_2d!(Ex, Ey, phi_f_in, p.tau_U)
        end

        compute_macroscopic_guo_field_2d!(rho, ux, uy, f_in, Fx_prev, Fy_prev, Nx, Ny)
        enforce_free_side_macros_2d!(ux, uy, Nx, Ny)

        sample_cycle && copyto!(q_prev, qfield)
        if charge_scheme == :srt
            collide_electric_charge_srt_2d!(q_f_in, ux, uy, Ex, Ey, p.tau_q, p.K)
        else
            collide_electric_charge_regularized_2d!(q_f_in, ux, uy, Ex, Ey, p.tau_q, p.K)
        end
        stream_wall_x_wall_y_2d!(q_f_out, q_f_in, Nx, Ny)
        compute_ehd_scalar_2d!(qfield, q_f_out)
        apply_charge_nee_box_2d!(q_f_out, qfield, ux, uy, Ex, Ey, p.q_inj, zero(FT), p.K, Nx, Ny)
        compute_ehd_scalar_2d!(qfield, q_f_out)
        if sample_cycle
            ehd_rel_change_2d!(diag, qfield, q_prev, Nx, Ny)
            copyto!(diag_host, diag)
            q_rel_last = diag_host[1]
            diag_host[2] == zero(FT) && error("Charge field became non-finite at cycle $(cycle).")
        end
        q_f_in, q_f_out = q_f_out, q_f_in

        compute_coulomb_force_2d!(Fx, Fy, qfield, Ex, Ey, Nx, Ny)
        _project_coulomb_force_rows!(Fx, Fy, is_solid, force_projection)
        if ns_scheme == :bgk
            collide_guo_field_2d!(f_in, is_solid, Fx, Fy, p.omega)
        else
            ehd_collide_mrt_2d!(f_in, Fx, Fy, is_solid, p.nu)
        end
        stream_wall_x_wall_y_2d!(f_out, f_in, Nx, Ny)
        apply_free_slip_sidewalls_2d!(f_out, Nx, Ny)
        f_in, f_out = f_out, f_in

        if sample_cycle
            compute_macroscopic_guo_field_2d!(rho, ux, uy, f_in, Fx, Fy, Nx, Ny)
            enforce_free_side_macros_2d!(ux, uy, Nx, Ny)
            ehd_maxspeed_2d!(diag, ux, uy, Nx, Ny)
            copyto!(diag_host, diag)
            umax = diag_host[1]
            diag_host[2] == zero(FT) && error("Flow velocity became non-finite at cycle $(cycle).")
            umax > FT(velocity_stop) &&
                error("Flow field became unstable at cycle $(cycle): max(|u|) = $(umax).")
            hist_count += 1
            umax_history[hist_count] = umax
            cycle_history[hist_count] = cycle
        end
        copyto!(Fx_prev, Fx)
        copyto!(Fy_prev, Fy)
    end
    t1 = time_ns()

    phi_scheme === :lbm && compute_ehd_scalar_2d!(phi, phi_f_in)
    compute_ehd_scalar_2d!(qfield, q_f_in)
    if phi_scheme === :direct
        compute_electric_field_fd_2d!(Ex, Ey, phi, :neumann, Nx, Ny)
    else
        compute_electric_field_2d!(Ex, Ey, phi_f_in, p.tau_U)
    end
    compute_macroscopic_guo_field_2d!(rho, ux, uy, f_in, Fx, Fy, Nx, Ny)
    enforce_free_side_macros_2d!(ux, uy, Nx, Ny)

    return (ux=Array(ux), uy=Array(uy), rho=Array(rho), q=Array(qfield),
            phi=Array(phi), Ex=Array(Ex), Ey=Array(Ey), Fx=Array(Fx), Fy=Array(Fy),
            umax_history=umax_history[1:hist_count],
            cycle_history=cycle_history[1:hist_count],
            steps=steps_done, Nx=Nx, Ny=Ny, A=A, C=C, M=M, T=T,
            Ma_E=Ma_E, alpha=alpha, perturb_amplitude=perturb_amplitude,
            perturb_mode=perturb_mode, force_projection=force_projection,
            charge_scheme=charge_scheme, ns_scheme=ns_scheme,
            phi_scheme=phi_scheme, phi_substeps=phi_substeps,
            phi_iters_last=phi_iters_last, phi_rel_last=phi_rel_last,
            q_rel_change=q_rel_last, params=p,
            ns_collision=(ns_scheme == :bgk ? :bgk_guo : :mrt_guo_moment),
            sidewall_bc=:free_slip_ported,
            loop_ms_per_step=steps_done > 0 ? (t1 - t0) / 1e6 / steps_done : 0.0)
end
