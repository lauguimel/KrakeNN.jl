using Test
using KernelAbstractions
using KrakeNN

function _frozen_phi_lbm_box(q_cpu, analytic_phi, p; phi_tol=1e-6, phi_max_iter=30000)
    Nx, Ny = size(q_cpu)
    backend = KernelAbstractions.CPU()
    phi_f_in = KernelAbstractions.zeros(backend, Float64, Nx, Ny, 9)
    phi_f_out = KernelAbstractions.zeros(backend, Float64, Nx, Ny, 9)
    phi = KernelAbstractions.zeros(backend, Float64, Nx, Ny)
    phi_prev = KernelAbstractions.zeros(backend, Float64, Nx, Ny)
    qfield = KernelAbstractions.zeros(backend, Float64, Nx, Ny)
    diag = KernelAbstractions.zeros(backend, Float64, 2)
    diag_host = Vector{Float64}(undef, 2)

    phi_init = KrakeNN._fill_phi_populations!(zeros(Float64, Nx, Ny, 9), analytic_phi, Nx, Ny, Float64)
    copyto!(phi_f_in, phi_init)
    copyto!(phi_f_out, phi_init)
    copyto!(qfield, q_cpu)
    KrakeNN.compute_ehd_scalar_2d!(phi, phi_f_in)

    phi_rel = Inf
    for iter in 1:phi_max_iter
        copyto!(phi_prev, phi)
        KrakeNN.collide_electric_potential_2d!(phi_f_in, qfield, p.eps, p.omega_U, p.nu_U)
        KrakeNN.stream_wall_x_wall_y_2d!(phi_f_out, phi_f_in, Nx, Ny)
        KrakeNN.compute_ehd_scalar_2d!(phi, phi_f_out)
        KrakeNN.apply_phi_nee_box_2d!(phi_f_out, phi, 1.0, 0.0, Nx, Ny)
        KrakeNN.compute_ehd_scalar_2d!(phi, phi_f_out)
        KrakeNN.ehd_rel_change_2d!(diag, phi, phi_prev, Nx, Ny)
        copyto!(diag_host, diag)
        phi_rel = diag_host[1]
        phi_f_in, phi_f_out = phi_f_out, phi_f_in
        phi_rel <= phi_tol && return Array(phi), phi_rel, iter
    end
    error("Frozen-charge LBM phi solve did not converge; last rel=$(phi_rel).")
end

@testset "EHD direct phi hydrostatic base state" begin
    result = KrakeNN.run_ehd_hydrostatic_2d(;
        Nx=8,
        Ny=96,
        C=10.0,
        alpha=1e-4,
        charge_scheme=:srt,
        phi_scheme=:direct,
        backend=KernelAbstractions.CPU(),
        FT=Float64,
        charge_tol=1e-8,
        max_steps=100000,
    )

    @test result.phi_scheme === :direct
    @test result.err_q < 1e-2
    @test result.err_E < 1e-2
    @test result.xvar_q ≤ 1000eps(Float64) * maximum(abs, result.q)
    @test result.xvar_phi ≤ 1000eps(Float64) * maximum(abs, result.phi)
end

@testset "EHD direct phi frozen-charge consistency" begin
    Nx, Ny = 31, 48
    C = 10.0
    p = KrakeNN._ehd_ec_lattice_params(Ny, C, 10.0, 175.0, 1e-2, 1e-4, 1.0, 0.3; FT=Float64)
    analytic = KrakeNN.ehd_hydrostatic_profiles(C, Ny; FT=Float64)
    q_profile = p.q_inj .* analytic.q_star
    q_cpu = zeros(Float64, Nx, Ny)
    for j in 1:Ny
        q_cpu[:, j] .= q_profile[j]
    end
    q_cpu[:, 1] .= p.q_inj

    qfield = KernelAbstractions.zeros(KernelAbstractions.CPU(), Float64, Nx, Ny)
    phi_direct = KernelAbstractions.zeros(KernelAbstractions.CPU(), Float64, Nx, Ny)
    copyto!(qfield, q_cpu)
    setup = KrakeNN.ehd_poisson_setup(Nx, Ny, p.eps; xbc=:neumann)
    KrakeNN.ehd_poisson_solve!(phi_direct, setup, qfield)

    phi_lbm, phi_rel, phi_iters = _frozen_phi_lbm_box(q_cpu, analytic.phi, p)
    linf = maximum(abs.(Array(phi_direct) .- phi_lbm)) /
           max(maximum(abs, Array(phi_direct)), floatmin(Float64))
    @info "EHD direct frozen phi consistency" rel_linf=linf lbm_rel=phi_rel lbm_iters=phi_iters
    println("EHD direct frozen phi relative Linf = $(linf)")

    @test linf ≤ 5e-3
end

@testset "EHD direct phi MRT electroconvection smoke" begin
    common = (Nx=31, Ny=48, C=10.0, M=10.0, Ma_E=1e-2, alpha=1e-4,
              max_cycles=3_000, phi_scheme=:direct, history_interval=25,
              charge_scheme=:regularized, ns_scheme=:mrt, force_projection=:none,
              perturb_amplitude=1e-4, backend=KernelAbstractions.CPU(), FT=Float64)

    sup = KrakeNN.run_electroconvection_2d(; common..., T=220.0)
    sub = KrakeNN.run_electroconvection_2d(; common..., T=150.0)

    sup_final = last(sup.umax_history)
    sub_peak = maximum(sub.umax_history)
    sub_final = last(sub.umax_history)

    @info "EHD direct MRT smoke" sup_T220_final=sup_final sub_T150_peak=sub_peak sub_T150_final=sub_final

    @test all(isfinite, sup.umax_history)
    @test all(isfinite, sub.umax_history)
    @test sup_final > 1.4 * sub_final
    @test sub_final < sub_peak / 2
end
