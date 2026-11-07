using Test
using KernelAbstractions
using KrakeNN

@testset "EHD MRT electroconvection smoke" begin
    common = (Nx=31, Ny=48, C=10.0, M=10.0, Ma_E=1e-2, alpha=1e-4,
              max_cycles=3_000, phi_substeps=1, history_interval=25,
              charge_scheme=:regularized, force_projection=:none,
              perturb_amplitude=1e-4, backend=KernelAbstractions.CPU(), FT=Float64)

    bgk_sup = KrakeNN.run_electroconvection_2d(; common..., T=220.0, ns_scheme=:bgk)
    mrt_sup = KrakeNN.run_electroconvection_2d(; common..., T=220.0, ns_scheme=:mrt)
    bgk_sub = KrakeNN.run_electroconvection_2d(; common..., T=150.0, ns_scheme=:bgk)
    mrt_sub = KrakeNN.run_electroconvection_2d(; common..., T=150.0, ns_scheme=:mrt)

    bgk_final = last(bgk_sup.umax_history)
    mrt_final = last(mrt_sup.umax_history)
    bgk_sub_final = last(bgk_sub.umax_history)
    mrt_sub_peak = maximum(mrt_sub.umax_history)
    mrt_sub_final = last(mrt_sub.umax_history)

    @info "EHD MRT smoke" bgk_T220_maxu=bgk_final mrt_T220_maxu=mrt_final bgk_T150_final=bgk_sub_final mrt_T150_final=mrt_sub_final

    @test all(isfinite, bgk_sup.umax_history)
    @test all(isfinite, mrt_sup.umax_history)
    @test all(isfinite, bgk_sub.umax_history)
    @test all(isfinite, mrt_sub.umax_history)
    @test bgk_final > 2 * bgk_sub_final
    @test mrt_final > 1.4 * mrt_sub_final
    @test mrt_sub_final < mrt_sub_peak / 2
end
