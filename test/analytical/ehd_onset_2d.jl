using Test
using KernelAbstractions
using KrakeNN

@testset "Coupled EHD electroconvection onset bracket" begin
    # Coarse CPU canary for the coupled qE -> BGK+Guo NS path. The PRE
    # critical point shifts upward on this grid/model: T=175 and 190 were
    # marginal after 50k cycles, while T=220 grew cleanly. Measured on this
    # machine during setup: about 21 s per branch with phi_substeps=1.
    common = (Nx=59, Ny=96, C=10.0, M=10.0, Ma_E=1e-2, alpha=1e-4,
              max_cycles=50_000, phi_substeps=1, history_interval=100,
              charge_scheme=:regularized, force_projection=:none,
              backend=KernelAbstractions.CPU(), FT=Float64)

    sub = KrakeNN.run_electroconvection_2d(; common..., T=150.0)
    sup = KrakeNN.run_electroconvection_2d(; common..., T=220.0)

    sub_peak = maximum(sub.umax_history)
    sub_final = last(sub.umax_history)
    sup_final = last(sup.umax_history)

    @info "EHD onset canary" sub_T=sub.T sup_T=sup.T steps=sub.steps sub_peak sub_final sup_final

    @test sub_final < sub_peak / 10
    @test sup_final > 100 * sub_final
end
