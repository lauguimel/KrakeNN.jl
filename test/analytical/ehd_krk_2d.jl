using Test
using KernelAbstractions
using KrakeNN

@testset "EHD hydrostatic .krk runner path" begin
    fixture = joinpath(@__DIR__, "..", "..", "benchmarks", "krk", "ehd",
                       "hydrostatic_fast.krk")
    setup = KrakeNN.load_kraken(fixture)
    result = KrakeNN.run_simulation(setup; backend=KernelAbstractions.CPU(),
                                   T=Float64)

    @info "EHD .krk hydrostatic" err_q=result.err_q err_E=result.err_E xvar_q=result.xvar_q xvar_phi=result.xvar_phi steps=result.steps

    @test result.err_q < 1e-2
    @test result.err_E < 1e-2
    @test result.xvar_q ≤ 100eps(Float64) * maximum(abs, result.q)
    @test result.xvar_phi ≤ 100eps(Float64) * maximum(abs, result.phi)
end
