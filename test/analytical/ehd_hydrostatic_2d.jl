using Test
using KernelAbstractions
using KrakeNN

@testset "EHD hydrostatic D2Q9 base state" begin
    result = KrakeNN.run_ehd_hydrostatic_2d(;
        Nx=8,
        Ny=96,
        C=10.0,
        alpha=1e-4,
        charge_scheme=:srt,
        backend=KernelAbstractions.CPU(),
        FT=Float64,
        charge_tol=1e-8,
        max_steps=100000,
    )

    @test result.err_q < 1e-2
    @test result.err_E < 1e-2
    @test result.xvar_q ≤ 100eps(Float64) * maximum(abs, result.q)
    @test result.xvar_phi ≤ 100eps(Float64) * maximum(abs, result.phi)
end
