using Kraken
using Test

# .krk reproducibility (Kraken mandate §3.3) for the 3D FVFD log-conformation
# planar-extension canary. Mirrors test/test_viscoelastic_krk.jl: load the
# benchmark .krk, dispatch to run_viscoelastic_fvfd_extensional_3d, and assert
# the analytical Oldroyd-B fixed point C_xx = 1/(1-2λε̇), C_yy = 1/(1+2λε̇).

function _ext_center_mean(A)
    Nx, Ny, Nz = size(A)
    i1 = max(1, Nx ÷ 2)
    i2 = min(Nx, Nx ÷ 2 + 1)
    j1 = max(1, Ny ÷ 2)
    j2 = min(Ny, Ny ÷ 2 + 1)
    return sum(@view A[i1:i2, j1:j2, :]) / ((i2 - i1 + 1) * (j2 - j1 + 1) * Nz)
end

@testset "Viscoelastic .krk (FVFD extensional 3D)" begin
    root = normpath(joinpath(@__DIR__, ".."))
    case_path = joinpath(root, "benchmarks", "krk", "viscoelastic", "extensional_oldroyd_b.krk")

    cd(root) do
        setup = load_kraken(case_path)
        @test setup.lattice == :D3Q19
        @test :viscoelastic in setup.modules
        @test setup.max_steps == 1000

        r = run_simulation(case_path)

        @test haskey(r, :C_xx)
        @test haskey(r, :C_yy)
        @test r.velocity_mode === :imposed
        @test r.completed_steps == 1000
        @test all(isfinite, r.C_xx)
        @test all(isfinite, r.C_yy)
        @test all(isfinite, r.C_zz)

        lambda_epsilon = 50.0 * 0.005          # = 0.25
        Cxx_ref = 1 / (1 - 2 * lambda_epsilon) # = 2.0
        Cyy_ref = 1 / (1 + 2 * lambda_epsilon) # = 2/3

        Cxx_meas = _ext_center_mean(r.C_xx)
        Cyy_meas = _ext_center_mean(r.C_yy)
        rel_Cxx = abs(Cxx_meas - Cxx_ref) / Cxx_ref
        rel_Cyy = abs(Cyy_meas - Cyy_ref) / Cyy_ref

        @test rel_Cxx <= 0.01
        @test rel_Cyy <= 0.01

        println("KRK extensional: Cxx=$(Cxx_meas) (ref $(Cxx_ref), rel $(rel_Cxx))")
        println("KRK extensional: Cyy=$(Cyy_meas) (ref $(Cyy_ref), rel $(rel_Cyy))")
    end
end
