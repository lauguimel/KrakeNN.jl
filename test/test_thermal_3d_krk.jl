using KrakeNN
using Test

@testset "Thermal 3D natural convection .krk" begin
    root = normpath(joinpath(@__DIR__, ".."))
    case_path = joinpath(root, "examples", "natural_convection_3d.krk")

    cd(root) do
        setup = load_kraken(case_path)
        @test setup.lattice == :D3Q19
        @test setup.domain.Nz == setup.domain.Nx
        @test occursin("natural_convection", lowercase(setup.name))
        @test setup.physics.params[:Ra] ≈ 1e3
        @test setup.physics.params[:Pr] ≈ 0.71

        result = run_simulation(case_path)

        @test haskey(result, :Temp)
        @test haskey(result, :uz)
        @test all(isfinite, result.ρ)
        @test all(isfinite, result.ux)
        @test all(isfinite, result.uy)
        @test all(isfinite, result.uz)
        @test all(isfinite, result.Temp)
        @test isfinite(result.Nu)

        ρ_mean = sum(result.ρ) / length(result.ρ)
        ρ_max_dev = maximum(abs.(result.ρ .- ρ_mean)) / ρ_mean
        @test abs(ρ_mean - 1.0) < 0.05
        @test ρ_max_dev < 0.10

        west_mean = sum(result.Temp[1, :, :]) / length(result.Temp[1, :, :])
        east_mean = sum(result.Temp[end, :, :]) / length(result.Temp[end, :, :])
        @test west_mean > east_mean
        @test maximum(result.Temp) - minimum(result.Temp) > 0.1
        @test maximum(abs.(result.ux)) +
              maximum(abs.(result.uy)) +
              maximum(abs.(result.uz)) > 1e-12
    end
end
