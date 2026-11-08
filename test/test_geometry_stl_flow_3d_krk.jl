using KrakeNN
using Test

@testset "Geometry: 3D STL LI-BB flow units twin" begin
    root = normpath(joinpath(@__DIR__, ".."))
    example_dir = joinpath(root, "examples", "geometry_stl")
    mm_case = joinpath(example_dir, "sphere_stl_3d_mm.krk")
    lu_case = joinpath(example_dir, "sphere_stl_3d_lu.krk")

    cd(root) do
        setup_mm = load_kraken(mm_case)
        setup_lu = load_kraken(lu_case)

        res_mm = run_simulation(setup_mm)
        res_lu = run_simulation(setup_lu)

        @test maximum(abs.(res_mm.ux .- res_lu.ux)) <= 1e-8
        @test maximum(abs.(res_mm.uy .- res_lu.uy)) <= 1e-8
        @test maximum(abs.(res_mm.uz .- res_lu.uz)) <= 1e-8
        @test maximum(abs.(res_mm.ρ .- res_lu.ρ)) <= 1e-8

        @test !any(isnan, res_lu.ux)
        @test !any(isnan, res_lu.uy)
        @test !any(isnan, res_lu.uz)
        @test !any(isnan, res_lu.ρ)
        @test all(0.6 .< res_lu.ρ .< 1.6)
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    println("PASS M-GEO-6")
end
