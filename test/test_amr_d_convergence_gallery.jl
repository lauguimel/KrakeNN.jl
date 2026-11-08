using Test
using KrakeNN

include(joinpath(dirname(@__DIR__), "benchmarks", "amr_d_convergence_gallery_2d.jl"))

@testset "AMR D convergence gallery" begin
    @testset ".krk templates parse and nested probe is explicit" begin
        case_dir = joinpath(dirname(@__DIR__), "benchmarks", "krk",
                            "amr_d_convergence_2d")
        files = sort(filter(endswith(".krk"), readdir(case_dir; join=true)))
        @test length(files) == 9

        for file in files
            setup = load_kraken(file)
            @test setup.lattice == :D2Q9
            @test setup.domain.Nx > 0
            @test setup.domain.Ny > 0
            @test setup.max_steps > 0
            @test !isempty(setup.refinements)
        end

        nested = joinpath(case_dir, "cylinder_nested4_probe.krk")
        probe = probe_nested4_cylinder_2d(nested)
        @test probe.parsed_levels == 4
        @test probe.max_level == 4
        @test probe.spec_supported == true
        @test probe.supported == false
        @test occursin("nested obstacle", probe.reason)
    end

    @testset "smoke rows remain finite" begin
        rows = run_amr_d_convergence_gallery_2d(
            ; flows=(:couette, :poiseuille_xband, :bfs, :square, :cylinder),
            scales=(1,), base_steps=2, avg_window=1)
        @test length(rows) == 15
        @test count(row -> row.method == :amr_route_native, rows) == 5
        @test all(row -> isfinite(row.primary_error), rows)
        @test all(row -> isfinite(row.mass_rel_drift), rows)
        @test all(row -> row.cell_count > 0, rows)
    end
end
