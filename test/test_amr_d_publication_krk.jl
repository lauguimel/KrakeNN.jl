using Test
using KrakeNN

@testset "AMR D publication .krk cases" begin
    case_dir = joinpath(dirname(@__DIR__), "benchmarks", "krk", "amr_d_publication_2d")
    files = sort(filter(endswith(".krk"), readdir(case_dir; join=true)))

    @test length(files) == 6
    for file in files
        setup = load_kraken(file)
        @test setup.lattice == :D2Q9
        @test setup.domain.Nx > 0
        @test setup.domain.Ny > 0
        @test setup.max_steps > 0
        @test length(setup.refinements) == 1

        ref = only(setup.refinements)
        @test ref.ratio == 2
        @test ref.parent == ""
        @test !ref.is_3d

        ranges = conservative_tree_patch_ranges_from_krk_refines_2d(
            setup.domain, setup.refinements)
        i_range, j_range = only(ranges)
        @test first(i_range) >= 1
        @test last(i_range) <= setup.domain.Nx
        @test j_range == 1:setup.domain.Ny
    end
end
