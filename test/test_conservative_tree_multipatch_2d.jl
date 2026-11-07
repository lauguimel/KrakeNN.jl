using Test
using KrakeNN

@testset "Conservative tree multipatch ownership 2D" begin
    @testset "disjoint patches own parent and leaf cells" begin
        patch_set = create_conservative_tree_patch_set_2d(
            12, 10, ((2:3, 2:4), (7:8, 6:7)))

        @test length(patch_set.patches) == 2
        @test conservative_tree_patch_owner_counts_2d(patch_set) == [6, 4]
        @test isapprox(active_volume(patch_set), 120.0; atol=1e-14, rtol=0)

        @test conservative_tree_parent_owner_2d(patch_set, 2, 2) == 1
        @test conservative_tree_parent_owner_2d(patch_set, 8, 7) == 2
        @test conservative_tree_parent_owner_2d(patch_set, 1, 1) == 0

        @test conservative_tree_leaf_owner_2d(patch_set, 3, 3) == 1
        @test conservative_tree_leaf_owner_2d(patch_set, 16, 14) == 2
        @test conservative_tree_leaf_owner_2d(patch_set, 1, 1) == 0

        mask = active_coarse_mask(patch_set)
        @test !mask[2, 2]
        @test !mask[8, 7]
        @test mask[1, 1]
    end

    @testset "overlap and domain checks reject invalid ownership" begin
        @test_throws ArgumentError create_conservative_tree_patch_set_2d(
            12, 10, ((2:4, 2:4), (4:6, 3:5)))
        @test_throws ArgumentError create_conservative_tree_patch_set_2d(
            12, 10, ((0:2, 2:4),))
        @test_throws ArgumentError conservative_tree_parent_owner_2d(
            create_conservative_tree_patch_set_2d(12, 10, ((2:3, 2:4),)), 0, 1)
        @test_throws ArgumentError conservative_tree_leaf_owner_2d(
            create_conservative_tree_patch_set_2d(12, 10, ((2:3, 2:4),)), 25, 1)
    end

    @testset ".krk Refine helpers build ownership tables" begin
        text = """
        Simulation amr_multi D2Q9
        Domain L = 8 x 4  N = 32 x 16
        Physics nu = 0.1

        Refine fineA { region = [1.0, 0.5, 2.0, 1.5], ratio = 2 }
        Refine fineB { region = [4.0, 2.0, 5.0, 3.0], ratio = 2 }

        Boundary west periodic
        Boundary east periodic
        Boundary south wall
        Boundary north wall

        Run 10 steps
        """
        setup = parse_kraken(text)
        ranges = conservative_tree_patch_ranges_from_krk_refines_2d(
            setup.domain, setup.refinements)
        patch_set = create_conservative_tree_patch_set_from_krk_2d(setup)

        @test ranges == [(5:8, 3:6), (17:20, 9:12)]
        @test conservative_tree_patch_owner_counts_2d(patch_set) == [16, 16]
        @test conservative_tree_parent_owner_2d(patch_set, 5, 3) == 1
        @test conservative_tree_parent_owner_2d(patch_set, 20, 12) == 2
    end

    @testset ".krk helper rejects unsupported refine contracts" begin
        ratio3 = parse_kraken("""
        Simulation bad_ratio D2Q9
        Domain L = 1 x 1  N = 32 x 32
        Physics nu = 0.1
        Refine fine { region = [0.0, 0.0, 0.5, 0.5], ratio = 3 }
        Boundary west wall
        Boundary east wall
        Boundary south wall
        Boundary north wall
        Run 1 steps
        """)
        @test_throws ArgumentError create_conservative_tree_patch_set_from_krk_2d(ratio3)

        ratio4 = parse_kraken("""
        Simulation bad_ratio4 D2Q9
        Domain L = 1 x 1  N = 32 x 32
        Physics nu = 0.1
        Refine fine { region = [0.0, 0.0, 0.5, 0.5], ratio = 4 }
        Boundary west wall
        Boundary east wall
        Boundary south wall
        Boundary north wall
        Run 1 steps
        """)
        @test_throws ArgumentError create_conservative_tree_patch_set_from_krk_2d(ratio4)

        nested = parse_kraken("""
        Simulation bad_parent D2Q9
        Domain L = 1 x 1  N = 32 x 32
        Physics nu = 0.1
        Refine fine { region = [0.0, 0.0, 0.5, 0.5], ratio = 2, parent = base }
        Boundary west wall
        Boundary east wall
        Boundary south wall
        Boundary north wall
        Run 1 steps
        """)
        @test_throws ArgumentError create_conservative_tree_patch_set_from_krk_2d(nested)

        ref3d = parse_kraken("""
        Simulation bad_3d D3Q19
        Domain L = 1 x 1 x 1  N = 32 x 32 x 32
        Physics nu = 0.1
        Refine fine { region = [0.0, 0.0, 0.0, 0.5, 0.5, 0.5], ratio = 2 }
        Boundary west wall
        Boundary east wall
        Boundary south wall
        Boundary north wall
        Boundary bottom wall
        Boundary top wall
        Run 1 steps
        """)
        @test_throws ArgumentError create_conservative_tree_patch_set_from_krk_2d(ref3d)
    end
end
