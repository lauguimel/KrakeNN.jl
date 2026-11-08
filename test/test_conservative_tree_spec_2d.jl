using Test
using KrakeNN
using Random

@testset "Conservative tree nested spec 2D" begin
    @testset "programmatic nested tree preserves active volume" begin
        blocks = [
            ConservativeTreeRefineBlock2D("outer", 3:6, 2:5),
            ConservativeTreeRefineBlock2D("inner", 7:8, 5:6; parent="outer"),
        ]
        spec = create_conservative_tree_spec_2d(8, 6, blocks)

        @test spec.Nx == 8
        @test spec.Ny == 6
        @test spec.max_level == 2
        @test length(spec.cells) == 48 + 4 * 16 + 4 * 4
        @test length(spec.active_cells) == 48 - 16 + 64 - 4 + 16
        @test isapprox(active_volume(spec), 48.0; atol=1e-14, rtol=0)

        parent_id = conservative_tree_cell_id_2d(spec, 0, 3, 2)
        @test parent_id != 0
        @test !conservative_tree_is_active_leaf_2d(spec, parent_id)
        @test conservative_tree_children_2d(spec, parent_id) != (0, 0, 0, 0)

        inner_parent_id = conservative_tree_cell_id_2d(spec, 1, 7, 5)
        @test inner_parent_id != 0
        @test !conservative_tree_is_active_leaf_2d(spec, inner_parent_id)
        @test conservative_tree_children_2d(spec, inner_parent_id) != (0, 0, 0, 0)

        inner_leaf_id = conservative_tree_cell_id_2d(spec, 2, 13, 9)
        @test inner_leaf_id != 0
        @test conservative_tree_is_active_leaf_2d(spec, inner_leaf_id)
        @test spec.cells[inner_leaf_id].parent == inner_parent_id
    end

    @testset "four-level static nesting canary" begin
        blocks = [
            ConservativeTreeRefineBlock2D("L1", 17:48, 9:24),
            ConservativeTreeRefineBlock2D("L2", 49:80, 25:40; parent="L1"),
            ConservativeTreeRefineBlock2D("L3", 113:144, 57:72; parent="L2"),
            ConservativeTreeRefineBlock2D("L4", 241:272, 121:136; parent="L3"),
        ]
        spec = create_conservative_tree_spec_2d(64, 32, blocks)

        @test spec.max_level == 4
        @test isapprox(active_volume(spec), 64 * 32; atol=1e-12, rtol=0)
        @test spec.refine_level["L4"] == 4
        @test spec.refine_i_range["L4"] == 481:544
        @test spec.refine_j_range["L4"] == 241:272
    end

    @testset "invalid ownership and balance are rejected" begin
        @test_throws ArgumentError create_conservative_tree_spec_2d(8, 8, [
            ConservativeTreeRefineBlock2D("a", 2:4, 2:4),
            ConservativeTreeRefineBlock2D("b", 3:5, 3:5),
        ])

        @test_throws ArgumentError create_conservative_tree_spec_2d(8, 8, [
            ConservativeTreeRefineBlock2D("child", 3:4, 3:4; parent="missing"),
        ])

        @test_throws ArgumentError create_conservative_tree_spec_2d(8, 8, [
            ConservativeTreeRefineBlock2D("outer", 3:5, 3:5),
            ConservativeTreeRefineBlock2D("child", 1:2, 1:2; parent="outer"),
        ])

        unbalanced = [
            ConservativeTreeRefineBlock2D("outer", 3:4, 3:4),
            ConservativeTreeRefineBlock2D("edge_child", 5:5, 5:5; parent="outer"),
        ]
        @test_throws ArgumentError create_conservative_tree_spec_2d(8, 8, unbalanced)
        @test create_conservative_tree_spec_2d(
            8, 8, unbalanced; balance=2).max_level == 2
    end

    @testset ".krk nested refine blocks build a static spec" begin
        setup = parse_kraken("""
        Simulation nested_static D2Q9
        Domain L = 8 x 4  N = 16 x 8
        Physics nu = 0.1

        Refine outer { region = [2.0, 1.0, 6.0, 3.0], ratio = 2 }
        Refine inner { region = [3.0, 1.5, 4.0, 2.0], ratio = 2, parent = outer }

        Boundary west periodic
        Boundary east periodic
        Boundary south wall
        Boundary north wall
        Run 1 steps
        """)

        blocks = conservative_tree_refine_blocks_from_krk_2d(
            setup.domain, setup.refinements)
        @test blocks[1].name == "outer"
        @test blocks[1].i_range == 5:12
        @test blocks[1].j_range == 3:6
        @test blocks[2].name == "inner"
        @test blocks[2].parent == "outer"
        @test blocks[2].i_range == 13:16
        @test blocks[2].j_range == 7:8

        spec = create_conservative_tree_spec_from_krk_2d(setup)
        @test spec.max_level == 2
        @test isapprox(active_volume(spec), 16 * 8; atol=1e-12, rtol=0)
    end

    @testset ".krk ratio > 2 expands to buffered nested blocks" begin
        setup = parse_kraken("""
        Simulation auto_nested D2Q9
        Domain L = 8 x 4  N = 16 x 8
        Physics nu = 0.1

        Refine zoom { region = [3.0, 1.5, 4.0, 2.5], ratio = 4 }

        Boundary west periodic
        Boundary east periodic
        Boundary south wall
        Boundary north wall
        Run 1 steps
        """)

        blocks = conservative_tree_refine_blocks_from_krk_2d(
            setup.domain, setup.refinements)
        @test length(blocks) == 2
        @test blocks[1].name == "zoom_L1"
        @test blocks[1].parent == ""
        @test blocks[1].i_range == 6:9
        @test blocks[1].j_range == 3:6
        @test blocks[2].name == "zoom"
        @test blocks[2].parent == "zoom_L1"
        @test blocks[2].i_range == 13:16
        @test blocks[2].j_range == 7:10

        spec = create_conservative_tree_spec_from_krk_2d(setup)
        @test spec.max_level == 2
        @test spec.refine_level["zoom"] == 2
        @test isapprox(active_volume(spec), 16 * 8; atol=1e-12, rtol=0)
    end

    @testset ".krk ratio 8 expands to three nested levels" begin
        setup = parse_kraken("""
        Simulation auto_nested3 D2Q9
        Domain L = 8 x 4  N = 16 x 8
        Physics nu = 0.1

        Refine zoom { region = [3.0, 1.5, 4.0, 2.5], ratio = 8 }

        Boundary west periodic
        Boundary east periodic
        Boundary south wall
        Boundary north wall
        Run 1 steps
        """)

        blocks = conservative_tree_refine_blocks_from_krk_2d(
            setup.domain, setup.refinements)
        @test length(blocks) == 3
        @test blocks[1].name == "zoom_L1"
        @test blocks[1].parent == ""
        @test blocks[1].i_range == 6:9
        @test blocks[1].j_range == 3:6
        @test blocks[2].name == "zoom_L2"
        @test blocks[2].parent == "zoom_L1"
        @test blocks[2].i_range == 12:17
        @test blocks[2].j_range == 6:11
        @test blocks[3].name == "zoom"
        @test blocks[3].parent == "zoom_L2"
        @test blocks[3].i_range == 25:32
        @test blocks[3].j_range == 13:20

        spec = create_conservative_tree_spec_from_krk_2d(setup)
        @test spec.max_level == 3
        @test spec.refine_level["zoom"] == 3
        @test isapprox(active_volume(spec), 16 * 8; atol=1e-12, rtol=0)
    end

    @testset "existing cylinder nested4 probe is a static DSL canary" begin
        setup = load_kraken("benchmarks/krk/amr_d_convergence_2d/cylinder_nested4_probe.krk")
        spec = create_conservative_tree_spec_from_krk_2d(setup)

        @test spec.max_level == 4
        @test length(spec.active_cells) == 8040
        @test isapprox(active_volume(spec), 24 * 14; atol=1e-12, rtol=0)
    end

    @testset ".krk static spec rejects bad nested contracts" begin
        missing_parent = parse_kraken("""
        Simulation missing_parent D2Q9
        Domain L = 1 x 1  N = 16 x 16
        Physics nu = 0.1
        Refine child { region = [0.25, 0.25, 0.5, 0.5], ratio = 2, parent = nope }
        Boundary west wall
        Boundary east wall
        Boundary south wall
        Boundary north wall
        Run 1 steps
        """)
        @test_throws ArgumentError create_conservative_tree_spec_from_krk_2d(missing_parent)

        child_outside = parse_kraken("""
        Simulation child_outside D2Q9
        Domain L = 1 x 1  N = 16 x 16
        Physics nu = 0.1
        Refine outer { region = [0.25, 0.25, 0.5, 0.5], ratio = 2 }
        Refine child { region = [0.75, 0.75, 0.875, 0.875], ratio = 2, parent = outer }
        Boundary west wall
        Boundary east wall
        Boundary south wall
        Boundary north wall
        Run 1 steps
        """)
        @test_throws ArgumentError create_conservative_tree_spec_from_krk_2d(child_outside)

        non_power_two = parse_kraken("""
        Simulation non_power_two D2Q9
        Domain L = 1 x 1  N = 16 x 16
        Physics nu = 0.1
        Refine odd { region = [0.25, 0.25, 0.5, 0.5], ratio = 6 }
        Boundary west wall
        Boundary east wall
        Boundary south wall
        Boundary north wall
        Run 1 steps
        """)
        @test_throws ArgumentError create_conservative_tree_spec_from_krk_2d(non_power_two)
    end

    @testset "recursive ledger coalesce preserves active populations" begin
        spec = create_conservative_tree_spec_2d(16, 12, [
            ConservativeTreeRefineBlock2D("L1", 5:12, 3:10),
            ConservativeTreeRefineBlock2D("L2", 13:20, 7:14; parent="L1"),
            ConservativeTreeRefineBlock2D("L3", 29:36, 17:24; parent="L2"),
        ])
        F = allocate_conservative_tree_F_2d(spec)
        rng = MersenneTwister(0x51d)
        for cell_id in spec.active_cells, q in 1:9
            F[cell_id, q] = rand(rng)
        end

        active_before = active_population_sums_F_2d(F, spec)
        coalesce_conservative_tree_ledgers_F_2d!(F, spec)

        @test isapprox(level_population_sums_F_2d(F, spec, 0),
                       active_before; atol=1e-11, rtol=0)
        for (cell_id, children) in pairs(spec.children)
            children == (0, 0, 0, 0) && continue
            for q in 1:9
                child_sum = F[children[1], q] + F[children[2], q] +
                            F[children[3], q] + F[children[4], q]
                @test isapprox(F[cell_id, q], child_sum; atol=1e-12, rtol=0)
            end
        end
    end

    @testset "recursive ledger explosion roundtrips coarse populations" begin
        spec = create_conservative_tree_spec_2d(16, 12, [
            ConservativeTreeRefineBlock2D("L1", 5:12, 3:10),
            ConservativeTreeRefineBlock2D("L2", 13:20, 7:14; parent="L1"),
            ConservativeTreeRefineBlock2D("L3", 29:36, 17:24; parent="L2"),
            ConservativeTreeRefineBlock2D("L4", 61:68, 37:44; parent="L3"),
        ])
        F = allocate_conservative_tree_F_2d(spec)
        rng = MersenneTwister(0x5eaf)
        for (cell_id, cell) in pairs(spec.cells)
            cell.level == 0 || continue
            for q in 1:9
                F[cell_id, q] = rand(rng)
            end
        end
        F_level0 = copy(F[1:(spec.Nx * spec.Ny), :])
        level0_before = level_population_sums_F_2d(F, spec, 0)

        explode_conservative_tree_ledgers_F_2d!(F, spec)
        @test isapprox(active_population_sums_F_2d(F, spec),
                       level0_before; atol=1e-12, rtol=0)

        coalesce_conservative_tree_ledgers_F_2d!(F, spec)
        @test isapprox(F[1:(spec.Nx * spec.Ny), :],
                       F_level0; atol=1e-12, rtol=0)
    end

    @testset "ledger helpers reject incompatible matrices" begin
        spec = create_conservative_tree_spec_2d(4, 4, [
            ConservativeTreeRefineBlock2D("L1", 2:3, 2:3),
        ])
        @test_throws ArgumentError active_population_sums_F_2d(zeros(3, 9), spec)
        @test_throws ArgumentError coalesce_conservative_tree_ledgers_F_2d!(
            zeros(length(spec.cells), 8), spec)
    end

    @testset "route table conserves one packet per source direction" begin
        spec = create_conservative_tree_spec_2d(6, 5, [
            ConservativeTreeRefineBlock2D("fine", 3:4, 2:3),
        ])
        table = create_conservative_tree_route_table_2d(spec)

        @test length(table.links) == length(spec.active_cells) * 9
        @test !isempty(table.direct_routes)
        @test !isempty(table.coarse_to_fine_links)
        @test !isempty(table.fine_to_coarse_links)
        @test !isempty(table.boundary_links)

        route_weights = Dict{Tuple{Int,Int},Float64}()
        for route in table.routes
            key = (route.src, route.q)
            route_weights[key] = get(route_weights, key, 0.0) + route.weight
            if route.dst != 0
                src_level = spec.cells[route.src].level
                dst_level = spec.cells[route.dst].level
                @test abs(src_level - dst_level) <= 1
            end
        end
        for src in spec.active_cells, q in 1:9
            @test isapprox(route_weights[(src, q)], 1.0; atol=1e-14, rtol=0)
        end

        coarse_west = conservative_tree_cell_id_2d(spec, 0, 2, 2)
        split_routes = [route for route in table.routes
                        if route.src == coarse_west && route.q == 2]
        @test any(route.kind == SPLIT_FACE for route in split_routes)
        @test isapprox(sum(route.weight for route in split_routes),
                       1.0; atol=1e-14, rtol=0)

        fine_west = conservative_tree_cell_id_2d(spec, 1, 5, 3)
        coalesce_routes = [route for route in table.routes
                           if route.src == fine_west && route.q == 4]
        @test length(coalesce_routes) == 1
        @test coalesce_routes[1].kind == COALESCE_FACE
        @test spec.cells[coalesce_routes[1].dst].level == 0
    end

    @testset "route table handles direct and boundary packets" begin
        spec = create_conservative_tree_spec_2d(4, 4, ConservativeTreeRefineBlock2D[])
        table = create_conservative_tree_route_table_2d(spec)

        center = conservative_tree_cell_id_2d(spec, 0, 2, 2)
        east = conservative_tree_cell_id_2d(spec, 0, 3, 2)
        east_routes = [route for route in table.routes
                       if route.src == center && route.q == 2]
        @test length(east_routes) == 1
        @test east_routes[1].dst == east
        @test east_routes[1].kind == DIRECT
        @test isapprox(east_routes[1].weight, 1.0; atol=1e-14, rtol=0)

        west_boundary_src = conservative_tree_cell_id_2d(spec, 0, 1, 1)
        west_routes = [route for route in table.routes
                       if route.src == west_boundary_src && route.q == 4]
        @test length(west_routes) == 1
        @test west_routes[1].dst == 0
        @test west_routes[1].kind == ROUTE_BOUNDARY
        @test isapprox(west_routes[1].weight, 1.0; atol=1e-14, rtol=0)
    end

    @testset "route table can wrap x while keeping y physical" begin
        spec = create_conservative_tree_spec_2d(4, 4, ConservativeTreeRefineBlock2D[])
        table = create_conservative_tree_route_table_2d(spec; periodic_x=true)

        west_src = conservative_tree_cell_id_2d(spec, 0, 1, 2)
        west_dst = conservative_tree_cell_id_2d(spec, 0, 4, 2)
        west_routes = [route for route in table.routes
                       if route.src == west_src && route.q == 4]
        @test length(west_routes) == 1
        @test west_routes[1].dst == west_dst
        @test west_routes[1].kind == DIRECT

        east_src = conservative_tree_cell_id_2d(spec, 0, 4, 2)
        east_dst = conservative_tree_cell_id_2d(spec, 0, 1, 2)
        east_routes = [route for route in table.routes
                       if route.src == east_src && route.q == 2]
        @test length(east_routes) == 1
        @test east_routes[1].dst == east_dst
        @test east_routes[1].kind == DIRECT

        south_src = conservative_tree_cell_id_2d(spec, 0, 2, 1)
        south_routes = [route for route in table.routes
                        if route.src == south_src && route.q == 5]
        @test length(south_routes) == 1
        @test south_routes[1].dst == 0
        @test south_routes[1].kind == ROUTE_BOUNDARY
    end

    @testset "route table builds for the existing nested4 cylinder canary" begin
        setup = load_kraken("benchmarks/krk/amr_d_convergence_2d/cylinder_nested4_probe.krk")
        spec = create_conservative_tree_spec_from_krk_2d(setup)
        table = create_conservative_tree_route_table_2d(spec)

        @test length(table.links) == length(spec.active_cells) * 9
        @test !isempty(table.coarse_to_fine_links)
        @test !isempty(table.fine_to_coarse_links)
        @test !isempty(table.interface_routes)
    end

    @testset "route streaming scatters direct split and coalesce packets" begin
        spec = create_conservative_tree_spec_2d(6, 5, [
            ConservativeTreeRefineBlock2D("fine", 3:4, 2:3),
        ])
        table = create_conservative_tree_route_table_2d(spec)
        Fin = allocate_conservative_tree_F_2d(spec)
        Fout = allocate_conservative_tree_F_2d(spec)

        coarse_west = conservative_tree_cell_id_2d(spec, 0, 2, 2)
        Fin[coarse_west, 2] = 4.0
        stream_conservative_tree_routes_F_2d!(Fout, Fin, spec, table)
        @test isapprox(sum(Fout[:, 2]), 4.0; atol=1e-14, rtol=0)
        @test count(!iszero, Fout[:, 2]) > 1

        fill!(Fin, 0.0)
        fine_west = conservative_tree_cell_id_2d(spec, 1, 5, 3)
        coarse_dst = conservative_tree_cell_id_2d(spec, 0, 2, 2)
        Fin[fine_west, 4] = 2.5
        stream_conservative_tree_routes_F_2d!(Fout, Fin, spec, table)
        @test isapprox(Fout[coarse_dst, 4], 2.5; atol=1e-14, rtol=0)

        base_spec = create_conservative_tree_spec_2d(
            4, 4, ConservativeTreeRefineBlock2D[])
        base_table = create_conservative_tree_route_table_2d(base_spec)
        base_in = allocate_conservative_tree_F_2d(base_spec)
        base_out = allocate_conservative_tree_F_2d(base_spec)
        src = conservative_tree_cell_id_2d(base_spec, 0, 2, 2)
        dst = conservative_tree_cell_id_2d(base_spec, 0, 3, 2)
        base_in[src, 2] = 7.0
        stream_conservative_tree_routes_F_2d!(base_out, base_in,
                                              base_spec, base_table)
        @test isapprox(base_out[dst, 2], 7.0; atol=1e-14, rtol=0)
    end

    @testset "single-level route streaming preserves closed rest state" begin
        spec = create_conservative_tree_spec_2d(6, 5, [
            ConservativeTreeRefineBlock2D("fine", 3:4, 2:3),
        ])
        table = create_conservative_tree_route_table_2d(spec)
        Fin = allocate_conservative_tree_F_2d(spec)
        Fout = allocate_conservative_tree_F_2d(spec)
        w = weights(D2Q9())
        for cell_id in spec.active_cells
            volume = spec.cells[cell_id].metrics.volume
            for q in 1:9
                Fin[cell_id, q] = w[q] * volume
            end
        end

        stream_conservative_tree_routes_F_2d!(
            Fout, Fin, spec, table; boundary=:bounceback)

        @test maximum(abs.(Fout - Fin)) <= 1e-14
    end

    @testset "bounceback route streaming preserves closed nested rest state" begin
        spec = create_conservative_tree_spec_2d(16, 12, [
            ConservativeTreeRefineBlock2D("L1", 5:12, 3:10),
            ConservativeTreeRefineBlock2D("L2", 13:20, 7:14; parent="L1"),
            ConservativeTreeRefineBlock2D("L3", 29:36, 17:24; parent="L2"),
            ConservativeTreeRefineBlock2D("L4", 61:68, 37:44; parent="L3"),
        ])
        table = create_conservative_tree_route_table_2d(spec)
        Fin = allocate_conservative_tree_F_2d(spec)
        Fout = allocate_conservative_tree_F_2d(spec)
        w = weights(D2Q9())
        for cell_id in spec.active_cells
            volume = spec.cells[cell_id].metrics.volume
            for q in 1:9
                Fin[cell_id, q] = w[q] * volume
            end
        end

        stream_conservative_tree_routes_F_2d!(
            Fout, Fin, spec, table; boundary=:bounceback)

        @test isapprox(sum(Fout), sum(Fin); atol=1e-12, rtol=0)
        @test isapprox(sum(active_population_sums_F_2d(Fout, spec)),
                       sum(active_population_sums_F_2d(Fin, spec));
                       atol=1e-12, rtol=0)
        @test maximum(abs.(Fout - Fin)) <= 1e-14
    end

    @testset "route streaming rejects bad matrices and boundary policies" begin
        spec = create_conservative_tree_spec_2d(4, 4, ConservativeTreeRefineBlock2D[])
        table = create_conservative_tree_route_table_2d(spec)
        F = allocate_conservative_tree_F_2d(spec)
        @test_throws ArgumentError stream_conservative_tree_routes_F_2d!(
            zeros(3, 9), F, spec, table)
        @test_throws ArgumentError stream_conservative_tree_routes_F_2d!(
            F, F, spec, table; boundary=:periodic)
    end
end
