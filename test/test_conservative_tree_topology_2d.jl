using Test
using KrakeNN

function _topology_cell_id(topology, level, i, j)
    for (id, cell) in pairs(topology.cells)
        if cell.level == level && cell.i == i && cell.j == j
            return id
        end
    end
    error("cell not found: level=$level i=$i j=$j")
end

function _topology_routes_for(topology, src, q)
    return [route for route in topology.routes if route.src == src && route.q == q]
end

function _topology_link_for(topology, src, q)
    for link in topology.links
        if link.src == src && link.q == q
            return link
        end
    end
    error("link not found: src=$src q=$q")
end

@inline function _test_inside_parent_patch(i, j, patch)
    return first(patch.parent_i_range) <= i <= last(patch.parent_i_range) &&
           first(patch.parent_j_range) <= j <= last(patch.parent_j_range)
end

@inline function _test_inside_fine_patch(i, j, patch)
    return 2 * first(patch.parent_i_range) - 1 <= i <= 2 * last(patch.parent_i_range) &&
           2 * first(patch.parent_j_range) - 1 <= j <= 2 * last(patch.parent_j_range)
end

@inline function _test_inside_leaf_domain(i, j, Nx, Ny)
    return 1 <= i <= 2 * Nx && 1 <= j <= 2 * Ny
end

@testset "Conservative tree topology 2D" begin
    Nx, Ny = 9, 10
    patch = create_conservative_tree_patch_2d(3:5, 4:6)
    topology = create_conservative_tree_topology_2d(Nx, Ny, patch)
    n_parent_patch = length(patch.parent_i_range) * length(patch.parent_j_range)
    n_active = Nx * Ny - n_parent_patch + 4 * n_parent_patch

    @testset "active cells and volumes" begin
        @test length(topology.cells) == Nx * Ny + 4 * n_parent_patch
        @test length(topology.active_cells) == n_active
        @test isapprox(active_volume(topology), Nx * Ny; atol=1e-14, rtol=0)

        for J in 1:Ny, I in 1:Nx
            id = _topology_cell_id(topology, 0, I, J)
            @test topology.cells[id].active == !_test_inside_parent_patch(I, J, patch)
            @test topology.cells[id].metrics.volume == 1.0
        end

        for J in patch.parent_j_range, I in patch.parent_i_range
            parent = _topology_cell_id(topology, 0, I, J)
            @test !topology.cells[parent].active
            for iy in 1:2, ix in 1:2
                id = _topology_cell_id(topology, 1, 2 * I - 2 + ix, 2 * J - 2 + iy)
                @test topology.cells[id].active
                @test topology.cells[id].parent == parent
                @test topology.cells[id].metrics.volume == 0.25
            end
        end
    end

    @testset "logical links and compact tables" begin
        @test length(topology.links) == 9 * length(topology.active_cells)

        link_counts = Dict{Int,Int}()
        for link in topology.links
            link_counts[link.src] = get(link_counts, link.src, 0) + 1
        end
        @test all(get(link_counts, id, 0) == 9 for id in topology.active_cells)

        link_indices = vcat(topology.same_level_links,
                            topology.coarse_to_fine_links,
                            topology.fine_to_coarse_links,
                            topology.boundary_links)
        @test sort(link_indices) == collect(eachindex(topology.links))
        @test length(topology.same_level_links) >
              length(topology.coarse_to_fine_links) + length(topology.fine_to_coarse_links)

        route_indices = vcat(topology.direct_routes,
                             topology.interface_routes,
                             topology.boundary_routes)
        @test sort(route_indices) == collect(eachindex(topology.routes))

        for idx in topology.same_level_links
            @test topology.links[idx].kind == SAME_LEVEL
        end
        for idx in topology.coarse_to_fine_links
            @test topology.links[idx].kind == COARSE_TO_FINE
        end
        for idx in topology.fine_to_coarse_links
            @test topology.links[idx].kind == FINE_TO_COARSE
        end
        for idx in topology.boundary_links
            @test topology.links[idx].kind == BOUNDARY
        end
    end

    @testset "boundary and interface links are localized" begin
        for idx in topology.boundary_links
            link = topology.links[idx]
            cell = topology.cells[link.src]
            cx = d2q9_cx(link.q)
            cy = d2q9_cy(link.q)
            if cell.level == 0
                @test !(1 <= cell.i + cx <= Nx && 1 <= cell.j + cy <= Ny)
            else
                @test !_test_inside_leaf_domain(cell.i + cx, cell.j + cy, Nx, Ny)
            end
        end

        for idx in topology.coarse_to_fine_links
            link = topology.links[idx]
            cell = topology.cells[link.src]
            @test cell.level == 0
            @test cell.active
            @test _test_inside_parent_patch(cell.i + d2q9_cx(link.q),
                                            cell.j + d2q9_cy(link.q),
                                            patch)
        end

        for idx in topology.fine_to_coarse_links
            link = topology.links[idx]
            cell = topology.cells[link.src]
            i_dst = cell.i + d2q9_cx(link.q)
            j_dst = cell.j + d2q9_cy(link.q)
            @test cell.level == 1
            @test !_test_inside_fine_patch(i_dst, j_dst, patch)
            @test _test_inside_leaf_domain(i_dst, j_dst, Nx, Ny)
            for route in _topology_routes_for(topology, link.src, link.q)
                @test topology.cells[route.dst].level == 0
                @test topology.cells[route.dst].active
            end
        end
    end

    @testset "routes conserve every logical packet" begin
        route_sums = Dict{Tuple{Int,Int},Float64}()
        for route in topology.routes
            @test route.weight > 0
            if route.kind == ROUTE_BOUNDARY
                @test route.dst == 0
            else
                @test topology.cells[route.dst].active
            end
            key = (route.src, route.q)
            route_sums[key] = get(route_sums, key, 0.0) + route.weight
        end

        for link in topology.links
            @test isapprox(route_sums[(link.src, link.q)], 1.0; atol=1e-14, rtol=0)
        end
    end

    @testset "coarse to fine routes reproduce leaf-equivalent fractions" begin
        coarse_src = zeros(Float64, Nx, Ny, 9)
        for q in 1:9, J in 1:Ny, I in 1:Nx
            coarse_src[I, J, q] = 100q + 10I + J / 10
        end

        expected_fine = zeros(size(patch.fine_F))
        expected_coarse = zeros(size(coarse_src))
        routed = zeros(size(patch.fine_F))
        routed_coarse = zeros(size(coarse_src))
        i0 = 2 * first(patch.parent_i_range) - 1
        j0 = 2 * first(patch.parent_j_range) - 1
        for idx in topology.coarse_to_fine_links
            link = topology.links[idx]
            src_cell = topology.cells[link.src]
            packet = coarse_src[src_cell.i, src_cell.j, link.q]
            cx = d2q9_cx(link.q)
            cy = d2q9_cy(link.q)
            for iy in 1:2, ix in 1:2
                i_dst = 2 * src_cell.i - 2 + ix + cx
                j_dst = 2 * src_cell.j - 2 + iy + cy
                if _test_inside_fine_patch(i_dst, j_dst, patch)
                    expected_fine[i_dst - i0 + 1, j_dst - j0 + 1, link.q] +=
                        0.25 * packet
                else
                    I_dst = (i_dst + 1) >>> 1
                    J_dst = (j_dst + 1) >>> 1
                    expected_coarse[I_dst, J_dst, link.q] += 0.25 * packet
                end
            end

            for route in _topology_routes_for(topology, link.src, link.q)
                dst_cell = topology.cells[route.dst]
                if dst_cell.level == 1
                    routed[dst_cell.i - i0 + 1, dst_cell.j - j0 + 1, link.q] +=
                        packet * route.weight
                else
                    routed_coarse[dst_cell.i, dst_cell.j, link.q] +=
                        packet * route.weight
                end
            end
        end

        @test isapprox(routed, expected_fine; atol=1e-14, rtol=0)
        @test isapprox(routed_coarse, expected_coarse; atol=1e-14, rtol=0)
    end

    @testset "interface route primitives are explicit" begin
        west_source = _topology_cell_id(topology, 0, 2, 5)
        west_link = _topology_link_for(topology, west_source, 2)
        west_routes = _topology_routes_for(topology, west_source, 2)
        west_split = [route for route in west_routes if route.kind == SPLIT_FACE]
        west_residual = [route for route in west_routes if route.kind == DIRECT]
        @test west_link.kind == COARSE_TO_FINE
        @test length(west_routes) == 3
        @test length(west_split) == 2
        @test length(west_residual) == 1
        @test all(isapprox(route.weight, 0.25; atol=0, rtol=0) for route in west_split)
        @test sort(collect((topology.cells[route.dst].i, topology.cells[route.dst].j)
                           for route in west_split)) == [(5, 9), (5, 10)]
        @test west_residual[1].dst == west_source
        @test isapprox(west_residual[1].weight, 0.5; atol=0, rtol=0)

        southwest_source = _topology_cell_id(topology, 0, 2, 3)
        corner_routes = _topology_routes_for(topology, southwest_source, 6)
        corner_split = [route for route in corner_routes if route.kind == SPLIT_CORNER]
        corner_residual = [route for route in corner_routes if route.kind == DIRECT]
        @test length(corner_routes) == 4
        @test length(corner_split) == 1
        @test length(corner_residual) == 3
        @test corner_split[1].weight == 0.25
        @test (topology.cells[corner_split[1].dst].i,
               topology.cells[corner_split[1].dst].j) == (5, 7)
        @test all(route.weight == 0.25 for route in corner_residual)
        @test sort(collect((topology.cells[route.dst].i, topology.cells[route.dst].j)
                           for route in corner_residual)) == [(2, 3), (2, 4), (3, 3)]

        fine_face_src_a = _topology_cell_id(topology, 1, 5, 9)
        fine_face_src_b = _topology_cell_id(topology, 1, 5, 10)
        coarse_west = _topology_cell_id(topology, 0, 2, 5)
        for src in (fine_face_src_a, fine_face_src_b)
            routes = _topology_routes_for(topology, src, 4)
            @test length(routes) == 1
            @test routes[1].dst == coarse_west
            @test routes[1].kind == COALESCE_FACE
            @test routes[1].weight == 1.0
        end

        fine_corner_src = _topology_cell_id(topology, 1, 5, 7)
        coarse_southwest = _topology_cell_id(topology, 0, 2, 3)
        routes = _topology_routes_for(topology, fine_corner_src, 8)
        @test length(routes) == 1
        @test routes[1].dst == coarse_southwest
        @test routes[1].kind == COALESCE_CORNER
        @test routes[1].weight == 1.0
    end

    @testset "leaf-equivalent coarse mode splits same-level coarse routes" begin
        leaf_topology = create_conservative_tree_topology_2d(
            Nx, Ny, patch; coarse_route_mode=:leaf_equivalent)
        src = _topology_cell_id(leaf_topology, 0, 1, 5)
        link = _topology_link_for(leaf_topology, src, 2)
        routes = _topology_routes_for(leaf_topology, src, 2)

        @test link.kind == SAME_LEVEL
        @test length(routes) == 2
        @test all(route.kind == DIRECT for route in routes)
        @test all(isapprox(route.weight, 0.5; atol=0, rtol=0) for route in routes)
        @test sort(collect((leaf_topology.cells[route.dst].i,
                            leaf_topology.cells[route.dst].j)
                           for route in routes)) == [(1, 5), (2, 5)]

        boundary_routes = _topology_routes_for(leaf_topology, src, 4)
        direct = [route for route in boundary_routes if route.kind == DIRECT]
        boundary = [route for route in boundary_routes if route.kind == ROUTE_BOUNDARY]
        @test length(direct) == 1
        @test length(boundary) == 1
        @test direct[1].dst == src
        @test isapprox(direct[1].weight, 0.5; atol=0, rtol=0)
        @test isapprox(boundary[1].weight, 0.5; atol=0, rtol=0)

        @test_throws ArgumentError create_conservative_tree_topology_2d(
            Nx, Ny, patch; coarse_route_mode=:bad)
    end

    @testset "fine to coarse routes reproduce local coalesce primitives" begin
        fine_src = zeros(size(patch.fine_F))
        for q in 1:9, j in axes(fine_src, 2), i in axes(fine_src, 1)
            fine_src[i, j, q] = q / 2 + i / 128 + j / 256 + i * j / 8192
        end

        i0 = 2 * first(patch.parent_i_range) - 1
        j0 = 2 * first(patch.parent_j_range) - 1

        west_parent_block = @view fine_src[1:2, 3:4, :]
        coarse_west = _topology_cell_id(topology, 0, 2, 5)
        west_routed = 0.0
        for src in (_topology_cell_id(topology, 1, 5, 9),
                    _topology_cell_id(topology, 1, 5, 10))
            packet = fine_src[topology.cells[src].i - i0 + 1,
                              topology.cells[src].j - j0 + 1, 4]
            routes = _topology_routes_for(topology, src, 4)
            @test length(routes) == 1
            @test routes[1].dst == coarse_west
            west_routed += packet * routes[1].weight
        end
        @test isapprox(west_routed,
                       coalesce_fine_to_coarse_face_F(west_parent_block, 4, :west);
                       atol=1e-14, rtol=0)

        southwest_parent_block = @view fine_src[1:2, 1:2, :]
        fine_corner_src = _topology_cell_id(topology, 1, 5, 7)
        coarse_southwest = _topology_cell_id(topology, 0, 2, 3)
        routes = _topology_routes_for(topology, fine_corner_src, 8)
        @test length(routes) == 1
        @test routes[1].dst == coarse_southwest
        packet = fine_src[topology.cells[fine_corner_src].i - i0 + 1,
                          topology.cells[fine_corner_src].j - j0 + 1, 8]
        @test isapprox(packet * routes[1].weight,
                       coalesce_fine_to_coarse_corner_F(southwest_parent_block, 8, :southwest);
                       atol=1e-14, rtol=0)
    end

    @testset "argument checks" begin
        @test_throws ArgumentError create_conservative_tree_topology_2d(2, 10, patch)
        @test_throws ArgumentError create_conservative_tree_topology_2d(Nx, Ny, patch; coarse_volume=0.0)
    end

    @testset "Morton keys" begin
        @test morton_key_2d(1, 1) == 0
        @test morton_key_2d(2, 1) == 1
        @test morton_key_2d(1, 2) == 2
        @test morton_key_2d(2, 2) == 3
        @test morton_key_2d(3, 1) == 4
        @test_throws ArgumentError morton_key_2d(0, 1)
        @test_throws ArgumentError morton_key_2d(1, 0)
    end

    @testset "block packing remaps active cells and routes" begin
        packed = pack_conservative_tree_topology_2d(topology; cells_per_block=8)

        @test packed.cells_per_block == 8
        @test length(packed.packed_cell_ids) == length(topology.active_cells)
        @test sort(packed.packed_cell_ids) == sort(topology.active_cells)
        @test length(packed.packed_morton_keys) == length(packed.packed_cell_ids)
        @test all(block.count <= packed.cells_per_block for block in packed.blocks)
        @test all(block.count > 0 for block in packed.blocks)

        for (block_id, block) in pairs(packed.blocks)
            ids = @view packed.packed_cell_ids[block.first_cell:block.first_cell+block.count-1]
            keys = @view packed.packed_morton_keys[block.first_cell:block.first_cell+block.count-1]
            @test all(topology.cells[id].level == block.level for id in ids)
            @test collect(keys) == sort(collect(keys))
            @test block.morton_first == keys[1]

            for local_index in 1:block.count
                cell_id = ids[local_index]
                @test packed.logical_cell_to_block[cell_id] == block_id
                @test packed.logical_cell_to_local[cell_id] == local_index
            end
        end

        level_by_block = [block.level for block in packed.blocks]
        @test level_by_block == sort(level_by_block)

        for (id, cell) in pairs(topology.cells)
            if cell.active
                @test packed.logical_cell_to_block[id] > 0
                @test packed.logical_cell_to_local[id] > 0
            else
                @test packed.logical_cell_to_block[id] == 0
                @test packed.logical_cell_to_local[id] == 0
            end
        end

        @test length(packed.routes) == length(topology.routes)
        route_indices = vcat(packed.direct_routes,
                             packed.interface_routes,
                             packed.boundary_routes)
        @test sort(route_indices) == collect(eachindex(packed.routes))
        @test length(packed.direct_routes) == length(topology.direct_routes)
        @test length(packed.interface_routes) == length(topology.interface_routes)
        @test length(packed.boundary_routes) == length(topology.boundary_routes)

        for (route_index, route) in pairs(topology.routes)
            packed_route = packed.routes[route_index]
            @test packed_route.src_block == packed.logical_cell_to_block[route.src]
            @test packed_route.src_local == packed.logical_cell_to_local[route.src]
            @test packed_route.q == route.q
            @test packed_route.weight == route.weight
            @test packed_route.kind == route.kind

            if route.kind == ROUTE_BOUNDARY
                @test packed_route.dst_block == 0
                @test packed_route.dst_local == 0
            else
                @test packed_route.dst_block == packed.logical_cell_to_block[route.dst]
                @test packed_route.dst_local == packed.logical_cell_to_local[route.dst]
            end
        end

        @test_throws ArgumentError pack_conservative_tree_topology_2d(
            topology; cells_per_block=0)
    end
end
