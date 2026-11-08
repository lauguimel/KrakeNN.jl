using Test
using KrakeNN

function _topology_cell_id_3d(topology, level, i, j, k)
    for (id, cell) in pairs(topology.cells)
        if cell.level == level && cell.i == i && cell.j == j && cell.k == k
            return id
        end
    end
    error("cell not found: level=$level i=$i j=$j k=$k")
end

function _topology_routes_for_3d(topology, src, q)
    return [route for route in topology.routes if route.src == src && route.q == q]
end

function _topology_link_for_3d(topology, src, q)
    for link in topology.links
        if link.src == src && link.q == q
            return link
        end
    end
    error("link not found: src=$src q=$q")
end

@inline function _test_inside_parent_patch_3d(i, j, k, patch)
    return first(patch.parent_i_range) <= i <= last(patch.parent_i_range) &&
           first(patch.parent_j_range) <= j <= last(patch.parent_j_range) &&
           first(patch.parent_k_range) <= k <= last(patch.parent_k_range)
end

@inline function _test_inside_fine_patch_3d(i, j, k, patch)
    return 2 * first(patch.parent_i_range) - 1 <= i <= 2 * last(patch.parent_i_range) &&
           2 * first(patch.parent_j_range) - 1 <= j <= 2 * last(patch.parent_j_range) &&
           2 * first(patch.parent_k_range) - 1 <= k <= 2 * last(patch.parent_k_range)
end

@inline function _test_inside_leaf_domain_3d(i, j, k, Nx, Ny, Nz)
    return 1 <= i <= 2 * Nx && 1 <= j <= 2 * Ny && 1 <= k <= 2 * Nz
end

@inline function _test_offset_count_3d(di, dj, dk)
    return (di == 0 ? 0 : 1) + (dj == 0 ? 0 : 1) + (dk == 0 ? 0 : 1)
end

@inline function _test_face_from_offset_3d(di, dj, dk)
    if di < 0 && dj == 0 && dk == 0
        return :west
    elseif di > 0 && dj == 0 && dk == 0
        return :east
    elseif di == 0 && dj < 0 && dk == 0
        return :south
    elseif di == 0 && dj > 0 && dk == 0
        return :north
    elseif di == 0 && dj == 0 && dk < 0
        return :bottom
    elseif di == 0 && dj == 0 && dk > 0
        return :top
    else
        error("offset does not identify one face")
    end
end

@inline function _test_edge_from_offset_3d(di, dj, dk)
    if di < 0 && dj < 0 && dk == 0
        return :southwest
    elseif di > 0 && dj < 0 && dk == 0
        return :southeast
    elseif di < 0 && dj > 0 && dk == 0
        return :northwest
    elseif di > 0 && dj > 0 && dk == 0
        return :northeast
    elseif di < 0 && dj == 0 && dk < 0
        return :bottomwest
    elseif di > 0 && dj == 0 && dk < 0
        return :bottomeast
    elseif di < 0 && dj == 0 && dk > 0
        return :topwest
    elseif di > 0 && dj == 0 && dk > 0
        return :topeast
    elseif di == 0 && dj < 0 && dk < 0
        return :bottomsouth
    elseif di == 0 && dj > 0 && dk < 0
        return :bottomnorth
    elseif di == 0 && dj < 0 && dk > 0
        return :topsouth
    elseif di == 0 && dj > 0 && dk > 0
        return :topnorth
    else
        error("offset does not identify one edge")
    end
end

@testset "Conservative tree topology 3D" begin
    Nx, Ny, Nz = 7, 8, 6
    patch = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
    topology = create_conservative_tree_topology_3d(Nx, Ny, Nz, patch)
    n_parent_patch = length(patch.parent_i_range) *
                     length(patch.parent_j_range) *
                     length(patch.parent_k_range)
    n_active = Nx * Ny * Nz - n_parent_patch + 8 * n_parent_patch

    @testset "active cells and volumes" begin
        @test length(topology.cells) == Nx * Ny * Nz + 8 * n_parent_patch
        @test length(topology.active_cells) == n_active
        @test isapprox(active_volume(topology), Nx * Ny * Nz; atol=1e-14, rtol=0)

        for K in 1:Nz, J in 1:Ny, I in 1:Nx
            id = _topology_cell_id_3d(topology, 0, I, J, K)
            @test topology.cells[id].active == !_test_inside_parent_patch_3d(I, J, K, patch)
            @test topology.cells[id].metrics.volume == 1.0
        end

        for K in patch.parent_k_range, J in patch.parent_j_range, I in patch.parent_i_range
            parent = _topology_cell_id_3d(topology, 0, I, J, K)
            @test !topology.cells[parent].active
            for iz in 1:2, iy in 1:2, ix in 1:2
                id = _topology_cell_id_3d(topology, 1,
                                          2 * I - 2 + ix,
                                          2 * J - 2 + iy,
                                          2 * K - 2 + iz)
                @test topology.cells[id].active
                @test topology.cells[id].parent == parent
                @test topology.cells[id].metrics.volume == 0.125
            end
        end
    end

    @testset "logical links and route weights" begin
        @test length(topology.links) == 19 * length(topology.active_cells)

        link_counts = Dict{Int,Int}()
        for link in topology.links
            link_counts[link.src] = get(link_counts, link.src, 0) + 1
        end
        @test all(get(link_counts, id, 0) == 19 for id in topology.active_cells)

        link_indices = vcat(topology.same_level_links,
                            topology.coarse_to_fine_links,
                            topology.fine_to_coarse_links,
                            topology.boundary_links)
        @test sort(link_indices) == collect(eachindex(topology.links))

        route_indices = vcat(topology.direct_routes,
                             topology.interface_routes,
                             topology.boundary_routes)
        @test sort(route_indices) == collect(eachindex(topology.routes))

        route_sums = Dict{Tuple{Int,Int},Float64}()
        for route in topology.routes
            @test route.weight > 0
            if route.kind == ROUTE_BOUNDARY_3D
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

    @testset "boundary and interface links are localized" begin
        for idx in topology.boundary_links
            link = topology.links[idx]
            cell = topology.cells[link.src]
            cx = d3q19_cx(link.q)
            cy = d3q19_cy(link.q)
            cz = d3q19_cz(link.q)
            if cell.level == 0
                @test !(1 <= cell.i + cx <= Nx &&
                        1 <= cell.j + cy <= Ny &&
                        1 <= cell.k + cz <= Nz)
            else
                @test !_test_inside_leaf_domain_3d(cell.i + cx,
                                                   cell.j + cy,
                                                   cell.k + cz,
                                                   Nx, Ny, Nz)
            end
        end

        for idx in topology.coarse_to_fine_links
            link = topology.links[idx]
            cell = topology.cells[link.src]
            @test cell.level == 0
            @test cell.active
            @test _test_inside_parent_patch_3d(cell.i + d3q19_cx(link.q),
                                               cell.j + d3q19_cy(link.q),
                                               cell.k + d3q19_cz(link.q),
                                               patch)
        end

        for idx in topology.fine_to_coarse_links
            link = topology.links[idx]
            cell = topology.cells[link.src]
            i_dst = cell.i + d3q19_cx(link.q)
            j_dst = cell.j + d3q19_cy(link.q)
            k_dst = cell.k + d3q19_cz(link.q)
            @test cell.level == 1
            @test !_test_inside_fine_patch_3d(i_dst, j_dst, k_dst, patch)
            @test _test_inside_leaf_domain_3d(i_dst, j_dst, k_dst, Nx, Ny, Nz)
            for route in _topology_routes_for_3d(topology, link.src, link.q)
                @test topology.cells[route.dst].level == 0
                @test topology.cells[route.dst].active
            end
        end
    end

    @testset "interface route primitives are explicit" begin
        west_source = _topology_cell_id_3d(topology, 0, 2, 4, 2)
        west_link = _topology_link_for_3d(topology, west_source, 2)
        west_routes = _topology_routes_for_3d(topology, west_source, 2)
        @test west_link.kind == COARSE_TO_FINE
        @test length(west_routes) == 4
        @test all(route.kind == SPLIT_FACE_3D for route in west_routes)
        @test all(isapprox(route.weight, 0.25; atol=0, rtol=0) for route in west_routes)
        @test sort(collect((topology.cells[route.dst].i,
                            topology.cells[route.dst].j,
                            topology.cells[route.dst].k)
                           for route in west_routes)) ==
              [(5, 7, 3), (5, 7, 4), (5, 8, 3), (5, 8, 4)]

        southwest_source = _topology_cell_id_3d(topology, 0, 2, 3, 2)
        edge_routes = _topology_routes_for_3d(topology, southwest_source, 8)
        @test length(edge_routes) == 2
        @test all(route.kind == SPLIT_EDGE_3D for route in edge_routes)
        @test all(route.weight == 0.5 for route in edge_routes)
        @test sort(collect((topology.cells[route.dst].i,
                            topology.cells[route.dst].j,
                            topology.cells[route.dst].k)
                           for route in edge_routes)) == [(5, 7, 3), (5, 7, 4)]

        fine_face_src = _topology_cell_id_3d(topology, 1, 5, 7, 3)
        coarse_west = _topology_cell_id_3d(topology, 0, 2, 4, 2)
        face_routes = _topology_routes_for_3d(topology, fine_face_src, 3)
        @test length(face_routes) == 1
        @test face_routes[1].dst == coarse_west
        @test face_routes[1].kind == COALESCE_FACE_3D

        coarse_southwest = _topology_cell_id_3d(topology, 0, 2, 3, 2)
        edge_routes = _topology_routes_for_3d(topology, fine_face_src, 11)
        @test length(edge_routes) == 1
        @test edge_routes[1].dst == coarse_southwest
        @test edge_routes[1].kind == COALESCE_EDGE_3D
    end

    @testset "coarse to fine routes reproduce local 3D split primitives" begin
        coarse_src = zeros(Float64, Nx, Ny, Nz, 19)
        for q in 1:19, K in 1:Nz, J in 1:Ny, I in 1:Nx
            coarse_src[I, J, K, q] = 100q + 10I + J / 10 + K / 100
        end

        expected = zeros(size(patch.fine_F))
        routed = zeros(size(patch.fine_F))
        i0 = 2 * first(patch.parent_i_range) - 1
        j0 = 2 * first(patch.parent_j_range) - 1
        k0 = 2 * first(patch.parent_k_range) - 1

        for idx in topology.coarse_to_fine_links
            link = topology.links[idx]
            src_cell = topology.cells[link.src]
            packet = coarse_src[src_cell.i, src_cell.j, src_cell.k, link.q]
            I_dst = src_cell.i + d3q19_cx(link.q)
            J_dst = src_cell.j + d3q19_cy(link.q)
            K_dst = src_cell.k + d3q19_cz(link.q)
            di = src_cell.i < first(patch.parent_i_range) ? -1 :
                 src_cell.i > last(patch.parent_i_range) ? 1 : 0
            dj = src_cell.j < first(patch.parent_j_range) ? -1 :
                 src_cell.j > last(patch.parent_j_range) ? 1 : 0
            dk = src_cell.k < first(patch.parent_k_range) ? -1 :
                 src_cell.k > last(patch.parent_k_range) ? 1 : 0

            ip = I_dst - first(patch.parent_i_range) + 1
            jp = J_dst - first(patch.parent_j_range) + 1
            kp = K_dst - first(patch.parent_k_range) + 1
            block = @view expected[2ip-1:2ip, 2jp-1:2jp, 2kp-1:2kp, :]
            if _test_offset_count_3d(di, dj, dk) == 1
                split_coarse_to_fine_face_F_3d!(
                    block, packet, link.q, _test_face_from_offset_3d(di, dj, dk))
            else
                split_coarse_to_fine_edge_F_3d!(
                    block, packet, link.q, _test_edge_from_offset_3d(di, dj, dk))
            end

            for route in _topology_routes_for_3d(topology, link.src, link.q)
                dst_cell = topology.cells[route.dst]
                routed[dst_cell.i - i0 + 1,
                       dst_cell.j - j0 + 1,
                       dst_cell.k - k0 + 1,
                       link.q] += packet * route.weight
            end
        end

        @test isapprox(routed, expected; atol=1e-14, rtol=0)
    end
end
