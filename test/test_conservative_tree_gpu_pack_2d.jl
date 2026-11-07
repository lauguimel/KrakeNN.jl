using Test
using KrakeNN

@testset "Conservative tree GPU route pack 2D" begin
    @testset "route pack uses primitive structure-of-arrays" begin
        patch = create_conservative_tree_patch_2d(4:6, 3:5)
        topology = create_conservative_tree_topology_2d(9, 8, patch)
        pack = pack_conservative_tree_gpu_routes_2d(
            topology; cells_per_block=4, T=Float32)

        @test pack.cells_per_block == Int32(4)
        @test eltype(pack.block_level) == UInt8
        @test eltype(pack.block_first_cell) == Int32
        @test eltype(pack.cell_logical_id) == Int32
        @test eltype(pack.route_src) == Int32
        @test eltype(pack.route_dst) == Int32
        @test eltype(pack.route_q) == UInt8
        @test eltype(pack.route_kind) == UInt8
        @test eltype(pack.route_weight) == Float32

        @test length(pack.cell_logical_id) == length(topology.active_cells)
        @test length(pack.route_src) == length(topology.routes)
        @test length(pack.direct_routes) == length(topology.direct_routes)
        @test length(pack.interface_routes) == length(topology.interface_routes)
        @test length(pack.boundary_routes) == length(topology.boundary_routes)
        @test all(pack.block_count .<= pack.cells_per_block)
        @test all(pack.block_count .> 0)

        logical_to_packed = Dict(Int(id) => Int(k)
                                 for (k, id) in pairs(pack.cell_logical_id))
        for (route_index, route) in pairs(topology.routes)
            @test pack.route_src[route_index] == logical_to_packed[route.src]
            @test pack.route_q[route_index] == UInt8(route.q)
            @test pack.route_kind[route_index] == UInt8(route.kind)
            @test pack.route_weight[route_index] == Float32(route.weight)
            if route.kind == ROUTE_BOUNDARY
                @test pack.route_dst[route_index] == 0
            else
                @test pack.route_dst[route_index] == logical_to_packed[route.dst]
            end
        end

        @test_throws ArgumentError pack_conservative_tree_gpu_routes_2d(
            topology; cells_per_block=0)
    end

    @testset "packed route weight sums stay conservative" begin
        patch = create_conservative_tree_patch_2d(3:5, 3:6)
        topology = create_conservative_tree_topology_2d(8, 8, patch)
        pack = pack_conservative_tree_gpu_routes_2d(topology; T=Float64)
        sums = conservative_tree_gpu_route_weight_sums_2d(pack)

        @test length(sums) == 9 * length(topology.active_cells)
        @test all(isapprox(value, 1.0; atol=1e-14, rtol=0)
                  for value in values(sums))
    end

    @testset "CPU replay from GPU pack matches logical route streaming" begin
        nx, ny = 10, 9
        patch_in = create_conservative_tree_patch_2d(4:7, 3:6)
        patch_logical = create_conservative_tree_patch_2d(4:7, 3:6)
        patch_packed = create_conservative_tree_patch_2d(4:7, 3:6)
        topology = create_conservative_tree_topology_2d(nx, ny, patch_in)
        pack = pack_conservative_tree_gpu_routes_2d(
            topology; cells_per_block=5, T=Float64)

        coarse_in = zeros(Float64, nx, ny, 9)
        for q in 1:9, j in axes(coarse_in, 2), i in axes(coarse_in, 1)
            if !(i in patch_in.parent_i_range && j in patch_in.parent_j_range)
                coarse_in[i, j, q] = 0.2 + q / 17 + i / 31 + j / 43 + i * j / 4096
            end
        end
        for q in 1:9, j in axes(patch_in.fine_F, 2), i in axes(patch_in.fine_F, 1)
            patch_in.fine_F[i, j, q] = 0.05 + q / 29 + i / 37 + j / 53 + i * j / 8192
        end

        coarse_logical = similar(coarse_in)
        coarse_packed = similar(coarse_in)
        stream_composite_routes_interior_F_2d!(
            coarse_logical, patch_logical, coarse_in, patch_in, topology)
        stream_conservative_tree_gpu_pack_interior_F_2d!(
            coarse_packed, patch_packed, coarse_in, patch_in, pack)

        @test isapprox(coarse_packed, coarse_logical; atol=1e-14, rtol=0)
        @test isapprox(patch_packed.fine_F, patch_logical.fine_F; atol=1e-14, rtol=0)
        @test isapprox(patch_packed.coarse_shadow_F,
                       patch_logical.coarse_shadow_F; atol=1e-14, rtol=0)

        wrong_patch = create_conservative_tree_patch_2d(4:6, 3:6)
        @test_throws ArgumentError stream_conservative_tree_gpu_pack_interior_F_2d!(
            similar(coarse_in), wrong_patch, coarse_in, patch_in, pack)
    end
end
