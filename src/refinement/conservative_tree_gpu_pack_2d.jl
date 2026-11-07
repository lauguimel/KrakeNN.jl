# GPU-ready route packs for conservative-tree D2Q9 AMR.
#
# The pack is deliberately a structure-of-arrays made only of primitive Julia
# arrays. CPU tests replay the pack directly; a later patch can transfer these
# arrays to CUDA/Metal without changing the route contract.

"""
    ConservativeTreeGPURoutePack2D

Public type or module in the grid-refinement and conservative-tree AMR API.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using KrakeNN

KrakeNN.ConservativeTreeGPURoutePack2D
```
"""
struct ConservativeTreeGPURoutePack2D{T}
    cells_per_block::Int32
    block_level::Vector{UInt8}
    block_first_cell::Vector{Int32}
    block_count::Vector{Int32}
    cell_logical_id::Vector{Int32}
    cell_level::Vector{UInt8}
    cell_i::Vector{Int32}
    cell_j::Vector{Int32}
    route_src::Vector{Int32}
    route_dst::Vector{Int32}
    route_q::Vector{UInt8}
    route_kind::Vector{UInt8}
    route_weight::Vector{T}
    direct_routes::Vector{Int32}
    interface_routes::Vector{Int32}
    boundary_routes::Vector{Int32}
end

function _logical_to_packed_index_2d(packed::ConservativeTreePackedTopology2D)
    logical_to_packed = zeros(Int32, length(packed.logical_cell_to_block))
    @inbounds for (packed_index, logical_id) in pairs(packed.packed_cell_ids)
        logical_to_packed[logical_id] = Int32(packed_index)
    end
    return logical_to_packed
end

"""
    pack_conservative_tree_gpu_routes_2d(topology; cells_per_block=128, T=Float32)

Convert a conservative-tree topology into primitive route arrays suitable for
GPU transfer. Route destinations use packed-cell index `0` for boundaries.
"""
function pack_conservative_tree_gpu_routes_2d(
        topology::ConservativeTreeTopology2D;
        cells_per_block::Integer=128,
        T::Type{<:Real}=Float32)
    packed = pack_conservative_tree_topology_2d(topology;
                                                cells_per_block=cells_per_block)
    logical_to_packed = _logical_to_packed_index_2d(packed)

    block_level = UInt8[UInt8(block.level) for block in packed.blocks]
    block_first_cell = Int32[Int32(block.first_cell) for block in packed.blocks]
    block_count = Int32[Int32(block.count) for block in packed.blocks]

    cell_logical_id = Int32[Int32(id) for id in packed.packed_cell_ids]
    cell_level = Vector{UInt8}(undef, length(packed.packed_cell_ids))
    cell_i = Vector{Int32}(undef, length(packed.packed_cell_ids))
    cell_j = Vector{Int32}(undef, length(packed.packed_cell_ids))
    @inbounds for (packed_index, logical_id) in pairs(packed.packed_cell_ids)
        cell = topology.cells[logical_id]
        cell_level[packed_index] = UInt8(cell.level)
        cell_i[packed_index] = Int32(cell.i)
        cell_j[packed_index] = Int32(cell.j)
    end

    n_routes = length(topology.routes)
    route_src = Vector{Int32}(undef, n_routes)
    route_dst = Vector{Int32}(undef, n_routes)
    route_q = Vector{UInt8}(undef, n_routes)
    route_kind = Vector{UInt8}(undef, n_routes)
    route_weight = Vector{T}(undef, n_routes)

    @inbounds for (route_index, route) in pairs(topology.routes)
        src = logical_to_packed[route.src]
        src > 0 || throw(ArgumentError("route source is not packed"))
        route_src[route_index] = src
        route_dst[route_index] =
            route.kind == ROUTE_BOUNDARY ? Int32(0) : logical_to_packed[route.dst]
        route_q[route_index] = UInt8(route.q)
        route_kind[route_index] = UInt8(route.kind)
        route_weight[route_index] = T(route.weight)
    end

    return ConservativeTreeGPURoutePack2D{T}(
        Int32(packed.cells_per_block),
        block_level, block_first_cell, block_count,
        cell_logical_id, cell_level, cell_i, cell_j,
        route_src, route_dst, route_q, route_kind, route_weight,
        Int32.(packed.direct_routes),
        Int32.(packed.interface_routes),
        Int32.(packed.boundary_routes))
end

"""
    conservative_tree_gpu_route_weight_sums_2d(

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.conservative_tree_gpu_route_weight_sums_2d)
```
"""
function conservative_tree_gpu_route_weight_sums_2d(
        pack::ConservativeTreeGPURoutePack2D{T}) where T
    sums = Dict{Tuple{Int32,UInt8},T}()
    @inbounds for route_index in eachindex(pack.route_src)
        key = (pack.route_src[route_index], pack.route_q[route_index])
        sums[key] = get(sums, key, zero(T)) + pack.route_weight[route_index]
    end
    return sums
end

@inline function _gpu_pack_fine_local_2d(patch::ConservativeTreePatch2D,
                                         i_leaf::Int,
                                         j_leaf::Int)
    i0 = 2 * first(patch.parent_i_range) - 1
    j0 = 2 * first(patch.parent_j_range) - 1
    return i_leaf - i0 + 1, j_leaf - j0 + 1
end

@inline function _gpu_pack_cell_Fq_2d(coarse_F,
                                      patch::ConservativeTreePatch2D,
                                      pack::ConservativeTreeGPURoutePack2D,
                                      packed_cell::Int,
                                      q::Int)
    level = pack.cell_level[packed_cell]
    i = Int(pack.cell_i[packed_cell])
    j = Int(pack.cell_j[packed_cell])
    if level == UInt8(0)
        return coarse_F[i, j, q]
    end
    il, jl = _gpu_pack_fine_local_2d(patch, i, j)
    return patch.fine_F[il, jl, q]
end

@inline function _gpu_pack_add_cell_Fq_2d!(coarse_F,
                                           patch::ConservativeTreePatch2D,
                                           pack::ConservativeTreeGPURoutePack2D,
                                           packed_cell::Int,
                                           q::Int,
                                           value)
    level = pack.cell_level[packed_cell]
    i = Int(pack.cell_i[packed_cell])
    j = Int(pack.cell_j[packed_cell])
    if level == UInt8(0)
        coarse_F[i, j, q] += value
    else
        il, jl = _gpu_pack_fine_local_2d(patch, i, j)
        patch.fine_F[il, jl, q] += value
    end
    return nothing
end

"""
    stream_conservative_tree_gpu_pack_interior_F_2d!(

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.stream_conservative_tree_gpu_pack_interior_F_2d!)
```
"""
function stream_conservative_tree_gpu_pack_interior_F_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D,
        pack::ConservativeTreeGPURoutePack2D)
    _check_composite_coarse_layout(coarse_in, patch_in)
    _check_composite_coarse_layout(coarse_out, patch_out)
    size(coarse_out) == size(coarse_in) ||
        throw(ArgumentError("coarse_out and coarse_in must have the same size"))
    patch_out.parent_i_range == patch_in.parent_i_range &&
        patch_out.parent_j_range == patch_in.parent_j_range ||
        throw(ArgumentError("patch_out and patch_in must cover the same parent range"))

    coarse_out .= 0
    patch_out.fine_F .= 0
    patch_out.coarse_shadow_F .= 0

    boundary_kind = UInt8(ROUTE_BOUNDARY)
    @inbounds for route_index in eachindex(pack.route_src)
        pack.route_kind[route_index] == boundary_kind && continue
        src = Int(pack.route_src[route_index])
        dst = Int(pack.route_dst[route_index])
        dst > 0 || throw(ArgumentError("non-boundary packed route has zero destination"))
        q = Int(pack.route_q[route_index])
        packet = _gpu_pack_cell_Fq_2d(coarse_in, patch_in, pack, src, q)
        _gpu_pack_add_cell_Fq_2d!(
            coarse_out, patch_out, pack, dst, q,
            pack.route_weight[route_index] * packet)
    end
    coalesce_patch_to_shadow_F_2d!(patch_out)
    return coarse_out, patch_out
end
