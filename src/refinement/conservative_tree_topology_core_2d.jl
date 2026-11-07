# Static AMR topology tables for the conservative-tree D2Q9 prototype.
#
# This layer describes cells, logical D2Q9 links, and packet routes. It does
# not own LBM state and deliberately stays separate from the conservative
# projection/restriction oracle in conservative_tree_2d.jl.

"""
    LinkKind

Public enumeration for the grid-refinement and conservative-tree AMR API. Exported values documented by this definition include `LinkKind`, `SAME_LEVEL`, `COARSE_TO_FINE`, `FINE_TO_COARSE`, `BOUNDARY`.
Use these tags when inspecting or constructing route and topology metadata.

```julia
using KrakeNN

KrakeNN.LinkKind
```
"""
@enum LinkKind::UInt8 SAME_LEVEL=0 COARSE_TO_FINE=1 FINE_TO_COARSE=2 BOUNDARY=3
"""
    RouteKind

Public enumeration for the grid-refinement and conservative-tree AMR API. Exported values documented by this definition include `RouteKind`, `DIRECT`, `SPLIT_FACE`, `SPLIT_CORNER`, `COALESCE_FACE`, `COALESCE_CORNER`, `ROUTE_BOUNDARY`.
Use these tags when inspecting or constructing route and topology metadata.

```julia
using KrakeNN

KrakeNN.RouteKind
```
"""
@enum RouteKind::UInt8 DIRECT=0 SPLIT_FACE=1 SPLIT_CORNER=2 COALESCE_FACE=3 COALESCE_CORNER=4 ROUTE_BOUNDARY=5

"""
    AbstractCellMetrics

Public type or module in the grid-refinement and conservative-tree AMR API.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using KrakeNN

KrakeNN.AbstractCellMetrics
```
"""
abstract type AbstractCellMetrics end

"""
    CartesianMetrics2D

Public type or module in the grid-refinement and conservative-tree AMR API.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using KrakeNN

KrakeNN.CartesianMetrics2D
```
"""
struct CartesianMetrics2D <: AbstractCellMetrics
    volume::Float64
end

"""
    ConservativeTreeCell2D

Public type or module in the grid-refinement and conservative-tree AMR API.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using KrakeNN

KrakeNN.ConservativeTreeCell2D
```
"""
struct ConservativeTreeCell2D
    level::Int
    i::Int
    j::Int
    active::Bool
    metrics::CartesianMetrics2D
    parent::Int
end

"""
    ConservativeTreeLink2D

Public type or module in the grid-refinement and conservative-tree AMR API.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using KrakeNN

KrakeNN.ConservativeTreeLink2D
```
"""
struct ConservativeTreeLink2D
    src::Int
    q::Int
    kind::LinkKind
end

"""
    ConservativeTreeRoute2D

Public type or module in the grid-refinement and conservative-tree AMR API.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using KrakeNN

KrakeNN.ConservativeTreeRoute2D
```
"""
struct ConservativeTreeRoute2D
    src::Int
    dst::Int
    q::Int
    weight::Float64
    kind::RouteKind
end

"""
    ConservativeTreeTopology2D

Public type or module in the grid-refinement and conservative-tree AMR API.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using KrakeNN

KrakeNN.ConservativeTreeTopology2D
```
"""
struct ConservativeTreeTopology2D
    cells::Vector{ConservativeTreeCell2D}
    links::Vector{ConservativeTreeLink2D}
    routes::Vector{ConservativeTreeRoute2D}
    active_cells::Vector{Int}
    same_level_links::Vector{Int}
    coarse_to_fine_links::Vector{Int}
    fine_to_coarse_links::Vector{Int}
    boundary_links::Vector{Int}
    direct_routes::Vector{Int}
    interface_routes::Vector{Int}
    boundary_routes::Vector{Int}
end

"""
    ConservativeTreeBlock2D

Public type or module in the grid-refinement and conservative-tree AMR API.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using KrakeNN

KrakeNN.ConservativeTreeBlock2D
```
"""
struct ConservativeTreeBlock2D
    level::Int
    first_cell::Int
    count::Int
    morton_first::UInt64
end

"""
    ConservativeTreePackedRoute2D

Public type or module in the grid-refinement and conservative-tree AMR API.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using KrakeNN

KrakeNN.ConservativeTreePackedRoute2D
```
"""
struct ConservativeTreePackedRoute2D
    src_block::Int
    src_local::Int
    dst_block::Int
    dst_local::Int
    q::Int
    weight::Float64
    kind::RouteKind
end

"""
    ConservativeTreePackedTopology2D

Public type or module in the grid-refinement and conservative-tree AMR API.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using KrakeNN

KrakeNN.ConservativeTreePackedTopology2D
```
"""
struct ConservativeTreePackedTopology2D
    cells_per_block::Int
    blocks::Vector{ConservativeTreeBlock2D}
    packed_cell_ids::Vector{Int}
    packed_morton_keys::Vector{UInt64}
    logical_cell_to_block::Vector{Int}
    logical_cell_to_local::Vector{Int}
    routes::Vector{ConservativeTreePackedRoute2D}
    direct_routes::Vector{Int}
    interface_routes::Vector{Int}
    boundary_routes::Vector{Int}
end

@inline _tree_topology_key(level::Int, i::Int, j::Int) = (level, i, j)

@inline function _fine_global_i(I::Int, ix::Int)
    return 2 * I - 2 + ix
end

@inline function _fine_global_j(J::Int, iy::Int)
    return 2 * J - 2 + iy
end

@inline function _coarse_parent_from_fine(i::Int, j::Int)
    return (i + 1) >>> 1, (j + 1) >>> 1
end

@inline function _child_index_from_fine(i::Int, j::Int)
    return isodd(i) ? 1 : 2, isodd(j) ? 1 : 2
end

@inline function _inside_fine_patch(i::Int, j::Int, patch::ConservativeTreePatch2D)
    imin = 2 * first(patch.parent_i_range) - 1
    imax = 2 * last(patch.parent_i_range)
    jmin = 2 * first(patch.parent_j_range) - 1
    jmax = 2 * last(patch.parent_j_range)
    return imin <= i <= imax && jmin <= j <= jmax
end

@inline function _inside_leaf_domain(i::Int, j::Int, Nx::Int, Ny::Int)
    return 1 <= i <= 2 * Nx && 1 <= j <= 2 * Ny
end

function _check_conservative_tree_topology_args(Nx::Int,
                                                Ny::Int,
                                                patch::ConservativeTreePatch2D,
                                                coarse_volume::Real)
    Nx > 0 || throw(ArgumentError("Nx must be positive"))
    Ny > 0 || throw(ArgumentError("Ny must be positive"))
    coarse_volume > 0 || throw(ArgumentError("coarse_volume must be positive"))
    _check_conservative_tree_patch_layout(patch)
    first(patch.parent_i_range) >= 1 ||
        throw(ArgumentError("patch.parent_i_range starts outside the domain"))
    last(patch.parent_i_range) <= Nx ||
        throw(ArgumentError("patch.parent_i_range ends outside the domain"))
    first(patch.parent_j_range) >= 1 ||
        throw(ArgumentError("patch.parent_j_range starts outside the domain"))
    last(patch.parent_j_range) <= Ny ||
        throw(ArgumentError("patch.parent_j_range ends outside the domain"))
    return nothing
end

function _push_link_route!(links::Vector{ConservativeTreeLink2D},
                           routes::Vector{ConservativeTreeRoute2D},
                           same_level_links::Vector{Int},
                           coarse_to_fine_links::Vector{Int},
                           fine_to_coarse_links::Vector{Int},
                           boundary_links::Vector{Int},
                           direct_routes::Vector{Int},
                           interface_routes::Vector{Int},
                           boundary_routes::Vector{Int},
                           src::Int,
                           q::Int,
                           link_kind::LinkKind,
                           route_specs)
    push!(links, ConservativeTreeLink2D(src, q, link_kind))
    link_index = length(links)
    if link_kind == SAME_LEVEL
        push!(same_level_links, link_index)
    elseif link_kind == COARSE_TO_FINE
        push!(coarse_to_fine_links, link_index)
    elseif link_kind == FINE_TO_COARSE
        push!(fine_to_coarse_links, link_index)
    else
        push!(boundary_links, link_index)
    end

    for spec in route_specs
        dst, weight, route_kind = spec
        push!(routes, ConservativeTreeRoute2D(src, dst, q, Float64(weight), route_kind))
        route_index = length(routes)
        if route_kind == DIRECT
            push!(direct_routes, route_index)
        elseif route_kind == ROUTE_BOUNDARY
            push!(boundary_routes, route_index)
        else
            push!(interface_routes, route_index)
        end
    end
    return nothing
end

function _push_or_accumulate_route_spec!(dsts::Vector{Int},
                                         weights::Vector{Float64},
                                         kinds::Vector{RouteKind},
                                         dst::Int,
                                         weight::Float64,
                                         kind::RouteKind)
    @inbounds for idx in eachindex(dsts)
        if dsts[idx] == dst && kinds[idx] == kind
            weights[idx] += weight
            return nothing
        end
    end
    push!(dsts, dst)
    push!(weights, weight)
    push!(kinds, kind)
    return nothing
end

function _coarse_to_fine_route_specs(cell_id_by_coord,
                                     src_i::Int,
                                     src_j::Int,
                                     q::Int,
                                     patch::ConservativeTreePatch2D,
                                     Nx::Int,
                                     Ny::Int;
                                     periodic_x::Bool=false)
    cx = d2q9_cx(q)
    cy = d2q9_cy(q)
    split_kind = (src_i < first(patch.parent_i_range) ||
                  src_i > last(patch.parent_i_range)) &&
                 (src_j < first(patch.parent_j_range) ||
                  src_j > last(patch.parent_j_range)) ? SPLIT_CORNER : SPLIT_FACE

    dsts = Int[]
    weights = Float64[]
    kinds = RouteKind[]
    @inbounds for iy in 1:2, ix in 1:2
        i_dst = _fine_global_i(src_i, ix) + cx
        j_dst = _fine_global_j(src_j, iy) + cy
        if periodic_x
            i_dst = mod1(i_dst, 2 * Nx)
        end

        if !_inside_leaf_domain(i_dst, j_dst, Nx, Ny)
            _push_or_accumulate_route_spec!(dsts, weights, kinds,
                                            0, 0.25, ROUTE_BOUNDARY)
        elseif _inside_fine_patch(i_dst, j_dst, patch)
            dst = cell_id_by_coord[_tree_topology_key(1, i_dst, j_dst)]
            _push_or_accumulate_route_spec!(dsts, weights, kinds,
                                            dst, 0.25, split_kind)
        else
            I_dst, J_dst = _coarse_parent_from_fine(i_dst, j_dst)
            dst = cell_id_by_coord[_tree_topology_key(0, I_dst, J_dst)]
            _push_or_accumulate_route_spec!(dsts, weights, kinds,
                                            dst, 0.25, DIRECT)
        end
    end

    return Tuple((dsts[idx], weights[idx], kinds[idx]) for idx in eachindex(dsts))
end

function _coarse_leaf_equivalent_link_kind(specs)
    has_boundary = false
    has_split = false
    @inbounds for spec in specs
        _, _, kind = spec
        if kind == ROUTE_BOUNDARY
            has_boundary = true
        elseif kind == SPLIT_FACE || kind == SPLIT_CORNER
            has_split = true
        end
    end
    return has_boundary ? BOUNDARY : (has_split ? COARSE_TO_FINE : SAME_LEVEL)
end

@inline function _fine_to_coarse_route_kind(i_dst_leaf::Int,
                                            j_dst_leaf::Int,
                                            patch::ConservativeTreePatch2D)
    out_x = i_dst_leaf < 2 * first(patch.parent_i_range) - 1 ||
            i_dst_leaf > 2 * last(patch.parent_i_range)
    out_y = j_dst_leaf < 2 * first(patch.parent_j_range) - 1 ||
            j_dst_leaf > 2 * last(patch.parent_j_range)
    return out_x && out_y ? COALESCE_CORNER : COALESCE_FACE
end

function _build_conservative_tree_cells_2d(Nx::Int,
                                           Ny::Int,
                                           patch::ConservativeTreePatch2D,
                                           coarse_volume::Float64)
    cells = ConservativeTreeCell2D[]
    active_cells = Int[]
    cell_id_by_coord = Dict{Tuple{Int,Int,Int},Int}()

    coarse_metrics = CartesianMetrics2D(coarse_volume)
    fine_metrics = CartesianMetrics2D(coarse_volume / 4)

    for J in 1:Ny, I in 1:Nx
        active = !_inside_range(I, J, patch.parent_i_range, patch.parent_j_range)
        push!(cells, ConservativeTreeCell2D(0, I, J, active, coarse_metrics, 0))
        id = length(cells)
        cell_id_by_coord[_tree_topology_key(0, I, J)] = id
        active && push!(active_cells, id)
    end

    for J in patch.parent_j_range, I in patch.parent_i_range
        parent = cell_id_by_coord[_tree_topology_key(0, I, J)]
        for iy in 1:2, ix in 1:2
            i_f = _fine_global_i(I, ix)
            j_f = _fine_global_j(J, iy)
            push!(cells, ConservativeTreeCell2D(1, i_f, j_f, true, fine_metrics, parent))
            id = length(cells)
            cell_id_by_coord[_tree_topology_key(1, i_f, j_f)] = id
            push!(active_cells, id)
        end
    end

    return cells, active_cells, cell_id_by_coord
end

function _build_conservative_tree_links_2d(cells::Vector{ConservativeTreeCell2D},
                                           active_cells::Vector{Int},
                                           cell_id_by_coord,
                                           Nx::Int,
                                           Ny::Int,
                                           patch::ConservativeTreePatch2D,
                                           coarse_route_mode::Symbol)
    links = ConservativeTreeLink2D[]
    routes = ConservativeTreeRoute2D[]
    same_level_links = Int[]
    coarse_to_fine_links = Int[]
    fine_to_coarse_links = Int[]
    boundary_links = Int[]
    direct_routes = Int[]
    interface_routes = Int[]
    boundary_routes = Int[]

    for src in active_cells
        cell = cells[src]
        for q in 1:9
            cx = d2q9_cx(q)
            cy = d2q9_cy(q)

            if cell.level == 0
                if coarse_route_mode == :leaf_equivalent
                    specs = _coarse_to_fine_route_specs(
                        cell_id_by_coord, cell.i, cell.j, q, patch, Nx, Ny)
                    link_kind = _coarse_leaf_equivalent_link_kind(specs)
                    _push_link_route!(links, routes,
                                      same_level_links, coarse_to_fine_links,
                                      fine_to_coarse_links, boundary_links,
                                      direct_routes, interface_routes,
                                      boundary_routes,
                                      src, q, link_kind, specs)
                else
                    I_dst = cell.i + cx
                    J_dst = cell.j + cy
                    if !(1 <= I_dst <= Nx && 1 <= J_dst <= Ny)
                        _push_link_route!(links, routes,
                                          same_level_links, coarse_to_fine_links,
                                          fine_to_coarse_links, boundary_links,
                                          direct_routes, interface_routes,
                                          boundary_routes,
                                          src, q, BOUNDARY,
                                          ((0, 1.0, ROUTE_BOUNDARY),))
                    elseif _inside_range(I_dst, J_dst,
                                         patch.parent_i_range,
                                         patch.parent_j_range)
                        specs = _coarse_to_fine_route_specs(cell_id_by_coord,
                                                            cell.i, cell.j, q,
                                                            patch, Nx, Ny)
                        _push_link_route!(links, routes,
                                          same_level_links, coarse_to_fine_links,
                                          fine_to_coarse_links, boundary_links,
                                          direct_routes, interface_routes,
                                          boundary_routes,
                                          src, q, COARSE_TO_FINE, specs)
                    else
                        dst = cell_id_by_coord[_tree_topology_key(0, I_dst, J_dst)]
                        _push_link_route!(links, routes,
                                          same_level_links, coarse_to_fine_links,
                                          fine_to_coarse_links, boundary_links,
                                          direct_routes, interface_routes,
                                          boundary_routes,
                                          src, q, SAME_LEVEL,
                                          ((dst, 1.0, DIRECT),))
                    end
                end
            else
                i_dst = cell.i + cx
                j_dst = cell.j + cy
                if _inside_fine_patch(i_dst, j_dst, patch)
                    dst = cell_id_by_coord[_tree_topology_key(1, i_dst, j_dst)]
                    _push_link_route!(links, routes,
                                      same_level_links, coarse_to_fine_links,
                                      fine_to_coarse_links, boundary_links,
                                      direct_routes, interface_routes,
                                      boundary_routes,
                                      src, q, SAME_LEVEL,
                                      ((dst, 1.0, DIRECT),))
                elseif !_inside_leaf_domain(i_dst, j_dst, Nx, Ny)
                    _push_link_route!(links, routes,
                                      same_level_links, coarse_to_fine_links,
                                      fine_to_coarse_links, boundary_links,
                                      direct_routes, interface_routes,
                                      boundary_routes,
                                      src, q, BOUNDARY,
                                      ((0, 1.0, ROUTE_BOUNDARY),))
                else
                    I_dst, J_dst = _coarse_parent_from_fine(i_dst, j_dst)
                    dst = cell_id_by_coord[_tree_topology_key(0, I_dst, J_dst)]
                    route_kind = _fine_to_coarse_route_kind(i_dst, j_dst, patch)
                    _push_link_route!(links, routes,
                                      same_level_links, coarse_to_fine_links,
                                      fine_to_coarse_links, boundary_links,
                                      direct_routes, interface_routes,
                                      boundary_routes,
                                      src, q, FINE_TO_COARSE,
                                      ((dst, 1.0, route_kind),))
                end
            end
        end
    end

    return links, routes,
           same_level_links, coarse_to_fine_links, fine_to_coarse_links,
           boundary_links, direct_routes, interface_routes, boundary_routes
end
