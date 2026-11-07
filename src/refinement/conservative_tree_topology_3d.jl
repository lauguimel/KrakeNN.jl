# Static AMR topology tables for the conservative-tree D3Q19 prototype.
#
# This layer describes active cells, logical D3Q19 links, and conservative
# route weights. It does not own LBM state and intentionally stops before
# streaming or GPU packing.

"""
    RouteKind3D

Public enumeration for the grid-refinement and conservative-tree AMR API. Exported values documented by this definition include `RouteKind3D`, `DIRECT_3D`, `SPLIT_FACE_3D`, `SPLIT_EDGE_3D`, `COALESCE_FACE_3D`, `COALESCE_EDGE_3D`, `ROUTE_BOUNDARY_3D`.
Use these tags when inspecting or constructing route and topology metadata.

```julia
using KrakeNN

KrakeNN.RouteKind3D
```
"""
@enum RouteKind3D::UInt8 DIRECT_3D=0 SPLIT_FACE_3D=1 SPLIT_EDGE_3D=2 COALESCE_FACE_3D=3 COALESCE_EDGE_3D=4 ROUTE_BOUNDARY_3D=5

"""
    CartesianMetrics3D

Public type or module in the grid-refinement and conservative-tree AMR API.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using KrakeNN

KrakeNN.CartesianMetrics3D
```
"""
struct CartesianMetrics3D <: AbstractCellMetrics
    volume::Float64
end

"""
    ConservativeTreeCell3D

Public type or module in the grid-refinement and conservative-tree AMR API.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using KrakeNN

KrakeNN.ConservativeTreeCell3D
```
"""
struct ConservativeTreeCell3D
    level::Int
    i::Int
    j::Int
    k::Int
    active::Bool
    metrics::CartesianMetrics3D
    parent::Int
end

"""
    ConservativeTreeLink3D

Public type or module in the grid-refinement and conservative-tree AMR API.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using KrakeNN

KrakeNN.ConservativeTreeLink3D
```
"""
struct ConservativeTreeLink3D
    src::Int
    q::Int
    kind::LinkKind
end

"""
    ConservativeTreeRoute3D

Public type or module in the grid-refinement and conservative-tree AMR API.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using KrakeNN

KrakeNN.ConservativeTreeRoute3D
```
"""
struct ConservativeTreeRoute3D
    src::Int
    dst::Int
    q::Int
    weight::Float64
    kind::RouteKind3D
end

"""
    ConservativeTreeTopology3D

Public type or module in the grid-refinement and conservative-tree AMR API.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using KrakeNN

KrakeNN.ConservativeTreeTopology3D
```
"""
struct ConservativeTreeTopology3D
    cells::Vector{ConservativeTreeCell3D}
    links::Vector{ConservativeTreeLink3D}
    routes::Vector{ConservativeTreeRoute3D}
    active_cells::Vector{Int}
    same_level_links::Vector{Int}
    coarse_to_fine_links::Vector{Int}
    fine_to_coarse_links::Vector{Int}
    boundary_links::Vector{Int}
    direct_routes::Vector{Int}
    interface_routes::Vector{Int}
    boundary_routes::Vector{Int}
end

@inline _tree_topology_key_3d(level::Int, i::Int, j::Int, k::Int) = (level, i, j, k)

@inline _fine_global_i_3d(I::Int, ix::Int) = 2 * I - 2 + ix
@inline _fine_global_j_3d(J::Int, iy::Int) = 2 * J - 2 + iy
@inline _fine_global_k_3d(K::Int, iz::Int) = 2 * K - 2 + iz

@inline function _coarse_parent_from_fine_3d(i::Int, j::Int, k::Int)
    return (i + 1) >>> 1, (j + 1) >>> 1, (k + 1) >>> 1
end

@inline function _inside_range_3d(i::Int, j::Int, k::Int,
                                  irange::UnitRange{Int},
                                  jrange::UnitRange{Int},
                                  krange::UnitRange{Int})
    return first(irange) <= i <= last(irange) &&
           first(jrange) <= j <= last(jrange) &&
           first(krange) <= k <= last(krange)
end

@inline function _inside_fine_patch_3d(i::Int, j::Int, k::Int,
                                       patch::ConservativeTreePatch3D)
    imin = 2 * first(patch.parent_i_range) - 1
    imax = 2 * last(patch.parent_i_range)
    jmin = 2 * first(patch.parent_j_range) - 1
    jmax = 2 * last(patch.parent_j_range)
    kmin = 2 * first(patch.parent_k_range) - 1
    kmax = 2 * last(patch.parent_k_range)
    return imin <= i <= imax && jmin <= j <= jmax && kmin <= k <= kmax
end

@inline function _inside_leaf_domain_3d(i::Int, j::Int, k::Int,
                                        Nx::Int, Ny::Int, Nz::Int)
    return 1 <= i <= 2 * Nx && 1 <= j <= 2 * Ny && 1 <= k <= 2 * Nz
end

function _check_conservative_tree_topology_args_3d(Nx::Int,
                                                   Ny::Int,
                                                   Nz::Int,
                                                   patch::ConservativeTreePatch3D,
                                                   coarse_volume::Real)
    Nx > 0 || throw(ArgumentError("Nx must be positive"))
    Ny > 0 || throw(ArgumentError("Ny must be positive"))
    Nz > 0 || throw(ArgumentError("Nz must be positive"))
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
    first(patch.parent_k_range) >= 1 ||
        throw(ArgumentError("patch.parent_k_range starts outside the domain"))
    last(patch.parent_k_range) <= Nz ||
        throw(ArgumentError("patch.parent_k_range ends outside the domain"))
    return nothing
end

function _push_link_route_3d!(links::Vector{ConservativeTreeLink3D},
                              routes::Vector{ConservativeTreeRoute3D},
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
    push!(links, ConservativeTreeLink3D(src, q, link_kind))
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
        push!(routes, ConservativeTreeRoute3D(src, dst, q, Float64(weight), route_kind))
        route_index = length(routes)
        if route_kind == DIRECT_3D
            push!(direct_routes, route_index)
        elseif route_kind == ROUTE_BOUNDARY_3D
            push!(boundary_routes, route_index)
        else
            push!(interface_routes, route_index)
        end
    end
    return nothing
end

@inline function _nonzero_offset_count_3d(di::Int, dj::Int, dk::Int)
    return (di == 0 ? 0 : 1) + (dj == 0 ? 0 : 1) + (dk == 0 ? 0 : 1)
end

function _coarse_to_fine_face_route_specs_3d(cell_id_by_coord,
                                             I_dst::Int,
                                             J_dst::Int,
                                             K_dst::Int,
                                             di::Int,
                                             dj::Int,
                                             dk::Int)
    ixs = di == 0 ? (1, 2) : (di < 0 ? (1,) : (2,))
    iys = dj == 0 ? (1, 2) : (dj < 0 ? (1,) : (2,))
    izs = dk == 0 ? (1, 2) : (dk < 0 ? (1,) : (2,))
    specs = Tuple{Int,Float64,RouteKind3D}[]
    for iz in izs, iy in iys, ix in ixs
        dst = cell_id_by_coord[_tree_topology_key_3d(
            1,
            _fine_global_i_3d(I_dst, ix),
            _fine_global_j_3d(J_dst, iy),
            _fine_global_k_3d(K_dst, iz))]
        push!(specs, (dst, 0.25, SPLIT_FACE_3D))
    end
    length(specs) == 4 ||
        throw(ArgumentError("face route must target four children"))
    return Tuple(specs)
end

function _coarse_to_fine_edge_route_specs_3d(cell_id_by_coord,
                                             I_dst::Int,
                                             J_dst::Int,
                                             K_dst::Int,
                                             di::Int,
                                             dj::Int,
                                             dk::Int)
    ixs = di == 0 ? (1, 2) : (di < 0 ? (1,) : (2,))
    iys = dj == 0 ? (1, 2) : (dj < 0 ? (1,) : (2,))
    izs = dk == 0 ? (1, 2) : (dk < 0 ? (1,) : (2,))
    specs = Tuple{Int,Float64,RouteKind3D}[]
    for iz in izs, iy in iys, ix in ixs
        dst = cell_id_by_coord[_tree_topology_key_3d(
            1,
            _fine_global_i_3d(I_dst, ix),
            _fine_global_j_3d(J_dst, iy),
            _fine_global_k_3d(K_dst, iz))]
        push!(specs, (dst, 0.5, SPLIT_EDGE_3D))
    end
    length(specs) == 2 ||
        throw(ArgumentError("edge route must target two children"))
    return Tuple(specs)
end

function _coarse_to_fine_route_specs_3d(cell_id_by_coord,
                                        I_dst::Int,
                                        J_dst::Int,
                                        K_dst::Int,
                                        di::Int,
                                        dj::Int,
                                        dk::Int)
    n = _nonzero_offset_count_3d(di, dj, dk)
    if n == 1
        return _coarse_to_fine_face_route_specs_3d(
            cell_id_by_coord, I_dst, J_dst, K_dst, di, dj, dk)
    elseif n == 2
        return _coarse_to_fine_edge_route_specs_3d(
            cell_id_by_coord, I_dst, J_dst, K_dst, di, dj, dk)
    else
        throw(ArgumentError("D3Q19 coarse-to-fine routes support face or edge crossings only"))
    end
end

@inline function _fine_to_coarse_route_kind_3d(i_dst_leaf::Int,
                                               j_dst_leaf::Int,
                                               k_dst_leaf::Int,
                                               patch::ConservativeTreePatch3D)
    out_x = i_dst_leaf < 2 * first(patch.parent_i_range) - 1 ||
            i_dst_leaf > 2 * last(patch.parent_i_range)
    out_y = j_dst_leaf < 2 * first(patch.parent_j_range) - 1 ||
            j_dst_leaf > 2 * last(patch.parent_j_range)
    out_z = k_dst_leaf < 2 * first(patch.parent_k_range) - 1 ||
            k_dst_leaf > 2 * last(patch.parent_k_range)
    n = (out_x ? 1 : 0) + (out_y ? 1 : 0) + (out_z ? 1 : 0)
    if n == 1
        return COALESCE_FACE_3D
    elseif n == 2
        return COALESCE_EDGE_3D
    else
        throw(ArgumentError("D3Q19 fine-to-coarse routes support face or edge crossings only"))
    end
end

function _build_conservative_tree_cells_3d(Nx::Int,
                                           Ny::Int,
                                           Nz::Int,
                                           patch::ConservativeTreePatch3D,
                                           coarse_volume::Float64)
    cells = ConservativeTreeCell3D[]
    active_cells = Int[]
    cell_id_by_coord = Dict{Tuple{Int,Int,Int,Int},Int}()

    coarse_metrics = CartesianMetrics3D(coarse_volume)
    fine_metrics = CartesianMetrics3D(coarse_volume / 8)

    for K in 1:Nz, J in 1:Ny, I in 1:Nx
        active = !_inside_range_3d(I, J, K,
                                   patch.parent_i_range,
                                   patch.parent_j_range,
                                   patch.parent_k_range)
        push!(cells, ConservativeTreeCell3D(0, I, J, K, active, coarse_metrics, 0))
        id = length(cells)
        cell_id_by_coord[_tree_topology_key_3d(0, I, J, K)] = id
        active && push!(active_cells, id)
    end

    for K in patch.parent_k_range, J in patch.parent_j_range, I in patch.parent_i_range
        parent = cell_id_by_coord[_tree_topology_key_3d(0, I, J, K)]
        for iz in 1:2, iy in 1:2, ix in 1:2
            i_f = _fine_global_i_3d(I, ix)
            j_f = _fine_global_j_3d(J, iy)
            k_f = _fine_global_k_3d(K, iz)
            push!(cells, ConservativeTreeCell3D(1, i_f, j_f, k_f,
                                                true, fine_metrics, parent))
            id = length(cells)
            cell_id_by_coord[_tree_topology_key_3d(1, i_f, j_f, k_f)] = id
            push!(active_cells, id)
        end
    end

    return cells, active_cells, cell_id_by_coord
end

function _build_conservative_tree_links_3d(cells::Vector{ConservativeTreeCell3D},
                                           active_cells::Vector{Int},
                                           cell_id_by_coord,
                                           Nx::Int,
                                           Ny::Int,
                                           Nz::Int,
                                           patch::ConservativeTreePatch3D)
    links = ConservativeTreeLink3D[]
    routes = ConservativeTreeRoute3D[]
    same_level_links = Int[]
    coarse_to_fine_links = Int[]
    fine_to_coarse_links = Int[]
    boundary_links = Int[]
    direct_routes = Int[]
    interface_routes = Int[]
    boundary_routes = Int[]

    for src in active_cells
        cell = cells[src]
        for q in 1:19
            cx = d3q19_cx(q)
            cy = d3q19_cy(q)
            cz = d3q19_cz(q)

            if cell.level == 0
                I_dst = cell.i + cx
                J_dst = cell.j + cy
                K_dst = cell.k + cz
                if !(1 <= I_dst <= Nx && 1 <= J_dst <= Ny && 1 <= K_dst <= Nz)
                    _push_link_route_3d!(links, routes,
                                         same_level_links, coarse_to_fine_links,
                                         fine_to_coarse_links, boundary_links,
                                         direct_routes, interface_routes,
                                         boundary_routes,
                                         src, q, BOUNDARY,
                                         ((0, 1.0, ROUTE_BOUNDARY_3D),))
                elseif _inside_range_3d(I_dst, J_dst, K_dst,
                                        patch.parent_i_range,
                                        patch.parent_j_range,
                                        patch.parent_k_range)
                    di = cell.i < first(patch.parent_i_range) ? -1 :
                         cell.i > last(patch.parent_i_range) ? 1 : 0
                    dj = cell.j < first(patch.parent_j_range) ? -1 :
                         cell.j > last(patch.parent_j_range) ? 1 : 0
                    dk = cell.k < first(patch.parent_k_range) ? -1 :
                         cell.k > last(patch.parent_k_range) ? 1 : 0
                    specs = _coarse_to_fine_route_specs_3d(
                        cell_id_by_coord, I_dst, J_dst, K_dst, di, dj, dk)
                    _push_link_route_3d!(links, routes,
                                         same_level_links, coarse_to_fine_links,
                                         fine_to_coarse_links, boundary_links,
                                         direct_routes, interface_routes,
                                         boundary_routes,
                                         src, q, COARSE_TO_FINE, specs)
                else
                    dst = cell_id_by_coord[_tree_topology_key_3d(0, I_dst, J_dst, K_dst)]
                    _push_link_route_3d!(links, routes,
                                         same_level_links, coarse_to_fine_links,
                                         fine_to_coarse_links, boundary_links,
                                         direct_routes, interface_routes,
                                         boundary_routes,
                                         src, q, SAME_LEVEL,
                                         ((dst, 1.0, DIRECT_3D),))
                end
            else
                i_dst = cell.i + cx
                j_dst = cell.j + cy
                k_dst = cell.k + cz
                if _inside_fine_patch_3d(i_dst, j_dst, k_dst, patch)
                    dst = cell_id_by_coord[_tree_topology_key_3d(1, i_dst, j_dst, k_dst)]
                    _push_link_route_3d!(links, routes,
                                         same_level_links, coarse_to_fine_links,
                                         fine_to_coarse_links, boundary_links,
                                         direct_routes, interface_routes,
                                         boundary_routes,
                                         src, q, SAME_LEVEL,
                                         ((dst, 1.0, DIRECT_3D),))
                elseif !_inside_leaf_domain_3d(i_dst, j_dst, k_dst, Nx, Ny, Nz)
                    _push_link_route_3d!(links, routes,
                                         same_level_links, coarse_to_fine_links,
                                         fine_to_coarse_links, boundary_links,
                                         direct_routes, interface_routes,
                                         boundary_routes,
                                         src, q, BOUNDARY,
                                         ((0, 1.0, ROUTE_BOUNDARY_3D),))
                else
                    I_dst, J_dst, K_dst = _coarse_parent_from_fine_3d(i_dst, j_dst, k_dst)
                    dst = cell_id_by_coord[_tree_topology_key_3d(0, I_dst, J_dst, K_dst)]
                    route_kind = _fine_to_coarse_route_kind_3d(i_dst, j_dst, k_dst, patch)
                    _push_link_route_3d!(links, routes,
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

"""
    create_conservative_tree_topology_3d(Nx, Ny, Nz, patch; coarse_volume=1.0)

Build static D3Q19 topology tables for a fixed ratio-2 conservative-tree patch.

The topology stores active coarse cells outside `patch`, inactive coarse
ledger cells under `patch`, and active fine children inside `patch`. Links and
routes are CPU-built tables intended to be packed or copied to a GPU-oriented
layout later; no LBM state is stored here.
"""
function create_conservative_tree_topology_3d(Nx::Integer,
                                              Ny::Integer,
                                              Nz::Integer,
                                              patch::ConservativeTreePatch3D;
                                              coarse_volume::Real=1.0)
    Nx_i = Int(Nx)
    Ny_i = Int(Ny)
    Nz_i = Int(Nz)
    _check_conservative_tree_topology_args_3d(Nx_i, Ny_i, Nz_i,
                                              patch, coarse_volume)

    cells, active_cells, cell_id_by_coord =
        _build_conservative_tree_cells_3d(Nx_i, Ny_i, Nz_i,
                                          patch, Float64(coarse_volume))

    links, routes,
    same_level_links, coarse_to_fine_links, fine_to_coarse_links,
    boundary_links, direct_routes, interface_routes, boundary_routes =
        _build_conservative_tree_links_3d(cells, active_cells, cell_id_by_coord,
                                          Nx_i, Ny_i, Nz_i, patch)

    return ConservativeTreeTopology3D(cells, links, routes, active_cells,
                                      same_level_links, coarse_to_fine_links,
                                      fine_to_coarse_links, boundary_links,
                                      direct_routes, interface_routes,
                                      boundary_routes)
end

function active_volume(topology::ConservativeTreeTopology3D)
    total = 0.0
    @inbounds for id in topology.active_cells
        total += topology.cells[id].metrics.volume
    end
    return total
end
