# Route-table construction for nested 2D conservative-tree specs.
#
# This is still a static correctness layer. It builds packet routes for active
# leaves, but it does not stream populations.

"""
    ConservativeTreeRouteTable2D

Public type or module in the grid-refinement and conservative-tree AMR API.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using KrakeNN

KrakeNN.ConservativeTreeRouteTable2D
```
"""
struct ConservativeTreeRouteTable2D
    links::Vector{ConservativeTreeLink2D}
    routes::Vector{ConservativeTreeRoute2D}
    same_level_links::Vector{Int}
    coarse_to_fine_links::Vector{Int}
    fine_to_coarse_links::Vector{Int}
    boundary_links::Vector{Int}
    direct_routes::Vector{Int}
    interface_routes::Vector{Int}
    boundary_routes::Vector{Int}
    direct_route_ranges_by_level::Vector{UnitRange{Int}}
    boundary_route_ranges_by_level::Vector{UnitRange{Int}}
    split_route_ranges_by_parent_level::Vector{UnitRange{Int}}
    coalesce_route_ranges_by_child_level::Vector{UnitRange{Int}}
end

function _active_leaf_covering_sample_2d(spec::ConservativeTreeSpec2D,
                                         sample_level::Int,
                                         i::Int,
                                         j::Int)
    0 <= sample_level <= spec.max_level ||
        throw(ArgumentError("sample_level is outside the tree"))
    nx = _conservative_tree_level_size_2d(spec.Nx, sample_level)
    ny = _conservative_tree_level_size_2d(spec.Ny, sample_level)
    (1 <= i <= nx && 1 <= j <= ny) || return 0

    for level in sample_level:-1:0
        scale = 1 << (sample_level - level)
        ai = div(i - 1, scale) + 1
        aj = div(j - 1, scale) + 1
        cell_id = conservative_tree_cell_id_2d(spec, level, ai, aj)
        cell_id == 0 && continue
        spec.cells[cell_id].active && return cell_id
    end
    return 0
end

@inline function _route_kind_for_level_pair_2d(src::ConservativeTreeCell2D,
                                               dst::ConservativeTreeCell2D,
                                               q::Int)
    delta = dst.level - src.level
    if delta == 0
        return DIRECT
    end
    abs(delta) == 1 ||
        throw(ArgumentError("route crosses more than one AMR level"))
    is_corner = d2q9_cx(q) != 0 && d2q9_cy(q) != 0
    if delta == 1
        return is_corner ? SPLIT_CORNER : SPLIT_FACE
    end
    return is_corner ? COALESCE_CORNER : COALESCE_FACE
end

@inline function _link_kind_for_route_kinds_2d(kinds::Vector{RouteKind})
    has_boundary = false
    has_split = false
    has_coalesce = false
    @inbounds for kind in kinds
        if kind == ROUTE_BOUNDARY
            has_boundary = true
        elseif kind == SPLIT_FACE || kind == SPLIT_CORNER
            has_split = true
        elseif kind == COALESCE_FACE || kind == COALESCE_CORNER
            has_coalesce = true
        end
    end
    has_boundary && return BOUNDARY
    has_split && return COARSE_TO_FINE
    has_coalesce && return FINE_TO_COARSE
    return SAME_LEVEL
end

function _accumulate_route_spec_2d!(dsts::Vector{Int},
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

function _sampled_route_specs_for_active_cell_2d(spec::ConservativeTreeSpec2D,
                                                 src_id::Int,
                                                 q::Int,
                                                 sample_level::Int;
                                                 periodic_x::Bool=false)
    dsts = Int[]
    weights = Float64[]
    kinds = RouteKind[]
    return _sampled_route_specs_for_active_cell_2d!(
        dsts, weights, kinds, spec, src_id, q, sample_level;
        periodic_x=periodic_x)
end

function _sampled_route_specs_for_active_cell_2d!(
        dsts::Vector{Int},
        weights::Vector{Float64},
        kinds::Vector{RouteKind},
        spec::ConservativeTreeSpec2D,
        src_id::Int,
        q::Int,
        sample_level::Int;
        periodic_x::Bool=false)
    empty!(dsts)
    empty!(weights)
    empty!(kinds)
    src = spec.cells[src_id]
    scale = 1 << (sample_level - src.level)
    cx = d2q9_cx(q)
    cy = d2q9_cy(q)
    nx_sample = _conservative_tree_level_size_2d(spec.Nx, sample_level)
    weight = 1.0 / Float64(scale * scale)

    for sj in 1:scale, si in 1:scale
        sample_i = (src.i - 1) * scale + si + cx
        sample_j = (src.j - 1) * scale + sj + cy
        if periodic_x
            sample_i = mod1(sample_i, nx_sample)
        end
        dst_id = _active_leaf_covering_sample_2d(spec, sample_level,
                                                 sample_i, sample_j)
        if dst_id == 0
            _accumulate_route_spec_2d!(dsts, weights, kinds,
                                       0, weight, ROUTE_BOUNDARY)
        else
            dst = spec.cells[dst_id]
            kind = _route_kind_for_level_pair_2d(src, dst, q)
            _accumulate_route_spec_2d!(dsts, weights, kinds,
                                       dst_id, weight, kind)
        end
    end
    return dsts, weights, kinds
end

function _route_source_touches_finer_2d(spec::ConservativeTreeSpec2D,
                                        src::ConservativeTreeCell2D)
    src.level < spec.max_level || return false
    nx = _conservative_tree_level_size_2d(spec.Nx, src.level)
    ny = _conservative_tree_level_size_2d(spec.Ny, src.level)

    for dj in -1:1, di in -1:1
        di == 0 && dj == 0 && continue
        i = src.i + di
        j = src.j + dj
        1 <= i <= nx || continue
        1 <= j <= ny || continue
        cell_id = conservative_tree_cell_id_2d(spec, src.level, i, j)
        cell_id == 0 && continue
        spec.cells[cell_id].active || return true
    end
    return false
end

function _route_specs_for_active_cell_2d(spec::ConservativeTreeSpec2D,
                                         src_id::Int,
                                         q::Int;
                                         periodic_x::Bool=false)
    return _sampled_route_specs_for_active_cell_2d(
        spec, src_id, q, spec.max_level; periodic_x=periodic_x)
end

function _route_specs_for_active_cell_2d!(
        dsts::Vector{Int},
        weights::Vector{Float64},
        kinds::Vector{RouteKind},
        spec::ConservativeTreeSpec2D,
        src_id::Int,
        q::Int;
        periodic_x::Bool=false)
    return _sampled_route_specs_for_active_cell_2d!(
        dsts, weights, kinds, spec, src_id, q, spec.max_level;
        periodic_x=periodic_x)
end

function _push_multilevel_link_routes_2d!(
        links::Vector{ConservativeTreeLink2D},
        routes::Vector{ConservativeTreeRoute2D},
        same_level_links::Vector{Int},
        coarse_to_fine_links::Vector{Int},
        fine_to_coarse_links::Vector{Int},
        boundary_links::Vector{Int},
        direct_routes::Vector{Int},
        interface_routes::Vector{Int},
        boundary_routes::Vector{Int},
        src_id::Int,
        q::Int,
        dsts::Vector{Int},
        weights::Vector{Float64},
        kinds::Vector{RouteKind})
    link_kind = _link_kind_for_route_kinds_2d(kinds)
    push!(links, ConservativeTreeLink2D(src_id, q, link_kind))
    link_id = length(links)
    if link_kind == SAME_LEVEL
        push!(same_level_links, link_id)
    elseif link_kind == COARSE_TO_FINE
        push!(coarse_to_fine_links, link_id)
    elseif link_kind == FINE_TO_COARSE
        push!(fine_to_coarse_links, link_id)
    else
        push!(boundary_links, link_id)
    end

    @inbounds for idx in eachindex(dsts)
        push!(routes, ConservativeTreeRoute2D(src_id, dsts[idx], q,
                                              weights[idx], kinds[idx]))
        route_id = length(routes)
        if kinds[idx] == DIRECT
            push!(direct_routes, route_id)
        elseif kinds[idx] == ROUTE_BOUNDARY
            push!(boundary_routes, route_id)
        else
            push!(interface_routes, route_id)
        end
    end
    return nothing
end

function _empty_unit_range_2d(start::Int)
    return start:(start - 1)
end

function _ranges_from_counts_2d(counts::Vector{Int}, first_index::Int)
    ranges = Vector{UnitRange{Int}}(undef, length(counts))
    cursor = first_index
    @inbounds for idx in eachindex(counts)
        n = counts[idx]
        ranges[idx] = n == 0 ? _empty_unit_range_2d(cursor) :
                      cursor:(cursor + n - 1)
        cursor += n
    end
    return ranges
end

function _partition_route_ids_by_source_level_2d!(
        route_ids::Vector{Int},
        spec::ConservativeTreeSpec2D,
        routes::Vector{ConservativeTreeRoute2D})
    counts = zeros(Int, spec.max_level + 1)
    @inbounds for route_id in route_ids
        level = spec.cells[routes[route_id].src].level
        counts[level + 1] += 1
    end
    sort!(route_ids; by=route_id -> spec.cells[routes[route_id].src].level)
    ranges = _ranges_from_counts_2d(counts, 1)
    return ranges
end

@inline function _interface_route_partition_key_2d(
        spec::ConservativeTreeSpec2D,
        routes::Vector{ConservativeTreeRoute2D},
        route_id::Int)
    route = routes[route_id]
    src_level = spec.cells[route.src].level
    if route.kind == SPLIT_FACE || route.kind == SPLIT_CORNER
        return src_level
    end
    return spec.max_level + 1 + src_level
end

function _partition_interface_route_ids_by_level_2d!(
        interface_routes::Vector{Int},
        spec::ConservativeTreeSpec2D,
        routes::Vector{ConservativeTreeRoute2D})
    split_counts = zeros(Int, spec.max_level)
    coalesce_counts = zeros(Int, spec.max_level + 1)
    @inbounds for route_id in interface_routes
        route = routes[route_id]
        src_level = spec.cells[route.src].level
        if route.kind == SPLIT_FACE || route.kind == SPLIT_CORNER
            src_level < spec.max_level ||
                throw(ArgumentError("split route source cannot be at max_level"))
            split_counts[src_level + 1] += 1
        elseif route.kind == COALESCE_FACE || route.kind == COALESCE_CORNER
            src_level > 0 ||
                throw(ArgumentError("coalesce route source cannot be at level 0"))
            coalesce_counts[src_level + 1] += 1
        end
    end

    split_ranges = _ranges_from_counts_2d(split_counts, 1)
    split_total = sum(split_counts)
    coalesce_ranges = _ranges_from_counts_2d(coalesce_counts, split_total + 1)
    sort!(interface_routes;
          by=route_id -> _interface_route_partition_key_2d(
              spec, routes, route_id))

    return split_ranges, coalesce_ranges
end

"""
    create_conservative_tree_route_table_2d(spec; periodic_x=false)

Build a static packet route table for every active leaf and D2Q9 population in
a nested conservative-tree spec. Routes are generated by leaf-equivalent
sampling on the finest level of the tree, then mapped back to the active leaf
that owns each sample.

Using the global finest level is more expensive than local one-level sampling,
but it is the conservative-tree reference path: the active leaves form an exact
partition at that level, which keeps closed rest states well-balanced through
nested refinement.

When `periodic_x=true`, sampled routes that leave the x-domain are wrapped
before the active-leaf lookup. The y-domain remains physical and still
produces `ROUTE_BOUNDARY` routes for wall/open-boundary closures.
"""
function create_conservative_tree_route_table_2d(spec::ConservativeTreeSpec2D;
                                                 periodic_x::Bool=false)
    links = ConservativeTreeLink2D[]
    routes = ConservativeTreeRoute2D[]
    same_level_links = Int[]
    coarse_to_fine_links = Int[]
    fine_to_coarse_links = Int[]
    boundary_links = Int[]
    direct_routes = Int[]
    interface_routes = Int[]
    boundary_routes = Int[]
    link_hint = 9 * length(spec.active_cells)
    sizehint!(links, link_hint)
    sizehint!(same_level_links, link_hint)
    sizehint!(direct_routes, link_hint)

    dsts = Int[]
    weights = Float64[]
    kinds = RouteKind[]
    sizehint!(dsts, 8)
    sizehint!(weights, 8)
    sizehint!(kinds, 8)

    for src_id in spec.active_cells
        for q in 1:9
            _route_specs_for_active_cell_2d!(
                dsts, weights, kinds, spec, src_id, q;
                periodic_x=periodic_x)
            _push_multilevel_link_routes_2d!(
                links, routes, same_level_links, coarse_to_fine_links,
                fine_to_coarse_links, boundary_links, direct_routes,
                interface_routes, boundary_routes, src_id, q,
                dsts, weights, kinds)
        end
    end

    direct_ranges = _partition_route_ids_by_source_level_2d!(
        direct_routes, spec, routes)
    boundary_ranges = _partition_route_ids_by_source_level_2d!(
        boundary_routes, spec, routes)
    split_ranges, coalesce_ranges =
        _partition_interface_route_ids_by_level_2d!(
            interface_routes, spec, routes)

    return ConservativeTreeRouteTable2D(
        links, routes, same_level_links, coarse_to_fine_links,
        fine_to_coarse_links, boundary_links, direct_routes, interface_routes,
        boundary_routes, direct_ranges, boundary_ranges,
        split_ranges, coalesce_ranges)
end
