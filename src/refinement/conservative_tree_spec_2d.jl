# Nested 2D conservative-tree specification.
#
# This file is deliberately state-free: it builds and validates a static tree of
# active leaves, but it does not own LBM populations and does not dispatch a
# multilevel runtime.

"""
    ConservativeTreeRefineBlock2D

Public type or module in the grid-refinement and conservative-tree AMR API.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using KrakeNN

KrakeNN.ConservativeTreeRefineBlock2D
```
"""
struct ConservativeTreeRefineBlock2D
    name::String
    parent::String
    i_range::UnitRange{Int}
    j_range::UnitRange{Int}
end

"""
    ConservativeTreeSpec2D

Public type or module in the grid-refinement and conservative-tree AMR API.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using KrakeNN

KrakeNN.ConservativeTreeSpec2D
```
"""
struct ConservativeTreeSpec2D
    Nx::Int
    Ny::Int
    max_level::Int
    cells::Vector{ConservativeTreeCell2D}
    active_cells::Vector{Int}
    key_to_id::Dict{Tuple{Int,Int,Int},Int}
    children::Vector{NTuple{4,Int}}
    refine_parent::Dict{String,String}
    refine_level::Dict{String,Int}
    refine_i_range::Dict{String,UnitRange{Int}}
    refine_j_range::Dict{String,UnitRange{Int}}
    balance::Int
end

function ConservativeTreeRefineBlock2D(name::AbstractString,
                                       i_range::AbstractUnitRange{<:Integer},
                                       j_range::AbstractUnitRange{<:Integer};
                                       parent::AbstractString="")
    isempty(name) && throw(ArgumentError("refine block name must be nonempty"))
    String(name) == "base" &&
        throw(ArgumentError("refine block name 'base' is reserved"))
    isempty(i_range) && throw(ArgumentError("i_range must be nonempty"))
    isempty(j_range) && throw(ArgumentError("j_range must be nonempty"))
    i = Int(first(i_range)):Int(last(i_range))
    j = Int(first(j_range)):Int(last(j_range))
    return ConservativeTreeRefineBlock2D(String(name), String(parent), i, j)
end

@inline _conservative_tree_base_parent_name_2d(parent::AbstractString) =
    isempty(parent) || parent == "base"

@inline function _conservative_tree_level_size_2d(N::Int, level::Int)
    return N << level
end

@inline function _conservative_tree_child_range_2d(r::UnitRange{Int})
    return (2 * first(r) - 1):(2 * last(r))
end

@inline function _conservative_tree_parent_range_from_child_2d(r::UnitRange{Int})
    return (div(first(r) - 1, 2) + 1):(div(last(r) - 1, 2) + 1)
end

@inline function _conservative_tree_pad_range_2d(r::UnitRange{Int},
                                                 pad::Int,
                                                 upper::Int)
    lo = max(1, first(r) - pad)
    hi = min(upper, last(r) + pad)
    return lo:hi
end

function _conservative_tree_ratio_levels_2d(ratio::Integer)
    ratio_i = Int(ratio)
    ratio_i >= 2 ||
        throw(ArgumentError("conservative-tree specs require Refine ratio >= 2"))
    ispow2(ratio_i) ||
        throw(ArgumentError("conservative-tree specs require power-of-two Refine ratios"))

    levels = 0
    r = ratio_i
    while r > 1
        r = div(r, 2)
        levels += 1
    end
    return levels
end

function _conservative_tree_auto_block_name_2d(name::String,
                                               level_index::Int,
                                               nlevels::Int)
    level_index == nlevels && return name
    return "$(name)_L$(level_index)"
end

@inline function _conservative_tree_cell_key_2d(level::Int, i::Int, j::Int)
    return (level, i, j)
end

function _check_refine_block_domain_2d(block::ConservativeTreeRefineBlock2D,
                                       Nx::Int,
                                       Ny::Int,
                                       level::Int)
    nx = _conservative_tree_level_size_2d(Nx, level)
    ny = _conservative_tree_level_size_2d(Ny, level)
    first(block.i_range) >= 1 ||
        throw(ArgumentError("refine block $(block.name) starts outside x domain"))
    last(block.i_range) <= nx ||
        throw(ArgumentError("refine block $(block.name) ends outside x domain"))
    first(block.j_range) >= 1 ||
        throw(ArgumentError("refine block $(block.name) starts outside y domain"))
    last(block.j_range) <= ny ||
        throw(ArgumentError("refine block $(block.name) ends outside y domain"))
    return nothing
end

function _push_conservative_tree_cell_2d!(
        cells::Vector{ConservativeTreeCell2D},
        children::Vector{NTuple{4,Int}},
        key_to_id::Dict{Tuple{Int,Int,Int},Int},
        level::Int,
        i::Int,
        j::Int,
        active::Bool,
        parent::Int,
        coarse_volume::Float64)
    volume = coarse_volume / Float64(4^level)
    push!(cells, ConservativeTreeCell2D(level, i, j, active,
                                        CartesianMetrics2D(volume), parent))
    push!(children, (0, 0, 0, 0))
    id = length(cells)
    key = _conservative_tree_cell_key_2d(level, i, j)
    haskey(key_to_id, key) &&
        throw(ArgumentError("duplicate conservative-tree cell key $key"))
    key_to_id[key] = id
    return id
end

function _deactivate_conservative_tree_cell_2d!(
        cells::Vector{ConservativeTreeCell2D},
        cell_id::Int)
    cell = cells[cell_id]
    cell.active ||
        throw(ArgumentError("refine overlaps an already refined cell at " *
                            "level $(cell.level), i=$(cell.i), j=$(cell.j)"))
    cells[cell_id] = ConservativeTreeCell2D(cell.level, cell.i, cell.j, false,
                                            cell.metrics, cell.parent)
    return nothing
end

function _refine_conservative_tree_parent_cell_2d!(
        cells::Vector{ConservativeTreeCell2D},
        children::Vector{NTuple{4,Int}},
        key_to_id::Dict{Tuple{Int,Int,Int},Int},
        parent_level::Int,
        i::Int,
        j::Int,
        coarse_volume::Float64)
    parent_key = _conservative_tree_cell_key_2d(parent_level, i, j)
    parent_id = get(key_to_id, parent_key, 0)
    parent_id != 0 ||
        throw(ArgumentError("refine targets a cell that does not exist at " *
                            "level $parent_level, i=$i, j=$j"))
    _deactivate_conservative_tree_cell_2d!(cells, parent_id)

    child_level = parent_level + 1
    child_ids = Int[]
    for jy in 1:2, ix in 1:2
        child_i = 2 * i - 2 + ix
        child_j = 2 * j - 2 + jy
        child_id = _push_conservative_tree_cell_2d!(
            cells, children, key_to_id, child_level, child_i, child_j,
            true, parent_id, coarse_volume)
        push!(child_ids, child_id)
    end
    children[parent_id] = (child_ids[1], child_ids[2], child_ids[3], child_ids[4])
    return nothing
end

function _candidate_indices_for_axis_2d(lo::Int,
                                        hi::Int,
                                        width::Int,
                                        dir::Int,
                                        n_level::Int)
    if dir == 0
        first_idx = div(lo, width) + 1
        last_idx = div(hi - 1, width) + 1
        return max(first_idx, 1):min(last_idx, n_level)
    elseif dir > 0
        lo % width == 0 || return 1:0
        idx = div(lo, width) + 1
        return 1 <= idx <= n_level ? (idx:idx) : (1:0)
    else
        hi % width == 0 || return 1:0
        idx = div(hi, width)
        return 1 <= idx <= n_level ? (idx:idx) : (1:0)
    end
end

function _validate_conservative_tree_balance_2d(cells::Vector{ConservativeTreeCell2D},
                                                active_cells::Vector{Int},
                                                key_to_id,
                                                Nx::Int,
                                                Ny::Int,
                                                max_level::Int,
                                                balance::Int)
    balance >= 0 || throw(ArgumentError("balance must be nonnegative"))
    isempty(active_cells) && return nothing

    for src_id in active_cells
        src = cells[src_id]
        src_width = 1 << (max_level - src.level)
        src_x0 = (src.i - 1) * src_width
        src_x1 = src.i * src_width
        src_y0 = (src.j - 1) * src_width
        src_y1 = src.j * src_width

        for dy in -1:1, dx in -1:1
            dx == 0 && dy == 0 && continue
            for level in 0:max_level
                width = 1 << (max_level - level)
                nx_level = _conservative_tree_level_size_2d(Nx, level)
                ny_level = _conservative_tree_level_size_2d(Ny, level)

                i_candidates = _candidate_indices_for_axis_2d(
                    dx > 0 ? src_x1 : src_x0,
                    dx < 0 ? src_x0 : src_x1,
                    width, dx, nx_level)
                j_candidates = _candidate_indices_for_axis_2d(
                    dy > 0 ? src_y1 : src_y0,
                    dy < 0 ? src_y0 : src_y1,
                    width, dy, ny_level)

                for j in j_candidates, i in i_candidates
                    dst_id = get(key_to_id,
                                 _conservative_tree_cell_key_2d(level, i, j), 0)
                    dst_id == 0 && continue
                    dst_id == src_id && continue
                    dst = cells[dst_id]
                    dst.active || continue
                    abs(src.level - dst.level) <= balance ||
                        throw(ArgumentError("2:1 balance violation between " *
                            "level $(src.level) cell ($(src.i), $(src.j)) " *
                            "and level $(dst.level) cell ($(dst.i), $(dst.j))"))
                end
            end
        end
    end
    return nothing
end

function _validate_conservative_tree_refine_parent_2d(
        block::ConservativeTreeRefineBlock2D,
        parent_name::String,
        refine_level::Dict{String,Int},
        refine_i_range::Dict{String,UnitRange{Int}},
        refine_j_range::Dict{String,UnitRange{Int}})
    _conservative_tree_base_parent_name_2d(parent_name) && return 0
    haskey(refine_level, parent_name) ||
        throw(ArgumentError("refine block $(block.name) references missing " *
                            "parent '$parent_name'"))
    parent_i = refine_i_range[parent_name]
    parent_j = refine_j_range[parent_name]
    first(block.i_range) >= first(parent_i) &&
        last(block.i_range) <= last(parent_i) ||
        throw(ArgumentError("refine block $(block.name) is outside parent " *
                            "'$parent_name' in x"))
    first(block.j_range) >= first(parent_j) &&
        last(block.j_range) <= last(parent_j) ||
        throw(ArgumentError("refine block $(block.name) is outside parent " *
                            "'$parent_name' in y"))
    return refine_level[parent_name]
end

"""
    create_conservative_tree_spec_2d(Nx, Ny, blocks; balance=1,
                                     coarse_volume=1.0)

Build a static nested conservative-tree specification from named ratio-2 refine
blocks. `blocks` ranges are expressed in their parent level coordinates:
base-level coordinates for root blocks, and the named parent's leaf coordinates
for child blocks.

This function validates ownership and 2:1 balance only. It does not allocate
LBM populations and it does not enable multilevel streaming.
"""
function create_conservative_tree_spec_2d(Nx::Integer,
                                          Ny::Integer,
                                          blocks;
                                          balance::Integer=1,
                                          coarse_volume::Real=1.0)
    Nx_i = Int(Nx)
    Ny_i = Int(Ny)
    _check_multipatch_domain_2d(Nx_i, Ny_i)
    coarse_volume > 0 || throw(ArgumentError("coarse_volume must be positive"))
    balance_i = Int(balance)

    cells = ConservativeTreeCell2D[]
    children = NTuple{4,Int}[]
    key_to_id = Dict{Tuple{Int,Int,Int},Int}()
    for j in 1:Ny_i, i in 1:Nx_i
        _push_conservative_tree_cell_2d!(
            cells, children, key_to_id, 0, i, j, true, 0, Float64(coarse_volume))
    end

    refine_parent = Dict{String,String}()
    refine_level = Dict{String,Int}()
    refine_i_range = Dict{String,UnitRange{Int}}()
    refine_j_range = Dict{String,UnitRange{Int}}()
    max_level = 0

    for raw_block in blocks
        block = raw_block isa ConservativeTreeRefineBlock2D ? raw_block :
            ConservativeTreeRefineBlock2D(raw_block.name,
                                          raw_block.i_range,
                                          raw_block.j_range;
                                          parent=raw_block.parent)
        haskey(refine_level, block.name) &&
            throw(ArgumentError("duplicate refine block name '$(block.name)'"))

        parent_name = block.parent
        parent_level = _validate_conservative_tree_refine_parent_2d(
            block, parent_name, refine_level, refine_i_range, refine_j_range)
        _check_refine_block_domain_2d(block, Nx_i, Ny_i, parent_level)

        for j in block.j_range, i in block.i_range
            _refine_conservative_tree_parent_cell_2d!(
                cells, children, key_to_id, parent_level, i, j,
                Float64(coarse_volume))
        end

        child_level = parent_level + 1
        refine_parent[block.name] = parent_name
        refine_level[block.name] = child_level
        refine_i_range[block.name] = _conservative_tree_child_range_2d(block.i_range)
        refine_j_range[block.name] = _conservative_tree_child_range_2d(block.j_range)
        max_level = max(max_level, child_level)
    end

    active_cells = Int[]
    @inbounds for (id, cell) in pairs(cells)
        cell.active && push!(active_cells, id)
    end
    _validate_conservative_tree_balance_2d(cells, active_cells, key_to_id,
                                           Nx_i, Ny_i, max_level, balance_i)

    return ConservativeTreeSpec2D(Nx_i, Ny_i, max_level, cells, active_cells,
                                  key_to_id, children, refine_parent,
                                  refine_level, refine_i_range, refine_j_range,
                                  balance_i)
end

function active_volume(spec::ConservativeTreeSpec2D)
    total = 0.0
    @inbounds for id in spec.active_cells
        total += spec.cells[id].metrics.volume
    end
    return total
end

"""
    conservative_tree_cell_id_2d(spec::ConservativeTreeSpec2D,

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.conservative_tree_cell_id_2d)
```
"""
function conservative_tree_cell_id_2d(spec::ConservativeTreeSpec2D,
                                      level::Integer,
                                      i::Integer,
                                      j::Integer)
    return get(spec.key_to_id,
               _conservative_tree_cell_key_2d(Int(level), Int(i), Int(j)), 0)
end

"""
    conservative_tree_children_2d(spec::ConservativeTreeSpec2D,

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.conservative_tree_children_2d)
```
"""
function conservative_tree_children_2d(spec::ConservativeTreeSpec2D,
                                       cell_id::Integer)
    id = Int(cell_id)
    1 <= id <= length(spec.children) ||
        throw(ArgumentError("cell_id is outside the tree"))
    return spec.children[id]
end

"""
    conservative_tree_is_active_leaf_2d(spec::ConservativeTreeSpec2D,

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.conservative_tree_is_active_leaf_2d)
```
"""
function conservative_tree_is_active_leaf_2d(spec::ConservativeTreeSpec2D,
                                             cell_id::Integer)
    id = Int(cell_id)
    1 <= id <= length(spec.cells) ||
        throw(ArgumentError("cell_id is outside the tree"))
    return spec.cells[id].active
end

"""
    conservative_tree_refine_blocks_from_krk_2d(domain, refinements)

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.conservative_tree_refine_blocks_from_krk_2d)
```
"""
function conservative_tree_refine_blocks_from_krk_2d(domain, refinements)
    Nx = Int(getproperty(domain, :Nx))
    Ny = Int(getproperty(domain, :Ny))
    Lx = getproperty(domain, :Lx)
    Ly = getproperty(domain, :Ly)
    _check_multipatch_domain_2d(Nx, Ny)

    blocks = ConservativeTreeRefineBlock2D[]
    refine_level = Dict{String,Int}()
    refine_i_range = Dict{String,UnitRange{Int}}()
    refine_j_range = Dict{String,UnitRange{Int}}()

    for ref in refinements
        name = String(getproperty(ref, :name))
        name == "base" &&
            throw(ArgumentError("Refine name 'base' is reserved"))
        haskey(refine_level, name) &&
            throw(ArgumentError("duplicate Refine name '$name'"))
        getproperty(ref, :is_3d) &&
            throw(ArgumentError("3D Refine blocks are not valid for 2D conservative-tree specs"))
        nlevels = _conservative_tree_ratio_levels_2d(getproperty(ref, :ratio))

        parent = String(getproperty(ref, :parent))
        if _conservative_tree_base_parent_name_2d(parent)
            parent_level = 0
        else
            haskey(refine_level, parent) ||
                throw(ArgumentError("Refine $name references missing parent '$parent'"))
            parent_level = refine_level[parent]
        end

        target_parent_level = parent_level + nlevels - 1
        nx_level = _conservative_tree_level_size_2d(Nx, target_parent_level)
        ny_level = _conservative_tree_level_size_2d(Ny, target_parent_level)
        target_i_range, target_j_range = _conservative_tree_region_to_parent_ranges_2d(
            getproperty(ref, :region), nx_level, ny_level, Lx, Ly)

        i_ranges = Vector{UnitRange{Int}}(undef, nlevels)
        j_ranges = Vector{UnitRange{Int}}(undef, nlevels)
        i_ranges[nlevels] = target_i_range
        j_ranges[nlevels] = target_j_range
        for k in (nlevels - 1):-1:1
            child_level = parent_level + k
            child_nx = _conservative_tree_level_size_2d(Nx, child_level)
            child_ny = _conservative_tree_level_size_2d(Ny, child_level)
            padded_i = _conservative_tree_pad_range_2d(i_ranges[k + 1], 1, child_nx)
            padded_j = _conservative_tree_pad_range_2d(j_ranges[k + 1], 1, child_ny)
            i_ranges[k] = _conservative_tree_parent_range_from_child_2d(padded_i)
            j_ranges[k] = _conservative_tree_parent_range_from_child_2d(padded_j)
        end

        if !_conservative_tree_base_parent_name_2d(parent)
            parent_i = refine_i_range[parent]
            parent_j = refine_j_range[parent]
            first(i_ranges[1]) >= first(parent_i) &&
                last(i_ranges[1]) <= last(parent_i) ||
                throw(ArgumentError("Refine $name is outside parent '$parent' in x"))
            first(j_ranges[1]) >= first(parent_j) &&
                last(j_ranges[1]) <= last(parent_j) ||
                throw(ArgumentError("Refine $name is outside parent '$parent' in y"))
        end

        block_parent = parent
        for k in 1:nlevels
            block_name = _conservative_tree_auto_block_name_2d(name, k, nlevels)
            block_name == "base" &&
                throw(ArgumentError("Refine name 'base' is reserved"))
            haskey(refine_level, block_name) &&
                throw(ArgumentError("duplicate Refine name '$block_name'"))

            block = ConservativeTreeRefineBlock2D(
                block_name, i_ranges[k], j_ranges[k]; parent=block_parent)
            push!(blocks, block)
            child_level = parent_level + k
            refine_level[block_name] = child_level
            refine_i_range[block_name] =
                _conservative_tree_child_range_2d(i_ranges[k])
            refine_j_range[block_name] =
                _conservative_tree_child_range_2d(j_ranges[k])
            block_parent = block_name
        end
    end

    return blocks
end

"""
    create_conservative_tree_spec_from_krk_2d(setup; balance=1,
                                              coarse_volume=1.0)

Build the static nested conservative-tree specification described by a parsed
2D `.krk` setup. This is a DSL canary only; it intentionally does not dispatch
the multilevel conservative-tree runtime.
"""
function create_conservative_tree_spec_from_krk_2d(setup;
                                                   balance::Integer=1,
                                                   coarse_volume::Real=1.0)
    getproperty(setup, :lattice) == :D2Q9 ||
        throw(ArgumentError("conservative-tree 2D specs require D2Q9"))
    domain = getproperty(setup, :domain)
    blocks = conservative_tree_refine_blocks_from_krk_2d(
        domain, getproperty(setup, :refinements))
    return create_conservative_tree_spec_2d(
        Int(getproperty(domain, :Nx)), Int(getproperty(domain, :Ny)), blocks;
        balance=balance, coarse_volume=coarse_volume)
end
