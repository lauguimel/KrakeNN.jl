# Conservative projection helpers for nested 2D conservative-tree specs.
#
# The storage contract is intentionally minimal: one row per tree cell, nine
# D2Q9 integrated populations per row. Active leaves and inactive parent ledgers
# live in the same matrix so projection tests can run before the runtime state
# backend exists.

function _check_conservative_tree_F_2d(F::AbstractMatrix,
                                       spec::ConservativeTreeSpec2D)
    size(F, 1) == length(spec.cells) ||
        throw(ArgumentError("F must have one row per conservative-tree cell"))
    size(F, 2) == 9 ||
        throw(ArgumentError("F must have 9 D2Q9 populations per row"))
    return nothing
end

"""
    allocate_conservative_tree_F_2d(spec::ConservativeTreeSpec2D;

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.allocate_conservative_tree_F_2d)
```
"""
function allocate_conservative_tree_F_2d(spec::ConservativeTreeSpec2D;
                                         T::Type{<:Real}=Float64)
    return zeros(T, length(spec.cells), 9)
end

"""
    active_population_sums_F_2d(F::AbstractMatrix,

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.active_population_sums_F_2d)
```
"""
function active_population_sums_F_2d(F::AbstractMatrix,
                                     spec::ConservativeTreeSpec2D)
    _check_conservative_tree_F_2d(F, spec)
    sums = zeros(eltype(F), 9)
    @inbounds for cell_id in spec.active_cells
        for q in 1:9
            sums[q] += F[cell_id, q]
        end
    end
    return sums
end

"""
    level_population_sums_F_2d(F::AbstractMatrix,

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.level_population_sums_F_2d)
```
"""
function level_population_sums_F_2d(F::AbstractMatrix,
                                    spec::ConservativeTreeSpec2D,
                                    level::Integer)
    _check_conservative_tree_F_2d(F, spec)
    level_i = Int(level)
    sums = zeros(eltype(F), 9)
    @inbounds for (cell_id, cell) in pairs(spec.cells)
        cell.level == level_i || continue
        for q in 1:9
            sums[q] += F[cell_id, q]
        end
    end
    return sums
end

function _coalesce_conservative_tree_parent_F_2d!(F::AbstractMatrix,
                                                  children::NTuple{4,Int},
                                                  parent_id::Int)
    @inbounds for q in 1:9
        F[parent_id, q] = F[children[1], q] + F[children[2], q] +
                          F[children[3], q] + F[children[4], q]
    end
    return nothing
end

"""
    coalesce_conservative_tree_ledgers_F_2d!(F::AbstractMatrix,

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.coalesce_conservative_tree_ledgers_F_2d!)
```
"""
function coalesce_conservative_tree_ledgers_F_2d!(F::AbstractMatrix,
                                                  spec::ConservativeTreeSpec2D)
    _check_conservative_tree_F_2d(F, spec)
    for level in (spec.max_level - 1):-1:0
        @inbounds for (cell_id, cell) in pairs(spec.cells)
            cell.level == level || continue
            children = spec.children[cell_id]
            children == (0, 0, 0, 0) && continue
            _coalesce_conservative_tree_parent_F_2d!(F, children, cell_id)
        end
    end
    return F
end

function _explode_conservative_tree_parent_F_2d!(F::AbstractMatrix,
                                                 children::NTuple{4,Int},
                                                 parent_id::Int)
    @inbounds for q in 1:9
        fq = F[parent_id, q] / 4
        F[children[1], q] = fq
        F[children[2], q] = fq
        F[children[3], q] = fq
        F[children[4], q] = fq
    end
    return nothing
end

"""
    explode_conservative_tree_ledgers_F_2d!(F::AbstractMatrix,

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.explode_conservative_tree_ledgers_F_2d!)
```
"""
function explode_conservative_tree_ledgers_F_2d!(F::AbstractMatrix,
                                                 spec::ConservativeTreeSpec2D)
    _check_conservative_tree_F_2d(F, spec)
    for level in 0:(spec.max_level - 1)
        @inbounds for (cell_id, cell) in pairs(spec.cells)
            cell.level == level || continue
            children = spec.children[cell_id]
            children == (0, 0, 0, 0) && continue
            _explode_conservative_tree_parent_F_2d!(F, children, cell_id)
        end
    end
    return F
end
