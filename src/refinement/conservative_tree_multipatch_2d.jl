# Multi-patch ownership tables for conservative-tree D2Q9 AMR.
#
# This layer is intentionally ownership-only: it defines which patch owns each
# refined parent and each fine leaf. It does not yet build routes between
# multiple refined regions.

"""
    ConservativeTreePatchSet2D

Public type or module in the grid-refinement and conservative-tree AMR API.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using KrakeNN

KrakeNN.ConservativeTreePatchSet2D
```
"""
struct ConservativeTreePatchSet2D{T}
    Nx::Int
    Ny::Int
    patches::Vector{ConservativeTreePatch2D{T}}
    parent_owner::Matrix{Int}
    leaf_owner::Matrix{Int}
end

function _check_multipatch_domain_2d(Nx::Int, Ny::Int)
    Nx > 0 || throw(ArgumentError("Nx must be positive"))
    Ny > 0 || throw(ArgumentError("Ny must be positive"))
    return nothing
end

function _check_multipatch_patch_layout_2d(Nx::Int,
                                           Ny::Int,
                                           patch::ConservativeTreePatch2D)
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

function _build_conservative_tree_patch_owners_2d(Nx::Int,
                                                  Ny::Int,
                                                  patches::Vector{<:ConservativeTreePatch2D})
    _check_multipatch_domain_2d(Nx, Ny)
    parent_owner = zeros(Int, Nx, Ny)
    leaf_owner = zeros(Int, 2 * Nx, 2 * Ny)

    @inbounds for (pid, patch) in pairs(patches)
        _check_multipatch_patch_layout_2d(Nx, Ny, patch)
        for J in patch.parent_j_range, I in patch.parent_i_range
            parent_owner[I, J] == 0 ||
                throw(ArgumentError("patches overlap at parent cell ($I, $J)"))
            parent_owner[I, J] = pid
            i0 = 2 * I - 1
            j0 = 2 * J - 1
            leaf_owner[i0, j0] = pid
            leaf_owner[i0 + 1, j0] = pid
            leaf_owner[i0, j0 + 1] = pid
            leaf_owner[i0 + 1, j0 + 1] = pid
        end
    end
    return parent_owner, leaf_owner
end

"""
    create_conservative_tree_patch_set_2d(Nx, Ny, patch_ranges; T=Float64)

Create an ownership table for several disjoint ratio-2 conservative-tree
patches. `patch_ranges` is an iterable of `(i_range, j_range)` tuples.
"""
function create_conservative_tree_patch_set_2d(
        Nx::Integer,
        Ny::Integer,
        patch_ranges;
        T::Type{<:Real}=Float64)
    Nx_i = Int(Nx)
    Ny_i = Int(Ny)
    _check_multipatch_domain_2d(Nx_i, Ny_i)

    patches = ConservativeTreePatch2D{T}[]
    for ranges in patch_ranges
        length(ranges) == 2 ||
            throw(ArgumentError("each patch range entry must be (i_range, j_range)"))
        push!(patches, create_conservative_tree_patch_2d(ranges[1], ranges[2]; T=T))
    end
    parent_owner, leaf_owner =
        _build_conservative_tree_patch_owners_2d(Nx_i, Ny_i, patches)
    return ConservativeTreePatchSet2D{T}(Nx_i, Ny_i, patches, parent_owner, leaf_owner)
end

function create_conservative_tree_patch_set_2d(
        Nx::Integer,
        Ny::Integer,
        patches::Vector{ConservativeTreePatch2D{T}}) where T
    Nx_i = Int(Nx)
    Ny_i = Int(Ny)
    parent_owner, leaf_owner =
        _build_conservative_tree_patch_owners_2d(Nx_i, Ny_i, patches)
    return ConservativeTreePatchSet2D{T}(Nx_i, Ny_i, patches, parent_owner, leaf_owner)
end

function active_volume(patch_set::ConservativeTreePatchSet2D)
    active_coarse = count(iszero, patch_set.parent_owner)
    active_fine = count(!iszero, patch_set.leaf_owner)
    return active_coarse + 0.25 * active_fine
end

"""
    active_coarse_mask(patch_set::ConservativeTreePatchSet2D)

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.active_coarse_mask)
```
"""
function active_coarse_mask(patch_set::ConservativeTreePatchSet2D)
    mask = trues(patch_set.Nx, patch_set.Ny)
    @inbounds for J in 1:patch_set.Ny, I in 1:patch_set.Nx
        patch_set.parent_owner[I, J] != 0 && (mask[I, J] = false)
    end
    return mask
end

"""
    conservative_tree_parent_owner_2d(patch_set::ConservativeTreePatchSet2D,

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.conservative_tree_parent_owner_2d)
```
"""
@inline function conservative_tree_parent_owner_2d(patch_set::ConservativeTreePatchSet2D,
                                                   I::Integer,
                                                   J::Integer)
    i = Int(I)
    j = Int(J)
    1 <= i <= patch_set.Nx || throw(ArgumentError("I is outside the coarse domain"))
    1 <= j <= patch_set.Ny || throw(ArgumentError("J is outside the coarse domain"))
    return patch_set.parent_owner[i, j]
end

"""
    conservative_tree_leaf_owner_2d(patch_set::ConservativeTreePatchSet2D,

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.conservative_tree_leaf_owner_2d)
```
"""
@inline function conservative_tree_leaf_owner_2d(patch_set::ConservativeTreePatchSet2D,
                                                 i_leaf::Integer,
                                                 j_leaf::Integer)
    i = Int(i_leaf)
    j = Int(j_leaf)
    1 <= i <= 2 * patch_set.Nx || throw(ArgumentError("i_leaf is outside the leaf domain"))
    1 <= j <= 2 * patch_set.Ny || throw(ArgumentError("j_leaf is outside the leaf domain"))
    return patch_set.leaf_owner[i, j]
end

"""
    conservative_tree_patch_owner_counts_2d(patch_set::ConservativeTreePatchSet2D)

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.conservative_tree_patch_owner_counts_2d)
```
"""
function conservative_tree_patch_owner_counts_2d(patch_set::ConservativeTreePatchSet2D)
    counts = zeros(Int, length(patch_set.patches))
    @inbounds for owner in patch_set.parent_owner
        owner == 0 && continue
        counts[owner] += 1
    end
    return counts
end

function _conservative_tree_region_to_parent_ranges_2d(region::NTuple{4,Float64},
                                                       Nx::Int,
                                                       Ny::Int,
                                                       Lx::Real,
                                                       Ly::Real)
    Nx > 0 || throw(ArgumentError("Nx must be positive"))
    Ny > 0 || throw(ArgumentError("Ny must be positive"))
    Lx > 0 || throw(ArgumentError("Lx must be positive"))
    Ly > 0 || throw(ArgumentError("Ly must be positive"))

    x_min, y_min, x_max, y_max = region
    x_max > x_min || throw(ArgumentError("refine region must have x_max > x_min"))
    y_max > y_min || throw(ArgumentError("refine region must have y_max > y_min"))

    dx = Float64(Lx) / Nx
    dy = Float64(Ly) / Ny
    i_start = max(1, floor(Int, x_min / dx) + 1)
    j_start = max(1, floor(Int, y_min / dy) + 1)
    i_end = min(Nx, ceil(Int, x_max / dx))
    j_end = min(Ny, ceil(Int, y_max / dy))
    i_start <= i_end || throw(ArgumentError("refine region does not overlap the x domain"))
    j_start <= j_end || throw(ArgumentError("refine region does not overlap the y domain"))
    return i_start:i_end, j_start:j_end
end

"""
    conservative_tree_patch_ranges_from_krk_refines_2d(domain, refinements)

Convert parsed `.krk` `RefineSetup` entries to conservative-tree parent-cell
ranges. This helper is intentionally parser-adjacent and route-native specific:
it accepts only 2D `ratio = 2` base-grid refine blocks.
"""
function conservative_tree_patch_ranges_from_krk_refines_2d(domain,
                                                            refinements)
    ranges = Tuple{UnitRange{Int},UnitRange{Int}}[]
    for ref in refinements
        getproperty(ref, :is_3d) &&
            throw(ArgumentError("3D Refine blocks are not valid for 2D conservative-tree AMR"))
        getproperty(ref, :ratio) == 2 ||
            throw(ArgumentError("conservative-tree AMR currently requires Refine ratio = 2"))
        isempty(getproperty(ref, :parent)) ||
            throw(ArgumentError("nested Refine parent blocks are not yet supported by conservative-tree AMR"))
        push!(ranges, _conservative_tree_region_to_parent_ranges_2d(
            getproperty(ref, :region),
            Int(getproperty(domain, :Nx)),
            Int(getproperty(domain, :Ny)),
            getproperty(domain, :Lx),
            getproperty(domain, :Ly)))
    end
    return ranges
end

"""
    create_conservative_tree_patch_set_from_krk_2d(setup; T=Float64)

Build a conservative-tree multi-patch ownership table from a parsed `.krk`
`SimulationSetup`. This does not dispatch a simulation; it is a small helper
for DSL-driven AMR setup tests and future route-native runners.
"""
function create_conservative_tree_patch_set_from_krk_2d(setup;
                                                        T::Type{<:Real}=Float64)
    getproperty(setup, :lattice) == :D2Q9 ||
        throw(ArgumentError("conservative-tree 2D patch sets require D2Q9"))
    domain = getproperty(setup, :domain)
    ranges = conservative_tree_patch_ranges_from_krk_refines_2d(
        domain, getproperty(setup, :refinements))
    return create_conservative_tree_patch_set_2d(
        Int(getproperty(domain, :Nx)), Int(getproperty(domain, :Ny)), ranges; T=T)
end
