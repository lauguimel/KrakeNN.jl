struct ConservativeTreePatch2D{T}
    parent_i_range::UnitRange{Int}
    parent_j_range::UnitRange{Int}
    ratio::Int
    fine_F::Array{T,3}
    coarse_shadow_F::Array{T,3}
end

"""
    create_conservative_tree_patch_2d(parent_i_range, parent_j_range; ratio=2, T=Float64)

Allocate an experimental fixed ratio-2 conservative-tree patch. No ghost cells
are allocated.
"""
function create_conservative_tree_patch_2d(parent_i_range::AbstractUnitRange{<:Integer},
                                           parent_j_range::AbstractUnitRange{<:Integer};
                                           ratio::Int=2,
                                           T::Type{<:Real}=Float64)
    ratio == 2 || throw(ArgumentError("only ratio=2 is implemented"))
    isempty(parent_i_range) && throw(ArgumentError("parent_i_range must be nonempty"))
    isempty(parent_j_range) && throw(ArgumentError("parent_j_range must be nonempty"))

    ip = Int(first(parent_i_range)):Int(last(parent_i_range))
    jp = Int(first(parent_j_range)):Int(last(parent_j_range))
    nx_parent = length(ip)
    ny_parent = length(jp)

    fine_F = zeros(T, ratio * nx_parent, ratio * ny_parent, 9)
    coarse_shadow_F = zeros(T, nx_parent, ny_parent, 9)
    return ConservativeTreePatch2D{T}(ip, jp, ratio, fine_F, coarse_shadow_F)
end

function _check_conservative_tree_patch_layout(patch::ConservativeTreePatch2D)
    patch.ratio == 2 || throw(ArgumentError("only ratio=2 is implemented"))
    nx_parent = length(patch.parent_i_range)
    ny_parent = length(patch.parent_j_range)
    size(patch.fine_F) == (2 * nx_parent, 2 * ny_parent, 9) ||
        throw(ArgumentError("patch.fine_F has inconsistent size"))
    size(patch.coarse_shadow_F) == (nx_parent, ny_parent, 9) ||
        throw(ArgumentError("patch.coarse_shadow_F has inconsistent size"))
    return nothing
end

"""
    coalesce_patch_to_shadow_F_2d!(patch)

Fill `patch.coarse_shadow_F` by coalescing every 2x2 fine block. This is the
fine-to-parent ledger update for the conservative tree route.
"""
function coalesce_patch_to_shadow_F_2d!(patch::ConservativeTreePatch2D)
    _check_conservative_tree_patch_layout(patch)

    nx_parent = length(patch.parent_i_range)
    ny_parent = length(patch.parent_j_range)
    @inbounds for jp in 1:ny_parent, ip in 1:nx_parent
        i0 = 2 * ip - 1
        j0 = 2 * jp - 1
        Fp = @view patch.coarse_shadow_F[ip, jp, :]
        Fc = @view patch.fine_F[i0:i0+1, j0:j0+1, :]
        coalesce_F_2d!(Fp, Fc)
    end
    return patch
end

"""
    explode_shadow_to_patch_uniform_F_2d!(patch)

Fill `patch.fine_F` by uniformly exploding each parent ledger cell into its
four fine children.
"""
function explode_shadow_to_patch_uniform_F_2d!(patch::ConservativeTreePatch2D)
    _check_conservative_tree_patch_layout(patch)

    nx_parent = length(patch.parent_i_range)
    ny_parent = length(patch.parent_j_range)
    @inbounds for jp in 1:ny_parent, ip in 1:nx_parent
        i0 = 2 * ip - 1
        j0 = 2 * jp - 1
        Fc = @view patch.fine_F[i0:i0+1, j0:j0+1, :]
        Fp = @view patch.coarse_shadow_F[ip, jp, :]
        explode_uniform_F_2d!(Fc, Fp)
    end
    return patch
end

function _check_composite_coarse_layout(coarse_F::AbstractArray{<:Any,3},
                                        patch::ConservativeTreePatch2D)
    _check_conservative_tree_patch_layout(patch)
    size(coarse_F, 3) == 9 ||
        throw(ArgumentError("coarse_F must have 9 D2Q9 populations in dimension 3"))
    first(patch.parent_i_range) >= first(axes(coarse_F, 1)) ||
        throw(ArgumentError("patch.parent_i_range starts outside coarse_F"))
    last(patch.parent_i_range) <= last(axes(coarse_F, 1)) ||
        throw(ArgumentError("patch.parent_i_range ends outside coarse_F"))
    first(patch.parent_j_range) >= first(axes(coarse_F, 2)) ||
        throw(ArgumentError("patch.parent_j_range starts outside coarse_F"))
    last(patch.parent_j_range) <= last(axes(coarse_F, 2)) ||
        throw(ArgumentError("patch.parent_j_range ends outside coarse_F"))
    return nothing
end

"""
    apply_composite_zou_he_west_F_2d!(

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.apply_composite_zou_he_west_F_2d!)
```
"""
function apply_composite_zou_he_west_F_2d!(
        coarse_F::AbstractArray{T,3},
        patch::ConservativeTreePatch2D{T},
        u_in,
        volume_coarse,
        volume_fine) where T
    _check_composite_coarse_layout(coarse_F, patch)
    @inbounds for J in axes(coarse_F, 2)
        if first(patch.parent_i_range) <= 1 <= last(patch.parent_i_range) &&
                J in patch.parent_j_range
            il, jl = _patch_local_parent_index(patch, 1, J)
            for jf in (2 * jl - 1):(2 * jl)
                apply_zou_he_west_cell_F_2d!(
                    @view(patch.fine_F[2 * il - 1, jf, :]), u_in, volume_fine)
            end
        else
            apply_zou_he_west_cell_F_2d!(
                @view(coarse_F[1, J, :]), u_in, volume_coarse)
        end
    end
    coalesce_patch_to_shadow_F_2d!(patch)
    return coarse_F, patch
end

"""
    apply_composite_zou_he_pressure_east_F_2d!(

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.apply_composite_zou_he_pressure_east_F_2d!)
```
"""
function apply_composite_zou_he_pressure_east_F_2d!(
        coarse_F::AbstractArray{T,3},
        patch::ConservativeTreePatch2D{T},
        volume_coarse,
        volume_fine;
        rho_out=one(T)) where T
    _check_composite_coarse_layout(coarse_F, patch)
    I = last(axes(coarse_F, 1))
    @inbounds for J in axes(coarse_F, 2)
        if first(patch.parent_i_range) <= I <= last(patch.parent_i_range) &&
                J in patch.parent_j_range
            il, jl = _patch_local_parent_index(patch, I, J)
            for jf in (2 * jl - 1):(2 * jl)
                apply_zou_he_pressure_east_cell_F_2d!(
                    @view(patch.fine_F[2 * il, jf, :]), volume_fine;
                    rho_out=rho_out)
            end
        else
            apply_zou_he_pressure_east_cell_F_2d!(
                @view(coarse_F[I, J, :]), volume_coarse; rho_out=rho_out)
        end
    end
    coalesce_patch_to_shadow_F_2d!(patch)
    return coarse_F, patch
end

function _check_composite_solid_mask_layout(coarse_F::AbstractArray{<:Any,3},
                                            patch::ConservativeTreePatch2D,
                                            is_solid::AbstractArray{Bool,2})
    size(is_solid) == (2 * size(coarse_F, 1), 2 * size(coarse_F, 2)) ||
        throw(ArgumentError("is_solid must have size (2*Nx, 2*Ny)"))
    @inbounds for J in axes(coarse_F, 2), I in axes(coarse_F, 1)
        _inside_range(I, J, patch.parent_i_range, patch.parent_j_range) && continue
        i0 = 2 * I - 1
        j0 = 2 * J - 1
        s11 = is_solid[i0, j0]
        s21 = is_solid[i0 + 1, j0]
        s12 = is_solid[i0, j0 + 1]
        s22 = is_solid[i0 + 1, j0 + 1]
        (s11 == s21 == s12 == s22) ||
            throw(ArgumentError("active coarse cells cannot be partially solid"))
    end
    return nothing
end

function apply_composite_zou_he_west_F_2d!(
        coarse_F::AbstractArray{T,3},
        patch::ConservativeTreePatch2D{T},
        is_solid::AbstractArray{Bool,2},
        u_in,
        volume_coarse,
        volume_fine) where T
    _check_composite_coarse_layout(coarse_F, patch)
    _check_composite_solid_mask_layout(coarse_F, patch, is_solid)
    @inbounds for J in axes(coarse_F, 2)
        if first(patch.parent_i_range) <= 1 <= last(patch.parent_i_range) &&
                J in patch.parent_j_range
            il, jl = _patch_local_parent_index(patch, 1, J)
            i_leaf = 2 * first(patch.parent_i_range) - 1
            for jf in (2 * jl - 1):(2 * jl)
                is_solid[i_leaf, jf] && continue
                apply_zou_he_west_cell_F_2d!(
                    @view(patch.fine_F[2 * il - 1, jf, :]), u_in, volume_fine)
            end
        else
            is_solid[1, 2 * J - 1] && continue
            apply_zou_he_west_cell_F_2d!(
                @view(coarse_F[1, J, :]), u_in, volume_coarse)
        end
    end
    coalesce_patch_to_shadow_F_2d!(patch)
    return coarse_F, patch
end

function apply_composite_zou_he_pressure_east_F_2d!(
        coarse_F::AbstractArray{T,3},
        patch::ConservativeTreePatch2D{T},
        is_solid::AbstractArray{Bool,2},
        volume_coarse,
        volume_fine;
        rho_out=one(T)) where T
    _check_composite_coarse_layout(coarse_F, patch)
    _check_composite_solid_mask_layout(coarse_F, patch, is_solid)
    I = last(axes(coarse_F, 1))
    @inbounds for J in axes(coarse_F, 2)
        if first(patch.parent_i_range) <= I <= last(patch.parent_i_range) &&
                J in patch.parent_j_range
            il, jl = _patch_local_parent_index(patch, I, J)
            i_leaf = 2 * last(patch.parent_i_range)
            for jf in (2 * jl - 1):(2 * jl)
                is_solid[i_leaf, jf] && continue
                apply_zou_he_pressure_east_cell_F_2d!(
                    @view(patch.fine_F[2 * il, jf, :]), volume_fine;
                    rho_out=rho_out)
            end
        else
            is_solid[2 * I - 1, 2 * J - 1] && continue
            apply_zou_he_pressure_east_cell_F_2d!(
                @view(coarse_F[I, J, :]), volume_coarse; rho_out=rho_out)
        end
    end
    coalesce_patch_to_shadow_F_2d!(patch)
    return coarse_F, patch
end

"""
    active_mass_F(coarse_F, patch)

Mass of a composite fixed-tree state: active coarse cells outside the refined
parent range plus active fine cells inside `patch`. The inactive parent ledger
region in `coarse_F` is deliberately skipped.
"""
function active_mass_F(coarse_F::AbstractArray{<:Any,3},
                       patch::ConservativeTreePatch2D)
    _check_composite_coarse_layout(coarse_F, patch)

    total = zero(coarse_F[begin, begin, 1] + patch.fine_F[begin, begin, 1])
    @inbounds for q in 1:9, j in axes(coarse_F, 2), i in axes(coarse_F, 1)
        _inside_range(i, j, patch.parent_i_range, patch.parent_j_range) && continue
        total += coarse_F[i, j, q]
    end
    return total + mass_F(patch.fine_F)
end

"""
    active_population_sums_F(coarse_F, patch)

Return the 9 active oriented-population totals of a composite fixed-tree state.
Inactive parent cells covered by the refined patch are skipped.
"""
function active_population_sums_F(coarse_F::AbstractArray{<:Any,3},
                                  patch::ConservativeTreePatch2D)
    _check_composite_coarse_layout(coarse_F, patch)

    totals = zeros(promote_type(eltype(coarse_F), eltype(patch.fine_F)), 9)
    @inbounds for q in 1:9
        for j in axes(coarse_F, 2), i in axes(coarse_F, 1)
            _inside_range(i, j, patch.parent_i_range, patch.parent_j_range) && continue
            totals[q] += coarse_F[i, j, q]
        end
        totals[q] += sum(@view patch.fine_F[:, :, q])
    end
    return totals
end

"""
    active_momentum_F(coarse_F, patch)

Momentum of a composite fixed-tree state, counting only active coarse cells and
active fine leaves.
"""
function active_momentum_F(coarse_F::AbstractArray{<:Any,3},
                           patch::ConservativeTreePatch2D)
    _check_composite_coarse_layout(coarse_F, patch)

    mx = zero(coarse_F[begin, begin, 1] + patch.fine_F[begin, begin, 1])
    my = zero(coarse_F[begin, begin, 1] + patch.fine_F[begin, begin, 1])
    @inbounds for q in 1:9
        cx = d2q9_cx(q)
        cy = d2q9_cy(q)
        for j in axes(coarse_F, 2), i in axes(coarse_F, 1)
            _inside_range(i, j, patch.parent_i_range, patch.parent_j_range) && continue
            fq = coarse_F[i, j, q]
            mx += cx * fq
            my += cy * fq
        end
    end
    fmx, fmy = momentum_F(patch.fine_F)
    return mx + fmx, my + fmy
end

"""
    active_moments_F(coarse_F::AbstractArray{<:Any,3},

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.active_moments_F)
```
"""
function active_moments_F(coarse_F::AbstractArray{<:Any,3},
                          patch::ConservativeTreePatch2D)
    m = active_mass_F(coarse_F, patch)
    mx, my = active_momentum_F(coarse_F, patch)
    return m, mx, my
end

@inline function _composite_parent_Fq(coarse_F::AbstractArray{<:Any,3},
                                      patch::ConservativeTreePatch2D,
                                      I::Int,
                                      J::Int,
                                      q::Int)
    if _inside_range(I, J, patch.parent_i_range, patch.parent_j_range)
        il, jl = _patch_local_parent_index(patch, I, J)
        return patch.coarse_shadow_F[il, jl, q]
    end
    return coarse_F[I, J, q]
end

@inline function _limited_parent_slope_x(coarse_F::AbstractArray{<:Any,3},
                                         patch::ConservativeTreePatch2D,
                                         I::Int,
                                         J::Int,
                                         q::Int)
    center = _composite_parent_Fq(coarse_F, patch, I, J, q)
    has_left = I > first(axes(coarse_F, 1))
    has_right = I < last(axes(coarse_F, 1))
    if has_left && has_right
        left = center - _composite_parent_Fq(coarse_F, patch, I - 1, J, q)
        right = _composite_parent_Fq(coarse_F, patch, I + 1, J, q) - center
        return _minmod(left, right)
    elseif has_left
        return center - _composite_parent_Fq(coarse_F, patch, I - 1, J, q)
    elseif has_right
        return _composite_parent_Fq(coarse_F, patch, I + 1, J, q) - center
    else
        return zero(center)
    end
end

@inline function _limited_parent_slope_y(coarse_F::AbstractArray{<:Any,3},
                                         patch::ConservativeTreePatch2D,
                                         I::Int,
                                         J::Int,
                                         q::Int)
    center = _composite_parent_Fq(coarse_F, patch, I, J, q)
    has_south = J > first(axes(coarse_F, 2))
    has_north = J < last(axes(coarse_F, 2))
    if has_south && has_north
        south = center - _composite_parent_Fq(coarse_F, patch, I, J - 1, q)
        north = _composite_parent_Fq(coarse_F, patch, I, J + 1, q) - center
        return _minmod(south, north)
    elseif has_south
        return center - _composite_parent_Fq(coarse_F, patch, I, J - 1, q)
    elseif has_north
        return _composite_parent_Fq(coarse_F, patch, I, J + 1, q) - center
    else
        return zero(center)
    end
end

function _explode_limited_linear_composite_F_2d!(
        leaf_block::AbstractArray{<:Any,3},
        coarse_F::AbstractArray{<:Any,3},
        patch::ConservativeTreePatch2D,
        I::Int,
        J::Int)
    _check_child_block_2d(leaf_block, "leaf_block")

    @inbounds for q in 1:9
        center = _composite_parent_Fq(coarse_F, patch, I, J, q)
        sx = _limited_parent_slope_x(coarse_F, patch, I, J, q)
        sy = _limited_parent_slope_y(coarse_F, patch, I, J, q)

        max_delta = (abs(sx) + abs(sy)) / 16
        base = center / 4
        if max_delta > zero(max_delta) && base < max_delta
            theta = base / max_delta
            sx *= theta
            sy *= theta
        end

        leaf_block[1, 1, q] = base - sx / 16 - sy / 16
        leaf_block[2, 1, q] = base + sx / 16 - sy / 16
        leaf_block[1, 2, q] = base - sx / 16 + sy / 16
        leaf_block[2, 2, q] = base + sx / 16 + sy / 16
    end
    return leaf_block
end

