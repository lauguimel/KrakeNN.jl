# Conservative cell-centered tree refinement primitives for D3Q19.
#
# These helpers manipulate integrated populations F_i = f_i * cell_volume.
# They are deliberately independent from the existing patch-based 3D
# refinement path.

const D3Q19_CX_INT = (0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0)
const D3Q19_CY_INT = (0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1)
const D3Q19_CZ_INT = (0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1)
const D3Q19_OPPOSITE_INT = (1, 3, 2, 5, 4, 7, 6, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16)

@inline function _check_d3q19_q(q::Int)
    1 <= q <= 19 || throw(ArgumentError("D3Q19 population index q must be in 1:19"))
    return q
end

"""
    d3q19_cx(q::Integer)

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.d3q19_cx)
```
"""
@inline function d3q19_cx(q::Integer)
    qi = _check_d3q19_q(Int(q))
    return D3Q19_CX_INT[qi]
end

"""
    d3q19_cy(q::Integer)

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.d3q19_cy)
```
"""
@inline function d3q19_cy(q::Integer)
    qi = _check_d3q19_q(Int(q))
    return D3Q19_CY_INT[qi]
end

"""
    d3q19_cz(q::Integer)

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.d3q19_cz)
```
"""
@inline function d3q19_cz(q::Integer)
    qi = _check_d3q19_q(Int(q))
    return D3Q19_CZ_INT[qi]
end

"""
    d3q19_opposite(q::Integer)

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.d3q19_opposite)
```
"""
@inline function d3q19_opposite(q::Integer)
    qi = _check_d3q19_q(Int(q))
    return D3Q19_OPPOSITE_INT[qi]
end

@inline function _check_d3q19_vector(F::AbstractVector, name::AbstractString)
    length(F) == 19 || throw(ArgumentError("$name must have length 19 for D3Q19"))
    return nothing
end

@inline function _check_child_block_3d(Fc::AbstractArray{<:Any,4}, name::AbstractString)
    size(Fc) == (2, 2, 2, 19) ||
        throw(ArgumentError("$name must have size (2, 2, 2, 19)"))
    return nothing
end

"""
    coalesce_F_3d!(Fp, Fc)

Coalesce the eight ratio-2 child cells `Fc[ix, iy, iz, q]` into one parent
population vector `Fp[q]`.
"""
function coalesce_F_3d!(Fp::AbstractVector, Fc::AbstractArray{<:Any,4})
    _check_d3q19_vector(Fp, "Fp")
    _check_child_block_3d(Fc, "Fc")

    @inbounds for q in 1:19
        Fp[q] = Fc[1, 1, 1, q] + Fc[2, 1, 1, q] +
                Fc[1, 2, 1, q] + Fc[2, 2, 1, q] +
                Fc[1, 1, 2, q] + Fc[2, 1, 2, q] +
                Fc[1, 2, 2, q] + Fc[2, 2, 2, q]
    end
    return Fp
end

"""
    explode_uniform_F_3d!(Fc, Fp)

Uniformly split a parent integrated D3Q19 population vector over its eight
ratio-2 children.
"""
function explode_uniform_F_3d!(Fc::AbstractArray{<:Any,4}, Fp::AbstractVector)
    _check_child_block_3d(Fc, "Fc")
    _check_d3q19_vector(Fp, "Fp")

    @inbounds for q in 1:19
        fq = Fp[q] / 8
        for iz in 1:2, iy in 1:2, ix in 1:2
            Fc[ix, iy, iz, q] = fq
        end
    end
    return Fc
end

"""
    mass_F_3d(F::AbstractVector)

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.mass_F_3d)
```
"""
function mass_F_3d(F::AbstractVector)
    _check_d3q19_vector(F, "F")
    return sum(F)
end

"""
    momentum_F_3d(F::AbstractVector)

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.momentum_F_3d)
```
"""
function momentum_F_3d(F::AbstractVector)
    _check_d3q19_vector(F, "F")

    mx = zero(F[1])
    my = zero(F[1])
    mz = zero(F[1])
    @inbounds for q in 1:19
        fq = F[q]
        mx += d3q19_cx(q) * fq
        my += d3q19_cy(q) * fq
        mz += d3q19_cz(q) * fq
    end
    return mx, my, mz
end

"""
    moments_F_3d(F::AbstractVector)

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.moments_F_3d)
```
"""
function moments_F_3d(F::AbstractVector)
    m = mass_F_3d(F)
    mx, my, mz = momentum_F_3d(F)
    return m, mx, my, mz
end

function mass_F_3d(F::AbstractArray{<:Any,4})
    size(F, 4) == 19 ||
        throw(ArgumentError("F must have 19 D3Q19 populations in dimension 4"))
    return sum(F)
end

function momentum_F_3d(F::AbstractArray{<:Any,4})
    size(F, 4) == 19 ||
        throw(ArgumentError("F must have 19 D3Q19 populations in dimension 4"))

    mx = zero(F[begin, begin, begin, 1])
    my = zero(F[begin, begin, begin, 1])
    mz = zero(F[begin, begin, begin, 1])
    @inbounds for q in 1:19
        cx = d3q19_cx(q)
        cy = d3q19_cy(q)
        cz = d3q19_cz(q)
        for iz in axes(F, 3), iy in axes(F, 2), ix in axes(F, 1)
            fq = F[ix, iy, iz, q]
            mx += cx * fq
            my += cy * fq
            mz += cz * fq
        end
    end
    return mx, my, mz
end

function moments_F_3d(F::AbstractArray{<:Any,4})
    m = mass_F_3d(F)
    mx, my, mz = momentum_F_3d(F)
    return m, mx, my, mz
end

"""
    fill_equilibrium_integrated_D3Q19!(Fcell::AbstractVector,

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.fill_equilibrium_integrated_D3Q19!)
```
"""
function fill_equilibrium_integrated_D3Q19!(Fcell::AbstractVector,
                                            volume,
                                            rho,
                                            ux,
                                            uy,
                                            uz)
    _check_d3q19_vector(Fcell, "Fcell")
    @inbounds for q in 1:19
        Fcell[q] = volume * equilibrium(D3Q19(), rho, ux, uy, uz, q)
    end
    return Fcell
end

function fill_equilibrium_integrated_D3Q19!(F::AbstractArray{<:Any,4},
                                            volume,
                                            rho,
                                            ux,
                                            uy,
                                            uz)
    size(F, 4) == 19 ||
        throw(ArgumentError("F must have 19 D3Q19 populations in dimension 4"))
    @inbounds for iz in axes(F, 3), iy in axes(F, 2), ix in axes(F, 1)
        rho_ijk = rho isa Function ? rho(ix, iy, iz) : rho
        ux_ijk = ux isa Function ? ux(ix, iy, iz) : ux
        uy_ijk = uy isa Function ? uy(ix, iy, iz) : uy
        uz_ijk = uz isa Function ? uz(ix, iy, iz) : uz
        fill_equilibrium_integrated_D3Q19!(
            @view(F[ix, iy, iz, :]), volume, rho_ijk, ux_ijk, uy_ijk, uz_ijk)
    end
    return F
end

"""
    ConservativeTreePatch3D

Experimental fixed ratio-2 patch for the conservative 3D tree route.

`fine_F` is the active fine state inside the refined region. `coarse_shadow_F`
is only a parent ledger/aggregate over the same physical region; it is not an
active fluid state.
"""
struct ConservativeTreePatch3D{T}
    parent_i_range::UnitRange{Int}
    parent_j_range::UnitRange{Int}
    parent_k_range::UnitRange{Int}
    ratio::Int
    fine_F::Array{T,4}
    coarse_shadow_F::Array{T,4}
end

"""
    create_conservative_tree_patch_3d(parent_i_range, parent_j_range, parent_k_range; ratio=2, T=Float64)

Allocate an experimental fixed ratio-2 conservative-tree 3D patch. No ghost
cells are allocated.
"""
function create_conservative_tree_patch_3d(parent_i_range::AbstractUnitRange{<:Integer},
                                           parent_j_range::AbstractUnitRange{<:Integer},
                                           parent_k_range::AbstractUnitRange{<:Integer};
                                           ratio::Int=2,
                                           T::Type{<:Real}=Float64)
    ratio == 2 || throw(ArgumentError("only ratio=2 is implemented"))
    isempty(parent_i_range) && throw(ArgumentError("parent_i_range must be nonempty"))
    isempty(parent_j_range) && throw(ArgumentError("parent_j_range must be nonempty"))
    isempty(parent_k_range) && throw(ArgumentError("parent_k_range must be nonempty"))

    ip = Int(first(parent_i_range)):Int(last(parent_i_range))
    jp = Int(first(parent_j_range)):Int(last(parent_j_range))
    kp = Int(first(parent_k_range)):Int(last(parent_k_range))
    nx_parent = length(ip)
    ny_parent = length(jp)
    nz_parent = length(kp)

    fine_F = zeros(T, ratio * nx_parent, ratio * ny_parent, ratio * nz_parent, 19)
    coarse_shadow_F = zeros(T, nx_parent, ny_parent, nz_parent, 19)
    return ConservativeTreePatch3D{T}(ip, jp, kp, ratio, fine_F, coarse_shadow_F)
end

function _check_conservative_tree_patch_layout(patch::ConservativeTreePatch3D)
    patch.ratio == 2 || throw(ArgumentError("only ratio=2 is implemented"))
    nx_parent = length(patch.parent_i_range)
    ny_parent = length(patch.parent_j_range)
    nz_parent = length(patch.parent_k_range)
    size(patch.fine_F) == (2 * nx_parent, 2 * ny_parent, 2 * nz_parent, 19) ||
        throw(ArgumentError("patch.fine_F has inconsistent size"))
    size(patch.coarse_shadow_F) == (nx_parent, ny_parent, nz_parent, 19) ||
        throw(ArgumentError("patch.coarse_shadow_F has inconsistent size"))
    return nothing
end

"""
    coalesce_patch_to_shadow_F_3d!(patch::ConservativeTreePatch3D)

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.coalesce_patch_to_shadow_F_3d!)
```
"""
function coalesce_patch_to_shadow_F_3d!(patch::ConservativeTreePatch3D)
    _check_conservative_tree_patch_layout(patch)

    nx_parent = length(patch.parent_i_range)
    ny_parent = length(patch.parent_j_range)
    nz_parent = length(patch.parent_k_range)
    @inbounds for kp in 1:nz_parent, jp in 1:ny_parent, ip in 1:nx_parent
        i0 = 2 * ip - 1
        j0 = 2 * jp - 1
        k0 = 2 * kp - 1
        Fp = @view patch.coarse_shadow_F[ip, jp, kp, :]
        Fc = @view patch.fine_F[i0:i0+1, j0:j0+1, k0:k0+1, :]
        coalesce_F_3d!(Fp, Fc)
    end
    return patch
end

"""
    explode_shadow_to_patch_uniform_F_3d!(patch::ConservativeTreePatch3D)

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.explode_shadow_to_patch_uniform_F_3d!)
```
"""
function explode_shadow_to_patch_uniform_F_3d!(patch::ConservativeTreePatch3D)
    _check_conservative_tree_patch_layout(patch)

    nx_parent = length(patch.parent_i_range)
    ny_parent = length(patch.parent_j_range)
    nz_parent = length(patch.parent_k_range)
    @inbounds for kp in 1:nz_parent, jp in 1:ny_parent, ip in 1:nx_parent
        i0 = 2 * ip - 1
        j0 = 2 * jp - 1
        k0 = 2 * kp - 1
        Fc = @view patch.fine_F[i0:i0+1, j0:j0+1, k0:k0+1, :]
        Fp = @view patch.coarse_shadow_F[ip, jp, kp, :]
        explode_uniform_F_3d!(Fc, Fp)
    end
    return patch
end

function _check_composite_coarse_layout_3d(coarse_F::AbstractArray{<:Any,4},
                                           patch::ConservativeTreePatch3D)
    _check_conservative_tree_patch_layout(patch)
    size(coarse_F, 4) == 19 ||
        throw(ArgumentError("coarse_F must have 19 D3Q19 populations in dimension 4"))
    first(patch.parent_i_range) >= first(axes(coarse_F, 1)) ||
        throw(ArgumentError("patch.parent_i_range starts outside coarse_F"))
    last(patch.parent_i_range) <= last(axes(coarse_F, 1)) ||
        throw(ArgumentError("patch.parent_i_range ends outside coarse_F"))
    first(patch.parent_j_range) >= first(axes(coarse_F, 2)) ||
        throw(ArgumentError("patch.parent_j_range starts outside coarse_F"))
    last(patch.parent_j_range) <= last(axes(coarse_F, 2)) ||
        throw(ArgumentError("patch.parent_j_range ends outside coarse_F"))
    first(patch.parent_k_range) >= first(axes(coarse_F, 3)) ||
        throw(ArgumentError("patch.parent_k_range starts outside coarse_F"))
    last(patch.parent_k_range) <= last(axes(coarse_F, 3)) ||
        throw(ArgumentError("patch.parent_k_range ends outside coarse_F"))
    return nothing
end

function _check_composite_pair_layout_3d(coarse_out::AbstractArray{<:Any,4},
                                         patch_out::ConservativeTreePatch3D,
                                         coarse_in::AbstractArray{<:Any,4},
                                         patch_in::ConservativeTreePatch3D)
    size(coarse_out) == size(coarse_in) ||
        throw(ArgumentError("coarse_out and coarse_in must have the same size"))
    patch_out.parent_i_range == patch_in.parent_i_range ||
        throw(ArgumentError("patch_out and patch_in must have the same parent_i_range"))
    patch_out.parent_j_range == patch_in.parent_j_range ||
        throw(ArgumentError("patch_out and patch_in must have the same parent_j_range"))
    patch_out.parent_k_range == patch_in.parent_k_range ||
        throw(ArgumentError("patch_out and patch_in must have the same parent_k_range"))
    size(patch_out.fine_F) == size(patch_in.fine_F) ||
        throw(ArgumentError("patch_out and patch_in fine arrays must have the same size"))
    _check_composite_coarse_layout_3d(coarse_in, patch_in)
    _check_composite_coarse_layout_3d(coarse_out, patch_out)
    return nothing
end

function composite_to_leaf_F_3d!(leaf_F::AbstractArray{<:Any,4},
                                 coarse_F::AbstractArray{<:Any,4},
                                 patch::ConservativeTreePatch3D)
    _check_composite_coarse_layout_3d(coarse_F, patch)
    size(leaf_F) == (2 * size(coarse_F, 1),
                     2 * size(coarse_F, 2),
                     2 * size(coarse_F, 3),
                     19) ||
        throw(ArgumentError("leaf_F must have size (2*Nx, 2*Ny, 2*Nz, 19)"))

    leaf_F .= 0
    @inbounds for k in axes(coarse_F, 3), j in axes(coarse_F, 2), i in axes(coarse_F, 1)
        _inside_range_3d(i, j, k,
                         patch.parent_i_range,
                         patch.parent_j_range,
                         patch.parent_k_range) && continue
        i0 = 2 * i - 1
        j0 = 2 * j - 1
        k0 = 2 * k - 1
        explode_uniform_F_3d!(
            @view(leaf_F[i0:i0+1, j0:j0+1, k0:k0+1, :]),
            @view(coarse_F[i, j, k, :]))
    end

    i0 = 2 * first(patch.parent_i_range) - 1
    j0 = 2 * first(patch.parent_j_range) - 1
    k0 = 2 * first(patch.parent_k_range) - 1
    leaf_F[i0:i0+size(patch.fine_F, 1)-1,
           j0:j0+size(patch.fine_F, 2)-1,
           k0:k0+size(patch.fine_F, 3)-1, :] .= patch.fine_F
    return leaf_F
end

"""
    active_mass_F_3d(coarse_F::AbstractArray{<:Any,4},

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.active_mass_F_3d)
```
"""
function active_mass_F_3d(coarse_F::AbstractArray{<:Any,4},
                          patch::ConservativeTreePatch3D)
    _check_composite_coarse_layout_3d(coarse_F, patch)

    total = zero(coarse_F[begin, begin, begin, 1] +
                 patch.fine_F[begin, begin, begin, 1])
    @inbounds for q in 1:19, k in axes(coarse_F, 3), j in axes(coarse_F, 2), i in axes(coarse_F, 1)
        _inside_range_3d(i, j, k,
                         patch.parent_i_range,
                         patch.parent_j_range,
                         patch.parent_k_range) && continue
        total += coarse_F[i, j, k, q]
    end
    return total + mass_F_3d(patch.fine_F)
end

"""
    active_population_sums_F_3d(coarse_F::AbstractArray{<:Any,4},

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.active_population_sums_F_3d)
```
"""
function active_population_sums_F_3d(coarse_F::AbstractArray{<:Any,4},
                                     patch::ConservativeTreePatch3D)
    _check_composite_coarse_layout_3d(coarse_F, patch)

    totals = zeros(promote_type(eltype(coarse_F), eltype(patch.fine_F)), 19)
    @inbounds for q in 1:19
        for k in axes(coarse_F, 3), j in axes(coarse_F, 2), i in axes(coarse_F, 1)
            _inside_range_3d(i, j, k,
                             patch.parent_i_range,
                             patch.parent_j_range,
                             patch.parent_k_range) && continue
            totals[q] += coarse_F[i, j, k, q]
        end
        totals[q] += sum(@view patch.fine_F[:, :, :, q])
    end
    return totals
end

"""
    active_momentum_F_3d(coarse_F::AbstractArray{<:Any,4},

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.active_momentum_F_3d)
```
"""
function active_momentum_F_3d(coarse_F::AbstractArray{<:Any,4},
                              patch::ConservativeTreePatch3D)
    _check_composite_coarse_layout_3d(coarse_F, patch)

    mx = zero(coarse_F[begin, begin, begin, 1] +
              patch.fine_F[begin, begin, begin, 1])
    my = zero(coarse_F[begin, begin, begin, 1] +
              patch.fine_F[begin, begin, begin, 1])
    mz = zero(coarse_F[begin, begin, begin, 1] +
              patch.fine_F[begin, begin, begin, 1])
    @inbounds for q in 1:19
        cx = d3q19_cx(q)
        cy = d3q19_cy(q)
        cz = d3q19_cz(q)
        for k in axes(coarse_F, 3), j in axes(coarse_F, 2), i in axes(coarse_F, 1)
            _inside_range_3d(i, j, k,
                             patch.parent_i_range,
                             patch.parent_j_range,
                             patch.parent_k_range) && continue
            fq = coarse_F[i, j, k, q]
            mx += cx * fq
            my += cy * fq
            mz += cz * fq
        end
    end
    fmx, fmy, fmz = momentum_F_3d(patch.fine_F)
    return mx + fmx, my + fmy, mz + fmz
end

"""
    active_moments_F_3d(coarse_F::AbstractArray{<:Any,4},

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.active_moments_F_3d)
```
"""
function active_moments_F_3d(coarse_F::AbstractArray{<:Any,4},
                             patch::ConservativeTreePatch3D)
    m = active_mass_F_3d(coarse_F, patch)
    mx, my, mz = active_momentum_F_3d(coarse_F, patch)
    return m, mx, my, mz
end

"""
    collide_BGK_integrated_D3Q19!(Fcell::AbstractVector, volume, omega)

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.collide_BGK_integrated_D3Q19!)
```
"""
