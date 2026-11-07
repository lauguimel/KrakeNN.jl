# Conservative cell-centered tree refinement prototype for D2Q9.
#
# This file intentionally stays separate from the existing RefinementPatch
# implementation. The state manipulated here is the integrated population
# F_i = f_i * cell_volume, not the density population f_i.

const D2Q9_CX_INT = (0, 1, 0, -1, 0, 1, -1, -1, 1)
const D2Q9_CY_INT = (0, 0, 1, 0, -1, 1, 1, -1, -1)
const D2Q9_OPPOSITE_INT = (1, 4, 5, 2, 3, 8, 9, 6, 7)

@inline function _check_d2q9_q(q::Int)
    1 <= q <= 9 || throw(ArgumentError("D2Q9 population index q must be in 1:9"))
    return q
end

"""
    d2q9_cx(q::Integer)

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.d2q9_cx)
```
"""
@inline function d2q9_cx(q::Integer)
    qi = _check_d2q9_q(Int(q))
    return D2Q9_CX_INT[qi]
end

"""
    d2q9_cy(q::Integer)

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.d2q9_cy)
```
"""
@inline function d2q9_cy(q::Integer)
    qi = _check_d2q9_q(Int(q))
    return D2Q9_CY_INT[qi]
end

"""
    d2q9_opposite(q::Integer)

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.d2q9_opposite)
```
"""
@inline function d2q9_opposite(q::Integer)
    qi = _check_d2q9_q(Int(q))
    return D2Q9_OPPOSITE_INT[qi]
end

@inline function _check_d2q9_vector(F::AbstractVector, name::AbstractString)
    length(F) == 9 || throw(ArgumentError("$name must have length 9 for D2Q9"))
    return nothing
end

@inline function _check_child_block_2d(Fc::AbstractArray{<:Any,3}, name::AbstractString)
    size(Fc) == (2, 2, 9) ||
        throw(ArgumentError("$name must have size (2, 2, 9)"))
    return nothing
end

"""
    coalesce_F_2d!(Fp, Fc)

Coalesce the four ratio-2 child cells `Fc[ix, iy, q]` into one parent
population vector `Fp[q]`.

Both `Fp` and `Fc` store integrated populations `F_i = f_i * volume`.
The operation preserves each oriented population exactly up to floating-point
roundoff.
"""
function coalesce_F_2d!(Fp::AbstractVector, Fc::AbstractArray{<:Any,3})
    _check_d2q9_vector(Fp, "Fp")
    _check_child_block_2d(Fc, "Fc")

    @inbounds for q in 1:9
        Fp[q] = Fc[1, 1, q] + Fc[2, 1, q] + Fc[1, 2, q] + Fc[2, 2, q]
    end
    return Fp
end

"""
    explode_uniform_F_2d!(Fc, Fp)

Uniformly split a parent integrated population vector over its four ratio-2
children. This is a conservative low-order explosion operator.
"""
function explode_uniform_F_2d!(Fc::AbstractArray{<:Any,3}, Fp::AbstractVector)
    _check_child_block_2d(Fc, "Fc")
    _check_d2q9_vector(Fp, "Fp")

    @inbounds for q in 1:9
        fq = Fp[q] / 4
        Fc[1, 1, q] = fq
        Fc[2, 1, q] = fq
        Fc[1, 2, q] = fq
        Fc[2, 2, q] = fq
    end
    return Fc
end

"""
    mass_F(F)

Mass carried by a D2Q9 integrated population vector.
"""
function mass_F(F::AbstractVector)
    _check_d2q9_vector(F, "F")
    return sum(F)
end

"""
    momentum_F(F)

Momentum carried by a D2Q9 integrated population vector.
"""
function momentum_F(F::AbstractVector)
    _check_d2q9_vector(F, "F")

    mx = zero(F[1])
    my = zero(F[1])
    @inbounds for q in 1:9
        mx += d2q9_cx(q) * F[q]
        my += d2q9_cy(q) * F[q]
    end
    return mx, my
end

"""
    moments_F(F)

Return `(mass, momentum_x, momentum_y)` for a D2Q9 integrated population vector.
"""
function moments_F(F::AbstractVector)
    m = mass_F(F)
    mx, my = momentum_F(F)
    return m, mx, my
end

function mass_F(F::AbstractArray{<:Any,3})
    size(F, 3) == 9 || throw(ArgumentError("F must have 9 D2Q9 populations in dimension 3"))
    return sum(F)
end

function momentum_F(F::AbstractArray{<:Any,3})
    size(F, 3) == 9 || throw(ArgumentError("F must have 9 D2Q9 populations in dimension 3"))

    mx = zero(F[begin, begin, 1])
    my = zero(F[begin, begin, 1])
    @inbounds for q in 1:9
        cx = d2q9_cx(q)
        cy = d2q9_cy(q)
        for j in axes(F, 2), i in axes(F, 1)
            fq = F[i, j, q]
            mx += cx * fq
            my += cy * fq
        end
    end
    return mx, my
end

function moments_F(F::AbstractArray{<:Any,3})
    m = mass_F(F)
    mx, my = momentum_F(F)
    return m, mx, my
end

"""
    fill_equilibrium_integrated_D2Q9!(Fcell::AbstractVector,

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.fill_equilibrium_integrated_D2Q9!)
```
"""
function fill_equilibrium_integrated_D2Q9!(Fcell::AbstractVector,
                                           volume,
                                           rho,
                                           ux,
                                           uy)
    _check_d2q9_vector(Fcell, "Fcell")
    @inbounds for q in 1:9
        Fcell[q] = volume * equilibrium(D2Q9(), rho, ux, uy, q)
    end
    return Fcell
end

function fill_equilibrium_integrated_D2Q9!(F::AbstractArray{<:Any,3},
                                           volume,
                                           rho,
                                           ux,
                                           uy)
    size(F, 3) == 9 || throw(ArgumentError("F must have 9 D2Q9 populations in dimension 3"))
    @inbounds for j in axes(F, 2), i in axes(F, 1)
        rho_ij = rho isa Function ? rho(i, j) : rho
        ux_ij = ux isa Function ? ux(i, j) : ux
        uy_ij = uy isa Function ? uy(i, j) : uy
        fill_equilibrium_integrated_D2Q9!(@view(F[i, j, :]), volume, rho_ij, ux_ij, uy_ij)
    end
    return F
end

"""
    conservative_tree_parent_index(i_f::Int, j_f::Int)

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.conservative_tree_parent_index)
```
"""
@inline function conservative_tree_parent_index(i_f::Int, j_f::Int)
    i_f >= 1 || throw(ArgumentError("i_f must be >= 1"))
    j_f >= 1 || throw(ArgumentError("j_f must be >= 1"))

    i_parent = (i_f + 1) >>> 1
    j_parent = (j_f + 1) >>> 1
    i_child = isodd(i_f) ? 1 : 2
    j_child = isodd(j_f) ? 1 : 2
    return i_parent, j_parent, i_child, j_child
end

"""
    split_coarse_to_fine_vertical_F_2d!(Fc_dest, Fq, q)

Accumulate one integrated packet crossing a vertical coarse/fine interface into
the two fine children adjacent to the entry face.

For east-moving populations the entry face is the west side of the refined
parent, so children `(1, 1)` and `(1, 2)` receive `Fq / 2`. For west-moving
populations the entry face is the east side, so children `(2, 1)` and `(2, 2)`
receive `Fq / 2`.
"""
function split_coarse_to_fine_vertical_F_2d!(Fc_dest::AbstractArray{<:Any,3},
                                             Fq,
                                             q::Int)
    face = d2q9_cx(q) > 0 ? :west : :east
    return split_coarse_to_fine_face_F_2d!(Fc_dest, Fq, q, face)
end

"""
    coalesce_fine_to_coarse_vertical_F(Fc_src, q)

Return the integrated packet leaving a refined parent through a vertical
interface. The function sums the two interface-adjacent fine children for the
oriented population `q`.
"""
function coalesce_fine_to_coarse_vertical_F(Fc_src::AbstractArray{<:Any,3},
                                            q::Int)
    face = d2q9_cx(q) < 0 ? :west : :east
    return coalesce_fine_to_coarse_face_F(Fc_src, q, face)
end

@inline function _check_conservative_tree_face(face::Symbol)
    face in (:west, :east, :south, :north) ||
        throw(ArgumentError("face must be one of :west, :east, :south, :north"))
    return face
end

@inline function _face_normal(face::Symbol)
    _check_conservative_tree_face(face)
    if face == :west
        return -1, 0
    elseif face == :east
        return 1, 0
    elseif face == :south
        return 0, -1
    else
        return 0, 1
    end
end

@inline function _enters_patch_through_face(q::Int, face::Symbol)
    nx, ny = _face_normal(face)
    return d2q9_cx(q) * nx + d2q9_cy(q) * ny < 0
end

@inline function _leaves_patch_through_face(q::Int, face::Symbol)
    nx, ny = _face_normal(face)
    return d2q9_cx(q) * nx + d2q9_cy(q) * ny > 0
end

"""
    split_coarse_to_fine_face_F_2d!(Fc_dest, Fq, q, face)

Accumulate an integrated packet entering a refined parent through one face.

`face` names the boundary of the refined parent (`:west`, `:east`, `:south`,
`:north`). The operator is conservative and low order: it splits the packet
uniformly over the two child cells adjacent to the entry face. Diagonal corner
routing is intentionally left to the caller; this function is a face-interface
building block.
"""
function split_coarse_to_fine_face_F_2d!(Fc_dest::AbstractArray{<:Any,3},
                                         Fq,
                                         q::Int,
                                         face::Symbol)
    _check_child_block_2d(Fc_dest, "Fc_dest")
    qi = _check_d2q9_q(q)
    _check_conservative_tree_face(face)
    _enters_patch_through_face(qi, face) ||
        throw(ArgumentError("population q=$qi does not enter through face $face"))

    half = Fq / 2
    @inbounds begin
        if face == :west
            Fc_dest[1, 1, qi] += half
            Fc_dest[1, 2, qi] += half
        elseif face == :east
            Fc_dest[2, 1, qi] += half
            Fc_dest[2, 2, qi] += half
        elseif face == :south
            Fc_dest[1, 1, qi] += half
            Fc_dest[2, 1, qi] += half
        else
            Fc_dest[1, 2, qi] += half
            Fc_dest[2, 2, qi] += half
        end
    end
    return Fc_dest
end

"""
    coalesce_fine_to_coarse_face_F(Fc_src, q, face)

Return the integrated packet leaving a refined parent through one face by
summing the two child cells adjacent to that exit face.
"""
function coalesce_fine_to_coarse_face_F(Fc_src::AbstractArray{<:Any,3},
                                        q::Int,
                                        face::Symbol)
    _check_child_block_2d(Fc_src, "Fc_src")
    qi = _check_d2q9_q(q)
    _check_conservative_tree_face(face)
    _leaves_patch_through_face(qi, face) ||
        throw(ArgumentError("population q=$qi does not leave through face $face"))

    @inbounds begin
        if face == :west
            return Fc_src[1, 1, qi] + Fc_src[1, 2, qi]
        elseif face == :east
            return Fc_src[2, 1, qi] + Fc_src[2, 2, qi]
        elseif face == :south
            return Fc_src[1, 1, qi] + Fc_src[2, 1, qi]
        else
            return Fc_src[1, 2, qi] + Fc_src[2, 2, qi]
        end
    end
end

@inline function _check_conservative_tree_corner(corner::Symbol)
    corner in (:southwest, :southeast, :northwest, :northeast) ||
        throw(ArgumentError("corner must be one of :southwest, :southeast, :northwest, :northeast"))
    return corner
end

@inline function _entry_corner_for_q(q::Int)
    if q == 6
        return :southwest
    elseif q == 7
        return :southeast
    elseif q == 8
        return :northeast
    elseif q == 9
        return :northwest
    else
        throw(ArgumentError("only diagonal populations q=6:9 enter through corners"))
    end
end

@inline function _exit_corner_for_q(q::Int)
    if q == 6
        return :northeast
    elseif q == 7
        return :northwest
    elseif q == 8
        return :southwest
    elseif q == 9
        return :southeast
    else
        throw(ArgumentError("only diagonal populations q=6:9 leave through corners"))
    end
end

@inline function _corner_child_index(corner::Symbol)
    _check_conservative_tree_corner(corner)
    if corner == :southwest
        return 1, 1
    elseif corner == :southeast
        return 2, 1
    elseif corner == :northwest
        return 1, 2
    else
        return 2, 2
    end
end

@inline function _inside_range(i::Int, j::Int,
                              irange::UnitRange{Int},
                              jrange::UnitRange{Int})
    return first(irange) <= i <= last(irange) &&
           first(jrange) <= j <= last(jrange)
end

@inline function _inside_array_2d(A::AbstractArray, i::Int, j::Int)
    return first(axes(A, 1)) <= i <= last(axes(A, 1)) &&
           first(axes(A, 2)) <= j <= last(axes(A, 2))
end

@inline function _patch_local_parent_index(patch, i_parent::Int, j_parent::Int)
    return i_parent - first(patch.parent_i_range) + 1,
           j_parent - first(patch.parent_j_range) + 1
end

@inline function _child_block_view(Ffine, ip_local::Int, jp_local::Int)
    i0 = 2 * ip_local - 1
    j0 = 2 * jp_local - 1
    return @view Ffine[i0:i0+1, j0:j0+1, :]
end

@inline function _minmod(a, b)
    if a * b <= 0
        return zero(a + b)
    end
    return abs(a) < abs(b) ? a : b
end

@inline function _interface_face_from_offset(di::Int, dj::Int)
    if di < 0 && dj == 0
        return :west
    elseif di > 0 && dj == 0
        return :east
    elseif di == 0 && dj < 0
        return :south
    elseif di == 0 && dj > 0
        return :north
    else
        throw(ArgumentError("offset does not identify a single face"))
    end
end

@inline function _interface_corner_from_offset(di::Int, dj::Int)
    if di < 0 && dj < 0
        return :southwest
    elseif di > 0 && dj < 0
        return :southeast
    elseif di < 0 && dj > 0
        return :northwest
    elseif di > 0 && dj > 0
        return :northeast
    else
        throw(ArgumentError("offset does not identify a corner"))
    end
end

"""
    split_coarse_to_fine_corner_F_2d!(Fc_dest, Fq, q, corner)

Accumulate one diagonal integrated packet entering a refined parent through a
corner. The whole packet goes to the child cell adjacent to that corner.
"""
function split_coarse_to_fine_corner_F_2d!(Fc_dest::AbstractArray{<:Any,3},
                                           Fq,
                                           q::Int,
                                           corner::Symbol)
    _check_child_block_2d(Fc_dest, "Fc_dest")
    qi = _check_d2q9_q(q)
    _check_conservative_tree_corner(corner)
    _entry_corner_for_q(qi) == corner ||
        throw(ArgumentError("population q=$qi does not enter through corner $corner"))

    ix, iy = _corner_child_index(corner)
    @inbounds Fc_dest[ix, iy, qi] += Fq
    return Fc_dest
end

"""
    coalesce_fine_to_coarse_corner_F(Fc_src, q, corner)

Return the diagonal integrated packet leaving a refined parent through a
corner. The packet is read from the child cell adjacent to that corner.
"""
function coalesce_fine_to_coarse_corner_F(Fc_src::AbstractArray{<:Any,3},
                                          q::Int,
                                          corner::Symbol)
    _check_child_block_2d(Fc_src, "Fc_src")
    qi = _check_d2q9_q(q)
    _check_conservative_tree_corner(corner)
    _exit_corner_for_q(qi) == corner ||
        throw(ArgumentError("population q=$qi does not leave through corner $corner"))

    ix, iy = _corner_child_index(corner)
    @inbounds return Fc_src[ix, iy, qi]
end

function _check_patch_boundary_transfer_layout(Ffine::AbstractArray{<:Any,3},
                                               Fcoarse::AbstractArray{<:Any,3},
                                               patch)
    _check_conservative_tree_patch_layout(patch)
    size(Ffine) == size(patch.fine_F) ||
        throw(ArgumentError("Ffine must have the same size as patch.fine_F"))
    size(Fcoarse, 3) == 9 ||
        throw(ArgumentError("Fcoarse must have 9 D2Q9 populations in dimension 3"))
    return nothing
end

"""
    coarse_to_fine_patch_boundary_F_2d!(fine_dest, coarse_src, patch)

Accumulate all coarse populations that enter the refined patch during one
coarse-grid packet transfer. Only sources outside `patch.parent_*_range` and
inside `coarse_src` are considered. The destination is the fine child block
corresponding to the inactive parent ledger cell.
"""
function coarse_to_fine_patch_boundary_F_2d!(fine_dest::AbstractArray{<:Any,3},
                                             coarse_src::AbstractArray{<:Any,3},
                                             patch)
    _check_patch_boundary_transfer_layout(fine_dest, coarse_src, patch)

    @inbounds for jp_parent in patch.parent_j_range, ip_parent in patch.parent_i_range
        ip_local, jp_local = _patch_local_parent_index(patch, ip_parent, jp_parent)
        Fc_dest = _child_block_view(fine_dest, ip_local, jp_local)
        for q in 1:9
            cx = d2q9_cx(q)
            cy = d2q9_cy(q)
            cx == 0 && cy == 0 && continue

            isrc = ip_parent - cx
            jsrc = jp_parent - cy
            _inside_array_2d(coarse_src, isrc, jsrc) || continue
            _inside_range(isrc, jsrc, patch.parent_i_range, patch.parent_j_range) && continue

            di = isrc < first(patch.parent_i_range) ? -1 :
                 isrc > last(patch.parent_i_range) ? 1 : 0
            dj = jsrc < first(patch.parent_j_range) ? -1 :
                 jsrc > last(patch.parent_j_range) ? 1 : 0

            packet = coarse_src[isrc, jsrc, q]
            if di != 0 && dj != 0
                split_coarse_to_fine_corner_F_2d!(Fc_dest, packet, q,
                                                  _interface_corner_from_offset(di, dj))
            else
                split_coarse_to_fine_face_F_2d!(Fc_dest, packet, q,
                                                _interface_face_from_offset(di, dj))
            end
        end
    end
    return fine_dest
end

"""
    fine_to_coarse_patch_boundary_F_2d!(coarse_dest, fine_src, patch)

Accumulate all fine populations that leave the refined patch during a boundary
packet transfer. Only destinations outside `patch.parent_*_range` and inside
`coarse_dest` are considered.
"""
function fine_to_coarse_patch_boundary_F_2d!(coarse_dest::AbstractArray{<:Any,3},
                                             fine_src::AbstractArray{<:Any,3},
                                             patch)
    _check_patch_boundary_transfer_layout(fine_src, coarse_dest, patch)

    @inbounds for jp_parent in patch.parent_j_range, ip_parent in patch.parent_i_range
        ip_local, jp_local = _patch_local_parent_index(patch, ip_parent, jp_parent)
        Fc_src = _child_block_view(fine_src, ip_local, jp_local)
        for q in 1:9
            cx = d2q9_cx(q)
            cy = d2q9_cy(q)
            cx == 0 && cy == 0 && continue

            idst = ip_parent + cx
            jdst = jp_parent + cy
            _inside_array_2d(coarse_dest, idst, jdst) || continue
            _inside_range(idst, jdst, patch.parent_i_range, patch.parent_j_range) && continue

            di = idst < first(patch.parent_i_range) ? -1 :
                 idst > last(patch.parent_i_range) ? 1 : 0
            dj = jdst < first(patch.parent_j_range) ? -1 :
                 jdst > last(patch.parent_j_range) ? 1 : 0

            if di != 0 && dj != 0
                coarse_dest[idst, jdst, q] += coalesce_fine_to_coarse_corner_F(
                    Fc_src, q, _interface_corner_from_offset(di, dj))
            else
                coarse_dest[idst, jdst, q] += coalesce_fine_to_coarse_face_F(
                    Fc_src, q, _interface_face_from_offset(di, dj))
            end
        end
    end
    return coarse_dest
end

"""
    collide_BGK_integrated_D2Q9!(Fcell, volume, omega)

Apply a local BGK collision to one D2Q9 cell stored as integrated populations.
The conversion is `f_i = F_i / volume`; after collision the populations are
stored again as `F_i`.
"""
