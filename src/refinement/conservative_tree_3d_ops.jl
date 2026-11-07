function collide_BGK_integrated_D3Q19!(Fcell::AbstractVector, volume, omega)
    _check_d3q19_vector(Fcell, "Fcell")
    volume > zero(volume) || throw(ArgumentError("volume must be positive"))

    m = mass_F_3d(Fcell)
    iszero(m) && throw(ArgumentError("Fcell mass must be nonzero"))
    mx, my, mz = momentum_F_3d(Fcell)
    rho = m / volume
    ux = mx / m
    uy = my / m
    uz = mz / m

    @inbounds for q in 1:19
        f = Fcell[q] / volume
        feq = equilibrium(D3Q19(), rho, ux, uy, uz, q)
        Fcell[q] = (f - omega * (f - feq)) * volume
    end
    return Fcell
end

function collide_BGK_integrated_D3Q19!(F::AbstractArray{<:Any,4}, volume, omega)
    size(F, 4) == 19 ||
        throw(ArgumentError("F must have 19 D3Q19 populations in dimension 4"))
    @inbounds for k in axes(F, 3), j in axes(F, 2), i in axes(F, 1)
        collide_BGK_integrated_D3Q19!(@view(F[i, j, k, :]), volume, omega)
    end
    return F
end

"""
    collide_Guo_integrated_D3Q19!(Fcell::AbstractVector,

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.collide_Guo_integrated_D3Q19!)
```
"""
function collide_Guo_integrated_D3Q19!(Fcell::AbstractVector,
                                       volume,
                                       omega,
                                       Fx,
                                       Fy,
                                       Fz)
    _check_d3q19_vector(Fcell, "Fcell")
    volume > zero(volume) || throw(ArgumentError("volume must be positive"))

    m = mass_F_3d(Fcell)
    iszero(m) && throw(ArgumentError("Fcell mass must be nonzero"))
    mx, my, mz = momentum_F_3d(Fcell)
    rho = m / volume
    ux = (mx / volume + Fx / 2) / rho
    uy = (my / volume + Fy / 2) / rho
    uz = (mz / volume + Fz / 2) / rho
    guo_pref = 1 - omega / 2

    @inbounds for q in 1:19
        cx = d3q19_cx(q)
        cy = d3q19_cy(q)
        cz = d3q19_cz(q)
        w = weights(D3Q19())[q]
        ci_dot_u = cx * ux + cy * uy + cz * uz
        ci_dot_F = cx * Fx + cy * Fy + cz * Fz
        Sq = w * (3 * ((cx - ux) * Fx + (cy - uy) * Fy + (cz - uz) * Fz) +
                  9 * ci_dot_u * ci_dot_F)
        f = Fcell[q] / volume
        feq = equilibrium(D3Q19(), rho, ux, uy, uz, q)
        Fcell[q] = volume * (f - omega * (f - feq) + guo_pref * Sq)
    end
    return Fcell
end

function collide_Guo_integrated_D3Q19!(F::AbstractArray{<:Any,4},
                                       volume,
                                       omega,
                                       Fx,
                                       Fy,
                                       Fz)
    size(F, 4) == 19 ||
        throw(ArgumentError("F must have 19 D3Q19 populations in dimension 4"))
    @inbounds for k in axes(F, 3), j in axes(F, 2), i in axes(F, 1)
        collide_Guo_integrated_D3Q19!(@view(F[i, j, k, :]),
                                      volume, omega, Fx, Fy, Fz)
    end
    return F
end

"""
    collide_BGK_composite_F_3d!(coarse_F::AbstractArray{<:Any,4},

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.collide_BGK_composite_F_3d!)
```
"""
function collide_BGK_composite_F_3d!(coarse_F::AbstractArray{<:Any,4},
                                     patch::ConservativeTreePatch3D,
                                     volume_coarse,
                                     volume_fine,
                                     omega_coarse,
                                     omega_fine)
    _check_composite_coarse_layout_3d(coarse_F, patch)

    @inbounds for k in axes(coarse_F, 3), j in axes(coarse_F, 2), i in axes(coarse_F, 1)
        _inside_range_3d(i, j, k,
                         patch.parent_i_range,
                         patch.parent_j_range,
                         patch.parent_k_range) && continue
        collide_BGK_integrated_D3Q19!(
            @view(coarse_F[i, j, k, :]), volume_coarse, omega_coarse)
    end
    collide_BGK_integrated_D3Q19!(patch.fine_F, volume_fine, omega_fine)
    coalesce_patch_to_shadow_F_3d!(patch)
    return coarse_F, patch
end

"""
    collide_Guo_composite_F_3d!(coarse_F::AbstractArray{<:Any,4},

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.collide_Guo_composite_F_3d!)
```
"""
function collide_Guo_composite_F_3d!(coarse_F::AbstractArray{<:Any,4},
                                     patch::ConservativeTreePatch3D,
                                     volume_coarse,
                                     volume_fine,
                                     omega_coarse,
                                     omega_fine,
                                     Fx,
                                     Fy,
                                     Fz)
    _check_composite_coarse_layout_3d(coarse_F, patch)

    @inbounds for k in axes(coarse_F, 3), j in axes(coarse_F, 2), i in axes(coarse_F, 1)
        _inside_range_3d(i, j, k,
                         patch.parent_i_range,
                         patch.parent_j_range,
                         patch.parent_k_range) && continue
        collide_Guo_integrated_D3Q19!(
            @view(coarse_F[i, j, k, :]), volume_coarse,
            omega_coarse, Fx, Fy, Fz)
    end
    collide_Guo_integrated_D3Q19!(patch.fine_F, volume_fine, omega_fine, Fx, Fy, Fz)
    coalesce_patch_to_shadow_F_3d!(patch)
    return coarse_F, patch
end

"""
    conservative_tree_parent_index_3d(i_f::Int, j_f::Int, k_f::Int)

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.conservative_tree_parent_index_3d)
```
"""
@inline function conservative_tree_parent_index_3d(i_f::Int, j_f::Int, k_f::Int)
    i_f >= 1 || throw(ArgumentError("i_f must be >= 1"))
    j_f >= 1 || throw(ArgumentError("j_f must be >= 1"))
    k_f >= 1 || throw(ArgumentError("k_f must be >= 1"))

    i_parent = (i_f + 1) >>> 1
    j_parent = (j_f + 1) >>> 1
    k_parent = (k_f + 1) >>> 1
    i_child = isodd(i_f) ? 1 : 2
    j_child = isodd(j_f) ? 1 : 2
    k_child = isodd(k_f) ? 1 : 2
    return i_parent, j_parent, k_parent, i_child, j_child, k_child
end

@inline function _check_conservative_tree_face_3d(face::Symbol)
    face in (:west, :east, :south, :north, :bottom, :top) ||
        throw(ArgumentError("face must be one of :west, :east, :south, :north, :bottom, :top"))
    return face
end

@inline function _face_normal_3d(face::Symbol)
    _check_conservative_tree_face_3d(face)
    if face == :west
        return -1, 0, 0
    elseif face == :east
        return 1, 0, 0
    elseif face == :south
        return 0, -1, 0
    elseif face == :north
        return 0, 1, 0
    elseif face == :bottom
        return 0, 0, -1
    else
        return 0, 0, 1
    end
end

@inline function _enters_patch_through_face_3d(q::Int, face::Symbol)
    nx, ny, nz = _face_normal_3d(face)
    return d3q19_cx(q) * nx + d3q19_cy(q) * ny + d3q19_cz(q) * nz < 0
end

@inline function _leaves_patch_through_face_3d(q::Int, face::Symbol)
    nx, ny, nz = _face_normal_3d(face)
    return d3q19_cx(q) * nx + d3q19_cy(q) * ny + d3q19_cz(q) * nz > 0
end

@inline function _face_child_indices_3d(face::Symbol)
    _check_conservative_tree_face_3d(face)
    if face == :west
        return ((1, 1, 1), (1, 2, 1), (1, 1, 2), (1, 2, 2))
    elseif face == :east
        return ((2, 1, 1), (2, 2, 1), (2, 1, 2), (2, 2, 2))
    elseif face == :south
        return ((1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2))
    elseif face == :north
        return ((1, 2, 1), (2, 2, 1), (1, 2, 2), (2, 2, 2))
    elseif face == :bottom
        return ((1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1))
    else
        return ((1, 1, 2), (2, 1, 2), (1, 2, 2), (2, 2, 2))
    end
end

"""
    split_coarse_to_fine_face_F_3d!(Fc_dest, Fq, q, face)

Accumulate an integrated packet entering a refined parent through one face.
The packet is split uniformly over the four child cells adjacent to that face.
"""
function split_coarse_to_fine_face_F_3d!(Fc_dest::AbstractArray{<:Any,4},
                                         Fq,
                                         q::Int,
                                         face::Symbol)
    _check_child_block_3d(Fc_dest, "Fc_dest")
    qi = _check_d3q19_q(q)
    _check_conservative_tree_face_3d(face)
    _enters_patch_through_face_3d(qi, face) ||
        throw(ArgumentError("population q=$qi does not enter through face $face"))

    share = Fq / 4
    @inbounds for (ix, iy, iz) in _face_child_indices_3d(face)
        Fc_dest[ix, iy, iz, qi] += share
    end
    return Fc_dest
end

"""
    coalesce_fine_to_coarse_face_F_3d(Fc_src, q, face)

Return the integrated packet leaving a refined parent through one face by
summing the four child cells adjacent to that exit face.
"""
function coalesce_fine_to_coarse_face_F_3d(Fc_src::AbstractArray{<:Any,4},
                                          q::Int,
                                          face::Symbol)
    _check_child_block_3d(Fc_src, "Fc_src")
    qi = _check_d3q19_q(q)
    _check_conservative_tree_face_3d(face)
    _leaves_patch_through_face_3d(qi, face) ||
        throw(ArgumentError("population q=$qi does not leave through face $face"))

    packet = zero(Fc_src[1, 1, 1, qi])
    @inbounds for (ix, iy, iz) in _face_child_indices_3d(face)
        packet += Fc_src[ix, iy, iz, qi]
    end
    return packet
end

@inline function _check_conservative_tree_edge_3d(edge::Symbol)
    edge in (:southwest, :southeast, :northwest, :northeast,
             :bottomwest, :bottomeast, :topwest, :topeast,
             :bottomsouth, :bottomnorth, :topsouth, :topnorth) ||
        throw(ArgumentError("edge must name one of the 12 3D parent edges"))
    return edge
end

@inline function _edge_faces_3d(edge::Symbol)
    _check_conservative_tree_edge_3d(edge)
    if edge == :southwest
        return :south, :west
    elseif edge == :southeast
        return :south, :east
    elseif edge == :northwest
        return :north, :west
    elseif edge == :northeast
        return :north, :east
    elseif edge == :bottomwest
        return :bottom, :west
    elseif edge == :bottomeast
        return :bottom, :east
    elseif edge == :topwest
        return :top, :west
    elseif edge == :topeast
        return :top, :east
    elseif edge == :bottomsouth
        return :bottom, :south
    elseif edge == :bottomnorth
        return :bottom, :north
    elseif edge == :topsouth
        return :top, :south
    else
        return :top, :north
    end
end

@inline function _enters_patch_through_edge_3d(q::Int, edge::Symbol)
    f1, f2 = _edge_faces_3d(edge)
    return _enters_patch_through_face_3d(q, f1) &&
           _enters_patch_through_face_3d(q, f2)
end

@inline function _leaves_patch_through_edge_3d(q::Int, edge::Symbol)
    f1, f2 = _edge_faces_3d(edge)
    return _leaves_patch_through_face_3d(q, f1) &&
           _leaves_patch_through_face_3d(q, f2)
end

@inline function _edge_child_indices_3d(edge::Symbol)
    _check_conservative_tree_edge_3d(edge)
    if edge == :southwest
        return ((1, 1, 1), (1, 1, 2))
    elseif edge == :southeast
        return ((2, 1, 1), (2, 1, 2))
    elseif edge == :northwest
        return ((1, 2, 1), (1, 2, 2))
    elseif edge == :northeast
        return ((2, 2, 1), (2, 2, 2))
    elseif edge == :bottomwest
        return ((1, 1, 1), (1, 2, 1))
    elseif edge == :bottomeast
        return ((2, 1, 1), (2, 2, 1))
    elseif edge == :topwest
        return ((1, 1, 2), (1, 2, 2))
    elseif edge == :topeast
        return ((2, 1, 2), (2, 2, 2))
    elseif edge == :bottomsouth
        return ((1, 1, 1), (2, 1, 1))
    elseif edge == :bottomnorth
        return ((1, 2, 1), (2, 2, 1))
    elseif edge == :topsouth
        return ((1, 1, 2), (2, 1, 2))
    else
        return ((1, 2, 2), (2, 2, 2))
    end
end

"""
    split_coarse_to_fine_edge_F_3d!(Fc_dest, Fq, q, edge)

Accumulate one D3Q19 edge-aligned packet entering a refined parent through a
parent edge. The packet is split uniformly over the two child cells adjacent
to that edge.
"""
function split_coarse_to_fine_edge_F_3d!(Fc_dest::AbstractArray{<:Any,4},
                                         Fq,
                                         q::Int,
                                         edge::Symbol)
    _check_child_block_3d(Fc_dest, "Fc_dest")
    qi = _check_d3q19_q(q)
    _check_conservative_tree_edge_3d(edge)
    _enters_patch_through_edge_3d(qi, edge) ||
        throw(ArgumentError("population q=$qi does not enter through edge $edge"))

    share = Fq / 2
    @inbounds for (ix, iy, iz) in _edge_child_indices_3d(edge)
        Fc_dest[ix, iy, iz, qi] += share
    end
    return Fc_dest
end

"""
    coalesce_fine_to_coarse_edge_F_3d(Fc_src, q, edge)

Return the integrated packet leaving a refined parent through one parent edge
by summing the two child cells adjacent to that edge.
"""
function coalesce_fine_to_coarse_edge_F_3d(Fc_src::AbstractArray{<:Any,4},
                                          q::Int,
                                          edge::Symbol)
    _check_child_block_3d(Fc_src, "Fc_src")
    qi = _check_d3q19_q(q)
    _check_conservative_tree_edge_3d(edge)
    _leaves_patch_through_edge_3d(qi, edge) ||
        throw(ArgumentError("population q=$qi does not leave through edge $edge"))

    packet = zero(Fc_src[1, 1, 1, qi])
    @inbounds for (ix, iy, iz) in _edge_child_indices_3d(edge)
        packet += Fc_src[ix, iy, iz, qi]
    end
    return packet
end

@inline function _check_conservative_tree_corner_3d(corner::Symbol)
    corner in (:bottomsouthwest, :bottomsoutheast, :bottomnorthwest, :bottomnortheast,
               :topsouthwest, :topsoutheast, :topnorthwest, :topnortheast) ||
        throw(ArgumentError("corner must name one of the 8 3D parent corners"))
    return corner
end

"""
    split_coarse_to_fine_corner_F_3d!(Fc_dest, Fq, q, corner)

D3Q19 has no body-diagonal populations, so the conservative corner route set
is empty. This function exists as an explicit canary for that topology fact.
"""
function split_coarse_to_fine_corner_F_3d!(Fc_dest::AbstractArray{<:Any,4},
                                           Fq,
                                           q::Int,
                                           corner::Symbol)
    _check_child_block_3d(Fc_dest, "Fc_dest")
    qi = _check_d3q19_q(q)
    _check_conservative_tree_corner_3d(corner)
    throw(ArgumentError("population q=$qi cannot enter through corner $corner with D3Q19"))
end

"""
    coalesce_fine_to_coarse_corner_F_3d(Fc_src, q, corner)

D3Q19 has no body-diagonal populations, so the conservative corner route set
is empty. This function exists as an explicit canary for that topology fact.
"""
function coalesce_fine_to_coarse_corner_F_3d(Fc_src::AbstractArray{<:Any,4},
                                            q::Int,
                                            corner::Symbol)
    _check_child_block_3d(Fc_src, "Fc_src")
    qi = _check_d3q19_q(q)
    _check_conservative_tree_corner_3d(corner)
    throw(ArgumentError("population q=$qi cannot leave through corner $corner with D3Q19"))
end
