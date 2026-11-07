# Equilibrium/non-equilibrium reconstruction for integrated D2Q9 populations.
#
# This is the scalar CPU reference contract used by AMR-D before it is lowered
# into route/interface kernels. Rows store integrated populations
# `F_q = f_q * cell_volume`.

"""
    macrostate_integrated_D2Q9(Fcell::AbstractVector, volume)

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.macrostate_integrated_D2Q9)
```
"""
function macrostate_integrated_D2Q9(Fcell::AbstractVector, volume)
    _check_d2q9_vector(Fcell, "Fcell")
    T = typeof(zero(eltype(Fcell)) + volume)
    vol = T(volume)
    vol > zero(T) || throw(ArgumentError("volume must be positive"))

    mass = zero(T)
    mx = zero(T)
    my = zero(T)
    @inbounds for q in 1:9
        Fq = T(Fcell[q])
        mass += Fq
        mx += T(d2q9_cx(q)) * Fq
        my += T(d2q9_cy(q)) * Fq
    end
    iszero(mass) && throw(ArgumentError("Fcell mass must be nonzero"))

    rho = mass / vol
    ux = mx / mass
    uy = my / mass
    return rho, ux, uy
end

"""
    reconstruct_integrated_D2Q9_eq_neq!(

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.reconstruct_integrated_D2Q9_eq_neq!)
```
"""
function reconstruct_integrated_D2Q9_eq_neq!(
        Fdst::AbstractVector,
        dst_volume,
        Fsrc::AbstractVector,
        src_volume;
        alpha=1)
    _check_d2q9_vector(Fdst, "Fdst")
    _check_d2q9_vector(Fsrc, "Fsrc")
    T = typeof(zero(eltype(Fdst)) + zero(eltype(Fsrc)) +
               dst_volume + src_volume + alpha)
    src_vol = T(src_volume)
    dst_vol = T(dst_volume)
    src_vol > zero(T) || throw(ArgumentError("src_volume must be positive"))
    dst_vol > zero(T) || throw(ArgumentError("dst_volume must be positive"))

    rho, ux, uy = macrostate_integrated_D2Q9(Fsrc, src_vol)
    a = T(alpha)
    @inbounds for q in 1:9
        feq = equilibrium(D2Q9(), T(rho), T(ux), T(uy), q)
        fsrc = T(Fsrc[q]) / src_vol
        Fdst[q] = dst_vol * (feq + a * (fsrc - feq))
    end
    return Fdst
end

"""
    reconstructed_integrated_D2Q9_packet(

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.reconstructed_integrated_D2Q9_packet)
```
"""
function reconstructed_integrated_D2Q9_packet(
        Fsrc::AbstractVector,
        src_volume,
        q::Integer,
        weight;
        alpha=1)
    _check_d2q9_vector(Fsrc, "Fsrc")
    qi = _check_d2q9_q(Int(q))
    T = typeof(zero(eltype(Fsrc)) + src_volume + weight + alpha)
    src_vol = T(src_volume)
    src_vol > zero(T) || throw(ArgumentError("src_volume must be positive"))
    mass = zero(T)
    @inbounds for qq in 1:9
        mass += T(Fsrc[qq])
    end
    iszero(mass) && return zero(T)

    rho, ux, uy = macrostate_integrated_D2Q9(Fsrc, src_vol)
    feq = equilibrium(D2Q9(), T(rho), T(ux), T(uy), qi)
    fsrc = T(Fsrc[qi]) / src_vol
    return T(weight) * src_vol * (feq + T(alpha) * (fsrc - feq))
end
