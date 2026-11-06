# Integrated physical-surface and discrete residual objectives.

"""
    FVWallTorque(center=(0,0); kind=:residual_flux, region=:solid, wall=0)

Scalar moment about `center`, per unit depth. `:residual_flux` uses the exact
complete wall momentum flux of the residual (including convection), with
the nominal Cartesian face centre as lever arm. `:surface` uses physical
cut points, reconstructed physical traction and analytic midpoint arc
weights. `wall=1,2,...` selects one analytic circle (inner/outer for annuli);
zero selects all walls. Geometry, weights and wall data carry parameter
duals. The default centre is fixed during differentiation.

The cell-centred moment sum also contains the internal-face transport
couple and the wall half-cell shift; it does not equal wall torque alone.
The surface quadrature of a constant stress is generally not exactly zero
on irregular cuts; no correction to its arc weights is applied.
"""
struct FVWallTorque{T} <: FVWallObjective
    center::NTuple{2,T}
    kind::Symbol
    region::Symbol
    wall::Int
end
function FVWallTorque(center=(0.0, 0.0); kind=:residual_flux, region=:solid, wall=0)
    f = FVWallForce(1; kind, region, wall)
    x, y = promote(center...)
    return FVWallTorque((x, y), f.kind, f.region, f.wall)
end

function _fv_surface_data(prob)
    prob.immersed !== nothing && prob.immersed.surface !== nothing ||
        throw(ArgumentError("surface objective requires analytic immersed surface metadata"))
    return prob.immersed.surface
end

_fv_copy_surface!(::Nothing, ::Nothing) = nothing
_fv_check_surface(::Nothing, ::Nothing) = nothing
function _fv_check_surface(a::FVImmersedSurface, b::FVImmersedSurface)
    a.given_viscosity == b.given_viscosity && a.min_transversality == b.min_transversality ||
        throw(ArgumentError("parameter map must preserve surface viscosity mode and transversality threshold"))
    size(a.values) == size(b.values) && size(a.wall) == size(b.wall) ||
        throw(DimensionMismatch("parameter map must preserve surface array shapes"))
    Array(a.wall) == Array(b.wall) ||
        throw(ArgumentError("parameter map changed surface wall identities"))
    return nothing
end
function _fv_copy_surface!(a::FVImmersedSurface, b::FVImmersedSurface)
    copyto!(a.values, b.values)
    copyto!(a.wall, b.wall)
    return nothing
end

function _fv_surface_force_face(f, U, prob, i, j, d, nuP, nuO)
    surface = _fv_surface_data(prob)
    s = surface.values
    ex, ey = (d == 1) - (d == 2), (d == 3) - (d == 4)
    opposite = nuO === nothing ? prob.nu[i - ex, j - ey] : nuO
    q = prob.immersed.qcut[d, i, j]
    nuw = surface.given_viscosity ? s[8, d, i, j] : nuP + q * (nuP - opposite)
    r = _fv_surface_reconstruction(U, prob, i, j, d,
        (s[3, d, i, j], s[4, d, i, j]), (s[6, d, i, j], s[7, d, i, j]), nuw;
        min_transversality=surface.min_transversality)
    T = promote_type(eltype(U), typeof(nuP), typeof(opposite), eltype(s),
                     _fv_visc_eltype(prob.visc), _fv_immersed_eltype(prob.immersed))
    return r.traction[f.dir] * s[5, d, i, j] + zero(T)
end

@inline function _fv_wall_force_face(f::FVWallTorque, U, prob, i, j, d,
                                     qcut, wvel, closure, nuP=prob.nu[i, j], nuO=nothing)
    fx = _fv_wall_force_face(FVWallForce(1; kind=f.kind, region=f.region),
                             U, prob, i, j, d, qcut, wvel, closure, nuP, nuO)
    fy = _fv_wall_force_face(FVWallForce(2; kind=f.kind, region=f.region),
                             U, prob, i, j, d, qcut, wvel, closure, nuP, nuO)
    if f.kind == :surface
        s = _fv_surface_data(prob).values
        x, y = s[1, d, i, j], s[2, d, i, j]
    else
        x = (i - 1//2 + ((d == 1) - (d == 2)) / 2) * prob.grid.h
        y = (j - 1//2 + ((d == 3) - (d == 4)) / 2) * prob.grid.h
    end
    return (x - f.center[1]) * fy - (y - f.center[2]) * fx
end
