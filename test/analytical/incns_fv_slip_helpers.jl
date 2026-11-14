# Shared cases of the FV wall-slip tests (src/methods/inc_ns/fv_slip.jl).
#
# Plane channel: y in [0, H], x-periodic, body force G along x, slip on both
# walls. τ_w = G H / 2 on each wall for every viscosity law (global balance),
# so the slip velocity is s = f(G H / 2) and u = u_noslip(y) + s.
# Newtonian: u_noslip = G/(2ν) y (H - y). The cell-centred FV solution with the
# half-cell wall difference is u_noslip + G h² / (8ν) + s exactly (the discrete
# wall traction equals G H/2 by the same balance), so the error is G h²/(8ν).
#
# Inclined immersed channel (incns_fv_immersed_walls_helpers.jl): with slip the
# exact velocity is (U(η) + U_s) t, U_s = f(τ_w), τ_w = g0 δ sin(1) (k δ = 1),
# W/E Dirichlet set to that profile; the flow stays parallel to the walls, so
# the tangential derivative neglected by the slip closure is zero.
using Test, KrakeNN, LinearAlgebra, ForwardDiff, Random
isdefined(@__MODULE__, :_iw_problem) || include(joinpath(@__DIR__, "incns_fv_immersed_walls_helpers.jl"))

const _SL_G, _SL_NU, _SL_H = 1.0, 0.5, 1.0

# nonlinear test law s = b τ + c τ³ (odd, increasing)
struct _SlipCubic{T} <: FVSlipLaw
    b::T
    c::T
end
KrakeNN.fv_slip_velocity(l::_SlipCubic, τ) = l.b * τ + l.c * τ^3
KrakeNN.fv_slip_parameters(l::_SlipCubic) = [l.b, l.c]
KrakeNN.fv_slip_law(::_SlipCubic, θ) = _SlipCubic(promote(θ[1], θ[2])...)

function _sl_channel(Ny; law = FVNavierSlip(0.1), conv = false, sides = (:N, :S),
                     rheology = nothing, eps = 0.0, Nx = 4, slip = :default)
    sl = slip === :default ? FVSlipWalls(law; sides) : slip
    return fv_steady_problem(Nx, Ny; h = _SL_H / Ny, nu = _SL_NU, conv,
        bc = (W = :periodic, E = :periodic, S = :wall, N = :wall),
        source = (x, y) -> (_SL_G, 0.0, 0.0), slip = sl, rheology, gnf_eps = eps)
end
_sl_noslip_u(y) = _SL_G / (2 * _SL_NU) * y * (_SL_H - y)
_sl_tauw() = _SL_G * _SL_H / 2

# regularised GNF channel reference (same inversion as incns_fv_gnf.jl)
function _sl_gdot(law, eps, tau)
    f(gd) = effective_viscosity(law, sqrt(gd^2 + eps^2)) * gd - tau
    lo, hi = 0.0, 1.0
    while f(hi) < 0
        hi *= 2
    end
    for _ in 1:200
        mid = (lo + hi) / 2
        f(mid) < 0 ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end
function _sl_gnf_noslip_u(law, eps, y; M = 4000)
    y = min(y, _SL_H - y)
    dy = y / M
    return sum(_sl_gdot(law, eps, _SL_G * (_SL_H / 2 - (k - 0.5) * dy)) for k in 1:M; init = 0.0) * dy
end

# inclined immersed channel with slip on both profile walls
_isl_tauw() = _IW_G0 * _IW_DELTA * sin(1.0)
_isl_us(law) = fv_slip_velocity(law, _isl_tauw())
_isl_velocity(x, y, us) = (u = _iw_U(_iw_eta(x, y)) + us; (u * _IW_C, u * _IW_S))

function _isl_problem(r; law = FVNavierSlip(0.04), conv = false, closure = :quadratic,
                      mass = :M3, rheology = nothing, eps = 0.0, slip = :default,
                      convection = :upwind)
    Nx, Ny, h = _iw_grid(r)
    ib = _iw_geometry(r; closure, mass)
    us = _isl_us(law)
    sl = slip === :default ? FVSlipWalls(law; immersed = true) : slip
    return fv_steady_problem(Nx, Ny; h, conv, immersed = ib, wall_flux = 0.0, nu = _IW_NU,
        bc = (W = :wall, E = :wall, S = :wall, N = :wall),
        boundary_velocity = (side, x, y) -> _isl_velocity(x, y, us),
        source = (x, y) -> _iw_force(x, y), slip = sl, rheology, gnf_eps = eps, convection)
end

_isl_state(p, us) = fv_state(p, (x, y) -> _isl_velocity(x, y, us)[1],
                             (x, y) -> _isl_velocity(x, y, us)[2], (x, y) -> 0.0)

function _isl_errors(p, U, us)
    g = p.grid
    f = fv_fields(p, U)
    e = Float64[]
    for j in 1:g.Ny, i in 1:g.Nx
        p.tag[i, j] == FV_CELL_SOLID && continue
        ue, ve = _isl_velocity((i - 0.5) * g.h, (j - 0.5) * g.h, us)
        push!(e, hypot(f[1][i, j] - ue, f[2][i, j] - ve))
    end
    return (inf = norm(e, Inf), l2 = sqrt(sum(abs2, e) / length(e)))
end

function _sl_dense_jacobian(p, U)
    D = ForwardDiff.Dual{KrakeNN.FVJacobianTag,Float64,1}
    X = D.(U); J = zeros(length(U), length(U)); R = similar(X)
    for col in eachindex(U)
        X[col] = D(U[col], ForwardDiff.Partials((1.0,)))
        fv_residual!(R, X, p)
        for row in eachindex(U)
            J[row, col] = ForwardDiff.partials(R[row])[1]
        end
        X[col] = D(U[col])
    end
    return J
end
