# Shared inclined-channel case for the generic immersed walls (FVProfileWall).
#
# Domain [0,2] x [0,1.5], h = 1/(16*2^r), Nx = 32*2^r, Ny = 24*2^r. Two straight
# walls of slope 1/4: lower y = 0.25 + x/4 (solid below), upper y = 0.75 + x/4
# (solid above). The wall intercepts are multiples of h on every grid and the
# slope is < 1/3, so the first/last x-links are never cut (every cut keeps its
# opposite fluid cell inside the domain). The channel is point-symmetric about
# (1, 0.75), so the prescribed W/E Dirichlet fluxes balance exactly.
#
# Exact flow (Stokes and Navier-Stokes): body force g0*cos(k*eta) along the
# channel axis t = (cos a, sin a), tan a = 1/4, eta = (y - 0.5 - x/4) cos a the
# signed distance to the centreline, half-width delta = cos(a)/4, k = 1/delta:
#   u = U(eta) t,  U = g0/(nu k^2) (cos(k eta) - cos(k delta)),  p = const.
# u.grad(u) = 0 because U does not vary along t. The profile is not polynomial,
# so the quadratic ghost does not reproduce it and an order can be measured.
#
# Exact temperature (manufactured, Dirichlet everywhere):
#   T = Tw + sin(pi x/2) [A (cos(k eta) - cos(k delta)) + B eta/delta]
# equals Tw on the W/E sides and Tw -+ B sin(pi x/2) on the walls (a wall
# temperature function). The cell source is q = u.grad(T) - alpha lap(T).
using Test, KrakeNN, LinearAlgebra, ForwardDiff, Random

const _IW_A = atan(0.25)
const _IW_C, _IW_S = cos(_IW_A), sin(_IW_A)
const _IW_DELTA = 0.25 * _IW_C
const _IW_K = 1 / _IW_DELTA
const _IW_NU = 1.0
const _IW_G0 = 20.0
const _IW_TW = 1.0
const _IW_TA = 1.0
const _IW_TB = 0.5
const _IW_ALPHA = 1.0

_iw_eta(x, y) = (y - 0.5 - 0.25x) * _IW_C
_iw_U(eta) = _IW_G0 / (_IW_NU * _IW_K^2) * (cos(_IW_K * eta) - cos(_IW_K * _IW_DELTA))
_iw_velocity(x, y) = (u = _iw_U(_iw_eta(x, y)); (u * _IW_C, u * _IW_S))
_iw_force(x, y) = (f = _IW_G0 * cos(_IW_K * _iw_eta(x, y)); (f * _IW_C, f * _IW_S, 0.0))

_iw_F(eta) = _IW_TA * (cos(_IW_K * eta) - cos(_IW_K * _IW_DELTA)) + _IW_TB * eta / _IW_DELTA
_iw_dF(eta) = -_IW_TA * _IW_K * sin(_IW_K * eta) + _IW_TB / _IW_DELTA
_iw_d2F(eta) = -_IW_TA * _IW_K^2 * cos(_IW_K * eta)
_iw_temperature(x, y) = _IW_TW + sin(pi * x / 2) * _iw_F(_iw_eta(x, y))
function _iw_heat_source(x, y; alpha = _IW_ALPHA)
    eta = _iw_eta(x, y)
    s, ds, d2s = sin(pi * x / 2), pi / 2 * cos(pi * x / 2), -(pi / 2)^2 * sin(pi * x / 2)
    F, dF, d2F = _iw_F(eta), _iw_dF(eta), _iw_d2F(eta)
    lap = d2s * F - 2 * ds * dF * _IW_S + s * d2F
    convection = _iw_U(eta) * _IW_C * ds * F
    return convection - alpha * lap
end
# Diffusive heat leaving the fluid through wall k (k=1 lower, k=2 upper):
# integral of -alpha grad(T).n over x in [0,2], n = fluid -> solid unit normal.
function _iw_wall_heat_exact(k; alpha = _IW_ALPHA, n = 4096)
    total = 0.0
    nx, ny = k == 1 ? (_IW_S, -_IW_C) : (-_IW_S, _IW_C)
    ds = 2 / n / _IW_C
    for l in 1:n
        x = 2 * (l - 0.5) / n
        y = (k == 1 ? 0.25 : 0.75) + 0.25x
        eta = _iw_eta(x, y)
        s, dsx = sin(pi * x / 2), pi / 2 * cos(pi * x / 2)
        Tx = dsx * _iw_F(eta) - s * _iw_dF(eta) * _IW_S
        Ty = s * _iw_dF(eta) * _IW_C
        total += -alpha * (Tx * nx + Ty * ny) * ds
    end
    return total
end

_iw_walls() = (FVProfileWall([0.0, 2.0], [0.25, 0.75]; solid=:below),
               FVProfileWall([0.0, 2.0], [0.75, 1.25]; solid=:above))
_iw_grid(r) = (32 * 2^r, 24 * 2^r, 1 / (16 * 2^r))

function _iw_geometry(r; closure=:quadratic, mass=:M3, pressure=:linear)
    Nx, Ny, h = _iw_grid(r)
    return fv_immersed_walls(Nx, Ny; walls=_iw_walls(), h, closure, mass, pressure)
end

function _iw_problem(r; conv=false, thermal=false, closure=:quadratic, mass=:M3,
                     viscosity=nothing, nu=_IW_NU, energy=nothing)
    Nx, Ny, h = _iw_grid(r)
    ib = _iw_geometry(r; closure, mass)
    if thermal && energy === nothing
        energy = FVEnergy(; alpha=_IW_ALPHA, left=_IW_TW, right=_IW_TW,
            solid=(k, x, y) -> _iw_temperature(x, y),
            source=(x, y) -> _iw_heat_source(x, y))
    end
    kw = viscosity === nothing ? (; nu) : (; viscosity)
    return fv_steady_problem(Nx, Ny; h, conv, immersed=ib, wall_flux=0.0,
        bc=(W=:wall, E=:wall, S=:wall, N=:wall),
        boundary_velocity=(side, x, y) -> _iw_velocity(x, y),
        source=(x, y) -> _iw_force(x, y), energy, kw...)
end

function _iw_exact_state(p)
    u = (x, y) -> _iw_velocity(x, y)[1]
    v = (x, y) -> _iw_velocity(x, y)[2]
    pr = (x, y) -> 0.0
    return p.energy === nothing ? fv_state(p, u, v, pr) : fv_state(p, u, v, pr, _iw_temperature)
end

function _iw_errors(p, U)
    g = p.grid
    fields = fv_fields(p, U)
    u, v = fields[1], fields[2]
    eu = Float64[]; et = Float64[]
    for j in 1:g.Ny, i in 1:g.Nx
        p.tag[i, j] == FV_CELL_SOLID && continue
        x, y = (i - 0.5) * g.h, (j - 0.5) * g.h
        ue, ve = _iw_velocity(x, y)
        push!(eu, hypot(u[i, j] - ue, v[i, j] - ve))
        length(fields) == 4 && push!(et, fields[4][i, j] - _iw_temperature(x, y))
    end
    vel = (inf=norm(eu, Inf), l2=sqrt(sum(abs2, eu) / length(eu)))
    temp = isempty(et) ? nothing : (inf=norm(et, Inf), l2=sqrt(sum(abs2, et) / length(et)))
    return vel, temp
end

_iw_order(a, b) = log2(a / b)
