# Analytic, host-side lowering for the 2D immersed FV wall contract.
import ForwardDiff

"""
    FVImmersedClosure([closure]; mass=:M3, pressure=:linear)

Isbits wall configuration. `closure=:quadratic` uses the opposite fluid cell;
`:linear` uses the wall and fluid cell. Mass is `:M3` (pairwise corrected nominal-face velocity), `:M1`
(uncorrected nominal-face velocity), or `:M2` (wall velocity); pressure is `:linear` or `:zero_gradient`.
`:c1` forces linear interpolation at q=1/2, M2 and zero pressure gradient.
The integer codes are ghost=0/1/2 (C1/linear/quadratic), mass=1/2/3 and
pressure=0/1 (zero-gradient/linear).
"""
struct FVImmersedClosure
    ghost::UInt8
    mass::UInt8
    pressure::UInt8
end
function FVImmersedClosure(closure::Symbol; mass=:M3, pressure=:linear)
    closure in (:c1, :linear, :quadratic) || throw(ArgumentError("unknown immersed closure $closure"))
    mass in (:M1, :M2, :M3) || throw(ArgumentError("unknown immersed mass closure $mass"))
    pressure in (:zero_gradient, :linear) || throw(ArgumentError("unknown immersed pressure closure $pressure"))
    closure == :c1 && return FVImmersedClosure(0x00, 0x02, 0x00)
    return FVImmersedClosure(closure == :linear ? 0x01 : 0x02,
                             mass == :M1 ? 0x01 : mass == :M2 ? 0x02 : 0x03,
                             pressure == :zero_gradient ? 0x00 : 0x01)
end
FVImmersedClosure(; closure=:quadratic, mass=:M3, pressure=:linear) =
    FVImmersedClosure(closure; mass, pressure)

"""
    FVImmersedBoundary2D(qcut, wvel, tag, h, closure)

Lowered 2D geometry: `qcut[direction,i,j]` in E,W,N,S order and
`wvel[component,direction,i,j]` at each physical wall crossing. Uncut links
have zero q. `tag` contains only fluid/solid cells; pressure pinning belongs
to the problem. Arrays may have independent scalar types, including duals.
Construct using [`fv_immersed_boundary`](@ref); the raw constructor assumes
the resolution contract has already been checked. The five-argument raw
constructor carries no analytic surface metadata and supports residual-flux
objectives only. M3 pairing is lowered by `fv_steady_problem`, which requires
an exact `wall_flux` for raw geometry. No 3D lowering is supplied.
"""
struct FVImmersedBoundary2D{Q<:AbstractArray,W<:AbstractArray,T<:AbstractMatrix,H,S,M}
    qcut::Q
    wvel::W
    tag::T
    h::H
    closure::FVImmersedClosure
    surface::S
    mass::M
end
FVImmersedBoundary2D(q, w, tag, h, closure) = FVImmersedBoundary2D(q, w, tag, h, closure, nothing)
FVImmersedBoundary2D(q, w, tag, h, closure, surface) =
    FVImmersedBoundary2D(q, w, tag, h, closure, surface, nothing)
fv_adapt(to, b::FVImmersedBoundary2D) =
    FVImmersedBoundary2D(to(b.qcut), to(b.wvel), to(b.tag), b.h, b.closure,
                        _fv_adapt_surface(to, b.surface), _fv_adapt_mass(to, b.mass))

# ForwardDiff comparisons can consult partials when primal values tie.
# Geometry topology and root selection must depend on primal values only.
_fv_ib_primal(x) = x
_fv_ib_primal(x::ForwardDiff.Dual) = _fv_ib_primal(ForwardDiff.value(x))
_fv_ib_phi(x, y, c) = (x - c[1])^2 + (y - c[2])^2 - c[3]^2
_fv_ib_cell(c, h, Nx, Ny) = (clamp(round(Int, _fv_ib_primal(c[1] / h + 1//2)), 1, Nx),
                            clamp(round(Int, _fv_ib_primal(c[2] / h + 1//2)), 1, Ny))
_fv_ib_error(message, cells...) = throw(ArgumentError("immersed geometry: $message; cells $(join(cells, ", "))"))

# Both roots of one circle, independently of global union/annulus tags.
# At an exact endpoint use a translated quadratic. This is an algebraic
# evaluation, not a tolerance snap: the primal endpoint and its derivative
# are both retained without clipping q.
function _fv_ib_roots(x, y, ex, ey, h, circle)
    a = h^2
    b = 2h * ((x - circle[1]) * ex + (y - circle[2]) * ey)
    c = _fv_ib_phi(x, y, circle)
    ce = _fv_ib_phi(x + h * ex, y + h * ey, circle)
    if iszero(_fv_ib_primal(c))
        iszero(_fv_ib_primal(b)) && return (zero(c),)
        sd = sqrt(b^2 - 4a * c)
        t = -2c / (b + sign(_fv_ib_primal(b)) * sd)
        return (t, -b / a - t)
    elseif iszero(_fv_ib_primal(ce))
        be = b + 2a
        iszero(_fv_ib_primal(be)) && return (one(ce),)
        sd = sqrt(be^2 - 4a * ce)
        t = one(ce) - 2ce / (be + sign(_fv_ib_primal(be)) * sd)
        return (t, -b / a - t)
    end
    disc = b^2 - 4a * c
    _fv_ib_primal(disc) < 0 && return ()
    iszero(_fv_ib_primal(disc)) && return (-b / (2a),)
    sd = sqrt(disc)
    # Product of roots is c/a: evaluate the smaller root without subtracting
    # two nearly equal numbers when a fluid centre approaches the wall.
    r = -(b + (_fv_ib_primal(b) >= 0 ? sd : -sd)) / 2
    return (r / a, c / r)
end

function _fv_ib_resolution(circles, annular, h, Nx, Ny)
    for (k, c) in enumerate(circles)
        cell = _fv_ib_cell(c, h, Nx, Ny)
        _fv_ib_primal(c[3] - 2h) >= 0 || _fv_ib_error("wall $k radius must be >= 2h", cell)
        gap = min(c[1] - c[3], Nx * h - c[1] - c[3],
                  c[2] - c[3], Ny * h - c[2] - c[3])
        _fv_ib_primal(gap - 3h) >= 0 || _fv_ib_error("wall $k/domain gap must be >= 3h", cell)
    end
    for k in eachindex(circles), l in (k + 1):length(circles)
        a, b = circles[k], circles[l]
        gap = annular ? b[3] - a[3] : hypot(a[1] - b[1], a[2] - b[2]) - a[3] - b[3]
        _fv_ib_primal(gap - 3h) >= 0 || _fv_ib_error("walls $k/$l gap must be >= 3h",
            _fv_ib_cell(a, h, Nx, Ny), _fv_ib_cell(b, h, Nx, Ny))
    end
end

function _fv_ib_tags(circles, annular, h, Nx, Ny, frozen_tag)
    tag = fill(FV_CELL_FLUID, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        x, y = (i - 1//2) * h, (j - 1//2) * h
        solid = if annular
            _fv_ib_primal(_fv_ib_phi(x, y, circles[1])) <= 0 ||
                _fv_ib_primal(_fv_ib_phi(x, y, circles[2])) >= 0
        else
            any(c -> _fv_ib_primal(_fv_ib_phi(x, y, c)) <= 0, circles)
        end
        solid && (tag[i, j] = FV_CELL_SOLID)
    end
    if frozen_tag !== nothing
        size(frozen_tag) == (Nx, Ny) || throw(DimensionMismatch("frozen_tag must have size ($Nx, $Ny)"))
        for j in 1:Ny, i in 1:Nx
            frozen_tag[i, j] in (FV_CELL_FLUID, FV_CELL_SOLID, FV_CELL_PINNED) ||
                _fv_ib_error("invalid frozen tag", (i, j))
            oldsolid = frozen_tag[i, j] == FV_CELL_SOLID
            oldsolid == (tag[i, j] == FV_CELL_SOLID) ||
                _fv_ib_error("frozen fluid/solid tag changed", (i, j))
        end
    end
    return tag
end

"""
    fv_immersed_boundary(Nx, Ny; h=1.0, circles=(), annulus=nothing,
        wall_velocity=(wall,x,y)->(0,0), frozen_tag=nothing,
        wall_tangent=nothing, wall_viscosity=nothing,
        min_transversality=1e-8, closure=:quadratic, mass=:M3, pressure=:linear)

Lower disjoint solid circles `(cx,cy,R)`, or an annular fluid region
`annulus=(cx,cy,Rinner,Router)`, on cell centres `((i-1/2)h,(j-1/2)h)`.
The callback's wall index is the circle index, or 1=inner/2=outer for an
annulus; callback coordinates are the actual wall intersection.

Analytic surface metadata includes cut coordinates, fluid-to-solid normals,
midpoint arc weights and Dirichlet tangential derivatives. An optional
`wall_tangent(wall,x,y,tx,ty)` supplies `(du/dt,dv/dt)` along the unit tangent;
the default differentiates `wall_velocity` at the cut. An optional
`wall_viscosity(wall,x,y)` supplies physical cut viscosity. Without it, the
surface objective extrapolates `nuP + q*(nuP-nuO)` (exact for affine given
viscosity); a GNF law instead uses the reconstructed physical wall gradient.
All metadata carries parameter duals and is transferred by `fv_adapt`.

The checked contract is radius >=2h, distinct-wall and wall/domain gaps
>=3h, and exactly one boundary crossing on different-tag axial links and
none strictly inside same-tag links. A centre on any wall is solid, and
an intersection at the solid endpoint is retained (q=1). Every cut has an
opposite fluid cell and `abs(e dot normal)>min_transversality`. No small-q
regularization is used. Errors identify the affected cells.

`frozen_tag` rejects any changed fluid/solid classification, treating pinned
cells as fluid. Arithmetic remains generic in scalar type for geometry and
wall-velocity differentiation; only topology decisions use primal values.
`closure` accepts a symbol or [`FVImmersedClosure`](@ref).
"""
function fv_immersed_boundary(Nx::Int, Ny::Int; h=1.0, circles=(), annulus=nothing,
        wall_velocity=(wall, x, y) -> (zero(x), zero(y)), frozen_tag=nothing,
        wall_tangent=nothing, wall_viscosity=nothing,
        min_transversality=1e-8, closure=:quadratic, mass=:M3, pressure=:linear)
    Nx > 0 && Ny > 0 || throw(ArgumentError("immersed grid dimensions must be positive"))
    isfinite(_fv_ib_primal(h)) && _fv_ib_primal(h) > 0 || throw(ArgumentError("immersed h must be finite and positive"))
    0 <= min_transversality < 1 || throw(ArgumentError("min_transversality must lie in [0,1)"))
    annular = annulus !== nothing
    annular && !isempty(circles) && throw(ArgumentError("choose circles or annulus, not both"))
    raw = annular ? ((annulus[1], annulus[2], annulus[3]),
                     (annulus[1], annulus[2], annulus[4])) : Tuple(circles)
    isempty(raw) && throw(ArgumentError("at least one immersed circle is required"))
    all(c -> length(c) == 3, raw) || throw(ArgumentError("circles must contain (cx,cy,R) tuples"))
    T = promote_type(typeof(float(h)), (typeof(v) for c in raw for v in c)...)
    hh = T(h)
    walls = [Tuple(T.(c)) for c in raw]
    for c in walls
        all(v -> isfinite(_fv_ib_primal(v)), c) || _fv_ib_error("wall parameters must be finite", (1, 1))
    end
    _fv_ib_resolution(walls, annular, hh, Nx, Ny)
    tag = _fv_ib_tags(walls, annular, hh, Nx, Ny, frozen_tag)
    qcut = zeros(T, 4, Nx, Ny)
    samples = Tuple[]
    WT = T
    for j in 1:Ny, i in 1:Nx, axis in 1:2
        ex, ey = axis == 1 ? (1, 0) : (0, 1)
        ni, nj = i + ex, j + ey
        ni <= Nx && nj <= Ny || continue
        x, y = (i - 1//2) * hh, (j - 1//2) * hh
        different = tag[i, j] != tag[ni, nj]
        hits, wall, cut = 0, 0, zero(T)
        for (k, c) in enumerate(walls), t in _fv_ib_roots(x, y, ex, ey, hh, c)
            tp = _fv_ib_primal(t)
            inside = 0 < tp < 1
            endpoint = different && ((tp == 0 && tag[i, j] == FV_CELL_SOLID) ||
                                       (tp == 1 && tag[ni, nj] == FV_CELL_SOLID))
            if inside || endpoint
                hits += 1
                wall, cut = k, t
            end
        end
        hits == Int(different) || _fv_ib_error("axial link has $hits boundary intersections; expected $(Int(different))",
                                              (i, j), (ni, nj))
        different || continue
        forward = tag[i, j] == FV_CELL_FLUID
        pi, pj = forward ? (i, j) : (ni, nj)
        dx, dy = forward ? (ex, ey) : (-ex, -ey)
        oi, oj = pi - dx, pj - dy
        1 <= oi <= Nx && 1 <= oj <= Ny && tag[oi, oj] == FV_CELL_FLUID ||
            _fv_ib_error("opposite cell must be fluid", (pi, pj), (oi, oj))
        q = forward ? cut : one(cut) - cut
        0 < _fv_ib_primal(q) <= 1 || _fv_ib_error("cut fraction must lie in (0,1]", (pi, pj))
        wx, wy = x + hh * cut * ex, y + hh * cut * ey
        c = walls[wall]
        transverse = abs(((wx - c[1]) * ex + (wy - c[2]) * ey) / c[3])
        _fv_ib_primal(transverse) > min_transversality ||
            _fv_ib_error("wall $wall transversality $(_fv_ib_primal(transverse)) must exceed $min_transversality",
                         (pi, pj), (pi + dx, pj + dy))
        d = axis == 1 ? (forward ? 1 : 2) : (forward ? 3 : 4)
        qcut[d, pi, pj] = q
        uv = wall_velocity(wall, wx, wy)
        length(uv) == 2 || throw(ArgumentError("wall_velocity must return two components"))
        WT = promote_type(WT, typeof(uv[1]), typeof(uv[2]))
        push!(samples, (d, pi, pj, uv[1], uv[2]))
    end
    wvel = zeros(WT, 2, 4, Nx, Ny)
    for (d, i, j, u, v) in samples
        wvel[1, d, i, j], wvel[2, d, i, j] = u, v
    end
    config = closure isa FVImmersedClosure ? closure : FVImmersedClosure(closure; mass, pressure)
    surface = _fv_immersed_surface(qcut, walls, annular, hh, wall_velocity,
                                  wall_tangent, wall_viscosity, min_transversality)
    mass_input = config.mass == 3 ? _fv_wall_flux_quadrature(walls, annular, wall_velocity) : nothing
    return FVImmersedBoundary2D(qcut, wvel, tag, h, config, surface, mass_input)
end

"""Construct one solid circle; keywords follow [`fv_immersed_boundary`](@ref)."""
fv_immersed_circle(Nx, Ny, cx, cy, R; kwargs...) =
    fv_immersed_boundary(Nx, Ny; circles=((cx, cy, R),), kwargs...)

"""Construct an annular fluid region; keywords follow [`fv_immersed_boundary`](@ref)."""
fv_immersed_annulus(Nx, Ny, cx, cy, Ri, Ro; kwargs...) =
    fv_immersed_boundary(Nx, Ny; annulus=(cx, cy, Ri, Ro), kwargs...)
