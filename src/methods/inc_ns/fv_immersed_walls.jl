# Generic 2D immersed walls: piecewise-linear profiles y = f(x) and circles,
# lowered to the same FVImmersedBoundary2D arrays as the analytic circles.
# The circle/annulus constructor `fv_immersed_boundary(...; circles, annulus)`
# keeps its own code path; this file adds a separate lowering.

"""Abstract 2D immersed wall: a curve separating a solid side from the fluid."""
abstract type FVImmersedWall end

"""
    FVProfileWall(x, y; solid=:above)

Piecewise-linear wall `y = f(x)` through the vertices `(x[k], y[k])` (strictly
increasing `x`). `solid=:above` makes `y >= f(x)` solid, `:below` makes
`y <= f(x)` solid. The vertices must cover the whole domain width: the wall
crosses the west and east sides, which must then be inlet/outlet/wall sides.
Vertex ordinates may be parameter duals (shape maps).
"""
struct FVProfileWall{T} <: FVImmersedWall
    x::Vector{T}
    y::Vector{T}
    solid_above::Bool
end
function FVProfileWall(x::AbstractVector, y::AbstractVector; solid::Symbol=:above)
    solid in (:above, :below) || throw(ArgumentError("profile wall solid side must be :above or :below"))
    length(x) == length(y) && length(x) >= 2 ||
        throw(ArgumentError("profile wall needs at least two (x, y) vertices"))
    all(k -> _fv_ib_primal(x[k+1]) > _fv_ib_primal(x[k]), 1:length(x)-1) ||
        throw(ArgumentError("profile wall abscissae must be strictly increasing"))
    T = promote_type(eltype(x), eltype(y))
    T <: Integer && (T = Float64)
    return FVProfileWall{T}(T.(collect(x)), T.(collect(y)), solid === :above)
end

"""    FVCircleWall(cx, cy, R)

Solid disk of radius `R`: the circle as one case of the generic wall lowering."""
struct FVCircleWall{T} <: FVImmersedWall
    cx::T
    cy::T
    R::T
end
FVCircleWall(cx, cy, R) = (T = float(promote_type(typeof(cx), typeof(cy), typeof(R)));
                           FVCircleWall{T}(T(cx), T(cy), T(R)))

_fv_wall_eltype(w::FVProfileWall{T}) where {T} = T
_fv_wall_eltype(w::FVCircleWall{T}) where {T} = T
_fv_wall_open(::FVProfileWall) = true
_fv_wall_open(::FVCircleWall) = false

# ---------------------------------------------------------------- profile helpers
function _fv_profile_segment(w::FVProfileWall, x)
    xp = _fv_ib_primal(x); n = length(w.x)
    for k in 1:n-2
        xp < _fv_ib_primal(w.x[k+1]) && return k
    end
    return n - 1
end
_fv_profile_slope(w, k) = (w.y[k+1] - w.y[k]) / (w.x[k+1] - w.x[k])
function _fv_profile_y(w::FVProfileWall, x)
    k = _fv_profile_segment(w, x)
    return w.y[k] + _fv_profile_slope(w, k) * (x - w.x[k])
end
# Arc length from the first vertex to abscissa x along the polyline.
function _fv_profile_arc(w::FVProfileWall, x)
    k = _fv_profile_segment(w, x)
    s = zero(x) + zero(eltype(w.y))
    for l in 1:k-1
        s += hypot(w.x[l+1] - w.x[l], w.y[l+1] - w.y[l])
    end
    return s + (x - w.x[k]) * sqrt(1 + _fv_profile_slope(w, k)^2)
end

# ---------------------------------------------------------------- generic interface
# Signed side function: <= 0 is solid. Topology uses primal values only.
_fv_wall_phi(w::FVCircleWall, x, y) = _fv_ib_phi(x, y, (w.cx, w.cy, w.R))
function _fv_wall_phi(w::FVProfileWall, x, y)
    f = _fv_profile_y(w, x)
    return w.solid_above ? f - y : y - f
end

# Link parameters t (x + t*h*e) where the axial link crosses the wall.
_fv_wall_roots(w::FVCircleWall, x, y, ex, ey, h) = _fv_ib_roots(x, y, ex, ey, h, (w.cx, w.cy, w.R))
function _fv_wall_roots(w::FVProfileWall, x, y, ex, ey, h)
    if ex == 0
        return ((_fv_profile_y(w, x) - y) / h,)
    end
    T = promote_type(typeof(x), eltype(w.y), typeof(h))
    roots = T[]
    n = length(w.x)
    for k in 1:n-1
        s = _fv_profile_slope(w, k)
        if iszero(_fv_ib_primal(s))
            _fv_ib_primal(w.y[k]) == _fv_ib_primal(y) &&
                _fv_ib_primal(w.x[k+1]) > _fv_ib_primal(x) && _fv_ib_primal(w.x[k]) < _fv_ib_primal(x + h) &&
                _fv_ib_error("profile wall segment lies on a horizontal link", (0, 0))
            continue
        end
        xs = w.x[k] + (y - w.y[k]) / s
        xp = _fv_ib_primal(xs)
        inseg = xp >= _fv_ib_primal(w.x[k]) && (xp < _fv_ib_primal(w.x[k+1]) ||
                                                 (k == n - 1 && xp == _fv_ib_primal(w.x[k+1])))
        inseg && _fv_ib_primal(x) <= xp <= _fv_ib_primal(x + h) && push!(roots, (xs - x) / h)
    end
    return roots
end

# Unit normal from fluid into solid at a wall point.
function _fv_wall_normal(w::FVCircleWall, x, y)
    return -(x - w.cx) / w.R, -(y - w.cy) / w.R
end
function _fv_wall_normal(w::FVProfileWall, x, y)
    s = _fv_profile_slope(w, _fv_profile_segment(w, x))
    r = sqrt(1 + s^2)
    return w.solid_above ? (-s / r, one(r) / r) : (s / r, -one(r) / r)
end

# Arc coordinate used to order cut points along one wall.
_fv_wall_arcpos(w::FVCircleWall, x, y) = atan(y - w.cy, x - w.cx)
_fv_wall_arcpos(w::FVProfileWall, x, y) = _fv_profile_arc(w, x)

# Physical outward (fluid -> solid) wall flux of the prescribed wall velocity.
function _fv_wall_flux_integral(w::FVCircleWall, k, velocity, n, Lx)
    total = zero(w.R)
    for l in 0:n-1
        angle = 2pi * l / n
        nx, ny = cos(angle), sin(angle)
        u, v = velocity(k, w.cx + w.R * nx, w.cy + w.R * ny)
        total -= w.R * (u * nx + v * ny) * (2pi / n)
    end
    return total
end
function _fv_wall_flux_integral(w::FVProfileWall, k, velocity, n, Lx)
    total = zero(eltype(w.y))
    for s in 1:length(w.x)-1
        a = max(w.x[s], zero(w.x[s])); b = min(w.x[s+1], oftype(w.x[s+1], Lx))
        _fv_ib_primal(b) > _fv_ib_primal(a) || continue
        slope = _fv_profile_slope(w, s)
        nx, ny = _fv_wall_normal(w, (a + b) / 2, zero(a))
        ds = (b - a) * sqrt(1 + slope^2) / n
        for l in 1:n
            x = a + (b - a) * (l - 1//2) / n
            u, v = velocity(k, x, w.y[s] + slope * (x - w.x[s]))
            total += (u * nx + v * ny) * ds
        end
    end
    return total
end

# ---------------------------------------------------------------- resolution contract
function _fv_profile_breaks(a::FVProfileWall, b, Lx)
    xs = [zero(Lx), Lx]
    for w in (a, b)
        w isa FVProfileWall || continue
        for x in w.x
            0 < _fv_ib_primal(x) < _fv_ib_primal(Lx) && push!(xs, x)
        end
    end
    return xs
end

function _fv_walls_resolution(walls, h, Nx, Ny)
    Lx, Ly = Nx * h, Ny * h
    cell(x, y) = (clamp(round(Int, _fv_ib_primal(x / h + 1//2)), 1, Nx),
                  clamp(round(Int, _fv_ib_primal(y / h + 1//2)), 1, Ny))
    for (k, w) in enumerate(walls)
        if w isa FVCircleWall
            _fv_ib_primal(w.R - 2h) >= 0 || _fv_ib_error("wall $k radius must be >= 2h", cell(w.cx, w.cy))
            gap = min(w.cx - w.R, Lx - w.cx - w.R, w.cy - w.R, Ly - w.cy - w.R)
            _fv_ib_primal(gap - 3h) >= 0 || _fv_ib_error("wall $k/domain gap must be >= 3h", cell(w.cx, w.cy))
        else
            _fv_ib_primal(w.x[1]) <= 0 && _fv_ib_primal(w.x[end]) >= _fv_ib_primal(Lx) ||
                _fv_ib_error("profile wall $k must cover the domain width [0, $(_fv_ib_primal(Lx))]", (1, 1), (Nx, 1))
            for x in _fv_profile_breaks(w, w, Lx)
                f = _fv_profile_y(w, x)
                _fv_ib_primal(min(f, Ly - f) - 3h) >= 0 ||
                    _fv_ib_error("profile wall $k/domain gap must be >= 3h at x=$(_fv_ib_primal(x))", cell(x, f))
            end
        end
    end
    for k in eachindex(walls), l in (k + 1):length(walls)
        a, b = walls[k], walls[l]
        if a isa FVCircleWall && b isa FVCircleWall
            gap = hypot(a.cx - b.cx, a.cy - b.cy) - a.R - b.R
            _fv_ib_primal(gap - 3h) >= 0 || _fv_ib_error("walls $k/$l gap must be >= 3h",
                cell(a.cx, a.cy), cell(b.cx, b.cy))
        elseif a isa FVProfileWall && b isa FVProfileWall
            for x in _fv_profile_breaks(a, b, Lx)
                gap = abs(_fv_profile_y(a, x) - _fv_profile_y(b, x))
                _fv_ib_primal(gap - 3h) >= 0 || _fv_ib_error("walls $k/$l vertical gap must be >= 3h at x=$(_fv_ib_primal(x))",
                    cell(x, _fv_profile_y(a, x)), cell(x, _fv_profile_y(b, x)))
            end
        else
            c, p = a isa FVCircleWall ? (a, b) : (b, a)
            # distance from the centre to every segment of the polyline
            d = Inf
            for s in 1:length(p.x)-1
                ax, ay, bx, by = p.x[s], p.y[s], p.x[s+1], p.y[s+1]
                vx, vy = bx - ax, by - ay
                t = clamp(_fv_ib_primal(((c.cx - ax) * vx + (c.cy - ay) * vy) / (vx^2 + vy^2)), 0, 1)
                d = min(d, _fv_ib_primal(hypot(c.cx - ax - t * vx, c.cy - ay - t * vy)))
            end
            d - _fv_ib_primal(c.R + 3h) >= 0 || _fv_ib_error("walls $k/$l gap must be >= 3h", cell(c.cx, c.cy))
        end
    end
end

function _fv_walls_tags(walls, h, Nx, Ny, frozen_tag)
    tag = fill(FV_CELL_FLUID, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        x, y = (i - 1//2) * h, (j - 1//2) * h
        any(w -> _fv_ib_primal(_fv_wall_phi(w, x, y)) <= 0, walls) && (tag[i, j] = FV_CELL_SOLID)
    end
    if frozen_tag !== nothing
        size(frozen_tag) == (Nx, Ny) || throw(DimensionMismatch("frozen_tag must have size ($Nx, $Ny)"))
        for j in 1:Ny, i in 1:Nx
            frozen_tag[i, j] in (FV_CELL_FLUID, FV_CELL_SOLID, FV_CELL_PINNED) ||
                _fv_ib_error("invalid frozen tag", (i, j))
            (frozen_tag[i, j] == FV_CELL_SOLID) == (tag[i, j] == FV_CELL_SOLID) ||
                _fv_ib_error("frozen fluid/solid tag changed", (i, j))
        end
    end
    return tag
end

"""
    fv_immersed_walls(Nx, Ny; walls, h=1.0, wall_velocity=(wall,x,y)->(0,0),
        frozen_tag=nothing, wall_tangent=nothing, wall_viscosity=nothing,
        min_transversality=1e-8, closure=:quadratic, mass=:M3, pressure=:linear)

Lower generic immersed walls (a tuple of [`FVProfileWall`](@ref) and
[`FVCircleWall`](@ref)) on cell centres `((i-1/2)h,(j-1/2)h)` into an
[`FVImmersedBoundary2D`](@ref) with the same ghost closures, M1/M2/M3 mass
treatments, surface metadata and Jacobian pattern as the analytic circles.
A cell is solid when it lies on the solid side of any wall (a centre on a
wall is solid). The wall index of the callbacks is the position in `walls`.

Checked contract: circles of radius >= 2h; circle/domain gap >= 3h; profile
walls span the full domain width and keep >= 3h from the south and north
sides; distinct walls keep a gap >= 3h (vertical gap between profiles);
exactly one crossing on different-tag axial links and none inside same-tag
links; every cut has an opposite fluid cell inside the domain and
`abs(e dot normal) > min_transversality`. Profile walls cross the west and
east sides: their cut faces may touch those sides (M3 relaxes the closed-wall
check for them), but an axial cut whose opposite cell would lie outside the
domain is rejected, so profiles should meet the inlet/outlet with a slope
small enough to leave the first link uncut. For M3 with a nonzero wall
velocity, the default physical wall flux is midpoint quadrature along each
wall (256 points per segment or circle, error estimate from 128); prefer an
exact `wall_flux` in [`fv_steady_problem`](@ref).

Existing circle/annulus problems built by [`fv_immersed_boundary`](@ref) do
not go through this path and are unchanged.
"""
function fv_immersed_walls(Nx::Int, Ny::Int; walls, h=1.0,
        wall_velocity=(wall, x, y) -> (zero(x), zero(y)), frozen_tag=nothing,
        wall_tangent=nothing, wall_viscosity=nothing,
        min_transversality=1e-8, closure=:quadratic, mass=:M3, pressure=:linear)
    Nx > 0 && Ny > 0 || throw(ArgumentError("immersed grid dimensions must be positive"))
    isfinite(_fv_ib_primal(h)) && _fv_ib_primal(h) > 0 || throw(ArgumentError("immersed h must be finite and positive"))
    0 <= min_transversality < 1 || throw(ArgumentError("min_transversality must lie in [0,1)"))
    ws = Tuple(walls)
    isempty(ws) && throw(ArgumentError("at least one immersed wall is required"))
    all(w -> w isa FVImmersedWall, ws) || throw(ArgumentError("walls must be FVProfileWall or FVCircleWall"))
    T = promote_type(typeof(float(h)), map(_fv_wall_eltype, ws)...)
    hh = T(h)
    _fv_walls_resolution(ws, hh, Nx, Ny)
    tag = _fv_walls_tags(ws, hh, Nx, Ny, frozen_tag)
    qcut = zeros(T, 4, Nx, Ny)
    samples = Tuple[]
    cuts = Tuple[]
    WT = T
    for j in 1:Ny, i in 1:Nx, axis in 1:2
        ex, ey = axis == 1 ? (1, 0) : (0, 1)
        ni, nj = i + ex, j + ey
        ni <= Nx && nj <= Ny || continue
        x, y = (i - 1//2) * hh, (j - 1//2) * hh
        different = tag[i, j] != tag[ni, nj]
        hits, wall, cut = 0, 0, zero(T)
        for (k, w) in enumerate(ws), t in _fv_wall_roots(w, x, y, ex, ey, hh)
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
        nx, ny = _fv_wall_normal(ws[wall], wx, wy)
        transverse = abs(dx * nx + dy * ny)
        _fv_ib_primal(transverse) > min_transversality ||
            _fv_ib_error("wall $wall transversality $(_fv_ib_primal(transverse)) must exceed $min_transversality",
                         (pi, pj), (pi + dx, pj + dy))
        d = axis == 1 ? (forward ? 1 : 2) : (forward ? 3 : 4)
        qcut[d, pi, pj] = q
        uv = wall_velocity(wall, wx, wy)
        length(uv) == 2 || throw(ArgumentError("wall_velocity must return two components"))
        WT = promote_type(WT, typeof(uv[1]), typeof(uv[2]))
        push!(samples, (d, pi, pj, uv[1], uv[2]))
        push!(cuts, (d, pi, pj, wall))
    end
    wvel = zeros(WT, 2, 4, Nx, Ny)
    for (d, i, j, u, v) in samples
        wvel[1, d, i, j], wvel[2, d, i, j] = u, v
    end
    config = closure isa FVImmersedClosure ? closure : FVImmersedClosure(closure; mass, pressure)
    surface = _fv_walls_surface(qcut, cuts, ws, hh, Nx, wall_velocity, wall_tangent,
                                wall_viscosity, min_transversality)
    mass_input = config.mass == 3 ? _fv_walls_flux_quadrature(ws, Nx * hh, wall_velocity) : nothing
    return FVImmersedBoundary2D(qcut, wvel, tag, h, config, surface, mass_input)
end

function _fv_walls_surface(qcut, cuts, walls, h, Nx, velocity, tangent, viscosity, threshold)
    _, Nx, Ny = size(qcut)
    T = eltype(qcut)
    samples = Tuple[]
    for (d, i, j, k) in cuts
        q = qcut[d, i, j]
        ex, ey = (d == 1) - (d == 2), (d == 3) - (d == 4)
        x, y = (i - 1//2 + q * ex) * h, (j - 1//2 + q * ey) * h
        nx, ny = _fv_wall_normal(walls[k], x, y)
        tx, ty = -ny, nx
        wt = tangent === nothing ? _fv_surface_tangent(velocity, k, x, y, tx, ty) :
                                   tangent(k, x, y, tx, ty)
        nu = viscosity === nothing ? zero(x) : viscosity(k, x, y)
        T = promote_type(T, typeof(nx), typeof(wt[1]), typeof(wt[2]), typeof(nu))
        push!(samples, (i, j, d, k, _fv_wall_arcpos(walls[k], x, y), x, y, nx, ny, wt[1], wt[2], nu))
    end
    values = zeros(T, 8, 4, Nx, Ny)
    wall = zeros(Int32, 4, Nx, Ny)
    Lx = Nx * h
    for (k, w) in enumerate(walls)
        points = sort(filter(s -> s[4] == k, samples); by=s -> _fv_ib_primal(s[5]))
        n = length(points)
        if w isa FVCircleWall
            lo, hi, period = zero(T), zero(T), 2pi
        else
            lo, hi, period = _fv_profile_arc(w, zero(Lx)), _fv_profile_arc(w, Lx), 0
        end
        for l in 1:n
            s = points[l]
            if w isa FVCircleWall
                previous, next = points[mod1(l - 1, n)][5], points[mod1(l + 1, n)][5]
                before = s[5] - previous + (l == 1 ? period : 0)
                after = next - s[5] + (l == n ? period : 0)
                ds = w.R * (before + after) / 2
            else
                # open curve: the end points also own the stretch to the domain side
                before = l == 1 ? 2 * (s[5] - lo) : s[5] - points[l-1][5]
                after = l == n ? 2 * (hi - s[5]) : points[l+1][5] - s[5]
                ds = (before + after) / 2
            end
            i, j, d = s[1:3]
            values[:, d, i, j] .= (s[6], s[7], s[8], s[9], ds, s[10], s[11], s[12])
            wall[d, i, j] = k
        end
    end
    return FVImmersedSurface(values, wall, viscosity !== nothing, Float64(threshold),
                             any(_fv_wall_open, walls))
end

function _fv_walls_flux_quadrature(walls, Lx, velocity)
    fine = [_fv_wall_flux_integral(w, k, velocity, 256, Lx) for (k, w) in enumerate(walls)]
    coarse = [_fv_wall_flux_integral(w, k, velocity, 128, Lx) for (k, w) in enumerate(walls)]
    return _FVWallFluxQuadrature(fine, abs.(fine .- coarse))
end

# ---------------------------------------------------------------- wall heat flux
"""
    fv_immersed_wall_heat(U, prob) -> Vector

Heat leaving the fluid through the immersed walls, one entry per wall index
of the surface metadata (or one entry for raw geometry): the sum over cut
faces of exactly the residual's wall energy flux (M3-corrected upwind
convection plus the ghost-closure Dirichlet diffusion). The values are in
the units of the thermal row, built on `theta = T - T_ref`: the convective
part is measured relative to `prob.energy.Tref`, so compare it with inlet and
outlet fluxes taken with the same reference (it vanishes when the walls carry
no mass flux, the M3 `wall_flux = 0` case). With this quadrature the discrete
energy balance of the fluid (inlet, outlet, domain walls, immersed walls,
heating and sources) closes to round-off.
"""
function fv_immersed_wall_heat(U, prob::FVSteadyProblem2D)
    prob.energy === nothing && throw(ArgumentError("wall heat requires an active energy model"))
    prob.immersed === nothing && throw(ArgumentError("wall heat requires immersed geometry"))
    p = prob.nu isa Array ? prob : fv_adapt(Array, prob)
    Uh = Array(U)
    g = p.grid
    b = p.immersed
    energy, source = _fv_energy_args(p.energy)
    face_pair = _fv_face_pair(b)
    walls = b.surface === nothing ? 1 : max(1, Int(maximum(b.surface.wall)))
    heat = zeros(promote_type(eltype(Uh), _fv_immersed_eltype(b), _fv_energy_eltype(p.energy)), walls)
    for j in 1:g.Ny, i in 1:g.Nx
        p.tag[i, j] == FV_CELL_SOLID && continue
        for d in 1:4
            kind, ii, jj = _fv_face_kind(g, p.tag, i, j, d)
            kind == _FV_F_DIRICHLET && !_fv_energy_side_face(g, i, j, d) || continue
            k = b.surface === nothing ? 1 : max(1, Int(b.surface.wall[d, i, j]))
            heat[k] += _fv_energy_immersed_face(Uh, p.bvel, g, i, j, d, b.qcut, b.wvel,
                                                b.closure, face_pair, energy, source, Val(false))
        end
    end
    return heat
end
