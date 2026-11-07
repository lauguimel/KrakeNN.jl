# Local physical-surface reconstruction, separate from nominal FV face fluxes.
# Analytic arc quadrature is lowered with the geometry, including parameter duals.

struct FVImmersedSurface{A,B}
    values::A                # x,y,nx,ny,arc weight,du/dt,dv/dt,physical viscosity
    wall::B
    given_viscosity::Bool
    min_transversality::Float64
end

# This directional derivative is the innermost differentiation, outside
# parameter duals in the number representation. Explicit tag ordering avoids
# ForwardDiff's ambiguity between its generated tags and FVParamTag.
struct _FVSurfaceTangentTag end
ForwardDiff.:(≺)(::Type{T}, ::Type{_FVSurfaceTangentTag}) where {T} = true
ForwardDiff.:(≺)(::Type{_FVSurfaceTangentTag}, ::Type{T}) where {T} = false
ForwardDiff.:(≺)(::Type{_FVSurfaceTangentTag}, ::Type{_FVSurfaceTangentTag}) = false
_fv_surface_partial(x) = zero(x)
_fv_surface_partial(x::ForwardDiff.Dual{_FVSurfaceTangentTag}) = ForwardDiff.partials(x)[1]
function _fv_surface_tangent(velocity, k, x, y, tx, ty)
    xd = ForwardDiff.Dual{_FVSurfaceTangentTag}(x, tx)
    yd = ForwardDiff.Dual{_FVSurfaceTangentTag}(y, ty)
    uv = velocity(k, xd, yd)
    return _fv_surface_partial(uv[1]), _fv_surface_partial(uv[2])
end

"""Complete outward nominal-wall momentum and mass flux used by the residual.

Pressure, viscous stress and upwind convection use precisely the residual
closure. The physical-cut traction is deliberately a separate reconstruction.
"""
@inline function _fv_wall_flux_components(U, tag, nu, bvel, g, i, j, d, visc,
                                        qcut, wvel, closure, gP, face_pair = nothing)
    h = g.h
    isx, sx = d <= 2, (d == 1 || d == 3) ? 1 : -1
    uw, vw, ua, va, pf, uf, vf, dun, dvn =
        _fv_wall_data(U, bvel, g, i, j, d, qcut, wvel, closure)
    consistent = _fv_immersed_face(g, qcut, i, j, d) && closure.mass != 2
    mass = h * sx * (consistent ? (isx ? uf : vf) : (isx ? uw : vw))
    mass = _fv_corrected_wall_mass(mass, U, g, i, j, d, face_pair)
    gF = _fv_wall_gradient(U, tag, bvel, g, i, j, d, gP, qcut, wvel, closure)
    tx, ty = _fv_wall_stress(nu, visc, g, i, j, d, gF, dun, dvn, qcut, closure)
    au = g.conv * (_fv_posp(mass) * _fv_val(U, g, i, j, _fv_velocity_slot(g.layout, 1)) + _fv_negp(mass) * ua)
    av = g.conv * (_fv_posp(mass) * _fv_val(U, g, i, j, _fv_velocity_slot(g.layout, 2)) + _fv_negp(mass) * va)
    return au, av, tx, ty, pf, mass
end

@inline function _fv_wall_momentum_flux(U, tag, nu, bvel, g, i, j, d, visc,
                                        qcut, wvel, closure, gP, face_pair = nothing)
    au, av, tx, ty, pf, mass = _fv_wall_flux_components(U, tag, nu, bvel, g, i, j, d,
                                                       visc, qcut, wvel, closure, gP, face_pair)
    h, sx, isx = g.h, (d == 1 || d == 3) ? 1 : -1, d <= 2
    return au - h * tx + (isx ? h * sx * pf : zero(pf)),
           av - h * ty + (isx ? zero(pf) : h * sx * pf), mass
end
_fv_adapt_surface(to, ::Nothing) = nothing
_fv_adapt_surface(to, s::FVImmersedSurface) =
    FVImmersedSurface(to(s.values), to(s.wall), s.given_viscosity, s.min_transversality)

function _fv_immersed_surface(qcut, circles, annular, h, velocity, tangent, viscosity, threshold)
    _, Nx, Ny = size(qcut)
    samples = Tuple[]
    T = eltype(qcut)
    for j in 1:Ny, i in 1:Nx, d in 1:4
        q = qcut[d, i, j]
        _fv_ib_primal(q) > 0 || continue
        ex, ey = (d == 1) - (d == 2), (d == 3) - (d == 4)
        x, y = (i - 1//2 + q * ex) * h, (j - 1//2 + q * ey) * h
        k = argmin([abs(_fv_ib_primal(_fv_ib_phi(x, y, c))) for c in circles])
        c = circles[k]
        sign = annular && k == 2 ? 1 : -1
        nx, ny = sign * (x - c[1]) / c[3], sign * (y - c[2]) / c[3]
        tx, ty = -ny, nx
        wt = tangent === nothing ? _fv_surface_tangent(velocity, k, x, y, tx, ty) :
                                   tangent(k, x, y, tx, ty)
        nu = viscosity === nothing ? zero(x) : viscosity(k, x, y)
        angle = atan(y - c[2], x - c[1])
        T = promote_type(T, typeof(wt[1]), typeof(wt[2]), typeof(nu))
        push!(samples, (i, j, d, k, angle, x, y, nx, ny, wt[1], wt[2], nu))
    end
    values = zeros(T, 8, 4, Nx, Ny)
    wall = zeros(Int32, 4, Nx, Ny)
    for k in eachindex(circles)
        points = sort(filter(s -> s[4] == k, samples); by=s -> _fv_ib_primal(s[5]))
        n = length(points)
        for l in 1:n
            s = points[l]
            previous, next = points[mod1(l - 1, n)][5], points[mod1(l + 1, n)][5]
            before = s[5] - previous + (l == 1 ? 2pi : 0)
            after = next - s[5] + (l == n ? 2pi : 0)
            ds = circles[k][3] * (before + after) / 2
            i, j, d = s[1:3]
            values[:, d, i, j] .= (s[6], s[7], s[8], s[9], ds, s[10], s[11], s[12])
            wall[d, i, j] = k
        end
    end
    return FVImmersedSurface(values, wall, viscosity !== nothing, Float64(threshold))
end

"""
    _fv_surface_reconstruction(U, prob, i, j, d, normal, wall_tangent,
                               physical_nu; min_transversality=1e-8)

Reconstruct physical traction at the actual immersed cut, using O/P/wall
quadratic axial differentiation (R10). `normal` is the unit normal from fluid
to solid, `t = (-normal[2], normal[1])`, and `wall_tangent` contains the two
analytical Dirichlet derivatives along t at that cut. The normal derivative
is `(axial - (e dot t)*wall_tangent)/(e dot normal)`; no cut is replaced or
merged. A nonpositive threshold or a nearly tangent link raises an error
identifying the fluid cell and its solid neighbour.

`physical_nu` is the given viscosity evaluated at the cut, not a cell or
nominal-face value. A GNF model instead evaluates its law on the reconstructed
wall gradient. Pressure always uses `pP + q*(pP-pO)`, independently of the
residual's pressure closure. The quadratic derivative is also independent of
the residual's L/Q ghost choice.

Returns the full gradient `(ux,uy,vx,vy)`, wall pressure/viscosity, and the
pointwise fluid-on-wall traction `p*n - nu*(G+G')*n`, with pressure and viscous
parts separate. No quadrature weights or nominal-face fluxes are included.
With injected exact quadratic velocities the gradient is exact up to roundoff;
that identity does not establish accuracy for a solved velocity field.
"""
function _fv_surface_reconstruction(U, prob, i, j, d, normal, wall_tangent,
                                    physical_nu; min_transversality=1e-8)
    0 < min_transversality < 1 ||
        throw(ArgumentError("surface min_transversality must lie in (0,1); cell ($i, $j)"))
    1 <= d <= 4 || throw(ArgumentError("surface direction must lie in 1:4; cell ($i, $j)"))
    ib = prob.immersed
    ib === nothing && throw(ArgumentError("surface reconstruction needs immersed geometry; cell ($i, $j)"))
    g = prob.grid
    ex, ey = (d == 1) - (d == 2), (d == 3) - (d == 4)
    cells = ((i, j), (i + ex, j + ey))
    q = ib.qcut[d, i, j]
    0 < _fv_ib_primal(q) <= 1 || _fv_ib_error("surface cut fraction must lie in (0,1]", cells...)
    nx, ny = normal
    abs(_fv_ib_primal(nx^2 + ny^2) - 1) <= 1e-10 ||
        _fv_ib_error("surface normal must have unit length", cells...)
    en = ex * nx + ey * ny
    abs(_fv_ib_primal(en)) > min_transversality ||
        _fv_ib_error("surface transversality $(_fv_ib_primal(abs(en))) must exceed $min_transversality", cells...)
    oi, oj = i - ex, j - ey
    1 <= oi <= g.Nx && 1 <= oj <= g.Ny && prob.tag[oi, oj] != FV_CELL_SOLID ||
        _fv_ib_error("surface opposite cell must be fluid", (i, j), (oi, oj))
    tx, ty = -ny, nx
    et = ex * tx + ey * ty
    axial(c) = (q / (1 + q) * _fv_val(U, g, oi, oj, _fv_velocity_slot(g.layout, c)) -
        (1 + q) / q * _fv_val(U, g, i, j, _fv_velocity_slot(g.layout, c)) +
        (1 + 2q) / (q * (1 + q)) * ib.wvel[c, d, i, j]) / g.h
    un = (axial(1) - et * wall_tangent[1]) / en
    vn = (axial(2) - et * wall_tangent[2]) / en
    ux, uy = un * nx + wall_tangent[1] * tx, un * ny + wall_tangent[1] * ty
    vx, vy = vn * nx + wall_tangent[2] * tx, vn * ny + wall_tangent[2] * ty
    nuw = _fv_wall_nu(prob.visc, physical_nu, ux, uy, vx, vy, i, j)
    pP = _fv_val(U, g, i, j, _fv_pressure_slot(g.layout))
    pw = pP + q * (pP - _fv_val(U, g, oi, oj, _fv_pressure_slot(g.layout)))
    pressure = (pw * nx, pw * ny)
    viscous = (-nuw * (2ux * nx + (uy + vx) * ny),
               -nuw * ((uy + vx) * nx + 2vy * ny))
    traction = (pressure[1] + viscous[1], pressure[2] + viscous[2])
    return (; gradient=(ux, uy, vx, vy), pressure=pw, viscosity=nuw,
            pressure_traction=pressure, viscous_traction=viscous, traction)
end
