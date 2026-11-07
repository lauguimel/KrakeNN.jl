# C5-S3 local diagnostic fixture, not a production halfplane constructor.
using Test, LinearAlgebra, KrakeNN

const _C5M_NS = (16, 32, 64, 128)
const _C5M_ANGLES = (0, 15, 30, 45, 60, 75)
const _C5M_HALO = 5
const _C5M_ORDER_GATE = 1.8 # C5 plan, frozen before measuring these grids.

function _c5m_fields(x, y, degree, pressure_quadratic=degree>=3)
    a, b = degree >= 2, degree >= 3
    u = 0.3 + 0.4x - 0.7y + a * (0.2x^2 + 0.3x*y - 0.15y^2) + b * (0.07x^3 + 0.09x*y^2)
    v = -0.2 + 0.6x - 0.4y + a * (-0.15x^2 - 0.4x*y - 0.15y^2) - b * (0.21x^2*y + 0.03y^3)
    p = 0.2 + 0.8x + 0.5y + pressure_quadratic * (0.2x^2 + 0.15x*y + 0.25y^2)
    return u, v, p
end
function _c5m_gradient(x, y, degree)
    a, b = degree >= 2, degree >= 3
    return (0.4 + a*(0.4x + 0.3y) + b*(0.21x^2 + 0.09y^2),
            -0.7 + a*(0.3x - 0.3y) + b*0.18x*y,
            0.6 + a*(-0.3x - 0.4y) - b*0.42x*y,
            -0.4 + a*(-0.4x - 0.3y) - b*(0.21x^2 + 0.09y^2))
end
_c5m_nu(x, y) = 1 + 0.2x + 0.3y

# Continuous Stokes source grad(p) - div(nu*(G+G')), evaluated analytically.
# div(u)=0 for all three fields. NO source is derived from fv_residual.
function _c5m_source(x, y, degree, pressure_quadratic=degree>=3)
    a, b = degree >= 2, degree >= 3
    ux, uy, vx, vy = _c5m_gradient(x, y, degree)
    nu = _c5m_nu(x, y)
    sx = 0.8 + pressure_quadratic*(0.4x + 0.15y) - nu*(0.1a + 0.6b*x) - 0.4ux - 0.3*(uy + vx)
    sy = 0.5 + pressure_quadratic*(0.15x + 0.5y) - nu*(-0.6a - 0.6b*y) - 0.2*(uy + vx) - 0.6vy
    return sx, sy, 0.0
end

function _c5m_patch(N, angle, closure, mass, pressure, degree; pressure_quadratic=degree>=3)
    h, H = 1/N, _C5M_HALO
    M = N + 2H
    normal = (cosd(angle), sind(angle))
    xy(i, j) = ((i - 0.5 - H)*h, (j - 0.5 - H)*h)
    phi(x, y) = normal[1]*(x - 0.5) + normal[2]*(y - 0.5) - 0.073
    tag = fill(FV_CELL_FLUID, M, M)
    # The physical square is [0,1]^2, independent of N. A five-cell exact
    # halo surrounds it. Cap the solid halfplane outside that halo's inner
    # three layers, leaving a two-cell fluid collar at the outer grid edge.
    # The convex cap makes every O valid for the full production residual;
    # it cannot enter a radius-two stencil of a measured/solved physical cell.
    for j in 3:M-2, i in 3:M-2
        phi(xy(i,j)...) >= 0 && (tag[i,j] = FV_CELL_SOLID)
    end
    qcut, wvel = zeros(4,M,M), zeros(2,4,M,M)
    for j in 1:M, i in 1:M, d in 1:4
        tag[i,j] == FV_CELL_FLUID || continue
        ex, ey = (d == 1) - (d == 2), (d == 3) - (d == 4)
        ni, nj, oi, oj = i+ex, j+ey, i-ex, j-ey
        1 <= ni <= M && 1 <= nj <= M && tag[ni,nj] == FV_CELL_SOLID || continue
        @assert 1 <= oi <= M && 1 <= oj <= M && tag[oi,oj] == FV_CELL_FLUID
        x, y = xy(i,j)
        en = ex*normal[1] + ey*normal[2]
        # Entering the convex solid requires both the halfplane and cap.
        qp = phi(x,y) < 0 ? -phi(x,y)/(h*en) : 0.0
        qc = (i < 3 || i > M-2 || j < 3 || j > M-2) ? 0.5 : 0.0
        q = max(qp, qc)
        @assert 0 < q <= 1
        qcut[d,i,j] = q
        uw, vw, _ = _c5m_fields(x+q*h*ex, y+q*h*ey, degree, pressure_quadratic)
        wvel[:,d,i,j] .= (uw,vw)
    end
    ib = FVImmersedBoundary2D(qcut,wvel,tag,h,FVImmersedClosure(closure;mass,pressure))
    shift = H*h
    p = fv_steady_problem(M,M; h, immersed=ib, conv=false,
        nu=(x,y)->_c5m_nu(x-shift,y-shift),
        boundary_velocity=(s,x,y)->_c5m_fields(x-shift,y-shift,degree,pressure_quadratic)[1:2],
        source=(x,y)->_c5m_source(x-shift,y-shift,degree,pressure_quadratic))
    U = fv_state(p, ((x,y)->_c5m_fields(x-shift,y-shift,degree,pressure_quadratic)[c] for c in 1:3)...)
    cells = [(i,j) for j in H+1:H+N for i in H+1:H+N if p.tag[i,j] != FV_CELL_SOLID]
    @assert all(c -> p.tag[c...] == FV_CELL_FLUID, cells) # pressure gauge is fixed by halo
    cut = [c for c in cells if any(>(0),view(qcut,:,c...))]
    interior = [c for c in cells if all(p.tag[c[1]+di,c[2]+dj] != FV_CELL_SOLID
                  for di in -2:2 for dj in -2:2)]
    near = setdiff(cells,union(cut,interior))
    links = [(i,j,d) for (i,j) in cells for d in 1:4 if qcut[d,i,j] > 0]
    @assert !isempty(cut) && !isempty(interior) && !isempty(links)
    @assert all(abs(phi((xy(i,j) .+ (qcut[d,i,j]*h*((d==1)-(d==2)),
        qcut[d,i,j]*h*((d==3)-(d==4))))...)) < 1e-12 for (i,j,d) in links)
    idx = [fv_index(p.grid,i,j,c) for (i,j) in cells for c in 1:3]
    return (; p,U,cells,cut,interior,near,links,idx,normal,xy,N,angle,degree,pressure_quadratic)
end

_c5m_maxabs(a) = maximum(abs,a;init=0.0)
function _c5m_residual_metrics(f, U)
    R = fv_residual(U,f.p)
    # FV residuals are integrated over h^2; divide by h^2 to compare local
    # differential truncation. Cut, near, and strict interior partition all
    # measured fluid cells. No cut link is removed for its q or orientation.
    value(cells, comps) = maximum((abs(R[fv_index(f.p.grid,i,j,c)])/f.p.grid.h^2
        for (i,j) in cells for c in comps);init=0.0)
    return (; trunc_cut_momentum=value(f.cut,1:2),trunc_cut_mass=value(f.cut,3:3),
        trunc_near_momentum=value(f.near,1:2),trunc_near_mass=value(f.near,3:3),
        trunc_interior_momentum=value(f.interior,1:2),trunc_interior_mass=value(f.interior,3:3))
end

function _c5m_reconstruction_metrics(f,U)
    p, ib, g = f.p, f.p.immersed, f.p.grid
    ghost, face, nominal_pressure, cell_gradient = 0.0,0.0,0.0,0.0
    surface_gradient, surface_pressure = 0.0,0.0
    pressure2, viscous2, traction2 = 0.0,0.0,0.0
    pressureinf, viscousinf, tractioninf = 0.0,0.0,0.0
    for (i,j,d) in f.links
        x,y = f.xy(i,j)
        ex,ey = (d==1)-(d==2),(d==3)-(d==4)
        q = ib.qcut[d,i,j]
        xw,yw = x+q*g.h*ex,y+q*g.h*ey
        w = KrakeNN._fv_wall_data(U,p.bvel,g,i,j,d,ib.qcut,ib.wvel,ib.closure)
        eg = _c5m_fields(x+g.h*ex,y+g.h*ey,f.degree,f.pressure_quadratic)
        ef = _c5m_fields(x+g.h*ex/2,y+g.h*ey/2,f.degree,f.pressure_quadratic)
        ghost = max(ghost,abs(w[3]-eg[1]),abs(w[4]-eg[2]))
        face = max(face,abs(w[6]-ef[1]),abs(w[7]-ef[2]))
        nominal_pressure = max(nominal_pressure,abs(w[5]-ef[3]))
        cg = KrakeNN._fv_cell_grad(U,p.tag,p.bvel,g,i,j,ib.qcut,ib.wvel,ib.closure)
        cell_gradient = max(cell_gradient,_c5m_maxabs(cg[1:4] .- _c5m_gradient(x,y,f.degree)))
        G = _c5m_gradient(xw,yw,f.degree)
        nx,ny = f.normal
        dt = (-ny*G[1]+nx*G[2],-ny*G[3]+nx*G[4])
        nuw = _c5m_nu(xw,yw)
        s = KrakeNN._fv_surface_reconstruction(U,p,i,j,d,f.normal,dt,nuw)
        surface_gradient = max(surface_gradient,_c5m_maxabs(s.gradient .- G))
        pw = _c5m_fields(xw,yw,f.degree,f.pressure_quadratic)[3]
        surface_pressure = max(surface_pressure,abs(s.pressure-pw))
        pe = (pw*nx,pw*ny)
        ve = (-nuw*(2G[1]*nx+(G[2]+G[3])*ny),-nuw*((G[2]+G[3])*nx+2G[4]*ny))
        ep = sum(abs2,s.pressure_traction .- pe)
        ev = sum(abs2,s.viscous_traction .- ve)
        et = sum(abs2,s.traction .- (pe .+ ve))
        pressure2+=ep; viscous2+=ev; traction2+=et
        pressureinf=max(pressureinf,sqrt(ep)); viscousinf=max(viscousinf,sqrt(ev)); tractioninf=max(tractioninf,sqrt(et))
    end
    nl = length(f.links)
    velocity = maximum((abs(U[fv_index(g,i,j,c)]-f.U[fv_index(g,i,j,c)]) for (i,j) in f.cells for c in 1:2);init=0.0)
    pressure = maximum((abs(U[fv_index(g,i,j,3)]-f.U[fv_index(g,i,j,3)]) for (i,j) in f.cells);init=0.0)
    return (; ghost,face,nominal_pressure,cell_gradient,surface_gradient,surface_pressure,
        traction_rms=sqrt(traction2/nl),traction_linf=tractioninf,
        pressure_traction_rms=sqrt(pressure2/nl),pressure_traction_linf=pressureinf,
        viscous_traction_rms=sqrt(viscous2/nl),viscous_traction_linf=viscousinf,velocity,pressure)
end

# Linear Stokes system restricted to physical-square fluid unknowns. All halo
# velocity AND pressure remain exact. This isolates wall closure behaviour,
# not a global closed-domain flow or a pressure-pin/mass compatibility test.
function _c5m_solve(f,factor)
    U = copy(f.U)
    U[f.idx] .-= factor \ fv_residual(U,f.p)[f.idx]
    return U
end

function _c5m_triplet(N,angle,closure,mass,pressure)
    fs = [_c5m_patch(N,angle,closure,mass,pressure,k) for k in 1:3]
    f = fs[1]
    s = fv_newton_solver(f.p)
    J = fv_jacobian!(s,f.U)
    factor = lu(J[f.idx,f.idx]) # Same linear operator for all three sources.
    rows = NamedTuple[]
    for f in fs
        injected = _c5m_residual_metrics(f,f.U)
        for state in (:injected,:solved)
            U = state == :injected ? f.U : _c5m_solve(f,factor)
            measured = _c5m_reconstruction_metrics(f,U)
            rinf = maximum(abs,fv_residual(U,f.p)[f.idx])/f.p.grid.h^2
            push!(rows,(;N,angle,closure,mass,residual_pressure=pressure,degree=f.degree,state,
                cells=length(f.cells),cut_cells=length(f.cut),interior_cells=length(f.interior),
                near_cells=length(f.near),links=length(f.links),
                qmin=minimum(f.p.immersed.qcut[d,i,j] for (i,j,d) in f.links),
                injected...,measured...,equation_residual=rinf))
        end
    end
    return rows
end
