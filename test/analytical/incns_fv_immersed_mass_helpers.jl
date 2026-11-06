# Closed-domain mass diagnostics, Float64 Stokes. The historical default is
# Q/M1/linear-pressure; mass=:M3 evaluates the augmented compatibility system.
# No mass-source correction or projection is applied. Diagnostics restore
# the continuity equation replaced by the pressure pin.
using Kraken
using LinearAlgebra
using KernelAbstractions

function _c5mass_fields(case, x, y, cx, cy, ri, ro)
    X, Y = x-cx, y-cy
    s = X^2 + Y^2
    if case == :couette
        A, B = -ri^2/(ro^2-ri^2), ri^2*ro^2/(ro^2-ri^2)
        w = A + B/s
        return (-Y*w, X*w, 0.0, 0.0, 0.0)
    end
    # psi = (r^2-ri^2)(r^2-ro^2)(1+aX+bY+cXY)/ro^4.
    # u=psi_y, v=-psi_x: exactly divergence-free and tangent to both walls.
    # The affine/bilinear modulation gives a non-rigid, asymmetric wall speed.
    a, b, c = 0.4, 0.7, 0.3
    A, B = ri^2+ro^2, ri^2*ro^2
    f, fp, F = s^2-A*s+B, 2s-A, 1+a*X+b*Y+c*X*Y
    Fx, Fy, D = a+c*Y, b+c*X, a*X+b*Y+2c*X*Y
    scale = inv(ro^4)
    u, v = scale*(2Y*fp*F+f*Fy), -scale*(2X*fp*F+f*Fx)
    # Laplacian(psi)=(16r^2-4A)F+4(2r^2-A)D.
    lapu = scale*(32Y*F+(16s-4A)*Fy+16Y*D+4fp*(b+2c*X))
    lapv = -scale*(32X*F+(16s-4A)*Fx+16X*D+4fp*(a+2c*Y))
    pressure = X+0.3Y+X*Y
    return (u, v, pressure, 1+Y-lapu, 0.3+X-lapv)
end

function _c5mass_problem(N, case; pin=:first, mass=:M1)
    h, cx, cy, ri, ro = 1/N, 0.5113, 0.4937, 0.14, 0.34
    fields(x,y) = _c5mass_fields(case,x,y,cx,cy,ri,ro)
    ib = fv_immersed_annulus(N,N,cx,cy,ri,ro; h,
        closure=:quadratic, mass, pressure=:linear,
        wall_velocity=(wall,x,y)->fields(x,y)[1:2])
    mass_kw = mass == :M3 ? (;wall_flux=0.0) : (;)
    p = fv_steady_problem(N,N; h, immersed=ib, nu=1.0, conv=false,
        source=(x,y)->(fields(x,y)[4],fields(x,y)[5],0.0), mass_kw...)
    cells = findall(!=(FV_CELL_SOLID),p.tag)
    oldpin = only(findall(==(FV_CELL_PINNED),p.tag))
    pinpos = pin == :first ? first(cells) : last(cells)
    p.tag[oldpin] = FV_CELL_FLUID
    p.tag[pinpos] = FV_CELL_PINNED
    pressure_shift = fields((pinpos[1]-0.5)*h,(pinpos[2]-0.5)*h)[3]
    exact = fv_state(p,(x,y)->fields(x,y)[1],(x,y)->fields(x,y)[2],
        (x,y)->fields(x,y)[3]-pressure_shift)
    for idx in findall(==(FV_CELL_SOLID),p.tag), comp in 1:3
        exact[fv_index(p.grid,Tuple(idx)...,comp)] = 0
    end
    return (;p,exact,cells,pinpos,case,N,cx,cy,ri,ro,mass)
end

# Expanding (uP+uG)/2 for the Q ghost gives the following independent
# analytic discrete wall sum:
# B(U) = h sum_cut e . [ (1-q)/(2(1+q)) uO
#                      +(3q-2)/(2q) uP + uw/(q(1+q)) ].
# Every interior flux occurs twice with opposite signs, including Rhie-Chow.
# Therefore sum_cells Rc = B(U)-h^2 sum_cells s_mass. For a converged pinned
# solve, B(U) equals the omitted pin equation up to the solve residual.
function _c5mass_wall_sum(U,p)
    g,ib = p.grid,p.immersed
    total, absolute = 0.0,0.0
    for j in 1:g.Ny, i in 1:g.Nx, d in 1:4
        q = ib.qcut[d,i,j]
        q > 0 || continue
        ex,ey = (d==1)-(d==2),(d==3)-(d==4)
        c = d <= 2 ? 1 : 2
        uO = U[fv_index(g,i-ex,j-ey,c)]
        uP = U[fv_index(g,i,j,c)]
        uw = ib.wvel[c,d,i,j]
        flux = g.h*(ex+ey)*((1-q)/(2(1+q))*uO+(3q-2)/(2q)*uP+uw/(q*(1+q)))
        if ib.closure.mass == 0x03
            pair = ib.mass.face_pair[d,i,j]
            flux -= g.h * U[3g.Nx*g.Ny+pair]
        end
        total += flux
        absolute += abs(flux)
    end
    return total,absolute
end

function _c5mass_metrics(f,U)
    p,g,ib = f.p,f.p.grid,f.p.immersed
    tag = copy(p.tag)
    tag[f.pinpos] = FV_CELL_FLUID
    restored = FVSteadyProblem2D(g,tag,p.nu,p.bvel,p.src,p.visc,ib)
    R = fv_residual(U,restored)
    rows = [fv_index(g,Tuple(c)...,3) for c in f.cells]
    wall,wall_absolute = _c5mass_wall_sum(U,p)
    sum_mass = sum(p.src[rows])*g.h^2
    global_balance = wall-sum_mass
    pin_flux = R[fv_index(g,Tuple(f.pinpos)...,3)]
    velocity_rows = [fv_index(g,Tuple(c)...,k) for c in f.cells for k in 1:2]
    velocity_error = norm(U[velocity_rows]-f.exact[velocity_rows],Inf)
    velocity_rms = sqrt(sum(abs2,U[velocity_rows]-f.exact[velocity_rows])/length(velocity_rows))
    pressure_rows = [fv_index(g,Tuple(c)...,3) for c in f.cells]
    pressure_difference = U[pressure_rows]-f.exact[pressure_rows]
    pressure_difference .-= sum(pressure_difference)/length(pressure_difference)
    pressure_error = norm(pressure_difference,Inf)
    return (;global_balance,pin_flux,pin_divergence=pin_flux/g.h^2,
        sum_continuity=sum(R[rows]),wall_absolute,sum_mass,velocity_error,velocity_rms,
        pressure_error,
        residual=norm(fv_residual(U,p),Inf)/g.h^2)
end

function _c5mass_solve(f)
    U = copy(f.exact)
    @assert KernelAbstractions.get_backend(U) isa KernelAbstractions.CPU
    solver = fv_newton_solver(f.p)
    status = fv_newton!(solver,U; atol=2e-12,rtol=1e-13,maxit=4)
    return (;U,status,metrics=_c5mass_metrics(f,U))
end

function _c5mass_pin_difference(f,a,b)
    g = f.p.grid
    velocity_rows = [fv_index(g,Tuple(c)...,k) for c in f.cells for k in 1:2]
    pressure_rows = [fv_index(g,Tuple(c)...,3) for c in f.cells]
    dp = a[pressure_rows]-b[pressure_rows]
    dp .-= sum(dp)/length(dp) # Only gauge alignment, never mass projection.
    return (;velocity=norm(a[velocity_rows]-b[velocity_rows],Inf),pressure=norm(dp,Inf))
end

_c5mass_order(a,b,Na,Nb) = log(abs(a/b))/log(Nb/Na)
