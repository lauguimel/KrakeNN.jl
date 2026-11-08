# Open-channel C5 references: identical physical geometry and sampled inflow
# for M1/M3. All fluxes are outward from the fluid; inlet flux is negative.
using KrakeNN, Test, LinearAlgebra

function _c5channel_problem(N,mass)
    h,Nx,flow = 1/N,2N,0.1
    shape(y) = 6y*(1-y)
    sampled = h*sum(shape((j-0.5)*h) for j in 1:N)
    inlet(y) = flow*shape(y)/sampled
    ib = fv_immersed_circle(Nx,N,0.513,0.487,0.16;h,
        closure=:quadratic,mass,pressure=:linear)
    kw = mass == :M3 ? (;wall_flux=0.0) : (;)
    p = fv_steady_problem(Nx,N;h,nu=1.0,conv=false,immersed=ib,
        bc=(W=:inlet,E=:outlet,S=:wall,N=:wall),
        boundary_velocity=(s,x,y)->s==:W ? (inlet(y),0.0) : (0.0,0.0),kw...)
    return (;p,N,mass,flow,sampled)
end

function _c5channel_wall_flux(U,p)
    g,ib = p.grid,p.immersed
    raw,corrected,absolute = 0.0,0.0,0.0
    for j in 1:g.Ny,i in 1:g.Nx,d in 1:4
        q = ib.qcut[d,i,j]
        q > 0 || continue
        ex,ey = (d==1)-(d==2),(d==3)-(d==4)
        c = d <= 2 ? 1 : 2
        uO,uP = U[fv_index(g,i-ex,j-ey,c)],U[fv_index(g,i,j,c)]
        uw = ib.wvel[c,d,i,j]
        # Exact expansion of Q ghost evaluated at the nominal face.
        F = g.h*(ex+ey)*((1-q)/(2(1+q))*uO+(3q-2)/(2q)*uP+uw/(q*(1+q)))
        raw += F
        if ib.closure.mass == 0x03
            F -= g.h*U[3g.Nx*g.Ny+ib.mass.face_pair[d,i,j]]
        end
        corrected += F
        absolute += abs(F)
    end
    return (;raw,corrected,absolute)
end

function _c5channel_pressure(U,p,x)
    g = p.grid
    left = floor(Int,x/g.h+0.5)
    fraction = (x-(left-0.5)*g.h)/g.h
    1 <= left < g.Nx || error("pressure section outside interpolation domain")
    all(p.tag[i,j] != FV_CELL_SOLID for i in (left,left+1),j in 1:g.Ny) ||
        error("pressure section crosses the immersed solid")
    return sum((1-fraction)*U[fv_index(g,left,j,3)]+
        fraction*U[fv_index(g,left+1,j,3)] for j in 1:g.Ny)/g.Ny
end

function _c5channel_solve(f)
    p,g = f.p,f.p.grid
    solver = fv_newton_solver(p)
    U = zeros(fv_nunknowns(p))
    status = fv_newton!(solver,U;atol=2e-12,rtol=1e-13,maxit=4)
    R = fv_residual(U,p)
    cells = findall(!=(FV_CELL_SOLID),p.tag)
    continuity = [fv_index(g,Tuple(c)...,3) for c in cells]
    wall = _c5channel_wall_flux(U,p)
    inlet = -g.h*sum(p.bvel[1,j,2] for j in 1:g.Ny)
    outlet = g.h*sum(U[fv_index(g,g.Nx,j,1)] for j in 1:g.Ny)
    ncell = 3g.Nx*g.Ny
    pair_residual = length(R)==ncell ? 0.0 : norm(R[ncell+1:end],Inf)
    metrics = (;N=f.N,mass=f.mass,h=g.h,Nx=g.Nx,Ny=g.Ny,
        inlet_target=f.flow,sampled_shape_flux=f.sampled,
        inlet_flux=inlet,outlet_flux=outlet,inlet_minus_outlet=-inlet-outlet,
        wall_flux=wall.corrected,
        uncorrected_wall_flux=wall.raw,wall_absolute=wall.absolute,
        global_balance=inlet+outlet+wall.corrected,
        continuity_sum=sum(R[continuity]),continuity_absolute=sum(abs,R[continuity]),
        pressure_drop=_c5channel_pressure(U,p,0.25)-_c5channel_pressure(U,p,1.75),
        drag_surface=fv_objective(FVWallForce(1;kind=:surface),U,p),
        drag_residual_flux=fv_objective(FVWallForce(1;kind=:residual_flux),U,p),
        lift_surface=fv_objective(FVWallForce(2;kind=:surface),U,p),
        lift_residual_flux=fv_objective(FVWallForce(2;kind=:residual_flux),U,p),
        residual=norm(R,Inf),normalized_residual=norm(R,Inf)/g.h^2,
        pair_residual,lambda=length(U)==ncell ? 0.0 : U[end],
        unknowns=length(U),colors=solver.asm.ncolors,iterations=status.iterations)
    return (;U,status,metrics)
end

# Richardson's three-grid ratio for unequal spacings. The p=0 limiting
# expression avoids a removable 0/0. No presumed order is imposed on data.
function _c5channel_richardson(h,values)
    d12,d23 = values[1]-values[2],values[2]-values[3]
    d12*d23 > 0 || return (;order=NaN,extrapolated=NaN,status="nonmonotonic")
    ratio = d12/d23
    a,b = log(h[1]/h[2]),log(h[3]/h[2])
    model(p) = abs(p) < 1e-10 ? -a/b : expm1(p*a)/(-expm1(p*b))
    left,right = -20.0,20.0
    low,high = model(left)-ratio,model(right)-ratio
    while low*high > 0 && right < 320
        left *= 2
        right *= 2
        low,high = model(left)-ratio,model(right)-ratio
    end
    low*high <= 0 || return (;order=NaN,extrapolated=NaN,status="unbracketed")
    for iteration in 1:100
        middle = (left+right)/2
        fm = model(middle)-ratio
        if low*fm <= 0
            right = middle
        else
            left,low = middle,fm
        end
    end
    order = (left+right)/2
    ratio23 = (h[3]/h[2])^order
    extrapolated = abs(1-ratio23) <= 1e-12 ? NaN :
        (values[3]-ratio23*values[2])/(1-ratio23)
    return (;order,extrapolated,status="estimated")
end

function _c5channel_csv(name,rows)
    get(ENV,"KRAKEN_WRITE_C5_CHANNEL_EVIDENCE","false") == "true" || return
    path = joinpath(@__DIR__,"../../benchmarks/fv_c5",name*".csv")
    open(path,"w") do io
        println(io,join(string.(keys(first(rows))),","))
        for row in rows
            println(io,join(string.(values(row)),","))
        end
    end
end
