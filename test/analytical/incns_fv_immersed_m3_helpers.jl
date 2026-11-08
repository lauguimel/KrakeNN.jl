# M3 test-only references. Flux sums below independently expand the L/Q ghost;
# they do not call the production wall-flux or pair-row implementations.
using Test, KrakeNN, ForwardDiff, LinearAlgebra, Random
using KrakeNN.SparseArrays: nnz

function _c5m3_problem(shape; closure=:quadratic, conv=false, radial=false, mass=:M3)
    velocity(k,x,y) = radial ? (0.17*(x-10.13)/((x-10.13)^2+(y-9.87)^2),
                                0.17*(y-9.87)/((x-10.13)^2+(y-9.87)^2)) :
                               (0.03x+0.02y,-0.01x-0.03y)
    N = shape == :circle ? 12 : 20
    ib = shape == :circle ? fv_immersed_circle(N,N,6.13,5.87,2.21;
        h=1.0,closure,mass,wall_velocity=velocity) :
        fv_immersed_annulus(N,N,10.13,9.87,2.6,6.2;
            h=1.0,closure,mass,wall_velocity=velocity)
    mass_kw = mass == :M3 ? (;wall_flux=0.0) : (;)
    p = fv_steady_problem(N,N;h=1.0,immersed=ib,conv,mass_kw...)
    if radial
        # Body numbering follows solid components, not analytic wall numbering.
        m = p.immersed.mass
        Q = [begin
            face = first(findall(==(k),m.face_pair))
            p.immersed.surface.wall[face] == 1 ? -2pi*0.17 : 2pi*0.17
        end for k in eachindex(m.Q)]
        p = fv_steady_problem(N,N;h=1.0,immersed=ib,conv,wall_flux=Q)
    end
    return p
end

function _c5m3_raw_flux(U,p)
    g,ib = p.grid,p.immersed
    flux = zeros(eltype(U),length(ib.mass.Q))
    absolute = zero(eltype(U))
    for j in 1:g.Ny,i in 1:g.Nx,d in 1:4
        q = ib.qcut[d,i,j]
        q > 0 || continue
        ex,ey = (d==1)-(d==2),(d==3)-(d==4)
        c = d <= 2 ? 1 : 2
        uO,uP = U[fv_index(g,i-ex,j-ey,c)],U[fv_index(g,i,j,c)]
        uw = ib.wvel[c,d,i,j]
        ghost = ib.closure.ghost == 0x01 ? uw+(uw-uP)*(1-q)/q :
            (1-q)/(1+q)*uO-2(1-q)/q*uP+2/(q*(1+q))*uw
        F = g.h*(ex+ey)*(uP+ghost)/2
        flux[ib.mass.face_pair[d,i,j]] += F
        absolute += abs(F)
    end
    return flux,absolute
end

function _c5m3_external_flux(p)
    g,total = p.grid,0.0
    for j in 1:g.Ny,i in 1:g.Nx,d in 1:4
        p.tag[i,j] == FV_CELL_SOLID && continue
        ex,ey = (d==1)-(d==2),(d==3)-(d==4)
        (1 <= i+ex <= g.Nx && 1 <= j+ey <= g.Ny) && continue
        component,l = d <= 2 ? (1,j) : (2,i)
        total += g.h*(ex+ey)*p.bvel[component,l,d]
    end
    return total
end

function _c5m3_restore(p)
    tag = copy(p.tag)
    tag[tag .== FV_CELL_PINNED] .= FV_CELL_FLUID
    return FVSteadyProblem2D(p.grid,tag,p.nu,p.bvel,p.src,p.visc,p.immersed)
end

function _c5m3_csv(name,rows)
    get(ENV,"KRAKEN_WRITE_C5_M3_EVIDENCE","false") == "true" || return
    destination = joinpath(@__DIR__,"../../benchmarks/fv_c5",name*".csv")
    open(destination,"w") do io
        println(io,join(string.(keys(first(rows))),","))
        for row in rows
            println(io,join(string.(values(row)),","))
        end
    end
end
