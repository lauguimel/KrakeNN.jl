# Shape derivatives of off-grid Taylor-Couette flow, CPU Float64 Stokes.
# All geometry constructors enforce the C5 resolution contract unchanged.
using Test, Kraken, ForwardDiff, LinearAlgebra

const _C5SHAPE_GRIDS = (24,40,48,64)
const _C5SHAPE_STEPS = (1e-5,2e-6,4e-7)
const _C5SHAPE_BASE = (ri=0.14,cx=0.5113,cy=0.4937,ro=0.34,nu=1.0,omega=1.0)

function _c5shape_problem(N,theta; frozen_tag=nothing)
    ri,cx = theta
    cy,ro,nu,omega = _C5SHAPE_BASE.cy,_C5SHAPE_BASE.ro,_C5SHAPE_BASE.nu,_C5SHAPE_BASE.omega
    velocity(wall,x,y) = wall == 1 ? (-omega*(y-cy),omega*(x-cx)) : (zero(x),zero(y))
    ib = fv_immersed_annulus(N,N,cx,cy,ri,ro; h=1/N, frozen_tag,
        closure=:quadratic,mass=:M3,pressure=:linear,wall_velocity=velocity)
    return fv_steady_problem(N,N;h=1/N,immersed=ib,wall_flux=0.0,nu,conv=false)
end

function _c5shape_initial(p,theta)
    ri,cx = theta
    cy,ro,omega = _C5SHAPE_BASE.cy,_C5SHAPE_BASE.ro,_C5SHAPE_BASE.omega
    A,B = -omega*ri^2/(ro^2-ri^2),omega*ri^2*ro^2/(ro^2-ri^2)
    w(x,y) = A+B/((x-cx)^2+(y-cy)^2)
    U = fv_state(p,(x,y)->-(y-cy)*w(x,y),(x,y)->(x-cx)*w(x,y),(x,y)->0.0)
    for cell in findall(==(FV_CELL_SOLID),p.tag), comp in 1:3
        U[fv_index(p.grid,Tuple(cell)...,comp)] = 0
    end
    return U
end

# Fluid-on-inner-wall torque, per unit depth, with fluid outward normal
# pointing radially inward: tau_rtheta = -2 nu B/r^2, T = -4pi nu B.
_c5shape_torque(ri) = -4pi*_C5SHAPE_BASE.nu*_C5SHAPE_BASE.omega*ri^2*
    _C5SHAPE_BASE.ro^2/(_C5SHAPE_BASE.ro^2-ri^2)
_c5shape_dtorque(ri) = -8pi*_C5SHAPE_BASE.nu*_C5SHAPE_BASE.omega*ri*
    _C5SHAPE_BASE.ro^4/(_C5SHAPE_BASE.ro^2-ri^2)^2

function _c5shape_csv(name,rows)
    get(ENV,"KRAKEN_WRITE_C5_SHAPE_EVIDENCE","false") == "true" || return
    isempty(rows) && return
    destination = joinpath(@__DIR__,"../../benchmarks/fv_c5/shape_"*name*".csv")
    open(destination,"w") do io
        println(io,join(string.(keys(first(rows))),","))
        for row in rows
            println(io,join(string.(values(row)),","))
        end
    end
end

function _c5shape_error(f)
    try
        f()
        return nothing
    catch error
        return error
    end
end
