# A0 local aligned-wall reference; no production geometry or residual changes.
include(joinpath(@__DIR__, "incns_fv_immersed_mms_helpers.jl"))
import KernelAbstractions

function _c5a_patch(N, pressure_quadratic)
    @assert iseven(N)
    h, H = 1/N, _C5M_HALO
    Nx, Ny = H + N÷2, N + 2H
    shift = H*h
    xy(i,j) = ((i-0.5-H)*h, (j-0.5-H)*h)
    p = fv_steady_problem(Nx,Ny; h, conv=false,
        nu=(x,y)->_c5m_nu(x-shift,y-shift),
        boundary_velocity=(s,x,y)->_c5m_fields(x-shift,y-shift,3,pressure_quadratic)[1:2],
        source=(x,y)->_c5m_source(x-shift,y-shift,3,pressure_quadratic))
    U = fv_state(p, ((x,y)->_c5m_fields(x-shift,y-shift,3,pressure_quadratic)[c] for c in 1:3)...)
    cells = [(i,j) for j in H+1:H+N for i in H+1:Nx]
    cut = [(Nx,j) for j in H+1:H+N]
    near = [(Nx-1,j) for j in H+1:H+N]
    interior = [(i,j) for (i,j) in cells if i <= Nx-2]
    links = [(i,j,1) for (i,j) in cut]
    idx = [fv_index(p.grid,i,j,c) for (i,j) in cells for c in 1:3]
    halo_mask = trues(length(U))
    halo_mask[idx] .= false
    halo = findall(halo_mask)
    @assert all(p.tag[i,j] == FV_CELL_FLUID for (i,j) in cells)
    @assert p.immersed === nothing
    @assert Nx*h-shift == 0.5
    # Only the observer needs q. Its same-shape raw geometry describes the
    # domain E face; the surface helper reads P/O and never the outside cell.
    # This object is NEVER passed to a residual or Jacobian assembly.
    qcut, wvel = zeros(4,Nx,Ny), zeros(2,4,Nx,Ny)
    for j in 1:Ny
        qcut[1,Nx,j] = 0.5
        wvel[:,1,Nx,j] .= p.bvel[:,j,1]
    end
    ib = FVImmersedBoundary2D(qcut,wvel,copy(p.tag),h,FVImmersedClosure(:c1))
    observation = FVSteadyProblem2D(p.grid,p.tag,p.nu,p.bvel,p.src,p.visc,ib)
    return (; p,observation,U,cells,cut,near,interior,links,idx,halo,xy,N,
        normal=(1.0,0.0),degree=3,pressure_quadratic)
end

function _c5a_extended(f)
    p, g = f.p, f.p.grid
    Nx, Ny, h = g.Nx+1, g.Ny, g.h
    tag = fill(FV_CELL_FLUID,Nx,Ny)
    tag[Nx,:] .= FV_CELL_SOLID
    qcut, wvel = zeros(4,Nx,Ny), zeros(2,4,Nx,Ny)
    for j in 1:Ny
        qcut[1,Nx-1,j] = 0.5
        wvel[:,1,Nx-1,j] .= p.bvel[:,j,1]
    end
    ib = FVImmersedBoundary2D(qcut,wvel,tag,h,FVImmersedClosure(:c1))
    shift, pq = _C5M_HALO*h, f.pressure_quadratic
    pe = fv_steady_problem(Nx,Ny; h, immersed=ib, conv=false,
        nu=(x,y)->_c5m_nu(x-shift,y-shift),
        boundary_velocity=(s,x,y)->_c5m_fields(x-shift,y-shift,3,pq)[1:2],
        source=(x,y)->_c5m_source(x-shift,y-shift,3,pq))
    Ue = fv_state(pe, ((x,y)->_c5m_fields(x-shift,y-shift,3,pq)[c] for c in 1:3)...)
    ie = [fv_index(pe.grid,i,j,c) for (i,j) in f.cells for c in 1:3]
    return pe,Ue,ie
end

# Test-local read counter: observes actual boundary reads by the unmodified
# production kernel, without replacing or redefining any production method.
struct _C5ABoundaryReads <: AbstractArray{Float64,3}
    data::Array{Float64,3}
    east_reads::Threads.Atomic{Int}
end
Base.size(a::_C5ABoundaryReads) = size(a.data)
function Base.getindex(a::_C5ABoundaryReads,c::Int,j::Int,d::Int)
    d == 1 && Threads.atomic_add!(a.east_reads,1)
    return a.data[c,j,d]
end

function _c5a_trace(f)
    p = f.p
    a = _C5ABoundaryReads(p.bvel,Threads.Atomic{Int}(0))
    traced = FVSteadyProblem2D(p.grid,p.tag,p.nu,a,p.src,p.visc,nothing)
    reference = fv_residual(f.U,p)
    for _ in 1:5
        @test fv_residual(f.U,traced) == reference
    end
    @test a.east_reads[] > 0
    for (i,j) in f.cells
        localR = Kraken.fv_cell_residual(f.U,p.tag,p.nu,p.bvel,p.src,p.grid,i,j,p.visc)
        @test collect(localR) == reference[[fv_index(p.grid,i,j,c) for c in 1:3]]
    end
    return a.east_reads[]
end

function _c5a_rows(N)
    fs = [_c5a_patch(N,pq) for pq in (true,false)]
    f = first(fs)
    solver = fv_newton_solver(f.p)
    J = fv_jacobian!(solver,f.U)
    factor = lu(J[f.idx,f.idx])
    rows = NamedTuple[]
    for f in fs
        @test f.p.immersed === nothing
        @test KernelAbstractions.get_backend(f.U) == KernelAbstractions.CPU()
        @test eltype(f.U) === Float64
        @test all(f.xy(i,j)[1]+f.p.grid.h/2 == 0.5 &&
            Tuple(f.p.bvel[:,j,1]) == _c5m_fields(0.5,f.xy(i,j)[2],3,f.pressure_quadratic)[1:2]
            for (i,j) in f.cut)
        # Both pressure degrees share this linear Stokes Jacobian.
        injected = _c5m_residual_metrics(f,f.U)
        for state in (:injected,:solved)
            U = state == :injected ? f.U : _c5m_solve(f,factor)
            @test U[f.halo] == f.U[f.halo]
            observed = merge(f,(p=f.observation,))
            measured = _c5m_reconstruction_metrics(observed,U)
            rinf = maximum(abs,fv_residual(U,f.p)[f.idx])/f.p.grid.h^2
            push!(rows,(;N,velocity_degree=3,pressure_degree=f.pressure_quadratic ? 2 : 1,
                state,cells=length(f.cells),wall_cells=length(f.cut),
                near_cells=length(f.near),interior_cells=length(f.interior),
                links=length(f.links),qmin=0.5,injected...,measured...,equation_residual=rinf))
        end
    end
    return rows
end

const _C5A_METRICS = (:velocity,:pressure,:surface_gradient,:surface_pressure,
    :traction_rms,:traction_linf,:pressure_traction_rms,:pressure_traction_linf,
    :viscous_traction_rms,:viscous_traction_linf,
    :trunc_cut_momentum,:trunc_cut_mass,:trunc_near_momentum,:trunc_near_mass,
    :trunc_interior_momentum,:trunc_interior_mass)

# Preserve the existing roundoff-budget convention without fitting exact
# injected state values or exact linear-pressure extrapolation as orders.
_c5m_roundoff(r) = 512eps(Float64)*(1+1/r.qmin)
include(joinpath(@__DIR__, "incns_fv_immersed_mms_evidence.jl"))
function _c5a_orders(rows)
    orders = NamedTuple[]
    for pd in (2,1), state in (:injected,:solved), metric in _C5A_METRICS
        group = sort(filter(r->r.pressure_degree==pd && r.state==state,rows);by=r->r.N)
        for k in 1:length(group)-1
            a,b = group[k],group[k+1]
            order = _c5m_pair_order(a,b,metric)
            push!(orders,(;velocity_degree=3,pressure_degree=pd,state,metric,
                coarse=a.N,fine=b.N,error_coarse=getproperty(a,metric),
                error_fine=getproperty(b,metric),order))
        end
    end
    return orders
end

function _c5a_write(out,rows,orders,timings,route)
    mkpath(out)
    for (name,data) in (("aligned_c1_metrics.csv",rows),("aligned_c1_orders.csv",orders),
            ("aligned_c1_timings.csv",timings),("aligned_c1_route.csv",[route]))
        open(joinpath(out,name),"w") do io
            println(io,join(string.(keys(first(data))),","))
            foreach(r->println(io,join(string.(values(r)),",")),data)
        end
    end
end
