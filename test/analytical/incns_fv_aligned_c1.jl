# Standalone A0 decision evidence, CPU Float64, local linear MMS only.
# No convergence gate is imposed: report every refinement pair and stop.
module C5AlignedC1Reference
using Test, LinearAlgebra, Kraken
include(joinpath(@__DIR__, "incns_fv_aligned_c1_helpers.jl"))

const route = Ref{NamedTuple}()
@testset "A0 aligned C1 production route and restricted equivalence" begin
    f = _c5a_patch(8,true)
    p,g = f.p,f.p.grid
    @test p.immersed === nothing
    @test Kraken._fv_immersed_args(p.immersed) === (nothing,nothing,nothing)
    @test KernelAbstractions.get_backend(f.U) == KernelAbstractions.CPU()
    @test eltype(f.U) === Float64
    @test g.Nx*g.h-_C5M_HALO*g.h == 0.5
    @test length(f.cells) == f.N^2÷2
    pe,Ue,ie = _c5a_extended(f)
    for (i,j,d) in f.links
        x,y = f.xy(i,j)
        exact = _c5m_fields(0.5,y,3,true)[1:2]
        @test x+g.h/2 == 0.5
        @test Kraken._fv_face_kind(g,p.tag,i,j,d)[1] == Kraken._FV_F_DIRICHLET
        @test Kraken._fv_dirichlet_velocity(g,p.bvel,i,j,d) == exact
        # Tagged internal faces hard-code no-slip without immersed data.
        @test Kraken._fv_dirichlet_velocity(pe.grid,pe.bvel,i,j,d) == (0.0,0.0)
        @test Kraken._fv_wall_data(f.U,p.bvel,g,i,j,d,nothing,nothing,nothing) ==
            Kraken._fv_wall_data(Ue,pe.bvel,pe.grid,i,j,d,
                pe.immersed.qcut,pe.immersed.wvel,pe.immersed.closure)
    end
    calls = _c5a_trace(f)
    # Compare both exact injection and an arbitrary perturbation on the same
    # restricted unknowns; the extended solid-column equations are excluded.
    residual_delta = 0.0
    for amplitude in (0.0,0.017)
        Ua,Ub = copy(f.U),copy(Ue)
        perturbation = amplitude .* sin.(collect(eachindex(f.idx)))
        Ua[f.idx] .+= perturbation
        Ub[ie] .+= perturbation
        delta = maximum(abs,fv_residual(Ua,p)[f.idx]-fv_residual(Ub,pe)[ie])
        residual_delta = max(residual_delta,delta)
        @test delta == 0.0
    end
    Ja = fv_jacobian!(fv_newton_solver(p),f.U)[f.idx,f.idx]
    Jb = fv_jacobian!(fv_newton_solver(pe),Ue)[ie,ie]
    jacobian_delta = maximum(abs,Ja-Jb)
    @test jacobian_delta == 0.0
    route[] = (;backend=:CPU,precision=:Float64,immersed_solve=false,
        wall_x=0.5,halo_layers=_C5M_HALO,trace_evaluations=5,east_boundary_reads=calls,
        tiny_grid=8,restricted_unknowns=length(f.idx),residual_delta,jacobian_delta)
    println("A0 route: ",route[])
end

const results = NamedTuple[]
const timings = NamedTuple[]
for N in _C5M_NS
    seconds = @elapsed @testset "A0 aligned C1 N=$N both pressure degrees" begin
        rows = _c5a_rows(N)
        append!(results,rows)
        for r in rows
            @test all(isfinite,v for v in values(r) if v isa AbstractFloat)
            @test r.cells == r.wall_cells+r.near_cells+r.interior_cells == N^2÷2
            @test r.links == r.wall_cells == N
            @test r.qmin == 0.5
            # Existing arithmetic roundoff budget, not a convergence gate.
            r.state == :solved && @test r.equation_residual <= _c5m_roundoff(r)*N^2
        end
    end
    push!(timings,(;N,seconds))
    println("A0 grid timing: N=",N," seconds=",seconds)
end

const orders = _c5a_orders(results)
@testset "A0 complete finite evidence without order selection" begin
    @test length(results) == 16
    @test length(orders) == 2*2*length(_C5A_METRICS)*3
    @test all(r->r.order == "roundoff_limited" || isfinite(r.order),orders)
    for pd in (2,1), metric in (:pressure,:traction_rms,:pressure_traction_rms,:viscous_traction_rms)
        selected = filter(r->r.pressure_degree==pd && r.state==:solved && r.metric==metric,orders)
        @test [(r.coarse,r.fine) for r in selected] == [(16,32),(32,64),(64,128)]
        @test all(r->r.order isa Number && isfinite(r.order),selected)
        println("A0 solved pressure_degree=",pd," metric=",metric,
            " orders=",[r.order for r in selected])
    end
end

let out = get(ENV,"KRAKEN_C5_A0_OUTPUT","")
    isempty(out) || _c5a_write(out,results,orders,timings,route[])
end
end # module
