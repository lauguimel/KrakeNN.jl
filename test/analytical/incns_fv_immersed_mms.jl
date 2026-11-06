# C5 step 3, CPU Float64 unit scale: exact injection AND a local solved MMS.
# Frozen grids/angles precede measurement. Q/M1/linear now uses the explicitly
# final A0-referenced gate for every pressure/traction component. All four
# choices use this contract; previous gate evidence remains reproducible.
# Every 16/32/64/128 pair and all six angles remain in every contract.
# No result here validates a closed domain, an integrated force or a GPU run.
using Test, LinearAlgebra, ForwardDiff, Kraken
include(joinpath(@__DIR__,"incns_fv_immersed_mms_helpers.jl"))

# Roundoff budget for O(100) stencil operations with coefficients O(1/q),
# with headroom for sparse solve/reduction: scale by h^-1 or h^-2 only for
# derivatives or differential residuals. This is not a truncation tolerance.
_c5m_roundoff(r) = 512eps(Float64)*(1+1/r.qmin)
_c5m_orders(rows,key) = [log2(getproperty(rows[k],key)/getproperty(rows[k+1],key)) for k in 1:length(rows)-1]
function _c5m_affine_exact(rows)
    return all(filter(r->r.degree==1 && r.state==:injected,rows)) do r
        t = _c5m_roundoff(r)
        max(r.ghost,r.face,r.nominal_pressure) <= t &&
            r.cell_gradient <= t*r.N && r.equation_residual <= t*r.N^2
    end
end
include(joinpath(@__DIR__,"incns_fv_immersed_mms_evidence.jl"))
const _C5M_A0_REFERENCE = _c5m_a0_reference()

@testset "C5 historical revised selection bookkeeping" begin
    reference = (;pressure=ones(3),traction=ones(3))
    check(v) = _c5m_revised_criteria(true,fill(1.8,3),v,fill(0.8,3),fill(0.8,3),reference)
    @test !check([1.8,1.7,1.7]).selected
    @test check([1.8,1.8,1.7]).selected
    @test check([1.8,1.8,1.8]).viscous_pairs == 3
    @test !check([1.8,"roundoff_limited",1.7]).selected
    @test !_c5m_revised_criteria(false,fill(2.0,3),fill(2.0,3),ones(3),ones(3),reference).selected
    @test length(_C5M_A0_REFERENCE.pressure)==length(_C5M_A0_REFERENCE.traction)==3
end

@testset "C5 final selection requires every metric and refinement pair" begin
    reference = (;pressure=ones(3),pressure_traction=ones(3),viscous=ones(3),traction=ones(3))
    boundary = (fill(1.8,3),fill(0.8,3),fill(0.8,3),fill(0.8,3),fill(0.8,3))
    @test _c5m_final_criteria(true,boundary...,reference).selected
    @test !_c5m_final_criteria(false,boundary...,reference).selected
    for metric in 1:5, pair in 1:3
        measured = deepcopy(boundary)
        measured[metric][pair] = prevfloat(measured[metric][pair])
        @test !_c5m_final_criteria(true,measured...,reference).selected
    end
    for invalid in (NaN,Inf,"roundoff_limited")
        @test !_c5m_final_criteria(true,boundary[1:3]...,Any[0.8,invalid,0.8],boundary[5],reference).selected
    end
    @test_throws ErrorException _c5m_final_criteria(true,boundary[1:4]...,[0.8,0.8],reference)
end

@testset "C5 continuous manufactured sources and local surface contract" begin
    for degree in 1:3, pressure_quadratic in (false,true), x in ([0.2,0.3],[0.6,0.7],[0.9,0.4])
        vel(y) = collect(_c5m_fields(y...,degree,pressure_quadratic)[1:2])
        G(y) = ForwardDiff.jacobian(vel,y)
        stress(y) = vec(_c5m_nu(y...)*(G(y)+G(y)'))
        D = ForwardDiff.jacobian(stress,x)
        gp = ForwardDiff.gradient(y->_c5m_fields(y...,degree,pressure_quadratic)[3],x)
        exact = (gp[1]-D[1,1]-D[3,2],gp[2]-D[2,1]-D[4,2],tr(G(x)))
        @test maximum(abs,exact .- _c5m_source(x...,degree,pressure_quadratic)) < 64eps()
        @test maximum(abs,Tuple(vec(G(x))) .- _c5m_gradient(x...,degree)[[1,3,2,4]]) < 64eps()
    end
    f = _c5m_patch(8,30,:quadratic,:M1,:linear,3)
    i,j,d = first(f.links)
    nx,ny = f.normal
    x,y = f.xy(i,j)
    q,h = f.p.immersed.qcut[d,i,j],f.p.grid.h
    xw,yw = x+q*h*((d==1)-(d==2)),y+q*h*((d==3)-(d==4))
    G = _c5m_gradient(xw,yw,3)
    dt = (-ny*G[1]+nx*G[2],-ny*G[3]+nx*G[4])
    physical_nu = _c5m_nu(xw,yw)
    wall(U) = Kraken._fv_surface_reconstruction(U,f.p,i,j,d,f.normal,dt,physical_nu)
    s = @inferred wall(f.U)
    @test s.viscosity == physical_nu
    @test s.viscosity != f.p.nu[i,j]
    e = zeros(length(f.U)); e[fv_index(f.p.grid,i,j,3)] = 1
    Ud = ForwardDiff.Dual.(f.U,e)
    sd = @inferred wall(Ud)
    @test ForwardDiff.partials(sd.pressure)[1] ≈ 1+q atol=8eps()
    @test ForwardDiff.value.(sd.traction) == s.traction
    # For quadratic pressure, linear extrapolation has the exact error
    # -a_e*q*(q+1)*h^2. Nonzero error probes pressure at qh, not nominal h/2.
    ae = d <= 2 ? 0.2 : 0.25
    @test s.pressure-_c5m_fields(xw,yw,3)[3] ≈ -ae*q*(q+1)*h^2 atol=16eps()
    @test_throws ArgumentError Kraken._fv_surface_reconstruction(f.U,f.p,i,j,d,f.normal,dt,physical_nu;min_transversality=0)
    tangent = d <= 2 ? (1e-10,sqrt(1-1e-20)) : (sqrt(1-1e-20),1e-10)
    message = try
        Kraken._fv_surface_reconstruction(f.U,f.p,i,j,d,tangent,dt,physical_nu)
        ""
    catch e
        e isa ArgumentError || rethrow()
        sprint(showerror,e)
    end
    @test occursin("transversality",message) && occursin("($i, $j)",message)
    # A GNF wall law reads the reconstructed physical strain, not physical_nu.
    pg = FVSteadyProblem2D(f.p.grid,f.p.tag,f.p.nu,f.p.bvel,f.p.src,
        FVGNFViscosity(Newtonian(0.2),0.0),f.p.immersed)
    sg = Kraken._fv_surface_reconstruction(f.U,pg,i,j,d,f.normal,dt,100.0)
    @test sg.viscosity == 0.2
end

const _C5M_RESULTS = NamedTuple[]
const _C5M_TIMINGS = NamedTuple[]
for closure in (:linear,:quadratic), mass in (:M1,), pressure in (:zero_gradient,:linear), angle in _C5M_ANGLES
    elapsed = @elapsed @testset "C5 local MMS $closure/$mass/$pressure angle=$angle" begin
        for N in _C5M_NS
            seconds = @elapsed rows = _c5m_triplet(N,angle,closure,mass,pressure)
            push!(_C5M_TIMINGS,(;closure,mass,pressure,angle,N,seconds))
            append!(_C5M_RESULTS,rows)
            for r in rows
                tol = _c5m_roundoff(r)
                @test all(isfinite,v for v in values(r) if v isa AbstractFloat)
                @test r.cells == r.cut_cells+r.near_cells+r.interior_cells
                @test r.links >= r.cut_cells > 0
                @test 0 < r.qmin <= 1
                if r.state == :solved
                    @test r.equation_residual <= tol*N^2
                elseif r.degree == 1
                    @test max(r.ghost,r.face) <= tol
                    @test r.cell_gradient <= tol*N
                    @test r.surface_gradient <= tol*N
                    @test r.surface_pressure <= tol
                    @test r.traction_linf <= 4tol*N
                    if pressure == :linear
                        @test r.nominal_pressure <= tol
                        @test r.equation_residual <= tol*N^2
                    end
                elseif r.degree == 2
                    @test r.surface_gradient <= tol*N
                    @test r.surface_pressure <= tol
                    @test r.traction_linf <= 4tol*N
                    if closure == :quadratic
                        @test r.ghost <= tol
                        @test r.cell_gradient <= tol*N
                        @test r.trunc_interior_momentum <= tol*N^2
                        if pressure == :linear
                            @test r.trunc_cut_mass <= tol*N^2
                            @test r.trunc_cut_momentum <= tol*N^2
                        end
                    end
                else
                    # Analytical interpolation error bound: q*(q+1)<=2 and
                    # max quadratic pressure coefficient is 1/4.
                    @test r.surface_pressure <= 0.5/N^2+tol
                end
            end
        end
    end
    @info "C5 local MMS configuration timing" closure mass pressure angle seconds=elapsed
end

@testset "C5 final scientific selection gate: Q/M1/linear" begin
    for closure in (:linear,:quadratic), mass in (:M1,), pressure in (:zero_gradient,:linear)
        rows = filter(r->r.closure==closure && r.mass==mass && r.residual_pressure==pressure,_C5M_RESULTS)
        gate = _c5m_final_gate(rows,_C5M_A0_REFERENCE)
        println("C5 final gate ",closure,"/",mass,"/",pressure,": ",gate.checks)
        for metric in (:velocity,:pressure,:pressure_traction_rms,:viscous_traction_rms,:traction_rms)
            println("C5 final ",metric,": ",[r.min_order for r in gate.rows if r.metric==metric])
        end
        if closure==:quadratic && pressure==:linear
            @test gate.checks.selected
        else
            @test !gate.checks.selected
        end
    end
end

@testset "C5 Q/M1/linear quadratic solved polynomial identity" begin
    rows = filter(r->r.closure==:quadratic && r.residual_pressure==:linear &&
        r.degree==2 && r.state==:solved,_C5M_RESULTS)
    for r in rows
        t = _c5m_roundoff(r)
        @test max(r.velocity,r.pressure,r.ghost,r.surface_pressure) <= t
        @test r.cell_gradient <= t*r.N
        @test r.surface_gradient <= t*r.N
        @test r.traction_rms <= 4t*r.N
    end
end

# Opt-in evidence output only; ordinary unit-suite runs do not write files.
let out = get(ENV,"KRAKEN_C5_MMS_OUTPUT","")
    if !isempty(out)
        _c5m_write_evidence(out,_C5M_RESULTS,_C5M_TIMINGS;a0_reference=_C5M_A0_REFERENCE)
    end
end
