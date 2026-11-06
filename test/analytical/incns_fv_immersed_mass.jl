# C5 step 5: closed annulus, Q/M1/linear-pressure, CPU Float64 Stokes.
# Frozen scientific gates implement the plan's "same order as velocity error
# or better": |global mass|, |omitted continuity|/h^2 and pin-induced velocity
# change must converge at least as fast as velocity L-infinity error.
# Orders use every consecutive grid pair AND the 24->48 endpoint pair.
# No roundoff-zero global mass requirement is imposed on M1. The exact
# algebraic flux identity permits 512eps*max(sum(abs(wall flux)),1); Newton
# residual/h^2 <=1e-8 separates solver error from the scientific mass defect.
# Known failed gates are @test_broken, never relaxed: 2026-11-05, macOS arm64,
# Julia 1.12.5, CPU Float64. These preserve the failed historical M1 baseline;
# the separate M3 gates control continuation to shape derivatives/campaigns.
# The N=32 geometry violates the existing opposite-fluid-cell contract;
# grids 24,40,48 below retain every cut link and satisfy that contract.
using Test
using ForwardDiff
include(joinpath(@__DIR__,"incns_fv_immersed_mass_helpers.jl"))

@testset "C5 closed MMS continuous compatibility" begin
    f = _c5mass_problem(24,:mms)
    field(z) = _c5mass_fields(:mms,z[1],z[2],f.cx,f.cy,f.ri,f.ro)
    for z in ([0.62,0.57],[0.31,0.63],[0.61,0.27])
        du = ForwardDiff.gradient(z->field(z)[1],z)
        dv = ForwardDiff.gradient(z->field(z)[2],z)
        dp = ForwardDiff.gradient(z->field(z)[3],z)
        hu = ForwardDiff.hessian(z->field(z)[1],z)
        hv = ForwardDiff.hessian(z->field(z)[2],z)
        @test abs(du[1]+dv[2]) <= 256eps()*max(norm(du,Inf),norm(dv,Inf),1)
        @test field(z)[4] ≈ dp[1]-tr(hu) rtol=256eps() atol=256eps()
        @test field(z)[5] ≈ dp[2]-tr(hv) rtol=256eps() atol=256eps()
    end
    for R in (f.ri,f.ro), angle in (0.17,0.63,1.81,3.71)
        nx,ny = cos(angle),sin(angle)
        values = field([f.cx+R*nx,f.cy+R*ny])
        @test abs(values[1]*nx+values[2]*ny) <= 256eps()*max(abs(values[1]),abs(values[2]),1)
    end
end

let rows = NamedTuple[], pairs = Dict{Tuple{Symbol,Int},Any}()
    for case in (:couette,:mms), N in (24,40,48)
        elapsed = @elapsed @testset "C5 closed mass $case N=$N" begin
            first_case = _c5mass_problem(N,case)
            last_case = _c5mass_problem(N,case;pin=:last)
            a,b = _c5mass_solve(first_case),_c5mass_solve(last_case)
            difference = _c5mass_pin_difference(first_case,a.U,b.U)
            @test difference.velocity <= max(a.metrics.velocity_error,b.metrics.velocity_error)
            for (f,result,pin) in ((first_case,a,:first),(last_case,b,:last))
                @test result.status.converged
                @test result.metrics.residual <= 1e-8
                @test all(iszero,f.p.src[3:3:end])
                @test KernelAbstractions.get_backend(result.U) == KernelAbstractions.CPU()
                for (state,U,metrics) in ((:injected,f.exact,_c5mass_metrics(f,f.exact)),
                                           (:solved,result.U,result.metrics))
                    atol = 512eps()*max(metrics.wall_absolute,1)
                    @test abs(metrics.sum_continuity-metrics.global_balance) <= atol
                    if state == :solved
                        @test abs(metrics.global_balance-metrics.pin_flux) <= atol
                    end
                    push!(rows,(;case,N,pin,state,pin_i=f.pinpos[1],pin_j=f.pinpos[2],
                        cells=length(f.cells),links=count(>(0),f.p.immersed.qcut),metrics...,
                        pin_velocity_difference=difference.velocity,pin_pressure_difference=difference.pressure))
                end
            end
            pairs[(case,N)] = (;a=a.metrics,b=b.metrics,difference)
        end
        @info "C5 closed mass unit timing" case N seconds=elapsed
    end
    @testset "C5 closed mass scientific gates (historical M1 known failures)" begin
        for case in (:couette,:mms), (Na,Nb) in ((24,40),(40,48),(24,48))
            a,b = pairs[(case,Na)],pairs[(case,Nb)]
            velocity_order = _c5mass_order(a.a.velocity_error,b.a.velocity_error,Na,Nb)
            global_order = _c5mass_order(a.a.global_balance,b.a.global_balance,Na,Nb)
            pin_order = _c5mass_order(a.a.pin_divergence,b.a.pin_divergence,Na,Nb)
            sensitivity_order = _c5mass_order(a.difference.velocity,b.difference.velocity,Na,Nb)
            @info "C5 closed mass scientific orders" case Na Nb velocity_order global_order pin_order sensitivity_order
            if case == :mms && Na == 40 && Nb == 48
                # This one pair passes; it cannot override the other failed pairs.
                @test global_order >= velocity_order
                @test pin_order >= velocity_order
                @test sensitivity_order >= velocity_order
            else
                # Actual algebraic convergence does not restore the omitted
                # continuity row. In particular its normalized defect grows
                # from N=24 to N=48 for both independently compatible flows.
                @test_broken global_order >= velocity_order
                @test_broken pin_order >= velocity_order
                @test_broken sensitivity_order >= velocity_order
            end
        end
    end
    if get(ENV,"KRAKEN_WRITE_C5_MASS_EVIDENCE","false") == "true"
        output = joinpath(@__DIR__,"../../benchmarks/fv_c5/closed_mass.csv")
        open(output,"w") do io
            println(io,join(string.(keys(first(rows))),","))
            for row in rows
                println(io,join(string.(values(row)),","))
            end
        end
    end
end
