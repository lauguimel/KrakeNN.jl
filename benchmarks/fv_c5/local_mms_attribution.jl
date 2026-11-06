# Bounded C5-S3b diagnostic: separate velocity and pressure polynomials, then
# split the linear correction by momentum/continuity equations. Partial
# corrections are diagnostics, not solutions of the complete Stokes problem.
# CPU Float64, fixed physical square and exact halo from the registered MMS.
include(joinpath(@__DIR__,"../../test/analytical/incns_fv_immersed_mms_helpers.jl"))
rows, timings = NamedTuple[], NamedTuple[]
total_seconds = @elapsed for angle in (0,30), N in _C5M_NS
    seconds = @elapsed @testset "C5 source attribution N=$N angle=$angle" begin
        base = _c5m_patch(N,angle,:quadratic,:M1,:linear,2;pressure_quadratic=false)
        s = fv_newton_solver(base.p)
        J = fv_jacobian!(s,base.U)
        factor = lu(J[base.idx,base.idx])
        for degree in (2,3), pressure_quadratic in (false,true)
            f = _c5m_patch(N,angle,:quadratic,:M1,:linear,degree;pressure_quadratic)
            R = fv_residual(f.U,f.p)[f.idx]
            injected = _c5m_residual_metrics(f,f.U)
            for state in (:injected,:solved,:momentum_correction,:mass_correction)
                rhs = copy(R)
                if state == :momentum_correction
                    rhs[3:3:end] .= 0
                elseif state == :mass_correction
                    rhs[1:3:end] .= 0
                    rhs[2:3:end] .= 0
                end
                U = copy(f.U)
                state != :injected && (U[f.idx] .-= factor \ rhs)
                measured = _c5m_reconstruction_metrics(f,U)
                equation_residual = maximum(abs,fv_residual(U,f.p)[f.idx])/f.p.grid.h^2
                if state == :solved
                    qmin=minimum(f.p.immersed.qcut[d,i,j] for (i,j,d) in f.links)
                    @test equation_residual <= 512eps(Float64)*(1+1/qmin)*N^2
                end
                push!(rows,(;N,angle,velocity_degree=degree,
                    pressure_degree=pressure_quadratic ? 2 : 1,state,
                    injected...,measured...,equation_residual))
            end
        end
    end
    push!(timings,(;N,angle,seconds))
end
out = get(ENV,"KRAKEN_C5_MMS_OUTPUT",@__DIR__)
mkpath(out)
for (name,records) in (("local_mms_attribution.csv",rows),("local_mms_attribution_timings.csv",timings))
    open(joinpath(out,name),"w") do io
        println(io,join(string.(keys(first(records))),","))
        foreach(r->println(io,join(string.(values(r)),",")),records)
    end
end
println("C5 attribution total_seconds=",total_seconds)
