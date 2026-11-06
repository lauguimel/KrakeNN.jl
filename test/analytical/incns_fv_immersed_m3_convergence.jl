# C5-M3 gate 4: same 24/40/48 grids as the preserved M1 measurements.
# Frozen velocity order >=1.8 (reviewed C5 acceptance), both L-infinity/L2.
# Flux balance and restored pin must obey the actual residual summation bound;
# pin sensitivity after pressure gauge removal <=1e-8 (small solve tolerance).
# No @test_broken, geometry filtering, mass projection or tolerance relaxation.
isdefined(@__MODULE__,:_c5m3_csv) || include(joinpath(@__DIR__,"incns_fv_immersed_m3_helpers.jl"))
isdefined(@__MODULE__,:_c5mass_problem) || include(joinpath(@__DIR__,"incns_fv_immersed_mass_helpers.jl"))

let evidence = NamedTuple[],orders = NamedTuple[],cases = Dict{Tuple{Symbol,Int},Any}()
    for case in (:couette,:mms),N in (24,40,48)
        @testset "M3 closed $case N=$N" begin
            first_case = _c5mass_problem(N,case;mass=:M3)
            last_case = _c5mass_problem(N,case;mass=:M3,pin=:last)
            a,b = _c5mass_solve(first_case),_c5mass_solve(last_case)
            difference = _c5mass_pin_difference(first_case,a.U,b.U)
            @test difference.velocity <= 1e-8
            @test difference.pressure <= 1e-8
            for (f,result,pin) in ((first_case,a,:first),(last_case,b,:last))
                p,m = f.p,f.p.immersed.mass
                ncell = 3p.grid.Nx*p.grid.Ny
                residual = fv_residual(result.U,p)
                rows = [fv_index(p.grid,Tuple(c)...,3) for c in f.cells if c != f.pinpos]
                roundoff = 2048eps()*max(result.metrics.wall_absolute,1)
                pair_bound = sum(abs,residual[ncell+1:end])+roundoff
                pin_bound = pair_bound+sum(abs,residual[rows])
                @test result.status.converged
                @test result.metrics.residual <= 1e-8
                @test abs(result.metrics.global_balance-sum(m.Q)) <= pair_bound
                @test abs(result.metrics.pin_flux) <= pin_bound
                @test abs(result.metrics.sum_continuity-result.metrics.global_balance) <= roundoff
                push!(evidence,(;case,N,pin,cells=length(f.cells),pairs=length(m.Q),
                    result.metrics...,lambda_max=norm(result.U[ncell+1:end],Inf),
                    pair_residual=norm(residual[ncell+1:end],Inf),
                    balance_bound=pair_bound,restored_pin_bound=pin_bound,
                    pin_velocity_difference=difference.velocity,pin_pressure_difference=difference.pressure))
            end
            cases[(case,N)] = a.metrics
            _c5m3_csv("closed_mass_m3_convergence",evidence)
            if N != 24
                Na = N == 40 ? 24 : 40
                previous = cases[(case,Na)]
                for metric in (:velocity_error,:velocity_rms)
                    order = _c5mass_order(getproperty(previous,metric),getproperty(a.metrics,metric),Na,N)
                    push!(orders,(;case,Na,Nb=N,metric,order))
                    _c5m3_csv("closed_mass_m3_orders",orders)
                    @info "M3 closed velocity order" case Na Nb=N metric order
                    @test order >= 1.8
                    order >= 1.8 || error("M3 STOP: velocity order $order < 1.8; no larger campaign")
                end
            end
        end
    end
    @testset "M3 endpoint velocity orders" begin
        for case in (:couette,:mms),metric in (:velocity_error,:velocity_rms)
            order = _c5mass_order(getproperty(cases[(case,24)],metric),getproperty(cases[(case,48)],metric),24,48)
            push!(orders,(;case,Na=24,Nb=48,metric,order))
            @test order >= 1.8
        end
    end
    _c5m3_csv("closed_mass_m3_orders",orders)
end
