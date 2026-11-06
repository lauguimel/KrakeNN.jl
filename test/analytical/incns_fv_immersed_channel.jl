# C5 step 5 open channel, CPU Float64 Stokes unit cases.
# Fixed physical domain [0,2]x[0,1], circle (0.513,0.487), radius0.16, nu1.
# Discrete sampled inflow is normalized to exactly 0.1 on every grid.
# Algebraic bounds allow 2048eps times total absolute boundary flux. Newton
# residual/h^2 <=1e-8 is the existing closed-domain gate, not a continuum claim.
# Pressure drop and drag have no supplied analytical values: unequal-grid
# Richardson estimates are recorded, never converted into unsupported gates.
include(joinpath(@__DIR__,"incns_fv_immersed_channel_helpers.jl"))

@testset "Unequal-grid Richardson reference" begin
    h = (1/24,1/32,1/40)
    for order in (0.7,1.0,1.8,2.0,3.5)
        values = 2.0 .+ 9.0.*collect(h).^order
        result = _c5channel_richardson(h,values)
        @test result.status == "estimated"
        @test result.order ≈ order atol=1e-8 rtol=0
        @test result.extrapolated ≈ 2.0 atol=1e-10 rtol=0
    end
    @test _c5channel_richardson(h,(1.0,2.0,1.0)).status == "nonmonotonic"
end

let rows = NamedTuple[],orders = NamedTuple[],flux_orders = NamedTuple[],
    measurements=Dict{Tuple{Symbol,Int},Any}()
    for N in (24,32,40,48),mass in (:M1,:M3)
        @testset "C5 open channel $mass N=$N" begin
            elapsed = @elapsed begin
                f = _c5channel_problem(N,mass)
                result = _c5channel_solve(f)
            end
            p,m = f.p,result.metrics
            @test !any(==(FV_CELL_PINNED),p.tag)
            @test result.status.converged
            @test m.normalized_residual <= 1e-8
            @test -m.inlet_flux ≈ f.flow atol=64eps()*f.flow rtol=0
            roundoff = 2048eps()*max(abs(m.inlet_flux)+abs(m.outlet_flux)+m.wall_absolute,1)
            @test m.global_balance ≈ m.continuity_sum atol=roundoff rtol=0
            @test abs(m.global_balance) <= m.continuity_absolute+roundoff
            if mass == :M3
                @test length(p.immersed.mass.Q) == 1
                @test all(p.immersed.mass.region_open)
                @test only(p.immersed.mass.Q) == 0
                @test abs(m.wall_flux) <= m.pair_residual+roundoff
                @test abs(m.outlet_flux+m.inlet_flux) <=
                    m.continuity_absolute+m.pair_residual+roundoff
            end
            @test all(isfinite,(m.pressure_drop,m.drag_surface,m.drag_residual_flux))
            push!(rows,(;m...,seconds=elapsed))
            measurements[(mass,N)] = m
            _c5channel_csv("channel_mass",rows)
            @info "C5 open-channel evidence" mass N wall_flux=m.wall_flux inlet=m.inlet_flux outlet=m.outlet_flux pressure_drop=m.pressure_drop drag_surface=m.drag_surface drag_residual_flux=m.drag_residual_flux seconds=elapsed
        end
    end
    @testset "C5 open-channel unequal-grid orders (reported)" begin
        for mass in (:M1,:M3),grids in ((24,32,40),(32,40,48)),
            metric in (:pressure_drop,:drag_surface,:drag_residual_flux)
            data = [getproperty(measurements[(mass,N)],metric) for N in grids]
            result = _c5channel_richardson(Tuple(1/N for N in grids),data)
            push!(orders,(;mass,N1=grids[1],N2=grids[2],N3=grids[3],metric,
                value1=data[1],value2=data[2],value3=data[3],result...))
            @info "C5 open-channel Richardson estimate" mass grids metric result
        end
        @test length(orders) == 12
    end
    _c5channel_csv("channel_orders",orders)
    @testset "C5 open-channel inlet-minus-outlet orders (reported)" begin
        for mass in (:M1,:M3),(Na,Nb) in ((24,32),(32,40),(40,48),(24,48))
            a,b = measurements[(mass,Na)],measurements[(mass,Nb)]
            da,db = a.inlet_minus_outlet,b.inlet_minus_outlet
            roundoff = 2048eps()*max(abs(a.inlet_flux),abs(b.inlet_flux),1)
            limited = max(abs(da),abs(db)) <= roundoff
            order = limited ? NaN : log(abs(da/db))/log(Nb/Na)
            status = limited ? "roundoff_limited" : da*db < 0 ? "sign_change" : "magnitude_estimate"
            push!(flux_orders,(;mass,Na,Nb,defect_a=da,defect_b=db,order,status))
            mass == :M3 && @test limited
            @info "C5 inlet-minus-outlet magnitude order" mass Na Nb da db order status
        end
        @test length(flux_orders) == 8
    end
    _c5channel_csv("channel_flux_orders",flux_orders)
end
