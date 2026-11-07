using Test
using KrakeNN

@testset "KRK multiphase config" begin

    @testset "Parse Velocity block" begin
        setup = parse_kraken("""
            Simulation advtest D2Q9
            Domain L = 100 x 100  N = 100 x 100
            Physics nu = 0.1
            Module advection_only

            Velocity { ux = -(y - 50)  uy = (x - 50) }
            Initial { C = 0.5*(1 - tanh((sqrt((x-50)^2 + (y-50)^2) - 15) / 2)) }

            Boundary x periodic
            Boundary y periodic
            Run 10 steps
        """)

        @test setup.velocity_field !== nothing
        @test haskey(setup.velocity_field.fields, :ux)
        @test haskey(setup.velocity_field.fields, :uy)
        @test :advection_only in setup.modules
        @test setup.initial !== nothing
        @test haskey(setup.initial.fields, :C)
        @info "Velocity block parsed OK"
    end

    @testset "Parse twophase_vof module" begin
        setup = parse_kraken("""
            Simulation droplet D2Q9
            Domain L = 64 x 64  N = 64 x 64
            Physics nu = 0.1  sigma = 0.01  rho_l = 1.0  rho_g = 0.001
            Module twophase_vof

            Boundary x periodic
            Boundary y periodic

            Initial { C = 0.5*(1 - tanh((sqrt((x-32)^2 + (y-32)^2) - 10) / 2)) }

            Run 100 steps
        """)

        @test :twophase_vof in setup.modules
        @test setup.physics.params[:sigma] ≈ 0.01
        @test setup.physics.params[:rho_l] ≈ 1.0
        @test setup.physics.params[:rho_g] ≈ 0.001
        @info "twophase_vof config parsed OK"
    end

    @testset "Run advection_only via .krk" begin
        setup = parse_kraken("""
            Simulation circle_rot D2Q9
            Domain L = 50 x 50  N = 50 x 50
            Physics nu = 0.1
            Module advection_only

            Velocity { ux = -(y - 25)*0.05  uy = (x - 25)*0.05 }
            Initial { C = 0.5*(1 - tanh((sqrt((x-25)^2 + (y-25)^2) - 10) / 2)) }

            Boundary x periodic
            Boundary y periodic
            Run 20 steps
        """)

        result = run_simulation(setup)
        @test haskey(pairs(result) |> Dict, :C)
        @test !any(isnan, result.C)

        # Check mass conservation
        mass_initial = result.mass_history[1]
        mass_final = result.mass_history[end]
        rel_err = abs(mass_final - mass_initial) / mass_initial
        @test rel_err < 0.1
        @info "advection_only via .krk: mass error = $(round(rel_err*100, digits=2))%"
    end

    @testset "Run twophase_vof via .krk (short)" begin
        setup = parse_kraken("""
            Simulation drop_vof D2Q9
            Domain L = 32 x 32  N = 32 x 32
            Physics nu = 0.1  sigma = 0.01  rho_l = 1.0  rho_g = 0.01
            Module twophase_vof

            Boundary x periodic
            Boundary y periodic

            Initial { C = 0.5*(1 - tanh((sqrt((x-16)^2 + (y-16)^2) - 8) / 2)) }

            Run 50 steps
        """)

        result = run_simulation(setup)
        @test !any(isnan, result.ρ)
        @test !any(isnan, result.C)
        @info "twophase_vof via .krk runs OK, max|u| = $(round(result.max_u_spurious, sigdigits=3))"
    end

    @testset "Velocity field with Refine block" begin
        setup = parse_kraken("""
            Simulation refined_advect D2Q9
            Domain L = 100 x 100  N = 100 x 100
            Physics nu = 0.1
            Module advection_only

            Velocity { ux = -(y - 50)  uy = (x - 50) }
            Initial { C = 0.5*(1 - tanh((sqrt((x-50)^2 + (y-75)^2) - 15) / 2)) }

            Refine disk { region = [25, 50, 75, 100], ratio = 2 }

            Boundary x periodic
            Boundary y periodic
            Run 10 steps
        """)

        @test length(setup.refinements) == 1
        @test setup.refinements[1].name == "disk"
        @test setup.refinements[1].ratio == 2
        @info "Velocity + Refine block parsed OK"
    end
end
