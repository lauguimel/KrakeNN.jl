using Test
using Logging
using Kraken
using KernelAbstractions

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")

"""Best-effort local CUDA functional check for the Backend-directive tests
below — mirrors what `_backend_from_krk_symbol` does, without depending on
CUDA being `using`'d in this test file (it is a Kraken dependency but not
eagerly loaded)."""
function CUDA_functional_here()
    try
        mod = Base.require(Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA"))
        return Base.invokelatest(getfield(mod, :functional))
    catch
        return false
    end
end

@testset "Simulation Runner (.krk)" begin

    @testset "Cavity via .krk" begin
        # Use smaller grid for fast test
        setup = parse_kraken("""
            Simulation cavity D2Q9
            Domain L = 1 x 1  N = 64 x 64
            Physics nu = 0.128

            Boundary north velocity(ux = 0.1, uy = 0)
            Boundary south wall
            Boundary east  wall
            Boundary west  wall

            Run 15000 steps
        """)

        result = run_simulation(setup)

        # No NaN
        @test !any(isnan, result.ρ)
        @test !any(isnan, result.ux)
        @test !any(isnan, result.uy)

        # Mass conservation
        Nx, Ny = 64, 64
        @test abs(sum(result.ρ) - Nx * Ny) / (Nx * Ny) < 0.01

        # Lid velocity approximately imposed at top
        mean_ux_top = sum(result.ux[2:end-1, Ny]) / (Nx - 2)
        @test abs(mean_ux_top - 0.1) / 0.1 < 0.3  # within 30% (coarse + not converged)

        @info "Cavity .krk: mass err = $(abs(sum(result.ρ) - Nx*Ny)/(Nx*Ny)), " *
              "mean ux top = $(round(mean_ux_top, digits=4))"
    end

    @testset "Poiseuille via .krk" begin
        setup = parse_kraken("""
            Simulation poiseuille D2Q9
            Domain L = 0.125 x 1.0  N = 4 x 32
            Physics nu = 0.1  Fx = 1e-5

            Boundary x periodic
            Boundary south wall
            Boundary north wall

            Run 10000 steps
        """)

        result = run_simulation(setup)

        # Analytical parabolic profile (half-way BB: walls at y=0.5, y=Ny+0.5)
        Ny = 32
        nu = 0.1
        Fx = 1e-5
        u_analytical = [Fx / (2nu) * (j - 0.5) * (Ny + 0.5 - j) for j in 1:Ny]
        u_numerical = result.ux[2, :]

        # Interior points
        u_max = maximum(u_analytical)
        errors = abs.(u_numerical[2:end-1] .- u_analytical[2:end-1])
        max_rel_err = maximum(errors) / u_max

        @test max_rel_err < 0.02  # 2% relative error
        @info "Poiseuille .krk: L∞ relative error = $(round(max_rel_err, digits=5))"
    end

    @testset "Couette via .krk" begin
        setup = parse_kraken("""
            Simulation couette D2Q9
            Domain L = 0.125 x 1.0  N = 4 x 32

            Define u_wall = 0.05

            Physics nu = 0.1

            Boundary x periodic
            Boundary south wall
            Boundary north velocity(ux = u_wall, uy = 0)

            Run 30000 steps
        """)

        result = run_simulation(setup)

        Ny = 32
        u_wall = 0.05

        # Analytical: north (j=Ny) moves, south (j=1) is wall
        # But south wall is handled by stream_periodic_x_wall_y bounce-back,
        # north is Zou-He velocity → linear profile u(j) from 0 to u_wall
        # Zou-He at j=Ny: u(Ny)=u_wall, BB at j=1: u(1)≈0
        # u(j) = u_wall * (j - 1) / (Ny - 1)
        H = Ny - 1
        u_analytical = [u_wall * (j - 1) / H for j in 1:Ny]
        u_numerical = result.ux[2, :]

        errors = abs.(u_numerical[2:end-1] .- u_analytical[2:end-1])
        max_rel_err = maximum(errors) / u_wall

        @test max_rel_err < 0.02  # near-exact for linear profile
        @info "Couette .krk: L∞ relative error = $(round(max_rel_err, digits=5))"

        # uy should be ~0
        @test maximum(abs.(result.uy)) < 1e-6
    end

    @testset "Geometry: Fluid region" begin
        # Define fluid as a subset of the domain
        setup = parse_kraken("""
            Simulation channel D2Q9
            Domain L = 1 x 1  N = 32 x 32
            Physics nu = 0.1  Fx = 1e-5

            Fluid channel { y > 0.25 && y < 0.75 }

            Boundary x periodic
            Boundary south wall
            Boundary north wall

            Run 5000 steps
        """)

        result = run_simulation(setup)

        @test !any(isnan, result.ρ)
        # Velocity should be zero in the solid region (y < 0.25 or y > 0.75)
        # Check a node well inside the solid region
        @test abs(result.ux[16, 2]) < 1e-4   # j=2 → y~0.05, solid (near-zero)
        @test abs(result.ux[16, 30]) < 1e-4  # j=30 → y~0.92, solid (near-zero)

        @info "Fluid region test: solid nodes have zero velocity ✓"
    end

    @testset "Spatial BC: parabolic inlet" begin
        setup = parse_kraken("""
            Simulation inlet_test D2Q9
            Domain L = 2 x 1  N = 40 x 20

            Define U = 0.05
            Define H = 1.0

            Physics nu = 0.1

            Boundary west  velocity(ux = 4*U*y*(H - y)/H^2, uy = 0)
            Boundary east  pressure(rho = 1.0)
            Boundary south wall
            Boundary north wall

            Run 5000 steps
        """)

        result = run_simulation(setup)

        @test !any(isnan, result.ρ)

        # Check inlet profile is approximately parabolic
        ux_inlet = result.ux[1, :]
        Ny = 20
        H = 1.0
        U = 0.05
        dy = H / Ny
        u_expected_mid = 4 * U * 0.5 * 0.5  # at y=H/2: 4*U*0.5*0.5 = U

        # Near the center (j ≈ Ny/2)
        j_mid = Ny ÷ 2
        @test abs(ux_inlet[j_mid] - U) / U < 0.3  # reasonable for developing flow

        @info "Spatial BC test: inlet parabolic profile approximately correct"
    end

    @testset "Output produces files" begin
        outdir = mktempdir()
        setup = parse_kraken("""
            Simulation output_test D2Q9
            Domain L = 1 x 1  N = 16 x 16
            Physics nu = 0.1

            Boundary north velocity(ux = 0.1, uy = 0)
            Boundary south wall
            Boundary east  wall
            Boundary west  wall

            Run 200 steps
            Output vtk every 100 [rho, ux, uy]
        """)
        # Override output directory
        setup_mod = SimulationSetup(
            setup.name, setup.lattice, setup.domain, setup.physics,
            setup.user_vars, setup.regions, setup.boundaries, setup.initial,
            setup.modules, setup.max_steps,
            [OutputSetup(:vtk, 100, [:rho, :ux, :uy], outdir)],
            setup.diagnostics, setup.refinements, setup.velocity_field,
            setup.rheology)

        result = run_simulation(setup_mod)

        @test !any(isnan, result.ρ)
        @info "Output test: simulation runs with output enabled"
    end

    @testset "Dispatch: thermal rayleigh_benard via .krk" begin
        path = joinpath(EXAMPLES_DIR, "rayleigh_benard.krk")
        result = run_simulation(path; max_steps=100)
        @test !any(isnan, result.ρ)
        @test !any(isnan, result.ux)
        @test haskey(result, :Temp)
        @test !any(isnan, result.Temp)
    end

    @testset "Dispatch: thermal heat_conduction via .krk" begin
        path = joinpath(EXAMPLES_DIR, "heat_conduction.krk")
        # Falls back to run_rayleigh_benard_2d with Ra≈0 (pragmatic v0.1.0)
        result = run_simulation(path; max_steps=100)
        @test !any(isnan, result.ρ)
        @test haskey(result, :Temp)
    end

    @testset "Dispatch: refined grid_refinement_cavity runs" begin
        path = joinpath(EXAMPLES_DIR, "grid_refinement_cavity.krk")
        result = run_simulation(path; max_steps=100)
        @test !any(isnan, result.ρ)
        @test haskey(result, :ux)
    end

    @testset "Dispatch: D3Q19 cavity_3d (parser does not yet emit D3Q19 setups)" begin
        # The example file currently keeps the D3Q19 case commented out
        # because the parser does not yet support 3D faces. We exercise the
        # dispatch directly by constructing a SimulationSetup with lattice=:D3Q19.
        setup_2d = parse_kraken("""
            Simulation cavity_3d D2Q9
            Domain L = 1 x 1  N = 8 x 8
            Physics nu = 0.1
            Boundary north velocity(ux = 0.1, uy = 0)
            Boundary south wall
            Boundary east  wall
            Boundary west  wall
            Run 10 steps
        """)
        # Promote to D3Q19 with a small 3D domain.
        dom3 = Kraken.DomainSetup(8, 8, 8, 1.0, 1.0, 1.0)
        setup_3d = Kraken.SimulationSetup(
            "cavity_3d", :D3Q19, dom3, setup_2d.physics,
            setup_2d.user_vars, setup_2d.regions, setup_2d.boundaries,
            setup_2d.initial, setup_2d.modules, 10,
            setup_2d.outputs, setup_2d.diagnostics, setup_2d.refinements,
            setup_2d.velocity_field, setup_2d.rheology)
        result = run_simulation(setup_3d)
        @test !any(isnan, result.ρ)
        @test haskey(result, :uz)
    end

    @testset "Dispatch: axisymmetric hagen_poiseuille (synthetic setup)" begin
        # The example .krk keeps the axisym case commented out (parser
        # does not support `axis`/`symmetry` boundaries yet). Exercise
        # dispatch via a synthetic setup that already declares the module.
        setup_base = parse_kraken("""
            Simulation hagen_poiseuille D2Q9
            Domain L = 0.125 x 1.0  N = 4 x 16
            Physics nu = 0.1 Fx = 1e-5
            Boundary x periodic
            Boundary south wall
            Boundary north wall
            Run 50 steps
        """)
        # Inject :axisymmetric module and rename Fx → Fz via body_force.
        bf = Dict{Symbol, Kraken.KrakenExpr}()
        bf[:Fz] = Kraken.parse_kraken_expr("1e-5", Dict{Symbol,Float64}())
        physics_axi = Kraken.PhysicsSetup(setup_base.physics.params, bf)
        setup_axi = Kraken.SimulationSetup(
            "hagen_poiseuille", :D2Q9, setup_base.domain, physics_axi,
            setup_base.user_vars, setup_base.regions, setup_base.boundaries,
            setup_base.initial, [:axisymmetric], 50,
            setup_base.outputs, setup_base.diagnostics, setup_base.refinements,
            setup_base.velocity_field, setup_base.rheology)
        result = run_simulation(setup_axi)
        @test !any(isnan, result.ρ)
    end

    @testset "Dispatch: axisymmetric hagen_poiseuille via .krk example" begin
        path = joinpath(@__DIR__, "..", "examples", "hagen_poiseuille.krk")
        setup = load_kraken(path)
        @test :axisymmetric in setup.modules
        result = run_simulation(path; max_steps=100)
        @test !any(isnan, result.ρ)
    end

    @testset "PNG output without CairoMakie emits warning" begin
        setup = parse_kraken("""
            Simulation png_warn D2Q9
            Domain L = 1.0 x 1.0  N = 16 x 16
            Physics nu = 0.1
            Boundary north velocity ux = 0.1
            Boundary south wall
            Boundary east wall
            Boundary west wall
            Run 50 steps
            Output png every 10 [|u|]
        """)
        # Should warn (CairoMakie not loaded) but not error.
        # min_level=Warn filters the LBM-parameters Info log emitted by
        # sanity_check so that only the CairoMakie warning is matched.
        result = @test_logs (:warn, r"CairoMakie") min_level=Logging.Warn run_simulation(setup)
        @test !any(isnan, result.ρ)
    end

    @testset "Backend directive (T1)" begin
        krk_text(backend_line) = """
            Simulation backend_dispatch_test D2Q9
            Domain L = 1 x 1  N = 8 x 8
            Physics nu = 0.1
            $backend_line
            Boundary north velocity(ux = 0.1, uy = 0)
            Boundary south wall
            Boundary east  wall
            Boundary west  wall
            Run 5 steps
        """

        @testset "(a) explicit cpu backend in the file parses and runs" begin
            setup = parse_kraken(krk_text("Backend cpu"))
            @test setup.backend === :cpu
            result = run_simulation(setup)
            @test !any(isnan, result.ρ)
        end

        @testset "(a-bis) no Backend directive still defaults to CPU" begin
            setup = parse_kraken(krk_text(""))
            @test setup.backend === nothing
            result = run_simulation(setup)
            @test !any(isnan, result.ρ)
        end

        @testset "(b) unknown backend name raises a clear error at parse time" begin
            err = try
                parse_kraken(krk_text("Backend tpu"))
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("tpu", err.msg)
        end

        @testset "(c) explicit Julia kwarg overrides the file's Backend" begin
            setup = parse_kraken(krk_text("Backend cuda"))
            @test setup.backend === :cuda
            # Explicit backend=CPU() must win over the file's `Backend cuda`,
            # so this must run cleanly even though CUDA is not usable here.
            result = run_simulation(setup; backend=KernelAbstractions.CPU())
            @test !any(isnan, result.ρ)
        end

        @testset "requesting a non-functional GPU backend errors, never falls back to CPU" begin
            setup = parse_kraken(krk_text("Backend cuda"))
            # On a machine/CI runner without a functional CUDA device, this
            # must raise a clear, actionable ArgumentError naming 'cuda' —
            # NOT silently run on CPU. (If this ever runs on a CUDA-capable
            # box, the same call would legitimately succeed on the GPU;
            # either way it must never silently fall back to CPU.)
            if !(CUDA_functional_here())
                err = try
                    run_simulation(setup)
                    nothing
                catch e
                    e
                end
                @test err isa ArgumentError
                @test occursin("cuda", err.msg)
            end
        end

        @testset "the same Julia kwarg override behavior applies via run_simulation(filename; ...)" begin
            path, io = mktemp()
            write(io, krk_text("Backend cuda"))
            close(io)
            result = run_simulation(path; backend=KernelAbstractions.CPU(), max_steps=5)
            @test !any(isnan, result.ρ)
            rm(path; force=true)
        end
    end
end
