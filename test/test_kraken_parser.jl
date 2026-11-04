using Test

# Use Kraken if available, otherwise include directly for standalone testing
if !isdefined(@__MODULE__, :KrakenExpr)
    include(joinpath(@__DIR__, "..", "src", "io", "expression.jl"))
    include(joinpath(@__DIR__, "..", "src", "io", "kraken_parser.jl"))
end

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")

@testset "Kraken Parser" begin

    @testset "Cavity example" begin
        setup = load_kraken(joinpath(EXAMPLES_DIR, "cavity.krk"))

        @test setup.name == "cavity"
        @test setup.lattice == :D2Q9
        @test setup.domain.Lx ≈ 1.0
        @test setup.domain.Ly ≈ 1.0
        @test setup.domain.Nx == 128
        @test setup.domain.Ny == 128
        @test setup.physics.params[:nu] ≈ 0.128
        @test setup.max_steps == 60000

        # Boundaries
        @test length(setup.boundaries) == 4
        faces = Dict(b.face => b for b in setup.boundaries)
        @test faces[:north].type == :velocity
        @test evaluate(faces[:north].values[:ux]) ≈ 0.1
        @test faces[:south].type == :wall
        @test faces[:east].type == :wall
        @test faces[:west].type == :wall

        # Output
        @test !isempty(setup.outputs)
        @test setup.outputs[1].format == :vtk
        @test setup.outputs[1].interval == 10000
        @test :rho in setup.outputs[1].fields
        @test :ux in setup.outputs[1].fields

        # No obstacles, no modules
        @test isempty(setup.regions)
        @test isempty(setup.modules)
    end

    @testset "Poiseuille example" begin
        setup = load_kraken(joinpath(EXAMPLES_DIR, "poiseuille.krk"))

        @test setup.name == "poiseuille"
        @test setup.domain.Nx == 4
        @test setup.domain.Ny == 32
        @test setup.physics.params[:nu] ≈ 0.1

        # Body force from Physics line
        @test haskey(setup.physics.body_force, :Fx)
        @test evaluate(setup.physics.body_force[:Fx]) ≈ 1e-5

        # Periodic x → generates west + east periodic
        faces = Dict(b.face => b for b in setup.boundaries)
        @test faces[:west].type == :periodic
        @test faces[:east].type == :periodic
        @test faces[:south].type == :wall
        @test faces[:north].type == :wall
    end

    @testset "Cylinder example" begin
        setup = load_kraken(joinpath(EXAMPLES_DIR, "cylinder.krk"))

        @test setup.name == "cylinder"
        @test setup.domain.Lx ≈ 10.0
        @test setup.domain.Ly ≈ 2.5
        @test setup.domain.Nx == 200
        @test setup.domain.Ny == 50

        # User-defined variables
        @test setup.user_vars[:U] ≈ 0.05
        @test setup.user_vars[:H] ≈ 2.5
        @test setup.user_vars[:cx] ≈ 2.5
        @test setup.user_vars[:R] ≈ 0.5

        # Obstacle
        @test length(setup.regions) == 1
        obs = setup.regions[1]
        @test obs.name == "cylinder"
        @test obs.kind == :obstacle
        @test evaluate(obs.condition; x=2.5, y=1.25) == true  # center
        @test evaluate(obs.condition; x=0.0, y=0.0) == false  # far

        # Boundaries — west has spatial velocity
        faces = Dict(b.face => b for b in setup.boundaries)
        @test faces[:west].type == :velocity
        @test is_spatial(faces[:west].values[:ux])
        # Parabolic profile at y = H/2 → max = U
        @test evaluate(faces[:west].values[:ux]; y=1.25) ≈ 0.05 atol=1e-12
        @test faces[:east].type == :pressure

        # Diagnostics
        @test setup.diagnostics !== nothing
        @test setup.diagnostics.interval == 100
        @test :drag in setup.diagnostics.columns
    end

    @testset "Couette example" begin
        setup = load_kraken(joinpath(EXAMPLES_DIR, "couette.krk"))

        @test setup.name == "couette"
        @test setup.user_vars[:u_wall] ≈ 0.05

        faces = Dict(b.face => b for b in setup.boundaries)
        @test faces[:north].type == :velocity
        @test evaluate(faces[:north].values[:ux]) ≈ 0.05
        @test faces[:west].type == :periodic
        @test faces[:east].type == :periodic
    end

    @testset "Fluid region parsing" begin
        text = """
        Simulation channel D2Q9
        Domain L = 4 x 1  N = 80 x 20
        Physics nu = 0.1

        Fluid channel { y > 0.2 && y < 0.8 }

        Boundary west velocity(ux = 0.1)
        Boundary east pressure(rho = 1.0)
        Boundary south wall
        Boundary north wall

        Run 5000 steps
        """
        setup = parse_kraken(text)

        @test length(setup.regions) == 1
        @test setup.regions[1].kind == :fluid
        @test setup.regions[1].name == "channel"
        @test evaluate(setup.regions[1].condition; y=0.5) == true
        @test evaluate(setup.regions[1].condition; y=0.1) == false
    end

    @testset "Combined Fluid + Obstacle" begin
        text = """
        Simulation pipe_cylinder D2Q9
        Domain L = 10 x 2  N = 200 x 40
        Physics nu = 0.05

        Fluid pipe { y > 0.3 && y < 1.7 }
        Obstacle cylinder { (x - 5)^2 + (y - 1)^2 <= 0.3^2 }

        Boundary west velocity(ux = 0.05)
        Boundary east pressure(rho = 1.0)
        Boundary south wall
        Boundary north wall

        Run 10000 steps
        """
        setup = parse_kraken(text)

        @test length(setup.regions) == 2
        @test setup.regions[1].kind == :fluid
        @test setup.regions[2].kind == :obstacle
    end

    @testset "Initial conditions" begin
        text = """
        Simulation test D2Q9
        Domain L = 1 x 1  N = 32 x 32
        Physics nu = 0.1

        Boundary south wall
        Boundary north wall
        Boundary east wall
        Boundary west wall

        Initial { ux = 0.01*sin(2*pi*x/Lx) uy = 0 }

        Run 1000 steps
        """
        setup = parse_kraken(text)

        @test setup.initial !== nothing
        @test haskey(setup.initial.fields, :ux)
        @test is_spatial(setup.initial.fields[:ux])
    end

    @testset "Module parsing" begin
        text = """
        Simulation test D2Q9
        Domain L = 1 x 1  N = 32 x 32
        Physics nu = 0.1 Pr = 1.0 Ra = 2000

        Module thermal

        Boundary south wall
        Boundary north wall
        Boundary east wall
        Boundary west wall

        Run 1000 steps
        """
        setup = parse_kraken(text)
        @test :thermal in setup.modules
    end

    @testset "Backend directive" begin
        base = """
        Simulation backend_test D2Q9
        Domain L = 1 x 1  N = 8 x 8
        Physics nu = 0.1
        %BACKEND%
        Boundary north velocity(ux = 0.1, uy = 0)
        Boundary south wall
        Boundary east  wall
        Boundary west  wall
        Run 5 steps
        """

        # No Backend directive at all -> setup.backend stays nothing (caller/
        # default decides; a bare .krk file still defaults to CPU downstream).
        no_backend_text = replace(base, "%BACKEND%\n" => "")
        setup_nobackend = parse_kraken(no_backend_text)
        @test setup_nobackend.backend === nothing

        # Explicit cpu/cuda/metal all parse and are stored verbatim.
        for name in ("cpu", "cuda", "metal")
            setup = parse_kraken(replace(base, "%BACKEND%" => "Backend $name"))
            @test setup.backend === Symbol(name)
        end

        # Case-insensitive.
        setup_upper = parse_kraken(replace(base, "%BACKEND%" => "Backend CUDA"))
        @test setup_upper.backend === :cuda

        # Unknown backend name -> clear, actionable ArgumentError naming the
        # offending token (never a silent fallback to cpu).
        err = try
            parse_kraken(replace(base, "%BACKEND%" => "Backend tpu"))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("tpu", err.msg)
        @test occursin("cpu", err.msg)  # names the supported values

        # A second Backend directive is rejected (mirrors the "only one
        # Sensitivity/Units block" rule).
        @test_throws ArgumentError parse_kraken(
            replace(base, "%BACKEND%" => "Backend cpu\nBackend metal"))
    end

    @testset "Error handling" begin
        # Missing Simulation line
        @test_throws ArgumentError parse_kraken("Domain L = 1 x 1 N = 32 x 32\nRun 100 steps")

        # Missing Domain
        @test_throws ArgumentError parse_kraken("Simulation test D2Q9\nRun 100 steps")

        # Missing Run
        @test_throws ArgumentError parse_kraken("Simulation test D2Q9\nDomain L = 1 x 1 N = 32 x 32")

        # Unknown keyword
        @test_throws ArgumentError parse_kraken("""
            Simulation test D2Q9
            Domain L = 1 x 1 N = 32 x 32
            FooBar baz
            Run 100 steps
        """)

        # Invalid lattice
        @test_throws ArgumentError parse_kraken("""
            Simulation test D2Q99
            Domain L = 1 x 1 N = 32 x 32
            Run 100 steps
        """)
    end

    @testset "Setup reynolds helper" begin
        text = """
        Simulation re_test D2Q9
        Domain L = 1.0 x 1.0  N = 128 x 128
        Setup reynolds = 1000 L_ref = 128 U_ref = 0.1
        Boundary north velocity(ux = 0.1, uy = 0)
        Boundary south wall
        Boundary east wall
        Boundary west wall
        Run 1000 steps
        """
        setup = parse_kraken(text)
        @test setup.physics.params[:nu] ≈ 0.0128 atol=1e-9
        @test setup.physics.params[:Re] ≈ 1000
    end

    @testset "Setup reynolds conflict with nu" begin
        text = """
        Simulation bad D2Q9
        Domain L = 1 x 1  N = 64 x 64
        Physics nu = 0.01
        Setup reynolds = 1000
        Boundary north wall
        Boundary south wall
        Boundary east wall
        Boundary west wall
        Run 100 steps
        """
        @test_throws ArgumentError parse_kraken(text)
    end

    @testset "Setup rayleigh helper" begin
        text = """
        Simulation ra_test D2Q9
        Domain L = 1.0 x 1.0  N = 128 x 128
        Setup rayleigh = 1e5 prandtl = 0.71 L_ref = 128 U_ref = 0.1
        Module thermal
        Boundary south wall T = 1.0
        Boundary north wall T = 0.0
        Boundary east wall
        Boundary west wall
        Run 100 steps
        """
        setup = parse_kraken(text)
        @test setup.physics.params[:Ra] ≈ 1e5
        @test setup.physics.params[:Pr] ≈ 0.71
        @test haskey(setup.physics.params, :nu)
        @test haskey(setup.physics.params, :alpha)
        @test haskey(setup.physics.params, :gbeta_DT)
        # Consistency: Pr = nu/alpha
        @test setup.physics.params[:nu] / setup.physics.params[:alpha] ≈ 0.71 atol=1e-9
        # Ra = gβΔT L^3 / (ν α)
        ν = setup.physics.params[:nu]
        α = setup.physics.params[:alpha]
        gβΔT = setup.physics.params[:gbeta_DT]
        @test gβΔT * 128.0^3 / (ν * α) ≈ 1e5 atol=1e-3
    end

    @testset "Preset cavity_2d" begin
        text = "Preset cavity_2d"
        setup = parse_kraken(text)
        @test setup.name == "cavity_2d"
        @test setup.domain.Nx == 128
        faces = Dict(b.face => b for b in setup.boundaries)
        @test faces[:north].type == :velocity
        @test faces[:south].type == :wall
    end

    @testset "Preset override" begin
        text = """
        Preset cavity_2d
        Setup reynolds = 5000 L_ref = 128 U_ref = 0.1
        """
        # Preset sets nu=0.01; reynolds would conflict. Use a preset without nu,
        # or override by manually clearing? For now test that Re-only overrides:
        @test_throws ArgumentError parse_kraken(text)

        # Override via Physics works:
        text2 = """
        Preset cavity_2d
        Physics nu = 0.005
        """
        setup = parse_kraken(text2)
        @test setup.physics.params[:nu] ≈ 0.005
    end

    @testset "Spell-correction for directives" begin
        text = """
        Simulation test D2Q9
        Domain L = 1 x 1 N = 32 x 32
        Physics nu = 0.1
        Boundayr north wall
        Run 100 steps
        """
        err = try
            parse_kraken(text)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("Boundary", err.msg)
    end

    @testset "Spell-correction for BC type" begin
        text = """
        Simulation test D2Q9
        Domain L = 1 x 1 N = 32 x 32
        Physics nu = 0.1
        Boundary north velocty(ux = 0.1)
        Boundary south wall
        Boundary east wall
        Boundary west wall
        Run 100 steps
        """
        err = try
            parse_kraken(text)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("velocity", err.msg)
    end

    @testset "Sanity check tau < 0.5" begin
        # nu = -0.1 would give tau = 0.2; use a valid-parseable Setup
        text = """
        Simulation unstable D2Q9
        Domain L = 1 x 1 N = 64 x 64
        Physics nu = -0.1
        Boundary north wall
        Boundary south wall
        Boundary east wall
        Boundary west wall
        Run 100 steps
        """
        err = try
            parse_kraken(text)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("increase", err.msg) && occursin("ν", err.msg)
    end

    @testset "Sanity check Mach warning" begin
        # U_ref = 0.15 triggers compressibility warn (> 0.1) but NOT error (< 0.3)
        text = """
        Simulation fast D2Q9
        Domain L = 1 x 1 N = 64 x 64
        Physics nu = 0.1
        Boundary north velocity(ux = 0.15, uy = 0)
        Boundary south wall
        Boundary east wall
        Boundary west wall
        Run 100 steps
        """
        io = IOBuffer()
        Base.CoreLogging.with_logger(Base.CoreLogging.ConsoleLogger(io)) do
            parse_kraken(text)
        end
        out = String(take!(io))
        @test occursin("Mach", out)
    end

    @testset "Sanity check CFL error (U_ref > 0.3)" begin
        text = """
        Simulation cfl D2Q9
        Domain L = 1 x 1 N = 64 x 64
        Physics nu = 0.1
        Boundary north velocity(ux = 0.5, uy = 0)
        Boundary south wall
        Boundary east wall
        Boundary west wall
        Run 100 steps
        """
        err = try
            parse_kraken(text)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("CFL", err.msg)
    end

    @testset "Sanity check high tau (diffusion-dominated)" begin
        # Re=0.01, N=64, U=0.1 → ν = 0.1*64/0.01 = 640 → τ = 1920.5
        text = """
        Simulation slow D2Q9
        Domain L = 1 x 1 N = 64 x 64
        Setup reynolds = 0.01 L_ref = 64 U_ref = 0.1
        Boundary north velocity(ux = 0.1, uy = 0)
        Boundary south wall
        Boundary east wall
        Boundary west wall
        Run 100 steps
        """
        err = try
            parse_kraken(text)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("absurdly large", err.msg)
    end

    @testset "Sanity check moderate high tau (warning)" begin
        # τ ~ 15 → warning but not error
        # ν = 5 → τ = 15.5
        text = """
        Simulation diffusive D2Q9
        Domain L = 1 x 1 N = 64 x 64
        Physics nu = 5.0
        Boundary north velocity(ux = 0.1, uy = 0)
        Boundary south wall
        Boundary east wall
        Boundary west wall
        Run 100 steps
        """
        io = IOBuffer()
        Base.CoreLogging.with_logger(Base.CoreLogging.ConsoleLogger(io)) do
            parse_kraken(text)
        end
        out = String(take!(io))
        @test occursin("diffusion dominates", out) || occursin("very large", out)
    end

    @testset "Sanity check returns SanityIssue vector" begin
        text = """
        Simulation ok D2Q9
        Domain L = 1 x 1 N = 64 x 64
        Physics nu = 0.01
        Boundary north velocity(ux = 0.1, uy = 0)
        Boundary south wall
        Boundary east wall
        Boundary west wall
        Run 100 steps
        """
        setup = parse_kraken(text)
        issues = sanity_check(setup; verbose=false)
        @test issues isa Vector{SanityIssue}
        # ν=0.01, τ=0.53, U=0.1, Ma=0.17 — should be clean
        @test isempty(issues)
    end

    @testset "Sanity check resolution warning" begin
        # N=5 → should warn about coarse grid
        text = """
        Simulation tiny D2Q9
        Domain L = 1 x 1 N = 5 x 5
        Physics nu = 0.1
        Boundary north wall
        Boundary south wall
        Boundary east wall
        Boundary west wall
        Run 10 steps
        """
        setup = parse_kraken(text)
        issues = sanity_check(setup; verbose=false)
        @test any(i -> i.category == :resolution && i.level == :warn, issues)
    end

    @testset "Sanity check sweep" begin
        text = """
        Simulation sweep_sc D2Q9
        Domain L = 1 x 1 N = 64 x 64
        Setup reynolds = Re L_ref = 64 U_ref = 0.1
        Boundary north velocity(ux = 0.1, uy = 0)
        Boundary south wall
        Boundary east wall
        Boundary west wall
        Sweep Re = [1, 100, 1000]
        Run 100 steps
        """
        setups = parse_kraken_sweep(text)
        all_issues = sanity_check_sweep(setups; verbose=false)
        @test length(all_issues) == 3
        @test all(isa.(all_issues, Vector{SanityIssue}))
    end

    @testset "Sweep directive" begin
        text = """
        Simulation sweep_test D2Q9
        Domain L = 1 x 1 N = 64 x 64
        Physics nu = 0.01
        Boundary north velocity(ux = 0.1, uy = 0)
        Boundary south wall
        Boundary east wall
        Boundary west wall
        Sweep Re = [100, 200, 500]
        Run 100 steps
        """
        setups = parse_kraken_sweep(text)
        @test length(setups) == 3
        # Re kwarg doesn't affect nu directly since no Setup reynolds present,
        # but it should be applied as kwarg override (no :Re param in physics).
        # Test with Setup reynolds instead:
        text2 = """
        Simulation sweep_re D2Q9
        Domain L = 1 x 1 N = 64 x 64
        Setup reynolds = Re L_ref = 64 U_ref = 0.1
        Boundary north velocity(ux = 0.1, uy = 0)
        Boundary south wall
        Boundary east wall
        Boundary west wall
        Sweep Re = [100, 200, 500]
        Run 100 steps
        """
        # This won't work since Setup parses literal numbers, not kwargs.
        # Simpler: check Sweep just produces 3 setups and kwargs are applied
        # by verifying any downstream effect. Use Physics nu override with Sweep:
        text3 = """
        Simulation sweep_nu D2Q9
        Domain L = 1 x 1 N = 64 x 64
        Physics nu = 0.01
        Boundary north velocity(ux = 0.1, uy = 0)
        Boundary south wall
        Boundary east wall
        Boundary west wall
        Sweep nu = [0.01, 0.02, 0.05]
        Run 100 steps
        """
        setups3 = parse_kraken_sweep(text3)
        @test length(setups3) == 3
        nus = sort([s.physics.params[:nu] for s in setups3])
        @test nus ≈ [0.01, 0.02, 0.05]
    end

    @testset "3D domain" begin
        text = """
        Simulation cavity3d D3Q19
        Domain L = 1 x 1 x 1  N = 32 x 32 x 32
        Physics nu = 0.1

        Boundary south wall
        Boundary north velocity(ux = 0.1)
        Boundary east wall
        Boundary west wall

        Run 1000 steps
        """
        setup = parse_kraken(text)
        @test setup.lattice == :D3Q19
        @test setup.domain.Lz ≈ 1.0
        @test setup.domain.Nz == 32
    end

    @testset "3D face names (D3Q19)" begin
        # All six physical faces including top/bottom must parse cleanly
        text = """
        Simulation box3d D3Q19
        Domain L = 1 x 1 x 1  N = 16 x 16 x 16
        Physics nu = 0.01
        Boundary west  wall
        Boundary east  wall
        Boundary south wall
        Boundary north wall
        Boundary bottom wall
        Boundary top    velocity(ux = 0.05, uy = 0)
        Run 10 steps
        """
        setup = parse_kraken(text)
        @test setup.lattice == :D3Q19
        @test length(setup.boundaries) == 6
        faces = Set(b.face for b in setup.boundaries)
        @test :top in faces
        @test :bottom in faces
    end

    @testset "3D face names rejected on D2Q9" begin
        text = """
        Simulation bad2d D2Q9
        Domain L = 1 x 1 N = 32 x 32
        Physics nu = 0.01
        Boundary top wall
        Boundary south wall
        Boundary east wall
        Boundary west wall
        Run 10 steps
        """
        err = try
            parse_kraken(text); nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("D2Q9", err.msg)
        @test occursin("north/south/east/west", err.msg)
    end

    @testset "No Mach warning at default U_ref = 0.1" begin
        # U_ref falls back to 0.1 (default / probed from BC) → no warning
        text = """
        Simulation quiet D2Q9
        Domain L = 1 x 1 N = 64 x 64
        Physics nu = 0.01
        Boundary north velocity(ux = 0.1, uy = 0)
        Boundary south wall
        Boundary east wall
        Boundary west wall
        Run 10 steps
        """
        io = IOBuffer()
        Base.CoreLogging.with_logger(Base.CoreLogging.ConsoleLogger(io)) do
            parse_kraken(text)
        end
        out = String(take!(io))
        @test !occursin("Mach", out)
        @test !occursin("compressibility", out)
    end

    @testset "Mach warning fires when U_ref > 0.1" begin
        text = """
        Simulation loud D2Q9
        Domain L = 1 x 1 N = 64 x 64
        Physics nu = 0.01
        Boundary north velocity(ux = 0.15, uy = 0)
        Boundary south wall
        Boundary east wall
        Boundary west wall
        Run 10 steps
        """
        io = IOBuffer()
        Base.CoreLogging.with_logger(Base.CoreLogging.ConsoleLogger(io)) do
            parse_kraken(text)
        end
        out = String(take!(io))
        @test occursin("U_ref", out)
        @test occursin("0.15", out)
    end

    @testset "Axisymmetric boundary aliases" begin
        # With Module axisymmetric, the (z, r) face aliases must resolve:
        #   z    -> x      (streamwise, periodic)
        #   wall -> north  (outer radial wall at r = R)
        #   axis -> south  (axis of symmetry at r = 0)
        text = """
        Simulation hagen_poiseuille D2Q9
        Domain L = 0.125 x 1.0 N = 4 x 32
        Physics nu = 0.1 Fz = 1e-5
        Module axisymmetric
        Boundary z    periodic
        Boundary wall wall
        Boundary axis symmetry
        Run 100 steps
        """
        setup = parse_kraken(text)
        @test :axisymmetric in setup.modules
        faces = Dict(b.face => b for b in setup.boundaries)
        # `z periodic` expands to west + east periodic
        @test haskey(faces, :west)
        @test haskey(faces, :east)
        @test faces[:west].type == :periodic
        @test faces[:east].type == :periodic
        # `wall` alias -> north wall
        @test haskey(faces, :north)
        @test faces[:north].type == :wall
        # `axis` alias -> south symmetry
        @test haskey(faces, :south)
        @test faces[:south].type == :symmetry
        # Fz was promoted to body_force
        @test haskey(setup.physics.body_force, :Fz)

        # Without Module axisymmetric, `Boundary z periodic` must still error.
        bad_text = """
        Simulation bad D2Q9
        Domain L = 1.0 x 1.0 N = 8 x 8
        Physics nu = 0.1
        Boundary z periodic
        Boundary south wall
        Boundary north wall
        Run 10 steps
        """
        @test_throws ArgumentError parse_kraken(bad_text)

        bad_wall = """
        Simulation bad D2Q9
        Domain L = 1.0 x 1.0 N = 8 x 8
        Physics nu = 0.1
        Boundary x periodic
        Boundary wall wall
        Boundary south wall
        Run 10 steps
        """
        @test_throws ArgumentError parse_kraken(bad_wall)
    end

    @testset "Hagen-Poiseuille example parses" begin
        setup = load_kraken(joinpath(EXAMPLES_DIR, "hagen_poiseuille.krk"))
        @test setup.name == "hagen_poiseuille"
        @test :axisymmetric in setup.modules
        @test haskey(setup.physics.body_force, :Fz)
        faces = Set(b.face for b in setup.boundaries)
        @test :west in faces && :east in faces
        @test :north in faces && :south in faces
    end

    @testset "Output png parses" begin
        setup = parse_kraken("""
            Simulation test_png D2Q9
            Domain L = 1.0 x 1.0  N = 16 x 16
            Physics nu = 0.1
            Boundary north velocity ux = 0.1
            Boundary south wall
            Boundary east wall
            Boundary west wall
            Run 100 steps
            Output png every 1000 [|u|]
        """)
        @test length(setup.outputs) == 1
        @test setup.outputs[1].format == :png
        @test setup.outputs[1].interval == 1000
        @test Symbol("|u|") in setup.outputs[1].fields
        @test setup.outputs[1].fps == 10  # default
    end

    @testset "Output gif with fps parses" begin
        setup = parse_kraken("""
            Simulation test_gif D2Q9
            Domain L = 1.0 x 1.0  N = 16 x 16
            Physics nu = 0.1
            Boundary north velocity ux = 0.1
            Boundary south wall
            Boundary east wall
            Boundary west wall
            Run 100 steps
            Output gif every 100 [|u|] fps=15
        """)
        @test length(setup.outputs) == 1
        @test setup.outputs[1].format == :gif
        @test setup.outputs[1].interval == 100
        @test setup.outputs[1].fps == 15
    end

    @testset "Multiple Output directives (vtk + png)" begin
        setup = parse_kraken("""
            Simulation test_multi D2Q9
            Domain L = 1.0 x 1.0  N = 16 x 16
            Physics nu = 0.1
            Boundary north velocity ux = 0.1
            Boundary south wall
            Boundary east wall
            Boundary west wall
            Run 100 steps
            Output vtk every 10000 [rho, ux, uy]
            Output png every 1000 [|u|]
        """)
        @test length(setup.outputs) == 2
        fmts = Set(o.format for o in setup.outputs)
        @test :vtk in fmts
        @test :png in fmts
    end
end
