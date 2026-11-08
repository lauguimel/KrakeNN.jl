using Test
using KrakeNN

const K1_ROOT = normpath(joinpath(@__DIR__, ".."))

function _k1_velocity_l2(a, b, fluid)
    n = count(fluid)
    @test n > 0
    du2 = sum((a.ux[fluid] .- b.ux[fluid]) .^ 2 .+
              (a.uy[fluid] .- b.uy[fluid]) .^ 2) / n
    return sqrt(du2)
end

function _k1_solid_mask(setup)
    dom = setup.domain
    mask = falses(dom.Nx, dom.Ny)
    KrakeNN._apply_geometry!(mask, setup, dom.Lx / dom.Nx, dom.Ly / dom.Ny)
    return mask
end

function _k1_collision_setup(collision_line::AbstractString)
    return parse_kraken("""
Simulation k1_collision D2Q9
Domain L = 1.0 x 1.0  N = 24 x 24
Physics nu = 0.02$collision_line
Boundary north wall
Boundary south wall
Boundary east wall
Boundary west wall
Initial { ux = 0.02*sin(2*pi*x)*sin(pi*y) uy = -0.01*cos(pi*x)*sin(2*pi*y) rho = 1 }
Run 30 steps
""")
end

function _k1_stl_wall_setup(wall_line::AbstractString)
    return parse_kraken("""
Simulation k1_wall D2Q9
Define U = 0.03
Define H = 1.0
Domain L = 4.0 x 1.0  N = 80 x 20
Physics nu = 0.05$wall_line
Obstacle cyl stl(file = "examples/geometry_stl/cylinder.stl", z_slice = 0.5)
Boundary west velocity(ux = 4*U*y*(H-y)/H^2, uy = 0)
Boundary east pressure(rho = 1.0)
Boundary south wall
Boundary north wall
Initial { ux = U uy = 0 rho = 1 }
Run 80 steps
""")
end

@testset "K1 .krk symbolic knobs" begin
    @testset "Physics and Define store bareword symbols" begin
        setup = parse_kraken("""
Simulation k1_symbols D2Q9
Define selected_collision = trt
Domain L = 1.0 x 1.0  N = 8 x 8
Physics { nu = 0.02 collision = selected_collision wall_bc = halfwaybb advection_scheme = weno }
Boundary north wall
Boundary south wall
Boundary east wall
Boundary west wall
Run 1 steps
""")

        @test setup.user_vars[:selected_collision] === :trt
        @test setup.physics.params[:collision] === :trt
        @test setup.physics.params[:wall_bc] === :halfwaybb
        @test setup.physics.params[:advection_scheme] === :weno
        @test setup.collision === :trt
        @test setup.wall_bc === :halfwaybb
    end

    @testset "collision = trt reaches a different kernel than BGK" begin
        bgk = run_simulation(_k1_collision_setup(""); T=Float64)
        trt = run_simulation(_k1_collision_setup(" collision = trt"); T=Float64)

        @test bgk.setup.collision === :bgk
        @test trt.setup.collision === :trt
        @test all(isfinite.(bgk.ux))
        @test all(isfinite.(trt.ux))
        fluid = trues(size(bgk.ux))
        @test _k1_velocity_l2(bgk, trt, fluid) > 1e-10
    end

    @testset "wall_bc = libb reaches the STL LI-BB path" begin
        cd(K1_ROOT) do
            half = run_simulation(_k1_stl_wall_setup(""); T=Float64)
            libb = run_simulation(_k1_stl_wall_setup(" wall_bc = libb"); T=Float64)

            @test half.setup.wall_bc === :halfwaybb
            @test libb.setup.wall_bc === :libb
            @test all(isfinite.(half.ux))
            @test all(isfinite.(libb.ux))

            fluid = .!(_k1_solid_mask(half.setup) .| _k1_solid_mask(libb.setup))
            @test _k1_velocity_l2(half, libb, fluid) > 1e-10
        end
    end

    @testset "natural_convection_2d preset example runs" begin
        result = run_simulation(joinpath(K1_ROOT, "examples", "natural_convection.krk"))
        @test haskey(result, :Temp)
        @test all(isfinite.(result.ux))
        @test all(isfinite.(result.uy))
        @test all(isfinite.(result.Temp))
        @test :thermal in result.setup.modules
    end
end
