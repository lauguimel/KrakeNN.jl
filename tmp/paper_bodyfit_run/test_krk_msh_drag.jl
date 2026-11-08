using Test
using KrakeNN
using Gmsh
using KernelAbstractions

include(joinpath(@__DIR__, "..", "gen_ogrid_rect_8block.jl"))

@testset "paper .krk -> .msh -> SLBM drag smoke" begin
    mktempdir() do dir
        mesh_dir = joinpath(dir, "meshes")
        mkpath(mesh_dir)
        geo = joinpath(mesh_dir, "cylinder_ogrid.geo")
        msh = joinpath(mesh_dir, "cylinder_ogrid.msh")
        write_ogrid_rect_8block_geo(geo; Lx=2.2, Ly=0.41,
            cx_p=0.2, cy_p=0.205, R_in=0.05,
            N_arc=4, N_radial=6, radial_progression=0.9)

        gmsh.initialize()
        try
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.open(geo)
            gmsh.model.mesh.generate(2)
            gmsh.write(msh)
        finally
            gmsh.finalize()
        end

        krk = joinpath(dir, "case.krk")
        write(krk, """
        Simulation krk_msh_drag_smoke D2Q9
        Module slbm_drag
        Domain L = 2.2 x 0.41 N = 40 x 40
        Mesh gmsh(file = "meshes/cylinder_ogrid.msh", layout = topological, multiblock = true)
        Physics Re = 20 u_max = 0.04 cx = 0.2 cy = 0.205 R = 0.05 avg_window = 2 sample_every = 1 check_every = 1
        Boundary west velocity(ux = 0.04, uy = 0)
        Boundary east pressure(rho = 1.0)
        Boundary south wall
        Boundary north wall
        Run 2 steps
        """)

        setup = load_kraken(krk)
        @test setup.mesh !== nothing
        @test setup.mesh.file == msh

        result = run_simulation(setup; backend=KernelAbstractions.CPU(), T=Float64)
        @test result.blocks == 8
        @test result.nodes == 192
        @test isfinite(result.Cd)
        @test isfinite(result.Cl)
        @test abs(result.Cl) < 1e-8
        @test !isempty(result.history)
        @test isfinite(result.rho_min)
        @test isfinite(result.rho_max)
    end
end
