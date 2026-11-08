using Test
using KrakeNN
using Gmsh

# ===========================================================================
# WP-MESH — gmsh / external-mesh import pipeline.
#
#   1. Phase A axis-aligned loader matches cartesian_mesh to 1e-12.
#   2. Phase A reads physical groups (inlet/outlet/wall/fluid).
#   3. Phase B topological loader (4-corner case) recovers a half-annulus.
#   4. CurvilinearMesh(X, Y) via BSpline+ForwardDiff matches polar_mesh
#      analytic mapping to ~1e-6 (interpolation accuracy).
# ===========================================================================

@testset "gmsh loader & BSpline mesh" begin

    # ------------------------------------------------------------------
    # 1. Axis-aligned Schäfer-Turek-style channel block
    # ------------------------------------------------------------------
    @testset "Phase A axis-aligned vs cartesian_mesh" begin
        Lx, Ly = 2.2, 0.41
        Nx_n, Ny_n = 31, 11

        mktemp() do path, _io
            fpath = path * ".msh"
            gmsh.initialize()
            try
                gmsh.option.setNumber("General.Terminal", 0)
                gmsh.model.add("st_block")
                gmsh.model.geo.addPoint(0.0, 0.0, 0.0, 0.1, 1)
                gmsh.model.geo.addPoint(Lx,  0.0, 0.0, 0.1, 2)
                gmsh.model.geo.addPoint(Lx,  Ly,  0.0, 0.1, 3)
                gmsh.model.geo.addPoint(0.0, Ly,  0.0, 0.1, 4)
                gmsh.model.geo.addLine(1, 2, 1)
                gmsh.model.geo.addLine(2, 3, 2)
                gmsh.model.geo.addLine(3, 4, 3)
                gmsh.model.geo.addLine(4, 1, 4)
                gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
                gmsh.model.geo.addPlaneSurface([1], 1)
                gmsh.model.geo.mesh.setTransfiniteCurve(1, Nx_n)
                gmsh.model.geo.mesh.setTransfiniteCurve(3, Nx_n)
                gmsh.model.geo.mesh.setTransfiniteCurve(2, Ny_n)
                gmsh.model.geo.mesh.setTransfiniteCurve(4, Ny_n)
                gmsh.model.geo.mesh.setTransfiniteSurface(1)
                gmsh.model.geo.mesh.setRecombine(2, 1)
                gmsh.model.geo.synchronize()
                gmsh.model.addPhysicalGroup(1, [4], 100, "inlet")
                gmsh.model.addPhysicalGroup(1, [2], 200, "outlet")
                gmsh.model.addPhysicalGroup(1, [1], 300, "south")
                gmsh.model.addPhysicalGroup(1, [3], 400, "north")
                gmsh.model.addPhysicalGroup(2, [1], 1000, "fluid")
                gmsh.model.mesh.generate(2)
                gmsh.write(fpath)
            finally
                gmsh.finalize()
            end

            mesh, groups = load_gmsh_mesh_2d(fpath)
            @test mesh.Nξ == Nx_n
            @test mesh.Nη == Ny_n
            @test minimum(mesh.X) ≈ 0.0 atol=1e-10
            @test maximum(mesh.X) ≈ Lx  atol=1e-10
            @test minimum(mesh.Y) ≈ 0.0 atol=1e-10
            @test maximum(mesh.Y) ≈ Ly  atol=1e-10

            mesh_ref = cartesian_mesh(; x_min=0.0, x_max=Lx,
                                        y_min=0.0, y_max=Ly,
                                        Nx=Nx_n, Ny=Ny_n)
            @test maximum(abs.(mesh.X .- mesh_ref.X)) < 1e-10
            @test maximum(abs.(mesh.Y .- mesh_ref.Y)) < 1e-10
            @test maximum(abs.(mesh.J .- mesh_ref.J)) < 1e-10

            # Physical-group readback
            @test groups.by_name["inlet"]  == 100
            @test groups.by_name["outlet"] == 200
            @test groups.by_name["south"]  == 300
            @test groups.by_name["north"]  == 400
            @test groups.by_name["fluid"]  == 1000
            @test groups.dim[1000] == 2
            @test groups.dim[100]  == 1
        end
    end

    # ------------------------------------------------------------------
    # 2. CurvilinearMesh(X, Y) BSpline+ForwardDiff vs polar_mesh
    # ------------------------------------------------------------------
    @testset "CurvilinearMesh from arrays vs polar_mesh" begin
        m_ref = polar_mesh(; cx=0.0, cy=0.0, r_inner=1.0, r_outer=2.0,
                             n_theta=64, n_r=17, FT=Float64)
        m_arr = CurvilinearMesh(m_ref.X, m_ref.Y;
                                 periodic_ξ=true, periodic_η=false,
                                 type=:from_polar)
        @test maximum(abs.(m_arr.X    .- m_ref.X))    < 1e-10
        @test maximum(abs.(m_arr.Y    .- m_ref.Y))    < 1e-10
        @test maximum(abs.(m_arr.dYdη .- m_ref.dYdη)) < 1e-10
        # Cubic spline on cos(2πξ): expected truncation ~1e-5
        @test maximum(abs.(m_arr.dXdξ .- m_ref.dXdξ)) < 1e-4
        @test maximum(abs.(m_arr.J    .- m_ref.J))    < 1e-4
    end

    # ------------------------------------------------------------------
    # 3. Phase B topological — 4-corner half-annulus
    # ------------------------------------------------------------------
    @testset "Phase B topological half-annulus (4 corners)" begin
        R_in, R_out = 1.0, 2.0
        N_θ_h, N_r = 32, 17

        mktemp() do path, _io
            fpath = path * ".msh"
            gmsh.initialize()
            try
                gmsh.option.setNumber("General.Terminal", 0)
                gmsh.model.add("half_annulus")
                gmsh.model.geo.addPoint(R_in,  0, 0, 0.1, 1)
                gmsh.model.geo.addPoint(-R_in, 0, 0, 0.1, 2)
                gmsh.model.geo.addPoint(R_out, 0, 0, 0.1, 3)
                gmsh.model.geo.addPoint(-R_out, 0, 0, 0.1, 4)
                gmsh.model.geo.addPoint(0, 0, 0, 0.1, 5)
                gmsh.model.geo.addCircleArc(1, 5, 2, 1)
                gmsh.model.geo.addCircleArc(3, 5, 4, 2)
                gmsh.model.geo.addLine(1, 3, 3)
                gmsh.model.geo.addLine(2, 4, 4)
                gmsh.model.geo.addCurveLoop([3, 2, -4, -1], 1)
                gmsh.model.geo.addPlaneSurface([1], 1)
                gmsh.model.geo.mesh.setTransfiniteCurve(1, N_θ_h)
                gmsh.model.geo.mesh.setTransfiniteCurve(2, N_θ_h)
                gmsh.model.geo.mesh.setTransfiniteCurve(3, N_r)
                gmsh.model.geo.mesh.setTransfiniteCurve(4, N_r)
                gmsh.model.geo.mesh.setTransfiniteSurface(1)
                gmsh.model.geo.mesh.setRecombine(2, 1)
                gmsh.model.geo.synchronize()
                gmsh.model.addPhysicalGroup(1, [1], 100, "inner")
                gmsh.model.addPhysicalGroup(1, [2], 200, "outer")
                gmsh.model.addPhysicalGroup(1, [3, 4], 300, "seam")
                gmsh.model.addPhysicalGroup(2, [1], 1000, "fluid")
                gmsh.model.mesh.generate(2)
                gmsh.write(fpath)
            finally
                gmsh.finalize()
            end

            mesh, groups = load_gmsh_mesh_2d(fpath; layout=:topological,
                                              periodic_ξ=false, periodic_η=false)
            @test mesh.Nξ * mesh.Nη == N_θ_h * N_r
            @test minimum(mesh.J) > 0.0    # no folds
            @test groups.by_name["inner"] == 100
            @test groups.by_name["outer"] == 200
            # All nodes within the (R_in, R_out) ring
            for j in 1:mesh.Nη, i in 1:mesh.Nξ
                r = hypot(mesh.X[i, j], mesh.Y[i, j])
                @test R_in - 1e-8 ≤ r ≤ R_out + 1e-8
            end
        end
    end
end
