using Test
using KrakeNN

# =====================================================================
# v0.3 Phase B.2.3 — extended-grid mesh for multi-block SLBM.
# =====================================================================

@testset "Multi-block mesh extension (v0.3 Phase B.2.3)" begin

    @testset "extend_mesh_2d preserves interior, linearly extrapolates ghosts (Cartesian)" begin
        mesh = cartesian_mesh(; x_min=0.0, x_max=4.0, y_min=0.0, y_max=2.0,
                                Nx=5, Ny=3)
        mesh_ext = extend_mesh_2d(mesh; n_ghost=1)
        @test size(mesh_ext.X) == (7, 5)
        @test size(mesh_ext.Y) == (7, 5)
        # Interior matches.
        @test mesh_ext.X[2:6, 2:4] ≈ mesh.X
        @test mesh_ext.Y[2:6, 2:4] ≈ mesh.Y
        # West ghost: one-dx shift to the left (dx=1.0 for this mesh).
        @test mesh_ext.X[1, 2:4] ≈ mesh.X[1, :] .- 1.0
        # East ghost: one-dx shift to the right.
        @test mesh_ext.X[7, 2:4] ≈ mesh.X[end, :] .+ 1.0
        # South / north: dy=1.0.
        @test mesh_ext.Y[2:6, 1] ≈ mesh.Y[:, 1] .- 1.0
        @test mesh_ext.Y[2:6, 5] ≈ mesh.Y[:, end] .+ 1.0
        # Corner SW: bi-extrapolated via pass 2 from pass-1 west-ghost.
        @test mesh_ext.X[1, 1] ≈ -1.0
        @test mesh_ext.Y[1, 1] ≈ -1.0
    end

    @testset "extend_mesh_2d with n_ghost=2" begin
        mesh = cartesian_mesh(; x_min=0.0, x_max=3.0, y_min=0.0, y_max=3.0,
                                Nx=4, Ny=4)
        mesh_ext = extend_mesh_2d(mesh; n_ghost=2)
        @test size(mesh_ext.X) == (8, 8)
        # West ghosts k=1,2 at x = -1.0, -2.0
        @test mesh_ext.X[2, 3:6] ≈ mesh.X[1, :] .- 1.0
        @test mesh_ext.X[1, 3:6] ≈ mesh.X[1, :] .- 2.0
        @test mesh_ext.X[7, 3:6] ≈ mesh.X[end, :] .+ 1.0
        @test mesh_ext.X[8, 3:6] ≈ mesh.X[end, :] .+ 2.0
    end

    @testset "extend_mesh_2d n_ghost=0 returns input unchanged" begin
        mesh = cartesian_mesh(; x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0,
                                Nx=2, Ny=2)
        @test extend_mesh_2d(mesh; n_ghost=0) === mesh
    end

    @testset "build_block_slbm_geometry_extended returns shape-consistent geom" begin
        mesh = cartesian_mesh(; x_min=0.0, x_max=4.0, y_min=0.0, y_max=2.0,
                                Nx=5, Ny=3)
        blk = Block(:A, mesh; west=:w, east=:e, south=:s, north=:n)
        mesh_ext, geom = build_block_slbm_geometry_extended(blk; n_ghost=1,
                                                              local_cfl=false)
        @test mesh_ext.Nξ == 7 && mesh_ext.Nη == 5
        # The SLBMGeometry carries (Nξ_ext, Nη_ext, 9) stencil arrays.
        # We check via `i_dep` / `j_dep` (departure indices).
        @test size(geom.i_dep) == (7, 5, 9)
        @test size(geom.j_dep) == (7, 5, 9)
    end

    @testset "extend_interior_field_2d lifts 2D array" begin
        is_solid = rand(Bool, 3, 4)
        ext = extend_interior_field_2d(is_solid, 1; pad_value=false)
        @test size(ext) == (5, 6)
        @test ext[2:4, 2:5] == is_solid
        # Ghost cells all false (padding)
        @test all(.!ext[1, :]); @test all(.!ext[end, :])
        @test all(.!ext[:, 1]); @test all(.!ext[:, end])
    end

    @testset "extend_interior_field_2d lifts 3D array" begin
        q_wall = rand(Float64, 3, 4, 9)
        ext = extend_interior_field_2d(q_wall, 2)
        @test size(ext) == (7, 8, 9)
        @test ext[3:5, 3:6, :] == q_wall
        @test all(ext[1:2, :, :] .== 0)
        @test all(ext[:, 1:2, :] .== 0)
        @test all(ext[6:7, :, :] .== 0)
    end
end
