using Test
using KrakeNN
using KrakeNN: INTERFACE_TAG

# =====================================================================
# v0.3 Phase B.2.2 — block reorientation + BFS autoreorient.
# =====================================================================

@testset "Multi-block reorientation (v0.3 Phase B.2.2)" begin

    @testset "reorient_block flip_ξ: west ↔ east tags + mesh mirrored" begin
        mesh = cartesian_mesh(; x_min=0.0, x_max=3.0, y_min=0.0, y_max=1.0,
                                Nx=4, Ny=3)
        blk = Block(:A, mesh; west=:inlet, east=:outlet,
                              south=:wall, north=:wall)
        blk_f = reorient_block(blk; flip_ξ=true)
        # Tags swapped
        @test blk_f.boundary_tags.west  === :outlet
        @test blk_f.boundary_tags.east  === :inlet
        @test blk_f.boundary_tags.south === :wall
        @test blk_f.boundary_tags.north === :wall
        # X reversed along ξ: X[1, :] of flipped = X[Nξ, :] of original
        # (atol tolerates 1e-16 drift from the spline refit.)
        @test isapprox(blk_f.mesh.X[1, :], blk.mesh.X[end, :]; atol=1e-10)
        @test isapprox(blk_f.mesh.X[end, :], blk.mesh.X[1, :]; atol=1e-10)
        # Y unchanged along ξ (uniform mesh)
        @test isapprox(blk_f.mesh.Y, blk.mesh.Y; atol=1e-10)
        @test blk_f.mesh.Nξ == blk.mesh.Nξ
        @test blk_f.mesh.Nη == blk.mesh.Nη
    end

    @testset "reorient_block flip_η: south ↔ north tags + mesh mirrored" begin
        mesh = cartesian_mesh(; x_min=0.0, x_max=2.0, y_min=0.0, y_max=2.0,
                                Nx=3, Ny=5)
        blk = Block(:B, mesh; west=:wall, east=:wall,
                              south=:floor, north=:ceiling)
        blk_f = reorient_block(blk; flip_η=true)
        @test blk_f.boundary_tags.south === :ceiling
        @test blk_f.boundary_tags.north === :floor
        @test isapprox(blk_f.mesh.Y[:, 1], blk.mesh.Y[:, end]; atol=1e-10)
        @test isapprox(blk_f.mesh.Y[:, end], blk.mesh.Y[:, 1]; atol=1e-10)
    end

    @testset "reorient_block flip_both: 180° rotation" begin
        mesh = cartesian_mesh(; x_min=0.0, x_max=2.0, y_min=0.0, y_max=1.0,
                                Nx=4, Ny=3)
        blk = Block(:C, mesh; west=:w, east=:e, south=:s, north=:n)
        blk_rot = reorient_block(blk; flip_ξ=true, flip_η=true)
        @test blk_rot.boundary_tags.west  === :e
        @test blk_rot.boundary_tags.east  === :w
        @test blk_rot.boundary_tags.south === :n
        @test blk_rot.boundary_tags.north === :s
        # Mesh: X_new[i, j] = X[Nξ+1-i, Nη+1-j]
        Nξ, Nη = blk.mesh.Nξ, blk.mesh.Nη
        for j in 1:Nη, i in 1:Nξ
            @test isapprox(blk_rot.mesh.X[i, j], blk.mesh.X[Nξ+1-i, Nη+1-j]; atol=1e-10)
        end
    end

    @testset "reorient_block identity (no flip) returns same block" begin
        mesh = cartesian_mesh(; x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0,
                                Nx=3, Ny=3)
        blk = Block(:D, mesh; west=:w, east=:e, south=:s, north=:n)
        @test reorient_block(blk) === blk
    end

    # ---- BFS autoreorient: 2-block non-overlap canal, already aligned ----
    @testset "autoreorient on already-canonical 2-block canal: no change" begin
        mesh_A = cartesian_mesh(; x_min=0.0, x_max=4.0, y_min=0.0, y_max=4.0, Nx=5, Ny=5)
        mesh_B = cartesian_mesh(; x_min=5.0, x_max=9.0, y_min=0.0, y_max=4.0, Nx=5, Ny=5)
        blk_A = Block(:A, mesh_A; west=:inlet, east=:interface, south=:wall, north=:wall)
        blk_B = Block(:B, mesh_B; west=:interface, east=:outlet, south=:wall, north=:wall)
        mbm = MultiBlockMesh2D([blk_A, blk_B];
                                interfaces=[Interface(; from=(:A, :east), to=(:B, :west))])
        mbm2 = autoreorient_blocks(mbm; verbose=false)
        @test length(mbm2.blocks) == 2
        @test length(mbm2.interfaces) == 1
        # Tags preserved
        @test mbm2.blocks[1].boundary_tags === blk_A.boundary_tags
        @test mbm2.blocks[2].boundary_tags === blk_B.boundary_tags
        @test all(iss -> iss.severity !== :error,
                   sanity_check_multiblock(mbm2; verbose=false))
    end

    # ---- BFS autoreorient: 2-block with B flipped → BFS fixes it ----
    @testset "autoreorient on 2-block canal with B flipped along ξ" begin
        mesh_A = cartesian_mesh(; x_min=0.0, x_max=4.0, y_min=0.0, y_max=4.0, Nx=5, Ny=5)
        mesh_B = cartesian_mesh(; x_min=5.0, x_max=9.0, y_min=0.0, y_max=4.0, Nx=5, Ny=5)
        blk_A = Block(:A, mesh_A; west=:inlet, east=:interface, south=:wall, north=:wall)
        # Flip B's edge tags manually to simulate the topological-walker mess:
        # the shared curve is now labelled :east in B (should be :west after flip)
        # and the other labels rotate accordingly.
        blk_B_orig = Block(:B, mesh_B; west=:interface, east=:outlet, south=:wall, north=:wall)
        blk_B_flipped = reorient_block(blk_B_orig; flip_ξ=true)
        @test blk_B_flipped.boundary_tags.east === :interface
        mbm = MultiBlockMesh2D([blk_A, blk_B_flipped];
                                interfaces=[Interface(; from=(:A, :east), to=(:B, :east))])
        # Before autoreorient: the interface is east↔east (same-normal),
        # which the exchange kernel can't handle. After autoreorient: clean.
        mbm2 = autoreorient_blocks(mbm; verbose=false)
        @test all(iss -> iss.severity !== :error,
                   sanity_check_multiblock(mbm2; verbose=false))
        # The interface should now reference B's :west (canonical).
        a_edge = mbm2.interfaces[1].from[2]
        b_edge = mbm2.interfaces[1].to[2]
        @test a_edge === :east
        @test b_edge === :west
        @test mbm2.blocks[2].boundary_tags.east === :outlet
    end

    @testset "autoreorient keeps physical inlet/outlet on west/east" begin
        mesh_A = cartesian_mesh(; x_min=0.0, x_max=2.0, y_min=0.0, y_max=1.0, Nx=5, Ny=5)
        mesh_B = cartesian_mesh(; x_min=2.0, x_max=4.0, y_min=0.0, y_max=1.0, Nx=5, Ny=5)
        blk_A = Block(:A, mesh_A; west=:inlet, east=:interface,
                                south=:wall_bot, north=:wall_top)
        blk_B = Block(:B, mesh_B; west=:interface, east=:outlet,
                                south=:wall_bot, north=:wall_top)

        # Simulate a Gmsh topological walk that loaded the left block with
        # physical inlet on logical north and the interface on logical south.
        blk_A_bad = reorient_block(blk_A; transpose=true, flip_η=true)
        @test blk_A_bad.boundary_tags.north === :inlet
        @test blk_A_bad.boundary_tags.south === :interface

        mbm = MultiBlockMesh2D([blk_A_bad, blk_B];
                                interfaces=[Interface(; from=(:A, :south), to=(:B, :west))])
        mbm2 = autoreorient_blocks(mbm; verbose=false)
        issues = sanity_check_multiblock(mbm2; verbose=false)
        @test all(iss -> iss.severity !== :error, issues)
        @test mbm2.blocks[1].boundary_tags.west === :inlet
        @test mbm2.blocks[1].boundary_tags.east === :interface
        @test mbm2.blocks[1].boundary_tags.south === :wall_bot
        @test mbm2.blocks[1].boundary_tags.north === :wall_top
        @test mbm2.blocks[2].boundary_tags.west === :interface
        @test mbm2.blocks[2].boundary_tags.east === :outlet
        @test mbm2.interfaces[1].from === (:A, :east)
        @test mbm2.interfaces[1].to === (:B, :west)
    end

    # ---- BFS autoreorient: 3-block chain reorients all non-root blocks ----
    @testset "autoreorient on 3-block chain with middle block rotated" begin
        mesh_A = cartesian_mesh(; x_min=0.0, x_max=2.0, y_min=0.0, y_max=2.0, Nx=3, Ny=3)
        mesh_B = cartesian_mesh(; x_min=3.0, x_max=5.0, y_min=0.0, y_max=2.0, Nx=3, Ny=3)
        mesh_C = cartesian_mesh(; x_min=6.0, x_max=8.0, y_min=0.0, y_max=2.0, Nx=3, Ny=3)
        blk_A = Block(:A, mesh_A; west=:inlet, east=:interface, south=:wall, north=:wall)
        blk_B = Block(:B, mesh_B; west=:interface, east=:interface, south=:wall, north=:wall)
        blk_C = Block(:C, mesh_C; west=:interface, east=:outlet, south=:wall, north=:wall)
        # Flip B end-to-end (180° rotation) to mess up orientations.
        blk_B_flipped = reorient_block(blk_B; flip_ξ=true, flip_η=true)
        mbm = MultiBlockMesh2D([blk_A, blk_B_flipped, blk_C];
                                interfaces=[
                                    Interface(; from=(:A, :east), to=(:B, :east)),
                                    Interface(; from=(:B, :west), to=(:C, :west)),
                                ])
        mbm2 = autoreorient_blocks(mbm; verbose=false)
        issues = sanity_check_multiblock(mbm2; verbose=false)
        @test all(iss -> iss.severity !== :error, issues)
        # After reorient, A's east couples with B's west and B's east couples with C's west.
        @test any(i -> i.from === (:A, :east) && i.to === (:B, :west), mbm2.interfaces)
        @test any(i -> i.from === (:B, :east) && i.to === (:C, :west), mbm2.interfaces)
    end
end
