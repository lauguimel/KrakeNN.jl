using Test
using KrakeNN

@testset "Multi-block topology (v0.3 Phase A.1)" begin

    # ---- helpers --------------------------------------------------------
    # Two blocks forming a 2 × 1 horizontal split: left block [0, 0.5] ×
    # [0, 1], right block [0.5, 1.0] × [0, 1]. Shared edge at x = 0.5,
    # aligned indices.
    mk_two_blocks(; Nx=5, Ny=4, T=Float64) = begin
        mesh_left  = cartesian_mesh(; x_min=0.0, x_max=0.5,
                                       y_min=0.0, y_max=1.0, Nx=Nx, Ny=Ny, FT=T)
        mesh_right = cartesian_mesh(; x_min=0.5, x_max=1.0,
                                       y_min=0.0, y_max=1.0, Nx=Nx, Ny=Ny, FT=T)
        blk_l = Block(:left, mesh_left;
                      west=:inlet, east=:interface,
                      south=:wall, north=:wall)
        blk_r = Block(:right, mesh_right;
                      west=:interface, east=:outlet,
                      south=:wall, north=:wall)
        iface = Interface(; from=(:left, :east), to=(:right, :west))
        return blk_l, blk_r, iface
    end

    # ---- Block construction --------------------------------------------
    @testset "Block constructor stores all four edge tags" begin
        mesh = cartesian_mesh(; x_min=0.0, x_max=1.0,
                                y_min=0.0, y_max=1.0, Nx=4, Ny=4)
        blk = Block(:only, mesh;
                     west=:a, east=:b, south=:c, north=:d)
        @test blk.id == :only
        @test blk.boundary_tags.west  == :a
        @test blk.boundary_tags.east  == :b
        @test blk.boundary_tags.south == :c
        @test blk.boundary_tags.north == :d
    end

    @testset "edge_length is Nη for west/east, Nξ for south/north" begin
        mesh = cartesian_mesh(; x_min=0.0, x_max=1.0,
                                y_min=0.0, y_max=2.0, Nx=6, Ny=11)
        blk = Block(:b, mesh; west=:w, east=:w, south=:w, north=:w)
        @test edge_length(blk, :west)  == 11
        @test edge_length(blk, :east)  == 11
        @test edge_length(blk, :south) == 6
        @test edge_length(blk, :north) == 6
        @test_throws ErrorException edge_length(blk, :up)
    end

    @testset "edge_coords returns nodes along the edge" begin
        mesh = cartesian_mesh(; x_min=0.0, x_max=1.0,
                                y_min=0.0, y_max=1.0, Nx=3, Ny=3)
        blk = Block(:b, mesh; west=:w, east=:w, south=:w, north=:w)
        xs, ys = edge_coords(blk, :west)
        @test length(xs) == 3 && length(ys) == 3
        @test all(xs .≈ 0.0)
        @test ys ≈ [0.0, 0.5, 1.0]
        xs, ys = edge_coords(blk, :east)
        @test all(xs .≈ 1.0)
        xs, ys = edge_coords(blk, :south)
        @test all(ys .≈ 0.0)
        xs, ys = edge_coords(blk, :north)
        @test all(ys .≈ 1.0)
    end

    # ---- MultiBlockMesh2D + getblock -----------------------------------
    @testset "MultiBlockMesh2D stores blocks + id lookup" begin
        blk_l, blk_r, iface = mk_two_blocks()
        mbm = MultiBlockMesh2D([blk_l, blk_r]; interfaces=[iface])
        @test length(mbm.blocks) == 2
        @test length(mbm.interfaces) == 1
        @test getblock(mbm, :left).id == :left
        @test getblock(mbm, :right).id == :right
        @test_throws KeyError getblock(mbm, :nonexistent)
    end

    @testset "MultiBlockMesh2D empty blocks rejected" begin
        @test_throws ErrorException MultiBlockMesh2D(Block[])
    end

    # ---- sanity_check_multiblock: HAPPY path ---------------------------
    @testset "sanity passes on well-formed 2-block mesh" begin
        blk_l, blk_r, iface = mk_two_blocks()
        mbm = MultiBlockMesh2D([blk_l, blk_r]; interfaces=[iface])
        issues = sanity_check_multiblock(mbm; verbose=false)
        # Shared-node topology (edges colocated) now raises a :warning
        # about the exchange semantics. No :error issues expected.
        @test all(iss -> iss.severity !== :error, issues)
    end

    # ---- sanity: individual failure modes ------------------------------
    @testset "NoDuplicateBlockIDs catches repeated id" begin
        blk_l, blk_r, iface = mk_two_blocks()
        blk_r2 = Block(:left, blk_r.mesh;   # same :left id as the first
                        west=:interface, east=:outlet,
                        south=:wall, north=:wall)
        mbm = MultiBlockMesh2D([blk_l, blk_r2]; interfaces=Interface[])
        issues = sanity_check_multiblock(mbm; verbose=false)
        @test any(iss -> iss.code === :NoDuplicateBlockIDs, issues)
    end

    @testset "InterfaceRefsExist flags unknown block id" begin
        blk_l, _, _ = mk_two_blocks()
        bad_iface = Interface(; from=(:left, :east), to=(:ghost, :west))
        mbm = MultiBlockMesh2D([blk_l]; interfaces=[bad_iface])
        issues = sanity_check_multiblock(mbm; verbose=false)
        @test any(iss -> iss.code === :InterfaceRefsExist, issues)
    end

    @testset "InterfaceRefsExist flags unknown edge" begin
        blk_l, blk_r, _ = mk_two_blocks()
        bad_iface = Interface(; from=(:left, :EAST), to=(:right, :west))
        mbm = MultiBlockMesh2D([blk_l, blk_r]; interfaces=[bad_iface])
        issues = sanity_check_multiblock(mbm; verbose=false)
        @test any(iss -> iss.code === :InterfaceRefsExist, issues)
    end

    @testset "InterfaceBothEdgesMarked: tag must be :interface" begin
        mesh_l = cartesian_mesh(; x_min=0.0, x_max=0.5, y_min=0.0, y_max=1.0, Nx=4, Ny=4)
        mesh_r = cartesian_mesh(; x_min=0.5, x_max=1.0, y_min=0.0, y_max=1.0, Nx=4, Ny=4)
        # Intentionally tag east of :left with :wall instead of :interface
        blk_l = Block(:left,  mesh_l; west=:inlet, east=:wall,
                                       south=:wall, north=:wall)
        blk_r = Block(:right, mesh_r; west=:interface, east=:outlet,
                                       south=:wall, north=:wall)
        iface = Interface(; from=(:left, :east), to=(:right, :west))
        mbm = MultiBlockMesh2D([blk_l, blk_r]; interfaces=[iface])
        issues = sanity_check_multiblock(mbm; verbose=false)
        @test any(iss -> iss.code === :InterfaceBothEdgesMarked, issues)
    end

    @testset "InterfaceEdgesSameLength: Nη of east ≠ Nη of west" begin
        mesh_l = cartesian_mesh(; x_min=0.0, x_max=0.5, y_min=0.0, y_max=1.0, Nx=4, Ny=4)
        mesh_r = cartesian_mesh(; x_min=0.5, x_max=1.0, y_min=0.0, y_max=1.0, Nx=4, Ny=5) # Ny=5
        blk_l = Block(:left,  mesh_l; west=:inlet, east=:interface,
                                       south=:wall, north=:wall)
        blk_r = Block(:right, mesh_r; west=:interface, east=:outlet,
                                       south=:wall, north=:wall)
        iface = Interface(; from=(:left, :east), to=(:right, :west))
        mbm = MultiBlockMesh2D([blk_l, blk_r]; interfaces=[iface])
        issues = sanity_check_multiblock(mbm; verbose=false)
        @test any(iss -> iss.code === :InterfaceEdgesSameLength, issues)
    end

    @testset "InterfaceEdgesColocated: shifted edge fails" begin
        mesh_l = cartesian_mesh(; x_min=0.0, x_max=0.5, y_min=0.0, y_max=1.0, Nx=4, Ny=4)
        # right block shifted vertically — the shared edge is now at a
        # different y range, so colocation fails.
        mesh_r = cartesian_mesh(; x_min=0.5, x_max=1.0, y_min=0.1, y_max=1.1, Nx=4, Ny=4)
        blk_l = Block(:left,  mesh_l; west=:inlet, east=:interface,
                                       south=:wall, north=:wall)
        blk_r = Block(:right, mesh_r; west=:interface, east=:outlet,
                                       south=:wall, north=:wall)
        iface = Interface(; from=(:left, :east), to=(:right, :west))
        mbm = MultiBlockMesh2D([blk_l, blk_r]; interfaces=[iface])
        issues = sanity_check_multiblock(mbm; verbose=false)
        @test any(iss -> iss.code === :InterfaceEdgesColocated, issues)
    end

    @testset "InterfaceEveryMarkedEdgeUsed: unused :interface edge" begin
        mesh = cartesian_mesh(; x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, Nx=4, Ny=4)
        blk = Block(:solo, mesh; west=:inlet, east=:interface,  # :interface but nobody connects
                                  south=:wall, north=:wall)
        mbm = MultiBlockMesh2D([blk]; interfaces=Interface[])
        issues = sanity_check_multiblock(mbm; verbose=false)
        @test any(iss -> iss.code === :InterfaceEveryMarkedEdgeUsed, issues)
    end

    @testset "UniformElementType: mixed FT rejected" begin
        mesh_l = cartesian_mesh(; x_min=0.0, x_max=0.5, y_min=0.0, y_max=1.0,
                                   Nx=4, Ny=4, FT=Float64)
        mesh_r = cartesian_mesh(; x_min=0.5, x_max=1.0, y_min=0.0, y_max=1.0,
                                   Nx=4, Ny=4, FT=Float32)
        blk_l = Block(:left,  mesh_l; west=:inlet, east=:interface,
                                       south=:wall, north=:wall)
        blk_r = Block(:right, mesh_r; west=:interface, east=:outlet,
                                       south=:wall, north=:wall)
        iface = Interface(; from=(:left, :east), to=(:right, :west))
        # Constructor parameterises on first-block type; second block
        # of different type will be rejected by `typed_blocks` cast.
        @test_throws MethodError MultiBlockMesh2D([blk_l, blk_r]; interfaces=[iface])
    end
end
