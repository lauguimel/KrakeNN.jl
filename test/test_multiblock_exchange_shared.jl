using Test
using KrakeNN

# =====================================================================
# v0.3 Phase B.2.1 — shared-node exchange kernel.
#
# `exchange_ghost_shared_node_2d!` is the natural-gmsh counterpart of
# `exchange_ghost_2d!`: when two adjacent blocks share an interface
# curve, their interface columns coincide physically (rather than
# being one-cell apart as in the non-overlap convention). The ghost
# fill must therefore skip the duplicated column and read from
# interior+2 (i.e. extended index ng+2) rather than interior+1.
#
# Tests mirror the non-overlap version but with colocated blocks:
#   A spans x ∈ [0, Nxp - 1] with Nxp cells
#   B spans x ∈ [Nxp - 1, 2(Nxp - 1)] — SHARED east edge at x = Nxp - 1
# Both blocks have Nxp physical columns; the column at i = Nxp in A
# and i = 1 in B represent the SAME physical cell.
# =====================================================================

@testset "Multi-block shared-node ghost exchange (v0.3 Phase B.2.1)" begin

    Nxp = 6; Nyp = 4

    function build_shared_node_mbm()
        mesh_A = cartesian_mesh(; x_min=0.0, x_max=Float64(Nxp - 1),
                                   y_min=0.0, y_max=Float64(Nyp - 1),
                                   Nx=Nxp, Ny=Nyp)
        mesh_B = cartesian_mesh(; x_min=Float64(Nxp - 1), x_max=Float64(2 * (Nxp - 1)),
                                   y_min=0.0, y_max=Float64(Nyp - 1),
                                   Nx=Nxp, Ny=Nyp)
        blk_A = Block(:A, mesh_A; west=:inlet, east=:interface,
                                   south=:wall, north=:wall)
        blk_B = Block(:B, mesh_B; west=:interface, east=:outlet,
                                   south=:wall, north=:wall)
        iface = Interface(; from=(:A, :east), to=(:B, :west))
        return MultiBlockMesh2D([blk_A, blk_B]; interfaces=[iface])
    end

    @testset "basic W-E shared-node fill (ng = 1)" begin
        mbm = build_shared_node_mbm()
        # Sanity: this MBM should be flagged shared-node with a warning,
        # not an error.
        issues = sanity_check_multiblock(mbm; verbose=false)
        @test all(iss -> iss.severity !== :error, issues)
        @test any(iss -> iss.severity === :warning &&
                          iss.code === :InterfaceEdgesColocated, issues)

        ng = 1
        states = [allocate_block_state_2d(b; n_ghost=ng) for b in mbm.blocks]
        # Tag interior columns so we can trace which value ended up where.
        Nx_ext_A, Ny_ext_A = ext_dims(states[1])
        Nx_ext_B, Ny_ext_B = ext_dims(states[2])
        @inbounds for q in 1:9, j in 1:Ny_ext_A, i in 1:Nx_ext_A
            states[1].f[i, j, q] = 100 + i + 0.1 * j + 0.01 * q
        end
        @inbounds for q in 1:9, j in 1:Ny_ext_B, i in 1:Nx_ext_B
            states[2].f[i, j, q] = 200 + i + 0.1 * j + 0.01 * q
        end

        exchange_ghost_shared_node_2d!(mbm, states)

        # A's east ghost at extended i = Nxp + ng + 1 = Nxp + 2 should
        # contain B's interior at extended i = ng + 2 = 3 (skipping the
        # duplicated shared column at ng + 1 = 2).
        j_range = (ng + 1):(ng + Nyp)
        for j in j_range, q in 1:9
            @test states[1].f[Nxp + ng + 1, j, q] == 200 + 3 + 0.1 * j + 0.01 * q
        end
        # B's west ghost at extended i = 1 should contain A's interior
        # at extended i = Nxp + ng - 1 = Nxp (skipping the duplicated
        # shared column at Nxp + ng = Nxp + 1).
        for j in j_range, q in 1:9
            @test states[2].f[1, j, q] == 100 + Nxp + 0.1 * j + 0.01 * q
        end
        # No other ghost rows should have been touched — check A's south,
        # north, west ghosts are unchanged.
        # A's west ghost at extended i = 1: untouched (tagged via 100+1...)
        for j in j_range, q in 1:9
            @test states[1].f[1, j, q] == 100 + 1 + 0.1 * j + 0.01 * q
        end
    end

    @testset "ng = 2 shared-node" begin
        mbm = build_shared_node_mbm()
        ng = 2
        states = [allocate_block_state_2d(b; n_ghost=ng) for b in mbm.blocks]
        Nx_ext_A, Ny_ext_A = ext_dims(states[1])
        Nx_ext_B, Ny_ext_B = ext_dims(states[2])
        @inbounds for q in 1:9, j in 1:Ny_ext_A, i in 1:Nx_ext_A
            states[1].f[i, j, q] = 100 + i + 0.1 * j + 0.01 * q
        end
        @inbounds for q in 1:9, j in 1:Ny_ext_B, i in 1:Nx_ext_B
            states[2].f[i, j, q] = 200 + i + 0.1 * j + 0.01 * q
        end

        exchange_ghost_shared_node_2d!(mbm, states)

        j_range = (ng + 1):(ng + Nyp)
        # A's east ghost k=1 (extended Nxp + ng + 1) ← B interior ng+2
        # A's east ghost k=2 (extended Nxp + ng + 2) ← B interior ng+3
        for k in 1:ng, j in j_range, q in 1:9
            @test states[1].f[Nxp + ng + k, j, q] == 200 + (ng + 1 + k) + 0.1 * j + 0.01 * q
        end
        # B's west ghost k=1 (extended 1) ← A interior Nxp + ng - 1
        # B's west ghost k=2 (extended 2) ← A interior Nxp + ng
        for k in 1:ng, j in j_range, q in 1:9
            # state_left source extended index: Nx_l + ng - 1 - (ng - k) = Nxp + k - 1
            @test states[2].f[k, j, q] == 100 + (Nxp + k - 1) + 0.1 * j + 0.01 * q
        end
    end

    @testset "S-N shared-node fill" begin
        mesh_A = cartesian_mesh(; x_min=0.0, x_max=Float64(Nxp - 1),
                                   y_min=0.0, y_max=Float64(Nyp - 1),
                                   Nx=Nxp, Ny=Nyp)
        mesh_B = cartesian_mesh(; x_min=0.0, x_max=Float64(Nxp - 1),
                                   y_min=Float64(Nyp - 1), y_max=Float64(2 * (Nyp - 1)),
                                   Nx=Nxp, Ny=Nyp)
        blk_A = Block(:A, mesh_A; west=:inlet, east=:outlet,
                                   south=:wall, north=:interface)
        blk_B = Block(:B, mesh_B; west=:inlet, east=:outlet,
                                   south=:interface, north=:wall)
        iface = Interface(; from=(:A, :north), to=(:B, :south))
        mbm = MultiBlockMesh2D([blk_A, blk_B]; interfaces=[iface])

        ng = 1
        states = [allocate_block_state_2d(b; n_ghost=ng) for b in mbm.blocks]
        @inbounds for q in 1:9, j in 1:size(states[1].f, 2), i in 1:size(states[1].f, 1)
            states[1].f[i, j, q] = 100 + i + 0.1 * j + 0.01 * q
        end
        @inbounds for q in 1:9, j in 1:size(states[2].f, 2), i in 1:size(states[2].f, 1)
            states[2].f[i, j, q] = 200 + i + 0.1 * j + 0.01 * q
        end

        exchange_ghost_shared_node_2d!(mbm, states)

        i_range = (ng + 1):(ng + Nxp)
        # A's north ghost (extended j = Nyp + ng + 1) ← B interior j = ng + 2
        for i in i_range, q in 1:9
            @test states[1].f[i, Nyp + ng + 1, q] == 200 + i + 0.1 * (ng + 2) + 0.01 * q
        end
        # B's south ghost (extended j = 1) ← A interior j = Nyp + ng - 1 = Nyp
        for i in i_range, q in 1:9
            @test states[2].f[i, 1, q] == 100 + i + 0.1 * Nyp + 0.01 * q
        end
    end

    @testset "mismatched n_ghost errors" begin
        mbm = build_shared_node_mbm()
        states = [allocate_block_state_2d(mbm.blocks[1]; n_ghost=1),
                  allocate_block_state_2d(mbm.blocks[2]; n_ghost=2)]
        @test_throws ErrorException exchange_ghost_shared_node_2d!(mbm, states)
    end
end
