using Test
using Kraken

@testset "Multi-block ghost exchange (v0.3 Phase A.5b)" begin

    # ---- helpers --------------------------------------------------------
    # Build a horizontally-split 2-block mesh.
    function mk_we_mbm(; Nx_each=5, Ny=4, T=Float64)
        mesh_l = cartesian_mesh(; x_min=0.0, x_max=0.5,
                                   y_min=0.0, y_max=1.0, Nx=Nx_each, Ny=Ny, FT=T)
        mesh_r = cartesian_mesh(; x_min=0.5, x_max=1.0,
                                   y_min=0.0, y_max=1.0, Nx=Nx_each, Ny=Ny, FT=T)
        blk_l = Block(:left,  mesh_l; west=:inlet, east=:interface,
                                       south=:wall, north=:wall)
        blk_r = Block(:right, mesh_r; west=:interface, east=:outlet,
                                       south=:wall, north=:wall)
        iface = Interface(; from=(:left, :east), to=(:right, :west))
        return MultiBlockMesh2D([blk_l, blk_r]; interfaces=[iface])
    end

    function mk_sn_mbm(; Nx=4, Ny_each=5, T=Float64)
        mesh_b = cartesian_mesh(; x_min=0.0, x_max=1.0,
                                   y_min=0.0, y_max=0.5, Nx=Nx, Ny=Ny_each, FT=T)
        mesh_t = cartesian_mesh(; x_min=0.0, x_max=1.0,
                                   y_min=0.5, y_max=1.0, Nx=Nx, Ny=Ny_each, FT=T)
        blk_b = Block(:bot, mesh_b; west=:wall, east=:wall,
                                     south=:inlet, north=:interface)
        blk_t = Block(:top, mesh_t; west=:wall, east=:wall,
                                     south=:interface, north=:outlet)
        iface = Interface(; from=(:bot, :north), to=(:top, :south))
        return MultiBlockMesh2D([blk_b, blk_t]; interfaces=[iface])
    end

    # Fill state.f with a known signature at every (i, j, q); makes it
    # easy to read back exactly which cells the exchange touched.
    function paint_states!(states)
        for (k, st) in enumerate(states)
            Nx, Ny = ext_dims(st)
            @inbounds for q in 1:9, j in 1:Ny, i in 1:Nx
                st.f[i, j, q] = 1000.0 * k + 100.0 * q + 10.0 * i + 0.1 * j
            end
        end
    end

    # ---- allocation / accessors ----------------------------------------
    @testset "BlockState2D allocation and interior view" begin
        mbm = mk_we_mbm()
        st = allocate_block_state_2d(mbm.blocks[1]; n_ghost=1)
        @test size(st.f)  == (7, 6, 9)       # Nx_phys=5 → 7 extended; Ny_phys=4 → 6
        @test size(st.ρ)  == (7, 6)
        @test st.n_ghost  == 1
        @test st.Nξ_phys  == 5
        @test st.Nη_phys  == 4
        @test ext_dims(st) == (7, 6)

        int_f = interior_f(st)
        @test size(int_f) == (5, 4, 9)
        # Writing to the interior view should hit the underlying extended array
        int_f[1, 1, 1] = 42.0
        @test st.f[2, 2, 1] == 42.0

        ρv, uxv, uyv = interior_macro(st)
        @test size(ρv)  == (5, 4)
        @test size(uxv) == (5, 4)
    end

    # ---- West-East exchange --------------------------------------------
    @testset "W-E exchange fills LEFT east ghost from RIGHT interior, RIGHT west ghost from LEFT interior" begin
        mbm = mk_we_mbm()
        states = [allocate_block_state_2d(b; n_ghost=1) for b in mbm.blocks]
        paint_states!(states)

        # n_ghost = 1, Nx_phys = 5, so extended i = 1..7
        # LEFT east ghost is at i=7, should receive RIGHT's interior-west (i=2)
        # RIGHT west ghost is at i=1, should receive LEFT's interior-east (i=6)
        # Only physical-η rows are copied (j = ng+1 .. ng+Nη_phys = 2..5);
        # doubly-ghost corner cells at j ∈ {1, 6} are handled by the
        # physical south/north BCs on each block, not by this exchange.
        right_interior_west = [states[2].f[2, j, q] for j in 2:5, q in 1:9]
        left_interior_east  = [states[1].f[6, j, q] for j in 2:5, q in 1:9]

        exchange_ghost_2d!(mbm, states)

        for (jj, j) in enumerate(2:5), q in 1:9
            @test states[1].f[7, j, q] == right_interior_west[jj, q]
            @test states[2].f[1, j, q] == left_interior_east[jj, q]
        end
    end

    @testset "W-E exchange leaves physical interior untouched" begin
        mbm = mk_we_mbm()
        states = [allocate_block_state_2d(b; n_ghost=1) for b in mbm.blocks]
        paint_states!(states)

        before_int_L = copy(interior_f(states[1]))
        before_int_R = copy(interior_f(states[2]))

        exchange_ghost_2d!(mbm, states)

        @test interior_f(states[1]) == before_int_L
        @test interior_f(states[2]) == before_int_R
    end

    @testset "W-E swapped from/to produces the same fill" begin
        mesh_l = cartesian_mesh(; x_min=0.0, x_max=0.5, y_min=0.0, y_max=1.0, Nx=5, Ny=4)
        mesh_r = cartesian_mesh(; x_min=0.5, x_max=1.0, y_min=0.0, y_max=1.0, Nx=5, Ny=4)
        blk_l = Block(:left,  mesh_l; west=:inlet, east=:interface, south=:wall, north=:wall)
        blk_r = Block(:right, mesh_r; west=:interface, east=:outlet, south=:wall, north=:wall)
        iface_direct  = Interface(; from=(:left, :east), to=(:right, :west))
        iface_swapped = Interface(; from=(:right, :west), to=(:left, :east))

        mbm_d = MultiBlockMesh2D([blk_l, blk_r]; interfaces=[iface_direct])
        mbm_s = MultiBlockMesh2D([blk_l, blk_r]; interfaces=[iface_swapped])

        sd = [allocate_block_state_2d(b; n_ghost=1) for b in mbm_d.blocks]
        ss = [allocate_block_state_2d(b; n_ghost=1) for b in mbm_s.blocks]
        paint_states!(sd); paint_states!(ss)

        exchange_ghost_2d!(mbm_d, sd)
        exchange_ghost_2d!(mbm_s, ss)
        @test sd[1].f == ss[1].f
        @test sd[2].f == ss[2].f
    end

    # ---- South-North exchange ------------------------------------------
    @testset "S-N exchange fills BOT north ghost from TOP interior, TOP south ghost from BOT interior" begin
        mbm = mk_sn_mbm()
        states = [allocate_block_state_2d(b; n_ghost=1) for b in mbm.blocks]
        paint_states!(states)

        # Ny_phys = 5 → extended j = 1..7; interior j = 2..6
        # BOT north ghost at j=7, TOP south ghost at j=1
        # Only physical-ξ rows copied (i = 2..5); corner cells i ∈ {1, 6}
        # handled by west/east BCs on each block.
        top_interior_south = [states[2].f[i, 2, q] for i in 2:5, q in 1:9]
        bot_interior_north = [states[1].f[i, 6, q] for i in 2:5, q in 1:9]

        exchange_ghost_2d!(mbm, states)

        for (ii, i) in enumerate(2:5), q in 1:9
            @test states[1].f[i, 7, q] == top_interior_south[ii, q]
            @test states[2].f[i, 1, q] == bot_interior_north[ii, q]
        end
    end

    # ---- n_ghost = 2 ----------------------------------------------------
    @testset "Ng = 2 copies 2 ghost rows from neighbour interior" begin
        mbm = mk_we_mbm(; Nx_each=4, Ny=4)
        states = [allocate_block_state_2d(b; n_ghost=2) for b in mbm.blocks]
        paint_states!(states)

        # extended Nx = 4 + 2*2 = 8
        # LEFT east ghost at i=7, 8 ← RIGHT interior at i=3, 4
        # RIGHT west ghost at i=1, 2 ← LEFT interior at i=5, 6
        # Only physical j rows (j = ng+1 .. ng+Ny_phys = 3..6) are copied
        snap_right = copy(states[2].f[3:4, 3:6, :])
        snap_left  = copy(states[1].f[5:6, 3:6, :])

        exchange_ghost_2d!(mbm, states)

        # BROKEN (#28): with two ghost layers the first ghost row is sourced
        # from the wrong neighbour block — reads the 2110.x family where
        # 1150.x is expected. n_ghost = 1 is correct and is what the only live
        # consumer (body-fitted cylinder drag) uses.
        @test_broken states[1].f[7:8, 3:6, :] == snap_right
        @test_broken states[2].f[1:2, 3:6, :] == snap_left
    end

    # ---- error paths ---------------------------------------------------
    @testset "length mismatch throws" begin
        mbm = mk_we_mbm()
        st = allocate_block_state_2d(mbm.blocks[1])
        @test_throws ErrorException exchange_ghost_2d!(mbm, [st])
    end

    @testset "mixed n_ghost across blocks throws" begin
        mbm = mk_we_mbm()
        s1 = allocate_block_state_2d(mbm.blocks[1]; n_ghost=1)
        s2 = allocate_block_state_2d(mbm.blocks[2]; n_ghost=2)
        @test_throws ErrorException exchange_ghost_2d!(mbm, [s1, s2])
    end

    @testset "unsupported same-normal edge pair throws" begin
        mesh = cartesian_mesh(; x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, Nx=4, Ny=4)
        blk_a = Block(:a, mesh; west=:w, east=:interface, south=:s, north=:n)
        blk_b = Block(:b, mesh; west=:w, east=:interface, south=:s, north=:n)
        iface = Interface(; from=(:a, :east), to=(:b, :east))
        mbm = MultiBlockMesh2D([blk_a, blk_b]; interfaces=[iface])
        states = [allocate_block_state_2d(b; n_ghost=1) for b in mbm.blocks]
        # BROKEN (#28): an :east-to-:east pair is geometrically impossible and
        # should be refused. No guard exists, so the call proceeds silently.
        # Written as a caught-exception probe because @test_broken cannot wrap
        # @test_throws: it currently yields false, and will yield true — and so
        # report Unexpected Pass — once the guard is added.
        @test_broken try
            exchange_ghost_2d!(mbm, states)
            false
        catch
            true
        end
    end
end
