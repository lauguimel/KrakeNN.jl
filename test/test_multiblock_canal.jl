using Test
using KrakeNN

# =====================================================================
# Phase A.5b canal validation with ghost-layer f arrays.
#
# With pre-step ghost-fill on extended block arrays, each block's step
# reads VALID neighbour data at the interior-boundary (no halfway-BB
# fallback) and the interface is bit-exact vs single-block.
#
# What's NOT yet implemented: a helper that pre-fills PHYSICAL-wall
# ghosts with halfway-BB reflection. Without that, rows adjacent to a
# physical wall still differ from single-block. For this test we use
# a CLEAN configuration with no y-walls (south/north tagged as
# :interface → we close the y-direction by wrapping with a small
# periodic self-block, but easier: we just omit south/north BCs and
# let the kernel's halfway-BB fire at the extended outer ghosts where
# nothing reads it). We check:
#
# 1. Ghost fill + step preserves uniform rest state bit-exact (sanity).
# 2. After step with uniform-flow init, the two blocks' interior-boundary
#    COLUMNS at the x-interface agree EXACTLY (the ghost fill makes them
#    consistent — halo-strict couldn't achieve this at u ≠ 0).
# 3. Long-run smoke: 1000 steps with ghost-layer pipeline stays finite.
# =====================================================================

@testset "Multi-block canal with ghost layer (v0.3 Phase A.5b)" begin

    # Test geometry: left = [0, 5] × [0, 4],  right = [5, 10] × [0, 4].
    # Both blocks same Ny_phys = 5, Nx_phys = 5 (= 5 cells interior each).
    # For bit-exact vs single-block we'd also need physical-wall ghost
    # fill — scope deferred; here we assert INTERFACE consistency only.
    Nx_phys = 5; Ny_phys = 5
    ν = 0.1; ω = 1.0 / (3ν + 0.5)

    function build_mbm()
        # Non-overlap: A spans [0, 4], B spans [5, 9], both Nxp=5 cells
        mesh_A = cartesian_mesh(; x_min=0.0, x_max=4.0, y_min=0.0, y_max=4.0,
                                   Nx=Nx_phys, Ny=Ny_phys)
        mesh_B = cartesian_mesh(; x_min=5.0, x_max=9.0, y_min=0.0, y_max=4.0,
                                   Nx=Nx_phys, Ny=Ny_phys)
        blk_A = Block(:A, mesh_A; west=:inlet, east=:interface,
                                   south=:wall, north=:wall)
        blk_B = Block(:B, mesh_B; west=:interface, east=:outlet,
                                   south=:wall, north=:wall)
        iface = Interface(; from=(:A, :east), to=(:B, :west))
        return MultiBlockMesh2D([blk_A, blk_B]; interfaces=[iface])
    end

    function init_uniform_eq!(state::BlockState2D, ρ0, ux0, uy0)
        usq = ux0 * ux0 + uy0 * uy0
        Nx_ext, Ny_ext = ext_dims(state)
        @inbounds for j in 1:Ny_ext, i in 1:Nx_ext, q in 1:9
            state.f[i, j, q] = KrakeNN.feq_2d(Val(q), ρ0, ux0, uy0, usq)
        end
    end

    # ---- 1. Rest (u = 0) uniform remains uniform ---------------------
    @testset "u = 0 uniform equilibrium is preserved by ghost-layer step" begin
        mbm = build_mbm()
        @test all(iss -> iss.severity !== :error,
                   sanity_check_multiblock(mbm; verbose=false))

        states = [allocate_block_state_2d(b; n_ghost=1) for b in mbm.blocks]
        init_uniform_eq!(states[1], 1.0, 0.0, 0.0)
        init_uniform_eq!(states[2], 1.0, 0.0, 0.0)

        f_out_A = similar(states[1].f); f_out_B = similar(states[2].f)
        is_solid_A = zeros(Bool, ext_dims(states[1])...)
        is_solid_B = zeros(Bool, ext_dims(states[2])...)

        for step in 1:10
            exchange_ghost_2d!(mbm, states)
            Nx_ext_A, Ny_ext_A = ext_dims(states[1])
            Nx_ext_B, Ny_ext_B = ext_dims(states[2])
            fused_bgk_step!(f_out_A, states[1].f, states[1].ρ, states[1].ux, states[1].uy,
                            is_solid_A, Nx_ext_A, Ny_ext_A, ω)
            fused_bgk_step!(f_out_B, states[2].f, states[2].ρ, states[2].ux, states[2].uy,
                            is_solid_B, Nx_ext_B, Ny_ext_B, ω)
            states[1].f, f_out_A = f_out_A, states[1].f
            states[2].f, f_out_B = f_out_B, states[2].f
        end

        # At u = 0 the feq is symmetric in ±x/±y so halfway-BB at outer
        # ghosts gives the same value as the init; interior should still
        # be exactly the initial equilibrium.
        ref = KrakeNN.feq_2d(Val(1), 1.0, 0.0, 0.0, 0.0)
        int_A = interior_f(states[1])
        int_B = interior_f(states[2])
        @test all(int_A[:, :, 1] .≈ ref)
        @test all(int_B[:, :, 1] .≈ ref)
        # Interface columns agree bit-exact in A and B
        @test int_A[Nx_phys, :, :] == int_B[1, :, :]
    end

    # ---- 2. u ≠ 0: step 1 interface bit-exact ----------------------
    # The ghost-layer exchange correctly fills A's east ghost from B's
    # interior-west (and vice versa) so the step at the physical-
    # interface column reads VALID populations on both sides. After ONE
    # step starting from uniform flow, A's interior-east column and B's
    # interior-west column match bit-exact.
    #
    # Multi-step bit-exactness additionally requires pre-filling
    # PHYSICAL-WALL ghosts with halfway-BB reflection so the wall-side
    # pollution does not propagate inward. That helper is deferred.
    @testset "u = 0.05 step-1 interface bit-exact match" begin
        mbm = build_mbm()
        @test all(iss -> iss.severity !== :error,
                   sanity_check_multiblock(mbm; verbose=false))
        u0 = 0.05
        states = [allocate_block_state_2d(b; n_ghost=1) for b in mbm.blocks]
        init_uniform_eq!(states[1], 1.0, u0, 0.0)
        init_uniform_eq!(states[2], 1.0, u0, 0.0)
        f_out_A = similar(states[1].f); f_out_B = similar(states[2].f)
        is_solid_A = zeros(Bool, ext_dims(states[1])...)
        is_solid_B = zeros(Bool, ext_dims(states[2])...)

        # Single step; post-step, A's interior-east column at
        # (i = Nx_phys + n_ghost) and B's interior-west column at
        # (i = n_ghost + 1) must match bit-exactly. This is the fix
        # over halo-strict (cf. test_multiblock_canal old version in git)
        # where the interface agreement was machine-epsilon garbage due
        # to both blocks computing BGK on halfway-BB-polluted moments.
        exchange_ghost_2d!(mbm, states)
        Nx_ext_A, Ny_ext_A = ext_dims(states[1])
        Nx_ext_B, Ny_ext_B = ext_dims(states[2])
        fused_bgk_step!(f_out_A, states[1].f, states[1].ρ, states[1].ux, states[1].uy,
                        is_solid_A, Nx_ext_A, Ny_ext_A, ω)
        fused_bgk_step!(f_out_B, states[2].f, states[2].ρ, states[2].ux, states[2].uy,
                        is_solid_B, Nx_ext_B, Ny_ext_B, ω)
        ng = states[1].n_ghost
        a_col = view(f_out_A, Nx_phys + ng, (ng + 1):(ng + Ny_phys), :)
        b_col = view(f_out_B, ng + 1,        (ng + 1):(ng + Ny_phys), :)
        @test a_col == b_col     # bit-exact: identical ops on identical data
    end

    # ---- 2b. u ≠ 0 bit-exact multi-step with wall ghost fill (A.5c) --
    # With BOTH interface-ghost (exchange_ghost_2d!) AND wall-ghost fill
    # (fill_physical_wall_ghost_2d!) applied pre-step, each block's step
    # reads valid ghosts everywhere. Multi-block is then bit-exact to
    # single-block after any number of steps — validates Phase A.5c.
    @testset "u = 0.05 multi-step bit-exact with wall-ghost fill (20 steps)" begin
        # Non-overlap geometry (required by exchange_ghost_2d!):
        # A owns x ∈ [0, Nxp-1] with Nxp cells, B owns x ∈ [Nxp, 2Nxp-1]
        # with Nxp cells. Total 2*Nxp cells, all distinct. Single-block
        # reference spans [0, 2Nxp-1] with 2*Nxp cells.
        Nx_ref = 2 * Nx_phys; Ny_ref = Ny_phys
        u0 = 0.05
        function init_eq_plain(Nx, Ny, ρ, ux, uy)
            f = zeros(Float64, Nx, Ny, 9)
            usq = ux*ux + uy*uy
            @inbounds for j in 1:Ny, i in 1:Nx, q in 1:9
                f[i, j, q] = KrakeNN.feq_2d(Val(q), ρ, ux, uy, usq)
            end
            return f
        end
        fref_in  = init_eq_plain(Nx_ref, Ny_ref, 1.0, u0, 0.0)
        fref_out = similar(fref_in)
        ρref = ones(Nx_ref, Ny_ref); uxref = fill(u0, Nx_ref, Ny_ref); uyref = zeros(Nx_ref, Ny_ref)
        is_solid_ref = zeros(Bool, Nx_ref, Ny_ref)

        mesh_A = cartesian_mesh(; x_min=0.0,                 x_max=Float64(Nx_phys - 1),
                                   y_min=0.0,                y_max=Float64(Ny_phys - 1),
                                   Nx=Nx_phys, Ny=Ny_phys)
        mesh_B = cartesian_mesh(; x_min=Float64(Nx_phys),    x_max=Float64(2 * Nx_phys - 1),
                                   y_min=0.0,                y_max=Float64(Ny_phys - 1),
                                   Nx=Nx_phys, Ny=Ny_phys)

        blk_A = Block(:A, mesh_A; west=:wall, east=:interface,
                                   south=:wall, north=:wall)
        blk_B = Block(:B, mesh_B; west=:interface, east=:wall,
                                   south=:wall, north=:wall)
        mbm = MultiBlockMesh2D([blk_A, blk_B];
                                interfaces=[Interface(; from=(:A, :east), to=(:B, :west))])
        # Sanity should pass (non-overlap topology with 1·dx offset)
        @test all(iss -> iss.severity !== :error,
                   sanity_check_multiblock(mbm; verbose=false))
        states = [allocate_block_state_2d(b; n_ghost=1) for b in mbm.blocks]
        init_uniform_eq!(states[1], 1.0, u0, 0.0)
        init_uniform_eq!(states[2], 1.0, u0, 0.0)

        f_out_A = similar(states[1].f); f_out_B = similar(states[2].f)
        is_solid_A = zeros(Bool, ext_dims(states[1])...)
        is_solid_B = zeros(Bool, ext_dims(states[2])...)

        max_err = 0.0
        for step in 1:20
            fused_bgk_step!(fref_out, fref_in, ρref, uxref, uyref,
                            is_solid_ref, Nx_ref, Ny_ref, ω)

            exchange_ghost_2d!(mbm, states)
            fill_physical_wall_ghost_2d!(mbm, states)
            Nx_ext_A, Ny_ext_A = ext_dims(states[1])
            Nx_ext_B, Ny_ext_B = ext_dims(states[2])
            fused_bgk_step!(f_out_A, states[1].f, states[1].ρ, states[1].ux, states[1].uy,
                            is_solid_A, Nx_ext_A, Ny_ext_A, ω)
            fused_bgk_step!(f_out_B, states[2].f, states[2].ρ, states[2].ux, states[2].uy,
                            is_solid_B, Nx_ext_B, Ny_ext_B, ω)

            ng = states[1].n_ghost
            int_A = view(f_out_A, (ng+1):(ng+Nx_phys), (ng+1):(ng+Ny_phys), :)
            int_B = view(f_out_B, (ng+1):(ng+Nx_phys), (ng+1):(ng+Ny_phys), :)
            # Disjoint mapping: A phys 1..Nxp ↔ single 1..Nxp; B phys 1..Nxp ↔ single Nxp+1..2Nxp
            err_A = maximum(abs.(int_A .- view(fref_out, 1:Nx_phys, :, :)))
            err_B = maximum(abs.(int_B .- view(fref_out, (Nx_phys+1):Nx_ref, :, :)))
            max_err = max(max_err, err_A, err_B)

            fref_in, fref_out = fref_out, fref_in
            states[1].f, f_out_A = f_out_A, states[1].f
            states[2].f, f_out_B = f_out_B, states[2].f
        end
        @test max_err < 1e-12
    end

    # ---- 3. Long-run smoke ------------------------------------------
    @testset "1000-step run produces finite populations" begin
        mbm = build_mbm()
        u0 = 0.02
        states = [allocate_block_state_2d(b; n_ghost=1) for b in mbm.blocks]
        init_uniform_eq!(states[1], 1.0, u0, 0.0)
        init_uniform_eq!(states[2], 1.0, u0, 0.0)
        f_out_A = similar(states[1].f); f_out_B = similar(states[2].f)
        is_solid_A = zeros(Bool, ext_dims(states[1])...)
        is_solid_B = zeros(Bool, ext_dims(states[2])...)

        for step in 1:1000
            exchange_ghost_2d!(mbm, states)
            Nx_ext_A, Ny_ext_A = ext_dims(states[1])
            Nx_ext_B, Ny_ext_B = ext_dims(states[2])
            fused_bgk_step!(f_out_A, states[1].f, states[1].ρ, states[1].ux, states[1].uy,
                            is_solid_A, Nx_ext_A, Ny_ext_A, ω)
            fused_bgk_step!(f_out_B, states[2].f, states[2].ρ, states[2].ux, states[2].uy,
                            is_solid_B, Nx_ext_B, Ny_ext_B, ω)
            states[1].f, f_out_A = f_out_A, states[1].f
            states[2].f, f_out_B = f_out_B, states[2].f
        end
        @test all(isfinite, interior_f(states[1]))
        @test all(isfinite, interior_f(states[2]))
    end
end
