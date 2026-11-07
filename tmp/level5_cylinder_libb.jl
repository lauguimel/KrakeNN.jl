# Level 5 — cylinder + LI-BB + drag, 1-block vs 3-block (W|C|E) comparison.
# Same total domain, same cylinder entirely inside the central block, same
# BCSpec pipeline, same kernels. Must give bit-exact drag at each step.
#
# If they agree → the corner BC fix resolves the multi-block E3 issue.

using KrakeNN, KernelAbstractions

const T = Float64
const backend = KernelAbstractions.CPU()

# Small setup for fast iteration (CPU F64)
const Nx_total = 200
const Ny = 80
const dx = 1.0
const cx_g = 50.0            # global cylinder center x (far from outlet)
const cy_g = 40.0
const R    = 10.0
const u_in = 0.04
const ν    = 0.04
const steps = 10_000
const avg_window = 2_000

function _build_mbm_1block()
    mesh = cartesian_mesh(; x_min=0.0, x_max=(Nx_total - 1) * dx,
                            y_min=0.0, y_max=(Ny - 1) * dx,
                            Nx=Nx_total, Ny=Ny, FT=T)
    blk = Block(:A, mesh; west=:inlet, east=:outlet,
                           south=:wall, north=:wall)
    return MultiBlockMesh2D([blk]; interfaces=Interface[])
end

function _build_mbm_3blocks(Nx_W, Nx_C, Nx_E)
    @assert Nx_W + Nx_C + Nx_E == Nx_total
    x_W_min = 0.0;                   x_W_max = (Nx_W - 1) * dx
    x_C_min = Nx_W * dx;             x_C_max = x_C_min + (Nx_C - 1) * dx
    x_E_min = (Nx_W + Nx_C) * dx;    x_E_max = x_E_min + (Nx_E - 1) * dx
    mesh_W = cartesian_mesh(; x_min=x_W_min, x_max=x_W_max, y_min=0.0, y_max=(Ny-1)*dx,
                              Nx=Nx_W, Ny=Ny, FT=T)
    mesh_C = cartesian_mesh(; x_min=x_C_min, x_max=x_C_max, y_min=0.0, y_max=(Ny-1)*dx,
                              Nx=Nx_C, Ny=Ny, FT=T)
    mesh_E = cartesian_mesh(; x_min=x_E_min, x_max=x_E_max, y_min=0.0, y_max=(Ny-1)*dx,
                              Nx=Nx_E, Ny=Ny, FT=T)
    blk_W = Block(:W, mesh_W; west=:inlet, east=:interface,
                              south=:wall, north=:wall)
    blk_C = Block(:C, mesh_C; west=:interface, east=:interface,
                              south=:wall, north=:wall)
    blk_E = Block(:E, mesh_E; west=:interface, east=:outlet,
                              south=:wall, north=:wall)
    return MultiBlockMesh2D([blk_W, blk_C, blk_E];
                             interfaces=[Interface(; from=(:W, :east), to=(:C, :west)),
                                         Interface(; from=(:C, :east), to=(:E, :west))])
end

function _setup_block(block, cx_g, cy_g, R)
    # Convert cx_g to this block's local coordinates; cell is solid if inside cylinder.
    x0 = block.mesh.X[1, 1]; y0 = block.mesh.Y[1, 1]
    cx_local = (cx_g - x0) / dx + 1  # 1-based index where cylinder center sits
    cy_local = (cy_g - y0) / dx + 1
    Nxk = block.mesh.Nξ; Nyk = block.mesh.Nη
    q_wall_int, is_solid_int = precompute_q_wall_cylinder(Nxk, Nyk,
                                                            cx_local, cy_local, R; FT=T)
    return q_wall_int, is_solid_int
end

function run_level(mbm, label)
    n_blocks = length(mbm.blocks)
    ng = 1
    states = [allocate_block_state_2d(b; n_ghost=ng, backend=backend) for b in mbm.blocks]

    # Parabolic inlet velocity, j=1..Ny
    u_prof_h = T.([4 * u_in * (j - 1) * (Ny - j) / (Ny - 1)^2 for j in 1:Ny])

    # Per-block precompute
    q_wall_ext = Vector{Any}(undef, n_blocks)
    uw_x_ext = Vector{Any}(undef, n_blocks)
    uw_y_ext = Vector{Any}(undef, n_blocks)
    is_solid_ext = Vector{Any}(undef, n_blocks)
    for (k, blk) in enumerate(mbm.blocks)
        q_int, solid_int = _setup_block(blk, cx_g, cy_g, R)
        Nxk = blk.mesh.Nξ; Nyk = blk.mesh.Nη
        q_wall_ext[k]   = extend_interior_field_2d(q_int, ng)
        is_solid_ext[k] = extend_interior_field_2d(solid_int, ng)
        uw_x_ext[k]     = extend_interior_field_2d(zeros(T, Nxk, Nyk, 9), ng)
        uw_y_ext[k]     = extend_interior_field_2d(zeros(T, Nxk, Nyk, 9), ng)
    end

    # Initial condition: equilibrium of parabolic inlet (except inside cylinder)
    for (k, blk) in enumerate(mbm.blocks)
        Nxk = blk.mesh.Nξ; Nyk = blk.mesh.Nη
        f_int = zeros(T, Nxk, Nyk, 9)
        ng_ = states[k].n_ghost
        solid_int = @view is_solid_ext[k][(ng_+1):(ng_+Nxk), (ng_+1):(ng_+Nyk)]
        for j in 1:Nyk, i in 1:Nxk, q in 1:9
            u = solid_int[i, j] ? zero(T) : u_prof_h[j]
            f_int[i, j, q] = T(KrakeNN.equilibrium(D2Q9(), 1.0, u, 0.0, q))
        end
        int_view = interior_f(states[k])
        int_view .= f_int
    end

    f_out = [similar(st.f) for st in states]
    for k in 1:n_blocks; fill!(f_out[k], zero(T)); end

    # BCSpecs: inlet for west of first block, outlet for east of last, HalfwayBB elsewhere
    bcspecs = Vector{Any}(undef, n_blocks)
    for (k, blk) in enumerate(mbm.blocks)
        tags = blk.boundary_tags
        west_bc = tags.west === :inlet ? ZouHeVelocity(u_prof_h) : HalfwayBB()
        east_bc = tags.east === :outlet ? ZouHePressure(one(T)) : HalfwayBB()
        bcspecs[k] = BCSpec2D(; west=west_bc, east=east_bc,
                               south=HalfwayBB(), north=HalfwayBB())
    end

    Fx_sum = 0.0; Fy_sum = 0.0; n_avg = 0

    for step in 1:steps
        exchange_ghost_2d!(mbm, states)
        fill_physical_wall_ghost_2d!(mbm, states)
        for k in 1:n_blocks
            Nx_ext, Ny_ext = ext_dims(states[k])
            fused_trt_libb_v2_step!(f_out[k], states[k].f,
                                     states[k].ρ, states[k].ux, states[k].uy,
                                     is_solid_ext[k], q_wall_ext[k],
                                     uw_x_ext[k], uw_y_ext[k],
                                     Nx_ext, Ny_ext, T(ν))
        end
        for k in 1:n_blocks
            Nξp = states[k].Nξ_phys; Nηp = states[k].Nη_phys; ng_ = states[k].n_ghost
            int_out = view(f_out[k],     (ng_+1):(ng_+Nξp), (ng_+1):(ng_+Nηp), :)
            int_in  = view(states[k].f,  (ng_+1):(ng_+Nξp), (ng_+1):(ng_+Nηp), :)
            apply_bc_rebuild_2d!(int_out, int_in, bcspecs[k], ν, Nξp, Nηp)
        end
        # Drag: aggregate on all blocks (only cylinder block C contributes nonzero)
        if step > steps - avg_window
            for k in 1:n_blocks
                Nξp = states[k].Nξ_phys; Nηp = states[k].Nη_phys; ng_ = states[k].n_ghost
                int_f_out = view(f_out[k],    (ng_+1):(ng_+Nξp), (ng_+1):(ng_+Nηp), :)
                int_q     = view(q_wall_ext[k], (ng_+1):(ng_+Nξp), (ng_+1):(ng_+Nηp), :)
                int_uwx   = view(uw_x_ext[k],   (ng_+1):(ng_+Nξp), (ng_+1):(ng_+Nηp), :)
                int_uwy   = view(uw_y_ext[k],   (ng_+1):(ng_+Nξp), (ng_+1):(ng_+Nηp), :)
                drag = compute_drag_libb_mei_2d(int_f_out, int_q, int_uwx, int_uwy, Nξp, Nηp)
                Fx_sum += drag.Fx; Fy_sum += drag.Fy
            end
            n_avg += 1
        end
        for k in 1:n_blocks
            states[k].f, f_out[k] = f_out[k], states[k].f
        end
    end

    Fx = Fx_sum / n_avg; Fy = Fy_sum / n_avg
    u_ref = (2 / 3) * u_in   # parabolic mean
    D = 2 * R
    Cd = 2.0 * Fx / (u_ref^2 * D)
    Cl = 2.0 * Fy / (u_ref^2 * D)
    println("[$label] Cd=$(round(Cd, digits=4))  Cl=$(round(Cl, sigdigits=3))  Fx=$(round(Fx, sigdigits=6))")
    return (; Cd, Cl, Fx, Fy)
end

println("=== Level 5 : cylinder + LI-BB, 1 block vs 3 blocks ===")
println("Reference (Schäfer-Turek 2D-1, Re=20): Cd ≈ 5.58")
println("Setup: Nx=$Nx_total Ny=$Ny  R=$R  cx=$cx_g  u_in=$u_in  ν=$ν  steps=$steps")
println()

mbm_1 = _build_mbm_1block()
r1 = run_level(mbm_1, "1-block")

# 3-block split: W is short, C covers cylinder, E is short
# Cylinder x-range: cx_g ± R = [40, 60] must be entirely inside C
Nx_W = 30; Nx_C = 80; Nx_E = Nx_total - Nx_W - Nx_C
mbm_3 = _build_mbm_3blocks(Nx_W, Nx_C, Nx_E)
println("3-block W|C|E = $Nx_W|$Nx_C|$Nx_E  (interfaces at x=$((Nx_W-1)*dx) and x=$(((Nx_W+Nx_C)-1)*dx))")
r3 = run_level(mbm_3, "3-block")

println()
println("=== Comparison ===")
ΔCd = abs(r3.Cd - r1.Cd)
println("ΔCd = $(round(ΔCd, sigdigits=3))  (rel $(round(100*ΔCd/abs(r1.Cd), sigdigits=3))%)")
if ΔCd < 1e-6
    println("✅ 1-block == 3-block bit-exact-ish (corner fix OK for multi-block cylinder)")
elseif ΔCd < 1e-3
    println("⚠️  close but not bit-exact — check LI-BB aggregation")
else
    println("❌ still diverges — deeper bug remains")
end
