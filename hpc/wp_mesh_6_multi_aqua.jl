# WP-MESH-6 extension — multi-block Cartesian cylinder (approach E, MVP).
#
# Same physical setup as (A)(B)(C)(D) runs: cylinder Re=100 in
# 1.0 × 0.5 domain, blockage 10%, uniform inlet u=0.04. The multi-block
# design demonstrates the v0.3 infrastructure end-to-end:
#
#   ┌──────────────┬──────────────┬──────────────┐
#   │   block W    │   block C    │   block E    │
#   │  (upstream   │  (cylinder   │  (downstream │
#   │   Cartesian, │   centred on │   Cartesian, │
#   │   BGK)       │   LI-BB)     │   BGK)       │
#   └──────────────┴──────────────┴──────────────┘
#     x ∈ [0, xW]   [xW+dx, xC]    [xC+dx, Lx]
#
# All three blocks use identical uniform dx (non-overlap convention).
# Block C carries the cylinder; its LI-BB precompute runs in that
# block's local coordinates. Blocks W and E use the same
# `fused_trt_libb_v2_step!` kernel with `is_solid=0, q_wall=0` →
# reduces to standard TRT collide-stream.
#
# Pipeline per step (B.2.1 + A.5c):
#   1. exchange_ghost_2d! (non-overlap: W.east↔C.west, C.east↔E.west)
#   2. fill_physical_wall_ghost_2d! (south/north walls in all blocks,
#      west of W and east of E where inlet/outlet live — but those
#      wall_ghost-fills are harmless because apply_bc_rebuild overrides)
#   3. Per-block step on extended arrays (`Nξ+2Ng, Nη+2Ng, 9`)
#   4. Per-block BC apply on the INTERIOR view:
#         W: west=ZouHeVelocity(u_in) + walls + interface (HalfwayBB no-op)
#         C: all walls + interfaces (HalfwayBB no-op) → BC call is a no-op
#         E: east=ZouHePressure(1.0) + walls + interface
#   5. Swap states[k].f ↔ f_out[k]
#
# Drag sampling: only block C touches the cylinder, so `compute_drag_libb_mei_2d`
# on block C's INTERIOR view yields the full Fx, Fy.

using KrakeNN, CUDA, KernelAbstractions
using FFTW: rfft, rfftfreq

const Lx, Ly         = 1.0, 0.5
const cx_p, cy_p     = 0.5, 0.245
const R_p            = 0.025
const Re_target      = 100.0
const u_max          = 0.04
const u_mean         = u_max
const Cd_ref         = 1.4
const Cl_ref         = 0.33
const St_ref         = 0.165
# Interface splits (symmetric around cylinder). Block C spans
# [x_W_end - R_bubble, x_W_end + R_bubble] with R_bubble ≈ 3R so the
# wake envelope stays inside C (the wall-ghost BB on W/E is harmless
# only if the flow at the W↔C and C↔E interfaces is far from the body).
const R_bubble       = 0.15  # half-width of central block around cylinder
T = Float64

backend = CUDABackend()

function _setup(D_lu)
    dx_ref = 2 * R_p / D_lu
    Nx_total = round(Int, Lx / dx_ref) + 1
    Ny = round(Int, Ly / dx_ref) + 1
    # Split into three blocks along x:
    #  Block W: x ∈ [0, x_W_end]           physical cells 1..Nx_W (last cell at x=x_W_end)
    #  Block C: x ∈ [x_W_end+dx, x_E_start-dx]
    #  Block E: x ∈ [x_E_start, Lx]        physical cells 1..Nx_E
    # Non-overlap: consecutive blocks are 1·dx apart.
    x_C_west = cx_p - R_bubble            # physical x at block C's west interior boundary
    x_C_east = cx_p + R_bubble
    Nx_W = round(Int, x_C_west / dx_ref)                   # last cell at x=(Nx_W-1)*dx
    Nx_C = round(Int, (x_C_east - x_C_west) / dx_ref) + 1  # cells covering [x_C_west, x_C_east]
    Nx_E = Nx_total - Nx_W - Nx_C                           # remainder
    cx_lu = cx_p / dx_ref; cy_lu = cy_p / dx_ref; R_lu = R_p / dx_ref
    ν = u_mean * D_lu / Re_target
    return (; Nx_total, Nx_W, Nx_C, Nx_E, Ny, dx_ref, cx_lu, cy_lu, R_lu, ν)
end

function _inlet_uniform(Ny)
    u_prof_h = fill(T(u_max), Ny)
    return CuArray(u_prof_h), u_prof_h
end

function _init_f_interior(Nx, Ny, u_prof_h, is_solid_h)
    f_h = zeros(T, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        u = is_solid_h[i, j] ? 0.0 : Float64(u_prof_h[j])
        f_h[i, j, q] = T(KrakeNN.equilibrium(D2Q9(), 1.0, u, 0.0, q))
    end
    return f_h
end

function _strouhal(cl, dt_lu, D_lu)
    cl_z = cl .- (sum(cl) / length(cl))
    F = abs.(rfft(cl_z))
    fs = rfftfreq(length(cl_z), 1 / dt_lu)
    idx = argmax(F[2:end]) + 1
    return fs[idx] * D_lu / Float64(u_mean)
end

function _stats(cd, cl)
    cd_mean = sum(cd) / length(cd)
    cl_mean = sum(cl) / length(cl)
    cl_rms  = sqrt(sum((cl .- cl_mean).^2) / length(cl))
    return cd_mean, cl_rms
end

function _format(label, cells, cd, clr, St, mlups, elapsed)
    err_cd  = 100*abs(cd - Cd_ref)/Cd_ref
    err_clr = 100*abs(clr - Cl_ref)/Cl_ref
    err_st  = 100*abs(St - St_ref)/St_ref
    println("[$label cells=$cells] Cd=$(round(cd,digits=3)) (err $(round(err_cd,digits=2))%)  ",
            "Cl_RMS=$(round(clr,digits=3)) (err $(round(err_clr,digits=2))%)  ",
            "St=$(round(St,digits=3)) (err $(round(err_st,digits=2))%)  | ",
            "$(round(mlups,digits=0)) MLUPS  $(round(elapsed,digits=1))s")
end

# Build the 3-block MBM. All blocks share dx = s.dx_ref; non-overlap
# spacing between consecutive blocks is 1·dx_ref.
function _build_mbm(s)
    dx = s.dx_ref
    x_W_min = 0.0;                     x_W_max = (s.Nx_W - 1) * dx
    x_C_min = s.Nx_W * dx;              x_C_max = x_C_min + (s.Nx_C - 1) * dx
    x_E_min = (s.Nx_W + s.Nx_C) * dx;   x_E_max = x_E_min + (s.Nx_E - 1) * dx
    y_min = 0.0; y_max = (s.Ny - 1) * dx
    mesh_W = cartesian_mesh(; x_min=x_W_min, x_max=x_W_max, y_min=y_min, y_max=y_max,
                              Nx=s.Nx_W, Ny=s.Ny)
    mesh_C = cartesian_mesh(; x_min=x_C_min, x_max=x_C_max, y_min=y_min, y_max=y_max,
                              Nx=s.Nx_C, Ny=s.Ny)
    mesh_E = cartesian_mesh(; x_min=x_E_min, x_max=x_E_max, y_min=y_min, y_max=y_max,
                              Nx=s.Nx_E, Ny=s.Ny)
    blk_W = Block(:W, mesh_W; west=:inlet,     east=:interface,
                              south=:wall,     north=:wall)
    blk_C = Block(:C, mesh_C; west=:interface, east=:interface,
                              south=:wall,     north=:wall)
    blk_E = Block(:E, mesh_E; west=:interface, east=:outlet,
                              south=:wall,     north=:wall)
    mbm = MultiBlockMesh2D([blk_W, blk_C, blk_E];
                            interfaces=[Interface(; from=(:W, :east), to=(:C, :west)),
                                        Interface(; from=(:C, :east), to=(:E, :west))])
    return mbm
end

run_E3(D_lu, steps, sw, se) = begin
    s = _setup(D_lu)
    ng = 1
    mbm = _build_mbm(s)
    issues = sanity_check_multiblock(mbm; verbose=false)
    any(iss -> iss.severity === :error, issues) &&
        error("approach E (3-block) sanity has :error issues: $issues")

    # --- Allocate per-block state (extended Nξ+2Ng, Nη+2Ng, 9) ---
    states = [allocate_block_state_2d(b; n_ghost=ng, backend=backend) for b in mbm.blocks]

    # --- Cylinder geometry: compute on GLOBAL grid, then slice per block ---
    # Computing cx_C_lu via (cx_p - x0_C)/dx + 1 introduces ~1e-14 FP
    # drift because x0_C = Nx_W * dx is not exactly representable. Even
    # this tiny offset QUALITATIVELY changes precompute_q_wall_cylinder's
    # output (cells at the cylinder circumference flip solid↔fluid), so
    # we compute the cylinder mask ONCE on the global Nx_total × Ny grid
    # and slice into each block by integer offsets.
    Nx_C = mbm.blocks[2].mesh.Nξ
    cx_g_lu = cx_p / s.dx_ref + 1
    cy_g_lu = cy_p / s.dx_ref + 1
    q_g, is_solid_g = precompute_q_wall_cylinder(s.Nx_total, s.Ny,
                                                   cx_g_lu, cy_g_lu, s.R_lu)
    # Block C starts at global i = Nx_W + 1
    i_C_lo = s.Nx_W + 1
    i_C_hi = i_C_lo + Nx_C - 1
    q_wall_C_int   = q_g[i_C_lo:i_C_hi, 1:s.Ny, :]
    is_solid_C_int = is_solid_g[i_C_lo:i_C_hi, 1:s.Ny]
    # uw zeros (stationary cylinder)
    uw_x_C_int = zeros(T, Nx_C, s.Ny, 9)
    uw_y_C_int = zeros(T, Nx_C, s.Ny, 9)

    # Lift interior arrays to extended (pad with 0 on ghost layer).
    q_wall_ext = [
        CuArray(extend_interior_field_2d(zeros(T, mbm.blocks[1].mesh.Nξ, s.Ny, 9), ng)),
        CuArray(extend_interior_field_2d(q_wall_C_int, ng)),
        CuArray(extend_interior_field_2d(zeros(T, mbm.blocks[3].mesh.Nξ, s.Ny, 9), ng)),
    ]
    uw_x_ext = [
        CuArray(extend_interior_field_2d(zeros(T, mbm.blocks[1].mesh.Nξ, s.Ny, 9), ng)),
        CuArray(extend_interior_field_2d(uw_x_C_int, ng)),
        CuArray(extend_interior_field_2d(zeros(T, mbm.blocks[3].mesh.Nξ, s.Ny, 9), ng)),
    ]
    uw_y_ext = [
        CuArray(extend_interior_field_2d(zeros(T, mbm.blocks[1].mesh.Nξ, s.Ny, 9), ng)),
        CuArray(extend_interior_field_2d(uw_y_C_int, ng)),
        CuArray(extend_interior_field_2d(zeros(T, mbm.blocks[3].mesh.Nξ, s.Ny, 9), ng)),
    ]
    is_solid_ext = [
        CuArray(extend_interior_field_2d(zeros(Bool, mbm.blocks[1].mesh.Nξ, s.Ny), ng)),
        CuArray(extend_interior_field_2d(is_solid_C_int, ng)),
        CuArray(extend_interior_field_2d(zeros(Bool, mbm.blocks[3].mesh.Nξ, s.Ny), ng)),
    ]

    # --- Initialise populations to uniform-flow equilibrium in each block's interior ---
    u_prof_full, u_prof_h_full = _inlet_uniform(s.Ny)
    for k in 1:3
        Nξk = mbm.blocks[k].mesh.Nξ
        solid_int = k == 2 ? is_solid_C_int : zeros(Bool, Nξk, s.Ny)
        f_int = _init_f_interior(Nξk, s.Ny, u_prof_h_full, solid_int)
        int_view = interior_f(states[k])
        copyto!(int_view, CuArray(f_int))
    end

    # --- Per-block f_out buffers + per-block bcspec ---
    f_out = [similar(st.f) for st in states]
    for k in 1:3; fill!(f_out[k], zero(T)); end
    u_prof_cu, _ = _inlet_uniform(s.Ny)
    bcspec_W = BCSpec2D(; west=ZouHeVelocity(u_prof_cu), east=HalfwayBB(),
                          south=HalfwayBB(),              north=HalfwayBB())
    bcspec_C = BCSpec2D(; west=HalfwayBB(),              east=HalfwayBB(),
                          south=HalfwayBB(),              north=HalfwayBB())
    bcspec_E = BCSpec2D(; west=HalfwayBB(),              east=ZouHePressure(one(T)),
                          south=HalfwayBB(),              north=HalfwayBB())
    bcspecs = (bcspec_W, bcspec_C, bcspec_E)

    cd_h = Float64[]; cl_h = Float64[]
    norm = u_mean^2 * (s.R_lu * 2)

    t0 = time()
    for step in 1:steps
        # 1. Interface ghost exchange (non-overlap, W.east↔C.west and C.east↔E.west).
        exchange_ghost_2d!(mbm, states)
        # 2. Physical-wall ghost fill (halfway-BB reflection on wall rows).
        fill_physical_wall_ghost_2d!(mbm, states)
        # 3. Per-block step on extended arrays.
        for k in 1:3
            Nx_ext, Ny_ext = ext_dims(states[k])
            fused_trt_libb_v2_step!(f_out[k], states[k].f, states[k].ρ, states[k].ux, states[k].uy,
                                      is_solid_ext[k], q_wall_ext[k], uw_x_ext[k], uw_y_ext[k],
                                      Nx_ext, Ny_ext, s.ν)
        end
        # 4. Apply physical BCs on INTERIOR views. HalfwayBB on interface
        #    sides is a no-op, so this only overwrites inlet/outlet physical columns.
        for k in 1:3
            int_out  = interior_f(states[k])   # sits in states[k].f wrapper
            # Actually we need a view of f_out[k] interior — create from Nx_phys × Nη_phys.
            Nξ_phys = states[k].Nξ_phys; Nη_phys = states[k].Nη_phys
            ng_k = states[k].n_ghost
            int_f_out = view(f_out[k], (ng_k+1):(ng_k+Nξ_phys), (ng_k+1):(ng_k+Nη_phys), :)
            int_f_in  = view(states[k].f, (ng_k+1):(ng_k+Nξ_phys), (ng_k+1):(ng_k+Nη_phys), :)
            apply_bc_rebuild_2d!(int_f_out, int_f_in, bcspecs[k], s.ν, Nξ_phys, Nη_phys)
        end
        # 5. Drag sampling on block C (the only one touching the cylinder).
        if step > steps - sw && step % se == 0
            Nξ_phys_C = states[2].Nξ_phys; Nη_phys_C = states[2].Nη_phys
            ng_C = states[2].n_ghost
            int_f_out_C = view(f_out[2], (ng_C+1):(ng_C+Nξ_phys_C), (ng_C+1):(ng_C+Nη_phys_C), :)
            # Drag on interior q_wall / uw arrays (non-extended).
            q_int_C = view(q_wall_ext[2], (ng_C+1):(ng_C+Nξ_phys_C), (ng_C+1):(ng_C+Nη_phys_C), :)
            uwx_int_C = view(uw_x_ext[2], (ng_C+1):(ng_C+Nξ_phys_C), (ng_C+1):(ng_C+Nη_phys_C), :)
            uwy_int_C = view(uw_y_ext[2], (ng_C+1):(ng_C+Nξ_phys_C), (ng_C+1):(ng_C+Nη_phys_C), :)
            Fx, Fy = compute_drag_libb_mei_2d(int_f_out_C, q_int_C, uwx_int_C, uwy_int_C,
                                                Nξ_phys_C, Nη_phys_C)
            push!(cd_h, 2*Fx/norm); push!(cl_h, 2*Fy/norm)
        end
        # Swap
        for k in 1:3
            states[k].f, f_out[k] = f_out[k], states[k].f
        end
    end
    CUDA.synchronize()
    elapsed = time() - t0
    cd, clr = _stats(cd_h, cl_h)
    St = _strouhal(cl_h, se, 2 * s.R_lu)
    cells = (s.Nx_W + s.Nx_C + s.Nx_E) * s.Ny
    mlups = cells * steps / elapsed / 1e6
    _format("E3 D=$D_lu", cells, cd, clr, St, mlups, elapsed)
    println("  (E3): blocks W=$(s.Nx_W)×$(s.Ny) C=$(s.Nx_C)×$(s.Ny) E=$(s.Nx_E)×$(s.Ny) " *
            "= $cells total")
    return (; cells, cd, cl_rms=clr, St, mlups, elapsed)
end

println("=== WP-MESH-6 (E3) — 3-block Cartesian multi-block cylinder (CUDA H100, FP64) ===")
println("Reference (Williamson 1996, Park 1998): Cd≈$Cd_ref  Cl_RMS≈$Cl_ref  St≈$St_ref")
println("Domain $Lx × $Ly  cylinder ($cx_p, $cy_p) R=$R_p  blockage = $(round(2*R_p/Ly*100, digits=1))%")
println("Block C around cylinder spans 2×R_bubble=$(2*R_bubble) in x\n")

for D_lu in (20, 40, 80)
    steps = D_lu == 20 ? 80_000 : D_lu == 40 ? 160_000 : 320_000
    sw = steps ÷ 2
    se = D_lu == 80 ? 20 : 10
    println("--- D_lu=$D_lu  steps=$steps  sample_every=$se  window=$sw ---")
    try
        run_E3(D_lu, steps, sw, se)
    catch e
        println("  E3 FAILED: ", sprint(showerror, e)[1:min(end,200)])
    end
end

println("\n=== Done ===")
