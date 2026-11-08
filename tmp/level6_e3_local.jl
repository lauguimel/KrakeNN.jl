# Level 6 — reproduce E3 setup LOCALLY (CPU F64) at small D_lu to predict
# what Aqua will output with the corner fix. E3 on Aqua prior to fix:
#   D=40 Cd=15.50, D=80 Cd=-5.28  (clearly wrong).
# With this fix, Cd should at least be sensible (1–5 range typical for
# Re=100 blocked cylinder).

using KrakeNN, KernelAbstractions

const T = Float64
const backend = KernelAbstractions.CPU()

const Lx = 1.0; const Ly = 0.5
const cx_p = 0.5; const cy_p = 0.245
const R_p = 0.025
const u_max = 0.04; const u_mean = u_max
const Re = 100.0
const R_bubble = 0.15

D_lu = 20   # still small for CPU, but stable at Re=100 (τ ≈ 0.52)
steps = 10_000   # short; won't reach shedding cycle but avoids NaN masking
avg_window = 3_000

dx = 2 * R_p / D_lu
Nx_total = round(Int, Lx / dx) + 1
Ny = round(Int, Ly / dx) + 1
x_C_west = cx_p - R_bubble
x_C_east = cx_p + R_bubble
Nx_W = round(Int, x_C_west / dx)
Nx_C = round(Int, (x_C_east - x_C_west) / dx) + 1
Nx_E = Nx_total - Nx_W - Nx_C
R_lu = R_p / dx
ν = u_mean * 2 * R_lu / Re

println("=== Level 6 : E3 local CPU, D_lu=$D_lu ===")
println("Nx_total=$Nx_total Ny=$Ny  Nx_W=$Nx_W Nx_C=$Nx_C Nx_E=$Nx_E")
println("R_lu=$R_lu  ν=$(round(ν, sigdigits=4))  Re=$Re")

function _build_mbm()
    x_W_min = 0.0;                  x_W_max = (Nx_W - 1) * dx
    x_C_min = Nx_W * dx;             x_C_max = x_C_min + (Nx_C - 1) * dx
    x_E_min = (Nx_W + Nx_C) * dx;    x_E_max = x_E_min + (Nx_E - 1) * dx
    y_min = 0.0; y_max = (Ny - 1) * dx
    mesh_W = cartesian_mesh(; x_min=x_W_min, x_max=x_W_max, y_min=y_min, y_max=y_max,
                              Nx=Nx_W, Ny=Ny, FT=T)
    mesh_C = cartesian_mesh(; x_min=x_C_min, x_max=x_C_max, y_min=y_min, y_max=y_max,
                              Nx=Nx_C, Ny=Ny, FT=T)
    mesh_E = cartesian_mesh(; x_min=x_E_min, x_max=x_E_max, y_min=y_min, y_max=y_max,
                              Nx=Nx_E, Ny=Ny, FT=T)
    blk_W = Block(:W, mesh_W; west=:inlet, east=:interface, south=:wall, north=:wall)
    blk_C = Block(:C, mesh_C; west=:interface, east=:interface, south=:wall, north=:wall)
    blk_E = Block(:E, mesh_E; west=:interface, east=:outlet, south=:wall, north=:wall)
    return MultiBlockMesh2D([blk_W, blk_C, blk_E];
                             interfaces=[Interface(; from=(:W, :east), to=(:C, :west)),
                                         Interface(; from=(:C, :east), to=(:E, :west))])
end

function _build_mbm_1block()
    mesh = cartesian_mesh(; x_min=0.0, x_max=(Nx_total - 1) * dx,
                            y_min=0.0, y_max=(Ny - 1) * dx,
                            Nx=Nx_total, Ny=Ny, FT=T)
    blk = Block(:A, mesh; west=:inlet, east=:outlet, south=:wall, north=:wall)
    return MultiBlockMesh2D([blk]; interfaces=Interface[])
end

function run_E3_local(mbm, label)
    n_blocks = length(mbm.blocks)
    ng = 1
    states = [allocate_block_state_2d(b; n_ghost=ng, backend=backend) for b in mbm.blocks]

    # Compute q_wall on the GLOBAL Nx_total×Ny grid first, then slice per
    # block by integer offsets — avoids FP drift in cx_local across blocks.
    cx_global = (cx_p - 0.0) / dx + 1
    cy_global = (cy_p - 0.0) / dx + 1
    q_g, solid_g = precompute_q_wall_cylinder(Nx_total, Ny, cx_global, cy_global, R_lu; FT=T)
    # Compute each block's starting i-offset (in global integer units).
    i_offsets = Int[]
    for blk in mbm.blocks
        off = round(Int, (blk.mesh.X[1, 1] - 0.0) / dx)  # global i - 1
        push!(i_offsets, off)
    end
    q_wall_ext = Vector{Any}(undef, n_blocks)
    is_solid_ext = Vector{Any}(undef, n_blocks)
    uw_x_ext = Vector{Any}(undef, n_blocks)
    uw_y_ext = Vector{Any}(undef, n_blocks)
    for (k, blk) in enumerate(mbm.blocks)
        Nxk = blk.mesh.Nξ; Nyk = blk.mesh.Nη
        i_lo = i_offsets[k] + 1; i_hi = i_lo + Nxk - 1
        q_int = q_g[i_lo:i_hi, 1:Nyk, :]
        solid_int = solid_g[i_lo:i_hi, 1:Nyk]
        q_wall_ext[k]   = extend_interior_field_2d(q_int, ng)
        is_solid_ext[k] = extend_interior_field_2d(solid_int, ng)
        uw_x_ext[k]     = extend_interior_field_2d(zeros(T, Nxk, Nyk, 9), ng)
        uw_y_ext[k]     = extend_interior_field_2d(zeros(T, Nxk, Nyk, 9), ng)
    end

    # Uniform inlet u=u_max
    u_prof_h = fill(T(u_max), Ny)
    # Init with uniform flow everywhere (zero inside cylinder)
    for (k, blk) in enumerate(mbm.blocks)
        Nxk = blk.mesh.Nξ; Nyk = blk.mesh.Nη; ng_ = states[k].n_ghost
        f_int = zeros(T, Nxk, Nyk, 9)
        solid_int = @view is_solid_ext[k][(ng_+1):(ng_+Nxk), (ng_+1):(ng_+Nyk)]
        for j in 1:Nyk, i in 1:Nxk, q in 1:9
            u = solid_int[i, j] ? zero(T) : T(u_max)
            f_int[i, j, q] = T(KrakeNN.equilibrium(D2Q9(), 1.0, u, 0.0, q))
        end
        int_view = interior_f(states[k])
        int_view .= f_int
    end

    f_out = [similar(st.f) for st in states]
    for k in 1:n_blocks; fill!(f_out[k], zero(T)); end

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
            int_out = view(f_out[k],    (ng_+1):(ng_+Nξp), (ng_+1):(ng_+Nηp), :)
            int_in  = view(states[k].f, (ng_+1):(ng_+Nξp), (ng_+1):(ng_+Nηp), :)
            apply_bc_rebuild_2d!(int_out, int_in, bcspecs[k], ν, Nξp, Nηp)
        end
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
    norm = u_mean^2 * (2 * R_lu)
    Cd = 2 * Fx / norm
    Cl = 2 * Fy / norm
    println("[$label] Cd=$(round(Cd, digits=4))  Cl=$(round(Cl, sigdigits=3))  Fx=$(round(Fx, sigdigits=6))")
    return (; Cd, Cl)
end

mbm_1 = _build_mbm_1block()
r1 = run_E3_local(mbm_1, "1-block")
mbm_3 = _build_mbm()
r3 = run_E3_local(mbm_3, "3-block (E3)")

println()
ΔCd = abs(r3.Cd - r1.Cd)
println("ΔCd = $(round(ΔCd, sigdigits=3))")
if ΔCd < 1e-6
    println("✅ E3 local 1 == 3 bit-exact")
else
    println("⚠️  E3 local 1 ≠ 3 by $(round(ΔCd, sigdigits=3))")
end
println("Previous (pre-fix) E3 Aqua: Cd=15.50 at D=40, Cd=-5.28 at D=80 — clearly wrong")
