# Level 6 1-step bit-exact check: do 1-block and 3-block give SAME f_out
# after 1 step at E3 setup (uniform inlet, Re=100, D=20)?

using KrakeNN, KernelAbstractions

const T = Float64
const backend = KernelAbstractions.CPU()
const Lx = 1.0; const Ly = 0.5
const cx_p = 0.5; const cy_p = 0.245
const R_p = 0.025
const u_max = 0.04; const Re = 100.0
const R_bubble = 0.15
const D_lu = 20
const dx = 2 * R_p / D_lu

const Nx_total = round(Int, Lx / dx) + 1
const Ny = round(Int, Ly / dx) + 1
const R_lu = R_p / dx
const ν = u_max * 2 * R_lu / Re

function _build_mbm_1block()
    mesh = cartesian_mesh(; x_min=0.0, x_max=(Nx_total - 1) * dx,
                            y_min=0.0, y_max=(Ny - 1) * dx,
                            Nx=Nx_total, Ny=Ny, FT=T)
    blk = Block(:A, mesh; west=:inlet, east=:outlet, south=:wall, north=:wall)
    return MultiBlockMesh2D([blk]; interfaces=Interface[])
end

function _build_mbm_3blocks()
    x_C_west = cx_p - R_bubble; x_C_east = cx_p + R_bubble
    Nx_W = round(Int, x_C_west / dx)
    Nx_C = round(Int, (x_C_east - x_C_west) / dx) + 1
    Nx_E = Nx_total - Nx_W - Nx_C
    x_W_max = (Nx_W - 1) * dx
    x_C_min = Nx_W * dx;           x_C_max = x_C_min + (Nx_C - 1) * dx
    x_E_min = (Nx_W + Nx_C) * dx;  x_E_max = x_E_min + (Nx_E - 1) * dx
    mesh_W = cartesian_mesh(; x_min=0.0, x_max=x_W_max, y_min=0.0, y_max=(Ny-1)*dx,
                              Nx=Nx_W, Ny=Ny, FT=T)
    mesh_C = cartesian_mesh(; x_min=x_C_min, x_max=x_C_max, y_min=0.0, y_max=(Ny-1)*dx,
                              Nx=Nx_C, Ny=Ny, FT=T)
    mesh_E = cartesian_mesh(; x_min=x_E_min, x_max=x_E_max, y_min=0.0, y_max=(Ny-1)*dx,
                              Nx=Nx_E, Ny=Ny, FT=T)
    blk_W = Block(:W, mesh_W; west=:inlet, east=:interface, south=:wall, north=:wall)
    blk_C = Block(:C, mesh_C; west=:interface, east=:interface, south=:wall, north=:wall)
    blk_E = Block(:E, mesh_E; west=:interface, east=:outlet, south=:wall, north=:wall)
    return MultiBlockMesh2D([blk_W, blk_C, blk_E];
                             interfaces=[Interface(; from=(:W, :east), to=(:C, :west)),
                                         Interface(; from=(:C, :east), to=(:E, :west))])
end

function setup_and_step(mbm, label)
    n_blocks = length(mbm.blocks); ng = 1
    states = [allocate_block_state_2d(b; n_ghost=ng, backend=backend) for b in mbm.blocks]
    q_wall_ext = Vector{Any}(undef, n_blocks)
    is_solid_ext = Vector{Any}(undef, n_blocks)
    uw_x_ext = Vector{Any}(undef, n_blocks); uw_y_ext = Vector{Any}(undef, n_blocks)
    for (k, blk) in enumerate(mbm.blocks)
        Nxk = blk.mesh.Nξ; Nyk = blk.mesh.Nη
        x0 = blk.mesh.X[1, 1]; y0 = blk.mesh.Y[1, 1]
        cx_local = (cx_p - x0) / dx + 1
        cy_local = (cy_p - y0) / dx + 1
        q_int, solid_int = precompute_q_wall_cylinder(Nxk, Nyk, cx_local, cy_local, R_lu; FT=T)
        q_wall_ext[k]   = extend_interior_field_2d(q_int, ng)
        is_solid_ext[k] = extend_interior_field_2d(solid_int, ng)
        uw_x_ext[k]     = extend_interior_field_2d(zeros(T, Nxk, Nyk, 9), ng)
        uw_y_ext[k]     = extend_interior_field_2d(zeros(T, Nxk, Nyk, 9), ng)
    end
    u_prof_h = fill(T(u_max), Ny)
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
        bcspecs[k] = BCSpec2D(; west=west_bc, east=east_bc, south=HalfwayBB(), north=HalfwayBB())
    end
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
        int_out = view(f_out[k], (ng_+1):(ng_+Nξp), (ng_+1):(ng_+Nηp), :)
        int_in  = view(states[k].f, (ng_+1):(ng_+Nξp), (ng_+1):(ng_+Nηp), :)
        apply_bc_rebuild_2d!(int_out, int_in, bcspecs[k], ν, Nξp, Nηp)
    end
    # Compute drag on all blocks and sum
    Fx_sum = 0.0; Fy_sum = 0.0
    for k in 1:n_blocks
        Nξp = states[k].Nξ_phys; Nηp = states[k].Nη_phys; ng_ = states[k].n_ghost
        int_f_out = view(f_out[k], (ng_+1):(ng_+Nξp), (ng_+1):(ng_+Nηp), :)
        int_q     = view(q_wall_ext[k], (ng_+1):(ng_+Nξp), (ng_+1):(ng_+Nηp), :)
        int_uwx   = view(uw_x_ext[k],   (ng_+1):(ng_+Nξp), (ng_+1):(ng_+Nηp), :)
        int_uwy   = view(uw_y_ext[k],   (ng_+1):(ng_+Nξp), (ng_+1):(ng_+Nηp), :)
        drag = compute_drag_libb_mei_2d(int_f_out, int_q, int_uwx, int_uwy, Nξp, Nηp)
        Fx_sum += drag.Fx; Fy_sum += drag.Fy
        if Nξp > 50
            println("  [$label] block k=$k  Nξ=$Nξp  Fx=$(round(drag.Fx, sigdigits=4))  q_wall non-zero: $(count(x -> x > 0, int_q))")
        end
    end
    return f_out, Fx_sum, Fy_sum
end

println("=== Level 6 1-step E3 setup ===")
mbm_1 = _build_mbm_1block()
f_1, Fx_1, Fy_1 = setup_and_step(mbm_1, "1-block")
mbm_3 = _build_mbm_3blocks()
f_3, Fx_3, Fy_3 = setup_and_step(mbm_3, "3-block")
println("\n1-block drag: Fx=$(round(Fx_1, sigdigits=6))  Fy=$(round(Fy_1, sigdigits=6))")
println("3-block drag: Fx=$(round(Fx_3, sigdigits=6))  Fy=$(round(Fy_3, sigdigits=6))")
println("ΔFx = $(round(abs(Fx_1 - Fx_3), sigdigits=3))")

# Check if f_out matches in the cylinder region
Nx_W_3 = mbm_3.blocks[1].mesh.Nξ
Nx_C_3 = mbm_3.blocks[2].mesh.Nξ
# 1-block cell i in [Nx_W_3 + 1, Nx_W_3 + Nx_C_3] corresponds to 3-block C i in [1, Nx_C_3]
function _find_max_diff()
    md = 0.0; ml = (0,0,0,0)
    for j in 1:Ny, i in 1:Nx_C_3, q in 1:9
        i_global = Nx_W_3 + i
        a = f_1[1][i_global + 1, j + 1, q]
        b = f_3[2][i + 1, j + 1, q]
        d = abs(a - b)
        if d > md; md = d; ml = (i, j, q, i_global); end
    end
    return md, ml
end
max_diff, max_loc = _find_max_diff()
println("Max |f_1 - f_3_C| in cylinder block region: $(round(max_diff, sigdigits=3)) at (i_C=$(max_loc[1]), j=$(max_loc[2]), q=$(max_loc[3]))  [i_global=$(max_loc[4])]")
