# Level 4 — SLBM (with extended mesh via build_block_slbm_geometry_extended)
# on 1-block vs 2-block Poiseuille. Uniform Cartesian mesh, so SLBM's
# metric is identity — the SLBM step must produce the SAME result as
# BGK/TRT on the same Cartesian mesh. If multi-block SLBM == single-block
# SLBM bit-exact, then the per-block extended geometry pipeline is OK.

using KrakeNN, KernelAbstractions

const T = Float64
const Lx = 1.0
const Ly = 0.25
const Nx = 65
const Ny = 17
const dx = Lx / (Nx - 1)
const u_max = 0.04
const ν = 0.01
const steps = 4000
const backend = KernelAbstractions.CPU()

const H_eff = Ly + dx
u_analytical(y) = u_max * 4 * (y + dx/2) * (H_eff - (y + dx/2)) / H_eff^2
parabolic_inlet(Ny) = T.([u_analytical((j - 1) * dx) for j in 1:Ny])

function _make_block(id, xmin, xmax, Nx_k; west_tag, east_tag)
    mesh = cartesian_mesh(; x_min=xmin, x_max=xmax, y_min=0.0, y_max=Ly,
                           Nx=Nx_k, Ny=Ny, FT=T)
    return Block(id, mesh; west=west_tag, east=east_tag,
                            south=:wall, north=:wall)
end

function init_parabolic!(state, u_prof_h, T)
    Nξp = state.Nξ_phys; Nηp = state.Nη_phys; ng = state.n_ghost
    fill!(state.f, zero(T))
    @inbounds for j in 1:Nηp, i in 1:Nξp, q in 1:9
        u = u_prof_h[j]
        feq = KrakeNN.equilibrium(D2Q9(), 1.0, u, 0.0, q)
        state.f[i + ng, j + ng, q] = T(feq)
    end
end

function run_slbm(mbm, bcspecs, label)
    n_blocks = length(mbm.blocks)
    ng = 1
    states = [allocate_block_state_2d(b; n_ghost=ng, backend=backend) for b in mbm.blocks]
    u_prof_h = parabolic_inlet(Ny)
    for st in states
        init_parabolic!(st, u_prof_h, T)
    end

    # Per-block extended SLBM geometry + local-τ fields.
    geom_ext = Vector{Any}(undef, n_blocks)
    sp_ext = Vector{Any}(undef, n_blocks); sm_ext = Vector{Any}(undef, n_blocks)
    sp_int = Vector{Any}(undef, n_blocks); sm_int = Vector{Any}(undef, n_blocks)
    for (k, blk) in enumerate(mbm.blocks)
        mesh_ext, g = build_block_slbm_geometry_extended(blk; n_ghost=ng, local_cfl=false)
        geom_ext[k] = g
        sp_h, sm_h = compute_local_omega_2d(mesh_ext; ν=Float64(ν),
                                              scaling=:none, τ_floor=0.51)
        sp_ext[k] = T.(sp_h); sm_ext[k] = T.(sm_h)
        sp_int[k] = T.(sp_h[(ng+1):(ng+blk.mesh.Nξ), (ng+1):(ng+blk.mesh.Nη)])
        sm_int[k] = T.(sm_h[(ng+1):(ng+blk.mesh.Nξ), (ng+1):(ng+blk.mesh.Nη)])
    end
    # All-zero LI-BB precompute (no cylinder)
    is_solid_ext = [zeros(Bool, ext_dims(st)...) for st in states]
    qwall_ext = Vector{Any}(undef, n_blocks)
    uwx_ext = Vector{Any}(undef, n_blocks)
    uwy_ext = Vector{Any}(undef, n_blocks)
    for (k, blk) in enumerate(mbm.blocks)
        Nxk = blk.mesh.Nξ; Nyk = blk.mesh.Nη
        qwall_ext[k] = extend_interior_field_2d(zeros(T, Nxk, Nyk, 9), ng)
        uwx_ext[k]   = extend_interior_field_2d(zeros(T, Nxk, Nyk, 9), ng)
        uwy_ext[k]   = extend_interior_field_2d(zeros(T, Nxk, Nyk, 9), ng)
    end

    f_out = [similar(st.f) for st in states]
    for k in 1:n_blocks; fill!(f_out[k], zero(T)); end

    for step in 1:steps
        exchange_ghost_2d!(mbm, states)
        fill_physical_wall_ghost_2d!(mbm, states)
        for k in 1:n_blocks
            slbm_trt_libb_step_local_2d!(f_out[k], states[k].f,
                                          states[k].ρ, states[k].ux, states[k].uy,
                                          is_solid_ext[k], qwall_ext[k],
                                          uwx_ext[k], uwy_ext[k],
                                          geom_ext[k], sp_ext[k], sm_ext[k])
        end
        for k in 1:n_blocks
            Nξp = states[k].Nξ_phys; Nηp = states[k].Nη_phys; ng_ = states[k].n_ghost
            int_out = view(f_out[k],    (ng_+1):(ng_+Nξp), (ng_+1):(ng_+Nηp), :)
            int_in  = view(states[k].f, (ng_+1):(ng_+Nξp), (ng_+1):(ng_+Nηp), :)
            apply_bc_rebuild_2d!(int_out, int_in, bcspecs[k], ν, Nξp, Nηp;
                                   sp_field=sp_int[k], sm_field=sm_int[k])
        end
        for k in 1:n_blocks
            states[k].f, f_out[k] = f_out[k], states[k].f
        end
    end

    # Extract centerline ux along x
    x_all = Float64[]; u_all = Float64[]
    j_c = (Ny + 1) ÷ 2
    for (k, blk) in enumerate(mbm.blocks)
        st = states[k]; ng_ = st.n_ghost
        for i in 1:blk.mesh.Nξ
            push!(x_all, blk.mesh.X[i, 1])
            push!(u_all, st.ux[i + ng_, j_c + ng_])
        end
    end
    println("  [$label] centerline ux at mid: $(round(u_all[argmin(abs.(x_all .- Lx/2))], sigdigits=6))")
    return x_all, u_all
end

u_in = parabolic_inlet(Ny)

# Single-block
blk0 = _make_block(:A, 0.0, Lx, Nx; west_tag=:inlet, east_tag=:outlet)
mbm_0 = MultiBlockMesh2D([blk0]; interfaces=Interface[])
bcspec_0 = (BCSpec2D(; west=ZouHeVelocity(u_in),
                       east=ZouHePressure(one(T)),
                       south=HalfwayBB(), north=HalfwayBB()),)
x0, u0 = run_slbm(mbm_0, bcspec_0, "SLBM 1-block")

# 2-block W|E
Nx_W = 32; Nx_E = Nx - Nx_W
x_W_max = (Nx_W - 1) * dx; x_E_min = Nx_W * dx
blk_W = _make_block(:W, 0.0, x_W_max, Nx_W; west_tag=:inlet, east_tag=:interface)
blk_E = _make_block(:E, x_E_min, x_E_min + (Nx_E - 1) * dx, Nx_E;
                     west_tag=:interface, east_tag=:outlet)
mbm_1 = MultiBlockMesh2D([blk_W, blk_E];
                          interfaces=[Interface(; from=(:W, :east), to=(:E, :west))])
bcspec_1 = (BCSpec2D(; west=ZouHeVelocity(u_in), east=HalfwayBB(),
                       south=HalfwayBB(), north=HalfwayBB()),
            BCSpec2D(; west=HalfwayBB(), east=ZouHePressure(one(T)),
                       south=HalfwayBB(), north=HalfwayBB()))
x1, u1 = run_slbm(mbm_1, bcspec_1, "SLBM 2-block")

@assert length(u0) == length(u1)
Δ = maximum(abs.(u0 .- u1))
println("\n=== Level 4 SLBM result ===")
println("max |u_1blk - u_2blk| = $(round(Δ, sigdigits=3))")
if Δ < 1e-10
    println("✅ SLBM multi-block == single-block bit-exact")
elseif Δ < 1e-6
    println("⚠️  SLBM nearly bit-exact (spline refit noise $(round(Δ, sigdigits=3)))")
else
    println("❌ SLBM multi-block diverges ($(round(Δ, sigdigits=3)))")
end
