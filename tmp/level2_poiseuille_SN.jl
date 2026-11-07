# Level 2: 2-block S|N split (interface is horizontal) on Poiseuille.
# Validates the south-north exchange path and north-south BC paths, which
# are orthogonal to L1 (W|E split). Must match L0 single-block bit-exact.

using KrakeNN, KernelAbstractions

const Lx = 1.0
const Ly = 0.25
const Nx = 65
const Ny = 17
const dx = Lx / (Nx - 1)
const u_max = 0.04
const ν = 0.01
const ω = 1.0 / (3ν + 0.5)
const steps = 2000
const T = Float64

const H_eff = Ly + dx
u_analytical(y) = u_max * 4 * (y + dx/2) * (H_eff - (y + dx/2)) / H_eff^2

function _make_block(id, xmin, xmax, ymin, ymax, Nx_k, Ny_k;
                      west_tag, east_tag, south_tag, north_tag)
    mesh = cartesian_mesh(; x_min=xmin, x_max=xmax, y_min=ymin, y_max=ymax,
                           Nx=Nx_k, Ny=Ny_k, FT=T)
    return Block(id, mesh; west=west_tag, east=east_tag,
                            south=south_tag, north=north_tag)
end

function init_parabolic!(state, u_prof_offset_j, T)
    Nξp = state.Nξ_phys; Nηp = state.Nη_phys; ng = state.n_ghost
    fill!(state.f, zero(T))
    @inbounds for j in 1:Nηp, i in 1:Nξp, q in 1:9
        # j is LOCAL to this block; convert to global via offset
        j_global = u_prof_offset_j + j
        u = u_analytical((j_global - 1) * dx)
        feq = KrakeNN.equilibrium(D2Q9(), 1.0, u, 0.0, q)
        state.f[i + ng, j + ng, q] = T(feq)
    end
end

function run_level(mbm, bcspecs, y_offsets, label)
    n_blocks = length(mbm.blocks)
    states = [allocate_block_state_2d(b; n_ghost=1) for b in mbm.blocks]
    for k in 1:n_blocks
        init_parabolic!(states[k], y_offsets[k], T)
    end
    f_out = [similar(st.f) for st in states]
    for k in 1:n_blocks; fill!(f_out[k], zero(T)); end
    is_solid_ext = [zeros(Bool, ext_dims(states[k])...) for k in 1:n_blocks]

    for step in 1:steps
        exchange_ghost_2d!(mbm, states)
        fill_physical_wall_ghost_2d!(mbm, states)
        for k in 1:n_blocks
            Nx_ext, Ny_ext = ext_dims(states[k])
            fused_bgk_step!(f_out[k], states[k].f,
                             states[k].ρ, states[k].ux, states[k].uy,
                             is_solid_ext[k], Nx_ext, Ny_ext, ω)
        end
        for k in 1:n_blocks
            Nξp = states[k].Nξ_phys; Nηp = states[k].Nη_phys; ng = states[k].n_ghost
            int_out = view(f_out[k], (ng+1):(ng+Nξp), (ng+1):(ng+Nηp), :)
            int_in  = view(states[k].f, (ng+1):(ng+Nξp), (ng+1):(ng+Nηp), :)
            apply_bc_rebuild_2d!(int_out, int_in, bcspecs[k], ν, Nξp, Nηp)
        end
        for k in 1:n_blocks
            states[k].f, f_out[k] = f_out[k], states[k].f
        end
    end

    # Reconstruct global ux(x, y) by concatenating blocks vertically
    ux_global = zeros(T, Nx, Ny)
    for k in 1:n_blocks
        blk = mbm.blocks[k]; st = states[k]; ng = st.n_ghost
        Nyk = blk.mesh.Nη; offset_j = y_offsets[k]
        for j in 1:Nyk, i in 1:Nx
            ux_global[i, j + offset_j] = st.ux[i + ng, j + ng]
        end
    end
    println("  [$label] assembled global ux grid")
    return ux_global
end

# ---- L0 single block ----
u_in_full = T.([u_analytical((j - 1) * dx) for j in 1:Ny])
blk0 = _make_block(:A, 0.0, Lx, 0.0, Ly, Nx, Ny;
                    west_tag=:inlet, east_tag=:outlet,
                    south_tag=:wall, north_tag=:wall)
mbm_0 = MultiBlockMesh2D([blk0]; interfaces=Interface[])
bcspec_0 = (BCSpec2D(; west=ZouHeVelocity(u_in_full),
                       east=ZouHePressure(one(T)),
                       south=HalfwayBB(), north=HalfwayBB()),)
ux_0 = run_level(mbm_0, bcspec_0, (0,), "L0")

# ---- L2 : S|N split ----
# Bottom block: y ∈ [0, (Ny_S-1)*dx], Ny_S cells
# Top block:    y ∈ [Ny_S*dx, (Ny_S + Ny_N - 1)*dx], Ny_N cells
# Non-overlap between them: 1·dx gap.
Ny_S = 8; Ny_N = Ny - Ny_S  # 8 + 9 = 17
y_S_max = (Ny_S - 1) * dx
y_N_min = Ny_S * dx
y_N_max = y_N_min + (Ny_N - 1) * dx

# Inlet profile for S block (j=1..Ny_S) and N block (j=Ny_S+1..Ny_S+Ny_N)
u_in_S = T.([u_analytical((j - 1) * dx)        for j in 1:Ny_S])
u_in_N = T.([u_analytical((Ny_S + j - 1) * dx) for j in 1:Ny_N])

blk_S = _make_block(:S, 0.0, Lx, 0.0,    y_S_max, Nx, Ny_S;
                     west_tag=:inlet, east_tag=:outlet,
                     south_tag=:wall, north_tag=:interface)
blk_N = _make_block(:N, 0.0, Lx, y_N_min, y_N_max, Nx, Ny_N;
                     west_tag=:inlet, east_tag=:outlet,
                     south_tag=:interface, north_tag=:wall)
mbm_2 = MultiBlockMesh2D([blk_S, blk_N];
                          interfaces=[Interface(; from=(:S, :north), to=(:N, :south))])
bcspec_2 = (BCSpec2D(; west=ZouHeVelocity(u_in_S),
                       east=ZouHePressure(one(T)),
                       south=HalfwayBB(), north=HalfwayBB()),
            BCSpec2D(; west=ZouHeVelocity(u_in_N),
                       east=ZouHePressure(one(T)),
                       south=HalfwayBB(), north=HalfwayBB()))
ux_2 = run_level(mbm_2, bcspec_2, (0, Ny_S), "L2")

max_diff = maximum(abs.(ux_0 .- ux_2))
println("\n=== Level 2 result ===")
println("L0 u_max_domain = $(maximum(ux_0))")
println("L2 u_max_domain = $(maximum(ux_2))")
println("max|L0 - L2|    = $max_diff")
if max_diff < 1e-10
    println("✅ L2 bit-exact vs L0 (multi-block S|N OK)")
else
    println("❌ L2 diverges (max diff $max_diff)")
    i_max, j_max = argmax(abs.(ux_0 .- ux_2)).I
    println("   at (i=$i_max, j=$j_max) → x=$(round((i_max-1)*dx,digits=4)), y=$(round((j_max-1)*dx,digits=4))")
end
