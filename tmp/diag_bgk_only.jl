# Isolate whether the discrepancy is in BGK step or in apply_bc_rebuild.
# Run exchange + wall_ghost + BGK ONLY (skip apply_bc), then compare.

using KrakeNN, KernelAbstractions

const Lx = 1.0
const Ly = 0.25
const Nx = 65
const Ny = 17
const dx = Lx / (Nx - 1)
const u_max = 0.04
const ν = 0.01
const ω = 1.0 / (3ν + 0.5)
const T = Float64

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

function bgk_only_step!(mbm, states, f_out, is_solid_ext)
    exchange_ghost_2d!(mbm, states)
    fill_physical_wall_ghost_2d!(mbm, states)
    for k in eachindex(states)
        Nx_ext, Ny_ext = ext_dims(states[k])
        fused_bgk_step!(f_out[k], states[k].f,
                         states[k].ρ, states[k].ux, states[k].uy,
                         is_solid_ext[k], Nx_ext, Ny_ext, ω)
    end
end

u_in = parabolic_inlet(Ny)
# L0
blk0 = _make_block(:A, 0.0, Lx, Nx; west_tag=:inlet, east_tag=:outlet)
mbm_0 = MultiBlockMesh2D([blk0]; interfaces=Interface[])
states_0 = [allocate_block_state_2d(mbm_0.blocks[1]; n_ghost=1)]
init_parabolic!(states_0[1], u_in, T)
f_out_0 = [similar(states_0[1].f)]; fill!(f_out_0[1], zero(T))
is_solid_0 = [zeros(Bool, ext_dims(states_0[1])...)]
bgk_only_step!(mbm_0, states_0, f_out_0, is_solid_0)

# L1
Nx_W = 32; Nx_E = Nx - Nx_W
blk_W = _make_block(:W, 0.0, (Nx_W - 1) * dx, Nx_W; west_tag=:inlet, east_tag=:interface)
blk_E = _make_block(:E, Nx_W * dx, Nx_W * dx + (Nx_E - 1) * dx, Nx_E;
                     west_tag=:interface, east_tag=:outlet)
mbm_1 = MultiBlockMesh2D([blk_W, blk_E];
                          interfaces=[Interface(; from=(:W, :east), to=(:E, :west))])
states_1 = [allocate_block_state_2d(b; n_ghost=1) for b in mbm_1.blocks]
for st in states_1; init_parabolic!(st, u_in, T); end
f_out_1 = [similar(st.f) for st in states_1]
for k in eachindex(f_out_1); fill!(f_out_1[k], zero(T)); end
is_solid_1 = [zeros(Bool, ext_dims(st)...) for st in states_1]
bgk_only_step!(mbm_1, states_1, f_out_1, is_solid_1)

function _max_diff_W()
    max_diff = 0.0; max_loc = (0, 0, 0)
    for j in 1:Ny, i in 1:Nx_W, q in 1:9
        a = f_out_0[1][i + 1, j + 1, q]
        b = f_out_1[1][i + 1, j + 1, q]
        if abs(a - b) > max_diff
            max_diff = abs(a - b); max_loc = (i, j, q)
        end
    end
    return max_diff, max_loc
end
function _max_diff_E()
    max_diff = 0.0; max_loc = (0, 0, 0)
    for j in 1:Ny, i in 1:Nx_E, q in 1:9
        a = f_out_0[1][(Nx_W + i) + 1, j + 1, q]
        b = f_out_1[2][i + 1, j + 1, q]
        if abs(a - b) > max_diff
            max_diff = abs(a - b); max_loc = (i, j, q)
        end
    end
    return max_diff, max_loc
end

max_diff_W, max_loc_W = _max_diff_W()
max_diff_E, max_loc_E = _max_diff_E()

println("BGK-only (no apply_bc): after 1 step")
println("  Max |f_out_L0 - f_out_L1_W| = $max_diff_W at $max_loc_W")
println("  Max |f_out_L0 - f_out_L1_E| = $max_diff_E at $max_loc_E")

# Also: check at (33, 2) directly
for q in 1:9
    a = f_out_0[1][33, 2, q]
    b = f_out_1[1][33, 2, q]
    if abs(a - b) > 1e-12
        println("  (33, 2, q=$q): L0=$(round(a, sigdigits=10))  L1_W=$(round(b, sigdigits=10))  Δ=$(round(abs(a-b), sigdigits=3))")
    end
end
