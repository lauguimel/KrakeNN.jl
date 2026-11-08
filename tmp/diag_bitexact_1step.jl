# After-1-step bit-exactness diagnostic.
#
# Both L0 (single block) and L1 (2-block W|E) are initialised with the
# SAME parabolic equilibrium on their interior. Then we run exactly
# ONE step of {exchange_ghost + wall_ghost + fused_bgk + apply_bc}.
# Because the initial conditions are identical and the exchange is
# supposed to produce bit-exact ghost data, f_out should match between
# levels at every shared (physical) cell.
#
# If it doesn't, the bug is in exchange_ghost_2d!, wall_ghost fill,
# or the kernel's treatment of ghost cells.

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

function init_parabolic!(state::BlockState2D, u_prof_h, T)
    Nξp = state.Nξ_phys; Nηp = state.Nη_phys; ng = state.n_ghost
    fill!(state.f, zero(T))
    @inbounds for j in 1:Nηp, i in 1:Nξp, q in 1:9
        u = u_prof_h[j]
        feq = KrakeNN.equilibrium(D2Q9(), 1.0, u, 0.0, q)
        state.f[i + ng, j + ng, q] = T(feq)
    end
end

function one_step!(mbm, states, f_out, is_solid_ext, bcspecs)
    exchange_ghost_2d!(mbm, states)
    fill_physical_wall_ghost_2d!(mbm, states)
    for k in eachindex(states)
        Nx_ext, Ny_ext = ext_dims(states[k])
        fused_bgk_step!(f_out[k], states[k].f,
                         states[k].ρ, states[k].ux, states[k].uy,
                         is_solid_ext[k], Nx_ext, Ny_ext, ω)
    end
    for k in eachindex(states)
        Nξp = states[k].Nξ_phys; Nηp = states[k].Nη_phys; ng = states[k].n_ghost
        int_out = view(f_out[k], (ng+1):(ng+Nξp), (ng+1):(ng+Nηp), :)
        int_in  = view(states[k].f, (ng+1):(ng+Nξp), (ng+1):(ng+Nηp), :)
        apply_bc_rebuild_2d!(int_out, int_in, bcspecs[k], ν, Nξp, Nηp)
    end
    return nothing
end

# ---- L0 ----
u_in = parabolic_inlet(Ny)
blk0 = _make_block(:A, 0.0, Lx, Nx; west_tag=:inlet, east_tag=:outlet)
mbm_0 = MultiBlockMesh2D([blk0]; interfaces=Interface[])
states_0 = [allocate_block_state_2d(mbm_0.blocks[1]; n_ghost=1)]
init_parabolic!(states_0[1], u_in, T)
f_out_0 = [similar(states_0[1].f)]; fill!(f_out_0[1], zero(T))
is_solid_0 = [zeros(Bool, ext_dims(states_0[1])...)]
bcspecs_0 = (BCSpec2D(; west=ZouHeVelocity(u_in),
                       east=ZouHePressure(one(T)),
                       south=HalfwayBB(), north=HalfwayBB()),)
one_step!(mbm_0, states_0, f_out_0, is_solid_0, bcspecs_0)

# ---- L1 ----
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
bcspecs_1 = (BCSpec2D(; west=ZouHeVelocity(u_in), east=HalfwayBB(),
                       south=HalfwayBB(), north=HalfwayBB()),
             BCSpec2D(; west=HalfwayBB(), east=ZouHePressure(one(T)),
                       south=HalfwayBB(), north=HalfwayBB()))
one_step!(mbm_1, states_1, f_out_1, is_solid_1, bcspecs_1)

# ---- Compare interior f_out at every common cell ----
# L0 f_out extended [1..67, 1..19, 9]; interior at [2..66, 2..18].
# L1 W extended [1..34, 1..19]; interior [2..33]. Global i=1..32 → W local i=1..32.
# L1 E extended [1..35, 1..19]; interior [2..34]. Global i=33..65 → E local i=1..33.

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

println("After 1 step of {exchange + wall_ghost + BGK + apply_bc}:")
println("")
println("Max |f_out_L0 - f_out_L1_W|  at interior cells: $max_diff_W")
println("  at (i_W=$(max_loc_W[1]), j=$(max_loc_W[2]), q=$(max_loc_W[3])) → x=$(round((max_loc_W[1]-1)*dx, digits=4))")
a = f_out_0[1][max_loc_W[1] + 1, max_loc_W[2] + 1, max_loc_W[3]]
b = f_out_1[1][max_loc_W[1] + 1, max_loc_W[2] + 1, max_loc_W[3]]
println("  L0=$(round(a, sigdigits=10))  L1_W=$(round(b, sigdigits=10))")

println("")
println("Max |f_out_L0 - f_out_L1_E|  at interior cells: $max_diff_E")
println("  at (i_E=$(max_loc_E[1]), j=$(max_loc_E[2]), q=$(max_loc_E[3])) → x=$(round(Nx_W*dx + (max_loc_E[1]-1)*dx, digits=4))")
a = f_out_0[1][(Nx_W + max_loc_E[1]) + 1, max_loc_E[2] + 1, max_loc_E[3]]
b = f_out_1[2][max_loc_E[1] + 1, max_loc_E[2] + 1, max_loc_E[3]]
println("  L0=$(round(a, sigdigits=10))  L1_E=$(round(b, sigdigits=10))")

# Also check: cells at interface (W's last and E's first)
println("\n=== Cells at the interface ===")
for j in [1, 8, 17]
    # W's east-most interior: W local i=Nx_W=32
    # Global L0 i = Nx_W = 32, extended index i=33
    # L1 W extended index i = Nx_W + 1 = 33
    i_W = Nx_W  # W local i
    i_E = 1     # E local i
    println("j=$j (W i=$i_W = L0 i=$(Nx_W), E i=$i_E = L0 i=$(Nx_W+1)):")
    for q in 1:9
        a0_W = f_out_0[1][Nx_W + 1, j + 1, q]
        a1_W = f_out_1[1][i_W + 1, j + 1, q]
        a0_E = f_out_0[1][Nx_W + 2, j + 1, q]
        a1_E = f_out_1[2][i_E + 1, j + 1, q]
        dW = abs(a0_W - a1_W); dE = abs(a0_E - a1_E)
        if dW > 1e-15 || dE > 1e-15
            println("  q=$q  ΔW=$(round(dW, sigdigits=3))  ΔE=$(round(dE, sigdigits=3))")
        end
    end
end
