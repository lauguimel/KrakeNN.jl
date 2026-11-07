# Ultra-focused diagnostic: print the 9 pulled values at L1 W's east-most
# south-most interior cell (extended (33, 2)) and compare to L0 at same
# coordinates. Then print the 9 f_out values produced by BGK.

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

# D2Q9 velocity vectors: q=1 rest; q=2 E; q=3 N; q=4 W; q=5 S;
#                       q=6 NE; q=7 NW; q=8 SW; q=9 SE
const CQX = (0,  1, 0, -1,  0,  1, -1, -1,  1)
const CQY = (0,  0, 1,  0, -1,  1,  1, -1, -1)

function print_pulls(label, f, i, j)
    println("=== $label pulls at extended (i=$i, j=$j) ===")
    Nx_ext = size(f, 1); Ny_ext = size(f, 2)
    for q in 1:9
        im = max(i - 1, 1); ip = min(i + 1, Nx_ext)
        jm = max(j - 1, 1); jp = min(j + 1, Ny_ext)
        # Pull source for each q
        if q == 1
            src_i, src_j = i, j
        elseif q == 2
            src_i, src_j = (i > 1 ? im : i), j   # pulls from west
        elseif q == 3
            src_i, src_j = i, (j > 1 ? jm : j)
        elseif q == 4
            src_i, src_j = (i < Nx_ext ? ip : i), j
        elseif q == 5
            src_i, src_j = i, (j < Ny_ext ? jp : j)
        elseif q == 6
            src_i = i > 1 ? im : i
            src_j = j > 1 ? jm : j
        elseif q == 7
            src_i = i < Nx_ext ? ip : i
            src_j = j > 1 ? jm : j
        elseif q == 8
            src_i = i < Nx_ext ? ip : i
            src_j = j < Ny_ext ? jp : j
        elseif q == 9
            src_i = i > 1 ? im : i
            src_j = j < Ny_ext ? jp : j
        end
        println("  q=$q (cqx=$(CQX[q]), cqy=$(CQY[q]))  src=($src_i, $src_j)  val=$(round(f[src_i, src_j, q], sigdigits=8))")
    end
end

u_in = parabolic_inlet(Ny)

# ---- L0 setup + 1 exchange+wall_ghost (no step) ----
blk0 = _make_block(:A, 0.0, Lx, Nx; west_tag=:inlet, east_tag=:outlet)
mbm_0 = MultiBlockMesh2D([blk0]; interfaces=Interface[])
states_0 = [allocate_block_state_2d(mbm_0.blocks[1]; n_ghost=1)]
init_parabolic!(states_0[1], u_in, T)
exchange_ghost_2d!(mbm_0, states_0)  # no-op (no interfaces)
fill_physical_wall_ghost_2d!(mbm_0, states_0)

# ---- L1 setup + 1 exchange+wall_ghost (no step) ----
Nx_W = 32; Nx_E = Nx - Nx_W
blk_W = _make_block(:W, 0.0, (Nx_W - 1) * dx, Nx_W; west_tag=:inlet, east_tag=:interface)
blk_E = _make_block(:E, Nx_W * dx, Nx_W * dx + (Nx_E - 1) * dx, Nx_E;
                     west_tag=:interface, east_tag=:outlet)
mbm_1 = MultiBlockMesh2D([blk_W, blk_E];
                          interfaces=[Interface(; from=(:W, :east), to=(:E, :west))])
states_1 = [allocate_block_state_2d(b; n_ghost=1) for b in mbm_1.blocks]
for st in states_1; init_parabolic!(st, u_in, T); end
exchange_ghost_2d!(mbm_1, states_1)
fill_physical_wall_ghost_2d!(mbm_1, states_1)

# Print pulls at L0 extended (33, 2), L1 W extended (33, 2)
print_pulls("L0", states_0[1].f, 33, 2)
print_pulls("L1 W", states_1[1].f, 33, 2)

# Direct comparison of relevant ghost values
println("\n=== Direct cells comparison ===")
for (i, j) in [(32, 1), (33, 1), (34, 1), (32, 2), (33, 2), (34, 2)]
    for q in 1:9
        a = states_0[1].f[i, j, q]
        b = states_1[1].f[i, j, q]
        if abs(a - b) > 1e-10
            println("  (i=$i, j=$j, q=$q) L0=$a  L1_W=$b  Δ=$(round(abs(a-b), sigdigits=3))")
        end
    end
end
