# Level 2 simplified — uniform flow (equilibrium) in S|N multi-block.
# All 4 sides = HalfwayBB (walls) on EACH block except interface.
# If exchange is correct, uniform equilibrium remains bit-exact.

using KrakeNN, KernelAbstractions

const Lx = 1.0
const Ly = 0.25
const Nx = 33
const Ny = 17
const dx = Lx / (Nx - 1)
const u0 = 0.04
const ν = 0.01
const ω = 1.0 / (3ν + 0.5)
const steps = 1000
const T = Float64

function _make_block(id, xmin, xmax, ymin, ymax, Nx_k, Ny_k;
                      west_tag, east_tag, south_tag, north_tag)
    mesh = cartesian_mesh(; x_min=xmin, x_max=xmax, y_min=ymin, y_max=ymax,
                           Nx=Nx_k, Ny=Ny_k, FT=T)
    return Block(id, mesh; west=west_tag, east=east_tag,
                            south=south_tag, north=north_tag)
end

function init_uniform!(state, u, T)
    Nξp = state.Nξ_phys; Nηp = state.Nη_phys; ng = state.n_ghost
    fill!(state.f, zero(T))
    @inbounds for j in 1:Nηp, i in 1:Nξp, q in 1:9
        feq = KrakeNN.equilibrium(D2Q9(), 1.0, u, 0.0, q)
        state.f[i + ng, j + ng, q] = T(feq)
    end
end

# Setup: all-wall box, S|N split. Uniform u=0.04 throughout (non-zero).
Ny_S = 8; Ny_N = Ny - Ny_S
y_S_max = (Ny_S - 1) * dx
y_N_min = Ny_S * dx
y_N_max = y_N_min + (Ny_N - 1) * dx

blk_S = _make_block(:S, 0.0, Lx, 0.0, y_S_max, Nx, Ny_S;
                     west_tag=:wall, east_tag=:wall,
                     south_tag=:wall, north_tag=:interface)
blk_N = _make_block(:N, 0.0, Lx, y_N_min, y_N_max, Nx, Ny_N;
                     west_tag=:wall, east_tag=:wall,
                     south_tag=:interface, north_tag=:wall)
mbm_2 = MultiBlockMesh2D([blk_S, blk_N];
                          interfaces=[Interface(; from=(:S, :north), to=(:N, :south))])

bcspec_2 = (BCSpec2D(; west=HalfwayBB(), east=HalfwayBB(),
                       south=HalfwayBB(), north=HalfwayBB()),
            BCSpec2D(; west=HalfwayBB(), east=HalfwayBB(),
                       south=HalfwayBB(), north=HalfwayBB()))

states = [allocate_block_state_2d(b; n_ghost=1) for b in mbm_2.blocks]
for st in states
    init_uniform!(st, u0, T)
end
f_out = [similar(st.f) for st in states]
for k in 1:2; fill!(f_out[k], zero(T)); end
is_solid_ext = [zeros(Bool, ext_dims(states[k])...) for k in 1:2]

# Compute initial equilibrium sum (should be uniform)
println("Initial ρ: ", [sum(states[k].f[2:end-1, 2:end-1, :])/((Nx*size(states[k].f,2)-2)) for k in 1:2])

for step in 1:steps
    exchange_ghost_2d!(mbm_2, states)
    fill_physical_wall_ghost_2d!(mbm_2, states)
    for k in 1:2
        Nx_ext, Ny_ext = ext_dims(states[k])
        fused_bgk_step!(f_out[k], states[k].f,
                         states[k].ρ, states[k].ux, states[k].uy,
                         is_solid_ext[k], Nx_ext, Ny_ext, ω)
    end
    for k in 1:2
        Nξp = states[k].Nξ_phys; Nηp = states[k].Nη_phys; ng = states[k].n_ghost
        int_out = view(f_out[k], (ng+1):(ng+Nξp), (ng+1):(ng+Nηp), :)
        int_in  = view(states[k].f, (ng+1):(ng+Nξp), (ng+1):(ng+Nηp), :)
        apply_bc_rebuild_2d!(int_out, int_in, bcspec_2[k], ν, Nξp, Nηp)
    end
    for k in 1:2
        states[k].f, f_out[k] = f_out[k], states[k].f
    end
    if step in (1, 10, 100, 500, 1000)
        nan_S = any(isnan, states[1].ρ); nan_N = any(isnan, states[2].ρ)
        u_max_S = nan_S ? NaN : maximum(abs.(states[1].ux))
        u_max_N = nan_N ? NaN : maximum(abs.(states[2].ux))
        println("step $step : S u_max=$(round(u_max_S,sigdigits=4)) nan=$nan_S  |  N u_max=$(round(u_max_N,sigdigits=4)) nan=$nan_N")
    end
end
