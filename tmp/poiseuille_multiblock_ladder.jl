# Validation ladder — pure multi-block mechanics on Poiseuille flow.
#
# Goal: isolate what's broken in the multi-block pipeline BEFORE we
# add LI-BB / SLBM / cylinder / drag. We compare profiles at steady
# state for the SAME total domain decomposed into 1, 2, or 3 blocks.
#
# Level 0: single-block Cartesian channel, Zou-He parabolic inlet +
#          ZouHePressure(1) outlet, HalfwayBB walls, BGK.
# Level 1: same domain split W|E (2 blocks), non-overlap interface.
# Level 3: same domain split W|C|E (3 blocks, same topology as E3).
#
# Success criterion: u(y) profile at x = Lx/2 is IDENTICAL (to ~1e-10)
# across all three levels. If Levels 0/1 agree but Level 3 diverges,
# the bug is in 3-block-specific code (BC ordering, interface count,
# etc.). If Level 0/1 diverge, the bug is in exchange_ghost_2d! or
# fill_physical_wall_ghost_2d!.

using KrakeNN, KernelAbstractions

const Lx = 1.0
const Ly = 0.25
const Nx = 65                        # dx = 1/64 = 0.015625
const Ny = 17                        # dy = 0.25/16 = 0.015625 (uniform)
const dx = Lx / (Nx - 1)
const u_max = 0.04
const ν = 0.01
const ω = 1.0 / (3ν + 0.5)
const steps = 8000
const backend = KernelAbstractions.CPU()
const T = Float64

# Halfway-BB wall placement: walls sit at y = Y[1] - dy/2 and
# y = Y[Ny] + dy/2. Effective channel height H = Ly + dy. The shifted
# coordinate y_eff = y + dy/2 ∈ [0, H] gives the standard parabola.
const H_eff = Ly + dx
u_analytical(y) = u_max * 4 * (y + dx/2) * (H_eff - (y + dx/2)) / H_eff^2

parabolic_inlet(Ny) = T.([u_analytical((j - 1) * dx) for j in 1:Ny])

"""Return (mesh, block) for a Cartesian block spanning x ∈ [xmin, xmax]
with Nx_k cells. Always y ∈ [0, Ly] with Ny cells.

`west_tag`, `east_tag` take `:inlet`, `:outlet`, or `:interface`.
"""
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

function run_ladder(mbm, bcspecs, label)
    n_blocks = length(mbm.blocks)
    states = [allocate_block_state_2d(b; n_ghost=1, backend=backend) for b in mbm.blocks]
    u_prof_h = parabolic_inlet(Ny)
    for k in 1:n_blocks
        init_parabolic!(states[k], u_prof_h, T)
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

    # Extract ux at x = Lx/2 AND the full centerline profile (y = mid)
    x_target = Lx / 2
    ux_mid = zeros(T, Ny)
    j_center = (Ny + 1) ÷ 2    # center of channel in j
    # Build global x array and ux_centerline(x) by concatenating blocks
    x_all = Float64[]
    u_all = Float64[]
    for k in 1:n_blocks
        blk = mbm.blocks[k]; st = states[k]
        ng = st.n_ghost
        for i in 1:blk.mesh.Nξ
            push!(x_all, blk.mesh.X[i, 1])
            push!(u_all, st.ux[i + ng, j_center + ng])
        end
        x_k_min = blk.mesh.X[1, 1]; x_k_max = blk.mesh.X[end, 1]
        if x_k_min ≤ x_target ≤ x_k_max
            i_best = argmin(abs.(blk.mesh.X[:, 1] .- x_target))
            for j in 1:Ny
                ux_mid[j] = st.ux[i_best + ng, j + ng]
            end
            println("  [$label] x_sampled=$(round(blk.mesh.X[i_best,1], digits=5))  (target $x_target, block :$(blk.id))")
        end
    end
    return ux_mid, x_all, u_all
end

# ================ Level 0 — single block ================
println("=== Level 0 : 1 block ===")
mbm_0 = let
    blk = _make_block(:A, 0.0, Lx, Nx; west_tag=:inlet, east_tag=:outlet)
    MultiBlockMesh2D([blk]; interfaces=Interface[])
end
u_in = parabolic_inlet(Ny)
bcspec_0 = (BCSpec2D(; west=ZouHeVelocity(u_in),
                       east=ZouHePressure(one(T)),
                       south=HalfwayBB(), north=HalfwayBB()),)
ux_0, x0, u0 = run_ladder(mbm_0, bcspec_0, "L0")

# ================ Level 1 — 2 blocks W|E ================
println("=== Level 1 : 2 blocks W|E ===")
# Non-overlap: W has Nx_W cells ending at x_W_end = (Nx_W-1)*dx;
# E starts at x_E_start = Nx_W*dx with Nx_E cells; Nx_total = Nx_W + Nx_E.
Nx_W_1 = 32; Nx_E_1 = Nx - Nx_W_1   # 32 + 33 = 65
x_W_max_1 = (Nx_W_1 - 1) * dx
x_E_min_1 = Nx_W_1 * dx
x_E_max_1 = x_E_min_1 + (Nx_E_1 - 1) * dx
mbm_1 = let
    blk_W = _make_block(:W, 0.0, x_W_max_1, Nx_W_1;
                         west_tag=:inlet, east_tag=:interface)
    blk_E = _make_block(:E, x_E_min_1, x_E_max_1, Nx_E_1;
                         west_tag=:interface, east_tag=:outlet)
    MultiBlockMesh2D([blk_W, blk_E];
                      interfaces=[Interface(; from=(:W, :east), to=(:E, :west))])
end
bcspec_1 = (BCSpec2D(; west=ZouHeVelocity(u_in),
                       east=HalfwayBB(),
                       south=HalfwayBB(), north=HalfwayBB()),
            BCSpec2D(; west=HalfwayBB(),
                       east=ZouHePressure(one(T)),
                       south=HalfwayBB(), north=HalfwayBB()))
ux_1, x1, u1 = run_ladder(mbm_1, bcspec_1, "L1")

# ================ Level 3 — 3 blocks W|C|E ================
println("=== Level 3 : 3 blocks W|C|E ===")
Nx_W_3 = 20; Nx_C_3 = 25; Nx_E_3 = Nx - Nx_W_3 - Nx_C_3   # 20 + 25 + 20 = 65
x_W_max_3 = (Nx_W_3 - 1) * dx
x_C_min_3 = Nx_W_3 * dx
x_C_max_3 = x_C_min_3 + (Nx_C_3 - 1) * dx
x_E_min_3 = (Nx_W_3 + Nx_C_3) * dx
x_E_max_3 = x_E_min_3 + (Nx_E_3 - 1) * dx
mbm_3 = let
    blk_W = _make_block(:W, 0.0,        x_W_max_3, Nx_W_3;
                         west_tag=:inlet, east_tag=:interface)
    blk_C = _make_block(:C, x_C_min_3, x_C_max_3, Nx_C_3;
                         west_tag=:interface, east_tag=:interface)
    blk_E = _make_block(:E, x_E_min_3, x_E_max_3, Nx_E_3;
                         west_tag=:interface, east_tag=:outlet)
    MultiBlockMesh2D([blk_W, blk_C, blk_E];
                      interfaces=[Interface(; from=(:W, :east), to=(:C, :west)),
                                  Interface(; from=(:C, :east), to=(:E, :west))])
end
bcspec_3 = (BCSpec2D(; west=ZouHeVelocity(u_in),
                       east=HalfwayBB(),
                       south=HalfwayBB(), north=HalfwayBB()),
            BCSpec2D(; west=HalfwayBB(), east=HalfwayBB(),
                       south=HalfwayBB(), north=HalfwayBB()),
            BCSpec2D(; west=HalfwayBB(),
                       east=ZouHePressure(one(T)),
                       south=HalfwayBB(), north=HalfwayBB()))
ux_3, x3, u3 = run_ladder(mbm_3, bcspec_3, "L3")

# ================ Report ================
u_ref = [u_analytical((j - 1) * dx) for j in 1:Ny]
err_0 = maximum(abs.(ux_0 .- u_ref))
err_1 = maximum(abs.(ux_1 .- u_ref))
err_3 = maximum(abs.(ux_3 .- u_ref))
diff_01 = maximum(abs.(ux_0 .- ux_1))
diff_03 = maximum(abs.(ux_0 .- ux_3))
diff_13 = maximum(abs.(ux_1 .- ux_3))

println("\n=== Summary ===")
println("u_max analytical = $(round(maximum(u_ref), sigdigits=6))")
println("u_max level 0    = $(round(maximum(ux_0), sigdigits=6))")
println("u_max level 1    = $(round(maximum(ux_1), sigdigits=6))")
println("u_max level 3    = $(round(maximum(ux_3), sigdigits=6))")
println("")
println("max|u - u_analytical| at x=Lx/2:")
println("  Level 0 (1 block)   : $(round(err_0, sigdigits=3))")
println("  Level 1 (2 blocks)  : $(round(err_1, sigdigits=3))")
println("  Level 3 (3 blocks)  : $(round(err_3, sigdigits=3))")
println("")
println("max|u_Li - u_Lj| :")
println("  L0 vs L1 : $(round(diff_01, sigdigits=3))")
println("  L0 vs L3 : $(round(diff_03, sigdigits=3))")
println("  L1 vs L3 : $(round(diff_13, sigdigits=3))")
println("")
println("=== Verdict ===")
if diff_01 < 1e-10 && diff_03 < 1e-10
    println("✅ multi-block mechanics = single-block (identical to 1e-10)")
elseif diff_01 < 1e-4 && diff_03 < 1e-4
    println("⚠️  multi-block ≈ single-block, but not bit-exact (drift $(max(diff_01, diff_03)))")
else
    println("❌ multi-block DIVERGES from single-block — interface bug")
end

# ================ Spatial diagnosis: where does L1 diverge from L0? ================
println("\n=== Spatial diagnosis (centerline ux along x) ===")
# Merge x into common grid (all three levels have same Nx total)
@assert length(x0) == length(x1) == length(x3) == Nx
du01 = u0 .- u1
du03 = u0 .- u3
i_max_01 = argmax(abs.(du01))
i_max_03 = argmax(abs.(du03))
println("L0-L1: max |Δu|=$(round(maximum(abs.(du01)), sigdigits=3)) at x=$(round(x0[i_max_01], digits=4)) (i=$i_max_01, u0=$(round(u0[i_max_01], sigdigits=4)), u1=$(round(u1[i_max_01], sigdigits=4)))")
println("L0-L3: max |Δu|=$(round(maximum(abs.(du03)), sigdigits=3)) at x=$(round(x0[i_max_03], digits=4)) (i=$i_max_03)")
# Print a few x positions around the interface
println("\nAt a few x positions (centerline):")
println("  x         u0           u1           u3           Δ01          Δ03")
for i in [1, 16, 32, 33, 34, 40, 48, 65]
    println("  $(round(x0[i],digits=4))   $(round(u0[i],sigdigits=6))   $(round(u1[i],sigdigits=6))   $(round(u3[i],sigdigits=6))   $(round(u0[i]-u1[i],sigdigits=3))   $(round(u0[i]-u3[i],sigdigits=3))")
end
println("\n(L1 interface at x=$(round(x_W_max_1, digits=4))→$(round(x_E_min_1, digits=4)); L3 interfaces at $(round(x_W_max_3, digits=4))→$(round(x_C_min_3, digits=4)) and $(round(x_C_max_3, digits=4))→$(round(x_E_min_3, digits=4)))")
