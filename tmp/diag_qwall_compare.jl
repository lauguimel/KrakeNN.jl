# Compare q_wall values between 1-block and 3-block C setups to see
# if precompute_q_wall_cylinder produces identical cut-link patterns.

using KrakeNN, KernelAbstractions

const T = Float64
const Lx = 1.0; const Ly = 0.5
const cx_p = 0.5; const cy_p = 0.245
const R_p = 0.025
const R_bubble = 0.15
const D_lu = 20
const dx = 2 * R_p / D_lu
const R_lu = R_p / dx
const Nx_total = round(Int, Lx / dx) + 1
const Ny = round(Int, Ly / dx) + 1

# 1-block: cx in lattice units (xf convention). x0=0.
cx_1 = (cx_p - 0.0) / dx + 1
cy_1 = (cy_p - 0.0) / dx + 1
q_1, s_1 = precompute_q_wall_cylinder(Nx_total, Ny, cx_1, cy_1, R_lu; FT=T)

# 3-block C: x0 = (Nx_W)*dx = 140*dx = 0.35
x_C_west = cx_p - R_bubble
Nx_W = round(Int, x_C_west / dx)
Nx_C = round(Int, ((cx_p + R_bubble) - x_C_west) / dx) + 1
x0_C = Nx_W * dx
cx_C = (cx_p - x0_C) / dx + 1
cy_C = (cy_p - 0.0) / dx + 1
q_C, s_C = precompute_q_wall_cylinder(Nx_C, Ny, cx_C, cy_C, R_lu; FT=T)

println("Nx=$Nx_total Ny=$Ny  Nx_C=$Nx_C")
println("1-block: cx=$cx_1 cy=$cy_1  solid cells: $(sum(s_1))")
println("3-block C: cx=$cx_C cy=$cy_C  solid cells: $(sum(s_C))")

function _count_nz(q, Nx, Ny)
    n = 0
    for j in 1:Ny, i in 1:Nx
        any(q[i, j, qq] > 0 for qq in 2:9) && (n += 1)
    end
    return n
end
nzq_1 = _count_nz(q_1, Nx_total, Ny)
nzq_C = _count_nz(q_C, Nx_C, Ny)
println("Non-zero q_wall cells: 1-block=$nzq_1  C=$nzq_C  Δ=$(nzq_1 - nzq_C)")

# Sum of q_wall (same geometry should give same sum)
sq_1 = sum(q_1)
sq_C = sum(q_C)
println("Σ q_wall: 1-block=$(round(sq_1, sigdigits=8))  C=$(round(sq_C, sigdigits=8))  diff=$(round(sq_1 - sq_C, sigdigits=3))")

# Map of cut-link pattern relative to cylinder center
println("\nCells near cylinder (1-block) with q_wall > 0:")
i_range_1 = (round(Int, cx_1) - 12):(round(Int, cx_1) + 12)
j_range   = (round(Int, cy_1) - 12):(round(Int, cy_1) + 12)
count_1 = 0
for j in j_range, i in i_range_1
    if 1 <= i <= Nx_total && any(q_1[i, j, q] > 0 for q in 2:9)
        count_1 += 1
    end
end
println("  1-block window count = $count_1")

# Same window shifted by Nx_W (so mapped onto C)
i_range_C = (round(Int, cx_C) - 12):(round(Int, cx_C) + 12)
count_C = 0
for j in j_range, i in i_range_C
    if 1 <= i <= Nx_C && any(q_C[i, j, q] > 0 for q in 2:9)
        count_C += 1
    end
end
println("  C window count = $count_C")

# Diff per-cell
println("\nPer-cell diff in window (should be 0):")
max_cell_diff = 0.0; max_loc = (0, 0)
for j in j_range, i_k in 1:length(i_range_1)
    i_1 = i_range_1[i_k]; i_C = i_range_C[i_k]
    (1 <= i_1 <= Nx_total && 1 <= i_C <= Nx_C) || continue
    for q in 1:9
        d = abs(q_1[i_1, j, q] - q_C[i_C, j, q])
        if d > max_cell_diff
            max_cell_diff = d
            max_loc = (i_k, j)
        end
    end
end
println("  Max per-cell diff = $(round(max_cell_diff, sigdigits=3)) at shift-index $(max_loc[1]), j=$(max_loc[2])")
