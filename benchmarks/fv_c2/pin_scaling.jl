# Pressure-pin row scaling (risk flagged by C1): the pinned continuity row is
# `p_P = 0`, coefficient 1, while the other continuity rows carry O(h) velocity
# and O(h²/ν) pressure coefficients and the momentum rows O(ν) / O(h). Measure
# its effect on the Newton linear solve (UMFPACK through the CPU seam) by
# solving J δ = R with the pinned row scaled by s ∈ {1, c, 1e-6, 1e6}, where
# c = median |∂R_c/∂p_P| of the unpinned continuity rows (the "natural" scale).
# The solutions are mathematically identical; their spread, the backward error
# and the 1-norm condition number (dense, small grids) measure the effect.
#
#   julia --project benchmarks/fv_c2/pin_scaling.jl
using KrakeNN, LinearAlgebra, SparseArrays, Printf, Statistics

lid(s, x, y) = s === :N ? (1.0, 0.0) : (0.0, 0.0)

function study(N, Re; dense = N <= 32)
    p = fv_steady_problem(N, N; nu = 1 / Re, boundary_velocity = lid)
    g = p.grid
    # mid-Newton state: two damped Newton steps from rest
    r = solve_incns_fv_newton(; problem = p, maxit = 2)
    U = Array(r.U)
    s = r.solver
    fv_jacobian!(s, U)
    J = copy(s.J)
    R = fv_residual(U, p)
    kpin = findfirst(==(FV_CELL_PINNED), vec(p.tag))
    rp = 3 * (kpin - 1) + 3
    cont_rows = [3 * (c - 1) + 3 for c in 1:N*N if c != kpin]
    c = median(abs(J[k, k]) for k in cont_rows)
    mom = median(abs(J[k, k]) for k in 1:3:3N*N)
    ref = nothing
    @printf("N=%4d Re=%5g  h=%.3g  median|J_cc|=%.3g  median|J_uu|=%.3g  pin row coefficient 1\n",
            N, Re, g.h, c, mom)
    for sc in (1.0, c, 1e-6, 1e6)
        D = ones(size(J, 1)); D[rp] = sc
        Js = Diagonal(D) * J
        Rs = D .* R
        F = lu(Js)
        x = F \ Rs
        be = norm(Js * x - Rs) / norm(Rs)
        ref === nothing && (ref = x)
        dx = norm(x - ref) / norm(ref)
        kap = dense ? cond(Matrix(Js), 1) : NaN
        @printf("    pin scale %-9.3g backward err %.1e | rel. diff vs scale 1 %.1e | cond1 %.2e\n",
                sc, be, dx, kap)
    end
end

for (N, Re) in ((16, 100.0), (32, 100.0), (32, 1000.0), (128, 100.0), (256, 1000.0))
    study(N, Re)
end
