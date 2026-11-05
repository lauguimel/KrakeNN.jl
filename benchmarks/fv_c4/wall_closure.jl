# C4 wall-closure study: does the first-order wall pressure of the C1 residual
# (half-cell closure p_f = p_P, ν_P; pressure first order in the wall cells when
# ν varies, MMS 2026-11-04) degrade a wall objective and its adjoint gradient?
#
# Stokes flow (conv = false, so first-order upwind convection does not mask the
# wall error) in the channel [0, 4] × [0, 1] past the square [1.5, 2] ×
# [0.25, 0.75] (aligned with the cells on every grid, so the geometry is exact),
# parabolic inlet (W), outlet (E), walls S/N. Viscosity
#     ν(x, y; θ) = 1 + θ φ(x, y),   φ = sin(π x / 4) (1 + y)
# θ0 = 0 (constant ν; the gradient direction is still a variable-ν one) and
# θ0 = 1 (ν varies by a factor ~3 over the domain). Grids N = 8 … Nmax cells per
# unit height (Nx = 4N). Objectives:
#   drag(cell)   — FVWallForce(1; region = :solid): the residual's own wall
#                  momentum flux, wall pressure p_f = p_P (C1 closure);
#   drag(linear) — same with the objective-level linear wall-pressure
#                  extrapolation p_f = p_P + (p_P − p_PP)/2 (residual unchanged);
#   Δp           — mean pressure on the face x = 0.5 minus that on x = 3.5
#                  (interior faces, column means).
# For each: J(h), the adjoint gradient dJ/dθ, and observed orders
# p = log2(|X_h − X_{h/2}| / |X_{h/2} − X_{h/4}|). At the coarsest grid the
# adjoint is checked against central differences.
#
#   julia --project benchmarks/fv_c4/wall_closure.jl [Nmax]   (default 128)
using Kraken, LinearAlgebra, Printf

const NMAX = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 128
φ(x, y) = sinpi(x / 4) * (1 + y)

function setup(N, θ0)
    Nx, Ny = 4N, N
    h = 1 / N
    solid = zeros(Bool, Nx, Ny)
    solid[round(Int, 1.5N)+1:2N, round(Int, 0.25N)+1:round(Int, 0.75N)] .= true
    p = fv_steady_problem(Nx, Ny; h, solid, conv = false, nu = (x, y) -> 1 + θ0 * φ(x, y),
                          bc = (W = :inlet, E = :outlet, S = :wall, N = :wall),
                          boundary_velocity = (s, x, y) -> s === :W ? (4y * (1 - y), 0.0) : (0.0, 0.0))
    Φ = vec([φ((i - 0.5) * h, (j - 0.5) * h) for i in 1:Nx, j in 1:Ny])
    map = FVProblemMap((q, θ) -> fv_param_problem(FVViscosityField(), q, 1 .+ θ[1] .* Φ), 1)
    # mean pressure on the faces x = 0.5 and x = 3.5 (column pairs straddling them)
    idx = Int[]; w = Float64[]
    for (xf, sg) in ((0.5, 1.0), (3.5, -1.0))
        i = round(Int, xf * N)
        for ii in (i, i + 1), j in 1:Ny
            push!(idx, fv_index(p.grid, ii, j, 3)); push!(w, sg / (2Ny))
        end
    end
    dp = FVLinearFunctional(idx, w)
    return p, map, dp
end

function run(N, θ0; check_fd = false)
    p, map, dp = setup(N, θ0)
    s = fv_newton_solver(p)
    U = zeros(fv_nunknowns(p))
    t = @elapsed st = fv_newton!(s, U; atol = 1e-13, rtol = 1e-14)
    st.converged || @warn "Newton not converged" N θ0
    objs = (FVWallForce(1; region = :solid), FVWallForce(1; region = :solid, pressure = :linear), dp)
    out = Tuple{Float64,Float64}[]
    for obj in objs
        r = fv_steady_gradient(s, U, [θ0], obj; params = map)
        push!(out, (r.J, r.grad[1]))
        if check_fd
            ε = 1e-5
            Jε(θ) = (s2 = fv_newton_solver(p); fv_set_params!(s2, map, [θ]); V = copy(U);
                     fv_newton!(s2, V; atol = 1e-13, rtol = 1e-14); fv_objective(obj, V, s2.prob))
            fd = (Jε(θ0 + ε) - Jε(θ0 - ε)) / 2ε
            @printf("    FD check N=%d θ0=%g %-28s adjoint %.10e  FD %.10e  rel %.1e\n", N, θ0,
                    string(nameof(typeof(obj)), obj isa FVWallForce ? "(" * string(obj.pressure) * ")" : ""),
                    r.grad[1], fd, abs(r.grad[1] - fd) / abs(fd))
        end
    end
    # pressure part of the drag (ν = 0 in the force evaluation), for diagnosis
    p0 = Kraken._fv_with(s.prob; nu = zero(s.prob.nu))
    jp = fv_objective(FVWallForce(1; region = :solid), U, p0)
    return out, jp, t
end

order(a, b, c) = log2(abs(a - b) / abs(b - c))

println("C4 wall-closure study (Stokes, obstacle channel), ", Sys.MACHINE, ", Julia ", VERSION)
Ns = Int[]
let N = 8
    while N <= NMAX
        push!(Ns, N); N *= 2
    end
end
names = ("drag(cell)", "drag(linear)", "Δp")
for θ0 in (0.0, 1.0)
    println("\n=== θ0 = $θ0 ", θ0 == 0 ? "(constant ν)" : "(variable ν, max/min ≈ 3)", " ===")
    res = []; jps = Float64[]
    for (k, N) in enumerate(Ns)
        out, jp, t = run(N, θ0; check_fd = k == 1)
        push!(res, out); push!(jps, jp)
        @printf("  N=%4d (%6d unknowns, Newton %.2f s) | ", N, 12N^2, t)
        for (m, nm) in enumerate(names)
            @printf("%s J=% .8e dJ/dθ=% .8e | ", nm, out[m][1], out[m][2])
        end
        @printf("pressure part of drag(cell) % .6e\n", jp)
        flush(stdout)
    end
    println("  observed orders (from three successive grids, finest triple last):")
    for (m, nm) in enumerate(names)
        oJ = [order(res[k][m][1], res[k+1][m][1], res[k+2][m][1]) for k in 1:length(Ns)-2]
        og = [order(res[k][m][2], res[k+1][m][2], res[k+2][m][2]) for k in 1:length(Ns)-2]
        @printf("    %-13s J: %s | dJ/dθ: %s\n", nm, join((@sprintf("%.2f", x) for x in oJ), ", "),
                join((@sprintf("%.2f", x) for x in og), ", "))
    end
    op = [order(jps[k], jps[k+1], jps[k+2]) for k in 1:length(Ns)-2]
    @printf("    %-13s J: %s\n", "drag pressure", join((@sprintf("%.2f", x) for x in op), ", "))
    # Richardson extrapolation of the finest pair assuming the observed order
    for (m, nm) in enumerate(names)
        a, b, c = res[end-2][m], res[end-1][m], res[end][m]
        for (q, lab) in ((1, "J"), (2, "dJ/dθ"))
            pobs = order(a[q], b[q], c[q])
            ext = c[q] + (c[q] - b[q]) / (2^pobs - 1)
            @printf("    %-13s %-6s finest %.8e, Richardson (p=%.2f) %.8e, finest-grid rel. error ≈ %.1e\n",
                    nm, lab, c[q], pobs, ext, abs(c[q] - ext) / abs(ext))
        end
    end
end
