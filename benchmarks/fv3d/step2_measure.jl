# Step-2 measurement of the 3D steady FV Newton solver on the GPU (plan
# "solveur FV stationnaire 3D", I1; cross-check with work package 1, the cuDSS
# memory/time proxy 32³–96³).
#
# (0) Kernel contract on a small grid (FV3D_CHECK_N, default 8): CPU vs GPU
#     residual (Float64 and dual state) and coloured Jacobian values within
#     1e-12 relative, for Newtonian, GNF and law viscosity; zero device bytes
#     allocated by fv_residual! (CUDA.@allocated). Gated.
# (1) For N in FV3D_NS (default 32,48,64): lid-driven cavity at Re = 100,
#     Newtonian and GNF (Carreau–Yasuda, ν0 = 0.01), from rest with
#     pseudo-transient continuation. Reported: unknowns, nnz, max row, colours,
#     pattern/colouring time, device memory in use after the solver build and
#     after the solve (driver-level, includes cuDSS factors), first
#     factorisation (analysis + numeric) time, median time per later Newton
#     iteration (assembly + refactorisation + solve + line search),
#     iterations. Gated only on convergence.
#
#   FV3D_BACKEND=cuda FV3D_NS=32,48,64 julia --project=<env> benchmarks/fv3d/step2_measure.jl
#   FV3D_BACKEND=cpu  FV3D_NS=6,8      julia --project benchmarks/fv3d/step2_measure.jl   (smoke)
if lowercase(get(ENV, "FV3D_BACKEND", "cuda")) == "cuda"
    using CUDA
    CUDA.functional() || error("CUDA not functional")
    using CUDSS
end
include(joinpath(@__DIR__, "fv3d_common.jl"))
using Statistics
using Random
const ForwardDiff = KrakeNN.ForwardDiff

const CY = CarreauYasuda(0.01, 0.001, 5.0, 2.0, 0.5)
lid(s, x, y, z) = s === :T ? (1.0, 0.0, 0.0) : (0.0, 0.0, 0.0)

function check_problems(N)
    field = [0.2 * i / N + 0.1 * j / N - 0.05 * k / N for i in 1:N, j in 1:N, k in 1:N]
    law = LearnedViscosity(TinyMLP{2,2,1}(); nu_ref = 0.05, monotone = false)
    theta = law_init(law; rng = MersenneTwister(33))
    io = (W = :inlet, E = :outlet, S = :wall, N = :wall, B = :periodic, T = :periodic)
    solid = falses(N, N, N); solid[N ÷ 2:N ÷ 2 + 1, 1:N ÷ 2, :] .= true
    inflow(s, x, y, z) = s === :W ? (1.0, 0.1, 0.0) : (0.0, 0.0, 0.0)
    return (("newtonian", fv_steady_problem(N, N, N; nu = (x, y, z) -> 0.05 * (1 + x * y), bc = io, solid,
                                            boundary_velocity = inflow)),
            ("gnf", fv_steady_problem(N, N, N; nu = 0.01, boundary_velocity = lid, rheology = CY, gnf_eps = 1e-3)),
            ("law", fv_steady_problem(N, N, N; boundary_velocity = lid,
                                      viscosity = FVLawViscosity(law, theta, (field,); eps = 0.02))))
end

function kernel_contract(N, failures)
    out = Dict{String,Any}()
    rng = MersenneTwister(7)
    S = ForwardDiff.Dual{KrakeNN.FVJacobianTag,Float64,4}
    for (name, p) in check_problems(N)
        n = fv_nunknowns(p)
        U = 0.3 .* randn(rng, n)
        Ud = [S(U[k], ForwardDiff.Partials(ntuple(_ -> randn(rng), 4))) for k in 1:n]
        R = fv_residual(U, p); Rd = fv_residual(Ud, p)
        to = fv3d_to()
        dp = fv_adapt(to, p)
        dU = to(U); dR = similar(dU); dUd = to(Ud); dRd = similar(dUd)
        fv_residual!(dR, dU, dp); fv_residual!(dRd, dUd, dp)
        e_r = norm(Array(dR) - R) / norm(R)
        e_d = maximum(norm(ForwardDiff.partials.(Array(dRd), q) .- ForwardDiff.partials.(Rd, q)) /
                      norm(ForwardDiff.partials.(Rd, q)) for q in 1:4)
        a = fv3d_devalloc(() -> fv_residual!(dR, dU, dp))
        ad = fv3d_devalloc(() -> fv_residual!(dRd, dUd, dp))
        sc = fv_newton_solver(p)
        fv_jacobian!(sc, U)
        sg = fv_newton_solver(p; to, linsolve = fv3d_linsolve())
        fv_jacobian!(sg, dU)
        # the device Jacobian is stored in CSR order (cuDSS), the host one in CSC
        Jc = FV3D_GPU ? nonzeros(sparse(transpose(sc.J))) : nonzeros(sc.J)
        Jg = Vector(Array(sg.nzval))
        e_j = norm(Jg - Jc) / norm(Jc)
        @printf("SUMMARY contract %-9s N=%d  residual %.1e  dual %.1e  jacobian %.1e  device bytes %d / %d\n",
                name, N, e_r, e_d, e_j, a, ad)
        ok = e_r <= 1e-12 && e_d <= 1e-12 && e_j <= 1e-12 && a <= 0 && ad <= 0
        ok || push!(failures, "kernel contract $name")
        out[name] = Dict("residual_rel" => e_r, "dual_rel" => e_d, "jacobian_rel" => e_j,
                         "device_bytes" => a, "device_bytes_dual" => ad, "pass" => ok)
    end
    return out
end

function measure(N, gnf)
    p = gnf ? fv_steady_problem(N, N, N; nu = 0.01, boundary_velocity = lid, rheology = CY, gnf_eps = 1e-3) :
              fv_steady_problem(N, N, N; nu = 0.01, boundary_velocity = lid)
    GC.gc(); FV3D_GPU && CUDA.reclaim()
    m0 = fv3d_devused()
    to = fv3d_to()
    t_build = @elapsed s = fv_newton_solver(p; to, linsolve = fv3d_linsolve())
    m1 = fv3d_devused()
    S = fv_jacobian_pattern(p)
    rows = diff(sparse(transpose(S)).colptr)
    ncol = s.asm.ncolors
    U = to(zeros(fv_nunknowns(p)))
    t = @elapsed begin
        st = fv_newton!(s, U; atol = 1e-10, rtol = 1e-12, maxit = 60, dtau0 = 1.0, dtau_max = 1e4)
        fv3d_sync()
    end
    m2 = fv3d_devused()
    per_it = st.t_assembly .+ st.t_factor .+ st.t_solve .+ st.t_linesearch
    later = length(per_it) > 1 ? median(per_it[2:end]) : NaN
    row = Dict{String,Any}("N" => N, "unknowns" => fv_nunknowns(p), "nnz" => nnz(S),
                           "max_row" => maximum(rows), "colours" => ncol,
                           "t_pattern" => s.t_pattern, "t_coloring" => s.t_coloring, "t_build" => t_build,
                           "mem_build_GiB" => (m1 - m0) / 2^30, "mem_after_solve_GiB" => (m2 - m0) / 2^30,
                           "t_first_factor" => isempty(st.t_factor) ? NaN : st.t_factor[1],
                           "t_iteration_median" => later, "t_solve_total" => t,
                           "iterations" => st.iterations, "converged" => st.converged)
    @printf("SUMMARY %-9s N=%3d  n=%d nnz=%d maxrow=%d colours=%d  pattern %.1f s colouring %.1f s  mem %.2f/%.2f GiB  first factor %.2f s  iter %.2f s  it=%d conv=%s  total %.1f s\n",
            gnf ? "gnf" : "newtonian", N, row["unknowns"], row["nnz"], row["max_row"], ncol, s.t_pattern,
            s.t_coloring, row["mem_build_GiB"], row["mem_after_solve_GiB"], row["t_first_factor"], later,
            st.iterations, st.converged, t)
    flush(stdout)
    return row
end

fv3d_banner("step2_measure")
failures = String[]
results = Dict{String,Any}("backend" => FV3D_BACKEND)
results["contract"] = kernel_contract(parse(Int, get(ENV, "FV3D_CHECK_N", FV3D_GPU ? "8" : "5")), failures)
for gnf in (false, true), N in fv3d_ints("FV3D_NS", FV3D_GPU ? "32,48,64" : "6,8")
    key = (gnf ? "gnf_N" : "newtonian_N") * string(N)
    row = try
        measure(N, gnf)
    catch err
        println("ERROR ", key, ": ", first(sprint(showerror, err), 400)); flush(stdout)
        push!(failures, "$key error"); nothing
    end
    row === nothing && continue
    row["converged"] || push!(failures, "$key not converged")
    results[key] = row
end
results["failures"] = failures
fv3d_write(fv3d_out("step2_measure"), results)
exit(fv3d_verdict("step2_measure", failures))
