#!/usr/bin/env julia
# Cost of one adjoint gradient of the steady GNF solver against the cost of the
# central finite-difference gradient it replaces (2p forward solves per
# forcing), on the obstacle channel of the adjoint check.
# See README.md in this directory.
#
#   julia --project=. benchmarks/validation/gnf_calibration/cost_bench.jl \
#         --backend=cuda [--grids=32x16,64x32,128x64,256x128] [options]
#
# On the CPU backend the timings depend on the Julia and BLAS thread counts
# printed in the log header.

include(joinpath(@__DIR__, "common.jl"))

# ------------------------------------------------ fixed before any run ---

# Bench points. `interior`: the four parameters (ln η∞*, ln k*, n, a) of the
# adjoint check at a = 2.5 instead of 2 (an interior value of a), p = 4, so the
# finite-difference gradient costs 2p = 8 forwards per forcing. `calibration`:
# the three parameters of calibrate_carreau.jl at their true values, a = 2
# fixed, p = 3, 2p = 6.
const CB_POINTS = Dict(
    "interior" => (; theta=[log(0.02), log(5.0), 0.7, 2.5], names=("ln_eta_inf", "ln_k", "n", "a"),
                   a=2.5, closure=gc_closure4),
    "calibration" => (; theta=[log(0.02), log(5.0), 0.7], names=("ln_eta_inf", "ln_k", "n"),
                      a=2.0, closure=c -> gc_closure3(c; a=2.0)))
const CB_GSCALES = GC_GSCALES             # (0.1, 0.2, 0.3), defined in common.jl

const CB_DEFAULTS = Dict("backend" => "cuda", "grids" => "32x16,64x32,128x64,256x128",
                         "precision" => "Float64", "solvers" => "kstep,gmres",
                         "points" => "interior,calibration", "forcings" => "1:3",
                         "repeats" => "3", "fd-step" => "1e-4",
                         "out" => "benchmarks/validation/gnf_calibration/results",
                         "allow-unknown-commit" => "no")

const CB_HEADER = (
    "grid", "backend", "precision", "point", "n_params", "a", "forcing", "g_scale", "task",
    "solver", "parameter", "sign", "repeat", "warmup", "wall_s", "steps", "converged",
    "dist_est", "mach", "t_lin_s", "t_copy_s", "t_solve_s", "t_grad_s", "adjoint_iters",
    "operator_products", "adjoint_relres", "adjoint_converged", "fd_gradient",
    "adjoint_gradient", "rel_diff")

const CB_SUMMARY_HEADER = (
    "grid", "backend", "precision", "point", "n_params", "forcing", "g_scale", "solver",
    "repeats", "forward_rest_median_s", "forward_rest_steps", "adjoint_median_s",
    "adjoint_solve_median_s", "adjoint_iters", "operator_products", "fd_forwards",
    "fd_forwards_median_s", "fd_forward_steps", "adjoint_over_fd", "adjoint_over_forward_rest",
    "max_rel_diff_fd_adjoint")

"Observation of the adjoint check: both components at every fluid node, targets 0.8 u/u_scale."
function cb_obs(case, ux, uy)
    obs = gc_obs(case, ux, uy, case.uscale)
    return GNFVelocityObs(obs.node, obs.weight, obs.comp, 0.8 .* obs.target, obs.scale)
end

function cb_row(csv; grid, opts, T, point, P, forcing, gscale, task, solver=nothing,
                parameter=nothing, sign=nothing, rep, wall, steps=nothing, converged=nothing,
                dist=nothing, mach=nothing, t_lin=nothing, t_copy=nothing, t_solve=nothing,
                t_grad=nothing, iters=nothing, nprod=nothing, relres=nothing, aconv=nothing,
                fd=nothing, adj=nothing, rel=nothing)
    gc_csvrow(csv, (grid, opts["backend"], string(T), point, length(P.theta), P.a, forcing, gscale,
                    task, solver, parameter, sign, rep, rep == 0, wall, steps, converged, dist, mach,
                    t_lin, t_copy, t_solve, t_grad, iters, nprod, relres, aconv, fd, adj, rel))
end

"""
    cb_bench_one(csv, summary, case, P, point, fi, opts; backend, T, solvers, reps, h)

All timings for one (grid, point, forcing): forward from rest, adjoint gradient
per solver, 2p central-difference forwards warm-started from the base state.
`reps` lists the repeat indices (0 = warm-up, compiled, excluded from medians).
"""
function cb_bench_one(csv, summary, case, P, point, fi, opts; backend, T, solvers, reps, h)
    gscale = CB_GSCALES[fi]; g = gc_force(case, gscale)
    grid = "$(case.Nx)x$(case.Ny)"
    closure_of = P.closure(case)
    θ = P.theta; p = length(θ)
    meta = (; grid, opts, T, point, P, forcing=fi, gscale)
    t_rest = Float64[]; steps_rest = 0
    t_adj = Dict(s => Float64[] for s in solvers); t_sol = Dict(s => Float64[] for s in solvers)
    it_adj = Dict(s => 0 for s in solvers); np_adj = Dict(s => 0 for s in solvers)
    t_fd = Float64[]; steps_fd = 0
    grad_adj = Dict{Symbol,Vector{Float64}}()
    maxrel = 0.0
    for rep in reps
        base, t = gc_timed(() -> gc_forward(case, closure_of(θ), g; backend, T), backend)
        (base.converged && base.finite) || error("forward from rest did not converge ($grid, $point, forcing $fi)")
        mach = sqrt(3) * gc_umax(case, base.ux, base.uy)
        cb_row(csv; meta..., task="forward_rest", rep, wall=t, steps=base.steps, converged=base.converged,
               dist=base.dist_est, mach)
        rep > 0 && (push!(t_rest, t); steps_rest = base.steps)
        obs = cb_obs(case, base.ux, base.uy)
        for s in solvers
            out, t = gc_timed(() -> gc_gradient(base, θ, obs, closure_of, g, case; backend, T, solver=s),
                              backend)
            cb_row(csv; meta..., task="adjoint", solver=string(s), rep, wall=t, t_lin=out.t_lin,
                   t_copy=out.t_copy, t_solve=out.t_solve, t_grad=out.t_grad, iters=out.info.iters,
                   nprod=gc_nvjp(out.info), relres=out.info.relres, aconv=out.info.converged)
            if rep > 0
                push!(t_adj[s], t); push!(t_sol[s], out.t_solve)
                it_adj[s] = out.info.iters; np_adj[s] = gc_nvjp(out.info)
            end
            grad_adj[s] = collect(Float64, out.grad)
        end
        tsum = 0.0; ssum = 0; Jpm = zeros(p, 2)
        for k in 1:p, (j, sg) in enumerate((+1, -1))
            θp = copy(θ); δ = h * abs(θ[k]); θp[k] += sg * δ
            r, t = gc_timed(backend) do
                gc_forward(case, closure_of(θp), g; f_init=base.f, tau_init=base.tau, backend, T)
            end
            (r.converged && r.finite) || error("finite-difference forward did not converge")
            Jpm[k, j] = gnf_obs_objective(obs, r.ux, r.uy)
            cb_row(csv; meta..., task="fd_forward", parameter=P.names[k], sign=sg > 0 ? "+" : "-", rep,
                   wall=t, steps=r.steps, converged=r.converged, dist=r.dist_est)
            tsum += t; ssum += r.steps
        end
        rep > 0 && (push!(t_fd, tsum); steps_fd = ssum)
        fdg = [(Jpm[k, 1] - Jpm[k, 2]) / (2h * abs(θ[k])) for k in 1:p]
        for s in solvers, k in 1:p
            ga = grad_adj[s][k]
            rel = abs(fdg[k] - ga) / max(abs(ga), eps())
            rep > 0 && (maxrel = max(maxrel, rel))
            cb_row(csv; meta..., task="fd_vs_adjoint", solver=string(s), parameter=P.names[k], rep,
                   wall=nothing, fd=fdg[k], adj=ga, rel)
        end
    end
    nrep = count(>(0), reps)
    fr = gc_median(t_rest); ff = gc_median(t_fd)
    for s in solvers
        ta = gc_median(t_adj[s])
        gc_csvrow(summary, (grid, opts["backend"], string(T), point, p, fi, gscale, string(s), nrep, fr,
                            steps_rest, ta, gc_median(t_sol[s]), it_adj[s], np_adj[s], 2p, ff, steps_fd,
                            ta / ff, ta / fr, maxrel))
        @printf("  %-8s %-11s p=%d forcing %d  forward(rest) %.4f s [%d steps]  adjoint(%s) %.4f s [solve %.4f s, %d it, %d products]  2p=%d FD forwards %.4f s [%d steps]  adjoint/FD %.3f  adjoint/forward %.3f  max|FD-adj|/|adj| %.1e\n",
                grid, point, p, fi, fr, steps_rest, s, ta, gc_median(t_sol[s]), it_adj[s], np_adj[s], 2p,
                ff, steps_fd, ta / ff, ta / fr, maxrel)
    end
    return nothing
end

function cb_run(opts, csv, summary; backend)
    T = gc_precision(opts["precision"])
    ms = [gc_parse_grid(gs) for gs in gc_parse_list(opts["grids"])]
    solvers = [gc_solver(s) for s in gc_parse_list(opts["solvers"])]
    points = gc_parse_list(opts["points"])
    all(haskey(CB_POINTS, pt) for pt in points) || error("--points must be among interior, calibration")
    forcings = gc_parse_ints(opts["forcings"])
    all(in(1:length(CB_GSCALES)), forcings) || error("--forcings must be in 1:$(length(CB_GSCALES))")
    R = parse(Int, opts["repeats"]); R >= 1 || error("--repeats must be ≥ 1")
    h = parse(Float64, opts["fd-step"])
    println("Forward: dist_est < ", GC_FWD_TOL, " (from rest; FD forwards warm-started from the base state)")
    println("Adjoint: tol ", GC_ADJ_TOL, ", solvers ", join(solvers, ", "),
            "; FD step delta = ", h, " |theta_p|; repeats ", R,
            " (+1 warm-up on the first grid, first forcing, excluded from medians)")
    for pt in points
        P = CB_POINTS[pt]
        println("Point ", pt, ": theta (", join(P.names, ", "), ") = ", P.theta)
    end
    println("Timings (medians over repeats):")
    for (gi, m) in enumerate(ms)
        case = gc_case(m)
        for pt in points, (j, fi) in enumerate(forcings)
            reps = (gi == 1 && j == 1) ? (0:R) : (1:R)
            cb_bench_one(csv, summary, case, CB_POINTS[pt], pt, fi, opts; backend, T, solvers, reps, h)
        end
    end
    return true
end

function main(args)
    opts = try
        o = gc_options(args, CB_DEFAULTS)
        o["allow-unknown-commit"] in ("yes", "no") || error("--allow-unknown-commit must be yes or no")
        foreach(gc_parse_grid, gc_parse_list(o["grids"])); gc_precision(o["precision"])
        foreach(gc_solver, gc_parse_list(o["solvers"])); parse(Int, o["repeats"]); parse(Float64, o["fd-step"])
        o
    catch err
        println(stderr, "error: ", sprint(showerror, err)); return 2
    end
    backend, prov = try
        gc_backend(opts["backend"]),
        gc_provenance(; allow_unknown_commit=opts["allow-unknown-commit"] == "yes")
    catch err
        println(stderr, "error: ", sprint(showerror, err)); return 1
    end
    mkpath(opts["out"])
    stem = joinpath(opts["out"], "gnf_cost_bench_$(opts["backend"])_$(gc_stamp())")
    ok = false
    t0 = time()
    gc_transcript(stem * ".log") do
        open(stem * ".csv", "w") do csv
            open(stem * "_summary.csv", "w") do summary
                gc_csvrow(csv, CB_HEADER); gc_csvrow(summary, CB_SUMMARY_HEADER)
                try
                    gc_print_provenance(prov, "benchmarks/validation/gnf_calibration/cost_bench.jl";
                                        backend_name=opts["backend"], T=gc_precision(opts["precision"]),
                                        extra=["arguments" => join(args, ' '),
                                               "options" => join(["$k=$(opts[k])" for k in sort(collect(keys(opts)))], ' ')])
                    opts["backend"] == "cpu" &&
                        println("Note: CPU backend; timings depend on the thread counts above.")
                    ok = cb_run(opts, csv, summary; backend)
                catch err
                    showerror(stderr, err, catch_backtrace()); println(stderr)
                    ok = false
                finally
                    println("end_local=", Libc.strftime("%Y-%m-%dT%H:%M:%S%z", time()))
                    @printf("total_wall_seconds=%.3f\n", time() - t0)
                    println("Result: ", ok ? "completed" : "run failed (exit status 1)")
                    println("CSV=", gc_relpath(stem * ".csv"), " SUMMARY=", gc_relpath(stem * "_summary.csv"),
                            " LOG=", gc_relpath(stem * ".log"))
                end
            end
        end
    end
    return ok ? 0 : 1
end

# Load CUDA at top level, before `main` runs, so its bindings are visible in
# main's world age (an `@eval using CUDA` inside a function is not).
const CB_IS_SCRIPT = abspath(PROGRAM_FILE) == (@__FILE__)
CB_IS_SCRIPT && gc_wants_cuda(ARGS, CB_DEFAULTS["backend"]) && @eval using CUDA
CB_IS_SCRIPT && exit(main(ARGS))
