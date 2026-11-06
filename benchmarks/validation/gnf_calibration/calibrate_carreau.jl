#!/usr/bin/env julia
# Carreau calibration from synthetic steady velocity data with the adjoint
# gradient of the steady GNF solver (`gnf_steady_gradient`), on the periodic
# obstacle channel of the adjoint check, for noise-free and noisy data (see
# README.md in this directory).
#
#   julia --project=. benchmarks/validation/gnf_calibration/calibrate_carreau.jl \
#         --backend=cuda --grid=32x16 [--mode=all|noisefree|noise|smoke] [options]
#
# The constants below (parameters, true values, forcings, starts, noise levels
# and seeds, stopping rule) are fixed before any campaign run; change them only
# with a new commit, never through options.

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "bfgs.jl"))

# ------------------------------------------------ fixed before any run ---

# Calibrated parameters: θ = (ln η∞*, ln k*, n) of the code's Carreau–Yasuda law
# `CarreauYasuda(eta_0, eta_inf, lambda, a, n)` with a = 2 (Carreau) and
# η0 = ν0 fixed; η∞* = η∞/η0, k* = λ ν0/m². This is the parameter map of the
# adjoint check (`gc_closure4`) with its fourth component a held at 2.
const CAL_A = 2.0
const CAL_NAMES = ("ln_eta_inf", "ln_k", "n")
# True values: the reference point of the adjoint check (η∞* = 0.02, k* = 5,
# n = 0.7), where the forward and the adjoint gradient were checked.
const CAL_THETA_TRUE = [log(0.02), log(5.0), 0.7]
# S = 3 forcings: scale factors of the reference reduced force G* = 5. The
# largest is the force of the adjoint check (Mach 0.108 on 32×16); the two
# smaller ones give flows at lower shear rates (ν/ν0 down to 0.59, 0.46 and
# 0.39 on 32×16 for the three forcings, measured 2026-11-04).
const CAL_GSCALES = GC_GSCALES          # (0.1, 0.2, 0.3), defined in common.jl
# Five starts: factors on (η∞*, k*, n) of the true values, each component 20 to
# 50 % away. Mach at the largest forcing on 32×16 at these starts: 0.080, 0.147,
# 0.067, 0.071, 0.129 (measured 2026-11-04).
const CAL_STARTS = ((1.2, 1.2, 1.2),
                    (0.8, 0.8, 0.8),
                    (1.5, 0.6, 1.3),
                    (0.5, 1.5, 1.3),
                    (1.4, 0.5, 0.7))
# Noise: Gaussian, standard deviation σ_s = level × u_ref,s on each velocity
# component at each fluid node, u_ref,s = maximum fluid speed of the noise-free
# data of forcing s. One standard-normal vector per seed (forcings in order,
# then ux then uy at the fluid nodes in linear-index order), scaled by each level,
# so the three levels share the same ten draws. Noisy runs start from start 1.
const CAL_NOISE_LEVELS = (0.001, 0.01, 0.05)
const CAL_NOISE_SEEDS = (2026110401, 2026110402, 2026110403, 2026110404, 2026110405,
                         2026110406, 2026110407, 2026110408, 2026110409, 2026110410)
const CAL_NOISE_START = 1
# Stopping rule (gc_bfgs): noise-free runs stop as soon as the RMS of
# (u − u_data)/u_ref,s over all observations is ≤ 1e-6; every run
# also stops when ‖∇J‖∞ has fallen by 1e8 from the start, when an accepted step
# is ≤ 1e-9 in every component, when J decreases by ≤ 1e-12 J over an
# iteration, when the line search finds no decrease after 30 reductions, or
# after 100 iterations. A step is limited to 0.25 in ln η∞*, 0.25 in ln k* and
# 0.1 in n. Points with n outside (0.05, 2) or with a forward that does not
# converge count as failed trials (J = Inf).
const CAL_RMS_TARGET = 1e-6
const CAL_GTOL_REL = 1e-8
const CAL_XTOL = 1e-9
const CAL_FTOL = 1e-12
const CAL_MAXITER = 100
const CAL_MAX_STEP = [0.25, 0.25, 0.1]
const CAL_N_RANGE = (0.05, 2.0)

const CAL_DEFAULTS = Dict("backend" => "cuda", "grid" => "32x16", "precision" => "Float64",
                          "solver" => "gmres", "mode" => "all", "starts" => "1:5",
                          "levels" => "1:3", "draws" => "1:10", "maxiter" => "",
                          "out" => "benchmarks/validation/gnf_calibration/results",
                          "allow-unknown-commit" => "no")

# ------------------------------------------------------------ objective ---

"""
Objective of one calibration: J(θ) = Σ_s ½ Σ_m ((u_m(θ) − d_m)/u_ref,s)² over the S
forcings. `fval` runs the S forwards (warm-started from the current iterate) and
`fgrad` the S adjoints at the states of the last `fval`; counters and wall times
are kept for the CSV.
"""
mutable struct CalObjective
    case::Any
    closure_of::Any
    gs::Vector{Float64}
    obs::Vector{GNFVelocityObs}
    backend::Any
    T::Type
    solver::Symbol
    base::Vector{Any}
    last_x::Vector{Float64}
    last::Vector{Any}
    last_ok::Bool
    n_eval::Int
    n_fwd::Int
    n_fwd_fail::Int
    n_domain::Int
    fwd_steps::Int
    t_fwd::Float64
    n_grad::Int
    n_adj::Int
    n_adj_unconv::Int
    adj_iters::Int
    adj_nvjp::Int
    adj_maxrel::Float64
    t_adj::Float64
end

function CalObjective(case, gs, obs; backend, T, solver)
    S = length(gs)
    return CalObjective(case, gc_closure3(case; a=CAL_A), collect(Float64, gs), obs, backend, T,
                        solver, Any[nothing for _ in 1:S], Float64[], Any[], false,
                        0, 0, 0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0.0, 0.0)
end

function cal_fval!(obj::CalObjective, x)
    obj.n_eval += 1
    obj.last_ok = false
    if !(all(isfinite, x) && CAL_N_RANGE[1] < x[3] < CAL_N_RANGE[2])
        obj.n_domain += 1
        return Inf
    end
    cl = obj.closure_of(x)
    states = Any[]
    J = 0.0
    for (s, g) in enumerate(obj.gs)
        init = obj.base[s]
        r, t = gc_timed(obj.backend) do
            gc_forward(obj.case, cl, g; f_init=init === nothing ? nothing : init.f,
                       tau_init=init === nothing ? nothing : init.tau,
                       backend=obj.backend, T=obj.T)
        end
        obj.n_fwd += 1; obj.t_fwd += t; obj.fwd_steps += r.steps
        if !(r.converged && r.finite)
            obj.n_fwd_fail += 1
            return Inf
        end
        push!(states, r)
        J += gnf_obs_objective(obj.obs[s], r.ux, r.uy)
    end
    obj.last_x = copy(x); obj.last = states; obj.last_ok = true
    return J
end

function cal_fgrad!(obj::CalObjective, x)
    (obj.last_ok && obj.last_x == x) ||
        error("cal_fgrad!: gradient requested away from the last evaluated point")
    grad = zeros(length(x))
    for (s, g) in enumerate(obj.gs)
        out, t = gc_timed(obj.backend) do
            gc_gradient(obj.last[s], x, obj.obs[s], obj.closure_of, g, obj.case;
                        backend=obj.backend, T=obj.T, solver=obj.solver)
        end
        obj.n_adj += 1; obj.t_adj += t
        obj.adj_iters += out.info.iters; obj.adj_nvjp += gc_nvjp(out.info)
        obj.adj_maxrel = max(obj.adj_maxrel, out.info.relres)
        out.info.converged || (obj.n_adj_unconv += 1)
        grad .+= out.grad
    end
    obj.n_grad += 1
    obj.base = obj.last          # accepted point: warm start of the next forwards
    return grad
end

# ------------------------------------------------------------------ data ---

"Noise-free synthetic data at the true θ, one entry per forcing."
function cal_data(case; backend, T)
    cl = gc_closure3(case; a=CAL_A)(CAL_THETA_TRUE)
    data = NamedTuple[]
    for gscale in CAL_GSCALES
        g = gc_force(case, gscale)
        r = gc_forward(case, cl, g; backend=backend, T=T)
        (r.converged && r.finite) ||
            error("forward at the true parameters did not converge (G scale $gscale)")
        uref = gc_umax(case, r.ux, r.uy)
        nu = Array(r.nu)[case.nodes] ./ case.nu0
        push!(data, (; gscale, g, ux=copy(r.ux), uy=copy(r.uy), uref, steps=r.steps,
                     dist=r.dist_est, mach=sqrt(3) * uref, nu_min=minimum(nu), nu_max=maximum(nu),
                     zero_shear=gc_zero_shear(case, r.I1)))
    end
    return data
end

"Data plus Gaussian noise of standard deviation level × u_ref,s (see CAL_NOISE_SEEDS)."
function cal_noisy(case, data, level, seed)
    rng = MersenneTwister(seed)
    out = NamedTuple[]
    for d in data
        ξx = randn(rng, case.nfluid); ξy = randn(rng, case.nfluid)
        ux = copy(d.ux); uy = copy(d.uy)
        for (k, nd) in enumerate(case.nodes)
            ux[nd] += level * d.uref * ξx[k]
            uy[nd] += level * d.uref * ξy[k]
        end
        push!(out, (; ux, uy))
    end
    return out
end

cal_start(i) = [log(exp(CAL_THETA_TRUE[1]) * CAL_STARTS[i][1]),
                log(exp(CAL_THETA_TRUE[2]) * CAL_STARTS[i][2]),
                CAL_THETA_TRUE[3] * CAL_STARTS[i][3]]

"Physical parameters (η∞*, k*, n) of θ."
cal_physical(θ) = (exp(θ[1]), exp(θ[2]), θ[3])

# ------------------------------------------------------------ one run ---

const CAL_HIST_HEADER = ("run_id", "iter", "J", "rms", "grad_inf", "step_inf", "alpha",
                         "backtracks", "theta_ln_eta_inf", "theta_ln_k", "theta_n")

"""
    cal_run(case, data, fields, θ0; backend, T, solver, maxiter, noise_free, run_id, hist)

One BFGS calibration against the velocity fields `fields` (one `(; ux, uy)` per
forcing). Returns the optimiser result, the objective (counters) and the RMS
values.
"""
function cal_run(case, data, fields, θ0; backend, T, solver, maxiter, noise_free::Bool,
                 run_id, hist=nothing)
    obs = [gc_obs(case, f.ux, f.uy, d.uref) for (f, d) in zip(fields, data)]
    obj = CalObjective(case, [d.g for d in data], obs; backend, T, solver)
    M = 2 * case.nfluid * length(data)
    rms(J) = sqrt(2J / M)
    ftarget = noise_free ? 0.5 * M * CAL_RMS_TARGET^2 : -Inf
    callback = rec -> begin
        hist === nothing && return
        gc_csvrow(hist, (run_id, rec.iter, rec.f, rms(rec.f), rec.gnorm, rec.step, rec.alpha,
                         rec.backtracks, rec.x...))
    end
    t0 = time()
    res = gc_bfgs(x -> cal_fval!(obj, x), x -> cal_fgrad!(obj, x), θ0; maxiter=maxiter,
                  ftarget=ftarget, gtol_rel=CAL_GTOL_REL, xtol=CAL_XTOL, ftol=CAL_FTOL,
                  max_step=CAL_MAX_STEP, callback=callback)
    wall = time() - t0
    # RMS of the final velocity against the noise-free data (equal to `rms` without noise)
    acc = 0.0
    for (s, d) in enumerate(data)
        st = obj.base[s]
        for nd in case.nodes
            acc += ((st.ux[nd] - d.ux[nd])^2 + (st.uy[nd] - d.uy[nd])^2) / d.uref^2
        end
    end
    return (; res, obj, wall, M, J0=res.history[1].f, rms0=rms(res.history[1].f),
            rms=rms(res.f), rms_truth=sqrt(acc / M))
end

const CAL_CSV_HEADER = (
    "run_id", "kind", "start", "noise_level", "noise_seed", "grid", "backend", "precision",
    "solver", "theta0_ln_eta_inf", "theta0_ln_k", "theta0_n", "theta_ln_eta_inf", "theta_ln_k",
    "theta_n", "eta_inf_star", "k_star", "n", "relerr_eta_inf_star", "relerr_k_star", "relerr_n",
    "J0", "J", "rms0", "rms", "rms_target", "met_rms_target", "rms_vs_noise_free", "stop",
    "iterations", "objective_evals", "gradient_evals", "forward_solves", "failed_forwards",
    "out_of_domain", "adjoint_solves", "adjoint_unconverged", "adjoint_iters",
    "adjoint_operator_products", "adjoint_max_relres", "forward_steps", "skipped_updates",
    "direction_resets", "wall_total_s", "wall_forward_s", "wall_adjoint_s")

function cal_csvrow(csv, out, meta)
    θ = out.res.x; θ0 = out.res.history[1].x
    p = cal_physical(θ); pt = cal_physical(CAL_THETA_TRUE)
    relerr = ntuple(i -> (p[i] - pt[i]) / pt[i], 3)
    nf = meta.kind == "noise-free"
    o = out.obj
    gc_csvrow(csv, (meta.run_id, meta.kind, meta.start, meta.level, meta.seed, meta.grid,
                    meta.backend, string(meta.T), string(meta.solver), θ0..., θ..., p..., relerr...,
                    out.J0, out.res.f, out.rms0, out.rms, nf ? CAL_RMS_TARGET : nothing,
                    nf ? out.rms <= CAL_RMS_TARGET : nothing, out.rms_truth, string(out.res.stop),
                    out.res.iterations, out.res.n_fval, out.res.n_fgrad, o.n_fwd, o.n_fwd_fail,
                    o.n_domain, o.n_adj, o.n_adj_unconv, o.adj_iters, o.adj_nvjp, o.adj_maxrel,
                    o.fwd_steps, out.res.n_skip, out.res.n_reset, out.wall, o.t_fwd, o.t_adj))
    @printf("  %-12s stop=%-10s it=%3d  rms %.3e -> %.3e (vs noise-free %.3e)  eta_inf* %+.3e  k* %+.3e  n %+.3e  fwd %d adj %d  %.1f s (fwd %.1f, adj %.1f)\n",
            meta.run_id, out.res.stop, out.res.iterations, out.rms0, out.rms, out.rms_truth,
            relerr..., o.n_fwd, o.n_adj, out.wall, o.t_fwd, o.t_adj)
    return nothing
end

# ------------------------------------------------------------- campaign ---

"""
    cal_campaign(opts, csv, hist; backend) -> (; ok, verdict)

Runs the calibrations selected by `opts` (already validated), writing one CSV
row per run and one history row per accepted iterate. `ok` is true when every
noise-free run met the RMS target (mode `smoke`: when the misfit of its single
short run decreased). Noise runs carry no pass/fail criterion.
"""
function cal_campaign(opts, csv, hist; backend)
    m = gc_parse_grid(opts["grid"]); case = gc_case(m)
    T = gc_precision(opts["precision"]); solver = gc_solver(opts["solver"])
    mode = opts["mode"]
    smoke = mode == "smoke"
    maxiter = isempty(opts["maxiter"]) ? (smoke ? 3 : CAL_MAXITER) : parse(Int, opts["maxiter"])
    starts = smoke ? [1] : gc_parse_ints(opts["starts"])
    all(in(1:length(CAL_STARTS)), starts) || error("--starts must be in 1:$(length(CAL_STARTS))")
    levels = gc_parse_ints(opts["levels"]); draws = gc_parse_ints(opts["draws"])
    all(in(1:length(CAL_NOISE_LEVELS)), levels) || error("--levels must be in 1:$(length(CAL_NOISE_LEVELS))")
    all(in(1:length(CAL_NOISE_SEEDS)), draws) || error("--draws must be in 1:$(length(CAL_NOISE_SEEDS))")
    do_nf = mode in ("all", "noisefree", "smoke")
    do_noise = mode in ("all", "noise")

    println("Case: obstacle channel ", case.Nx, "x", case.Ny, " (m = ", m, "), fluid nodes ",
            case.nfluid, ", nu0 = ", case.nu0, ", a = ", CAL_A, " fixed")
    println("True theta (ln eta_inf*, ln k*, n) = ", CAL_THETA_TRUE, " i.e. (eta_inf*, k*, n) = ",
            cal_physical(CAL_THETA_TRUE))
    println("Solver: ", solver, " tol ", GC_ADJ_TOL, "; forward dist_est < ", GC_FWD_TOL,
            "; optimiser BFGS + Armijo, maxiter ", maxiter, ", max step ", CAL_MAX_STEP)
    data, tdata = gc_timed(() -> cal_data(case; backend, T), backend)
    @printf("Synthetic data (%.2f s):\n", tdata)
    for (s, d) in enumerate(data)
        @printf("  forcing %d: G scale %.2f (G* = %.2f, G_lat = %.6e)  steps %d  dist_est %.1e  u_ref %.6e  Mach %.4f  nu/nu0 in [%.4f, %.4f]  zero-shear nodes %d\n",
                s, d.gscale, d.gscale * GC_GSTAR_REF, d.g, d.steps, d.dist, d.uref, d.mach,
                d.nu_min, d.nu_max, d.zero_shear)
    end
    meta0 = (; grid="$(case.Nx)x$(case.Ny)", backend=opts["backend"], T, solver)
    ok = true
    verdicts = String[]
    if do_nf
        println("Noise-free runs (stop at RMS <= ", CAL_RMS_TARGET, "):")
        fields = [(; ux=d.ux, uy=d.uy) for d in data]
        for i in starts
            run_id = "nf-s$i"
            out = cal_run(case, data, fields, cal_start(i); backend, T, solver, maxiter,
                          noise_free=true, run_id, hist)
            cal_csvrow(csv, out, (; meta0..., run_id, kind="noise-free", start=i, level=0.0,
                                  seed=nothing))
            if smoke
                ok &= out.res.f < out.J0
                push!(verdicts, @sprintf("smoke: misfit %s (J %.6e -> %.6e after %d iterations)",
                                         out.res.f < out.J0 ? "decreased" : "did NOT decrease",
                                         out.J0, out.res.f, out.res.iterations))
            else
                ok &= out.rms <= CAL_RMS_TARGET
            end
        end
        smoke || push!(verdicts, "noise-free RMS target " * (ok ? "met by every noise-free run" : "NOT met by every noise-free run"))
    end
    if do_noise
        println("Noise runs (start ", CAL_NOISE_START, "):")
        for il in levels, id in draws
            level = CAL_NOISE_LEVELS[il]; seed = CAL_NOISE_SEEDS[id]
            run_id = @sprintf("n%g-d%d", level, id)
            fields = cal_noisy(case, data, level, seed)
            out = cal_run(case, data, fields, cal_start(CAL_NOISE_START); backend, T, solver,
                          maxiter, noise_free=false, run_id, hist)
            cal_csvrow(csv, out, (; meta0..., run_id, kind="noise", start=CAL_NOISE_START,
                                  level, seed))
        end
        push!(verdicts, "noise runs: reported without a pass/fail criterion")
    end
    return (; ok, verdict=join(verdicts, "; "))
end

function main(args)
    opts = try
        o = gc_options(args, CAL_DEFAULTS)
        o["mode"] in ("all", "noisefree", "noise", "smoke") ||
            error("--mode must be all, noisefree, noise or smoke")
        o["allow-unknown-commit"] in ("yes", "no") || error("--allow-unknown-commit must be yes or no")
        gc_parse_grid(o["grid"]); gc_precision(o["precision"]); gc_solver(o["solver"])
        isempty(o["maxiter"]) || parse(Int, o["maxiter"])
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
    m = gc_parse_grid(opts["grid"])
    stem = joinpath(opts["out"], "carreau_calibration_$(opts["backend"])_$(8m)x$(4m)_$(opts["mode"])_$(gc_stamp())")
    ok = false; verdict = "run failed"
    t0 = time()
    gc_transcript(stem * ".log") do
        open(stem * ".csv", "w") do csv
            open(stem * "_history.csv", "w") do hist
                gc_csvrow(csv, CAL_CSV_HEADER); gc_csvrow(hist, CAL_HIST_HEADER)
                try
                    gc_print_provenance(prov, "benchmarks/validation/gnf_calibration/calibrate_carreau.jl";
                                        backend_name=opts["backend"], T=gc_precision(opts["precision"]),
                                        extra=["arguments" => join(args, ' '),
                                               "options" => join(["$k=$(opts[k])" for k in sort(collect(keys(opts)))], ' ')])
                    ok, verdict = cal_campaign(opts, csv, hist; backend)
                catch err
                    showerror(stderr, err, catch_backtrace()); println(stderr)
                    ok = false; verdict = "run failed"
                finally
                    println("end_local=", Libc.strftime("%Y-%m-%dT%H:%M:%S%z", time()))
                    @printf("total_wall_seconds=%.3f\n", time() - t0)
                    println("Result: ", verdict, ok ? "" : " (exit status 1)")
                    println("CSV=", gc_relpath(stem * ".csv"), " HISTORY=", gc_relpath(stem * "_history.csv"),
                            " LOG=", gc_relpath(stem * ".log"))
                end
            end
        end
    end
    return ok ? 0 : 1
end

# Load CUDA at top level, before `main` runs, so its bindings are visible in
# main's world age (an `@eval using CUDA` inside a function is not).
const CAL_IS_SCRIPT = abspath(PROGRAM_FILE) == (@__FILE__)
CAL_IS_SCRIPT && gc_wants_cuda(ARGS, CAL_DEFAULTS["backend"]) && @eval using CUDA
CAL_IS_SCRIPT && exit(main(ARGS))
