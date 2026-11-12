# Learn the second normal stress coefficient of the viscoelastic duct slice
# from velocities only, with the discrete adjoint of the steady FV Newton
# solver (case of slice_forward.jl: square duct H = 1, U = 1, eta_s = eta_p0 = 1,
# Giesekus alpha = 0.4, lambda = De, one periodic slice Nx = 1, N x N cells).
#
# Model: tau = 2 eta(gdot) D + 4 Psi2(gdot) D.D (FVReinerRivlinViscosity), eta
# the exact Giesekus viscosity (fixed: the axial flow pins it), Psi2 =
# -eta_p0 lambda g(Wi), Wi = lambda gdot, g a TinyMLP{1,H,L} of
# log(Wi + wi0) (NeuralPsi2Law). Start g = 0 (no secondary flow). eta_p0 and
# lambda are known; only the network weights are learned. One law g for all
# training De.
#
# Data: RheoTool Giesekus velocities (graded 200 x 200 mesh) interpolated on
# the N x N cell centres (truth_N<N>_De<de>.csv: j, k, y, z, u, v, w; written
# by prep_truth.py of the research folder). Objective, mean over training De:
#     J_De = 1/2 [ wax |u - u_T|^2 / |u_T|^2 + |(v, w) - (v, w)_T|^2 / |(v, w)_T|^2 ]
# (sums over the cells). Gradient: fv_steady_adjoint (one refactorization and
# one transposed solve per De, parameter duals through the residual kernel).
# Optimiser: BFGS, Armijo backtracking, step capped at VE_STEP in every weight;
# each De keeps its own warm-start state; a trial whose Newton solve fails is
# rejected (backtrack).
#
# Environment:
#   VE_BACKEND  cuda (default) | cpu
#   VE_N        cells per side (default 100)
#   VE_TRAIN    training De, comma separated (default 0.6)
#   VE_VAL      validation De (default 0.3,1.0), solved with the final law by
#               continuation from the nearest solved De (steps of 0.1 at most)
#   VE_TRUTH    folder of the truth CSVs (required)
#   VE_H, VE_L  network width and depth (8, 1);  VE_SEED (1);  VE_WI0 (1e-2)
#   VE_SCALE    input scale of log(Wi + wi0) (2.0);  VE_WAX axial weight (1.0)
#   VE_MAXITER  BFGS iterations (60);  VE_STEP max step per weight (0.5)
#   VE_LAW_CSV  fitted g table for the comparison (default ve2_law_fitted_g.csv)
#   VE_OUT      output folder (default benchmarks/ve_duct/results_learn)
#
# Output (VE_OUT): learn_history.csv (one row per accepted iteration),
# learn_law.csv (g learned / exact Giesekus / fitted on log10 Wi),
# learn_fields_De<de>.csv (model and truth fields), learn_summary.toml.
# Exit code 1 if the loss did not decrease or a validation solve failed.
#
#   VE_BACKEND=cuda VE_TRUTH=<dir> julia --project=<env> benchmarks/ve_duct/slice_learn.jl
#   VE_BACKEND=cpu VE_N=24 VE_MAXITER=3 VE_TRUTH=<dir> julia --project benchmarks/ve_duct/slice_learn.jl
if lowercase(get(ENV, "VE_BACKEND", "cuda")) == "cuda"
    using CUDA
    CUDA.functional() || error("CUDA not functional")
    using CUDSS
end
using KrakeNN, LinearAlgebra, Printf, Random, TOML

const BACKEND = lowercase(get(ENV, "VE_BACKEND", "cuda"))
BACKEND in ("cpu", "cuda") || error("VE_BACKEND must be cpu or cuda, got $BACKEND")
const GPU = BACKEND == "cuda"
to_device() = GPU ? CUDA.CuArray : Array
linsolve() = GPU ? CUDABackendTag() : CPUBackendTag()

envf(k, d) = parse(Float64, get(ENV, k, string(d)))
envi(k, d) = parse(Int, get(ENV, k, string(d)))
envlist(k, d) = parse.(Float64, split(get(ENV, k, d), ','; keepempty = false))
const N = envi("VE_N", 100)
const TRAIN = envlist("VE_TRAIN", "0.6")
const VAL = envlist("VE_VAL", "0.3,1.0")
const TRUTH = get(ENV, "VE_TRUTH", "")
isdir(TRUTH) || error("VE_TRUTH must be the folder of the truth CSVs")
const H = envi("VE_H", 8)
const L = envi("VE_L", 1)
const SEED = envi("VE_SEED", 1)
const WI0 = envf("VE_WI0", 1e-2)
const SCALE = envf("VE_SCALE", 2.0)
const WAX = envf("VE_WAX", 1.0)
const MAXITER = envi("VE_MAXITER", 60)
const STEP = envf("VE_STEP", 0.5)
const ALPHA = 0.4
const ETA_S = 1.0
const ETA_P = 1.0
const EPS = 1e-4
const NEWTON = (; atol = 1e-11, rtol = 1e-13, maxit = 60)
const OUT = get(ENV, "VE_OUT", joinpath(@__DIR__, "results_learn"))
const LAW_CSV = get(ENV, "VE_LAW_CSV", joinpath(@__DIR__, "ve2_law_fitted_g.csv"))
const LAW = NeuralPsi2Law(TinyMLP{1,H,L}(; shift = (0.0,), scale = (SCALE,)); wi0 = WI0)
const NPSI = law_nparams(LAW)
const OFF = 4 + 2                       # eta (4) and the fixed (eta_p0, lambda) of Psi2

detag(de) = @sprintf("De%.1f", de)
θfull(de, w) = [ETA_S, ETA_P, de, ALPHA, ETA_P, de, w...]

function read_truth(de)
    path = joinpath(TRUTH, "truth_N$(N)_$(detag(de)).csv")
    lines = readlines(path)
    strip(lines[1]) == "j,k,y,z,u,v,w" || error("$path: unexpected header")
    u = zeros(N, N); v = zeros(N, N); w = zeros(N, N)
    for l in lines[2:end]
        isempty(strip(l)) && continue
        r = split(l, ',')
        j = parse(Int, r[1]); k = parse(Int, r[2])
        u[j, k] = parse(Float64, r[5]); v[j, k] = parse(Float64, r[6]); w[j, k] = parse(Float64, r[7])
    end
    return (; u, v, w)
end

function misfit_vectors(p, T)
    g = p.grid; n = fv_nunknowns(p)
    W = zeros(n); D = zeros(n)
    sa = sum(abs2, T.u); ss = sum(abs2, T.v) + sum(abs2, T.w)
    for k in 1:N, j in 1:N
        a = fv_index(g, 1, j, k, 1)
        W[a] = WAX / sa; D[a] = T.u[j, k]
        W[a + 1] = 1 / ss; D[a + 1] = T.v[j, k]
        W[a + 2] = 1 / ss; D[a + 2] = T.w[j, k]
    end
    return to_device()(W), to_device()(D)
end

function metrics(hp, U, T)
    u, v, w, _ = fv_fields(hp, Vector{Float64}(Array(U)))
    u = u[1, :, :]; v = v[1, :, :]; w = w[1, :, :]
    sec = sqrt((sum(abs2, v .- T.v) + sum(abs2, w .- T.w)) / (sum(abs2, T.v) + sum(abs2, T.w)))
    ax = sqrt(sum(abs2, u .- T.u) / sum(abs2, T.u))
    um = sum(u) / length(u); umT = sum(T.u) / length(T.u)
    mx = maximum(hypot.(v, w)) / um; mxT = maximum(hypot.(T.v, T.w)) / umT
    # wall shear rate (axial gradient at the wall faces): Wi_max = De gdot_max
    gw = maximum(2 * N .* vcat(u[1, :], u[end, :], u[:, 1], u[:, end]))
    return (; sec, ax, mx, mxT, gwall = gw, u, v, w)
end

function fitted_g()
    lines = filter(!isempty, strip.(readlines(LAW_CSV)))
    head = strip.(split(lines[1], ','))
    ix = findfirst(==("log10Wi"), head); ic = findfirst(==("jointD1_r0.0001_c"), head)
    rows = [parse.(Float64, split(l, ',')) for l in lines[2:end]]
    law = TabulatedGiesekusPsi2Law([r[ix] for r in rows])
    c = [r[ic] for r in rows]
    return wi -> -law_value(law, (wi,), [1.0; 1.0; ALPHA; c])      # eta_p0 = lambda = 1
end

function main()
    mkpath(OUT)
    println("== slice_learn  backend=$BACKEND  N=$N  train De=$(join(TRAIN, ","))  val De=$(join(VAL, ","))  ",
            "TinyMLP{1,$H,$L} ($(NPSI - 2) weights)  wi0=$WI0  scale=$SCALE  wax=$WAX  KrakeNN ",
            pkgversion(KrakeNN))
    GPU && println("   GPU ", CUDA.name(CUDA.device()), "  CUDA.jl ", pkgversion(CUDA), "  CUDSS.jl ", pkgversion(CUDSS))
    flush(stdout)
    t0 = time()
    w0 = law_init(LAW; rng = MersenneTwister(SEED))[3:end]
    eta = FVLawViscosity(GiesekusViscosityLaw(), [ETA_S, ETA_P, TRAIN[1], ALPHA]; eps = EPS)
    hp = fv_steady_problem(1, N, N; h = 1 / N,
                           bc = (W = :periodic, E = :periodic, S = :wall, N = :wall, B = :wall, T = :wall),
                           viscosity = FVReinerRivlinViscosity(eta, FVLawViscosity(LAW, [ETA_P; TRAIN[1]; w0];
                                                                                   eps = EPS)),
                           source = (x, y, z) -> (28.45 * (ETA_S + ETA_P), 0.0, 0.0, 0.0), u_mean_target = 1.0)
    map = FVReinerRivlinParameters(4, NPSI)
    s = fv_newton_solver(hp; to = to_device(), linsolve = linsolve())
    ws = fv_adjoint_workspace(s, map; chunk = 8)
    alld = sort(unique([TRAIN; VAL]))
    truth = Dict(de => read_truth(de) for de in alld)
    WD = Dict(de => misfit_vectors(hp, truth[de]) for de in alld)
    Ugood = Dict{Float64,Any}()
    U0 = to_device()(KrakeNN._fv_initial_force!(zeros(fv_nunknowns(hp)), hp))
    for de in TRAIN
        Ugood[de] = copy(U0)
    end
    ntr = length(TRAIN)
    counters = Dict("fevals" => 0, "newton_iters" => 0, "newton_fail" => 0, "t_newton" => 0.0, "t_adjoint" => 0.0)

    # value and gradient at w; returns (J, grad, states) or (Inf, nothing, nothing)
    # (locals declared: a closure would otherwise share J, states, ... with main)
    function fg(w)
        local Jv, grad, sts, θ, U, st, tn, ta, cot, ok, W, D, r, Jd
        counters["fevals"] += 1
        Jv = 0.0; grad = zeros(length(w)); sts = Dict{Float64,Any}()
        for de in TRAIN
            θ = θfull(de, w)
            fv_set_params!(s, map, θ)
            U = copy(Ugood[de])
            tn = @elapsed begin
                st = fv_newton!(s, U; NEWTON...)
            end
            counters["t_newton"] += tn
            counters["newton_iters"] += st.iterations
            ok = (st.converged || (st.stagnated && last(st.inf) <= 1e3 * NEWTON.atol)) && all(isfinite, st.inf)
            if !ok
                counters["newton_fail"] += 1
                return Inf, nothing, nothing
            end
            W, D = WD[de]
            r = U .- D
            Jd = sum(W .* r .^ 2) / 2
            ta = @elapsed begin
                cot = first(fv_steady_adjoint(s, U, W .* r; params = map, θ, ws))
            end
            counters["t_adjoint"] += ta
            Jv += Jd / ntr
            grad .+= cot[(OFF + 1):end] ./ ntr
            sts[de] = U
        end
        return Jv, grad, sts
    end

    gwi = (0.1, 0.3, 1.0, 3.0)
    gfit = fitted_g()
    gl(w, wi) = neural_psi2_g(LAW, wi, [ETA_P; 1.0; w])
    hist = open(joinpath(OUT, "learn_history.csv"), "w")
    println(hist, "iter,J,grad_inf,fevals,newton_iters,seconds," *
                  join(["sec_relL2_$(detag(de)),ax_relL2_$(detag(de)),max_uxy_$(detag(de))" for de in TRAIN], ",") *
                  "," * join(["g_Wi$(wi)" for wi in gwi], ","))
    function report(it, w, J, g, states)
        local row, parts, m
        row = Any[it, J, maximum(abs, g), counters["fevals"], counters["newton_iters"], time() - t0]
        parts = String[]
        for de in TRAIN
            m = metrics(hp, states[de], truth[de])
            append!(row, [m.sec, m.ax, m.mx])
            push!(parts, @sprintf("%s sec %.2f%% ax %.3f%% max %.3e/%.3e", detag(de), 100m.sec, 100m.ax, m.mx, m.mxT))
        end
        append!(row, [gl(w, wi) for wi in gwi])
        println(hist, join(string.(row), ",")); flush(hist)
        @printf("it %3d  J %.6e  |g| %.2e  fev %d  %s  g(0.1,0.3,1,3) = %s  (%.0f s)\n", it, J, maximum(abs, g),
                counters["fevals"], join(parts, "  "), join([@sprintf("%.4f", gl(w, wi)) for wi in gwi], ","),
                time() - t0)
        flush(stdout)
    end

    # ---------------------------------------------------------------- BFGS
    w = copy(w0)
    J, g, states = fg(w)
    isfinite(J) || error("the g = 0 start did not solve")
    J0 = J
    for de in TRAIN; Ugood[de] = states[de]; end
    report(0, w, J, g, states)
    Hinv = Matrix{Float64}(I, length(w), length(w))
    stop = "maxiter"; it = 0; resets = 0
    for k in 1:MAXITER
        it = k
        d = -Hinv * g
        slope = dot(g, d)
        if !(slope < 0)
            Hinv = Matrix{Float64}(I, length(w), length(w)); d = -g; slope = dot(g, d)
        end
        sc = maximum(abs, d) / STEP
        sc > 1 && (d ./= sc; slope /= sc)
        t = 1.0; acc = false; Jn = Inf; gn = g; stn = states
        for _ in 1:14
            Jn, gn, stn = fg(w .+ t .* d)
            if isfinite(Jn) && Jn <= J + 1e-4 * t * slope
                acc = true; break
            end
            t /= 2
        end
        if !acc
            if resets == 0 && Hinv != I
                Hinv = Matrix{Float64}(I, length(w), length(w)); resets += 1
                println("   line search failed: BFGS memory reset"); flush(stdout)
                continue
            end
            stop = "linesearch"; it = k - 1; break
        end
        resets = 0
        sv = t .* d; yv = gn .- g
        w = w .+ sv
        sy = dot(sv, yv)
        if sy > 1e-12 * norm(sv) * norm(yv)
            k == 1 && (Hinv .*= sy / dot(yv, yv))
            ρ = 1 / sy
            V = I - ρ * yv * sv'
            Hinv = V' * Hinv * V + ρ * sv * sv'
        end
        dJ = J - Jn
        J, g, states = Jn, gn, stn
        for de in TRAIN; Ugood[de] = states[de]; end
        report(k, w, J, g, states)
        if maximum(abs, g) < 1e-8
            stop = "gradient"; break
        end
        if dJ < 1e-10 * J
            stop = "stalled"; break
        end
    end
    close(hist)
    @printf("SUMMARY training: J %.6e -> %.6e (ratio %.3e)  iterations %d  stop %s  fevals %d  newton iters %d (fails %d)  t_newton %.0f s  t_adjoint %.0f s\n",
            J0, J, J / J0, it, stop, counters["fevals"], counters["newton_iters"], counters["newton_fail"],
            counters["t_newton"], counters["t_adjoint"])
    flush(stdout)

    # ---------------------------------------------------------------- evaluation
    failures = String[]
    J < 0.5 * J0 || push!(failures, "loss did not decrease by half ($(J / J0))")
    solved = Dict{Float64,Any}(de => Ugood[de] for de in TRAIN)
    summary = Dict{String,Any}("backend" => BACKEND, "N" => N, "train" => TRAIN, "val" => VAL, "H" => H, "L" => L,
                               "n_weights" => NPSI - 2, "seed" => SEED, "wi0" => WI0, "scale" => SCALE, "wax" => WAX,
                               "J0" => J0, "J" => J, "iterations" => it, "stop" => stop, "weights" => w,
                               "counters" => counters, "wall_seconds" => time() - t0)
    for de in VAL
        de in keys(solved) && continue
        near = argmin(x -> abs(x - de), collect(keys(solved)))
        nst = max(1, ceil(Int, abs(de - near) / 0.1 - 1e-9))
        U = copy(solved[near]); ok = true
        for q in 1:nst
            dq = near + (de - near) * q / nst
            fv_set_params!(s, map, θfull(dq, w))
            st = fv_newton!(s, U; NEWTON...)
            ok = st.converged || (st.stagnated && last(st.inf) <= 1e3 * NEWTON.atol)
            ok || break
        end
        ok ? (solved[de] = U) : push!(failures, "validation De=$de did not converge")
    end
    for de in sort(collect(keys(solved)))
        m = metrics(hp, solved[de], truth[de])
        role = de in TRAIN ? "train" : "val"
        summary[detag(de)] = Dict{String,Any}("role" => role, "sec_relL2" => m.sec, "ax_relL2" => m.ax,
                                              "max_uxy_over_U" => m.mx, "truth_max_uxy_over_U" => m.mxT,
                                              "max_ratio" => m.mx / m.mxT, "Wi_wall_max" => de * m.gwall)
        @printf("SUMMARY %-5s %s  max|u_xy|/U %.4e  truth %.4e (%+.1f %%)  sec relL2 %.2f %%  axial relL2 %.3f %%  Wi_wall_max %.2f\n",
                role, detag(de), m.mx, m.mxT, 100 * (m.mx / m.mxT - 1), 100m.sec, 100m.ax, de * m.gwall)
        open(joinpath(OUT, "learn_fields_$(detag(de)).csv"), "w") do io
            println(io, "j,k,y,z,u,v,w,u_truth,v_truth,w_truth")
            T = truth[de]
            for k in 1:N, j in 1:N
                @printf(io, "%d,%d,%.6g,%.6g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g\n", j, k, (j - 0.5) / N, (k - 0.5) / N,
                        m.u[j, k], m.v[j, k], m.w[j, k], T.u[j, k], T.v[j, k], T.w[j, k])
            end
        end
    end
    open(joinpath(OUT, "learn_law.csv"), "w") do io
        println(io, "log10Wi,g_learned,g_exact,g_fitted,g_start")
        for lw in range(-2.0, 1.2; length = 65)
            wi = exp10(lw)
            @printf(io, "%.4f,%.8g,%.8g,%.8g,%.8g\n", lw, gl(w, wi), KrakeNN._giesekus_g(wi, ALPHA), gfit(wi), gl(w0, wi))
        end
    end
    for wi in (0.1, 0.3, 1.0, 3.0, 10.0)
        @printf("SUMMARY law Wi=%-5g g learned %.4f  fitted %.4f  Giesekus %.4f  learned/fitted %.3f\n", wi, gl(w, wi),
                gfit(wi), KrakeNN._giesekus_g(wi, ALPHA), gl(w, wi) / gfit(wi))
    end
    summary["failures"] = failures
    open(io -> TOML.print(io, summary), joinpath(OUT, "learn_summary.toml"), "w")
    isempty(failures) ? println("PASS slice_learn") : println("FAIL slice_learn: ", join(failures, "; "))
    return isempty(failures)
end

main() || exit(1)
