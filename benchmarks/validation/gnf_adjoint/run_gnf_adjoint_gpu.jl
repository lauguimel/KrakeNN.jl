#!/usr/bin/env julia
# Frozen T5/G validation of the periodic obstacle-channel case of
# test/ad/gnf_adjoint_test.jl, plus the accuracy checks (accuracy
# budget, FD step sweep, solve tolerances, GPU against CPU, zero-shear branch).
# Helpers below are copied from that file; only backend/type plumbing and the
# host Newton diagnostic transfer differ. Options: see README.md.
using Dates
using LinearAlgebra
using Printf
using Random
using KernelAbstractions
using KrakeNN

const _GA_NU0 = 0.17
const _GA_GSCALE = 0.3
const _GA_THETA_REF = [log(0.02), log(5.0), 0.7, 2.0]   # (ln η∞, ln k, n, a)

function _ga_case(m::Int; nu0=_GA_NU0, gscale=_GA_GSCALE)
    Nx, Ny = 8m, 4m
    solid = [(((i - 0.5) / m - 4)^2 + ((j - 0.5) / m)^2 < 1.5^2) ||
             (((i - 0.5) / m - 4)^2 + ((j - 0.5) / m - 4)^2 < 1.5^2) for i in 1:Nx, j in 1:Ny]
    g = gscale * 5 * nu0^2 / m^3
    closure_of = θ -> CarreauYasuda(nu0, exp(θ[1]) * nu0, exp(θ[2]) * m^2 / nu0, θ[4], θ[3])
    return (; m, Nx, Ny, solid, g, nu0, uref=nu0 / m, closure_of)
end

function _ga_forward(case, θ; tol, f_init=nothing, tau_init=nothing, max_steps=200_000,
                     backend=KernelAbstractions.CPU(), T=Float64)
    return run_gnf_periodic_steady(; is_solid=case.solid, closure=case.closure_of(θ), Fx=case.g,
                                   backend=backend, T=T, tol=tol, check_every=20, fit_checks=20, fused=true,
                                   f_init=f_init, tau_init=tau_init, max_steps=max_steps)
end

# Forward to dist_est ≤ tol1, then `nextra` further levels of K steps each
# (one trajectory). K from the fitted contraction of the first level, unless given.
function _ga_forward_chain(case, θ; tol1=1e-10, nextra=2, K=nothing, f_init=nothing,
                           tau_init=nothing, backend=KernelAbstractions.CPU(), T=Float64)
    r = _ga_forward(case, θ; tol=tol1, f_init=f_init, tau_init=tau_init, backend=backend, T=T)
    Kc = K === nothing ? clamp(ceil(Int, log(100) / -log(r.contraction)), 200, 20_000) : K
    out = [(; f=copy(r.f), tau=copy(r.tau), dist=r.dist_est, steps=r.steps, rho=r.contraction,
            converged=r.converged, ux=r.ux, uy=r.uy, I1=copy(r.I1))]
    for _ in 1:nextra
        r = run_gnf_periodic_steady(; is_solid=case.solid, closure=case.closure_of(θ),
                                    Fx=case.g, backend=backend, T=T, tol=0.0, check_every=20, fit_checks=20,
                                    fused=true, f_init=r.f, tau_init=r.tau, max_steps=Kc)
        push!(out, (; f=copy(r.f), tau=copy(r.tau), dist=r.dist_est, steps=r.steps,
                    rho=r.contraction, converged=true, ux=r.ux, uy=r.uy, I1=copy(r.I1)))
    end
    return out, Kc
end

# True relative distance of u to the fixed point, from one Newton step on the
# active space: (I − P A P) δ + ĉ s = P (G(w) − w), ĉᵀ δ = 0 (mass border).
function _ga_newton_dist(case, st, θ)
    st = (; f=Array(st.f), tau=Array(st.tau)) # Newton diagnostic is host-only.
    cl = case.closure_of(θ)
    lin = gnf_linearise(st.f, st.tau, case.solid, cl, case.g)
    op = GNFStepOperator(lin)
    n = op.n
    w = gnf_pack_state(st.f, st.tau)
    r = (gnf_step_state(w, case.solid, cl, fill(case.g, case.Nx, case.Ny),
                        zeros(case.Nx, case.Ny)) .- w) .* lin.active
    tmp = zeros(n)
    function apply!(y, x)
        xn = view(x, 1:n)
        gnf_jvp!(tmp, op, xn)
        view(y, 1:n) .= xn .- tmp .+ op.chat .* x[n+1]
        y[n+1] = dot(op.chat, xn)
        return y
    end
    x = zeros(n + 1)
    info = gnf_gmres!(x, apply!, vcat(r, 0.0); tol=1e-8, restart=500, maxiter=5000)
    f1, _ = gnf_unpack_state(w .+ x[1:n], case.Nx, case.Ny)
    _, ux0, uy0 = gnf_post_moments(st.f, case.solid, case.g)
    _, ux1, uy1 = gnf_post_moments(f1, case.solid, case.g)
    d = sqrt(sum(abs2, ux1 .- ux0) + sum(abs2, uy1 .- uy0)) / sqrt(sum(abs2, ux0) + sum(abs2, uy0))
    return d, info.iters
end

# Node-wise observation of both velocity components at every fluid node,
# u* = u_lat/u_ref, targets 0.8 u*_ref (so that ∇J ≠ 0).
function _ga_obs(case, ux, uy)
    nodes = findall(vec(.!case.solid))
    M = 2 * length(nodes)
    node = zeros(Int, 1, M); w = ones(1, M); comp = zeros(Int, M); tgt = zeros(M)
    for (k, nd) in enumerate(nodes)
        node[1, 2k-1] = nd; comp[2k-1] = 1; tgt[2k-1] = 0.8 * ux[nd] / case.uref
        node[1, 2k] = nd; comp[2k] = 2; tgt[2k] = 0.8 * uy[nd] / case.uref
    end
    return GNFVelocityObs(node, w, comp, tgt, 1 / case.uref)
end

_ga_J(case, obs, st) = (m = gnf_post_moments(st.f, case.solid, case.g);
                        gnf_obs_objective(obs, m[2], m[3]))

_ga_relerr(a, b) = norm(a .- b) / max(norm(b), 1e-300)

const NAMES = ("ln eta_inf", "ln k", "n", "a")
const DISTS = (1e-10, 1e-12, 1e-14)
# Steps of the frozen T5 check (one per decade) and of the accuracy sweep (three per
# decade from 1e-2 to 1e-5, containing the four decade steps exactly).
const HS_DECADES = (1e-2, 1e-3, 1e-4, 1e-5)
const HS_SWEEP = (1e-2, 1e-2 * 10^(-1 / 3), 1e-2 * 10^(-2 / 3), 1e-3, 1e-3 * 10^(-1 / 3),
                  1e-3 * 10^(-2 / 3), 1e-4, 1e-4 * 10^(-1 / 3), 1e-4 * 10^(-2 / 3), 1e-5)
# Accuracy check (gates fixed before the run): per component, the budget
# b_p = 1e-6 |g_p| + 1e-10 max(1, ‖g‖∞); accuracy |g_p − FD_p| ≤ b_p at two
# neighbouring steps; FD error slope 2 ± 0.3 on a window of four consecutive
# sweep steps (one decade); tolerance changes move g_p by ≤ b_p/10; GPU against
# CPU within b_p; no perturbation changes the number of zero-shear nodes.
const ACC_REL = 1e-6
const ACC_ABS = 1e-10
const ACC_SLOPE = (1.7, 2.3)
const ACC_WINDOW = 4
const ZERO_SHEAR_I1 = 1e-12      # threshold of `gnf_linearise` (n_small_I1), lattice units

function options(args)
    opts = Dict("backend" => "cuda", "out" => "benchmarks/validation/gnf_adjoint/results",
                "float32" => "yes", "a" => "2.5", "grid" => "32x16", "hs" => "sweep",
                "allow-unknown-commit" => "no")
    for arg in args
        parts = split(arg, '='; limit=2)
        length(parts) == 2 && startswith(parts[1], "--") || error("Expected --key=value: $arg")
        key = parts[1][3:end]
        haskey(opts, key) || error("Unknown option: $key")
        opts[key] = parts[2]
    end
    opts["backend"] in ("cpu", "cuda") || error("--backend must be cuda or cpu")
    opts["float32"] in ("yes", "no") || error("--float32 must be yes or no")
    opts["hs"] in ("sweep", "decades") || error("--hs must be sweep or decades")
    opts["allow-unknown-commit"] in ("yes", "no") || error("--allow-unknown-commit must be yes or no")
    isempty(opts["out"]) && error("--out must not be empty")
    a = tryparse(Float64, opts["a"])
    (a !== nothing && isfinite(a) && a > 0) || error("--a must be a positive number")
    grid_m(opts["grid"])
    return opts
end

"Grid `NXxNY` with Nx = 2 Ny and Ny a multiple of 4 (8m × 4m) → m."
function grid_m(s)
    parts = split(lowercase(s), 'x')
    nx = length(parts) == 2 ? tryparse(Int, parts[1]) : nothing
    ny = length(parts) == 2 ? tryparse(Int, parts[2]) : nothing
    (nx !== nothing && ny !== nothing && nx == 2ny && ny % 4 == 0 && ny >= 8) ||
        error("--grid must be (8m)x(4m) with m ≥ 2, e.g. 32x16 or 64x32")
    return ny ÷ 4
end

"Test point: the reference θ of the test with its a component replaced by `--a`."
test_point(opts) = [_GA_THETA_REF[1:3]; parse(Float64, opts["a"])]

# Capture stdout and stderr, including library diagnostics, while streaming to
# the original console. The reader drains asynchronously and is joined on exit.
function transcript(f, path)
    console = stdout
    open(path, "w") do log
        pipe = Pipe()
        Base.link_pipe!(pipe; reader_supports_async=true, writer_supports_async=true)
        copier = @async while !eof(pipe)
            bytes = readavailable(pipe)
            write(log, bytes); flush(log)
            write(console, bytes); flush(console)
        end
        try
            redirect_stdout(pipe) do
                redirect_stderr(pipe) do
                    f()
                end
            end
        finally
            close(pipe.in)
            wait(copier)
            close(pipe.out)
        end
    end
end

csvfield(x) = '"' * replace(string(x), '"' => "\"\"") * '"'
function row(csv, check, parameter; value="", reference="", abs_err="", rel_err="",
             bound="", pass="informational", printrow=true)
    fields = (check, parameter, value, reference, abs_err, rel_err, bound, pass)
    println(csv, join(csvfield.(fields), ',')); flush(csv)
    printrow && println("  ", join(string.(fields), " | "))
    return pass !== false
end

function timed(f, check, backend)
    KernelAbstractions.synchronize(backend)
    t0 = time()
    try
        return f()
    finally
        KernelAbstractions.synchronize(backend)
        @printf("  %s wall %.3f s\n", check, time() - t0)
    end
end

function device_rhs(op, bh)
    b = similar(op.chat, length(bh))
    copyto!(b, bh)
    KernelAbstractions.synchronize(KernelAbstractions.get_backend(b))
    return b
end

function gradient(lin, obs, case, θ, backend; T=Float64, tol=1e-8)
    op = GNFStepOperator(lin; backend=backend, T=T)
    _, bh = gnf_obs_rhs(obs, lin)
    λ, _, info = gnf_adjoint_solve(op, device_rhs(op, bh); tol=tol,
                                  restart=1000, maxiter=20_000)
    g, _ = gnf_parameter_gradient(lin, λ, θ, case.closure_of)
    return g, info
end

function solve_report(csv, check, info; tol)
    @printf("  %s GMRES: tol %.0e converged=%s relres=%.6e iters=%d\n",
            check, tol, info.converged, info.relres, info.iters)
    row(csv, check, "converged"; value=info.converged, printrow=false)
    row(csv, check, "relres"; value=info.relres, reference=tol, printrow=false)
    row(csv, check, "iters"; value=info.iters, printrow=false)
end

function parity(csv, case, θ, backend)
    if backend isa KernelAbstractions.CPU
        row(csv, "P", "CPU/device parity"; pass="skipped (CPU backend)")
        return true
    end
    # As in G: both operators use the same CPU steady state and Float64 blocks.
    base, _ = _ga_forward_chain(case, θ)
    st = base[end]
    obs = _ga_obs(case, st.ux, st.uy)
    lin = gnf_linearise(st.f, st.tau, case.solid, case.closure_of(θ), case.g)
    opc = GNFStepOperator(lin)
    opd = GNFStepOperator(lin; backend=backend)
    v = randn(MersenneTwister(20261020), opc.n) .* Float64.(lin.active)
    vd = device_rhs(opd, v)
    yc = similar(v); yd = similar(vd)
    passed = true
    for (name, product!) in (("JVP", gnf_jvp!), ("VJP", gnf_vjp!))
        product!(yc, opc, v); product!(yd, opd, vd)
        KernelAbstractions.synchronize(backend)
        err = _ga_relerr(Array(yd), yc)
        # Frozen gate: test/ad/gnf_adjoint_test.jl, G.
        passed &= row(csv, "P", name; value=norm(Array(yd)), reference=norm(yc),
                      abs_err=norm(Array(yd) .- yc), rel_err=err, bound=1e-14, pass=err <= 1e-14)
    end
    _, bh = gnf_obs_rhs(obs, lin)
    λc, _, ic = gnf_adjoint_solve(opc, bh; tol=1e-10)
    λd, _, id = gnf_adjoint_solve(opd, device_rhs(opd, bh); tol=1e-10)
    solve_report(csv, "P CPU", ic; tol=1e-10)
    solve_report(csv, "P device", id; tol=1e-10)
    err = _ga_relerr(Array(λd), λc)
    passed &= row(csv, "P", "adjoint solve"; value=norm(Array(λd)), reference=norm(λc),
                  abs_err=norm(Array(λd) .- λc), rel_err=err, bound=1e-8, pass=err <= 1e-8)
    return passed
end

# Same finest state and observation as A, through `gnf_steady_gradient` with the
# K-step solver (the code default); informational.
function gradient_kstep(st, obs, case, θ, backend; tol=1e-8)
    out = gnf_steady_gradient(st.f, st.tau, θ, obs; is_solid=case.solid, closure_of=case.closure_of,
                              Fx=case.g, backend=backend, T=Float64, solver=:kstep, tol=tol)
    return out.grad, out.info
end

"Fluid nodes on the zero-shear branch (I1 < 1e-12) of a forward state."
zero_shear(case, st) = count(Array(st.I1)[.!case.solid] .< ZERO_SHEAR_I1)

"Least-squares slope of log(err) against log(h)."
function loglog_slope(h, err)
    x = log.(collect(h)); y = log.(max.(collect(err), floatmin(Float64)))
    xm = sum(x) / length(x); ym = sum(y) / length(y)
    return sum((x .- xm) .* (y .- ym)) / sum(abs2, x .- xm)
end

function validate(opts, csv)
    backend = opts["backend"] == "cuda" ? CUDA.CUDABackend() : KernelAbstractions.CPU()
    case = _ga_case(grid_m(opts["grid"])); θ0 = test_point(opts)
    hs = opts["hs"] == "sweep" ? HS_SWEEP : HS_DECADES
    idec = [findfirst(==(h), hs) for h in HS_DECADES]
    @printf("Case %dx%d; test point (ln eta_inf, ln k, n, a) = (%.6f, %.6f, %.4f, %.4f); FD steps (%s): %s\n",
            case.Nx, case.Ny, θ0..., opts["hs"], join([@sprintf("%.3e", h) for h in hs], ", "))
    θ0[4] * (1 - maximum(hs)) < 2 &&
        println("  note: central differences in a cross a < 2 at the largest steps")
    println("Checks: check | parameter | value | reference | abs_err | rel_err | bound | pass")
    passed = timed(() -> parity(csv, case, θ0, backend), "P", backend)
    base, Kchain, ndist = timed("F", backend) do
        base, Kchain = _ga_forward_chain(case, θ0; backend=backend, T=Float64)
        ndist = [_ga_newton_dist(case, st, θ0) for st in base]
        @printf("  F forward levels: steps %s (K = %d, rho_1 = %.6f) ; dist_est %s ; Newton distance %s (GMRES it %s)\n",
                join([s.steps for s in base], "/"), Kchain, base[1].rho,
                join([@sprintf("%.1e", s.dist) for s in base], "/"),
                join([@sprintf("%.1e", d[1]) for d in ndist], "/"), join([d[2] for d in ndist], "/"))
        row(csv, "F", "K"; value=Kchain)
        for (i, st) in enumerate(base)
            row(csv, "F", "level $i nominal distance"; value=DISTS[i])
            for (name, value) in (("steps", st.steps), ("contraction", st.rho),
                                  ("dist_est", st.dist), ("Newton distance", ndist[i][1]),
                                  ("Newton GMRES iters", ndist[i][2]))
                row(csv, "F", "level $i $name"; value=value)
            end
        end
        # Frozen forward gates from the reference test's "case 32×16" testset.
        st = base[end]; fl = .!case.solid
        lin = gnf_linearise(st.f, st.tau, case.solid, case.closure_of(θ0), case.g)
        umax = maximum(sqrt.(st.ux[fl] .^ 2 .+ st.uy[fl] .^ 2))
        ratio = maximum(lin.nu[fl]) / minimum(lin.nu[fl])
        passed &= row(csv, "F", "level 1 converged"; value=base[1].converged,
                      reference=true, pass=base[1].converged)
        passed &= row(csv, "F", "level 1 Newton gate"; value=ndist[1][1], bound=1e-9,
                      pass=ndist[1][1] <= 1e-9)
        passed &= row(csv, "F", "finest Newton gate"; value=ndist[end][1], bound=ndist[1][1],
                      pass=ndist[end][1] <= ndist[1][1])
        passed &= row(csv, "F", "Mach (< bound)"; value=sqrt(3) * umax, bound=0.15,
                      pass=sqrt(3) * umax < 0.15)
        passed &= row(csv, "F", "viscosity ratio (> bound)"; value=ratio, bound=2, pass=ratio > 2)
        row(csv, "F", "zero-shear nodes (finest level)"; value=zero_shear(case, st))
        return base, Kchain, ndist
    end
    st = base[end]
    obs = _ga_obs(case, st.ux, st.uy)
    lin = gnf_linearise(st.f, st.tau, case.solid, case.closure_of(θ0), case.g)
    g, fd, zs = timed("A", backend) do
        g, info = gradient(lin, obs, case, θ0, backend)
        solve_report(csv, "A", info; tol=1e-8)
        fd = Dict{Tuple{Int,Int,Int},Float64}()
        zs = Dict{Tuple{Int,Int,Int},Int}()       # (p, step, sign) → zero-shear nodes, finest level
        for p in 1:4, (ih, h) in enumerate(hs)
            δ = h * abs(θ0[p])
            θp = copy(θ0); θp[p] += δ
            θm = copy(θ0); θm[p] -= δ
            cp, _ = _ga_forward_chain(case, θp; K=Kchain, f_init=base[1].f,
                                      tau_init=base[1].tau, backend=backend, T=Float64)
            cm, _ = _ga_forward_chain(case, θm; K=Kchain, f_init=base[1].f,
                                      tau_init=base[1].tau, backend=backend, T=Float64)
            for id in eachindex(DISTS)
                fd[(id, p, ih)] = (_ga_J(case, obs, cp[id]) - _ga_J(case, obs, cm[id])) / (2δ)
            end
            zs[(p, ih, 1)] = zero_shear(case, cp[end]); zs[(p, ih, -1)] = zero_shear(case, cm[end])
        end
        # Frozen T5 criterion: finest distance, tol 1e-8, best of the four decade steps.
        atol = 1e-3 * norm(g) / sqrt(4)
        for p in 1:4
            errs = [abs(g[p] - fd[(3, p, ih)]) for ih in idec]
            ib = argmin(errs)
            ref = fd[(3, p, idec[ib])]; bound = atol + 1e-3 * abs(ref)
            ok = errs[ib] <= bound
            passed &= row(csv, "A", NAMES[p]; value=g[p], reference=ref,
                          abs_err=errs[ib], rel_err=errs[ib] / abs(ref), bound=bound,
                          pass=ok, printrow=false)
            row(csv, "A", "$(NAMES[p]) best h"; value=HS_DECADES[ib], printrow=false)
            @printf("  A criterion %s: adj %.6e  DF(h=%.0e) %.6e  |diff| %.2e  bound %.2e  %s\n",
                    NAMES[p], g[p], HS_DECADES[ib], ref, errs[ib], bound, ok ? "ok" : "FAIL")
        end
        return g, fd, zs
    end
    b = [ACC_REL * abs(g[p]) + ACC_ABS * max(1, norm(g, Inf)) for p in 1:4]
    timed("accuracy", backend) do
        println("  accuracy budget b_p = 1e-6 |g_p| + 1e-10 max(1, |g|_inf): ",
                join([@sprintf("%.3e", x) for x in b], ", "))
        for p in 1:4, (ih, h) in enumerate(hs)
            e = abs(g[p] - fd[(3, p, ih)])
            row(csv, "accuracy FD data", @sprintf("%s h=%.6e", NAMES[p], h); value=fd[(3, p, ih)],
                reference=g[p], abs_err=e, rel_err=e / max(abs(g[p]), 1e-300), printrow=false)
        end
        # Accuracy: the budget met at two neighbouring steps (finest forward level).
        for p in 1:4
            errs = [abs(g[p] - fd[(3, p, ih)]) for ih in eachindex(hs)]
            pair = [max(errs[i], errs[i+1]) for i in 1:length(hs)-1]
            i = argmin(pair); ok = pair[i] <= b[p]
            passed &= row(csv, "accuracy budget", NAMES[p]; value=g[p], reference=fd[(3, p, i)],
                          abs_err=pair[i], rel_err=pair[i] / max(abs(g[p]), 1e-300), bound=b[p],
                          pass=ok, printrow=false)
            row(csv, "accuracy budget", "$(NAMES[p]) neighbouring steps";
                value=@sprintf("%.6e;%.6e", hs[i], hs[i+1]), printrow=false)
            @printf("  accuracy budget %s: max |g - FD| over h = %.2e, %.2e is %.2e, budget %.2e  %s\n",
                    NAMES[p], hs[i], hs[i+1], pair[i], b[p], ok ? "ok" : "FAIL")
        end
        # Sweep: FD error slope on windows of four consecutive steps (one decade).
        if opts["hs"] == "sweep"
            for p in 1:4
                errs = [abs(g[p] - fd[(3, p, ih)]) for ih in eachindex(hs)]
                slopes = [loglog_slope(hs[i:i+ACC_WINDOW-1], errs[i:i+ACC_WINDOW-1])
                          for i in 1:length(hs)-ACC_WINDOW+1]
                good = findall(s -> ACC_SLOPE[1] <= s <= ACC_SLOPE[2], slopes)
                i = isempty(good) ? argmin(abs.(slopes .- 2)) : first(good)
                passed &= row(csv, "accuracy sweep", NAMES[p]; value=slopes[i], reference=2,
                              bound=@sprintf("%.1f..%.1f", ACC_SLOPE...), pass=!isempty(good),
                              printrow=false)
                row(csv, "accuracy sweep", "$(NAMES[p]) window"; value=@sprintf("%.6e..%.6e", hs[i], hs[i+ACC_WINDOW-1]),
                    printrow=false)
                row(csv, "accuracy sweep", "$(NAMES[p]) slopes of all windows";
                    value=join([@sprintf("%.3f", s) for s in slopes], ";"), printrow=false)
                @printf("  accuracy sweep %s: slope %.3f on h = %.2e..%.2e (all windows, largest h first: %s)  %s\n",
                        NAMES[p], slopes[i], hs[i], hs[i+ACC_WINDOW-1],
                        join([@sprintf("%.2f", s) for s in slopes], " "), isempty(good) ? "FAIL" : "ok")
            end
        else
            row(csv, "accuracy sweep", "FD error slope"; pass="skipped (--hs=decades)")
        end
        # Solves: adjoint tolerance 1e-8 → 1e-10 (finest state), then the forward
        # level 1 (target 1e-10) against level 3 (target 1e-14), both at adjoint 1e-10.
        g10, info10 = gradient(lin, obs, case, θ0, backend; tol=1e-10)
        solve_report(csv, "accuracy solves finest level, adjoint", info10; tol=1e-10)
        lin1 = gnf_linearise(base[1].f, base[1].tau, case.solid, case.closure_of(θ0), case.g)
        g1, info1 = gradient(lin1, obs, case, θ0, backend; tol=1e-10)
        solve_report(csv, "accuracy solves level 1, adjoint", info1; tol=1e-10)
        row(csv, "accuracy solves", "Newton distance level 1"; value=ndist[1][1], printrow=false)
        row(csv, "accuracy solves", "Newton distance level 3"; value=ndist[end][1], printrow=false)
        for p in 1:4
            da = abs(g10[p] - g[p]); df = abs(g1[p] - g10[p])
            passed &= row(csv, "accuracy solves", "$(NAMES[p]) adjoint tol 1e-8 vs 1e-10"; value=g[p],
                          reference=g10[p], abs_err=da, rel_err=da / max(abs(g10[p]), 1e-300),
                          bound=b[p] / 10, pass=da <= b[p] / 10, printrow=false)
            passed &= row(csv, "accuracy solves", "$(NAMES[p]) forward level 1 vs level 3"; value=g1[p],
                          reference=g10[p], abs_err=df, rel_err=df / max(abs(g10[p]), 1e-300),
                          bound=b[p] / 10, pass=df <= b[p] / 10, printrow=false)
            @printf("  accuracy solves %s: |g(1e-8) - g(1e-10)| %.2e, |g(level 1) - g(level 3)| %.2e, bound b/10 = %.2e  %s\n",
                    NAMES[p], da, df, b[p] / 10, max(da, df) <= b[p] / 10 ? "ok" : "FAIL")
        end
        @printf("  accuracy solves: Newton distance level 1 %.2e, level 3 %.2e\n", ndist[1][1], ndist[end][1])
        # Branch report: zero-shear nodes of every perturbed finest state against the base.
        nz0 = zero_shear(case, st); nchanged = 0
        for p in 1:4, (ih, h) in enumerate(hs), sg in (1, -1)
            nz = zs[(p, ih, sg)]; nchanged += nz != nz0
            passed &= row(csv, "accuracy branch", @sprintf("%s %sh h=%.6e", NAMES[p], sg > 0 ? "+" : "-", h);
                          value=nz, reference=nz0, pass=nz == nz0, printrow=false)
        end
        row(csv, "accuracy branch", "nodes at a viscosity bound";
            pass="informational (CarreauYasuda has no viscosity bound)", printrow=false)
        @printf("  accuracy branch: %d perturbations, zero-shear nodes at the base %d, perturbations with a different count %d  %s\n",
                8 * length(hs), nz0, nchanged, nchanged == 0 ? "ok" : "FAIL")
    end
    timed("C", backend) do
        # Same finest state, observation, blocks, RHS, and solve settings; CPU operator.
        gc, info = gradient(lin, obs, case, θ0, KernelAbstractions.CPU())
        gpu = !(backend isa KernelAbstractions.CPU)
        solve_report(csv, "C", info; tol=1e-8)
        for p in 1:4
            d = abs(g[p] - gc[p])
            if gpu
                # GPU: device against CPU operator within the accuracy budget.
                passed &= row(csv, "C", NAMES[p]; value=g[p], reference=gc[p], abs_err=d,
                              rel_err=d / max(abs(gc[p]), 1e-300), bound=b[p], pass=d <= b[p])
            else
                row(csv, "C", NAMES[p]; value=g[p], reference=gc[p], abs_err=d,
                    rel_err=d / max(abs(gc[p]), 1e-300),
                    pass="informational (CPU backend: both operators on the CPU)")
            end
        end
        row(csv, "C", "gradient norm"; value=norm(g), reference=norm(gc),
            abs_err=norm(g .- gc), rel_err=_ga_relerr(g, gc))
    end
    timed("S", backend) do
        if opts["float32"] == "no"
            row(csv, "S", "Float32 operator informational"; pass="skipped (--float32=no)")
        else
            # Only products use Float32; the blocks are built in Float64, as in A.
            g32, info = gradient(lin, obs, case, θ0, backend; T=Float32, tol=1e-5)
            solve_report(csv, "S informational Float32 operator", info; tol=1e-5)
            for p in 1:4
                row(csv, "S", NAMES[p]; value=g32[p], reference=g[p], abs_err=abs(g32[p] - g[p]),
                    rel_err=abs(g32[p] - g[p]) / max(abs(g[p]), 1e-300))
            end
            row(csv, "S", "gradient norm"; value=norm(g32), reference=norm(g),
                abs_err=norm(g32 .- g), rel_err=_ga_relerr(g32, g))
        end
    end
    timed("K", backend) do
        gk, info = gradient_kstep(st, obs, case, θ0, backend)
        solve_report(csv, "K informational K-step solver", info; tol=1e-8)
        for p in 1:4
            row(csv, "K", NAMES[p]; value=gk[p], reference=g[p], abs_err=abs(gk[p] - g[p]),
                rel_err=abs(gk[p] - g[p]) / max(abs(g[p]), 1e-300))
        end
    end
    return passed
end

function git_read(args...)
    root = normpath(joinpath(@__DIR__, "../../.."))
    try
        return chomp(read(pipeline(`git -C $root $args`, stderr=devnull), String))
    catch
        return nothing
    end
end

# Commit of the code that runs: `git rev-parse HEAD`; for a copy without .git, the
# KRAKEN_GIT_COMMIT environment variable; otherwise an error unless
# --allow-unknown-commit=yes.
function provenance(opts)
    commit = git_read("rev-parse", "HEAD"); source = "git rev-parse HEAD"
    if commit === nothing || isempty(commit)
        env = strip(get(ENV, "KRAKEN_GIT_COMMIT", ""))
        if !isempty(env)
            commit = String(env); source = "KRAKEN_GIT_COMMIT environment variable"
        elseif opts["allow-unknown-commit"] == "yes"
            commit = "unknown"; source = "none (--allow-unknown-commit=yes)"
        else
            error("cannot determine the git commit of the code: run from a git checkout, set " *
                  "KRAKEN_GIT_COMMIT to the commit the code was copied from, or pass --allow-unknown-commit=yes")
        end
    end
    status = git_read("status", "--porcelain", "--untracked-files=no")
    dirty = status === nothing ? nothing : [String(l) for l in split(status, '\n') if !isempty(l)]
    branch = something(git_read("rev-parse", "--abbrev-ref", "HEAD"), "unknown")
    return (; commit, source, branch, dirty)
end

function pkg_version(name)
    for (id, mod) in Base.loaded_modules
        id.name == name && return string(something(pkgversion(mod), "unknown"))
    end
    return "not loaded"
end

function main(args)
    opts, prov = try
        o = options(args)
        if o["backend"] == "cuda"
            isdefined(@__MODULE__, :CUDA) || error("CUDA is not loaded")
            CUDA.functional() || error("CUDA.functional() is false: no usable NVIDIA GPU and driver here")
            CUDA.allowscalar(false)
        end
        o, provenance(o)
    catch err
        println(stderr, "error: ", sprint(showerror, err))
        return 1
    end
    start = now(UTC); t0 = time()
    mkpath(opts["out"])
    m = grid_m(opts["grid"])
    stem = joinpath(opts["out"], "gnf_adjoint_gpu_$(opts["backend"])_$(8m)x$(4m)_a$(opts["a"])_$(Dates.format(start, "yyyymmdd_HHMMSS"))")
    transcript(stem * ".log") do
        open(stem * ".csv", "w") do csv
            println(csv, "check,parameter,value,reference,abs_err,rel_err,bound,pass")
            passed = false
            try
                println("Provenance: script=benchmarks/validation/gnf_adjoint/run_gnf_adjoint_gpu.jl")
                println("git_commit=", prov.commit, " (source: ", prov.source, ") branch=", prov.branch)
                if prov.dirty === nothing
                    println("git_modified_tracked_files=unknown")
                else
                    println("git_modified_tracked_files=", length(prov.dirty))
                    foreach(l -> println("  ", l), first(prov.dirty, 20))
                end
                println("Julia=", VERSION, " platform=", Sys.KERNEL, " ", Sys.ARCH, " cpu=", Sys.CPU_NAME)
                println("KrakeNN=", pkg_version("KrakeNN"), " KernelAbstractions=", pkg_version("KernelAbstractions"),
                        " CUDA.jl=", pkg_version("CUDA"), " ForwardDiff=", pkg_version("ForwardDiff"))
                println("backend=", opts["backend"], " precision=Float64; Float32 operator=", opts["float32"])
                println("options=", join(["$k=$(opts[k])" for k in sort(collect(keys(opts)))], ' '))
                println("start_utc=", start, " BLAS_threads=", BLAS.get_num_threads(), " Julia_threads=", Threads.nthreads())
                if opts["backend"] == "cuda"
                    println("device=", CUDA.name(CUDA.device()), " CUDA_runtime=", CUDA.runtime_version(),
                            " CUDA_driver=", CUDA.driver_version())
                else
                    println("device=CPU CUDA_runtime=not applicable")
                end
                passed = validate(opts, csv)
            catch err
                showerror(stderr, err, catch_backtrace()); println(stderr)
            finally
                println("end_utc=", now(UTC))
                @printf("total_wall_seconds=%.3f\n", time() - t0)
                println("Gated checks: ", passed ? "PASS" : "FAIL")
                println("CSV=", relpath(abspath(stem * ".csv"), pwd()), " LOG=", relpath(abspath(stem * ".log"), pwd()))
            end
            return passed ? 0 : 1
        end
    end
end

# Load CUDA at top level, before main runs, so its bindings are visible in main's world age.
let o = try options(ARGS) catch; nothing end
    o !== nothing && o["backend"] == "cuda" && @eval using CUDA
end

exit(main(ARGS))
