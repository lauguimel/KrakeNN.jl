#!/usr/bin/env julia
# Frozen T5/G validation of the periodic obstacle-channel case of
# test/ad/gnf_adjoint_test.jl. Helpers below are copied from that file;
# only backend/type plumbing and the host Newton diagnostic transfer differ.
using Dates
using Sockets
using LinearAlgebra
using Printf
using Random
using KernelAbstractions
using Kraken

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
            converged=r.converged, ux=r.ux, uy=r.uy)]
    for _ in 1:nextra
        r = run_gnf_periodic_steady(; is_solid=case.solid, closure=case.closure_of(θ),
                                    Fx=case.g, backend=backend, T=T, tol=0.0, check_every=20, fit_checks=20,
                                    fused=true, f_init=r.f, tau_init=r.tau, max_steps=Kc)
        push!(out, (; f=copy(r.f), tau=copy(r.tau), dist=r.dist_est, steps=r.steps,
                    rho=r.contraction, converged=true, ux=r.ux, uy=r.uy))
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
const HS = (1e-2, 1e-3, 1e-4, 1e-5)

function options(args)
    opts = Dict("backend" => "cuda", "out" => "benchmarks/validation/gnf_adjoint/results",
                "float32" => "yes")
    for arg in args
        parts = split(arg, '='; limit=2)
        length(parts) == 2 && startswith(parts[1], "--") || error("Expected --key=value: $arg")
        key = parts[1][3:end]
        haskey(opts, key) || error("Unknown option: $key")
        opts[key] = parts[2]
    end
    opts["backend"] in ("cpu", "cuda") || error("--backend must be cuda or cpu")
    opts["float32"] in ("yes", "no") || error("--float32 must be yes or no")
    isempty(opts["out"]) && error("--out must not be empty")
    return opts
end

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

function validate(opts, csv)
    backend = opts["backend"] == "cuda" ? CUDA.CUDABackend() : KernelAbstractions.CPU()
    case = _ga_case(4); θ0 = _GA_THETA_REF
    println("Checks: check | parameter | value | reference | abs_err | rel_err | bound | pass")
    passed = timed(() -> parity(csv, case, θ0, backend), "P", backend)
    base, Kchain = timed("F", backend) do
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
        return base, Kchain
    end
    st = base[end]
    obs = _ga_obs(case, st.ux, st.uy)
    lin = gnf_linearise(st.f, st.tau, case.solid, case.closure_of(θ0), case.g)
    g = timed("A", backend) do
        g, info = gradient(lin, obs, case, θ0, backend)
        solve_report(csv, "A", info; tol=1e-8)
        fd = Dict{Tuple{Int,Int,Int},Float64}()
        for p in 1:4, (ih, h) in enumerate(HS)
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
        end
        # Frozen T5 criterion: finest distance, tol 1e-8, best h per parameter.
        atol = 1e-3 * norm(g) / sqrt(4)
        for p in 1:4
            errs = [abs(g[p] - fd[(3, p, ih)]) for ih in eachindex(HS)]
            ib = argmin(errs)
            ref = fd[(3, p, ib)]; bound = atol + 1e-3 * abs(ref)
            ok = errs[ib] <= bound
            passed &= row(csv, "A", NAMES[p]; value=g[p], reference=ref,
                          abs_err=errs[ib], rel_err=errs[ib] / abs(ref), bound=bound,
                          pass=ok, printrow=false)
            row(csv, "A", "$(NAMES[p]) best h"; value=HS[ib], printrow=false)
            @printf("  A criterion %s: adj %.6e  DF(h=%.0e) %.6e  |diff| %.2e  bound %.2e  %s\n",
                    NAMES[p], g[p], HS[ib], ref, errs[ib], bound, ok ? "ok" : "FAIL")
        end
        return g
    end
    timed("C", backend) do
        # Same finest state, observation, blocks, RHS, and solve settings; CPU operator.
        gc, info = gradient(lin, obs, case, θ0, KernelAbstractions.CPU())
        solve_report(csv, "C informational", info; tol=1e-8)
        for p in 1:4
            row(csv, "C", NAMES[p]; value=g[p], reference=gc[p], abs_err=abs(g[p] - gc[p]),
                rel_err=abs(g[p] - gc[p]) / max(abs(gc[p]), 1e-300))
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
    return passed
end

function git_read(args...)
    root = normpath(joinpath(@__DIR__, "../../.."))
    try
        return chomp(read(pipeline(`git -C $root $args`, stderr=devnull), String))
    catch
        return "unknown"
    end
end

function main(args)
    opts = options(args)
    start = now(UTC); t0 = time()
    mkpath(opts["out"])
    stem = joinpath(opts["out"], "gnf_adjoint_gpu_$(opts["backend"])_$(Dates.format(start, "yyyymmdd_HHMMSS"))")
    transcript(stem * ".log") do
        open(stem * ".csv", "w") do csv
            println(csv, "check,parameter,value,reference,abs_err,rel_err,bound,pass")
            passed = false
            try
                status = git_read("status", "--short")
                dirty = status == "unknown" ? "unknown" : isempty(status) ? 0 : length(split(status, '\n'))
                println("Provenance: git_commit=", git_read("rev-parse", "HEAD"))
                println("git_status_short_lines=", dirty, " hostname=", gethostname())
                println("Julia=", VERSION, " platform=", Sys.KERNEL, " ", Sys.ARCH)
                println("backend=", opts["backend"], " precision=Float64; Float32 operator=", opts["float32"])
                println("start_utc=", start, " BLAS_threads=", BLAS.get_num_threads(), " Julia_threads=", Threads.nthreads())
                if opts["backend"] == "cuda"
                    @eval using CUDA
                    Base.invokelatest(CUDA.functional) || error("CUDA.functional() is false")
                    Base.invokelatest() do
                        println("device=", CUDA.name(CUDA.device()), " CUDA_runtime=", CUDA.runtime_version())
                        CUDA.allowscalar(false)
                    end
                else
                    println("device=CPU CUDA_runtime=not applicable")
                end
                passed = Base.invokelatest(validate, opts, csv)
            catch err
                showerror(stderr, err, catch_backtrace()); println(stderr)
            finally
                println("end_utc=", now(UTC))
                @printf("total_wall_seconds=%.3f\n", time() - t0)
                println("Gated checks: ", passed ? "PASS" : "FAIL")
                println("CSV=", stem * ".csv", " LOG=", stem * ".log")
            end
            return passed ? 0 : 1
        end
    end
end

exit(main(ARGS))

