# Train a learned basal sliding law F_theta(|u|, N) on the Pine Island Glacier crop.
#   (1) Weertman C sweep (best C = start target),
#   (2) pretraining without PDE: the network imitates the Weertman law F = C (|u|^2+u_reg^2)^(m/2),
#   (3) training: minimise L = 1/2 sum_valid (u-u_obs)^2 / sigma^2 + lambda/2 |theta-theta_pre|^2
#       with L-BFGS + backtracking; gradient by one adjoint solve per accepted iterate;
#       each solve warm-starts Newton from the previous state (eps continuation on failure).
#
#   julia --project=. benchmarks/ssa_glacier/pig_train.jl
#
# Environment (data from prep_pig.py, see pig_forward.jl):
#   KRAKEN_SSA_BACKEND = cpu (default) | cuda
#   KRAKEN_PIG_COARSEN = 1 (2 km, default) | 2 (4 km)
#   KRAKEN_PIG_ITERS   = number of L-BFGS iterations (default 50)
#   KRAKEN_PIG_SIGMA   = misfit scale in m/yr (default 100; one value for all cells)
#   KRAKEN_PIG_LAMBDA  = weight decay toward the pretrained theta (default 1.0)
#   KRAKEN_PIG_C       = Weertman C to imitate (default: best of the sweep KRAKEN_PIG_CS)
#   KRAKEN_PIG_PRETRAIN_STEPS = Adam steps of the pretraining (default 20000)
# Outputs in benchmarks/ssa_glacier/out/ with the suffix _c<coarsen>.
include(joinpath(@__DIR__, "pig_forward.jl"))   # loads CUDA at top level when asked, defines the helpers
if lowercase(get(ENV, "KRAKEN_SSA_BACKEND", "cpu")) == "cuda"
    using CUDA, CUDSS
end

const MLP_SCALE = (1000.0, 1e4)
softplus(t) = KrakeNN._law_softplus(t)
logistic(t) = KrakeNN._law_logistic(t)

# ------------------------------------------------------------------ pretraining (no PDE)
"""Adam on the relative error of F_theta against the target F_t over the samples (sp, N)."""
function pretrain(mlp, theta0, F0, sp, Nv, Ft; steps = 20000, lr = 0.02, lr_end = 1e-4)
    theta = copy(theta0); np = length(theta); n = length(sp)
    m = zeros(np); v = zeros(np); g = zeros(np)
    b1, b2 = 0.9, 0.999
    for t in 1:steps
        fill!(g, 0.0)
        for k in 1:n
            z = law_value(mlp, (sp[k], Nv[k]), theta)
            F = F0 * softplus(z)
            r = F / Ft[k] - 1
            law_pullback!(g, mlp, (sp[k], Nv[k]), theta, r / Ft[k] * F0 * logistic(z) / n)
        end
        a = lr * (lr_end / lr)^(t / steps)
        @. m = b1 * m + (1 - b1) * g
        @. v = b2 * v + (1 - b2) * g^2
        @. theta -= a * (m / (1 - b1^t)) / (sqrt(v / (1 - b2^t)) + 1e-8)
    end
    return theta
end

drag_F(mlp, theta, F0, sp, N) = F0 * softplus(law_value(mlp, (sp, N), theta))

function fit_error(mlp, theta, F0, sp, Nv, Ft)
    r = [drag_F(mlp, theta, F0, sp[k], Nv[k]) / Ft[k] - 1 for k in eachindex(sp)]
    return (rms = sqrt(sum(abs2, r) / length(r)), max = maximum(abs, r))
end

# ------------------------------------------------------------------ L-BFGS
function lbfgs_dir(g, S, Y)
    q = copy(g); k = length(S); al = zeros(k)
    for i in k:-1:1
        al[i] = dot(S[i], q) / dot(Y[i], S[i]); q .-= al[i] .* Y[i]
    end
    k > 0 && (q .*= dot(S[end], Y[end]) / dot(Y[end], Y[end]))
    for i in 1:k
        b = dot(Y[i], q) / dot(Y[i], S[i]); q .+= (al[i] - b) .* S[i]
    end
    return -q
end

function main_train()
    be = select_backend()
    k = parse(Int, get(ENV, "KRAKEN_PIG_COARSEN", "1"))
    iters = parse(Int, get(ENV, "KRAKEN_PIG_ITERS", "50"))
    sigma = parse(Float64, get(ENV, "KRAKEN_PIG_SIGMA", "100"))
    lambda = parse(Float64, get(ENV, "KRAKEN_PIG_LAMBDA", "1.0"))
    d = load_pig(; coarsen = k)
    @printf("PIG crop %d x %d at %.0f m, backend %s, iters %d, sigma %.0f m/yr, lambda %.3g\n",
            d.Nx, d.Ny, d.dx, be.name, iters, sigma, lambda)
    Uobs = observed_state(d); w = misfit_weights(d)
    W = zeros(length(Uobs)); W[1:2:end] .= vec(w); W[2:2:end] .= vec(w)
    mkpath(OUT); tag = "_c$(k)"

    # (1) Weertman sweep
    println("\n== Weertman sweep (m = 1/3)")
    Cs = Tuple(parse.(Float64, split(get(ENV, "KRAKEN_PIG_CS", "5,10,20,40,80,160"), ",")))
    best = nothing
    for C in Cs
        p = pig_problem(d; drag = SSAWeertman(C, 1 / 3))
        s = ssa_newton_solver(p; to = be.to, linsolve = be.linsolve)
        U = be.to(copy(Uobs)); st = solve_continuation!(s, U, EPS_LADDER)
        m = misfit_stats(d, Array(U))
        @printf("C = %5.1f  converged = %s  RMS speed all/grounded/floating = %.0f/%.0f/%.0f\n",
                C, st[end].converged, m.rms_speed, m.rms_grounded, m.rms_floating)
        if st[end].converged && (best === nothing || m.rms_grounded < best.m.rms_grounded)
            best = (; C, m, U = Array(U))
        end
    end
    best === nothing && error("no Weertman case converged")
    C = haskey(ENV, "KRAKEN_PIG_C") ? parse(Float64, ENV["KRAKEN_PIG_C"]) : best.C
    pw = pig_problem(d; drag = SSAWeertman(C, 1 / 3))
    if C != best.C
        s = ssa_newton_solver(pw; to = be.to, linsolve = be.linsolve)
        U = be.to(copy(Uobs)); solve_continuation!(s, U, EPS_LADDER)
        best = (; C, m = misfit_stats(d, Array(U)), U = Array(U))
    end
    @printf("Weertman reference: C = %.1f, RMS speed all/grounded/floating = %.1f/%.1f/%.1f m/yr, max model %.0f obs %.0f\n",
            C, best.m.rms_speed, best.m.rms_grounded, best.m.rms_floating, best.m.max_model, best.m.max_obs)
    Uw = best.U

    # (2) pretraining on samples spanning the crop ranges
    println("\n== Pretraining: F_theta(|u|, N) ~ C sp^(1/3)")
    gr = (d.mask .== 2)
    Ncell = max.(KrakeNN.SSA_RHOG .* max.(d.H, 10.0) .- RHO_W_G .* max.(0.0, -d.bed), 10.0)
    sp_obs = hypot.(d.vx, d.vy)
    sel = gr .& (d.vvalid .> 0.5) .& (sp_obs .> 0)
    N_lo, N_hi = extrema(Ncell[sel]); u_lo = max(1.0, minimum(sp_obs[sel]) / 2); u_hi = 1.3 * maximum(sp_obs[sel])
    ug_ = range(log(u_lo), log(u_hi); length = 40); Ng_ = range(log(N_lo / 1.2), log(N_hi * 1.2); length = 20)
    gsp = [exp(a) for a in ug_ for _ in Ng_]; gN = [exp(b) for _ in ug_ for b in Ng_]
    rng = MersenneTwister(7)
    idx = findall(sel); idx = idx[randperm(rng, length(idx))[1:min(600, length(idx))]]
    csp = sp_obs[idx]; cN = Ncell[idx]
    usp(u) = sqrt.(u .^ 2 .+ 1e-4)
    Xsp = vcat(gsp, usp(csp)); XN = vcat(gN, cN)
    Ft = C .* Xsp .^ (1 / 3)
    @printf("sample ranges: |u| %.1f-%.0f m/yr, N %.0f-%.0f kPa (%d grid + %d crop cells)\n", u_lo, u_hi, N_lo, N_hi, length(gsp), length(idx))
    mlp = TinyMLP{2,4,1}(; shift = (0.0, 0.0), scale = MLP_SCALE)
    F0 = C * median(sp_obs[sel])^(1 / 3) / log(2)
    theta0 = law_init(mlp; rng = MersenneTwister(1))
    t_pre = @elapsed theta_pre = pretrain(mlp, theta0, F0, Xsp, XN, Ft;
                                           steps = parse(Int, get(ENV, "KRAKEN_PIG_PRETRAIN_STEPS", "20000")))
    fe_grid = fit_error(mlp, theta_pre, F0, gsp, gN, C .* gsp .^ (1 / 3))
    fe_cell = fit_error(mlp, theta_pre, F0, usp(csp), cN, C .* usp(csp) .^ (1 / 3))
    @printf("pretrain (%.1f s): relative error of F vs Weertman: grid RMS %.3f max %.3f; crop cells RMS %.3f max %.3f\n",
            t_pre, fe_grid.rms, fe_grid.max, fe_cell.rms, fe_cell.max)

    # solver with the learned drag; start from the Weertman state, eps continuation
    drag = SSALearnedDrag(mlp, copy(theta_pre); F0)
    pl = ssa_with(pw; drag, eps_reg = 1e-5)
    sl = ssa_newton_solver(pl; to = be.to, linsolve = be.linsolve)
    Ub = be.to(copy(Uobs)); Wd = be.to(W)
    Ucur = be.to(copy(Uw)); Utry = similar(Ucur)
    st = solve_continuation!(sl, Ucur, EPS_LADDER)
    mpre = misfit_stats(d, Array(Ucur))
    @printf("pretrained network: Newton converged = %s, its per stage = %s, RMS speed all/grounded/floating = %.1f/%.1f/%.1f (Weertman %.1f/%.1f/%.1f)\n",
            st[end].converged, join(getfield.(st, :iterations), "+"), mpre.rms_speed, mpre.rms_grounded, mpre.rms_floating,
            best.m.rms_speed, best.m.rms_grounded, best.m.rms_floating)
    st[end].converged || error("Newton did not converge with the pretrained network")
    Upre = Array(Ucur)

    # (3) training
    println("\n== Training (L-BFGS, backtracking)")
    misfitL(U) = (r = Wd .* (U .- Ub); (sum(r .* (U .- Ub)) / (2 * sigma^2), r ./ sigma^2))
    # one solve at theta from the state Ucur; returns (L_total, ok, its, fell_back)
    function trial!(theta)
        ssa_set_theta!(sl, theta)
        copyto!(Utry, Ucur)
        sl.prob = ssa_with(sl.prob; eps_reg = 1e-5)
        r = ssa_newton!(sl, Utry; atol = 1e-8, rtol = 1e-13, maxit = 30)
        its = r.iterations; fb = false
        if !r.converged
            fb = true; copyto!(Utry, Ucur)
            stc = solve_continuation!(sl, Utry, EPS_LADDER)
            its += sum(getfield.(stc, :iterations))
            stc[end].converged || return (Inf, false, its, fb)
        end
        Lm = misfitL(Utry)[1]
        return (Lm + lambda / 2 * sum(abs2, theta .- theta_pre), true, its, fb)
    end
    function gradient_at(theta)       # sl holds theta and Ucur = the solution at theta
        g = ssa_theta_gradient(sl, Ucur, misfitL(Ucur)[2])
        return g .+ lambda .* (theta .- theta_pre)
    end
    theta = copy(theta_pre)
    Lc, ok, its0, _ = trial!(theta); ok || error("initial solve failed")
    copyto!(Ucur, Utry)
    g = gradient_at(theta)
    if get(ENV, "KRAKEN_PIG_GRADCHECK", "0") == "1"      # directional derivative: adjoint vs central FD
        dirv = g ./ norm(g)
        for h in (1e-2, 1e-3, 1e-4)
            Lp = trial!(theta .+ h .* dirv)[1]; Lm = trial!(theta .- h .* dirv)[1]
            @printf("gradcheck h = %.0e  adjoint %.6e  FD %.6e\n", h, dot(g, dirv), (Lp - Lm) / (2h))
        end
        trial!(theta); copyto!(Ucur, Utry)
    end
    hist = open(joinpath(OUT, "pig_train_history$(tag).csv"), "w")
    println(hist, "iter,L,Lmisfit,rms_all,rms_grounded,rms_floating,newton_its,solves,fallback,grad_inf,grad_l2,step_norm")
    function log_iter(it, L, Ucurr, its, solves, fb, gg, sn)
        ms = misfit_stats(d, Array(Ucurr))
        @printf("it %3d  L %.6e  RMS speed all/grounded/floating %.1f/%.1f/%.1f  Newton its %d (%d solves%s)  |g|inf %.2e |g|2 %.2e  step %.3f\n",
                it, L, ms.rms_speed, ms.rms_grounded, ms.rms_floating, its, solves, fb ? ", fallback" : "", maximum(abs, gg), norm(gg), sn)
        println(hist, join((it, L, L - lambda / 2 * sum(abs2, theta .- theta_pre), ms.rms_speed, ms.rms_grounded, ms.rms_floating, its, solves, Int(fb), maximum(abs, gg), norm(gg), sn), ","))
        flush(hist)
    end
    log_iter(0, Lc, Ucur, its0, 1, false, g, 0.0)
    S = Vector{Float64}[]; Y = Vector{Float64}[]
    for it in 1:iters
        dir = lbfgs_dir(g, S, Y)
        dot(dir, g) >= 0 && (empty!(S); empty!(Y); dir = -g)
        maxstep = 0.3
        t = isempty(S) ? min(1.0, maxstep / norm(dir)) : min(1.0, maxstep / norm(dir))
        accepted = false; its_tot = 0; solves = 0; fb_any = false; thn = theta; Ln = Lc
        for _ in 1:10
            thn = theta .+ t .* dir
            Ln, ok, its, fb = trial!(thn); solves += 1; its_tot += its; fb_any |= fb
            if ok && Ln <= Lc + 1e-4 * t * dot(g, dir)
                accepted = true; break
            end
            t /= 2
        end
        if !accepted
            if isempty(S)
                println("line search failed on a steepest-descent step: stop"); break
            end
            println("line search failed: reset the L-BFGS memory")
            empty!(S); empty!(Y)
            # restore the solver at theta (state Ucur is still the solution there)
            ssa_set_theta!(sl, theta)
            continue
        end
        copyto!(Ucur, Utry)
        gn = gradient_at(thn)
        s_ = thn .- theta; y_ = gn .- g
        if dot(s_, y_) > 1e-10 * norm(s_) * norm(y_)
            push!(S, s_); push!(Y, y_); length(S) > 8 && (popfirst!(S); popfirst!(Y))
        end
        dL = Lc - Ln
        theta = thn; Lc = Ln; g = gn
        log_iter(it, Lc, Ucur, its_tot, solves, fb_any, g, norm(s_))
        if maximum(abs, g) < 1e-6 || dL < 1e-9 * abs(Lc)
            println("converged (gradient or decrease below tolerance)"); break
        end
    end
    close(hist)

    # outputs
    ssa_set_theta!(sl, theta)
    mtr = misfit_stats(d, Array(Ucur))
    @printf("\nRESULT %dx%d at %.0f m: RMS speed all/grounded/floating  Weertman %.1f/%.1f/%.1f  pretrained %.1f/%.1f/%.1f  trained %.1f/%.1f/%.1f m/yr; max model trained %.0f (obs %.0f)\n",
            d.Nx, d.Ny, d.dx, best.m.rms_speed, best.m.rms_grounded, best.m.rms_floating, mpre.rms_speed, mpre.rms_grounded, mpre.rms_floating,
            mtr.rms_speed, mtr.rms_grounded, mtr.rms_floating, mtr.max_model, mtr.max_obs)
    open(joinpath(OUT, "pig_theta$(tag).txt"), "w") do io
        println(io, "# trained theta of TinyMLP{2,4,1}(shift=(0,0), scale=$(MLP_SCALE)), F0 = $F0, C_weertman = $C, sigma = $sigma, lambda = $lambda")
        println(io, "# columns: index, pretrained, trained")
        for q in eachindex(theta)
            println(io, q, " ", theta_pre[q], " ", theta[q])
        end
    end
    for (nm, U) in (("weertman", Uw), ("pretrained", Upre), ("learned", Array(Ucur)))
        write(joinpath(OUT, "speed_$(nm)$(tag).bin"), Float64.(reshape(hypot.(U[1:2:end], U[2:2:end]), d.Nx, d.Ny)))
    end
    write(joinpath(OUT, "obs_speed$(tag).bin"), Float64.(hypot.(d.vx, d.vy)))
    write(joinpath(OUT, "mask$(tag).bin"), Float64.(d.mask))
    write(joinpath(OUT, "weights$(tag).bin"), Float64.(w))
    open(joinpath(OUT, "pig_train_meta$(tag).txt"), "w") do io
        println(io, "Nx = $(d.Nx)\nNy = $(d.Ny)\ndx = $(d.dx)\nC = $C")
    end
    # F(|u|, N) curves at four N (quantiles of the grounded-cell N) for the figure
    Nq = quantile(Ncell[sel], [0.05, 0.35, 0.65, 0.95])
    open(joinpath(OUT, "pig_law_curves$(tag).csv"), "w") do io
        println(io, "N,speed,F_weertman,F_pretrained,F_trained")
        for Nq_ in Nq, sp in exp.(range(log(1.0), log(u_hi); length = 80))
            println(io, join((Nq_, sp, C * sp^(1 / 3), drag_F(mlp, theta_pre, F0, sp, Nq_), drag_F(mlp, theta, F0, sp, Nq_)), ","))
        end
    end
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && main_train()
