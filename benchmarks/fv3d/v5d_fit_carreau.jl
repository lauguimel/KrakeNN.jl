# V5 stage (d): Carreau fit by adjoint gradients. Variables x = (ln ν0, ln λ, n)
# of the Carreau–Yasuda law (a and ν∞ fixed at their true values: the model
# class is known, its parameters are not); start ν0 × 2, λ × 0.5, n = 0.8
# (FV5_START = "f_nu0,f_lambda,n" overrides). Optimiser: the dense BFGS with
# Armijo backtracking of benchmarks/validation/gnf_calibration/bfgs.jl, step
# capped at (0.3, 0.5, 0.1) per variable. One Newton solver for the whole fit
# (pattern, colouring, Krylov/multigrid structures built once; the law is
# swapped by fv_set_params!), the last converged state as warm start of the
# next solve (fallbacks: best state with PTC, rest with PTC). One adjoint per
# accepted point (re-assembly at U + refactorisation of the existing hierarchy
# + one transposed GMRES solve).
#
# After the fit: uncertainty of the recovered law with noise AND model error.
#   Ŝ   whitened residual Jacobian at the fit, central differences (6 warm solves)
#   C = (ŜᵀŜ)⁻¹  noise covariance of x (linearised)
#   r_model = (FV planes at the true law − clean LBM planes)/σ_u, flow rate
#         (Q_fv − Q_lbm)/σ_Q: the residual of the true law on noise-free data,
#         read from the files of stage c (fv5_model_residual)
#   δx_model = −C Ŝᵀ r_model: the Gauss–Newton shift that r_model alone gives
#         the fit, i.e. the expected bias of x̂ (x̂ ≈ x_true + δx_model + noise)
#   along ln η(γ̇): σ(γ̇) = sqrt(gᵀCg), b(γ̇) = gᵀ δx_model, g = ∂ln η/∂x at x̂;
#   err(γ̇) = ln η_fit − ln η_true. Simultaneous 95 % band k σ with
#   k = sqrt(χ²_p(0.95)) (= 2.80 for p = 3: covers every linear combination of
#   x at once, hence every point of the γ̇ grid jointly; 1.96 is pointwise).
# Gates over the observed range (64 points, 5–95 % γ̇ quantiles of the fitted
# plane footprints); they are printed and saved, they do not set the exit code:
#   noise-only      |err| ≤ k σ        (expected to fail when model error matters)
#   bias-corrected  |err − b| ≤ k σ    (what is left after the predicted model bias is noise)
#   combined        |err| ≤ |b| + k σ  (the law is inside its noise + model-error band)
#   twin            FV5_DATA=twin|twin_noisy: no model error by construction,
#                   the noise-only gate is the gate
# Parameter-space checks: χ² = (x̂ − x_t)ᵀŜᵀŜ(x̂ − x_t), and with x_t + δx_model,
# against χ²_p(0.95); per-parameter 95 % marginal half-width (relative), a
# parameter counts as identifiable when it is ≤ FV5_PARAM_TOL (0.1).
# Reference only: the inflated white-noise band σ_eff² = σ_u² + σ_model² (option a).
#
# Figure data (FV5_OUT/figure/): d_law_<data>.csv, d_gamma_hist_<data>.csv,
# d_<data>_<plane>_{model,data,residual,clean}_c<k>.bin, index d_<data>.toml.
#
#   FV3D_BACKEND=cuda FV5_OUT=<dir> julia --project=<env> benchmarks/fv3d/v5d_fit_carreau.jl
# Knobs: FV5_DATA (noisy | clean | twin | twin_noisy), FV5_MAXITER (40 cuda /
# 15 cpu; 0 with FV5_RESUME=1 = no optimisation, post-fit analysis at the last
# point), FV5_RESUME=1 (start from <out>/d_carreau_progress_<data>.toml, or the
# older d_carreau_progress.toml for noisy), FV5_FD_STEP (1e-3), FV5_IDENT=0
# skips the identifiability solves, FV5_PARAM_TOL (0.1).
if lowercase(get(ENV, "FV3D_BACKEND", "cuda")) == "cuda"
    using CUDA
    CUDA.functional() || error("CUDA not functional")
    using CUDSS
end
isdefined(@__MODULE__, :FV5_COMMON_LOADED) || include(joinpath(@__DIR__, "v5_common.jl"))

const V5D_KP = 1.959964                     # two-sided 95 % normal quantile (pointwise)

"""95 % quantile of χ²_p (table for p ≤ 5, Wilson–Hilferty beyond)."""
v5d_chi2_95(p) = p <= 5 ? (3.841459, 5.991465, 7.814728, 9.487729, 11.070498)[p] :
                 p * (1 - 2 / (9p) + 1.644854 * sqrt(2 / (9p)))^3

"""Whitened residual Jacobian Ŝ at x by central differences (2 warm solves per variable)."""
function v5d_jacobian(F, P, x)
    δ = fv5_float("FV5_FD_STEP", 1e-3)
    r0 = fv5_residuals(F, P)
    S = zeros(length(r0), length(x))
    for j in eachindex(x)
        rp = rm = nothing
        for sgn in (1, -1)
            xs = copy(x); xs[j] += sgn * δ
            isfinite(fv5_fval!(F, xs)) || error("identifiability solve failed at variable $j")
            sgn == 1 ? (rp = fv5_residuals(F, P)) : (rm = fv5_residuals(F, P))
        end
        S[:, j] = (rp .- rm) ./ (2δ)
    end
    fv5_fval!(F, x)                       # back to the optimum
    return S
end

"""∂ ln η(γ̇)/∂x of the Carreau law at x (central differences, law evaluation only)."""
function v5d_lneta_grad(F, x, γ; δ = 1e-6)
    e(j) = δ .* ((1:length(x)) .== j)
    f(xx) = log(fv5_carreau_eta(F.thetafun(xx), γ))
    return [(f(x .+ e(j)) - f(x .- e(j))) / 2δ for j in eachindex(x)]
end

"""Noise std σ, model bias b and error ln η_fit − ln η_true of ln η on the grid γs."""
function v5d_lneta_bands(F, x, C, bias_x, θt, γs)
    s = zeros(length(γs)); b = zeros(length(γs)); err = zeros(length(γs))
    θf = F.thetafun(x)
    for (i, γ) in enumerate(γs)
        g = v5d_lneta_grad(F, x, γ)
        s[i] = sqrt(max(dot(g, C * g), 0.0))
        b[i] = dot(g, bias_x)
        err[i] = log(fv5_carreau_eta(θf, γ)) - log(fv5_carreau_eta(θt, γ))
    end
    return (; s, b, err)
end

"""Uncertainty of x and of ln η(γ̇) with noise and (when r_model is given) model error."""
function v5d_uncertainty(F, S, x, xt, θt, gr, rmodel, names, model_rms)
    p = length(x)
    A = Symmetric(S' * S)
    C = Matrix(inv(A))
    sd = sqrt.(diag(C))
    corr = C ./ (sd * sd')
    E = eigen(A)
    bias_x = rmodel === nothing ? zeros(p) : -(C * (S' * rmodel))
    q95 = v5d_chi2_95(p)
    ks = sqrt(q95)
    γs = fv5_gamma_grid(gr)
    bd = v5d_lneta_bands(F, x, C, bias_x, θt, γs)
    infl = isfinite(model_rms) ? sqrt(1 + model_rms^2) : NaN
    within(lhs, rhs) = count(lhs .<= rhs)
    inside = Dict{String,Int}("noise" => within(abs.(bd.err), ks .* bd.s),
                              "noise_pointwise" => within(abs.(bd.err), V5D_KP .* bd.s),
                              "bias_corrected" => within(abs.(bd.err .- bd.b), ks .* bd.s),
                              "combined" => within(abs.(bd.err), abs.(bd.b) .+ ks .* bd.s))
    isfinite(infl) && (inside["inflated"] = within(abs.(bd.err), (ks * infl) .* bd.s))
    dx = x .- xt
    dxb = dx .- bias_x
    # 95 % marginal half-width, relative: factor exp(1.96 sd) − 1 for log variables, 1.96 sd/|x| otherwise
    rel95 = [startswith(names[j], "ln_") ? exp(V5D_KP * sd[j]) - 1 : V5D_KP * sd[j] / abs(x[j]) for j in 1:p]
    return (; C, sd, corr, evals = E.values, evecs = E.vectors, bias_x, ks, q95, γs, bd, infl, inside,
            n = length(γs), q_noise = dot(dx, A * dx), q_bias = dot(dxb, A * dxb), rel95, cond = cond(Matrix(A)),
            rmodel_half_sq = rmodel === nothing ? NaN : sum(abs2, rmodel) / 2)
end

fv5_pf(b) = b ? "PASS" : "FAIL"

"""Figure files of stage d (law curves with bands, γ̇ histogram, planes) and their index."""
function v5d_write_figure(out, c, P, gr, F, x, θt, id, gates, u, v, w, T)
    dir = joinpath(out, "figure")
    tag = P.variant
    γp = fv5_plot_grid(gr)
    θf = F.thetafun(x)
    ηt = [fv5_carreau_eta(θt, γ) for γ in γp]
    ηf = [fv5_carreau_eta(θf, γ) for γ in γp]
    if id === nothing
        s = fill(NaN, length(γp)); b = fill(NaN, length(γp)); ks = NaN
    else
        bd = v5d_lneta_bands(F, x, id.C, id.bias_x, θt, γp)
        s = bd.s; b = bd.b; ks = id.ks
    end
    e0 = c.nu0
    lawcsv = fv5_wcsv(joinpath(dir, "d_law_$(tag).csv"),
                      ["gamma", "lambda_gamma", "eta_true", "eta_fit", "eta_true_over_eta0", "eta_fit_over_eta0",
                       "sigma_lneta", "bias_lneta", "noise_lo", "noise_hi", "total_lo", "total_hi",
                       "corrected_lo", "corrected_hi", "in_observed_range"],
                      [γp, c.lambda .* γp, ηt, ηf, ηt ./ e0, ηf ./ e0, s, b,
                       ηf .* exp.(-ks .* s) ./ e0, ηf .* exp.(ks .* s) ./ e0,
                       ηf .* exp.(-(abs.(b) .+ ks .* s)) ./ e0, ηf .* exp.(abs.(b) .+ ks .* s) ./ e0,
                       ηf .* exp.(-b .- ks .* s) ./ e0, ηf .* exp.(-b .+ ks .* s) ./ e0,
                       (γp .>= gr.lo) .& (γp .<= gr.hi)])
    hf = fv5_gamma_hist(c, (; u, v, w, solid = T.solid), gr)
    hl = fv5_gamma_hist(c, T, gr)
    histcsv = fv5_wcsv(joinpath(dir, "d_gamma_hist_$(tag).csv"),
                       ["gamma_lo", "gamma_hi", "lambda_gamma_lo", "lambda_gamma_hi", "count_footprint_fv",
                        "count_fluid_fv", "count_footprint_lbm"],
                       [hf.edges[1:(end - 1)], hf.edges[2:end], c.lambda .* hf.edges[1:(end - 1)],
                        c.lambda .* hf.edges[2:end], hf.count_footprint, hf.count_fluid, hl.count_footprint])
    planes = fv5_write_planes(out, "d_$(tag)", P, u, v, w)
    idx = Dict{String,Any}("stage" => "d", "data" => tag, "law_csv" => relpath(lawcsv, out),
                           "gamma_hist_csv" => relpath(histcsv, out), "case" => fv5_fig_case(c, P),
                           "gamma_range" => [gr.lo, gr.hi], "k_simultaneous" => ks, "k_pointwise" => V5D_KP,
                           "theta_fit" => θf, "theta_true" => θt,
                           "hist" => Dict("n_footprint_fv" => hf.n_footprint, "n_fluid_fv" => hf.n_fluid,
                                          "below_footprint_fv" => hf.below_footprint,
                                          "above_footprint_fv" => hf.above_footprint,
                                          "below_fluid_fv" => hf.below_fluid, "above_fluid_fv" => hf.above_fluid,
                                          "fields" => "FV state at the fit (fv), LBM truth (lbm)"),
                           "bands" => "ln eta: noise = fit ± k sigma; total = fit ± (|bias| + k sigma); corrected = fit − bias ± k sigma (confidence band of the true law); eta columns divided by eta0 = nu0 true, lambda_gamma uses the true lambda",
                           "gates" => gates, "planes" => planes)
    fv5_wtoml(joinpath(dir, "d_$(tag).toml"), idx)
end

function v5d_main()
    fv3d_banner("v5d_fit_carreau")
    t0 = time()
    out = fv5_outdir()
    c, _ = fv5_load_case(out)
    T = fv5_load_truth(out, c)
    P = fv5_load_piv(out, c)
    twin = P.variant in ("twin", "twin_noisy")
    gr = fv5_gamma_range(c, T)
    floor = isfile(joinpath(out, "c_floor.toml")) ? TOML.parsefile(joinpath(out, "c_floor.toml")) : nothing
    names = ("ln_nu0", "ln_lambda", "n")
    thetafun = x -> [exp(x[1]), c.nuinf, exp(x[2]), c.a, x[3]]
    chain = (x, g) -> [g[1] * exp(x[1]), g[3] * exp(x[2]), g[5]]
    valid = θ -> 0.05 < θ[5] < 1.5 && θ[1] > θ[2] && all(isfinite, θ)
    st = parse.(Float64, split(fv5_env("FV5_START", "2,0.5,0.8"), ','))
    x0 = [log(c.nu0 * st[1]), log(c.lambda * st[2]), st[3]]
    progress = joinpath(out, "d_carreau_progress_$(P.variant).toml")
    legacy = P.variant == "noisy" ? joinpath(out, "d_carreau_progress.toml") : nothing
    x0 = fv5_start(x0, progress; legacy)
    resfile = joinpath(out, "d_carreau_$(P.variant).toml")
    previous = fv5_env("FV5_RESUME", "0") == "1" && isfile(resfile) ? TOML.parsefile(resfile) : nothing
    F = fv5_fit_setup(c, P, fv5_carreau_visc(thetafun(x0)), 5; thetafun, chain, valid)
    xt = [log(c.nu0), log(c.lambda), c.n]
    maxiter = fv5_int("FV5_MAXITER", FV3D_GPU ? 40 : 15)
    @printf("   data %s  start x = %s  true x = %s  observations %d  sigma_u %.3e  gamma range %.3g-%.3g  maxiter %d\n",
            P.variant, string(round.(x0; sigdigits = 5)), string(round.(xt; sigdigits = 5)), fv5_nobs(P),
            P.sigma_u, gr.lo, gr.hi, maxiter)
    flush(stdout)
    res, t_opt = fv5_bfgs(F, x0, progress; maxiter, max_step = [0.3, 0.5, 0.1], names)
    x = res.x
    θf = thetafun(x)
    θt = fv5_theta_true(c)
    le = fv5_law_error(γ -> fv5_carreau_eta(θf, γ), c, gr)
    u, v, w, _ = fv_fields(F.s.prob, F.U)
    rows = fv5_plane_metrics(P, u, v, w)
    nobs = fv5_nobs(P)
    counters_fit = fv5_counters(F)
    out_d = Dict{String,Any}("data" => P.variant, "x_names" => collect(names), "x_start" => x0, "x" => x,
                             "x_true" => xt, "theta" => θf, "theta_true" => θt, "J" => res.f,
                             "J_per_obs" => 2res.f / nobs, "n_obs_fit" => nobs, "stop" => String(res.stop),
                             "iterations" => res.iterations, "grad_x" => res.g,
                             "rel_err_nu0" => θf[1] / θt[1] - 1, "rel_err_lambda" => θf[3] / θt[3] - 1,
                             "rel_err_n" => θf[5] / θt[5] - 1,
                             "law_rel_err_max" => le.max, "law_rel_err_rms" => le.rms,
                             "gamma_range" => [gr.lo, gr.hi], "planes" => rows, "t_optimiser" => t_opt,
                             "counters_fit" => counters_fit)
    floor === nothing || (out_d["J_true_same_data"] = P.variant == "noisy" ? floor["J_true_noisy"] : NaN)
    if previous !== nothing
        out_d["previous_run"] = haskey(previous, "previous_run") ? previous["previous_run"] :
            Dict{String,Any}(k => previous[k] for k in ("iterations", "stop", "J", "x_start", "t_optimiser",
                                                        "counters_fit", "wall_seconds") if haskey(previous, k))
    end
    # model discrepancy from the files of stage c (none in the twin variants, by construction)
    rmodel = nothing
    model_rms = NaN
    if !twin
        have = floor !== nothing && all(isfile(joinpath(out, "piv", "$(pl.name)_twin_c1.bin")) for pl in P.planes if pl.fit)
        if have
            rmodel = fv5_model_residual(out, P, floor)
            model_rms = floor["model_rms_over_sigma"]
        else
            println("   WARNING: c_floor.toml or piv/*_twin_c*.bin missing (run stage c): no model-bias band")
        end
    end
    id = nothing
    gates = Dict{String,Any}()
    if fv5_env("FV5_IDENT", "1") == "1"
        t_id = @elapsed S = v5d_jacobian(F, P, x)
        rmodel === nothing || length(rmodel) == size(S, 1) ||
            error("r_model has $(length(rmodel)) entries, the residual vector $(size(S, 1))")
        id = v5d_uncertainty(F, S, x, xt, θt, gr, rmodel, names, model_rms)
        bd = id.bd
        pass = k -> id.inside[k] == id.n
        ptol = fv5_float("FV5_PARAM_TOL", 0.1)
        ident_param = [id.rel95[j] <= ptol for j in eachindex(x)]
        if twin
            gates["twin_noise_only"] = pass("noise")
            law_ok = gates["twin_noise_only"]
        else
            gates["noise_only"] = pass("noise")
            if rmodel !== nothing
                gates["bias_corrected"] = pass("bias_corrected")
                gates["combined"] = pass("combined")
                isfinite(id.infl) && (gates["inflated_reference"] = pass("inflated"))
                law_ok = gates["combined"] && gates["bias_corrected"]
            else
                law_ok = false
            end
        end
        claim = law_ok && !ident_param[1] && !ident_param[2]
        gates["claim_supported"] = claim
        tot = abs.(bd.b) .+ id.ks .* bd.s
        out_d["ident"] = Dict{String,Any}(
            "std_x" => id.sd, "cov_x" => [id.C[i, :] for i in 1:size(id.C, 1)],
            "corr_x" => [id.corr[i, :] for i in 1:size(id.corr, 1)], "cond_StS" => id.cond,
            "eig_StS" => id.evals, "weakest_direction_x" => id.evecs[:, 1],
            "rel95_param" => id.rel95, "param_identifiable" => ident_param, "param_tol" => ptol,
            "k_simultaneous" => id.ks, "k_pointwise" => V5D_KP, "chi2_95" => id.q95, "points" => id.n,
            "model_bias_x" => id.bias_x, "x_minus_true" => x .- xt,
            "chi2_noise_only" => id.q_noise, "chi2_bias_corrected" => id.q_bias,
            "half_rmodel_sq" => id.rmodel_half_sq, "model_rms_over_sigma" => model_rms,
            "inflation_factor" => id.infl,
            "gamma" => id.γs, "sigma_lneta" => bd.s, "bias_lneta" => bd.b, "err_lneta" => bd.err,
            "band_noise_min" => id.ks * minimum(bd.s), "band_noise_max" => id.ks * maximum(bd.s),
            "bias_abs_max" => maximum(abs, bd.b), "band_total_max" => maximum(tot),
            "err_abs_max" => maximum(abs, bd.err), "err_minus_bias_abs_max" => maximum(abs, bd.err .- bd.b),
            "inside" => id.inside, "gates" => gates, "seconds" => t_id)
        @printf("SUMMARY v5d identifiability: std(ln nu0, ln lambda, n) = %s  corr(ln nu0, ln lambda) %+.4f  cond(S'S) %.2e  weakest direction of S'S %s (eigenvalue %.3e)  (%.0f s)\n",
                string(round.(id.sd; sigdigits = 3)), id.corr[1, 2], id.cond,
                string(round.(id.evecs[:, 1]; sigdigits = 3)), id.evals[1], t_id)
        @printf("SUMMARY v5d parameters, 95%% marginal half-width: nu0 x/÷ %.3g (rel %.2e)  lambda x/÷ %.3g (rel %.2e)  n ± %.3g (rel %.2e)  identifiable at %.0f %%: nu0 %s, lambda %s, n %s\n",
                1 + id.rel95[1], id.rel95[1], 1 + id.rel95[2], id.rel95[2], V5D_KP * id.sd[3], id.rel95[3],
                100ptol, ident_param[1] ? "yes" : "no", ident_param[2] ? "yes" : "no", ident_param[3] ? "yes" : "no")
        if rmodel !== nothing
            @printf("SUMMARY v5d model bias (Gauss-Newton shift of r_model = FV(true law) - LBM clean, 1/2|r_model|^2 = %.4e, stage c J(true, clean) %.4e): predicted dx = %s  observed x - x_true = %s  noise std %s\n",
                    id.rmodel_half_sq, floor["J_true_clean"], string(round.(id.bias_x; sigdigits = 3)),
                    string(round.(x .- xt; sigdigits = 3)), string(round.(id.sd; sigdigits = 3)))
        end
        @printf("SUMMARY v5d parameter-space chi2 (%d dof, 95%% threshold %.2f): noise only (x - x_true) %.3g  bias-corrected (x - x_true - dx_model) %.3g\n",
                length(x), id.q95, id.q_noise, id.q_bias)
        @printf("SUMMARY v5d ln eta over gamma %.3g-%.3g (%d points, simultaneous k = %.3f, pointwise 1.96): noise band k sigma %.2e-%.2e  |model bias| max %.2e  total band max %.2e (eta within x/÷ %.4f)  |err| max %.2e  |err - bias| max %.2e\n",
                gr.lo, gr.hi, id.n, id.ks, id.ks * minimum(bd.s), id.ks * maximum(bd.s), maximum(abs, bd.b),
                maximum(tot), exp(maximum(tot)), maximum(abs, bd.err), maximum(abs, bd.err .- bd.b))
        if twin
            @printf("GATE v5d twin (data = FV at the true law%s, no model error by construction): law inside the noise-only band: %s (%d/%d points)\n",
                    P.variant == "twin_noisy" ? " + noise" : ", no noise", fv5_pf(gates["twin_noise_only"]),
                    id.inside["noise"], id.n)
        else
            @printf("GATE v5d noise-only band (model error ignored): %s (%d/%d points; pointwise 1.96: %d/%d)\n",
                    fv5_pf(gates["noise_only"]), id.inside["noise"], id.n, id.inside["noise_pointwise"], id.n)
            if rmodel !== nothing
                @printf("GATE v5d bias-corrected (|err - model bias| inside the noise band): %s (%d/%d points)\n",
                        fv5_pf(gates["bias_corrected"]), id.inside["bias_corrected"], id.n)
                @printf("GATE v5d combined (|err| inside |model bias| + noise band): %s (%d/%d points)\n",
                        fv5_pf(gates["combined"]), id.inside["combined"], id.n)
                isfinite(id.infl) &&
                    @printf("INFO v5d inflated white-noise band (sigma_eff^2 = sigma_u^2 + sigma_model^2, factor %.4f): %d/%d points (reference only: the model error is systematic, its projection on x does not average out like white noise)\n",
                            id.infl, id.inside["inflated"], id.n)
            end
        end
        @printf("CLAIM v5d \"eta(gamma) recovered within its noise + model-error uncertainty over the observed range, nu0 and lambda not individually identifiable\": %s (law gates %s; nu0 identifiable %s, lambda identifiable %s)\n",
                claim ? "SUPPORTED" : "NOT SUPPORTED", law_ok ? "pass" : "fail",
                ident_param[1] ? "yes" : "no", ident_param[2] ? "yes" : "no")
    end
    out_d["counters"] = fv5_counters(F)
    out_d["wall_seconds"] = time() - t0
    fv5_wtoml(resfile, out_d)
    v5d_write_figure(out, c, P, gr, F, x, θt, id, gates, u, v, w, T)
    fv5_print_planes("v5d", rows)
    @printf("SUMMARY v5d carreau data=%s stop=%s iterations %d  J %.6e (2J/N_obs %.3f%s)  nu0 %.5g (rel err %+.2e)  lambda %.5g (%+.2e)  n %.5g (%+.2e)  eta(gamma) rel err over %.3g-%.3g: max %.2e rms %.2e\n",
            P.variant, res.stop, res.iterations, res.f, 2res.f / nobs,
            floor === nothing || P.variant != "noisy" ? "" : @sprintf(", J at the true law %.6e", floor["J_true_noisy"]),
            θf[1], θf[1] / θt[1] - 1, θf[3], θf[3] / θt[3] - 1, θf[5], θf[5] / θt[5] - 1, gr.lo, gr.hi,
            le.max, le.rms)
    @printf("SUMMARY v5d cost: optimiser %.0f s, wall %.0f s  %s\n", t_opt, time() - t0, fv5_counter_line(F))
    flush(stdout)
    failures = String[]
    res.stop === :linesearch && res.iterations == 0 && push!(failures, "no descent step accepted")
    F.gmres_unconverged == 0 || push!(failures, "$(F.gmres_unconverged) GMRES solves did not converge")
    return fv3d_verdict("v5d_fit_carreau", failures)
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(v5d_main())
end
