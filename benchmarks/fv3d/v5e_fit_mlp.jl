# V5 stage (e): learned law. ν(γ̇) = LearnedViscosity(TinyMLP{1,H,L}),
# monotone (stress strictly increasing in γ̇), fitted by adjoint gradients on
# the same data and objective as stage (d). No knowledge of the Carreau form.
#
# Parameterisation (see LearnedViscosity in src/platform/closures/tiny_mlp.jl):
# input u1 = (ln γ̇ − ln γ_ref) / s with γ_ref = 4 U_b / Ly taken from the
# measured flow rate (U_b = Q / (Ly Lz)) and s = FV5_MLP_SCALE (2). Start =
# Newtonian law ν = ν_start (FV5_MLP_NUSTART × ν0, default 2, the same wrong
# start as stage d): skip slope softplus(raw) + ε set equal to s, nu_ref =
# ν_start γ_ref, so ν = ν_start × exp(h(u1)) with the network part h small
# (output weights 0.01) at the start. Hidden weights from MersenneTwister
# (FV5_MLP_SEED, 7). All flat parameters are optimisation variables (x = θ).
# Optimiser: the same BFGS as stage (d), step capped at FV5_MLP_STEP (0.5) in
# every variable; same solver reuse and warm starts. A point is rejected
# (Inf, line search backtracks) when ν leaves [1e-3, 1e3] × ν_start on
# γ̇ ∈ γ_ref × [1e-4, 1e4].
#
#   FV3D_BACKEND=cuda FV5_OUT=<dir> julia --project=<env> benchmarks/fv3d/v5e_fit_mlp.jl
# Knobs: FV5_MLP_H (8 cuda / 4 cpu), FV5_MLP_L (1), FV5_MLP_EPS (1e-2: FV
# shear-rate regularisation, γ̇ ≥ ε), FV5_MAXITER (120 cuda / 10 cpu),
# FV5_DATA, FV5_RESUME=1 (from <out>/e_mlp_progress_<data>.toml, or the older
# e_mlp_progress.toml for noisy).
# Figure data (FV5_OUT/figure/): e_law_<data>.csv (true, fitted and start law on
# the plot grid of stage d), e_<data>_<plane>_{model,data,residual,clean}_c<k>.bin,
# index e_<data>.toml.
if lowercase(get(ENV, "FV3D_BACKEND", "cuda")) == "cuda"
    using CUDA
    CUDA.functional() || error("CUDA not functional")
    using CUDSS
end
isdefined(@__MODULE__, :FV5_COMMON_LOADED) || include(joinpath(@__DIR__, "v5_common.jl"))

function v5e_main()
    fv3d_banner("v5e_fit_mlp")
    t0 = time()
    out = fv5_outdir()
    c, _ = fv5_load_case(out)
    T = fv5_load_truth(out, c)
    P = fv5_load_piv(out, c)
    gr = fv5_gamma_range(c, T)
    H = fv5_int("FV5_MLP_H", FV3D_GPU ? 8 : 4)
    L = fv5_int("FV5_MLP_L", 1)
    s = fv5_float("FV5_MLP_SCALE", 2.0)
    epsl = 1e-3                                   # monotonicity margin of LearnedViscosity
    epsfv = fv5_float("FV5_MLP_EPS", 1e-2)
    ub = P.Q / (c.Ly * c.Lz)
    γref = 4 * ub / c.Ly
    shift = log(γref)
    νs = fv5_float("FV5_MLP_NUSTART", 2.0) * c.nu0
    law = LearnedViscosity(TinyMLP{1,H,L}(; shift = (shift,), scale = (s,)); nu_ref = νs * exp(shift),
                           monotone = true, epsilon = epsl)
    θ0 = law_init(law; rng = MersenneTwister(fv5_int("FV5_MLP_SEED", 7)))
    θ0[end] = KrakeNN._law_invsoftplus(s - epsl)
    nθ = length(θ0)
    eta = (θ, γ) -> KrakeNN.law_value(law, (γ,), θ)
    γwide = γref .* exp.(range(log(1e-4), log(1e4); length = 81))
    valid = θ -> all(isfinite, θ) && all(γ -> (ν = eta(θ, γ); isfinite(ν) && 1e-3νs <= ν <= 1e3νs), γwide)
    progress = joinpath(out, "e_mlp_progress_$(P.variant).toml")
    x0 = fv5_start(θ0, progress; legacy = P.variant == "noisy" ? joinpath(out, "e_mlp_progress.toml") : nothing)
    visc = FVLawViscosity(law, copy(x0); eps = epsfv)
    F = fv5_fit_setup(c, P, visc, nθ; thetafun = identity, chain = (x, g) -> g, valid)
    maxiter = fv5_int("FV5_MAXITER", FV3D_GPU ? 120 : 10)
    @printf("   data %s  TinyMLP{1,%d,%d} monotone, %d parameters  gamma_ref %.4g  scale %.3g  nu_start %.4g  FV eps %.1e  observations %d  gamma range %.3g-%.3g\n",
            P.variant, H, L, nθ, γref, s, νs, epsfv, fv5_nobs(P), gr.lo, gr.hi)
    flush(stdout)
    res, t_opt = fv5_bfgs(F, x0, progress; maxiter, max_step = fill(fv5_float("FV5_MLP_STEP", 0.5), nθ))
    θf = res.x
    le = fv5_law_error(γ -> eta(θf, γ), c, gr)
    le0 = fv5_law_error(γ -> eta(θ0, γ), c, gr)
    u, v, w, _ = fv_fields(F.s.prob, F.U)
    rows = fv5_plane_metrics(P, u, v, w)
    nobs = fv5_nobs(P)
    # law curves on the observed range (and one decade either side) for plotting
    γs = exp.(range(log(gr.lo) - log(10), log(gr.hi) + log(10); length = 49))
    θt = fv5_theta_true(c)
    dfile = joinpath(out, "d_carreau_$(P.variant).toml")
    dres = isfile(dfile) ? TOML.parsefile(dfile) : nothing
    floor = isfile(joinpath(out, "c_floor.toml")) ? TOML.parsefile(joinpath(out, "c_floor.toml")) : nothing
    out_e = Dict{String,Any}("data" => P.variant, "H" => H, "L" => L, "n_params" => nθ, "gamma_ref" => γref,
                             "scale" => s, "nu_start" => νs, "eps_fv" => epsfv, "epsilon_monotone" => epsl,
                             "theta_start" => θ0, "theta" => θf, "J" => res.f, "J_per_obs" => 2res.f / nobs,
                             "n_obs_fit" => nobs, "stop" => String(res.stop), "iterations" => res.iterations,
                             "grad_inf" => maximum(abs, res.g),
                             "law_rel_err_max" => le.max, "law_rel_err_rms" => le.rms,
                             "law_rel_err_start_max" => le0.max, "gamma_range" => [gr.lo, gr.hi],
                             "curve_gamma" => collect(γs), "curve_eta_fit" => [eta(θf, γ) for γ in γs],
                             "curve_eta_true" => [fv5_carreau_eta(θt, γ) for γ in γs],
                             "planes" => rows, "t_optimiser" => t_opt, "counters" => fv5_counters(F),
                             "wall_seconds" => time() - t0)
    dres === nothing || (out_e["carreau_law_rel_err_max"] = dres["law_rel_err_max"];
                         out_e["carreau_J"] = dres["J"])
    fv5_wtoml(joinpath(out, "e_mlp_$(P.variant).toml"), out_e)
    # figure data: law on the plot grid of stage d, planes at the fit, index
    γp = fv5_plot_grid(gr)
    ηt = [fv5_carreau_eta(θt, γ) for γ in γp]
    ηm = [eta(θf, γ) for γ in γp]
    lawcsv = fv5_wcsv(joinpath(out, "figure", "e_law_$(P.variant).csv"),
                      ["gamma", "lambda_gamma", "eta_true", "eta_mlp", "eta_mlp_start", "eta_true_over_eta0",
                       "eta_mlp_over_eta0", "in_observed_range"],
                      [γp, c.lambda .* γp, ηt, ηm, [eta(θ0, γ) for γ in γp], ηt ./ c.nu0, ηm ./ c.nu0,
                       (γp .>= gr.lo) .& (γp .<= gr.hi)])
    planes = fv5_write_planes(out, "e_$(P.variant)", P, u, v, w)
    fv5_wtoml(joinpath(out, "figure", "e_$(P.variant).toml"),
              Dict{String,Any}("stage" => "e", "data" => P.variant, "law_csv" => relpath(lawcsv, out),
                               "case" => fv5_fig_case(c, P), "gamma_range" => [gr.lo, gr.hi], "H" => H, "L" => L,
                               "n_params" => nθ, "law_rel_err_max" => le.max, "law_rel_err_rms" => le.rms,
                               "planes" => planes))
    fv5_print_planes("v5e", rows)
    @printf("SUMMARY v5e mlp data=%s TinyMLP{1,%d,%d} (%d parameters) stop=%s iterations %d  J %.6e (2J/N_obs %.3f)%s%s  eta(gamma) rel err over %.3g-%.3g: max %.2e rms %.2e (start: max %.2e)%s\n",
            P.variant, H, L, nθ, res.stop, res.iterations, res.f, 2res.f / nobs,
            floor === nothing || P.variant != "noisy" ? "" : @sprintf("  J at the true law %.6e", floor["J_true_noisy"]),
            dres === nothing ? "" : @sprintf("  J Carreau fit %.6e", dres["J"]), gr.lo, gr.hi, le.max, le.rms,
            le0.max, dres === nothing ? "" : @sprintf("  (Carreau fit: max %.2e)", dres["law_rel_err_max"]))
    @printf("SUMMARY v5e cost: optimiser %.0f s, wall %.0f s  %s\n", t_opt, time() - t0, fv5_counter_line(F))
    flush(stdout)
    failures = String[]
    res.stop === :linesearch && res.iterations == 0 && push!(failures, "no descent step accepted")
    F.gmres_unconverged == 0 || push!(failures, "$(F.gmres_unconverged) GMRES solves did not converge")
    return fv3d_verdict("v5e_fit_mlp", failures)
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(v5e_main())
end
