# V5 stage (c): model-error floor. FV Newton solve (branch K, P3 multigrid by
# default) with the TRUE Carreau–Yasuda law, from rest with pseudo-transient
# continuation; compared with the LBM truth (field and PIV planes) the
# difference is the discretisation mismatch LBM vs FV that no law fit can
# remove. Also: objective at the true law on noisy and on clean data, the
# value expected from the noise alone (N_obs / 2), and one adjoint gradient at
# the truth (Carreau variables ln ν0, ln λ, n) as a full-size check of the
# gradient path before stages (d) and (e). Writes <FV5_OUT>/c_floor.toml and
# the FV planes at the true law (piv/<plane>_twin_c<k>.bin, used by
# FV5_DATA=twin|twin_noisy in stages d/e).
#
#   FV3D_BACKEND=cuda FV5_OUT=<dir> julia --project=<env> benchmarks/fv3d/v5c_model_floor.jl
if lowercase(get(ENV, "FV3D_BACKEND", "cuda")) == "cuda"
    using CUDA
    CUDA.functional() || error("CUDA not functional")
    using CUDSS
end
isdefined(@__MODULE__, :FV5_COMMON_LOADED) || include(joinpath(@__DIR__, "v5_common.jl"))

function v5c_main()
    fv3d_banner("v5c_model_floor")
    t0 = time()
    out = fv5_outdir()
    c, _ = fv5_load_case(out)
    T = fv5_load_truth(out, c)
    P = fv5_load_piv(out, c; variant = "noisy")
    θt = fv5_theta_true(c)
    thetafun = x -> [exp(x[1]), c.nuinf, exp(x[2]), c.a, x[3]]
    chain = (x, g) -> [g[1] * exp(x[1]), g[3] * exp(x[2]), g[5]]
    F = fv5_fit_setup(c, P, fv5_carreau_visc(θt), 5; thetafun, chain, valid = θ -> true)
    xt = [log(c.nu0), log(c.lambda), c.n]
    J = fv5_fval!(F, xt)
    failures = String[]
    isfinite(J) || (push!(failures, "Newton with the true law did not converge");
                    return fv3d_verdict("v5c_model_floor", failures))
    t_newton = F.t_newton
    g = fv5_fgrad!(F, xt)
    u, v, w, _ = fv_fields(F.s.prob, F.U)
    fluid = .!T.solid
    dev(a, b) = sqrt(sum(abs2, (a .- b)[fluid])) / sqrt(sum(abs2, b[fluid]))
    du = dev(u, T.u)
    dvw = sqrt(sum(abs2, (v .- T.v)[fluid]) + sum(abs2, (w .- T.w)[fluid])) /
          sqrt(sum(abs2, T.v[fluid]) + sum(abs2, T.w[fluid]))
    dall = sqrt(sum(abs2, (u .- T.u)[fluid]) + sum(abs2, (v .- T.v)[fluid]) + sum(abs2, (w .- T.w)[fluid])) /
           sqrt(sum(abs2, T.u[fluid]) + sum(abs2, T.v[fluid]) + sum(abs2, T.w[fluid]))
    Qfv = fv5_flow_rate(u, T.solid, c.h)
    rows = fv5_plane_metrics(P, u, v, w)
    # objective on clean data = pure model error (flow-rate term against the clean Q)
    Pc = fv5_load_piv(out, c; variant = "clean")
    obj_c = fv5_objective(F.s.prob, c, Pc)
    Jc, _ = fv_objective_gradient!(obj_c, F.U, F.s.prob)
    nobs = fv5_nobs(P)
    # FV twin planes at the true law
    for (i, (pl, op)) in enumerate(zip(P.planes, P.ops))
        s = piv_apply(op, (u, v, w))
        m = P.masks[i]
        for (ci, sc) in enumerate(s)
            fv5_wbin(joinpath(out, "piv", "$(pl.name)_twin_c$(ci).bin"), sc .* m)
        end
    end
    γT = fv5_gamma_range(c, T)
    γF = fv5_gamma_range(c, (; u, v, w, solid = T.solid))
    res = Dict{String,Any}("J_true_noisy" => J, "J_true_clean" => Float64(Jc), "n_obs_fit" => nobs,
                           "J_noise_expected" => (nobs + 1) / 2,
                           "model_rms_over_sigma" => sqrt(2 * Float64(Jc) / nobs),
                           "field_rel_l2_all" => dall, "field_rel_l2_u" => du, "field_rel_l2_vw" => dvw,
                           "Q_fv_true" => Qfv, "Q_lbm" => P.meta["Q_clean"],
                           "Q_rel_diff" => (Qfv - P.meta["Q_clean"]) / P.meta["Q_clean"],
                           "grad_at_truth_x" => g, "x_names" => ["ln_nu0", "ln_lambda", "n"],
                           "planes" => rows, "t_newton_from_rest" => t_newton,
                           "gamma_lbm" => Dict("q05" => γT.lo, "q95" => γT.hi),
                           "gamma_fv" => Dict("q05" => γF.lo, "q95" => γF.hi),
                           "counters" => fv5_counters(F), "wall_seconds" => time() - t0)
    fv5_wtoml(joinpath(out, "c_floor.toml"), res)
    fv5_print_planes("v5c", rows)
    @printf("SUMMARY v5c FV(true law) vs LBM: field rel L2 all %.4f (u %.4f, v/w %.4f)  Q rel diff %+.4f  J(true, noisy) %.6e  J(true, clean) = model floor %.6e  noise-only expectation %.1f  model rms / σ_u %.3f  gamma q05-q95 LBM %.3g-%.3g FV %.3g-%.3g\n",
            dall, du, dvw, res["Q_rel_diff"], J, Jc, (nobs + 1) / 2, res["model_rms_over_sigma"],
            γT.lo, γT.hi, γF.lo, γF.hi)
    @printf("SUMMARY v5c adjoint gradient at the truth dJ/d(ln nu0, ln lambda, n) = %s  (nonzero: noise + model error)\n",
            string(round.(g; sigdigits = 4)))
    @printf("SUMMARY v5c cost: Newton from rest %.1f s, gradient %.1f s, wall %.0f s  %s\n", t_newton, F.t_grad,
            time() - t0, fv5_counter_line(F))
    flush(stdout)
    F.gmres_unconverged == 0 || push!(failures, "$(F.gmres_unconverged) GMRES solves did not converge")
    return fv3d_verdict("v5c_model_floor", failures)
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(v5c_main())
end
