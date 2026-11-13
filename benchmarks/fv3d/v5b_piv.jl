# V5 stage (b): synthetic PIV from the LBM truth of stage (a). The four planes
# of `fv5_planes` (three fitted, one held out) are sampled with `piv_operator`
# on the FV grid (the LBM nodes are the FV cell centres, so the truth is used
# as is); vectors whose footprint touches the obstacle are masked; Gaussian
# noise of standard deviation σ_u = FV5_NOISE (0.02) × max |u_truth| is added
# to every measured component, and FV5_SIGMA_Q (0.01) × Q to the flow rate
# through slab i = 1, from MersenneTwister(FV5_SEED) (default 20261110), planes
# in order, components in order, column-major samples, flow rate last.
# Writes <FV5_OUT>/piv/<plane>_{clean,noisy}_c<k>.bin, <plane>_mask.bin, piv.toml.
#
#   FV3D_BACKEND=cpu FV5_OUT=<dir> julia --project=<env> benchmarks/fv3d/v5b_piv.jl   (host only, seconds)
if lowercase(get(ENV, "FV3D_BACKEND", "cuda")) == "cuda"
    using CUDA
    CUDA.functional() || error("CUDA not functional")
    using CUDSS
end
isdefined(@__MODULE__, :FV5_COMMON_LOADED) || include(joinpath(@__DIR__, "v5_common.jl"))

function v5b_main()
    fv3d_banner("v5b_piv")
    t0 = time()
    out = fv5_outdir()
    c, ttoml = fv5_load_case(out)
    T = fv5_load_truth(out, c)
    noise = fv5_float("FV5_NOISE", 0.02)
    relQ = fv5_float("FV5_SIGMA_Q", 0.01)
    seed = fv5_int("FV5_SEED", 20261110)
    rng = MersenneTwister(seed)
    umax = maximum(sqrt.(T.u .^ 2 .+ T.v .^ 2 .+ T.w .^ 2))
    σ = noise * umax
    planes = fv5_planes(c)
    ops = fv5_host_ops(c, planes)
    d = joinpath(out, "piv")
    rows = Dict{String,Any}[]
    nobs = 0
    for (pl, op) in zip(planes, ops)
        m, _ = fv5_mask_footprint(op, T.solid)
        s = piv_apply(op, (T.u, T.v, T.w))
        fv5_wbin(joinpath(d, "$(pl.name)_mask.bin"), m)
        rms = Float64[]
        for (ci, sc) in enumerate(s)
            clean = sc .* m
            noisy = (sc .+ σ .* randn(rng, size(sc))) .* m
            fv5_wbin(joinpath(d, "$(pl.name)_clean_c$(ci).bin"), clean)
            fv5_wbin(joinpath(d, "$(pl.name)_noisy_c$(ci).bin"), noisy)
            push!(rms, sqrt(mean(sc[m .> 0] .^ 2)))
        end
        nv = count(>(0), m)
        pl.fit && (nobs += nv * length(s))
        push!(rows, Dict("name" => pl.name, "fit" => pl.fit, "normal" => String(pl.plane.normal),
                         "position" => pl.plane.position, "stereo" => pl.plane.stereo,
                         "samples" => [size(op.Wa, 1), size(op.Wb, 1)], "measured" => nv,
                         "masked" => length(m) - nv, "component_rms" => rms,
                         "window" => collect(pl.plane.window), "sheet_sigma" => pl.plane.thickness))
        @printf("SUMMARY v5b plane %-8s %s normal %s at %.4f  %d x %d vectors, %d measured, %d masked  %dC  component rms / σ_u %s\n",
                pl.name, pl.fit ? "fit " : "HELD", pl.plane.normal, pl.plane.position, size(op.Wa, 1),
                size(op.Wb, 1), nv, length(m) - nv, length(s), string(round.(rms ./ σ; sigdigits = 3)))
    end
    Qc = fv5_flow_rate(T.u, T.solid, c.h)
    σQ = relQ * abs(Qc)
    Qn = Qc + σQ * randn(rng)
    meta = Dict{String,Any}("sigma_u" => σ, "noise_rel_umax" => noise, "u_max" => umax, "seed" => seed,
                            "Q_clean" => Qc, "Q_noisy" => Qn, "sigma_Q" => σQ, "sigma_Q_rel" => relQ,
                            "flow_rate_slab" => 1, "planes" => rows, "n_obs_fit" => nobs,
                            "gamma_observed" => ttoml["gamma_observed"])
    fv5_wtoml(joinpath(d, "piv.toml"), meta)
    @printf("SUMMARY v5b sigma_u %.4e (%.1f %% of max |u| %.4f)  fitted observations %d (+1 flow rate)  Q clean %.6e noisy %.6e (sigma_Q %.2e)  %.1f s\n",
            σ, 100noise, umax, nobs, Qc, Qn, σQ, time() - t0)
    return fv3d_verdict("v5b_piv", nobs > 0 ? String[] : ["no measured vector"])
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(v5b_main())
end
