# V5 stage (a): LBM truth. D3Q19 TRT + Guo force + halfway bounce-back
# (`run_gnf_duct_obstacle_3d`) on the V5 duct (periodic x, off-centre bar on the
# bottom wall, uniform body force), true Carreau–Yasuda law converted to
# lattice units (see v5_common.jl). Writes <FV5_OUT>/truth/{u,v,w,nu,is_solid}.bin
# (FV units, Float64, column-major Nx × Ny × Nz) and truth/truth.toml (case,
# lattice scales, flow rate, Mach, convergence). Later stages read the case
# from that file.
#
#   FV3D_BACKEND=cuda FV5_N=48 FV5_OUT=<dir> julia --project=<env> benchmarks/fv3d/v5a_truth_lbm.jl
# Knobs: FV5_LBM_STEPS (400000 cuda / 40000 cpu), FV5_LBM_TOL (1e-9),
# FV5_LBM_CHECK (500), FV5_MACH_MAX (0.3), plus the case knobs of v5_common.jl.
if lowercase(get(ENV, "FV3D_BACKEND", "cuda")) == "cuda"
    using CUDA
    CUDA.functional() || error("CUDA not functional")
    using CUDSS
end
isdefined(@__MODULE__, :FV5_COMMON_LOADED) || include(joinpath(@__DIR__, "v5_common.jl"))

function v5a_main()
    fv3d_banner("v5a_truth_lbm")
    out = fv5_outdir()
    cfg = fv5_config_from_env()
    c = fv5_case(cfg)
    solid, ob = fv5_obstacle(c.Nx, c.Ny, c.Nz)
    # lattice law and force (h_lat = dt_lat = 1)
    vs = c.h / c.dt; nus = c.h^2 / c.dt; acc = c.h / c.dt^2
    law = CarreauYasuda(c.nu0 / nus, c.nuinf / nus, c.lambda / c.dt, c.a, c.n)
    F = c.force / acc
    steps = fv5_int("FV5_LBM_STEPS", FV3D_GPU ? 400_000 : 40_000)
    tol = fv5_float("FV5_LBM_TOL", 1e-9)
    check = fv5_int("FV5_LBM_CHECK", 500)
    backend = FV3D_GPU ? CUDA.CUDABackend() : KrakeNN.KernelAbstractions.CPU()
    taumin = 0.5 + 3 * law.eta_inf
    ma_newt = 2.1 * c.uref * c.dt / c.h * sqrt(3)      # Newtonian square-duct peak ≈ 2.1 × mean
    @printf("   grid %d x %d x %d, h = %.5g, dt = %.5g; obstacle i %s j %s k %s; lattice: nu0 %.4g (tau %.3f) nuinf %.4g (tau floor %.3f) lambda %.4g F %.4e; Newtonian-flow Mach %.3f (shear thinning raises it); max %d steps, tol %.1e\n",
            c.Nx, c.Ny, c.Nz, c.h, c.dt, string(ob.i), string(ob.j), string(ob.k), law.eta_0, 0.5 + 3 * law.eta_0,
            law.eta_inf, taumin, law.lambda, F, ma_newt, steps, tol)
    taumin > 0.505 || return fv3d_verdict("v5a_truth_lbm",
        [@sprintf("LBM tau floor %.4f too close to 1/2: raise FV5_NUINF_RATIO or FV5_NU_LAT", taumin)])
    flush(stdout)
    fv3d_free()
    m0 = fv3d_devused()
    t = @elapsed L = run_gnf_duct_obstacle_3d(; rheology = law, Nx = c.Nx, Ny = c.Ny, Nz = c.Nz, F,
                                              max_steps = steps, mask = Array{Bool}(solid), q_w = 0.5,
                                              tol_ss = tol, check_every = check, backend, FT = Float64)
    m1 = fv3d_devused()
    if !(all(isfinite, L.ux) && all(isfinite, L.uy) && all(isfinite, L.uz))
        bad = count(!isfinite, L.ux) + count(!isfinite, L.uy) + count(!isfinite, L.uz)
        @printf("SUMMARY v5a LBM diverged: %d non-finite velocity values after %d steps (%.1f s); tau floor %.3f, Newtonian Mach %.3f\n",
                bad, L.steps, t, taumin, ma_newt)
        return fv3d_verdict("v5a_truth_lbm",
            ["LBM velocity is NaN/Inf after $(L.steps) steps: lower FV5_UREF, or raise FV5_TAU_MIN / FV5_NUINF_RATIO"])
    end
    u = L.ux .* vs; v = L.uy .* vs; w = L.uz .* vs; nu = L.nu .* nus
    d = joinpath(out, "truth")
    for (name, a) in (("u", u), ("v", v), ("w", w), ("nu", nu), ("is_solid", Float64.(solid)))
        fv5_wbin(joinpath(d, name * ".bin"), a)
    end
    Q = fv5_flow_rate(u, solid, c.h)
    ub = Q / (c.Ly * c.Lz)
    fluid = .!solid
    umax = maximum(sqrt.(u .^ 2 .+ v .^ 2 .+ w .^ 2))
    gr = fv5_gamma_range(c, (; u, v, w, solid))
    mlups = L.steps * c.Nx * c.Ny * c.Nz / t / 1e6
    mf = L.mass_flux_planes
    res = Dict{String,Any}("case" => cfg,
        "units" => Dict("h" => c.h, "dt" => c.dt, "velocity_scale" => vs, "nu_scale" => nus,
                        "acceleration_scale" => acc, "force_fv" => c.force, "force_lattice" => F,
                        "lambda_lattice" => law.lambda, "nu0_lattice" => law.eta_0),
        "obstacle" => Dict("i" => [first(ob.i), last(ob.i)], "j" => [first(ob.j), last(ob.j)],
                           "k" => [first(ob.k), last(ob.k)]),
        "lbm" => Dict("steps" => L.steps, "converged" => L.converged, "relative_change" => L.convergence,
                      "tol" => tol, "check_every" => check, "wall_seconds" => t, "MLUPS" => mlups,
                      "Mach_max" => L.u_max * sqrt(3), "u_max_lattice" => L.u_max,
                      "mass_flux_plane_spread" => (maximum(mf) - minimum(mf)) / maximum(abs.(mf)),
                      "gpu_mem_GiB" => (m1 - m0) / 2^30),
        "flow" => Dict("Q" => Q, "u_bulk" => ub, "Re_bulk" => ub * c.Ly / c.nu0, "u_max" => umax,
                       "nu_min" => minimum(nu[fluid]), "nu_max" => maximum(nu[fluid])),
        "gamma_observed" => Dict("q05" => gr.lo, "q95" => gr.hi, "min" => gr.min, "max" => gr.max,
                                 "median" => gr.median, "cells" => gr.ncells))
    fv5_wtoml(joinpath(d, "truth.toml"), res)
    @printf("SUMMARY v5a LBM %dx%dx%d  steps %d converged %s (rel change %.2e)  %.1f s (%.0f MLUPS)  Mach max %.3f  tau min %.3f  Q %.5e  u_bulk %.4f  Re_bulk %.2f  nu/nu0 in [%.3f, %.3f]  gamma observed q05-q95 %.3g-%.3g (lambda*gamma %.3g-%.3g)  GPU mem %.2f GiB\n",
            c.Nx, c.Ny, c.Nz, L.steps, L.converged, L.convergence, t, mlups, L.u_max * sqrt(3), 0.5 + 3 * minimum(L.nu[fluid]), Q, ub,
            ub * c.Ly / c.nu0, minimum(nu[fluid]) / c.nu0, maximum(nu[fluid]) / c.nu0, gr.lo, gr.hi,
            c.lambda * gr.lo, c.lambda * gr.hi, (m1 - m0) / 2^30)
    flush(stdout)
    failures = String[]
    L.converged || push!(failures, "LBM not converged in $(L.steps) steps (raise FV5_LBM_STEPS)")
    mamax = fv5_float("FV5_MACH_MAX", 0.3)
    L.u_max * sqrt(3) <= mamax ||
        push!(failures, @sprintf("LBM Mach %.3f > %.2f (lower FV5_UREF; FV5_MACH_MAX relaxes the check)",
                                 L.u_max * sqrt(3), mamax))
    return fv3d_verdict("v5a_truth_lbm", failures)
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(v5a_main())
end
