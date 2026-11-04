#!/usr/bin/env julia
# L1 — planar Poiseuille Oldroyd-B (full pipeline), runner
#
# Invocation:
#   julia --project=. bench/viscoelastic_validation/L1_poiseuille_oldb/run.jl
#
# Writes results/L1_run_<timestamp>.json with the produced fields + analytic
# reference + per-quantity errors, ready for `compare.jl`.
#
# No assertions live here — `compare.jl` consumes the dump and rules
# PASS/FAIL. This separation lets a developer re-run `compare.jl` against
# an existing dump with tweaked thresholds without re-running the simulation.

using Dates
using JSON3
using KernelAbstractions
using Printf
using CairoMakie

using Kraken

const HERE = @__DIR__

"""
    wall_gamma_dot_factor(Ny)

Coefficient `c` such that `gamma_dot_wall = c * Fx_body / nu_total` for a
steady Bird-AH Poiseuille profile on the LBM half-cell grid
(walls at j=0.5 and j=Ny+0.5). Derived from the corrected HWBB-aware
stencil at j=1 applied to the analytic parabola
`u(j) = Fx/(2*nu) * (j-0.5)*(Ny+0.5-j)`:

    gamma_dot[1] = (-3*u[1] + 4*u[2] - u[3]) / 2
                 = Fx/(2*nu) * [ -3*(0.5)(Ny-0.5) + 4*(1.5)(Ny-1.5) - (2.5)(Ny-2.5) ] / 2
                 = Fx * (Ny - 1) / (2 * nu_total)

So `c = (Ny - 1) / 2`. Wi = lambda * gamma_dot_wall = lambda * c * Fx / nu_total.
"""
wall_gamma_dot_factor(Ny) = (Ny - 1) / 2

"""
    Fx_for_Wi(Wi_target, lambda, nu_total, Ny)

Solve for the body-force amplitude that produces a target wall-cell
Weissenberg number `Wi = lambda * gamma_dot_wall`.

`gamma_dot_wall = c * Fx / nu_total`, hence `Fx = Wi * nu_total / (lambda * c)`.
"""
function Fx_for_Wi(Wi_target, lambda, nu_total, Ny)
    c = wall_gamma_dot_factor(Ny)
    return Wi_target * nu_total / (lambda * c)
end

"""
    parse_kwargs()

Accept overrides for `Wi_target`, `Fx_body`, `max_steps`, `polymer_substeps`
from CLI args of the form `key=value` or environment variables
`L1_WI_TARGET`, `L1_FX_BODY`, `L1_MAX_STEPS`, `L1_POLYMER_SUBSTEPS`.
CLI overrides ENV.
"""
function parse_kwargs()
    opts = Dict{Symbol,Any}()
    if haskey(ENV, "L1_WI_TARGET")
        opts[:Wi_target] = parse(Float64, ENV["L1_WI_TARGET"])
    end
    if haskey(ENV, "L1_FX_BODY")
        opts[:Fx_body] = parse(Float64, ENV["L1_FX_BODY"])
    end
    if haskey(ENV, "L1_MAX_STEPS")
        opts[:max_steps] = parse(Int, ENV["L1_MAX_STEPS"])
    end
    if haskey(ENV, "L1_POLYMER_SUBSTEPS")
        v = ENV["L1_POLYMER_SUBSTEPS"]
        opts[:polymer_substeps] = v == "auto" ? :auto : parse(Int, v)
    end
    for a in ARGS
        m = match(r"^([A-Za-z_]+)=(.+)$", a)
        m === nothing && continue
        k, v = Symbol(m.captures[1]), m.captures[2]
        if k === :Wi_target
            opts[:Wi_target] = parse(Float64, v)
        elseif k === :Fx_body
            opts[:Fx_body] = parse(Float64, v)
        elseif k === :max_steps
            opts[:max_steps] = parse(Int, v)
        elseif k === :polymer_substeps
            opts[:polymer_substeps] = v == "auto" ? :auto : parse(Int, v)
        elseif k === :tag
            opts[:tag] = v
        end
    end
    return opts
end

function config(::Type{T}; overrides::Dict{Symbol,Any}=Dict{Symbol,Any}()) where {T}
    Nx = 8
    Ny = 32
    nu_s = T(0.04)
    nu_p = T(0.06)
    nu_total = nu_s + nu_p
    lambda = T(5.0)
    # Default Fx (matches M39 baseline; Wi ≈ 3.875e-3 with corrected stencil)
    Fx_default = T(5e-6)
    Fx_body = Fx_default
    polymer_substeps = :auto
    max_steps = 10_000
    if haskey(overrides, :Wi_target)
        Fx_body = T(Fx_for_Wi(overrides[:Wi_target], lambda, nu_total, Ny))
    end
    if haskey(overrides, :Fx_body)
        Fx_body = T(overrides[:Fx_body])
    end
    if haskey(overrides, :max_steps)
        max_steps = overrides[:max_steps]
    end
    if haskey(overrides, :polymer_substeps)
        polymer_substeps = overrides[:polymer_substeps]
    end
    return (;
        Nx=Nx,
        Ny=Ny,
        nu_s=nu_s,
        nu_p=nu_p,
        Fx_body=Fx_body,
        lambda=lambda,
        bsd_fraction=T(1.0),
        polymer_substeps=polymer_substeps,
        max_steps=max_steps,
        T=T,
    )
end

"""
    analytic_oldroydb_poiseuille(cfg)

Steady Bird-Armstrong-Hassager Vol 1 §3.4 analytic reference on the LBM
half-cell grid. y-coordinate of cell j is `(j - 0.5)` (LU); walls live at
j = 0 and j = Ny + 1 (half-cell beyond first/last interior). Half-channel
height in LU is `H = Ny / 2`; the centreline of the interior is at
`y_c = Ny / 2 + 0.5 - 0.5 = Ny / 2` (between cells Ny/2 and Ny/2 + 1).

Returns vectors of length Ny: `gamma_dot`, `u_x`, `tau_xy`, `tau_xx`,
plus the centreline peak value `u_peak_analytic`.
"""
function analytic_oldroydb_poiseuille(cfg)
    Tt = cfg.T
    Ny = cfg.Ny
    nu_total = cfg.nu_s + cfg.nu_p
    Fx = cfg.Fx_body
    lam = cfg.lambda

    # Mirror the LBM driver's reference (matches `_logfv_lbm_poiseuille_reference`)
    u_ref = [Fx / (2 * nu_total) * (Tt(j) - Tt(0.5)) * (Tt(Ny) + Tt(0.5) - Tt(j)) for j in 1:Ny]

    # Centreline (peak) value: maximum of the discrete profile
    u_peak_analytic = maximum(u_ref)

    # gamma_dot from d u_ref / dj  (Newtonian-equivalent shear rate)
    gamma_dot = zeros(Tt, Ny)
    for j in 1:Ny
        # central difference, BC: u(j=0) = 0 (HWBB midpoint), u(j=Ny+1) = 0
        u_jp1 = j == Ny ? zero(Tt) : u_ref[j + 1]
        u_jm1 = j == 1 ? zero(Tt) : u_ref[j - 1]
        gamma_dot[j] = (u_jp1 - u_jm1) / (2 * one(Tt))
    end

    tau_xy = cfg.nu_p .* gamma_dot
    tau_xx = 2 .* lam .* cfg.nu_p .* gamma_dot .^ 2
    tau_yy = zeros(Tt, Ny)

    return (;
        gamma_dot,
        u_x=u_ref,
        tau_xy,
        tau_xx,
        tau_yy,
        u_peak_analytic,
        nu_total,
    )
end

function profile_yaverage(arr::AbstractMatrix)
    Nx, Ny = size(arr)
    return [sum(@view arr[:, j]) / Nx for j in 1:Ny]
end

function tau_from_psi_cpu(psixx, psixy, psiyy, prefactor)
    Nx, Ny = size(psixx)
    tau_xx = similar(psixx)
    tau_xy = similar(psixy)
    tau_yy = similar(psiyy)
    for j in 1:Ny, i in 1:Nx
        cxx, cxy, cyy = Kraken.logfv_exp_sym2_2d(psixx[i, j], psixy[i, j], psiyy[i, j])
        tau_xx[i, j] = prefactor * (cxx - one(eltype(psixx)))
        tau_xy[i, j] = prefactor * cxy
        tau_yy[i, j] = prefactor * (cyy - one(eltype(psixx)))
    end
    return (tau_xx, tau_xy, tau_yy)
end

function relative_l2(values, reference)
    num = zero(eltype(values))
    den = zero(eltype(values))
    for k in eachindex(values, reference)
        num += abs2(values[k] - reference[k])
        den += abs2(reference[k])
    end
    return den == 0 ? oftype(num, NaN) : sqrt(num / den)
end

function max_abs_deviation(values, target)
    return maximum(abs.(values .- target))
end

"""
    make_diagnostic_plots(out_dir, cfg, result, ana, tau_xx_field, tau_xy_field, tau_yy_field, profiles)

Produce six diagnostic PNGs in `out_dir`. Saved unconditionally so that any
PASS/FAIL outcome is self-diagnosing. See `[[feedback_always_plot]]` in
MEMORY.md.
"""
function make_diagnostic_plots(out_dir, cfg, result, ana,
                               tau_xx_field, tau_xy_field, tau_yy_field,
                               u_profile, tau_xy_profile, tau_xx_profile, uy_profile)
    mkpath(out_dir)
    Nx, Ny = cfg.Nx, cfg.Ny
    j_axis = collect(1:Ny)

    # 1. Mesh / domain overview with walls highlighted
    fig1 = Figure(size=(700, 500))
    ax1 = Axis(fig1[1, 1]; title="L1 domain — periodic-x, HWBB walls at j=1, j=Ny",
               xlabel="i (x, LU)", ylabel="j (y, LU)")
    # mesh as scatter of cell centres
    xs = [Float64(i) for i in 1:Nx, _ in 1:Ny]
    ys = [Float64(j) for _ in 1:Nx, j in 1:Ny]
    scatter!(ax1, vec(xs), vec(ys); color=:lightgray, markersize=6, label="fluid cells")
    # highlight wall cells (j=1 and j=Ny) — half-cell wall lives at j=0.5 and j=Ny+0.5
    scatter!(ax1, collect(1:Nx), fill(1.0, Nx); color=:red, markersize=10, label="wall-adjacent (j=1)")
    scatter!(ax1, collect(1:Nx), fill(Float64(Ny), Nx); color=:red, markersize=10, label="wall-adjacent (j=Ny)")
    hlines!(ax1, [0.5, Ny + 0.5]; color=:black, linestyle=:dash, label="HWBB wall location")
    axislegend(ax1; position=:rb)
    save(joinpath(out_dir, "01_mesh.png"), fig1)

    # 2. Quiver of velocity field
    fig2 = Figure(size=(800, 500))
    ax2 = Axis(fig2[1, 1]; title="velocity field (Kraken, scaled)",
               xlabel="i (LU)", ylabel="j (LU)")
    # subsample to keep arrows readable
    step_i = max(1, Nx ÷ 8)
    step_j = max(1, Ny ÷ 16)
    xs_q = Float64[]
    ys_q = Float64[]
    us_q = Float64[]
    vs_q = Float64[]
    for j in 1:step_j:Ny, i in 1:step_i:Nx
        push!(xs_q, Float64(i))
        push!(ys_q, Float64(j))
        push!(us_q, Float64(result.ux[i, j]))
        push!(vs_q, Float64(result.uy[i, j]))
    end
    umax = maximum(abs, us_q) + eps()
    arrows!(ax2, xs_q, ys_q, us_q ./ umax .* 0.8, vs_q ./ umax .* 0.8;
            arrowsize=8, lengthscale=1.0, color=:steelblue)
    hlines!(ax2, [0.5, Ny + 0.5]; color=:black, linestyle=:dash)
    save(joinpath(out_dir, "02_quiver.png"), fig2)

    # 3. u(y) centerline profile with analytic overlay
    fig3 = Figure(size=(700, 500))
    ax3 = Axis(fig3[1, 1]; title="u_x(y): Kraken vs Bird-AH analytic Oldroyd-B",
               xlabel="j (LU)", ylabel="u_x (LU)")
    lines!(ax3, j_axis, ana.u_x; color=:black, linestyle=:dash, linewidth=2, label="analytic (BAH §3.4)")
    scatter!(ax3, j_axis, u_profile; color=:firebrick, markersize=8, label="Kraken (y-averaged)")
    axislegend(ax3; position=:rb)
    save(joinpath(out_dir, "03_u_profile.png"), fig3)

    # 4. rho(y) deviation from 1.0
    fig4 = Figure(size=(700, 500))
    ax4 = Axis(fig4[1, 1]; title="rho(y) - 1 (incompressibility / LBM density drift)",
               xlabel="j (LU)", ylabel="rho - 1")
    rho_mean = [sum(@view result.rho[:, j]) / Nx for j in 1:Ny]
    lines!(ax4, j_axis, rho_mean .- 1.0; color=:steelblue, linewidth=2)
    hlines!(ax4, [0.0]; color=:black, linestyle=:dash)
    hlines!(ax4, [1e-3, -1e-3]; color=:red, linestyle=:dot, label="±1e-3 threshold")
    axislegend(ax4; position=:rt)
    save(joinpath(out_dir, "04_rho.png"), fig4)

    # 5. tau_xy(y) Kraken vs analytic
    fig5 = Figure(size=(700, 500))
    ax5 = Axis(fig5[1, 1]; title="tau_xy(y): Kraken vs analytic (linear in y)",
               xlabel="j (LU)", ylabel="tau_xy (LU)")
    lines!(ax5, j_axis, ana.tau_xy; color=:black, linestyle=:dash, linewidth=2, label="analytic")
    scatter!(ax5, j_axis, tau_xy_profile; color=:firebrick, markersize=8, label="Kraken (y-averaged)")
    axislegend(ax5; position=:rb)
    save(joinpath(out_dir, "05_tau_xy.png"), fig5)

    # 6. tau_xx(y) Kraken vs analytic (first normal stress)
    fig6 = Figure(size=(700, 500))
    ax6 = Axis(fig6[1, 1]; title="tau_xx(y): Kraken vs analytic N1 = 2 lambda nu_p gamma_dot^2",
               xlabel="j (LU)", ylabel="tau_xx (LU)")
    lines!(ax6, j_axis, ana.tau_xx; color=:black, linestyle=:dash, linewidth=2, label="analytic")
    scatter!(ax6, j_axis, tau_xx_profile; color=:firebrick, markersize=8, label="Kraken (y-averaged)")
    axislegend(ax6; position=:rt)
    save(joinpath(out_dir, "06_tau_xx.png"), fig6)

    return nothing
end

function main()
    started_at = now()
    Tt = Float64
    overrides = parse_kwargs()
    cfg = config(Tt; overrides=overrides)
    tag = get(overrides, :tag, "default")
    backend = KernelAbstractions.CPU()

    @info "L1 Poiseuille Oldroyd-B starting" cfg.Nx cfg.Ny cfg.nu_s cfg.nu_p cfg.lambda cfg.Fx_body cfg.max_steps tag
    t0 = time()
    result = Kraken.run_viscoelastic_logfv_poiseuille_coupled_2d(;
        Nx=cfg.Nx,
        Ny=cfg.Ny,
        nu_s=cfg.nu_s,
        nu_p=cfg.nu_p,
        Fx_body=cfg.Fx_body,
        lambda=cfg.lambda,
        bsd_fraction=cfg.bsd_fraction,
        polymer_substeps=cfg.polymer_substeps,
        max_steps=cfg.max_steps,
        backend=backend,
        T=Tt,
    )
    wallclock = time() - t0
    @info "driver returned" wallclock max_abs_error=result.max_abs_error min_c_eig=result.min_c_eig

    ana = analytic_oldroydb_poiseuille(cfg)
    prefactor = cfg.nu_p / cfg.lambda
    tau_xx_field, tau_xy_field, tau_yy_field =
        tau_from_psi_cpu(result.psixx, result.psixy, result.psiyy, prefactor)

    # y-averaged profiles
    u_profile = result.ux_mean
    tau_xy_profile = profile_yaverage(tau_xy_field)
    tau_xx_profile = profile_yaverage(tau_xx_field)
    tau_yy_profile = profile_yaverage(tau_yy_field)
    uy_profile = profile_yaverage(result.uy)

    # Wall-adjacent quantities (j = 1 and j = Ny)
    tau_xy_wall_pair_kraken = (tau_xy_profile[1], tau_xy_profile[end])
    tau_xy_wall_pair_analytic = (ana.tau_xy[1], ana.tau_xy[end])
    tau_xx_wall_pair_kraken = (tau_xx_profile[1], tau_xx_profile[end])
    tau_xx_wall_pair_analytic = (ana.tau_xx[1], ana.tau_xx[end])

    # Relative L2 over interior (avoid the half-cell wall offset distorting metrics)
    interior = 2:(cfg.Ny - 1)
    u_rel_l2 = relative_l2(u_profile[interior], ana.u_x[interior])
    tau_xy_rel_l2 = relative_l2(tau_xy_profile[interior], ana.tau_xy[interior])
    tau_xx_rel_l2 = relative_l2(tau_xx_profile[interior], ana.tau_xx[interior])

    # Wall pair (j=1 and j=Ny only) — separate "wall" metric
    tau_xy_wall_rel = max(
        abs(tau_xy_wall_pair_kraken[1] - tau_xy_wall_pair_analytic[1]) / abs(tau_xy_wall_pair_analytic[1]),
        abs(tau_xy_wall_pair_kraken[2] - tau_xy_wall_pair_analytic[2]) / abs(tau_xy_wall_pair_analytic[2]),
    )
    tau_xx_wall_rel = max(
        abs(tau_xx_wall_pair_kraken[1] - tau_xx_wall_pair_analytic[1]) / abs(tau_xx_wall_pair_analytic[1]),
        abs(tau_xx_wall_pair_kraken[2] - tau_xx_wall_pair_analytic[2]) / abs(tau_xx_wall_pair_analytic[2]),
    )

    # Centreline (peak of profile)
    u_peak_kraken = maximum(u_profile)
    u_peak_rel = abs(u_peak_kraken - ana.u_peak_analytic) / abs(ana.u_peak_analytic)

    # Density / continuity
    rho_max_abs_dev = max_abs_deviation(result.rho, 1.0)
    uy_interior_max_abs = maximum(abs, result.uy[:, interior])

    # Conformation SPD
    min_eig_C = result.min_c_eig

    # NaN / Inf sentinel
    finite_ok = all(
        x -> all(isfinite, x),
        (
            result.ux, result.uy, result.rho,
            result.psixx, result.psixy, result.psiyy,
            tau_xx_field, tau_xy_field, tau_yy_field,
        ),
    )

    diagnostics = Dict(
        "wallclock_seconds" => wallclock,
        "u_centerline_kraken" => u_peak_kraken,
        "u_centerline_analytic" => ana.u_peak_analytic,
        "u_centerline_relative" => u_peak_rel,
        "u_profile_interior_relL2" => u_rel_l2,
        "tau_xy_wall_pair_kraken" => collect(tau_xy_wall_pair_kraken),
        "tau_xy_wall_pair_analytic" => collect(tau_xy_wall_pair_analytic),
        "tau_xy_wall_relative" => tau_xy_wall_rel,
        "tau_xy_profile_interior_relL2" => tau_xy_rel_l2,
        "tau_xx_wall_pair_kraken" => collect(tau_xx_wall_pair_kraken),
        "tau_xx_wall_pair_analytic" => collect(tau_xx_wall_pair_analytic),
        "tau_xx_wall_relative" => tau_xx_wall_rel,
        "tau_xx_profile_interior_relL2" => tau_xx_rel_l2,
        "tau_yy_profile_interior_max_abs" => maximum(abs, tau_yy_profile[interior]),
        "rho_max_abs_deviation" => rho_max_abs_dev,
        "uy_interior_max_abs" => uy_interior_max_abs,
        "min_eig_C" => min_eig_C,
        "no_nan_no_inf" => finite_ok,
        "Wi_estimate_lambda_gamma_dot_wall" => cfg.lambda * abs(ana.gamma_dot[1]),
    )
    profiles = Dict(
        "j_index" => collect(1:cfg.Ny),
        "u_kraken" => u_profile,
        "u_analytic" => ana.u_x,
        "tau_xy_kraken" => tau_xy_profile,
        "tau_xy_analytic" => ana.tau_xy,
        "tau_xx_kraken" => tau_xx_profile,
        "tau_xx_analytic" => ana.tau_xx,
        "uy_kraken" => uy_profile,
        "gamma_dot_analytic" => ana.gamma_dot,
    )

    Wi_target_used = haskey(overrides, :Wi_target) ?
        Float64(overrides[:Wi_target]) :
        Float64(cfg.lambda) * Float64(cfg.Fx_body) *
            wall_gamma_dot_factor(cfg.Ny) / (2 * Float64(cfg.nu_s + cfg.nu_p))
    payload = Dict(
        "schema_version" => 1,
        "test_id" => "L1_poiseuille_oldb",
        "started_at" => string(started_at),
        "completed_at" => string(now()),
        "tag" => tag,
        "config" => Dict(
            "Nx" => cfg.Nx,
            "Ny" => cfg.Ny,
            "nu_s" => cfg.nu_s,
            "nu_p" => cfg.nu_p,
            "Fx_body" => cfg.Fx_body,
            "lambda" => cfg.lambda,
            "bsd_fraction" => cfg.bsd_fraction,
            "polymer_substeps_requested" =>
                cfg.polymer_substeps === :auto ? "auto" : string(cfg.polymer_substeps),
            "polymer_substeps_selected" => result.polymer_substeps,
            "max_steps" => cfg.max_steps,
            "backend" => "CPU",
            "precision" => string(Tt),
            "Wi_target" => Wi_target_used,
        ),
        "diagnostics" => diagnostics,
        "profiles" => profiles,
    )

    # Diagnostic plots — always produced, BEFORE JSON write, BEFORE any later
    # exit/throw, so a FAIL is self-diagnosing.
    try
        make_diagnostic_plots(
            joinpath(HERE, "diagnostics"),
            cfg, result, ana,
            tau_xx_field, tau_xy_field, tau_yy_field,
            u_profile, tau_xy_profile, tau_xx_profile, uy_profile,
        )
        @info "diagnostic plots written" dir=joinpath(HERE, "diagnostics")
    catch err
        @warn "diagnostic plotting failed" exception=(err, catch_backtrace())
    end

    mkpath(joinpath(HERE, "results"))
    stamp = Dates.format(started_at, "yyyymmddTHHMMSS")
    out = joinpath(HERE, "results", "L1_run_$(tag)_$(stamp).json")
    open(out, "w") do io
        JSON3.pretty(io, payload)
    end
    tagged_latest = joinpath(HERE, "results", "L1_run_$(tag)_latest.json")
    open(tagged_latest, "w") do io
        JSON3.pretty(io, payload)
    end
    latest = joinpath(HERE, "results", "L1_run_latest.json")
    open(latest, "w") do io
        JSON3.pretty(io, payload)
    end
    println(@sprintf("L1 run complete: %.2f s", wallclock))
    println("  results -> $out")
    println("  latest  -> $latest")
    println("Next: julia --project=. bench/viscoelastic_validation/L1_poiseuille_oldb/compare.jl")
    return 0
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
