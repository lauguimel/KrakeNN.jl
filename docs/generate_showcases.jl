#!/usr/bin/env julia
"""
Generate showcase GIF animations for documentation.

Usage:
    julia --project=docs docs/generate_showcases.jl

Produces GIF files in docs/src/assets/showcases/
"""

using Metal
using KrakeNN
using CairoMakie
using KernelAbstractions

# CairoMakie re-exports the Colors/ColorTypes stack (Colorant, RGBf, cgrad),
# so no separate `using Colors` is needed (and Colors is not a direct dep here).
const Colorant = CairoMakie.Colorant

const OUTDIR = joinpath(@__DIR__, "src", "assets", "showcases")
mkpath(OUTDIR)

# --- Dark theme (matches the Documenter dark page background #1b1b1f) ---

const BG_DARK    = parse(Colorant, "#1b1b1f")              # page background
const FG_LIGHT   = RGBf(0.92, 0.92, 0.92)                  # titles / labels / spines
const FG_DIM     = RGBf(0.55, 0.55, 0.55)                  # ticks / gridlines

# Dark-centred diverging colormap for signed fields: the zero blends into the
# page background, only strong ± values pop. Replaces the light-centred :RdBu.
const CMAP_DIVERGING = cgrad([parse(Colorant, "#4ea1d3"),
                              parse(Colorant, "#1b1b1f"),
                              parse(Colorant, "#ff6b6b")])

# High-luminosity diverging map dedicated to the von Karman vortex street:
# vivid electric cyan on one side, hot magenta/amber on the other, with the
# page background "#1b1b1f" kept as the exact centre so zero-vorticity regions
# blend into the dark docs page while the vortices pop brilliantly. 5 stops
# (bright -> mid -> dark -> mid -> bright) give a richer, more luminous ramp.
const CMAP_VONKARMAN = cgrad([parse(Colorant, "#22d3ee"),   # electric cyan
                              parse(Colorant, "#2dd4bf"),   # bright teal
                              parse(Colorant, "#1b1b1f"),   # page background (zero)
                              parse(Colorant, "#fb7185"),   # hot rose
                              parse(Colorant, "#fbbf24")])  # vivid amber

# --- Utility ---

function make_gif(frames::Vector{<:Matrix}, path::String;
                  fps=10, colormap=:viridis, title="",
                  figsize=(800, 400), clims=nothing, decorate::Bool=true)
    isempty(frames) && return
    # Replace any NaN/Inf with 0
    for f in frames
        replace!(x -> isfinite(x) ? x : zero(x), f)
    end
    vmin = clims === nothing ? minimum(minimum.(frames)) : clims[1]
    vmax = clims === nothing ? maximum(maximum.(frames)) : clims[2]
    if !isfinite(vmin) || !isfinite(vmax)
        vmin, vmax = -1.0, 1.0
    end
    if vmin ≈ vmax
        vmax = vmin + one(vmin)
    end

    fig = Figure(; size=figsize, backgroundcolor=BG_DARK)
    ax = Axis(fig[1, 1]; title=(decorate ? title : ""), aspect=DataAspect(),
              backgroundcolor=BG_DARK,
              titlecolor=FG_LIGHT,
              xlabelcolor=FG_LIGHT, ylabelcolor=FG_LIGHT,
              xticklabelcolor=FG_LIGHT, yticklabelcolor=FG_LIGHT,
              leftspinecolor=FG_LIGHT, rightspinecolor=FG_LIGHT,
              topspinecolor=FG_LIGHT, bottomspinecolor=FG_LIGHT,
              xtickcolor=FG_DIM, ytickcolor=FG_DIM,
              xgridcolor=FG_DIM, ygridcolor=FG_DIM)
    obs = Observable(frames[1])
    heatmap!(ax, obs; colormap=colormap, colorrange=(vmin, vmax))
    if decorate
        Colorbar(fig[1, 2]; colormap=colormap, limits=(vmin, vmax),
                 ticklabelcolor=FG_LIGHT, labelcolor=FG_LIGHT, tickcolor=FG_DIM,
                 leftspinecolor=FG_LIGHT, rightspinecolor=FG_LIGHT,
                 topspinecolor=FG_LIGHT, bottomspinecolor=FG_LIGHT)
    else
        # Clean full-bleed variant: no axes/ticks/title/colorbar, just the field.
        hidedecorations!(ax)
        hidespines!(ax)
    end

    record(fig, path, 1:length(frames); framerate=fps) do i
        obs[] = frames[i]
    end
    # Save a representative still frame (last frame) so the dark styling can be
    # verified without playing the GIF.
    obs[] = frames[end]
    save(replace(path, ".gif" => "_frame.png"), fig)
    @info "Saved $(path) ($(length(frames)) frames, $(filesize(path)) bytes)"
end

function compute_vorticity(ux::Matrix, uy::Matrix)
    Nx, Ny = size(ux)
    ω = zeros(eltype(ux), Nx, Ny)
    for j in 2:Ny-1, i in 2:Nx-1
        duy_dx = (uy[i+1, j] - uy[i-1, j]) / 2
        dux_dy = (ux[i, j+1] - ux[i, j-1]) / 2
        val = duy_dx - dux_dy
        ω[i, j] = isfinite(val) ? val : zero(val)
    end
    return ω
end

# --- Try Metal backend, fall back to CPU ---

function get_backend()
    @info "Using Metal GPU backend"
    return MetalBackend()
end

# =====================================================================
# 1. Von Karman vortex street (cylinder Re=200)
# =====================================================================

function showcase_vonkarman(; backend=CPU())
    @info "Generating Von Karman vortex street (Re=200)..."
    T = Float32

    Nx, Ny = 800, 200
    radius = 20
    cx = Nx ÷ 5
    cy = Ny ÷ 2 + 1   # offset by 1 to break symmetry
    u_in = T(0.04)
    Re = 200
    D = 2 * radius
    nu = Float64(u_in) * D / Re   # 0.008
    max_steps = 80000
    snap_every = 200

    state, config = initialize_cylinder_2d(; Nx=Nx, Ny=Ny, cx=cx, cy=cy,
                                             radius=radius, u_in=Float64(u_in),
                                             ν=nu, backend=backend, T=T)
    f_in, f_out = state.f_in, state.f_out
    rho, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid

    frames = Matrix{Float32}[]

    for step in 1:max_steps
        stream_2d!(f_out, f_in, Nx, Ny)
        apply_zou_he_west_2d!(f_out, u_in, Nx, Ny)
        apply_zou_he_pressure_east_2d!(f_out, Nx, Ny)
        # MRT for stability at Re=200 (ω_ν ≈ 1.91)
        collide_mrt_2d!(f_out, is_solid, T(nu))
        compute_macroscopic_2d!(rho, ux, uy, f_out)
        f_in, f_out = f_out, f_in

        if step % snap_every == 0 && step > 30000
            vort = compute_vorticity(Array(ux), Array(uy))
            push!(frames, Float32.(vort))
        end

        if step % 10000 == 0
            @info "  step $step / $max_steps"
        end
    end

    # Use auto-computed symmetric limits from max vorticity
    all_max = [maximum(abs, f) for f in frames]
    vlim = Float64(quantile_approx(all_max, 0.9))
    if vlim < 1e-10
        vlim = 1.0  # fallback
    end
    make_gif(frames, joinpath(OUTDIR, "vonkarman_re200.gif");
             fps=15, colormap=CMAP_VONKARMAN, title="Von Karman vortex street (Re=200)",
             figsize=(800, 300), clims=(-vlim, vlim))
    # Clean decoration-free hero variant (no axes/ticks/title/colorbar) reusing
    # the same frames — used full-bleed by the docs hero/CSS at /showcases/.
    make_gif(frames, joinpath(OUTDIR, "vonkarman_re200_clean.gif");
             fps=15, colormap=CMAP_VONKARMAN,
             figsize=(800, 300), clims=(-vlim, vlim), decorate=false)
end

"Approximate quantile (avoids importing Statistics)."
function quantile_approx(v, p)
    s = sort(v)
    idx = clamp(round(Int, p * length(s)), 1, length(s))
    return s[idx]
end

# =====================================================================
# 2. Rayleigh-Benard convection rolls (Ra=1e5)
# =====================================================================

function showcase_rayleigh_benard(; backend=CPU())
    @info "Generating Rayleigh-Benard convection (Ra=1e5)..."
    FT = Float32

    Nx, Ny = 256, 128
    Ra = 1e5
    Pr = 0.71
    T_hot, T_cold = FT(1), FT(0)
    dT = T_hot - T_cold
    H = Ny
    nu = FT(0.05)
    alpha = nu / FT(Pr)
    beta_g = FT(Ra * nu * alpha / (dT * H^3))
    omega_f = FT(1.0 / (3.0 * nu + 0.5))
    omega_T = FT(1.0 / (3.0 * alpha + 0.5))
    max_steps = 40000
    snap_every = 400

    config = LBMConfig(D2Q9(); Nx=Nx, Ny=Ny, ν=Float64(nu), u_lid=0.0, max_steps=max_steps)
    state = initialize_2d(config, FT; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    rho, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid

    g_in  = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    g_out = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    Temp  = KernelAbstractions.zeros(backend, FT, Nx, Ny)

    w = weights(D2Q9())
    g_cpu = zeros(FT, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx
        T_init = FT(T_hot - dT * (j - 1) / (Ny - 1))
        T_init += FT(0.01 * dT) * sin(FT(2pi * i / Nx)) * sin(FT(pi * j / Ny))
        for q in 1:9
            g_cpu[i, j, q] = FT(w[q]) * T_init
        end
    end
    copyto!(g_in, g_cpu)
    copyto!(g_out, g_cpu)
    T_ref = FT((T_hot + T_cold) / 2)

    frames = Matrix{Float32}[]

    for step in 1:max_steps
        stream_periodic_x_wall_y_2d!(f_out, f_in, Nx, Ny)
        stream_periodic_x_wall_y_2d!(g_out, g_in, Nx, Ny)
        apply_fixed_temp_south_2d!(g_out, T_hot, Nx)
        apply_fixed_temp_north_2d!(g_out, T_cold, Nx, Ny)
        compute_temperature_2d!(Temp, g_out)
        compute_macroscopic_2d!(rho, ux, uy, f_out)
        collide_thermal_2d!(g_out, ux, uy, omega_T)
        collide_boussinesq_2d!(f_out, Temp, is_solid, omega_f, beta_g, T_ref)
        f_in, f_out = f_out, f_in
        g_in, g_out = g_out, g_in

        if step % snap_every == 0
            push!(frames, Float32.(Array(Temp)))
        end

        if step % 10000 == 0
            @info "  step $step / $max_steps"
        end
    end

    make_gif(frames, joinpath(OUTDIR, "rayleigh_benard_ra1e5.gif");
             fps=10, colormap=:inferno, title="Rayleigh-Benard convection (Ra=1e5)",
             figsize=(800, 400))
end

# =====================================================================
# 3. Taylor-Green vortex decay (N=256)
# =====================================================================

function showcase_taylor_green(; backend=CPU())
    @info "Generating Taylor-Green vortex decay..."
    T = Float32

    N = 256
    nu = T(0.001)
    u0 = T(0.04)
    max_steps = 5000
    snap_every = 50

    state = initialize_taylor_green_2d(; N=N, ν=Float64(nu), u0=Float64(u0),
                                         backend=backend, T=T)
    f_in, f_out = state.f_in, state.f_out
    rho, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid
    omega_val = T(1.0 / (3.0 * nu + 0.5))

    frames = Matrix{Float32}[]

    for step in 1:max_steps
        stream_fully_periodic_2d!(f_out, f_in, N, N)
        collide_2d!(f_out, is_solid, omega_val)
        f_in, f_out = f_out, f_in

        if step % snap_every == 0
            compute_macroscopic_2d!(rho, ux, uy, f_in)
            vort = compute_vorticity(Array(ux), Array(uy))
            push!(frames, Float32.(vort))
        end
    end

    vlim = maximum(abs, frames[1]) * 0.8
    make_gif(frames, joinpath(OUTDIR, "taylor_green_decay.gif");
             fps=15, colormap=CMAP_DIVERGING, title="Taylor-Green vortex decay (N=256)",
             figsize=(600, 600), clims=(-vlim, vlim))
end

# =====================================================================
# 4. Cavity Re=1000 (256x256)
# =====================================================================

function showcase_cavity(; backend=CPU())
    @info "Generating lid-driven cavity (Re=1000)..."
    T = Float32

    Nx, Ny = 256, 256
    u_lid = T(0.1)
    Re = 1000
    nu = u_lid * Nx / Re  # 0.0256
    omega_val = T(1.0 / (3.0 * nu + 0.5))
    max_steps = 100000
    snap_every = 2000

    config = LBMConfig(D2Q9(); Nx=Nx, Ny=Ny, ν=Float64(nu), u_lid=Float64(u_lid),
                       max_steps=max_steps)
    state = initialize_2d(config, T; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    rho, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid

    frames = Matrix{Float32}[]

    for step in 1:max_steps
        stream_2d!(f_out, f_in, Nx, Ny)
        apply_zou_he_north_2d!(f_out, u_lid, Nx, Ny)
        collide_2d!(f_out, is_solid, omega_val)
        compute_macroscopic_2d!(rho, ux, uy, f_out)
        f_in, f_out = f_out, f_in

        if step % snap_every == 0
            umag = sqrt.(Array(ux).^2 .+ Array(uy).^2)
            push!(frames, Float32.(umag))
        end

        if step % 20000 == 0
            @info "  step $step / $max_steps"
        end
    end

    make_gif(frames, joinpath(OUTDIR, "cavity_re1000.gif");
             fps=10, colormap=:magma, title="Lid-driven cavity (Re=1000, 256x256)",
             figsize=(600, 600))
end

# =====================================================================
# Main
# =====================================================================

function main()
    t0 = time()

    backend = get_backend()

    showcase_vonkarman(; backend)
    showcase_rayleigh_benard(; backend)
    showcase_taylor_green(; backend)
    showcase_cavity(; backend)

    elapsed = round(time() - t0; digits=1)
    @info "All showcases generated in $(elapsed)s"

    # Summary
    for f in readdir(OUTDIR)
        path = joinpath(OUTDIR, f)
        sz = round(filesize(path) / 1024; digits=1)
        println("  $f  $(sz) KB")
    end
end

main()
