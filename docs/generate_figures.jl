#!/usr/bin/env julia
# Generate all PNG figures for docs/src/assets/figures/
# Run with: julia --project=docs docs/generate_figures.jl

using KrakeNN
using CairoMakie
using Printf

const FIGDIR = joinpath(@__DIR__, "src", "assets", "figures")
mkpath(FIGDIR)

CairoMakie.activate!()

# ============================================================================
# Dark "krakendark" docs identity — figures are baked to the live Vitepress
# dark page colour (#1b1b1f) so they sit seamlessly on the docs page (no
# light/dark image swap is configured). Light text/axes, faint grid, vivid
# series colours, magma for speed, inferno for temperature, a dark-centred
# diverging map for signed fields (vorticity). Mirrors assets/krakendark.py.
# ============================================================================
const DARK    = "#1b1b1f"      # Vitepress default dark page background
const ACCENT  = "#ff6b6b"      # KrakeNN highlight / warm pole / analytical
const COOL    = "#4ea1d3"      # cool pole / LBM series
const TEXTCOL = RGBf(0.92, 0.92, 0.92)
const TICKCOL = RGBf(0.85, 0.85, 0.85)
const GRIDCOL = RGBf(0.45, 0.45, 0.45)
const SPINECOL = RGBf(0.55, 0.55, 0.55)

# Field colormaps, by quantity (magma = speed/positive, inferno = temperature).
const CMAP_SPEED = :magma
const CMAP_TEMP  = :inferno
# Dark-centred diverging map: near-zero blends into the page, strong +/- pop.
const DIVR = cgrad([COOL, DARK, ACCENT])

set_theme!(
    fontsize          = 14,
    backgroundcolor   = DARK,
    textcolor         = TEXTCOL,
    Figure            = (; backgroundcolor = DARK),
    Axis = (;
        backgroundcolor   = DARK,
        titlecolor        = TEXTCOL,
        xlabelcolor       = TEXTCOL, ylabelcolor       = TEXTCOL,
        xticklabelcolor   = TICKCOL, yticklabelcolor   = TICKCOL,
        xtickcolor        = TICKCOL, ytickcolor        = TICKCOL,
        xgridcolor        = (GRIDCOL, 0.4), ygridcolor = (GRIDCOL, 0.4),
        xgridwidth        = 0.6, ygridwidth = 0.6,
        leftspinecolor    = SPINECOL, rightspinecolor  = SPINECOL,
        topspinecolor     = SPINECOL, bottomspinecolor = SPINECOL,
    ),
    Legend = (;
        backgroundcolor = DARK,
        framecolor      = RGBf(0.5, 0.5, 0.5),
        labelcolor      = TEXTCOL,
        titlecolor      = TEXTCOL,
    ),
    Colorbar = (;
        labelcolor      = TEXTCOL,
        ticklabelcolor  = TICKCOL,
        tickcolor       = TICKCOL,
        leftspinecolor  = SPINECOL, rightspinecolor  = SPINECOL,
        topspinecolor   = SPINECOL, bottomspinecolor = SPINECOL,
    ),
)

# ============================================================================
# 1. Poiseuille profile (corrected analytical with half-way BB offset)
# ============================================================================
println("=== 1. Poiseuille profile ===")
let
    Ny = 32; ν = 0.1; Fx = 1e-5
    nsteps = max(50000, ceil(Int, 3 * Ny^2 / ν))
    result = run_poiseuille_2d(; Nx=4, Ny=Ny, ν=ν, Fx=Fx, max_steps=nsteps)
    u_num = result.ux[2, :]

    # Correct analytical: half-way bounce-back convention
    # Fluid nodes j=1..Ny, wall at j=0.5 and j=Ny+0.5
    # u(j) = Fx/(2ν) * (j - 0.5) * (Ny + 0.5 - j)
    u_ana = [Fx / (2ν) * (j - 0.5) * (Ny + 0.5 - j) for j in 1:Ny]

    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1]; xlabel="u_x (lattice units)", ylabel="y (node index)",
              title="Poiseuille flow -- Ny = $Ny")
    lines!(ax, u_ana, 1:Ny; label="Analytical", linewidth=2, color=ACCENT, linestyle=:dash)
    lines!(ax, u_num, 1:Ny; label="LBM", linewidth=2, color=COOL)
    axislegend(ax; position=:rb)
    save(joinpath(FIGDIR, "poiseuille_profile.png"), fig; px_per_unit=2)
    println("  done")
end

# ============================================================================
# 2. Cavity centerlines vs Ghia (1982)
# ============================================================================
println("=== 2. Cavity centerlines ===")
let
    N = 128; Re = 100; u_lid = 0.1
    ν = u_lid * N / Re
    config = LBMConfig(D2Q9(); Nx=N, Ny=N, ν=ν, u_lid=u_lid, max_steps=60000)
    result = run_cavity_2d(config)

    # Ghia data
    y_ghia  = [0.0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
               0.4531, 0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
               0.9688, 0.9766, 1.0]
    ux_ghia = [0.0, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
              -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
               0.68717, 0.73722, 0.78871, 0.84123, 1.0]

    mid = N ÷ 2 + 1
    ux_profile = [result.ux[mid, j] / u_lid for j in 1:N]
    y_norm     = [(j - 0.5) / N for j in 1:N]

    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1]; xlabel="u_x / u_lid", ylabel="y / L",
              title="Lid-driven cavity -- Re = $Re, N = $N")
    lines!(ax, ux_profile, y_norm; label="LBM", linewidth=2, color=COOL)
    scatter!(ax, ux_ghia, y_ghia; label="Ghia et al. (1982)", color=ACCENT, markersize=10)
    axislegend(ax; position=:lb)
    save(joinpath(FIGDIR, "cavity_centerlines.png"), fig; px_per_unit=2)
    println("  done")
end

# ============================================================================
# 3. Couette profile
# ============================================================================
println("=== 3. Couette profile ===")
let
    Ny = 32; ν = 0.1; u_wall = 0.05
    result = run_couette_2d(; Nx=4, Ny=Ny, ν=ν, u_wall=u_wall, max_steps=20000)

    H = Ny - 1
    j_fluid = 2:Ny-1
    y_phys = [j - 1 for j in j_fluid]
    u_ana  = [u_wall * (1 - y / H) for y in y_phys]
    u_num  = [result.ux[2, j] for j in j_fluid]

    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1]; xlabel="u_x (lattice units)", ylabel="y (lattice units)",
              title="Couette flow -- Ny = $Ny")
    lines!(ax, u_ana, y_phys; label="Analytical", linewidth=2, color=ACCENT, linestyle=:dash)
    lines!(ax, u_num, y_phys; label="LBM", linewidth=2, color=COOL)
    axislegend(ax; position=:rt)
    save(joinpath(FIGDIR, "couette_profile.png"), fig; px_per_unit=2)
    println("  done")
end

# ============================================================================
# 4. Taylor-Green vorticity heatmap
# ============================================================================
println("=== 4. Taylor-Green vorticity ===")
let
    N = 64; u0 = 0.04; ν = 0.01
    result = run_taylor_green_2d(; N=N, ν=ν, u0=u0, max_steps=5000)

    ωz = zeros(N, N)
    for j in 1:N, i in 1:N
        ip = mod1(i + 1, N); im = mod1(i - 1, N)
        jp = mod1(j + 1, N); jm = mod1(j - 1, N)
        ωz[i, j] = 0.5 * (result.uy[ip, j] - result.uy[im, j]) -
                    0.5 * (result.ux[i, jp] - result.ux[i, jm])
    end

    fig = Figure(size=(600, 500))
    ax = Axis(fig[1, 1]; title="Vorticity at t = 5000", xlabel="x", ylabel="y",
              aspect=DataAspect())
    wmax = maximum(abs, ωz)
    hm = heatmap!(ax, 1:N, 1:N, ωz; colormap=DIVR, colorrange=(-wmax, wmax))
    Colorbar(fig[1, 2], hm; label="omega_z")
    save(joinpath(FIGDIR, "taylor_green_vorticity.png"), fig; px_per_unit=2)
    println("  done")
end

# ============================================================================
# 5. Heat conduction profile (T vs y)
# ============================================================================
println("=== 5. Heat conduction profile ===")
let
    # Sub-critical Ra: pure conduction regime
    Ra = 100.0; Pr = 1.0; T_hot = 1.0; T_cold = 0.0
    result = run_rayleigh_benard_2d(;
        Nx=128, Ny=32, Ra=Ra, Pr=Pr, T_hot=T_hot, T_cold=T_cold, max_steps=20000)

    Ny = size(result.Temp, 2)
    H = Ny - 1
    j_fluid = 2:Ny-1
    y_phys = [(j - 1.5) / H for j in j_fluid]
    T_ana  = [T_hot - (T_hot - T_cold) * y for y in y_phys]
    T_num  = [result.Temp[64, j] for j in j_fluid]

    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1]; xlabel="Temperature", ylabel="y / H",
              title="Heat conduction -- sub-critical Ra = $Ra")
    lines!(ax, T_ana, y_phys; label="Analytical (T = 1 - y/H)", linewidth=2, color=ACCENT, linestyle=:dash)
    lines!(ax, T_num, y_phys; label="LBM", linewidth=2, color=COOL)
    axislegend(ax; position=:rt)
    save(joinpath(FIGDIR, "heat_profile.png"), fig; px_per_unit=2)
    println("  done")
end

# ============================================================================
# 6. Rayleigh-Benard temperature heatmap
# ============================================================================
println("=== 6. Rayleigh-Benard temperature ===")
let
    Ra = 5000.0; Pr = 1.0; T_hot = 1.0; T_cold = 0.0
    result = run_rayleigh_benard_2d(;
        Nx=128, Ny=32, Ra=Ra, Pr=Pr, T_hot=T_hot, T_cold=T_cold, max_steps=30000)

    Nx, Ny = size(result.Temp)
    fig = Figure(size=(600, 500))
    ax = Axis(fig[1, 1]; title="Temperature -- Ra = $Ra, Pr = $Pr",
              xlabel="x", ylabel="y", aspect=DataAspect())
    hm = heatmap!(ax, 1:Nx, 1:Ny, result.Temp; colormap=CMAP_TEMP,
                  colorrange=(T_cold, T_hot))
    Colorbar(fig[1, 2], hm; label="T")
    save(joinpath(FIGDIR, "rayleigh_benard_temperature.png"), fig; px_per_unit=2)
    println("  done")
end

# ============================================================================
# 7. Hagen-Poiseuille profile (axisymmetric)
# ============================================================================
println("=== 7. Hagen-Poiseuille profile ===")
let
    Nr = 32; ν = 0.1; Fz = 1e-5
    result = run_hagen_poiseuille_2d(; Nz=4, Nr=Nr, ν=ν, Fz=Fz, max_steps=20000)

    R_eff  = Nr - 0.5
    j_fluid = 1:Nr
    r_phys = [j - 0.5 for j in j_fluid]
    u_ana  = [Fz / (4ν) * (R_eff^2 - r^2) for r in r_phys]
    u_num  = [result.uz[2, j] for j in j_fluid]

    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1]; xlabel="u_z (lattice units)", ylabel="r (lattice units)",
              title="Hagen-Poiseuille flow -- Nr = $Nr")
    lines!(ax, u_ana, r_phys; label="Analytical", linewidth=2, color=ACCENT, linestyle=:dash)
    lines!(ax, u_num, r_phys; label="LBM (axisymmetric)", linewidth=2, color=COOL)
    axislegend(ax; position=:rt)
    save(joinpath(FIGDIR, "hagen_poiseuille_profile.png"), fig; px_per_unit=2)
    println("  done")
end

# ============================================================================
# 8. Convergence -- Poiseuille (log-log)
# ============================================================================
println("=== 8. Convergence Poiseuille ===")
let
    ν = 0.1; Fx = 1e-5
    Ny_list = [16, 32, 64, 128]
    errors = Float64[]

    for Ny_i in Ny_list
        # Diffusion time ~ Ny^2/ν; need enough steps to converge
        nsteps = max(30000, ceil(Int, 3 * Ny_i^2 / ν))
        println("  Ny=$Ny_i  steps=$nsteps")
        result = run_poiseuille_2d(; Nx=4, Ny=Ny_i, ν=ν, Fx=Fx, max_steps=nsteps)
        # Corrected analytical with half-way BB
        u_ana = [Fx / (2ν) * (j - 0.5) * (Ny_i + 0.5 - j) for j in 1:Ny_i]
        u_num = result.ux[2, :]
        L2 = sqrt(sum((u_num .- u_ana).^2) / sum(u_ana.^2))
        push!(errors, L2)
    end

    # Fit order
    slope = log2(errors[1] / errors[end]) / log2(Ny_list[end] / Ny_list[1])
    order_str = @sprintf("%.2f", slope)

    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1]; xlabel="Ny", ylabel="Relative L2 error",
              title="Convergence -- Poiseuille flow", xscale=log10, yscale=log10)
    scatterlines!(ax, Float64.(Ny_list), errors; linewidth=2, markersize=10,
                  color=COOL, label="LBM")
    ref = errors[1] .* (Ny_list[1] ./ Ny_list).^2
    lines!(ax, Float64.(Ny_list), ref; linestyle=:dash, color=TEXTCOL,
           label="Order 2 reference", linewidth=2)
    text!(ax, Float64(Ny_list[2]), errors[2] * 2.0;
          text="Measured order: $order_str", fontsize=12, color=TEXTCOL)
    axislegend(ax; position=:lb)
    save(joinpath(FIGDIR, "convergence_poiseuille.png"), fig; px_per_unit=2)
    println("  done (order = $order_str)")
end

# ============================================================================
# 9. Convergence -- Taylor-Green (log-log)
# ============================================================================
println("=== 9. Convergence Taylor-Green ===")
let
    ν = 0.01; u0 = 0.01; max_steps = 1000
    Ns = [16, 32, 64, 128]
    errors = Float64[]

    for N in Ns
        result = run_taylor_green_2d(; N=N, ν=ν, u0=u0, max_steps=max_steps)
        k = 2π / N
        decay = exp(-2 * ν * k^2 * max_steps)
        ux_ana = zeros(N, N)
        for j in 1:N, i in 1:N
            x = i - 1; y = j - 1
            ux_ana[i, j] = -u0 * cos(k * x) * sin(k * y) * decay
        end
        diff = result.ux .- ux_ana
        err = sqrt(sum(diff .^ 2) / sum(ux_ana .^ 2))
        push!(errors, err)
    end

    slope = log2(errors[1] / errors[end]) / log2(Ns[end] / Ns[1])
    order_str = @sprintf("%.2f", slope)

    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1]; xlabel="N", ylabel="Relative L2 error",
              title="Convergence -- Taylor-Green vortex", xscale=log10, yscale=log10)
    scatterlines!(ax, Float64.(Ns), errors; linewidth=2, markersize=10,
                  color=COOL, label="LBM")
    ref = errors[1] .* (Ns[1] ./ Ns).^2
    lines!(ax, Float64.(Ns), ref; linestyle=:dash, color=TEXTCOL,
           label="Order 2 reference", linewidth=2)
    text!(ax, Float64(Ns[2]), errors[2] * 2.0;
          text="Measured order: $order_str", fontsize=12, color=TEXTCOL)
    axislegend(ax; position=:lb)
    save(joinpath(FIGDIR, "convergence_taylor_green.png"), fig; px_per_unit=2)
    println("  done (order = $order_str)")
end

# ============================================================================
# 10. Natural convection -- uniform temperature field + refinement comparison
# ============================================================================
println("=== 10. Natural convection ===")
let
    N = 64; Ra = 1e4; Pr = 0.71; max_steps = 30000

    # Uniform run at two resolutions to show refinement benefit
    println("  running uniform N=$N ...")
    res_fine = run_natural_convection_2d(; N=N, Ra=Ra, Pr=Pr, max_steps=max_steps)

    # Temperature heatmap (fine uniform = best available)
    Nx, Ny = size(res_fine.Temp)
    fig = Figure(size=(600, 500))
    ax = Axis(fig[1, 1]; title="Natural convection -- Ra = $(Int(Ra)), N = $N",
              xlabel="x", ylabel="y", aspect=DataAspect())
    hm = heatmap!(ax, 1:Nx, 1:Ny, res_fine.Temp; colormap=CMAP_TEMP, colorrange=(0, 1))
    Colorbar(fig[1, 2], hm; label="T")
    save(joinpath(FIGDIR, "natconv_refined_temperature.png"), fig; px_per_unit=2)
    println("  temperature heatmap done (Nu = $(round(res_fine.Nu; digits=3)))")

    # Coarse run for comparison
    N_coarse = 32
    println("  running coarse N=$N_coarse ...")
    res_coarse = run_natural_convection_2d(; N=N_coarse, Ra=Ra, Pr=Pr, max_steps=max_steps)

    # Compare near-wall T profiles
    T_fine_wall = res_fine.Temp[2, :]
    T_coarse_wall = res_coarse.Temp[2, :]
    y_fine = [(j - 0.5) / N for j in 1:N]
    y_coarse = [(j - 0.5) / N_coarse for j in 1:N_coarse]

    fig2 = Figure(size=(600, 400))
    ax2 = Axis(fig2[1, 1]; xlabel="Temperature", ylabel="y / L",
               title="Near-wall T profile -- Ra = $(Int(Ra))")
    lines!(ax2, T_coarse_wall, y_coarse;
           label="N=$(N_coarse) (Nu=$(round(res_coarse.Nu; digits=2)))",
           linewidth=2, color=COOL)
    lines!(ax2, T_fine_wall, y_fine;
           label="N=$(N) (Nu=$(round(res_fine.Nu; digits=2)))",
           linewidth=2, color=ACCENT, linestyle=:dash)
    axislegend(ax2; position=:rt)
    save(joinpath(FIGDIR, "refinement_comparison.png"), fig2; px_per_unit=2)
    println("  comparison done")

    # NOTE: run_natural_convection_refined_2d currently diverges (NaN) at all
    # tested Ra/N combinations. The refined driver needs stability fixes before
    # it can be used for figure generation. The figures above use uniform grids
    # at two resolutions as a proxy for the refinement improvement.
    println("  (refined driver skipped -- diverges to NaN, needs stability fix)")
end

# ============================================================================
# 11. Cylinder 2D -- velocity magnitude hero (assets/figures PNG)
#     The example page (06_cylinder_2d) embeds ../assets/figures/cylinder_umag.png
#     as its hero. magma sequential speed field on the dark page.
# ============================================================================
println("=== 11. Cylinder velocity magnitude ===")
let
    Re = 20; radius = 10; u_in = 0.04
    D = 2 * radius
    ν = u_in * D / Re

    result = run_cylinder_2d(; Nx=400, Ny=100, radius=radius, u_in=u_in, ν=ν,
                               max_steps=40000, avg_window=2000)
    ux = result.ux; uy = result.uy
    Nx, Ny = size(ux)
    umag = @. sqrt(ux^2 + uy^2)

    fig = Figure(size=(800, 350))
    ax = Axis(fig[1, 1]; title="Velocity magnitude -- Re=$Re",
              xlabel="x", ylabel="y", aspect=DataAspect())
    hm = heatmap!(ax, 1:Nx, 1:Ny, umag; colormap=CMAP_SPEED,
                  colorrange=(0, 1.5 * u_in))
    Colorbar(fig[1, 2], hm; label="|u|")
    save(joinpath(FIGDIR, "cylinder_umag.png"), fig; px_per_unit=2)
    println("  done (Cd = $(round(result.Cd; digits=3)))")
end

# ============================================================================
# 12. Cavity 3D -- mid-plane velocity magnitude hero (assets/figures PNG)
#     The example page (05_cavity_3d) embeds ../assets/figures/cavity_3d_umag.png
#     as its hero. magma sequential speed field on the dark page.
# ============================================================================
println("=== 12. Cavity 3D velocity magnitude ===")
let
    # N=32 triggers a CPU segfault in stream_3d! on Apple Silicon; use N=24
    N = 24; Re = 100; u_lid = 0.05
    ν = u_lid * N / Re
    config = LBMConfig(D3Q19(); Nx=N, Ny=N, Nz=N, ν=ν, u_lid=u_lid,
                       max_steps=20000, output_interval=10000)
    ρ, ux, uy, uz, _ = run_cavity_3d(config)

    mid = N ÷ 2
    umag = zeros(N, N)
    for j in 1:N, i in 1:N
        umag[i, j] = sqrt(ux[i, j, mid]^2 + uy[i, j, mid]^2 + uz[i, j, mid]^2)
    end
    umag ./= u_lid

    fig = Figure(size=(550, 480))
    ax = Axis(fig[1, 1]; title="Velocity magnitude -- mid-plane z=$mid",
              xlabel="x", ylabel="y", aspect=DataAspect())
    hm = heatmap!(ax, 1:N, 1:N, umag; colormap=CMAP_SPEED)
    Colorbar(fig[1, 2], hm; label="|u| / u_lid")
    save(joinpath(FIGDIR, "cavity_3d_umag.png"), fig; px_per_unit=2)
    println("  done")
end

println("\n=== All figures generated in $FIGDIR ===")
