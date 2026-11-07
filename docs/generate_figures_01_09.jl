#!/usr/bin/env julia
# Generate SVG figures for examples 01–09
# Run: julia --project=docs docs/generate_figures_01_09.jl

using KrakeNN
using CairoMakie

# BC / geometry schematic primitives (pure CairoMakie, no KrakeNN dependency).
# The 7 *_geometry.svg blocks below are assembled exclusively from these so every
# schematic shares one visual identity. See docs/bc_schematic.jl for the contract.
include(joinpath(@__DIR__, "bc_schematic.jl"))

const KA = KrakeNN.KernelAbstractions
const OUTDIR = joinpath(@__DIR__, "src", "examples")

# --- Dark docs theme (#1b1b1f Vitepress page bg, light text/ticks/spines/grid) ---
# Every Figure()/Axis()/Colorbar()/Legend() below inherits this so the exported
# SVGs sit seamlessly on the dark Vitepress page. Mirrors the explicit dark block
# that already produces cylinder_umag.png. Blocks that pass their own
# backgroundcolor (e.g. cylinder_umag) still win locally.
const KRAKEN_DARK = "#1b1b1f"
set_theme!(Theme(
    backgroundcolor = KRAKEN_DARK,
    textcolor = "gray92",
    Axis = (
        backgroundcolor = KRAKEN_DARK,
        titlecolor = "gray92",
        xlabelcolor = "gray92", ylabelcolor = "gray92",
        xticklabelcolor = "gray85", yticklabelcolor = "gray85",
        xtickcolor = "gray70", ytickcolor = "gray70",
        xgridcolor = ("gray60", 0.18), ygridcolor = ("gray60", 0.18),
        leftspinecolor = "gray55", rightspinecolor = "gray55",
        topspinecolor = "gray55", bottomspinecolor = "gray55",
    ),
    Axis3 = (
        backgroundcolor = KRAKEN_DARK,
        titlecolor = "gray92",
        xlabelcolor = "gray92", ylabelcolor = "gray92", zlabelcolor = "gray92",
        xticklabelcolor = "gray85", yticklabelcolor = "gray85", zticklabelcolor = "gray85",
    ),
    Colorbar = (
        labelcolor = "gray92", ticklabelcolor = "gray85", tickcolor = "gray70",
        leftspinecolor = "gray55", rightspinecolor = "gray55",
        topspinecolor = "gray55", bottomspinecolor = "gray55",
    ),
    Legend = (
        backgroundcolor = KRAKEN_DARK, labelcolor = "gray92",
        titlecolor = "gray92", framecolor = "gray45",
    ),
))

# ============================================================================
# === 1. Poiseuille 2D ======================================================
# ============================================================================
println("=== 1. Poiseuille 2D ===")

# --- 1a. Geometry schematic (bc_schematic.jl primitives) ---
let
    fig = Figure(size=(610, 460), backgroundcolor=BC_DARK)
    ax = bc_axis(fig[1, 1]; title="Poiseuille 2D — geometry",
                 limits=(0, 5, 0, 4), pad=1.0)
    fluid_region!(ax, (0.0, 0.0), (5.0, 4.0); outline=false)
    wall!(ax, (0.0, 4.0), (5.0, 4.0); side=1,  label="top wall (bounce-back)",
          labelgap=0.4)
    wall!(ax, (0.0, 0.0), (5.0, 0.0); side=-1, label="bottom wall (bounce-back)",
          labelgap=0.4)
    periodic!(ax, (0.0, 0.0), (0.0, 4.0); label="periodic")
    periodic!(ax, (5.0, 0.0), (5.0, 4.0); label="periodic")
    body_force!(ax, 0.9, 2.0; dx=1.4, dy=0.0, label="Fx (body force)",
                n=3, spread=0.8, labelside=-1, labelgap=0.3)
    save(joinpath(OUTDIR, "poiseuille_geometry.svg"), fig)
    println("  ✓ poiseuille_geometry.svg")
end

# --- 1b. Velocity profile ---
let
    Ny = 32; ν = 0.1; Fx = 1e-5
    ρ, ux, uy, config = run_poiseuille_2d(; Nx=4, Ny=Ny, ν=ν, Fx=Fx, max_steps=20000)

    # Full-way bounce-back (stream_periodic_x_wall_y_2d!) places the no-slip wall
    # HALFWAY between the last fluid node and the ghost layer: at y = 0.5 (below
    # j=1) and y = Ny + 0.5 (above j=Ny). The effective channel height is H = Ny,
    # the physical coordinate of fluid node j is y = j - 0.5, and ALL nodes j=1..Ny
    # are fluid. With these wall-aware coordinates the D2Q9 parabola is exact.
    H = Ny
    j_fluid = 1:Ny
    y_phys = [j - 0.5 for j in j_fluid]
    u_ana  = [Fx / (2ν) * y * (H - y) for y in y_phys]
    u_num  = [ux[2, j] for j in j_fluid]

    fig = Figure(size=(600, 420))
    ax = Axis(fig[1, 1]; xlabel="u_x (lattice units)", ylabel="y (lattice units)",
              title="Poiseuille flow — Ny = $Ny")
    lines!(ax, u_ana, y_phys; label="Analytical", linewidth=2)
    scatter!(ax, u_num, y_phys; label="KrakeNN", markersize=8)
    axislegend(ax; position=:rb)
    save(joinpath(OUTDIR, "poiseuille_profile.svg"), fig)
    println("  ✓ poiseuille_profile.svg")
end

# --- 1c. Convergence ---
let
    ν = 0.1; Fx = 1e-5
    Ny_list = [16, 32, 64, 128]
    errors = Float64[]

    for Ny_i in Ny_list
        H_i = Ny_i                       # halfway-wall channel height (see 1b)
        # Diffusive time to steady state scales as H²/ν, so the step budget MUST
        # grow with resolution — a fixed budget leaves the fine grids transient
        # and corrupts the slope (spurious error blow-up at Ny≥64).
        nsteps = max(30_000, ceil(Int, 8 * H_i^2 / ν))
        ρ_i, ux_i, _, _ = run_poiseuille_2d(; Nx=4, Ny=Ny_i, ν=ν, Fx=Fx, max_steps=nsteps)
        jf  = 1:Ny_i                     # all nodes are fluid
        u_a = [Fx / (2ν) * (j - 0.5) * (H_i - (j - 0.5)) for j in jf]
        u_n = [ux_i[2, j] for j in jf]
        L2  = sqrt(sum((u_n .- u_a).^2) / sum(u_a.^2))
        push!(errors, L2)
    end

    fig = Figure(size=(500, 400))
    ax = Axis(fig[1, 1]; xlabel="Ny", ylabel="Relative L2 error",
              title="Convergence — Poiseuille flow", xscale=log10, yscale=log10)
    scatterlines!(ax, Float64.(Ny_list), errors; linewidth=2, markersize=10, label="KrakeNN")
    ref = errors[1] .* (Ny_list[1] ./ Ny_list).^2
    lines!(ax, Float64.(Ny_list), ref; linestyle=:dash, color=:gray, label="slope 2")
    axislegend(ax; position=:lb)
    save(joinpath(OUTDIR, "poiseuille_convergence.svg"), fig)
    println("  ✓ poiseuille_convergence.svg")
end

# --- 1d. Velocity magnitude field (lead illustration) ---
let
    Ny = 32; Nx = 96; ν = 0.1; Fx = 1e-5
    ρ, ux, uy, _ = run_poiseuille_2d(; Nx=Nx, Ny=Ny, ν=ν, Fx=Fx, max_steps=20000)
    umag = @. sqrt(ux^2 + uy^2)
    fig = Figure(size=(760, 320))
    ax = Axis(fig[1, 1]; title="Velocity magnitude — converged Poiseuille flow",
              xlabel="x", ylabel="y", aspect=DataAspect())
    hm = heatmap!(ax, 1:Nx, 1:Ny, umag; colormap=:viridis)
    Colorbar(fig[1, 2], hm; label="|u|")
    save(joinpath(OUTDIR, "poiseuille_umag.svg"), fig)
    println("  ✓ poiseuille_umag.svg")
end

# ============================================================================
# === 2. Couette 2D =========================================================
# ============================================================================
println("=== 2. Couette 2D ===")

# --- 2a. Geometry schematic (bc_schematic.jl primitives) ---
let
    fig = Figure(size=(600, 460), backgroundcolor=BC_DARK)
    ax = bc_axis(fig[1, 1]; title="Couette 2D — geometry",
                 limits=(0, 5, 0, 4), pad=1.0)
    fluid_region!(ax, (0.0, 0.0), (5.0, 4.0); outline=false)
    wall!(ax, (0.0, 4.0), (5.0, 4.0); side=1, label="top wall (Zou-He)",
          labelgap=0.4)
    moving_wall!(ax, (0.0, 0.0), (5.0, 0.0); side=-1, u_label="u_wall",
                 label="moving wall (Zou-He)", labelgap=0.5, ulabelgap=0.55)
    periodic!(ax, (0.0, 0.0), (0.0, 4.0); label="periodic")
    periodic!(ax, (5.0, 0.0), (5.0, 4.0); label="periodic")
    save(joinpath(OUTDIR, "couette_geometry.svg"), fig)
    println("  ✓ couette_geometry.svg")
end

# --- 2b. Velocity profile ---
let
    Ny = 32; ν = 0.1; u_wall = 0.05
    ρ, ux, uy, config = run_couette_2d(; Nx=4, Ny=Ny, ν=ν, u_wall=u_wall, max_steps=20000)

    H = Ny - 1
    j_fluid = 2:Ny-1
    y_phys = [j - 1 for j in j_fluid]
    u_ana  = [u_wall * (1 - y / H) for y in y_phys]
    u_num  = [ux[2, j] for j in j_fluid]

    fig = Figure(size=(600, 420))
    ax = Axis(fig[1, 1]; xlabel="u_x (lattice units)", ylabel="y (lattice units)",
              title="Couette flow — Ny = $Ny")
    lines!(ax, u_ana, y_phys; label="Analytical", linewidth=2)
    scatter!(ax, u_num, y_phys; label="KrakeNN", markersize=8)
    axislegend(ax; position=:rt)
    save(joinpath(OUTDIR, "couette_profile.svg"), fig)
    println("  ✓ couette_profile.svg")
end

# --- 2c. Convergence ---
let
    ν = 0.1; u_wall = 0.05
    Ny_list = [16, 32, 64, 128]
    errors = Float64[]

    for Ny_i in Ny_list
        H_i = Ny_i - 1
        nsteps = max(10_000, ceil(Int, 3 * H_i^2 / ν))
        ρ_i, ux_i, _, _ = run_couette_2d(; Nx=4, Ny=Ny_i, ν=ν, u_wall=u_wall, max_steps=nsteps)
        # Zou-He pins the wall velocity exactly ON nodes j=1 and j=Ny, so the
        # linear profile is reproduced over EVERY node. Measure over all of them.
        jf  = 1:Ny_i
        u_a = [u_wall * (1 - (j - 1) / H_i) for j in jf]
        u_n = [ux_i[2, j] for j in jf]
        L2  = sqrt(sum((u_n .- u_a).^2) / sum(u_a.^2))
        push!(errors, L2)
    end

    # Couette is EXACT for D2Q9 (linear profile is a 2nd-order-exact lattice
    # solution): the error is pure floating-point roundoff (~1e-14…1e-12) and has
    # NO convergence trend. Plotting a "slope 2" reference here would be a lie, so
    # we show the machine-precision floor honestly with a flat reference band.
    fig = Figure(size=(500, 400))
    ax = Axis(fig[1, 1]; xlabel="Ny", ylabel="Relative L2 error",
              title="Couette — exact to machine precision", xscale=log10, yscale=log10)
    scatterlines!(ax, Float64.(Ny_list), errors; linewidth=2, markersize=10, label="KrakeNN")
    hlines!(ax, [eps(Float64)]; linestyle=:dash, color=:gray, label="machine ε (Float64)")
    ylims!(ax, 1e-16, 1e-9)
    axislegend(ax; position=:lt)
    save(joinpath(OUTDIR, "couette_convergence.svg"), fig)
    println("  ✓ couette_convergence.svg")
end

# --- 2d. Velocity magnitude field (lead illustration) ---
let
    Ny = 32; Nx = 96; ν = 0.1; u_wall = 0.05
    ρ, ux, uy, _ = run_couette_2d(; Nx=Nx, Ny=Ny, ν=ν, u_wall=u_wall, max_steps=20000)
    umag = @. sqrt(ux^2 + uy^2)
    fig = Figure(size=(760, 320))
    ax = Axis(fig[1, 1]; title="Velocity magnitude — converged Couette flow",
              xlabel="x", ylabel="y", aspect=DataAspect())
    hm = heatmap!(ax, 1:Nx, 1:Ny, umag; colormap=:viridis)
    Colorbar(fig[1, 2], hm; label="|u|")
    save(joinpath(OUTDIR, "couette_umag.svg"), fig)
    println("  ✓ couette_umag.svg")
end

# ============================================================================
# === 3. Taylor-Green 2D ====================================================
# ============================================================================
println("=== 3. Taylor-Green 2D ===")

# --- 3a. Initial vorticity field ---
let
    N = 64; u0 = 0.04; ν = 0.01
    # Compute initial vorticity analytically
    k = 2pi / N
    ωz_init = zeros(N, N)
    for j in 1:N, i in 1:N
        x = i - 1.0
        y = j - 1.0
        # ω_z = ∂uy/∂x - ∂ux/∂y = u0*k*cos(kx)*cos(ky) + u0*k*cos(kx)*cos(ky)
        ωz_init[i, j] = 2.0 * u0 * k * cos(k * x) * cos(k * y)
    end

    fig = Figure(size=(500, 450))
    ax = Axis(fig[1, 1]; title="Initial vorticity field", aspect=DataAspect())
    hm = heatmap!(ax, 1:N, 1:N, ωz_init; colormap=:balance)
    Colorbar(fig[1, 2], hm; label="omega_z")
    save(joinpath(OUTDIR, "taylor_green_geometry.svg"), fig)
    println("  ✓ taylor_green_geometry.svg")
end

# --- 3b. Energy decay ---
let
    N = 64; u0 = 0.04; ν = 0.01
    k = 2pi / N
    # The velocity field u = u0·(−cos kx sin ky, sin kx cos ky) decays as
    # exp(−2νk²t), so the kinetic energy ∝ u² decays as exp(−4νk²t). The MEAN
    # kinetic-energy density of that field is E0 = ⟨½(ux²+uy²)⟩ = u0²/4 (the cos²/sin²
    # spatial averages each give ¼), NOT the peak ½u0². Using ½u0² over-predicts the
    # baseline by 2× and exp(−2νk²t) halves the slope — both were wrong before.
    E0 = u0^2 / 4

    steps_list = 0:500:5000
    E_num = Float64[]
    E_ana = Float64[]

    for s in steps_list
        if s == 0
            push!(E_num, E0)
        else
            res_s = run_taylor_green_2d(; N=N, ν=ν, u0=u0, max_steps=s)
            ux_s = res_s.ux; uy_s = res_s.uy
            KE = 0.0
            for j in 1:N, i in 1:N
                KE += 0.5 * (ux_s[i, j]^2 + uy_s[i, j]^2)
            end
            push!(E_num, KE / (N * N))
        end
        push!(E_ana, E0 * exp(-4ν * k^2 * s))
    end

    fig = Figure(size=(600, 420))
    ax = Axis(fig[1, 1]; xlabel="Time step", ylabel="Mean kinetic energy",
              title="Taylor-Green vortex decay — N = $N")
    lines!(ax, collect(steps_list), E_ana; label="Analytical", linewidth=2)
    scatter!(ax, collect(steps_list), E_num; label="KrakeNN", markersize=10)
    axislegend(ax; position=:rt)
    save(joinpath(OUTDIR, "taylor_green_decay.svg"), fig)
    println("  ✓ taylor_green_decay.svg")
end

# --- 3c. Vorticity at final time ---
let
    N = 64; u0 = 0.04; ν = 0.01
    ρ, ux, uy, config, u0_out, k, max_steps = run_taylor_green_2d(;
        N=N, ν=ν, u0=u0, max_steps=5000)

    ωz = zeros(N, N)
    for j in 1:N, i in 1:N
        ip = mod1(i + 1, N); im = mod1(i - 1, N)
        jp = mod1(j + 1, N); jm = mod1(j - 1, N)
        ωz[i, j] = 0.5 * (uy[ip, j] - uy[im, j]) - 0.5 * (ux[i, jp] - ux[i, jm])
    end

    fig = Figure(size=(500, 450))
    ax = Axis(fig[1, 1]; title="Vorticity at t = $max_steps", aspect=DataAspect())
    hm = heatmap!(ax, 1:N, 1:N, ωz; colormap=:balance)
    Colorbar(fig[1, 2], hm; label="omega_z")
    save(joinpath(OUTDIR, "taylor_green_vorticity.svg"), fig)
    println("  ✓ taylor_green_vorticity.svg")
end

# --- 3d. Velocity magnitude field (lead illustration) ---
let
    N = 64; u0 = 0.04; ν = 0.01
    res = run_taylor_green_2d(; N=N, ν=ν, u0=u0, max_steps=2000)
    ux = res.ux; uy = res.uy
    umag = @. sqrt(ux^2 + uy^2)
    fig = Figure(size=(520, 470))
    ax = Axis(fig[1, 1]; title="Velocity magnitude — Taylor-Green vortex",
              xlabel="x", ylabel="y", aspect=DataAspect())
    hm = heatmap!(ax, 1:N, 1:N, umag; colormap=:viridis)
    Colorbar(fig[1, 2], hm; label="|u|")
    save(joinpath(OUTDIR, "taylor_green_umag.svg"), fig)
    println("  ✓ taylor_green_umag.svg")
end

# ============================================================================
# === 4. Cavity 2D ==========================================================
# ============================================================================
println("=== 4. Cavity 2D ===")

# --- 4a. Geometry schematic (bc_schematic.jl primitives) ---
let
    fig = Figure(size=(500, 560), backgroundcolor=BC_DARK)
    ax = bc_axis(fig[1, 1]; title="Lid-driven cavity — geometry",
                 limits=(0, 5, 0, 5), pad=1.2)
    fluid_region!(ax, (0.0, 0.0), (5.0, 5.0); outline=false)
    moving_wall!(ax, (0.0, 5.0), (5.0, 5.0); side=1, u_label="u_lid",
                 label="moving lid (Zou-He)", labelgap=0.5, ulabelgap=0.55)
    wall!(ax, (0.0, 0.0), (5.0, 0.0); side=-1, label="bottom wall", labelgap=0.45)
    wall!(ax, (0.0, 0.0), (0.0, 5.0); side=1,  label="left wall", labelgap=0.45)
    wall!(ax, (5.0, 0.0), (5.0, 5.0); side=-1, label="right wall", labelgap=0.45)
    save(joinpath(OUTDIR, "cavity_geometry.svg"), fig)
    println("  ✓ cavity_geometry.svg")
end

# --- 4b. Centerline profiles vs Ghia ---
let
    N = 128; Re = 100; u_lid = 0.1
    ν = u_lid * N / Re
    config = LBMConfig(D2Q9(); Nx=N, Ny=N, ν=ν, u_lid=u_lid,
                       max_steps=60000, output_interval=10000)
    ρ, ux, uy, _ = run_cavity_2d(config)

    # Ghia et al. data for Re=100
    y_ghia  = [0.0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
               0.4531, 0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
               0.9688, 0.9766, 1.0]
    ux_ghia = [0.0, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
              -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
               0.68717, 0.73722, 0.78871, 0.84123, 1.0]

    mid = N ÷ 2 + 1
    ux_profile = [ux[mid, j] / u_lid for j in 1:N]
    y_norm     = [(j - 0.5) / N for j in 1:N]
    uy_profile = [uy[i, mid] / u_lid for i in 1:N]
    x_norm     = [(i - 0.5) / N for i in 1:N]

    fig = Figure(size=(900, 420))
    ax1 = Axis(fig[1, 1]; xlabel="u_x / u_lid", ylabel="y / N",
               title="Vertical centreline")
    lines!(ax1, ux_profile, y_norm; label="KrakeNN (N=$N)", linewidth=2)
    scatter!(ax1, ux_ghia, y_ghia; label="Ghia et al.", color=:red, markersize=8)
    axislegend(ax1; position=:lb)

    ax2 = Axis(fig[1, 2]; xlabel="x / N", ylabel="u_y / u_lid",
               title="Horizontal centreline")
    lines!(ax2, x_norm, uy_profile; label="KrakeNN (N=$N)", linewidth=2)
    axislegend(ax2; position=:rt)
    save(joinpath(OUTDIR, "cavity_centerlines.svg"), fig)
    println("  ✓ cavity_centerlines.svg")

    # --- 4c. Velocity magnitude ---
    umag = @. sqrt(ux^2 + uy^2) / u_lid
    fig2 = Figure(size=(500, 480))
    ax3 = Axis(fig2[1, 1]; title="Velocity magnitude — Re=$Re", aspect=DataAspect())
    hm = heatmap!(ax3, 1:N, 1:N, umag; colormap=:viridis)
    Colorbar(fig2[1, 2], hm; label="|u| / u_lid")
    save(joinpath(OUTDIR, "cavity_umag.svg"), fig2)
    println("  ✓ cavity_umag.svg")
end

# ============================================================================
# === 5. Cavity 3D ==========================================================
# ============================================================================
println("=== 5. Cavity 3D ===")

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
    ax = Axis(fig[1, 1]; title="Velocity magnitude — mid-plane z=$mid",
              xlabel="x", ylabel="y", aspect=DataAspect())
    hm = heatmap!(ax, 1:N, 1:N, umag; colormap=:viridis)
    Colorbar(fig[1, 2], hm; label="|u| / u_lid")
    save(joinpath(OUTDIR, "cavity_3d_umag.svg"), fig)
    println("  ✓ cavity_3d_umag.svg")
end

# ============================================================================
# === 6. Cylinder 2D ========================================================
# ============================================================================
println("=== 6. Cylinder 2D ===")

# --- 6a. Geometry schematic (bc_schematic.jl primitives) ---
let
    fig = Figure(size=(730, 360), backgroundcolor=BC_DARK)
    ax = bc_axis(fig[1, 1]; title="Flow around a cylinder — geometry",
                 limits=(0, 400, 0, 100), pad=22.0)
    fluid_region!(ax, (0.0, 0.0), (400.0, 100.0); outline=false)
    # top & bottom free-slip (plain grey line, NO hatch)
    free_slip!(ax, (0.0, 100.0), (400.0, 100.0); label="free-slip", labelside=-1,
               labelgap=11.0)
    free_slip!(ax, (0.0, 0.0), (400.0, 0.0); label="free-slip", labelside=1,
               labelgap=11.0)
    # uniform inlet on the left edge, arrows pointing INTO the domain (+x)
    inlet!(ax, (0.0, 8.0), (0.0, 92.0); profile=:uniform, label="u_in",
           depth=34.0, n=5, side=1, labelsize=12, labelgap=14.0)
    # outlet on the right edge, arrows leaving the domain (+x)
    outlet!(ax, (400.0, 8.0), (400.0, 92.0); label="outflow", side=-1, n=4,
            depth=30.0, labelsize=12, labelgap=42.0)
    # immersed cylinder
    obstacle!(ax, (90.0, 50.0), 12.0; label="cylinder", labelside=:below,
              labelgap=3.0)
    save(joinpath(OUTDIR, "cylinder_geometry.svg"), fig)
    println("  ✓ cylinder_geometry.svg")
end

# --- 6b. Velocity magnitude & drag ---
let
    Re = 20; radius = 10; u_in = 0.04
    D = 2 * radius
    ν = u_in * D / Re

    result = run_cylinder_2d(; Nx=400, Ny=100, radius=radius, u_in=u_in, ν=ν,
                               max_steps=40000, avg_window=2000)
    ux = result.ux; uy = result.uy; Cd = result.Cd
    Nx, Ny = size(ux)
    umag = @. sqrt(ux^2 + uy^2)

    # Dark docs styling (#1b1b1f Vitepress page bg, magma sequential field, light
    # text) to match the dark page. Saved as PNG — the per-cell heatmap
    # exports to a multi-MB SVG, so raster it directly.
    DARK = "#1b1b1f"
    fig = Figure(size=(800, 350), backgroundcolor=DARK)
    ax = Axis(fig[1, 1]; backgroundcolor=DARK,
              title="Velocity magnitude — Re=$Re", titlecolor="gray92",
              xlabel="x", ylabel="y", xlabelcolor="gray92", ylabelcolor="gray92",
              xticklabelcolor="gray85", yticklabelcolor="gray85",
              xtickcolor="gray70", ytickcolor="gray70",
              leftspinecolor="gray55", rightspinecolor="gray55",
              topspinecolor="gray55", bottomspinecolor="gray55",
              aspect=DataAspect())
    hm = heatmap!(ax, 1:Nx, 1:Ny, umag; colormap=:magma,
                  colorrange=(0, 1.5 * u_in))
    Colorbar(fig[1, 2], hm; label="|u|", labelcolor="gray92",
             ticklabelcolor="gray85", tickcolor="gray70")
    save(joinpath(OUTDIR, "cylinder_umag.png"), fig, px_per_unit=2)
    println("  ✓ cylinder_umag.png")

    # --- 6c. Drag comparison bar chart ---
    Cd_ref = 5.58
    fig2 = Figure(size=(400, 300))
    ax2 = Axis(fig2[1, 1]; title="Drag comparison at Re = $Re")
    barplot!(ax2, [1, 2], [Cd, Cd_ref]; color=[:steelblue, :tomato],
             bar_labels=[string(round(Cd; digits=3)), string(Cd_ref)],
             label_color="gray92")
    ax2.xticks = ([1, 2], ["KrakeNN", "Schafer-Turek"])
    ax2.ylabel = "Cd"
    save(joinpath(OUTDIR, "cylinder_drag.svg"), fig2)
    println("  ✓ cylinder_drag.svg")
end

# ============================================================================
# === 7. Heat Conduction =====================================================
# ============================================================================
println("=== 7. Heat Conduction ===")

# --- 7a. Geometry schematic (bc_schematic.jl primitives) ---
let
    fig = Figure(size=(630, 440), backgroundcolor=BC_DARK)
    ax = bc_axis(fig[1, 1]; title="Heat conduction — geometry",
                 limits=(0, 7, 0, 4), pad=1.1)
    fluid_region!(ax, (0.0, 0.0), (7.0, 4.0); outline=false)
    dirichlet_wall!(ax, (0.0, 4.0), (7.0, 4.0); kind=:cold, side=1,
                    label="T_cold (top)", labelgap=0.5)
    dirichlet_wall!(ax, (0.0, 0.0), (7.0, 0.0); kind=:hot, side=-1,
                    label="T_hot (bottom)", labelgap=0.5)
    periodic!(ax, (0.0, 0.0), (0.0, 4.0); label="periodic")
    periodic!(ax, (7.0, 0.0), (7.0, 4.0); label="periodic")
    save(joinpath(OUTDIR, "heat_geometry.svg"), fig)
    println("  ✓ heat_geometry.svg")
end

# --- 7b. Temperature profile ---
let
    Ra = 100.0; Pr = 1.0; T_hot = 1.0; T_cold = 0.0
    ρ, ux, uy, Temp, config, Ra_out, Pr_out, ν, α = run_rayleigh_benard_2d(;
        Nx=128, Ny=32, Ra=Ra, Pr=Pr, T_hot=T_hot, T_cold=T_cold, max_steps=20000)

    Ny = size(Temp, 2)
    H = Ny - 1
    j_fluid = 2:Ny-1
    y_phys = [(j - 1.5) / H for j in j_fluid]
    T_ana  = [T_hot - (T_hot - T_cold) * y for y in y_phys]
    T_num  = [Temp[64, j] for j in j_fluid]

    fig = Figure(size=(600, 420))
    ax = Axis(fig[1, 1]; xlabel="Temperature", ylabel="y / H",
              title="Heat conduction — Ra = $Ra (sub-critical)")
    lines!(ax, T_ana, y_phys; label="Analytical (linear)", linewidth=2)
    scatter!(ax, T_num, y_phys; label="KrakeNN", markersize=8)
    axislegend(ax; position=:rt)
    save(joinpath(OUTDIR, "heat_profile.svg"), fig)
    println("  ✓ heat_profile.svg")
end

# --- 7c. Temperature field (lead illustration; no flow → use the thermal field) ---
let
    Ra = 100.0; Pr = 1.0; T_hot = 1.0; T_cold = 0.0
    ρ, ux, uy, Temp, _ = run_rayleigh_benard_2d(;
        Nx=128, Ny=32, Ra=Ra, Pr=Pr, T_hot=T_hot, T_cold=T_cold, max_steps=20000)
    Nx, Ny = size(Temp)
    fig = Figure(size=(800, 320))
    ax = Axis(fig[1, 1]; title="Steady-state temperature — 1D heat conduction",
              xlabel="x", ylabel="y", aspect=DataAspect())
    hm = heatmap!(ax, 1:Nx, 1:Ny, Temp; colormap=:thermal, colorrange=(T_cold, T_hot))
    Colorbar(fig[1, 2], hm; label="T")
    save(joinpath(OUTDIR, "heat_temperature.svg"), fig)
    println("  ✓ heat_temperature.svg")
end

# ============================================================================
# === 8. Rayleigh-Benard =====================================================
# ============================================================================
println("=== 8. Rayleigh-Benard ===")

# --- 8a. Geometry schematic (bc_schematic.jl primitives) ---
let
    fig = Figure(size=(670, 440), backgroundcolor=BC_DARK)
    ax = bc_axis(fig[1, 1]; title="Rayleigh-Benard convection — geometry",
                 limits=(0, 9, 0, 4), pad=1.1)
    fluid_region!(ax, (0.0, 0.0), (9.0, 4.0); outline=false)
    dirichlet_wall!(ax, (0.0, 4.0), (9.0, 4.0); kind=:cold, side=1,
                    label="T_cold (top)", labelgap=0.5)
    dirichlet_wall!(ax, (0.0, 0.0), (9.0, 0.0); kind=:hot, side=-1,
                    label="T_hot (bottom)", labelgap=0.5)
    periodic!(ax, (0.0, 0.0), (0.0, 4.0); label="periodic")
    periodic!(ax, (9.0, 0.0), (9.0, 4.0); label="periodic")
    # light convection-roll cue in a NEUTRAL secondary grey (accent/cool reserved)
    for (xc, sgn) in ((2.5, 1.0), (6.5, -1.0))
        θ = range(0, 2π; length=64)
        rx, ry = 1.05, 1.25
        cx, cy = xc, 2.0
        lines!(ax, [cx + rx * cos(t) for t in θ], [cy + ry * sin(t) for t in θ];
               color=("gray70", 0.55), linewidth=1.3)
        ang = sgn > 0 ? π/2 : -π/2
        hx, hy = cx + rx * cos(ang), cy + ry * sin(ang)
        tx = -rx * sin(ang) * sgn
        ty =  ry * cos(ang) * sgn
        tn = sqrt(tx^2 + ty^2)
        arrows2d!(ax, [hx], [hy], [tx / tn * 0.6], [ty / tn * 0.6];
                  color=("gray70", 0.7), shaftwidth=1.4, tipwidth=7, tiplength=7)
    end
    text!(ax, 4.5, 2.0; text="convection rolls", color=BC_TEXT, font=BC_SERIF,
          fontsize=11, align=(:center, :center))
    save(joinpath(OUTDIR, "rayleigh_benard_geometry.svg"), fig)
    println("  ✓ rayleigh_benard_geometry.svg")
end

# --- 8b. Temperature field & velocity magnitude ---
let
    Ra = 5000.0; Pr = 1.0; T_hot = 1.0; T_cold = 0.0
    ρ, ux, uy, Temp, config, Ra_out, Pr_out, ν, α = run_rayleigh_benard_2d(;
        Nx=128, Ny=32, Ra=Ra, Pr=Pr, T_hot=T_hot, T_cold=T_cold, max_steps=30000)

    Nx, Ny = size(Temp)

    fig = Figure(size=(800, 350))
    ax = Axis(fig[1, 1]; title="Temperature — Ra = $Ra, Pr = $Pr",
              xlabel="x", ylabel="y", aspect=DataAspect())
    hm = heatmap!(ax, 1:Nx, 1:Ny, Temp; colormap=:thermal, colorrange=(T_cold, T_hot))
    Colorbar(fig[1, 2], hm; label="T")
    save(joinpath(OUTDIR, "rayleigh_benard_temperature.svg"), fig)
    println("  ✓ rayleigh_benard_temperature.svg")

    umag = @. sqrt(ux^2 + uy^2)
    fig2 = Figure(size=(800, 350))
    ax2 = Axis(fig2[1, 1]; title="Velocity magnitude — Ra = $Ra",
               xlabel="x", ylabel="y", aspect=DataAspect())
    hm2 = heatmap!(ax2, 1:Nx, 1:Ny, umag; colormap=:viridis)
    Colorbar(fig2[1, 2], hm2; label="|u|")
    save(joinpath(OUTDIR, "rayleigh_benard_velocity.svg"), fig2)
    println("  ✓ rayleigh_benard_velocity.svg")
end

# ============================================================================
# === 9. Hagen-Poiseuille ===================================================
# ============================================================================
println("=== 9. Hagen-Poiseuille ===")

# --- 9a. Geometry schematic (bc_schematic.jl primitives) ---
let
    fig = Figure(size=(630, 430), backgroundcolor=BC_DARK)
    ax = bc_axis(fig[1, 1]; title="Hagen-Poiseuille — pipe cross-section (z-r plane)",
                 limits=(0, 6, 0, 4), pad=1.2)
    fluid_region!(ax, (0.0, 0.0), (6.0, 4.0); outline=false)
    wall!(ax, (0.0, 4.0), (6.0, 4.0); side=1, label="wall r=R", labelgap=0.45)
    symmetry!(ax, (0.0, 0.0), (6.0, 0.0); label="symmetry r=0")
    periodic!(ax, (0.0, 0.0), (0.0, 4.0); label="periodic z")
    periodic!(ax, (6.0, 0.0), (6.0, 4.0); label="periodic z")
    body_force!(ax, 1.0, 2.0; dx=1.5, dy=0.0, label="Fz (body force)",
                n=3, spread=0.85, labelside=1, labelgap=0.35)
    # axis labels: z horizontal, r vertical
    text!(ax, 6.3, 0.0; text="z", color=BC_TEXT, font=BC_SERIF, fontsize=15,
          align=(:left, :center))
    text!(ax, 0.0, 4.4; text="r", color=BC_TEXT, font=BC_SERIF, fontsize=15,
          align=(:center, :bottom))
    save(joinpath(OUTDIR, "hagen_poiseuille_geometry.svg"), fig)
    println("  ✓ hagen_poiseuille_geometry.svg")
end

# --- 9b. Velocity profile ---
let
    Nr = 32; ν = 0.1; Fz = 1e-5
    ρ, uz, ur, config = run_hagen_poiseuille_2d(; Nz=4, Nr=Nr, ν=ν, Fz=Fz, max_steps=20000)

    R_eff  = Nr  # halfway BB wall at j=Nr → physical radius R = Nr (not Nr-0.5)
    j_fluid = 1:Nr
    r_phys = [j - 0.5 for j in j_fluid]
    u_ana  = [Fz / (4ν) * (R_eff^2 - r^2) for r in r_phys]
    u_num  = [uz[2, j] for j in j_fluid]

    fig = Figure(size=(600, 420))
    ax = Axis(fig[1, 1]; xlabel="u_z (lattice units)", ylabel="r (lattice units)",
              title="Hagen-Poiseuille flow — Nr = $Nr")
    lines!(ax, u_ana, r_phys; label="Analytical", linewidth=2)
    scatter!(ax, u_num, r_phys; label="KrakeNN (axisymmetric)", markersize=8)
    axislegend(ax; position=:rt)
    save(joinpath(OUTDIR, "hagen_poiseuille_profile.svg"), fig)
    println("  ✓ hagen_poiseuille_profile.svg")
end

println("\n=== All figures generated! ===")
