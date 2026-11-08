#!/usr/bin/env julia
# Export the real data behind the example-02 Couette velocity-profile figure
# (block 2b of generate_figures_01_09.jl) to a CSV served from the docs
# download dropdown. Columns: y, u_analytic, u_kraken — the SAME numbers the
# committed `couette_profile.svg` plots (no fabrication).
#
# Run: julia --project=docs docs/export_downloads_couette.jl

using KrakeNN

const OUTDIR = joinpath(@__DIR__, "src", "public", "downloads", "couette")
mkpath(OUTDIR)

# Same parameters as generate_figures_01_09.jl block "2b. Velocity profile".
Ny = 32
ν  = 0.1
u_wall = 0.05

ρ, ux, uy, config = run_couette_2d(; Nx=4, Ny=Ny, ν=ν, u_wall=u_wall, max_steps=20000)

# Zou-He moving wall at j=1 (u_wall) and stationary wall at j=Ny: effective
# channel height H = Ny - 1, fluid nodes j=2..Ny-1 with physical y = j - 1, and
# the analytic linear profile is u(y) = u_wall·(1 - y/H). Sampled at x = 2.
H       = Ny - 1
j_fluid = 2:Ny-1
y_phys  = [j - 1 for j in j_fluid]
u_ana   = [u_wall * (1 - y / H) for y in y_phys]
u_num   = [ux[2, j] for j in j_fluid]

csv = joinpath(OUTDIR, "couette.csv")
open(csv, "w") do io
    println(io, "y,u_analytic,u_kraken")
    for (y, ua, un) in zip(y_phys, u_ana, u_num)
        println(io, string(y, ",", ua, ",", un))
    end
end

println("✓ wrote $csv ($(length(y_phys)) rows)")
