#!/usr/bin/env julia
# Export the real data behind the example-01 Poiseuille velocity-profile figure
# (block 1b of generate_figures_01_09.jl) to a CSV served from the docs
# download dropdown. Columns: y, u_analytic, u_kraken — the SAME numbers the
# committed `poiseuille_profile.svg` plots (no fabrication).
#
# Run: julia --project=docs docs/export_downloads_poiseuille.jl

using KrakeNN

const OUTDIR = joinpath(@__DIR__, "src", "public", "downloads", "poiseuille")
mkpath(OUTDIR)

# Same parameters as generate_figures_01_09.jl block "1b. Velocity profile".
Ny = 32
ν  = 0.1
Fx = 1e-5

ρ, ux, uy, config = run_poiseuille_2d(; Nx=4, Ny=Ny, ν=ν, Fx=Fx, max_steps=20000)

# Half-way bounce-back: H = Ny, fluid node j has physical y = j - 0.5, the
# analytic parabola is u(y) = Fx/(2ν)·y·(H - y), and the numerical profile is
# sampled at x = 2 (fully developed → identical for every x).
H       = Ny
j_fluid = 1:Ny
y_phys  = [j - 0.5 for j in j_fluid]
u_ana   = [Fx / (2ν) * y * (H - y) for y in y_phys]
u_num   = [ux[2, j] for j in j_fluid]

csv = joinpath(OUTDIR, "poiseuille.csv")
open(csv, "w") do io
    println(io, "y,u_analytic,u_kraken")
    for (y, ua, un) in zip(y_phys, u_ana, u_num)
        println(io, string(y, ",", ua, ",", un))
    end
end

println("✓ wrote $csv ($(length(y_phys)) rows)")
