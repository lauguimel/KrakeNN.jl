#!/usr/bin/env julia
# Export the real data behind the example-09 Hagen-Poiseuille velocity-profile
# figure (block 9b of generate_figures_01_09.jl) to a CSV served from the docs
# download dropdown. Columns: r, u_analytic, u_kraken — the SAME numbers the
# committed `hagen_poiseuille_profile.svg` plots (no fabrication).
#
# Run: julia --project=docs docs/export_downloads_hagen_poiseuille.jl

using KrakeNN

const OUTDIR = joinpath(@__DIR__, "src", "public", "downloads", "hagen_poiseuille")
mkpath(OUTDIR)

# Same parameters as generate_figures_01_09.jl block "9b. Velocity profile".
Nr = 32; ν = 0.1; Fz = 1e-5

ρ, uz, ur, config = run_hagen_poiseuille_2d(; Nz=4, Nr=Nr, ν=ν, Fz=Fz, max_steps=20000)

# Axisymmetric pipe: halfway bounce-back at j=Nr puts the no-slip wall half a
# cell beyond the last fluid node, so the physical pipe radius is R = Nr (not
# Nr - 0.5). Fluid nodes j=1..Nr sit at r = j - 0.5; the analytic parabola is
# u(r) = Fz/(4ν)·(R² - r²). Sampled at z = 2.
R_eff   = Nr
j_fluid = 1:Nr
r_phys  = [j - 0.5 for j in j_fluid]
u_ana   = [Fz / (4ν) * (R_eff^2 - r^2) for r in r_phys]
u_num   = [uz[2, j] for j in j_fluid]

csv = joinpath(OUTDIR, "hagen_poiseuille.csv")
open(csv, "w") do io
    println(io, "r,u_analytic,u_kraken")
    for (r, ua, un) in zip(r_phys, u_ana, u_num)
        println(io, string(r, ",", ua, ",", un))
    end
end

println("✓ wrote $csv ($(length(r_phys)) rows)")
