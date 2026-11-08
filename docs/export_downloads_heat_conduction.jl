#!/usr/bin/env julia
# Export the real data behind the example-07 heat-conduction temperature-profile
# figure (block 7b of generate_figures_01_09.jl) to a CSV served from the docs
# download dropdown. Columns: y_over_H, T_analytic, T_kraken — the SAME numbers
# the committed `heat_profile.svg` plots (no fabrication).
#
# Run: julia --project=docs docs/export_downloads_heat_conduction.jl

using KrakeNN

const OUTDIR = joinpath(@__DIR__, "src", "public", "downloads", "heat_conduction")
mkpath(OUTDIR)

# Same parameters as generate_figures_01_09.jl block "7b. Temperature profile".
Ra = 100.0; Pr = 1.0; T_hot = 1.0; T_cold = 0.0
ρ, ux, uy, Temp, config, Ra_out, Pr_out, ν, α = run_rayleigh_benard_2d(;
    Nx=128, Ny=32, Ra=Ra, Pr=Pr, T_hot=T_hot, T_cold=T_cold, max_steps=20000)

# Sub-critical Ra -> pure conduction: linear T(y) = T_hot - (T_hot-T_cold)·y/H.
# Fluid nodes j=2..Ny-1 with y/H = (j - 1.5)/H, sampled at column x = 64.
Ny = size(Temp, 2)
H = Ny - 1
j_fluid = 2:Ny-1
y_phys = [(j - 1.5) / H for j in j_fluid]
T_ana  = [T_hot - (T_hot - T_cold) * y for y in y_phys]
T_num  = [Temp[64, j] for j in j_fluid]

csv = joinpath(OUTDIR, "heat_conduction.csv")
open(csv, "w") do io
    println(io, "y_over_H,T_analytic,T_kraken")
    for (y, ta, tn) in zip(y_phys, T_ana, T_num)
        println(io, string(y, ",", ta, ",", tn))
    end
end

println("✓ wrote $csv ($(length(y_phys)) rows)")
