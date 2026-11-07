#!/usr/bin/env julia
# Export the real data behind the example-08 Rayleigh-Benard temperature field
# (block 8b of generate_figures_01_09.jl) to a tidy CSV served from the docs
# download dropdown. Columns: x, y, T — the SAME temperature field the committed
# `rayleigh_benard_temperature.svg` heatmap plots (no fabrication). One row per
# lattice node.
#
# Run: julia --project=docs docs/export_downloads_rayleigh_benard.jl

using KrakeNN

const OUTDIR = joinpath(@__DIR__, "src", "public", "downloads", "rayleigh_benard")
mkpath(OUTDIR)

# Same parameters as generate_figures_01_09.jl block "8b. Temperature field".
Ra = 5000.0; Pr = 1.0; T_hot = 1.0; T_cold = 0.0
ρ, ux, uy, Temp, config, Ra_out, Pr_out, ν, α = run_rayleigh_benard_2d(;
    Nx=128, Ny=32, Ra=Ra, Pr=Pr, T_hot=T_hot, T_cold=T_cold, max_steps=30000)

Nx, Ny = size(Temp)

csv = joinpath(OUTDIR, "rayleigh_benard.csv")
open(csv, "w") do io
    println(io, "x,y,T")
    for j in 1:Ny, i in 1:Nx
        println(io, string(i, ",", j, ",", Temp[i, j]))
    end
end

println("✓ wrote $csv ($(Nx * Ny) rows, $(Nx)x$(Ny) field)")
