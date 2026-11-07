#!/usr/bin/env julia
# Export the real data behind the example-06 cylinder velocity-magnitude field
# (block 6b of generate_figures_01_09.jl) to a tidy CSV served from the docs
# download dropdown. Columns: x, y, umag — the SAME |u| field the committed
# `cylinder_umag.png` heatmap plots (no fabrication). One row per lattice node.
#
# Run: julia --project=docs docs/export_downloads_cylinder.jl

using KrakeNN

const OUTDIR = joinpath(@__DIR__, "src", "public", "downloads", "cylinder")
mkpath(OUTDIR)

# Same parameters as generate_figures_01_09.jl block "6b. Velocity magnitude".
Re = 20; radius = 10; u_in = 0.04
D = 2 * radius
ν = u_in * D / Re

result = run_cylinder_2d(; Nx=400, Ny=100, radius=radius, u_in=u_in, ν=ν,
                           max_steps=40000, avg_window=2000)
ux = result.ux; uy = result.uy
Nx, Ny = size(ux)
umag = @. sqrt(ux^2 + uy^2)

csv = joinpath(OUTDIR, "cylinder.csv")
open(csv, "w") do io
    println(io, "x,y,umag")
    for j in 1:Ny, i in 1:Nx
        println(io, string(i, ",", j, ",", umag[i, j]))
    end
end

println("✓ wrote $csv ($(Nx * Ny) rows, $(Nx)x$(Ny) field)")
