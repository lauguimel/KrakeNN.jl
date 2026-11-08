#!/usr/bin/env julia
# Export the real data behind the example-04 lid-driven-cavity centreline figure
# (block 4b of generate_figures_01_09.jl) to a CSV served from the docs
# download dropdown. Columns: y_norm, ux_kraken — the SAME vertical-centreline
# numbers the committed `cavity_centerlines.svg` plots (no fabrication). The
# Ghia (1982) reference points sit at different y, so they are exported in their
# own pair of columns (padded with empty cells) for overlay.
#
# Run: julia --project=docs docs/export_downloads_cavity.jl

using KrakeNN

const OUTDIR = joinpath(@__DIR__, "src", "public", "downloads", "cavity")
mkpath(OUTDIR)

# Same parameters as generate_figures_01_09.jl block "4b. Centerline profiles".
N = 128; Re = 100; u_lid = 0.1
ν = u_lid * N / Re
config = LBMConfig(D2Q9(); Nx=N, Ny=N, ν=ν, u_lid=u_lid,
                   max_steps=60000, output_interval=10000)
ρ, ux, uy, _ = run_cavity_2d(config)

# Ghia et al. (1982) reference for Re=100 (vertical centreline u_x / u_lid).
y_ghia  = [0.0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
           0.4531, 0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
           0.9688, 0.9766, 1.0]
ux_ghia = [0.0, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
          -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
           0.68717, 0.73722, 0.78871, 0.84123, 1.0]

mid = N ÷ 2 + 1
ux_profile = [ux[mid, j] / u_lid for j in 1:N]
y_norm     = [(j - 0.5) / N for j in 1:N]

# One tidy CSV: the KrakeNN curve is N rows; the 17 Ghia points are written into
# the same file's first 17 rows so the plot script can scatter them. Rows beyond
# the Ghia count leave those two columns empty.
csv = joinpath(OUTDIR, "cavity.csv")
open(csv, "w") do io
    println(io, "y_norm,ux_kraken,y_ghia,ux_ghia")
    for j in 1:N
        if j <= length(y_ghia)
            println(io, string(y_norm[j], ",", ux_profile[j], ",",
                               y_ghia[j], ",", ux_ghia[j]))
        else
            println(io, string(y_norm[j], ",", ux_profile[j], ",,"))
        end
    end
end

println("✓ wrote $csv ($N rows, $(length(y_ghia)) Ghia points)")
