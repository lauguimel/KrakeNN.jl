#!/usr/bin/env julia
# Export the real data behind the example-20 grid-refinement cavity centreline
# profile to a CSV served from the docs download dropdown. Columns:
# y, ux_centerline — the horizontal velocity along the vertical centreline of
# the uniform 64x64 reference lid-driven cavity (the `ux_mid_ref = ux_ref[32, :]`
# array the example computes and compares the refined run against; no
# fabrication). Mirrors the uniform reference `run_cavity_2d` call in
# 20_grid_refinement_cavity.jl.
#
# Run: julia --project=docs docs/export_downloads_grid_refinement_cavity.jl

using KrakeNN

const OUTDIR = joinpath(@__DIR__, "src", "public", "downloads", "grid_refinement_cavity")
mkpath(OUTDIR)

# Same parameters as the uniform reference run in
# docs/src/examples/20_grid_refinement_cavity.jl.
config_ref = LBMConfig(D2Q9(); Nx=64, Ny=64, ν=0.1, u_lid=0.1, max_steps=20000)
ρ_ref, ux_ref, uy_ref, _ = run_cavity_2d(config_ref)

ux_mid_ref = ux_ref[32, :]      # u_x along the vertical centreline
N = length(ux_mid_ref)
u_lid = 0.1

csv = joinpath(OUTDIR, "grid_refinement_cavity.csv")
open(csv, "w") do io
    println(io, "y,ux_centerline")
    for j in 1:N
        println(io, string((j - 0.5) / N, ",", ux_mid_ref[j] / u_lid))
    end
end

println("✓ wrote $csv ($(N) rows)")
