#!/usr/bin/env julia
# Export the real data behind the example-19 species-diffusion concentration
# profile to a CSV served from the docs download dropdown. Columns:
# y, C_analytic, C_kraken — the linear analytical profile C(y) = 1 - y/H and the
# LBM species-transport profile sampled at x = 2 (the SAME curves the committed
# `species_diffusion_profile.svg` figure shows; no fabrication). Mirrors the
# diffusion time loop in 19_species_diffusion.jl.
#
# Run: julia --project=docs docs/export_downloads_species_diffusion.jl

using KrakeNN

const OUTDIR = joinpath(@__DIR__, "src", "public", "downloads", "species_diffusion")
mkpath(OUTDIR)

# Same parameters as docs/src/examples/19_species_diffusion.jl.
Nx, Ny = 4, 32
D_coeff = 0.1
ω_D = 1.0 / (3.0 * D_coeff + 0.5)

h_in  = zeros(Float64, Nx, Ny, 9)
h_out = zeros(Float64, Nx, Ny, 9)
C     = zeros(Float64, Nx, Ny)
ux    = zeros(Float64, Nx, Ny)
uy    = zeros(Float64, Nx, Ny)

w = KrakeNN.weights(D2Q9())
for j in 1:Ny, i in 1:Nx, q in 1:9
    h_in[i, j, q] = w[q] * 0.5
end
copy!(h_out, h_in)

for step in 1:5000
    KrakeNN.stream_periodic_x_wall_y_2d!(h_out, h_in, Nx, Ny)
    apply_fixed_conc_south_2d!(h_out, 1.0, Nx)
    apply_fixed_conc_north_2d!(h_out, 0.0, Nx, Ny)
    collide_species_2d!(h_out, ux, uy, ω_D)
    global h_in, h_out = h_out, h_in
end

compute_concentration_2d!(C, h_in)

C_profile    = C[2, :]
C_analytical = [1.0 - (j - 0.5) / Ny for j in 1:Ny]

csv = joinpath(OUTDIR, "species_diffusion.csv")
open(csv, "w") do io
    println(io, "y,C_analytic,C_kraken")
    for j in 1:Ny
        println(io, string(j - 0.5, ",", C_analytical[j], ",", C_profile[j]))
    end
end

println("✓ wrote $csv ($(Ny) rows)")
