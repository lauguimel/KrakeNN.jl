#!/usr/bin/env julia
# Export the real data behind the example-17 non-Newtonian Poiseuille profile to
# a CSV served from the docs download dropdown. Columns: y, u_analytic, u_kraken
# — the power-law (n = 0.7) analytical profile and the LBM profile sampled at
# x = 2 (the SAME curves the `poiseuille_rheology_profile.svg` figure shows; no
# fabrication). Mirrors the power-law time loop in 17_poiseuille_rheology.jl.
#
# Run: julia --project=docs docs/export_downloads_poiseuille_rheology.jl

using KrakeNN

const OUTDIR = joinpath(@__DIR__, "src", "public", "downloads", "poiseuille_rheology")
mkpath(OUTDIR)

# Same parameters as docs/src/examples/17_poiseuille_rheology.jl (power-law).
Nx, Ny = 4, 32
K   = 0.1
n   = 0.7
Fx  = 1e-4
max_steps = 50000

rheology = PowerLaw(K, n)

f_in  = zeros(Float64, Nx, Ny, 9)
f_out = zeros(Float64, Nx, Ny, 9)
is_solid = falses(Nx, Ny)
tau_field = fill(3.0 * K + 0.5, Nx, Ny)

for j in 1:Ny, i in 1:Nx, q in 1:9
    f_in[i, j, q] = KrakeNN.equilibrium(D2Q9(), 1.0, 0.0, 0.0, q)
end
copy!(f_out, f_in)

Fx_arr = fill(Float64(Fx), Nx, Ny)
Fy_arr = zeros(Float64, Nx, Ny)

for step in 1:max_steps
    stream_periodic_x_wall_y_2d!(f_out, f_in, Nx, Ny)
    collide_rheology_guo_2d!(f_out, is_solid, rheology, tau_field, Fx_arr, Fy_arr)
    global f_in, f_out = f_out, f_in
end

# LBM velocity profile (sampled at x = 2, fully developed).
ρ  = zeros(Float64, Nx, Ny)
ux = zeros(Float64, Nx, Ny)
uy = zeros(Float64, Nx, Ny)
compute_macroscopic_2d!(ρ, ux, uy, f_in)
u_num = ux[2, :]

# Power-law analytical profile.
H = Float64(Ny)
u_ana = zeros(Ny)
for j in 1:Ny
    y = j - 0.5
    dist = abs(y - H / 2)
    u_ana[j] = n / (n + 1) * (Fx / K)^(1 / n) *
               ((H / 2)^((n + 1) / n) - dist^((n + 1) / n))
end

csv = joinpath(OUTDIR, "poiseuille_rheology.csv")
open(csv, "w") do io
    println(io, "y,u_analytic,u_kraken")
    for j in 1:Ny
        println(io, string(j - 0.5, ",", u_ana[j], ",", u_num[j]))
    end
end

println("✓ wrote $csv ($(Ny) rows)")
