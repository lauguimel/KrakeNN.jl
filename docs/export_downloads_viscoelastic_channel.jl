#!/usr/bin/env julia
# Export the real data behind the example-21 viscoelastic (Oldroyd-B) channel
# velocity profile to a CSV served from the docs download dropdown. Columns:
# y, u_analytic, u_kraken — the analytical Poiseuille profile with total
# viscosity and the LBM log-conformation profile sampled at x = 2 (the SAME
# curves the `viscoelastic_velocity.svg` figure shows; no fabrication). Mirrors
# the log-conformation time loop in 21_viscoelastic_channel.jl.
#
# Run: julia --project=docs docs/export_downloads_viscoelastic_channel.jl

using KrakeNN

const OUTDIR = joinpath(@__DIR__, "src", "public", "downloads", "viscoelastic_channel")
mkpath(OUTDIR)

# Same parameters as docs/src/examples/21_viscoelastic_channel.jl.
Nx, Ny = 4, 32
ν_s = 0.08
ν_p = 0.02
ν_total = ν_s + ν_p
lambda = 5.0
G = ν_p / lambda
Fx_val = 1e-5
max_steps = 30000

ω_s = 1.0 / (3.0 * ν_s + 0.5)

f_in  = zeros(Float64, Nx, Ny, 9)
f_out = zeros(Float64, Nx, Ny, 9)
is_solid = falses(Nx, Ny)
ux = zeros(Float64, Nx, Ny)
uy = zeros(Float64, Nx, Ny)
ρ  = ones(Float64, Nx, Ny)

for j in 1:Ny, i in 1:Nx, q in 1:9
    f_in[i, j, q] = KrakeNN.equilibrium(D2Q9(), 1.0, 0.0, 0.0, q)
end
copy!(f_out, f_in)

Θ_xx = zeros(Float64, Nx, Ny);  Θ_xy = zeros(Float64, Nx, Ny)
Θ_yy = zeros(Float64, Nx, Ny)
Θ_xx_new = similar(Θ_xx);  Θ_xy_new = similar(Θ_xy);  Θ_yy_new = similar(Θ_yy)

tau_p_xx = zeros(Float64, Nx, Ny)
tau_p_xy = zeros(Float64, Nx, Ny)
tau_p_yy = zeros(Float64, Nx, Ny)
Fx_p = zeros(Float64, Nx, Ny)
Fy_p = zeros(Float64, Nx, Ny)

Fx_total = fill(Float64(Fx_val), Nx, Ny)
Fy_total = zeros(Float64, Nx, Ny)

for step in 1:max_steps
    stream_periodic_x_wall_y_2d!(f_out, f_in, Nx, Ny)
    collide_guo_field_2d!(f_out, is_solid, Fx_total, Fy_total, Float64(ω_s))
    compute_macroscopic_2d!(ρ, ux, uy, f_out)

    evolve_logconf_2d!(Θ_xx_new, Θ_xy_new, Θ_yy_new,
                       Θ_xx, Θ_xy, Θ_yy,
                       ux, uy; lambda=lambda)
    copyto!(Θ_xx, Θ_xx_new)
    copyto!(Θ_xy, Θ_xy_new)
    copyto!(Θ_yy, Θ_yy_new)

    compute_stress_from_logconf_2d!(tau_p_xx, tau_p_xy, tau_p_yy,
                                    Θ_xx, Θ_xy, Θ_yy; G=G)
    compute_polymeric_force_2d!(Fx_p, Fy_p, tau_p_xx, tau_p_xy, tau_p_yy)

    Fx_total .= Fx_val .+ Fx_p
    Fy_total .= Fy_p
    global f_in, f_out = f_out, f_in
end

H = Float64(Ny)
u_analytical = [Fx_val / (2 * ν_total) * (j - 0.5) * (H + 0.5 - j) for j in 1:Ny]
u_numerical  = ux[2, :]

csv = joinpath(OUTDIR, "viscoelastic_channel.csv")
open(csv, "w") do io
    println(io, "y,u_analytic,u_kraken")
    for j in 1:Ny
        println(io, string(j - 0.5, ",", u_analytical[j], ",", u_numerical[j]))
    end
end

println("✓ wrote $csv ($(Ny) rows)")
