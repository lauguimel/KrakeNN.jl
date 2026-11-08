#!/usr/bin/env julia
# Export the real data behind the example-03 Taylor-Green energy-decay figure
# (block 3b of generate_figures_01_09.jl) to a CSV served from the docs
# download dropdown. Columns: step, E_analytic, E_kraken — the SAME numbers the
# committed `taylor_green_decay.svg` plots (no fabrication).
#
# Run: julia --project=docs docs/export_downloads_taylor_green.jl

using KrakeNN

const OUTDIR = joinpath(@__DIR__, "src", "public", "downloads", "taylor_green")
mkpath(OUTDIR)

# Same parameters as generate_figures_01_09.jl block "3b. Energy decay".
N = 64; u0 = 0.04; ν = 0.01
k = 2pi / N
# Mean kinetic-energy density of u = u0·(−cos kx sin ky, sin kx cos ky) is
# E0 = u0²/4; the energy decays as E(t) = E0·exp(−4νk²t).
E0 = u0^2 / 4

steps_list = 0:500:5000
E_num = Float64[]
E_ana = Float64[]

for s in steps_list
    if s == 0
        push!(E_num, E0)
    else
        res_s = run_taylor_green_2d(; N=N, ν=ν, u0=u0, max_steps=s)
        ux_s = res_s.ux; uy_s = res_s.uy
        KE = 0.0
        for j in 1:N, i in 1:N
            KE += 0.5 * (ux_s[i, j]^2 + uy_s[i, j]^2)
        end
        push!(E_num, KE / (N * N))
    end
    push!(E_ana, E0 * exp(-4ν * k^2 * s))
end

csv = joinpath(OUTDIR, "taylor_green.csv")
open(csv, "w") do io
    println(io, "step,E_analytic,E_kraken")
    for (s, ea, en) in zip(steps_list, E_ana, E_num)
        println(io, string(s, ",", ea, ",", en))
    end
end

println("✓ wrote $csv ($(length(E_num)) rows)")
