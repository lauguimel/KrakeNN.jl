#!/usr/bin/env julia
# Export the real data behind the example-12 reversed-vortex VOF field to a tidy
# CSV served from the docs download dropdown. Columns: x, y, C — the final
# volume-fraction field after the full reversal cycle (t = T), the SAME field
# the committed `reversed_vortex_snapshots.svg` / `_error.svg` figures show
# (no fabrication). Mirrors the VOF `run_advection_2d` call in
# 12_reversed_vortex.jl (the full-cycle `result`, not the half-deformation one).
#
# Run: julia --project=docs docs/export_downloads_reversed_vortex.jl

using KrakeNN

const OUTDIR = joinpath(@__DIR__, "src", "public", "downloads", "reversed_vortex")
mkpath(OUTDIR)

# Same parameters as docs/src/examples/12_reversed_vortex.jl.
N = 128
R = 0.15 * N
cx, cy = 0.5 * N, 0.75 * N

T_period = 8.0 * N  # full period (T = 1024)

function vortex_velocity(x, y, t)
    xn = x / N;  yn = y / N
    scale = cos(π * t / T_period) * 0.5
    return (-sin(π * xn) * cos(π * yn) * scale,
             cos(π * xn) * sin(π * yn) * scale)
end

init_fn(x, y) = 0.5 * (1 - tanh((sqrt((x - cx)^2 + (y - cy)^2) - R) / 2))

max_steps = round(Int, T_period)

result = run_advection_2d(; Nx=N, Ny=N, max_steps=max_steps,
                           velocity_fn=vortex_velocity, init_C_fn=init_fn)
C = result.C
Nx, Ny = size(C)

csv = joinpath(OUTDIR, "reversed_vortex.csv")
open(csv, "w") do io
    println(io, "x,y,C")
    for j in 1:Ny, i in 1:Nx
        println(io, string(i, ",", j, ",", C[i, j]))
    end
end

println("✓ wrote $csv ($(Nx * Ny) rows, $(Nx)x$(Ny) field)")
