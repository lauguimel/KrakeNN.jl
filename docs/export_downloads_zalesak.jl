#!/usr/bin/env julia
# Export the real data behind the example-11 Zalesak-disk VOF field to a tidy
# CSV served from the docs download dropdown. Columns: x, y, C — the final
# volume-fraction field after one full rotation, the SAME field the committed
# `zalesak_before_after.svg` / `zalesak_error_map.svg` figures are built from
# (no fabrication). Mirrors the `run_advection_2d` call in 11_zalesak_disk.jl.
#
# Run: julia --project=docs docs/export_downloads_zalesak.jl

using KrakeNN

const OUTDIR = joinpath(@__DIR__, "src", "public", "downloads", "zalesak")
mkpath(OUTDIR)

# Same parameters as docs/src/examples/11_zalesak_disk.jl.
N  = 100
R  = 15.0
cx, cy  = 50.0, 75.0
slot_w  = 5.0

angular_vel = 2π / (N * π)
max_steps   = round(Int, 2π / angular_vel)  # one full rotation

function zalesak_init(x, y)
    r = sqrt((x - cx)^2 + (y - cy)^2)
    disk = 0.5 * (1 - tanh((r - R) / 2))
    in_slot = abs(x - cx) < slot_w / 2 && y < cy && y > cy - R
    return in_slot ? 0.0 : disk
end

velocity_fn(x, y, t) = (-(y - 50.0) * angular_vel, (x - 50.0) * angular_vel)

result = run_advection_2d(; Nx=N, Ny=N, max_steps=max_steps,
                           velocity_fn=velocity_fn, init_C_fn=zalesak_init)
C = result.C
Nx, Ny = size(C)

csv = joinpath(OUTDIR, "zalesak.csv")
open(csv, "w") do io
    println(io, "x,y,C")
    for j in 1:Ny, i in 1:Nx
        println(io, string(i, ",", j, ",", C[i, j]))
    end
end

println("✓ wrote $csv ($(Nx * Ny) rows, $(Nx)x$(Ny) field)")
