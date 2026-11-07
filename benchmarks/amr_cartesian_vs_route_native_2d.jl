#!/usr/bin/env julia

using Dates
using KrakeNN
using Printf

function _amr_benchmark_flows()
    raw = get(ENV, "KRK_AMR_FLOWS", "bfs,square,cylinder")
    tokens = split(raw, ',')
    return Tuple(Symbol(strip(token)) for token in tokens if !isempty(strip(token)))
end

function _write_amr_benchmark_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, "flow,method,Nx,Ny,steps,ux_mean,uy_mean,mass_rel_drift,elapsed_s")
        for row in rows
            println(io, join((
                row.flow,
                row.method,
                row.Nx,
                row.Ny,
                row.steps,
                @sprintf("%.12e", row.ux_mean),
                @sprintf("%.12e", row.uy_mean),
                @sprintf("%.12e", row.mass_rel_drift),
                @sprintf("%.6f", row.elapsed_s),
            ), ","))
        end
    end
    return path
end

function main()
    steps = parse(Int, get(ENV, "KRK_AMR_STEPS", "1200"))
    tag = get(ENV, "KRK_AMR_TAG", Dates.format(now(), "yyyymmdd_HHMMSS"))
    flows = _amr_benchmark_flows()

    rows = benchmark_conservative_tree_cartesian_vs_amr_2d(
        ; flows=flows, steps=steps)

    outdir = joinpath(@__DIR__, "results")
    mkpath(outdir)
    path = joinpath(outdir, "amr_cartesian_vs_route_native_2d_$tag.csv")
    _write_amr_benchmark_csv(path, rows)

    println("wrote ", path)
    for row in rows
        println(@sprintf("%-8s %-16s steps=%d ux=%.6e uy=%.6e drift=%.3e elapsed=%.3fs",
                         String(row.flow), String(row.method), row.steps,
                         row.ux_mean, row.uy_mean, row.mass_rel_drift,
                         row.elapsed_s))
    end
    return rows
end

main()
