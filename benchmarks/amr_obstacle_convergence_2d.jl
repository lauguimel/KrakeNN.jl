#!/usr/bin/env julia

using Dates
using KrakeNN
using Printf

function _parse_symbols_env(name::AbstractString, default::AbstractString)
    raw = get(ENV, name, default)
    tokens = split(raw, ',')
    return Tuple(Symbol(strip(token)) for token in tokens if !isempty(strip(token)))
end

function _parse_ints_env(name::AbstractString, default::AbstractString)
    raw = get(ENV, name, default)
    tokens = split(raw, ',')
    return Tuple(parse(Int, strip(token)) for token in tokens if !isempty(strip(token)))
end

function _write_amr_obstacle_convergence_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, "flow,method,scale,Nx,Ny,steps,ux_mean,uy_mean,Fx_drag,Fy_drag,Cd,mass_rel_drift,elapsed_s")
        for row in rows
            println(io, join((
                row.flow,
                row.method,
                row.scale,
                row.Nx,
                row.Ny,
                row.steps,
                @sprintf("%.12e", row.ux_mean),
                @sprintf("%.12e", row.uy_mean),
                @sprintf("%.12e", row.Fx_drag),
                @sprintf("%.12e", row.Fy_drag),
                @sprintf("%.12e", row.Cd),
                @sprintf("%.12e", row.mass_rel_drift),
                @sprintf("%.6f", row.elapsed_s),
            ), ","))
        end
    end
    return path
end

function main()
    flows = _parse_symbols_env("KRK_AMR_CONV_FLOWS", "square,cylinder")
    scales = _parse_ints_env("KRK_AMR_CONV_SCALES", "1,2")
    base_steps = parse(Int, get(ENV, "KRK_AMR_CONV_BASE_STEPS", "1200"))
    step_exponent = parse(Float64, get(ENV, "KRK_AMR_CONV_STEP_EXPONENT", "1"))
    avg_window = parse(Int, get(ENV, "KRK_AMR_CONV_AVG_WINDOW", "300"))
    patch_strategy = Symbol(get(ENV, "KRK_AMR_CONV_PATCH_STRATEGY", "default"))
    tag = get(ENV, "KRK_AMR_TAG", Dates.format(now(), "yyyymmdd_HHMMSS"))

    rows = convergence_conservative_tree_obstacles_2d(
        ; flows=flows, scales=scales, base_steps=base_steps,
        step_exponent=step_exponent, avg_window=avg_window,
        patch_strategy=patch_strategy)

    outdir = joinpath(@__DIR__, "results")
    mkpath(outdir)
    path = joinpath(outdir, "amr_obstacle_convergence_2d_$tag.csv")
    _write_amr_obstacle_convergence_csv(path, rows)

    println("wrote ", path)
    println("patch_strategy=", patch_strategy)
    for row in rows
        println(@sprintf("%-8s %-16s scale=%d Nx=%d Ny=%d steps=%d ux=%.6e uy=%.6e Cd=%.6e drift=%.3e elapsed=%.3fs",
                         String(row.flow), String(row.method), row.scale,
                         row.Nx, row.Ny, row.steps, row.ux_mean, row.uy_mean,
                         row.Cd, row.mass_rel_drift, row.elapsed_s))
    end
    return rows
end

main()
