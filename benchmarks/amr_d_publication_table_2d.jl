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

function _write_amr_d_raw_csv(path::AbstractString, rows)
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

function _amr_d_patch_area_2d(scale::Int, Nx::Int, Ny::Int, patch_strategy::Symbol)
    if patch_strategy == :interface_buffered
        return length(((3 - 1) * scale + 1):(22 * scale)) * Ny
    elseif patch_strategy == :default
        return length(((8 - 1) * scale + 1):(17 * scale)) *
               length(((4 - 1) * scale + 1):(11 * scale))
    end
    throw(ArgumentError("unsupported patch_strategy: $patch_strategy"))
end

function _amr_d_cell_count(row, patch_strategy::Symbol)
    if row.method == :amr_route_native
        patch_area = _amr_d_patch_area_2d(row.scale, row.Nx, row.Ny, patch_strategy)
        return row.Nx * row.Ny - patch_area + 4 * patch_area
    end
    return 4 * row.Nx * row.Ny
end

function _finite_abs_error(value::Real, reference::Real)
    return isfinite(value) && isfinite(reference) ? abs(value - reference) : NaN
end

function _finite_rel_error(value::Real, reference::Real)
    return isfinite(value) && isfinite(reference) ?
           abs(value - reference) / max(abs(reference), eps(Float64)) : NaN
end

function _write_amr_d_summary_csv(path::AbstractString, rows, patch_strategy::Symbol)
    leaf_by_key = Dict{Tuple{Symbol,Int},Any}()
    for row in rows
        if row.method == :leaf_oracle
            leaf_by_key[(row.flow, row.scale)] = row
        end
    end

    open(path, "w") do io
        println(io, "flow,method,scale,Nx,Ny,steps,cell_count,ux_mean,uy_mean,Cd,ux_abs_error,uy_abs_error,Cd_abs_error,Cd_rel_error,mass_rel_drift,elapsed_s,speedup_vs_leaf,cell_count_ratio_vs_leaf,mlups")
        for row in rows
            leaf = leaf_by_key[(row.flow, row.scale)]
            cell_count = _amr_d_cell_count(row, patch_strategy)
            leaf_cell_count = _amr_d_cell_count(leaf, patch_strategy)
            speedup = leaf.elapsed_s / max(row.elapsed_s, eps(Float64))
            mlups = cell_count * row.steps / max(row.elapsed_s, eps(Float64)) / 1e6
            cd_abs_error = _finite_abs_error(row.Cd, leaf.Cd)
            println(io, join((
                row.flow,
                row.method,
                row.scale,
                row.Nx,
                row.Ny,
                row.steps,
                cell_count,
                @sprintf("%.12e", row.ux_mean),
                @sprintf("%.12e", row.uy_mean),
                @sprintf("%.12e", row.Cd),
                @sprintf("%.12e", abs(row.ux_mean - leaf.ux_mean)),
                @sprintf("%.12e", abs(row.uy_mean - leaf.uy_mean)),
                @sprintf("%.12e", cd_abs_error),
                @sprintf("%.12e", _finite_rel_error(row.Cd, leaf.Cd)),
                @sprintf("%.12e", row.mass_rel_drift),
                @sprintf("%.6f", row.elapsed_s),
                @sprintf("%.6f", speedup),
                @sprintf("%.6f", cell_count / leaf_cell_count),
                @sprintf("%.6f", mlups),
            ), ","))
        end
    end
    return path
end

function main()
    flows = _parse_symbols_env("KRK_AMR_D_FLOWS", "square,cylinder")
    scales = _parse_ints_env("KRK_AMR_D_SCALES", "1,2")
    base_steps = parse(Int, get(ENV, "KRK_AMR_D_BASE_STEPS", "1200"))
    step_exponent = parse(Float64, get(ENV, "KRK_AMR_D_STEP_EXPONENT", "1"))
    avg_window = parse(Int, get(ENV, "KRK_AMR_D_AVG_WINDOW", "300"))
    patch_strategy = Symbol(get(ENV, "KRK_AMR_D_PATCH_STRATEGY", "interface_buffered"))
    tag = get(ENV, "KRK_AMR_TAG", Dates.format(now(), "yyyymmdd_HHMMSS"))

    rows = convergence_conservative_tree_obstacles_2d(
        ; flows=flows, scales=scales, base_steps=base_steps,
        step_exponent=step_exponent, avg_window=avg_window,
        patch_strategy=patch_strategy, include_coarse_cartesian=true)

    outdir = joinpath(@__DIR__, "results")
    mkpath(outdir)
    raw_path = joinpath(outdir, "amr_d_publication_raw_2d_$tag.csv")
    summary_path = joinpath(outdir, "amr_d_publication_summary_2d_$tag.csv")
    _write_amr_d_raw_csv(raw_path, rows)
    _write_amr_d_summary_csv(summary_path, rows, patch_strategy)

    println("wrote ", raw_path)
    println("wrote ", summary_path)
    println("patch_strategy=", patch_strategy)
    for row in rows
        println(@sprintf("%-8s %-16s scale=%d ux=%.6e uy=%.6e Cd=%.6e drift=%.3e elapsed=%.3fs",
                         String(row.flow), String(row.method), row.scale,
                         row.ux_mean, row.uy_mean, row.Cd,
                         row.mass_rel_drift, row.elapsed_s))
    end
    return rows
end

main()
