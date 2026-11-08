#!/usr/bin/env julia

using Dates
using KrakeNN
using Printf

struct AMRDKrkRow2D
    flow::Symbol
    method::Symbol
    scale::Int
    Nx::Int
    Ny::Int
    steps::Int
    cell_count::Int
    ux_mean::Float64
    uy_mean::Float64
    Fx_drag::Float64
    Fy_drag::Float64
    Cd::Float64
    mass_rel_drift::Float64
    elapsed_s::Float64
end

function _krk_var(setup, name::Symbol, default=nothing)
    vars = getproperty(setup, :user_vars)
    if haskey(vars, name)
        return vars[name]
    end
    default === nothing && throw(ArgumentError("missing Define $name in $(setup.name)"))
    return default
end

function _krk_flow(setup)
    name = lowercase(getproperty(setup, :name))
    occursin("square", name) && return :square
    occursin("cylinder", name) && return :cylinder
    throw(ArgumentError("cannot infer AMR D flow from Simulation name $(setup.name)"))
end

function _steps_from_krk(setup)
    override = strip(get(ENV, "KRK_AMR_D_STEPS_OVERRIDE", ""))
    isempty(override) && return getproperty(setup, :max_steps)
    steps = parse(Int, override)
    steps > 0 || throw(ArgumentError("KRK_AMR_D_STEPS_OVERRIDE must be positive"))
    return steps
end

function _avg_window_from_krk(setup, steps::Int)
    override = strip(get(ENV, "KRK_AMR_D_AVG_WINDOW_OVERRIDE", ""))
    if isempty(override)
        return min(round(Int, _krk_var(setup, :avg_window)), steps)
    end
    avg = parse(Int, override)
    avg > 0 || throw(ArgumentError("KRK_AMR_D_AVG_WINDOW_OVERRIDE must be positive"))
    return min(avg, steps)
end

function _coarsen_leaf_range(range::UnitRange{Int})
    return ((first(range) + 1) >>> 1):((last(range) + 1) >>> 1)
end

function _coarsen_leaf_center(center_leaf::Real)
    return (Float64(center_leaf) - 0.5) / 2 + 0.5
end

function _mass_rel_drift(result)
    mass_initial = getproperty(result, :mass_initial)
    mass_drift = getproperty(result, :mass_drift)
    return abs(mass_drift) / max(abs(mass_initial), eps(typeof(float(mass_initial))))
end

function _row(flow::Symbol,
              method::Symbol,
              scale::Int,
              Nx::Int,
              Ny::Int,
              cell_count::Int,
              result,
              elapsed_s::Real)
    if flow == :cylinder
        return AMRDKrkRow2D(
            flow, method, scale, Nx, Ny, getproperty(result, :steps),
            cell_count,
            Float64(getproperty(result, :u_ref)),
            0.0,
            Float64(getproperty(result, :Fx_drag)),
            Float64(getproperty(result, :Fy_drag)),
            Float64(getproperty(result, :Cd)),
            Float64(_mass_rel_drift(result)),
            Float64(elapsed_s))
    end

    return AMRDKrkRow2D(
        flow, method, scale, Nx, Ny, getproperty(result, :steps),
        cell_count,
        Float64(getproperty(result, :ux_mean)),
        Float64(getproperty(result, :uy_mean)),
        NaN,
        NaN,
        NaN,
        Float64(_mass_rel_drift(result)),
        Float64(elapsed_s))
end

function _patch_ranges(setup)
    ranges = conservative_tree_patch_ranges_from_krk_refines_2d(
        getproperty(setup, :domain), getproperty(setup, :refinements))
    length(ranges) == 1 ||
        throw(ArgumentError("AMR D publication .krk files need exactly one Refine block"))
    return only(ranges)
end

function _publication_cell_count(method::Symbol,
                                 Nx::Int,
                                 Ny::Int,
                                 patch_i::UnitRange{Int}=1:Nx,
                                 patch_j::UnitRange{Int}=1:Ny)
    if method == :amr_route_native
        patch_area = length(patch_i) * length(patch_j)
        return Nx * Ny - patch_area + 4 * patch_area
    end
    return 4 * Nx * Ny
end

function _run_square_from_krk(setup, method::Symbol; T::Type{<:Real}=Float64)
    domain = getproperty(setup, :domain)
    scale = round(Int, _krk_var(setup, :scale))
    Fx = _krk_var(setup, :Fx)
    omega = _krk_var(setup, :omega)
    rho = _krk_var(setup, :rho0)
    steps = _steps_from_krk(setup)
    obstacle_i = (round(Int, _krk_var(setup, :obstacle_i0)):
                  round(Int, _krk_var(setup, :obstacle_i1)))
    obstacle_j = (round(Int, _krk_var(setup, :obstacle_j0)):
                  round(Int, _krk_var(setup, :obstacle_j1)))

    if method == :cartesian_coarse
        Nx = max(2, Int(domain.Nx) >>> 1)
        Ny = max(2, Int(domain.Ny) >>> 1)
        result = nothing
        elapsed = @elapsed result = run_conservative_tree_square_obstacle_macroflow_2d(
            ; Nx=Nx, Ny=Ny,
            patch_i_range=1:Nx,
            patch_j_range=1:Ny,
            obstacle_i_range=_coarsen_leaf_range(obstacle_i),
            obstacle_j_range=_coarsen_leaf_range(obstacle_j),
            Fx=Fx, omega=omega, rho=rho, steps=steps, T=T)
        return _row(:square, method, scale, Nx, Ny,
                    _publication_cell_count(method, Nx, Ny), result, elapsed)
    end

    patch_i, patch_j = _patch_ranges(setup)
    runner = method == :leaf_oracle ?
        run_conservative_tree_square_obstacle_macroflow_2d :
        run_conservative_tree_square_obstacle_route_native_2d
    result = nothing
    elapsed = @elapsed result = runner(
        ; Nx=Int(domain.Nx), Ny=Int(domain.Ny),
        patch_i_range=patch_i,
        patch_j_range=patch_j,
        obstacle_i_range=obstacle_i,
        obstacle_j_range=obstacle_j,
        Fx=Fx, omega=omega, rho=rho, steps=steps, T=T)
    Nx = Int(domain.Nx)
    Ny = Int(domain.Ny)
    return _row(:square, method, scale, Nx, Ny,
                _publication_cell_count(method, Nx, Ny, patch_i, patch_j),
                result, elapsed)
end

function _run_cylinder_from_krk(setup, method::Symbol; T::Type{<:Real}=Float64)
    domain = getproperty(setup, :domain)
    scale = round(Int, _krk_var(setup, :scale))
    Fx = _krk_var(setup, :Fx)
    omega = _krk_var(setup, :omega)
    rho = _krk_var(setup, :rho0)
    steps = _steps_from_krk(setup)
    avg_window = _avg_window_from_krk(setup, steps)
    cx_leaf = _krk_var(setup, :cx_leaf)
    cy_leaf = _krk_var(setup, :cy_leaf)
    radius_leaf = _krk_var(setup, :radius_leaf)

    if method == :cartesian_coarse
        Nx = max(2, Int(domain.Nx) >>> 1)
        Ny = max(2, Int(domain.Ny) >>> 1)
        result = nothing
        elapsed = @elapsed result = run_conservative_tree_cylinder_macroflow_2d(
            ; Nx=Nx, Ny=Ny,
            patch_i_range=1:Nx,
            patch_j_range=1:Ny,
            cx_leaf=_coarsen_leaf_center(cx_leaf),
            cy_leaf=_coarsen_leaf_center(cy_leaf),
            radius_leaf=radius_leaf / 2,
            Fx=Fx, omega=omega, rho=rho,
            steps=steps, avg_window=avg_window, T=T)
        return _row(:cylinder, method, scale, Nx, Ny,
                    _publication_cell_count(method, Nx, Ny), result, elapsed)
    end

    patch_i, patch_j = _patch_ranges(setup)
    runner = method == :leaf_oracle ?
        run_conservative_tree_cylinder_macroflow_2d :
        run_conservative_tree_cylinder_obstacle_route_native_2d
    result = nothing
    elapsed = @elapsed result = runner(
        ; Nx=Int(domain.Nx), Ny=Int(domain.Ny),
        patch_i_range=patch_i,
        patch_j_range=patch_j,
        cx_leaf=cx_leaf,
        cy_leaf=cy_leaf,
        radius_leaf=radius_leaf,
        Fx=Fx, omega=omega, rho=rho,
        steps=steps, avg_window=avg_window, T=T)
    Nx = Int(domain.Nx)
    Ny = Int(domain.Ny)
    return _row(:cylinder, method, scale, Nx, Ny,
                _publication_cell_count(method, Nx, Ny, patch_i, patch_j),
                result, elapsed)
end

function _run_case_from_krk(path::AbstractString; T::Type{<:Real}=Float64)
    setup = load_kraken(String(path))
    flow = _krk_flow(setup)
    methods = (:cartesian_coarse, :leaf_oracle, :amr_route_native)
    if flow == :square
        return [_run_square_from_krk(setup, method; T=T) for method in methods]
    end
    return [_run_cylinder_from_krk(setup, method; T=T) for method in methods]
end

function _write_raw_csv(path::AbstractString, rows)
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

function _cell_count(row)
    return row.cell_count
end

function _finite_abs_error(value::Real, reference::Real)
    return isfinite(value) && isfinite(reference) ? abs(value - reference) : NaN
end

function _finite_rel_error(value::Real, reference::Real)
    return isfinite(value) && isfinite(reference) ?
           abs(value - reference) / max(abs(reference), eps(Float64)) : NaN
end

function _write_summary_csv(path::AbstractString, rows)
    leaf_by_key = Dict{Tuple{Symbol,Int},AMRDKrkRow2D}()
    for row in rows
        row.method == :leaf_oracle && (leaf_by_key[(row.flow, row.scale)] = row)
    end

    open(path, "w") do io
        println(io, "flow,method,scale,Nx,Ny,steps,cell_count,ux_mean,uy_mean,Cd,ux_abs_error,uy_abs_error,Cd_abs_error,Cd_rel_error,mass_rel_drift,elapsed_s,speedup_vs_leaf,cell_count_ratio_vs_leaf,mlups")
        for row in rows
            leaf = leaf_by_key[(row.flow, row.scale)]
            cell_count = _cell_count(row)
            leaf_cell_count = _cell_count(leaf)
            speedup = leaf.elapsed_s / max(row.elapsed_s, eps(Float64))
            mlups = cell_count * row.steps / max(row.elapsed_s, eps(Float64)) / 1e6
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
                @sprintf("%.12e", _finite_abs_error(row.Cd, leaf.Cd)),
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
    case_dir = get(ENV, "KRK_AMR_D_KRK_DIR",
                   joinpath(@__DIR__, "krk", "amr_d_publication_2d"))
    tag = get(ENV, "KRK_AMR_TAG", Dates.format(now(), "yyyymmdd_HHMMSS"))
    files = sort(filter(endswith(".krk"), readdir(case_dir; join=true)))
    filter_raw = strip(get(ENV, "KRK_AMR_D_CASES", ""))
    if !isempty(filter_raw)
        wanted = Set(strip.(split(filter_raw, ",")))
        files = filter(file -> basename(file) in wanted, files)
    end
    isempty(files) && throw(ArgumentError("no .krk files found in $case_dir"))

    rows = AMRDKrkRow2D[]
    for file in files
        append!(rows, _run_case_from_krk(file))
    end
    sort!(rows; by=row -> (String(row.flow), row.scale, String(row.method)))

    outdir = joinpath(@__DIR__, "results")
    mkpath(outdir)
    raw_path = joinpath(outdir, "amr_d_publication_raw_2d_from_krk_$tag.csv")
    summary_path = joinpath(outdir, "amr_d_publication_summary_2d_from_krk_$tag.csv")
    _write_raw_csv(raw_path, rows)
    _write_summary_csv(summary_path, rows)

    println("wrote ", raw_path)
    println("wrote ", summary_path)
    for row in rows
        println(@sprintf("%-8s %-16s scale=%d ux=%.6e uy=%.6e Cd=%.6e drift=%.3e elapsed=%.3fs",
                         String(row.flow), String(row.method), row.scale,
                         row.ux_mean, row.uy_mean, row.Cd,
                         row.mass_rel_drift, row.elapsed_s))
    end
    return rows
end

main()
