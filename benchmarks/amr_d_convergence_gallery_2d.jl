#!/usr/bin/env julia

using CairoMakie
using Dates
using KrakeNN
using Printf

const AMR_D_GALLERY_METHODS = (:cartesian_coarse, :leaf_oracle, :amr_route_native)
const AMR_D_GALLERY_FLOWS = (
    :couette,
    :poiseuille_xband,
    :poiseuille_yband,
    :bfs,
    :square,
    :cylinder,
)

function _parse_symbols_env(name::AbstractString, default::AbstractString)
    raw = get(ENV, name, default)
    return Tuple(Symbol(strip(token)) for token in split(raw, ',')
                 if !isempty(strip(token)))
end

function _parse_ints_env(name::AbstractString, default::AbstractString)
    raw = get(ENV, name, default)
    return Tuple(parse(Int, strip(token)) for token in split(raw, ',')
                 if !isempty(strip(token)))
end

function _scale_range(range::UnitRange{Int}, scale::Int)
    scale > 0 || throw(ArgumentError("scale must be positive"))
    return ((first(range) - 1) * scale + 1):(last(range) * scale)
end

function _coarsen_range(range::UnitRange{Int})
    return ((first(range) + 1) >>> 1):((last(range) + 1) >>> 1)
end

function _steps_for_scale(scale::Int, base_steps::Int, exponent::Real)
    scale > 0 || throw(ArgumentError("scale must be positive"))
    base_steps > 0 || throw(ArgumentError("base_steps must be positive"))
    exponent >= 0 || throw(ArgumentError("step exponent must be nonnegative"))
    return max(1, round(Int, base_steps * scale^exponent))
end

function _mass_rel_drift(result)
    mass_initial = getproperty(result, :mass_initial)
    mass_drift = getproperty(result, :mass_drift)
    return abs(mass_drift) / max(abs(mass_initial), eps(typeof(float(mass_initial))))
end

function _cell_count(method::Symbol,
                     Nx::Int,
                     Ny::Int,
                     patch_i::UnitRange{Int},
                     patch_j::UnitRange{Int})
    if method == :amr_route_native
        patch_area = length(patch_i) * length(patch_j)
        return Nx * Ny - patch_area + 4 * patch_area
    end
    return 4 * Nx * Ny
end

function _finite_abs_error(value::Real, reference::Real)
    return isfinite(value) && isfinite(reference) ? abs(value - reference) : NaN
end

function _finite_rel_error(value::Real, reference::Real)
    return isfinite(value) && isfinite(reference) ?
           abs(value - reference) / max(abs(reference), eps(Float64)) : NaN
end

function _interp_profile(profile::AbstractVector, y::Real)
    n = length(profile)
    n == 1 && return Float64(only(profile))
    pos = Float64(y) * (n - 1) + 1
    lo = clamp(floor(Int, pos), 1, n)
    hi = clamp(lo + 1, 1, n)
    t = pos - lo
    return (1 - t) * Float64(profile[lo]) + t * Float64(profile[hi])
end

function _profile_linf_vs_reference(profile::AbstractVector,
                                    reference::AbstractVector)
    n = length(profile)
    return maximum(1:n) do k
        y = n == 1 ? 0.0 : (k - 1) / (n - 1)
        abs(Float64(profile[k]) - _interp_profile(reference, y))
    end
end

function _profile_patch(flow::Symbol, Nx::Int, Ny::Int, scale::Int)
    if flow == :couette || flow == :poiseuille_yband
        return 1:Nx, _scale_range(5:10, scale)
    elseif flow == :poiseuille_xband
        return _scale_range(7:12, scale), 1:Ny
    end
    throw(ArgumentError("unsupported profile flow: $flow"))
end

function _profile_runner(flow::Symbol, method::Symbol)
    if flow == :couette
        return method == :amr_route_native ?
               run_conservative_tree_couette_route_native_2d :
               run_conservative_tree_couette_macroflow_2d
    elseif flow == :poiseuille_xband || flow == :poiseuille_yband
        return method == :amr_route_native ?
               run_conservative_tree_poiseuille_route_native_2d :
               run_conservative_tree_poiseuille_macroflow_2d
    end
    throw(ArgumentError("unsupported profile flow: $flow"))
end

function _profile_row(flow::Symbol,
                      method::Symbol,
                      scale::Int,
                      Nx::Int,
                      Ny::Int,
                      patch_i::UnitRange{Int},
                      patch_j::UnitRange{Int},
                      result,
                      elapsed_s::Real,
                      leaf_profile::AbstractVector)
    profile = getproperty(result, :ux_profile)
    return (;
        flow,
        method,
        scale,
        Nx,
        Ny,
        steps=getproperty(result, :steps),
        metric=:linf_profile_error_vs_leaf,
        primary_error=method == :leaf_oracle ? 0.0 :
                      Float64(_profile_linf_vs_reference(profile, leaf_profile)),
        secondary_error=Float64(getproperty(result, :l2_error)),
        ux_mean=Float64(sum(profile) / length(profile)),
        uy_mean=0.0,
        Cd=NaN,
        mass_rel_drift=Float64(_mass_rel_drift(result)),
        elapsed_s=Float64(elapsed_s),
        cell_count=_cell_count(method, Nx, Ny, patch_i, patch_j),
        patch_i_first=first(patch_i),
        patch_i_last=last(patch_i),
        patch_j_first=first(patch_j),
        patch_j_last=last(patch_j),
    )
end

function _run_profile_case(flow::Symbol,
                           scale::Int,
                           steps::Int;
                           T::Type{<:Real}=Float64)
    Nx = 18 * scale
    Ny = 14 * scale
    patch_i, patch_j = _profile_patch(flow, Nx, Ny, scale)
    rows = NamedTuple[]

    leaf_patch_i = 1:Nx
    leaf_patch_j = 1:Ny
    leaf = nothing
    leaf_elapsed = @elapsed leaf = _profile_runner(flow, :leaf_oracle)(
        ; Nx=Nx, Ny=Ny, patch_i_range=leaf_patch_i,
        patch_j_range=leaf_patch_j, steps=steps, T=T)
    leaf_profile = getproperty(leaf, :ux_profile)
    push!(rows, _profile_row(flow, :leaf_oracle, scale, Nx, Ny,
                             leaf_patch_i, leaf_patch_j, leaf, leaf_elapsed,
                             leaf_profile))

    Nx_coarse = max(2, Nx >>> 1)
    Ny_coarse = max(2, Ny >>> 1)
    coarse_patch_i = 1:Nx_coarse
    coarse_patch_j = 1:Ny_coarse
    coarse = nothing
    coarse_elapsed = @elapsed coarse = _profile_runner(flow, :cartesian_coarse)(
        ; Nx=Nx_coarse, Ny=Ny_coarse, patch_i_range=coarse_patch_i,
        patch_j_range=coarse_patch_j, steps=steps, T=T)
    push!(rows, _profile_row(flow, :cartesian_coarse, scale,
                             Nx_coarse, Ny_coarse,
                             coarse_patch_i, coarse_patch_j,
                             coarse, coarse_elapsed, leaf_profile))

    route = nothing
    route_elapsed = @elapsed route = _profile_runner(flow, :amr_route_native)(
        ; Nx=Nx, Ny=Ny, patch_i_range=patch_i, patch_j_range=patch_j,
        steps=steps, T=T)
    push!(rows, _profile_row(flow, :amr_route_native, scale, Nx, Ny,
                             patch_i, patch_j, route, route_elapsed,
                             leaf_profile))
    return rows
end

function _solid_row(flow::Symbol,
                    method::Symbol,
                    scale::Int,
                    Nx::Int,
                    Ny::Int,
                    patch_i::UnitRange{Int},
                    patch_j::UnitRange{Int},
                    result,
                    elapsed_s::Real,
                    leaf)
    ux = hasproperty(result, :ux_mean) ? getproperty(result, :ux_mean) :
         getproperty(result, :u_ref)
    uy = hasproperty(result, :uy_mean) ? getproperty(result, :uy_mean) : zero(ux)
    leaf_ux = hasproperty(leaf, :ux_mean) ? getproperty(leaf, :ux_mean) :
              getproperty(leaf, :u_ref)
    leaf_uy = hasproperty(leaf, :uy_mean) ? getproperty(leaf, :uy_mean) : zero(leaf_ux)

    Cd = hasproperty(result, :Cd) ? Float64(getproperty(result, :Cd)) : NaN
    leaf_Cd = hasproperty(leaf, :Cd) ? Float64(getproperty(leaf, :Cd)) : NaN
    metric = flow == :cylinder ? :Cd_rel_error : :ux_abs_error_vs_leaf
    primary_error = flow == :cylinder ?
                    _finite_rel_error(Cd, leaf_Cd) :
                    _finite_abs_error(ux, leaf_ux)
    secondary_error = flow == :cylinder ?
                      _finite_abs_error(ux, leaf_ux) :
                      _finite_abs_error(uy, leaf_uy)

    return (;
        flow,
        method,
        scale,
        Nx,
        Ny,
        steps=getproperty(result, :steps),
        metric,
        primary_error=Float64(primary_error),
        secondary_error=Float64(secondary_error),
        ux_mean=Float64(ux),
        uy_mean=Float64(uy),
        Cd,
        mass_rel_drift=Float64(_mass_rel_drift(result)),
        elapsed_s=Float64(elapsed_s),
        cell_count=_cell_count(method, Nx, Ny, patch_i, patch_j),
        patch_i_first=first(patch_i),
        patch_i_last=last(patch_i),
        patch_j_first=first(patch_j),
        patch_j_last=last(patch_j),
    )
end

function _run_bfs_case(scale::Int,
                       steps::Int;
                       T::Type{<:Real}=Float64)
    Nx = 28 * scale
    Ny = 14 * scale
    patch_i = _scale_range(1:12, scale)
    patch_j = _scale_range(1:8, scale)
    step_i_leaf = 16 * scale
    step_height_leaf = 8 * scale

    leaf = nothing
    leaf_elapsed = @elapsed leaf = run_conservative_tree_bfs_macroflow_2d(
        ; Nx=Nx, Ny=Ny, patch_i_range=patch_i, patch_j_range=patch_j,
        step_i_leaf=step_i_leaf, step_height_leaf=step_height_leaf,
        steps=steps, T=T)

    rows = NamedTuple[]
    Nx_coarse = max(2, Nx >>> 1)
    Ny_coarse = max(2, Ny >>> 1)
    coarse = nothing
    coarse_elapsed = @elapsed coarse = run_conservative_tree_bfs_macroflow_2d(
        ; Nx=Nx_coarse, Ny=Ny_coarse, patch_i_range=1:Nx_coarse,
        patch_j_range=1:Ny_coarse, step_i_leaf=max(1, step_i_leaf >>> 1),
        step_height_leaf=max(1, step_height_leaf >>> 1), steps=steps, T=T)
    push!(rows, _solid_row(:bfs, :cartesian_coarse, scale,
                           Nx_coarse, Ny_coarse, 1:Nx_coarse, 1:Ny_coarse,
                           coarse, coarse_elapsed, leaf))
    push!(rows, _solid_row(:bfs, :leaf_oracle, scale, Nx, Ny,
                           patch_i, patch_j, leaf, leaf_elapsed, leaf))

    route = nothing
    route_elapsed = @elapsed route = run_conservative_tree_bfs_route_native_2d(
        ; Nx=Nx, Ny=Ny, patch_i_range=patch_i, patch_j_range=patch_j,
        step_i_leaf=step_i_leaf, step_height_leaf=step_height_leaf,
        steps=steps, T=T)
    push!(rows, _solid_row(:bfs, :amr_route_native, scale, Nx, Ny,
                           patch_i, patch_j, route, route_elapsed, leaf))
    return rows
end

function _run_square_case(scale::Int,
                          steps::Int;
                          T::Type{<:Real}=Float64)
    Nx = 24 * scale
    Ny = 14 * scale
    patch_i = _scale_range(3:22, scale)
    patch_j = 1:Ny
    obstacle_i = _scale_range(22:27, scale)
    obstacle_j = _scale_range(12:17, scale)

    leaf = nothing
    leaf_elapsed = @elapsed leaf = run_conservative_tree_square_obstacle_macroflow_2d(
        ; Nx=Nx, Ny=Ny, patch_i_range=patch_i, patch_j_range=patch_j,
        obstacle_i_range=obstacle_i, obstacle_j_range=obstacle_j,
        steps=steps, T=T)

    rows = NamedTuple[]
    Nx_coarse = max(2, Nx >>> 1)
    Ny_coarse = max(2, Ny >>> 1)
    coarse = nothing
    coarse_elapsed = @elapsed coarse =
        run_conservative_tree_square_obstacle_macroflow_2d(
            ; Nx=Nx_coarse, Ny=Ny_coarse,
            patch_i_range=1:Nx_coarse, patch_j_range=1:Ny_coarse,
            obstacle_i_range=_coarsen_range(obstacle_i),
            obstacle_j_range=_coarsen_range(obstacle_j),
            steps=steps, T=T)
    push!(rows, _solid_row(:square, :cartesian_coarse, scale,
                           Nx_coarse, Ny_coarse, 1:Nx_coarse, 1:Ny_coarse,
                           coarse, coarse_elapsed, leaf))
    push!(rows, _solid_row(:square, :leaf_oracle, scale, Nx, Ny,
                           patch_i, patch_j, leaf, leaf_elapsed, leaf))

    route = nothing
    route_elapsed = @elapsed route = run_conservative_tree_square_obstacle_route_native_2d(
        ; Nx=Nx, Ny=Ny, patch_i_range=patch_i, patch_j_range=patch_j,
        obstacle_i_range=obstacle_i, obstacle_j_range=obstacle_j,
        steps=steps, T=T)
    push!(rows, _solid_row(:square, :amr_route_native, scale, Nx, Ny,
                           patch_i, patch_j, route, route_elapsed, leaf))
    return rows
end

function _run_cylinder_case(scale::Int,
                            steps::Int,
                            avg_window::Int;
                            T::Type{<:Real}=Float64)
    Nx = 24 * scale
    Ny = 14 * scale
    patch_i = _scale_range(3:22, scale)
    patch_j = 1:Ny
    cx_leaf = 24 * scale + 0.5
    cy_leaf = 14 * scale + 0.5
    radius_leaf = 3 * scale
    avg = min(avg_window * scale, steps)

    leaf = nothing
    leaf_elapsed = @elapsed leaf = run_conservative_tree_cylinder_macroflow_2d(
        ; Nx=Nx, Ny=Ny, patch_i_range=patch_i, patch_j_range=patch_j,
        cx_leaf=cx_leaf, cy_leaf=cy_leaf, radius_leaf=radius_leaf,
        steps=steps, avg_window=avg, T=T)

    rows = NamedTuple[]
    Nx_coarse = max(2, Nx >>> 1)
    Ny_coarse = max(2, Ny >>> 1)
    coarse = nothing
    coarse_elapsed = @elapsed coarse = run_conservative_tree_cylinder_macroflow_2d(
        ; Nx=Nx_coarse, Ny=Ny_coarse, patch_i_range=1:Nx_coarse,
        patch_j_range=1:Ny_coarse,
        cx_leaf=(cx_leaf - 0.5) / 2 + 0.5,
        cy_leaf=(cy_leaf - 0.5) / 2 + 0.5,
        radius_leaf=radius_leaf / 2,
        steps=steps, avg_window=avg, T=T)
    push!(rows, _solid_row(:cylinder, :cartesian_coarse, scale,
                           Nx_coarse, Ny_coarse, 1:Nx_coarse, 1:Ny_coarse,
                           coarse, coarse_elapsed, leaf))
    push!(rows, _solid_row(:cylinder, :leaf_oracle, scale, Nx, Ny,
                           patch_i, patch_j, leaf, leaf_elapsed, leaf))

    route = nothing
    route_elapsed = @elapsed route =
        run_conservative_tree_cylinder_obstacle_route_native_2d(
            ; Nx=Nx, Ny=Ny, patch_i_range=patch_i, patch_j_range=patch_j,
            cx_leaf=cx_leaf, cy_leaf=cy_leaf, radius_leaf=radius_leaf,
            steps=steps, avg_window=avg, T=T)
    push!(rows, _solid_row(:cylinder, :amr_route_native, scale, Nx, Ny,
                           patch_i, patch_j, route, route_elapsed, leaf))
    return rows
end

function run_amr_d_convergence_gallery_2d(;
        flows::Tuple=AMR_D_GALLERY_FLOWS,
        scales::Tuple=(1,),
        base_steps::Int=80,
        step_exponent::Real=1,
        avg_window::Int=20,
        T::Type{<:Real}=Float64)
    rows = NamedTuple[]
    for scale in scales
        steps = _steps_for_scale(scale, base_steps, step_exponent)
        for flow in flows
            if flow in (:couette, :poiseuille_xband, :poiseuille_yband)
                append!(rows, _run_profile_case(flow, scale, steps; T=T))
            elseif flow == :bfs
                append!(rows, _run_bfs_case(scale, steps; T=T))
            elseif flow == :square
                append!(rows, _run_square_case(scale, steps; T=T))
            elseif flow == :cylinder
                append!(rows, _run_cylinder_case(scale, steps, avg_window; T=T))
            else
                throw(ArgumentError("unsupported AMR D gallery flow: $flow"))
            end
        end
    end
    return rows
end

function _leaf_key(row)
    return (row.flow, row.scale)
end

function _write_gallery_csv(path::AbstractString, rows)
    leaf_by_key = Dict{Tuple{Symbol,Int},NamedTuple}()
    for row in rows
        row.method == :leaf_oracle && (leaf_by_key[_leaf_key(row)] = row)
    end
    open(path, "w") do io
        println(io, "flow,method,scale,Nx,Ny,steps,metric,primary_error,secondary_error,ux_mean,uy_mean,Cd,mass_rel_drift,elapsed_s,cell_count,speedup_vs_leaf,cell_count_ratio_vs_leaf,patch_i_first,patch_i_last,patch_j_first,patch_j_last")
        for row in rows
            leaf = leaf_by_key[_leaf_key(row)]
            speedup = leaf.elapsed_s / max(row.elapsed_s, eps(Float64))
            cell_ratio = row.cell_count / leaf.cell_count
            println(io, join((
                row.flow,
                row.method,
                row.scale,
                row.Nx,
                row.Ny,
                row.steps,
                row.metric,
                @sprintf("%.12e", row.primary_error),
                @sprintf("%.12e", row.secondary_error),
                @sprintf("%.12e", row.ux_mean),
                @sprintf("%.12e", row.uy_mean),
                @sprintf("%.12e", row.Cd),
                @sprintf("%.12e", row.mass_rel_drift),
                @sprintf("%.6f", row.elapsed_s),
                row.cell_count,
                @sprintf("%.6f", speedup),
                @sprintf("%.6f", cell_ratio),
                row.patch_i_first,
                row.patch_i_last,
                row.patch_j_first,
                row.patch_j_last,
            ), ","))
        end
    end
    return path
end

function _group_rows(rows, flow::Symbol, method::Symbol)
    return sort(filter(row -> row.flow == flow && row.method == method, rows);
                by=row -> row.scale)
end

function _lineplot_rows!(ax, rows, flow::Symbol, method::Symbol, field::Symbol;
                         label, color, marker)
    series = _group_rows(rows, flow, method)
    isempty(series) && return false
    xs = [row.scale for row in series]
    ys = [max(getproperty(row, field), eps(Float64)) for row in series]
    lines!(ax, xs, ys; label, color, linewidth=2)
    scatter!(ax, xs, ys; color, marker, markersize=10)
    return true
end

function plot_amr_d_convergence_errors_2d(path_png::AbstractString,
                                          path_pdf::AbstractString,
                                          rows)
    panels = (
        (:couette, "Couette", "profile error vs leaf"),
        (:poiseuille_xband, "Poiseuille X band", "profile error vs leaf"),
        (:poiseuille_yband, "Poiseuille Y band", "profile error vs leaf"),
        (:bfs, "BFS", "|ux - leaf|"),
        (:square, "Square obstacle", "|ux - leaf|"),
        (:cylinder, "Cylinder", "Cd relative error"),
    )
    fig = Figure(size=(1320, 820), fontsize=17)
    colors = Dict(:cartesian_coarse => :gray35,
                  :leaf_oracle => :black,
                  :amr_route_native => :dodgerblue3)
    markers = Dict(:cartesian_coarse => :rect,
                   :leaf_oracle => :utriangle,
                   :amr_route_native => :circle)
    labels = Dict(:cartesian_coarse => "coarse Cartesian",
                  :leaf_oracle => "leaf oracle",
                  :amr_route_native => "AMR D")

    for (idx, (flow, title, ylabel)) in enumerate(panels)
        row = cld(idx, 3)
        col = idx - 3 * (row - 1)
        ax = Axis(fig[row, col]; title, xlabel="scale", ylabel,
                  yscale=log10, xticks=[1, 2, 4])
        for method in AMR_D_GALLERY_METHODS
            if method == :leaf_oracle
                continue
            end
            _lineplot_rows!(ax, rows, flow, method, :primary_error;
                            label=labels[method], color=colors[method],
                            marker=markers[method])
        end
        axislegend(ax; position=:rt)
    end
    save(path_png, fig)
    save(path_pdf, fig)
    return path_png, path_pdf
end

function plot_amr_d_convergence_cost_2d(path_png::AbstractString,
                                        path_pdf::AbstractString,
                                        rows)
    flows = collect(AMR_D_GALLERY_FLOWS)
    fig = Figure(size=(1320, 760), fontsize=17)
    for (idx, field_label) in enumerate(((:cell_count_ratio_vs_leaf, "cell-count ratio"),
                                         (:speedup_vs_leaf, "runtime speedup")))
        field, label = field_label
        ax = Axis(fig[idx, 1]; xlabel="case index", ylabel=label,
                  title=idx == 1 ? "AMR D cost relative to leaf oracle" :
                                   "AMR D runtime relative to leaf oracle")
        xs = Float64[]
        ys = Float64[]
        xticklabels = String[]
        for (k, flow) in enumerate(flows)
            series = _group_rows(rows, flow, :amr_route_native)
            leaf_series = _group_rows(rows, flow, :leaf_oracle)
            for row in series
                leaf = only(filter(r -> r.scale == row.scale, leaf_series))
                value = field == :cell_count_ratio_vs_leaf ?
                        row.cell_count / leaf.cell_count :
                        leaf.elapsed_s / max(row.elapsed_s, eps(Float64))
                push!(xs, length(xs) + 1)
                push!(ys, value)
                push!(xticklabels, "$(String(flow))\nS$(row.scale)")
            end
        end
        barplot!(ax, xs, ys; color=:dodgerblue3)
        hlines!(ax, [1.0]; color=:black, linestyle=:dash)
        ax.xticks = (xs, xticklabels)
    end
    save(path_png, fig)
    save(path_pdf, fig)
    return path_png, path_pdf
end

function probe_nested4_cylinder_2d(krk_path::AbstractString)
    setup = load_kraken(krk_path)
    parsed_levels = length(getproperty(setup, :refinements))
    spec_supported = true
    spec_reason = "static spec supported"
    max_level = 0
    try
        spec = create_conservative_tree_spec_from_krk_2d(setup)
        max_level = spec.max_level
    catch err
        spec_supported = false
        spec_reason = sprint(showerror, err)
    end
    case = conservative_tree_amr_d_case_from_krk_2d(setup)
    supported = case.runtime_supported
    reason = spec_supported ? case.reason : spec_reason
    return (; parsed_levels, max_level, spec_supported, supported, reason)
end

function _write_nested_probe_csv(path::AbstractString, probe)
    open(path, "w") do io
        println(io, "case,parsed_levels,max_level,spec_supported,runtime_supported,reason")
        println(io, join(("cylinder_nested4", probe.parsed_levels,
                          probe.max_level, probe.spec_supported,
                          probe.supported, repr(probe.reason)), ","))
    end
    return path
end

function plot_nested4_cylinder_probe(path_png::AbstractString,
                                     path_pdf::AbstractString,
                                     probe)
    fig = Figure(size=(980, 620), fontsize=18)
    ax = Axis(fig[1, 1]; title="Cylinder 4-level nesting probe",
              xlabel="x", ylabel="y", aspect=DataAspect())
    xlims!(ax, 0, 24)
    ylims!(ax, 0, 14)

    rects = (
        (2.0, 0.0, 22.0, 14.0, :dodgerblue3, "L1"),
        (6.0, 2.0, 18.0, 12.0, :seagreen4, "L2"),
        (8.5, 3.5, 15.5, 10.5, :orange3, "L3"),
        (10.0, 5.0, 14.0, 9.0, :crimson, "L4"),
    )
    for (xmin, ymin, xmax, ymax, color, label) in rects
        lines!(ax, [xmin, xmax, xmax, xmin, xmin],
                  [ymin, ymin, ymax, ymax, ymin];
               color, linewidth=3)
        text!(ax, xmin + 0.2, ymax - 0.35; text=label,
              color, fontsize=18)
    end
    theta = range(0, 2pi; length=160)
    lines!(ax, 12 .+ 1.5 .* cos.(theta), 7 .+ 1.5 .* sin.(theta);
           color=:black, linewidth=3)
    status = probe.supported ? "runtime supported" :
             probe.spec_supported ?
             "static spec supported; nested obstacle runtime blocked" :
             "blocked before D runtime: invalid static spec"
    Label(fig[2, 1], status; tellwidth=false)
    save(path_png, fig)
    save(path_pdf, fig)
    return path_png, path_pdf
end

function main()
    flows = _parse_symbols_env(
        "KRK_AMR_D_GALLERY_FLOWS",
        join(String.(AMR_D_GALLERY_FLOWS), ","))
    scales = _parse_ints_env("KRK_AMR_D_GALLERY_SCALES", "1")
    base_steps = parse(Int, get(ENV, "KRK_AMR_D_GALLERY_BASE_STEPS", "80"))
    step_exponent = parse(Float64, get(ENV, "KRK_AMR_D_GALLERY_STEP_EXPONENT", "2"))
    avg_window = parse(Int, get(ENV, "KRK_AMR_D_GALLERY_AVG_WINDOW", "20"))
    tag = get(ENV, "KRK_AMR_TAG", Dates.format(now(), "yyyymmdd_HHMMSS"))

    rows = run_amr_d_convergence_gallery_2d(
        ; flows=flows, scales=scales, base_steps=base_steps,
        step_exponent=step_exponent, avg_window=avg_window)

    result_dir = joinpath(@__DIR__, "results")
    fig_dir = joinpath(result_dir, "figures")
    mkpath(result_dir)
    mkpath(fig_dir)

    csv_path = joinpath(result_dir, "amr_d_convergence_gallery_2d_$tag.csv")
    _write_gallery_csv(csv_path, rows)

    errors_png = joinpath(fig_dir, "amr_d_convergence_errors_2d_$tag.png")
    errors_pdf = joinpath(fig_dir, "amr_d_convergence_errors_2d_$tag.pdf")
    plot_amr_d_convergence_errors_2d(errors_png, errors_pdf, rows)

    cost_png = joinpath(fig_dir, "amr_d_convergence_cost_2d_$tag.png")
    cost_pdf = joinpath(fig_dir, "amr_d_convergence_cost_2d_$tag.pdf")
    plot_amr_d_convergence_cost_2d(cost_png, cost_pdf, rows)

    nested_krk = joinpath(@__DIR__, "krk", "amr_d_convergence_2d",
                          "cylinder_nested4_probe.krk")
    probe = probe_nested4_cylinder_2d(nested_krk)
    probe_csv = joinpath(result_dir, "amr_d_cylinder_nested4_probe_2d_$tag.csv")
    _write_nested_probe_csv(probe_csv, probe)
    nested_png = joinpath(fig_dir, "amr_d_cylinder_nested4_probe_2d_$tag.png")
    nested_pdf = joinpath(fig_dir, "amr_d_cylinder_nested4_probe_2d_$tag.pdf")
    plot_nested4_cylinder_probe(nested_png, nested_pdf, probe)

    println("wrote ", csv_path)
    println("wrote ", errors_png)
    println("wrote ", cost_png)
    println("wrote ", probe_csv)
    println("wrote ", nested_png)
    for row in rows
        println(@sprintf("%-18s %-16s scale=%d metric=%s err=%.3e drift=%.3e elapsed=%.3fs",
                         String(row.flow), String(row.method), row.scale,
                         String(row.metric), row.primary_error,
                         row.mass_rel_drift, row.elapsed_s))
    end
    println("nested4 supported=", probe.supported, " reason=", probe.reason)
    return rows, probe
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
