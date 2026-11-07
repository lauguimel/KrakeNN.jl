#!/usr/bin/env julia

using CairoMakie
using KrakeNN
using Printf

const OUTDIR = get(ENV, "KRK_AMR_D_DEBUG_OUTDIR",
                   joinpath(dirname(@__DIR__), "benchmarks", "results",
                            "figures", "amr_d_complex_flow_debug_2d"))
const STEPS = parse(Int, get(ENV, "KRK_AMR_D_DEBUG_STEPS", "240"))
const AVG_WINDOW = parse(Int, get(ENV, "KRK_AMR_D_DEBUG_AVG_WINDOW", "60"))

mkpath(OUTDIR)

function _mass_rel_drift(result)
    m0 = getproperty(result, :mass_initial)
    dm = getproperty(result, :mass_drift)
    return abs(dm) / max(abs(m0), eps(typeof(float(m0))))
end

function _safe_get(result, field::Symbol, default=NaN)
    return hasproperty(result, field) ? getproperty(result, field) : default
end

function _leaf_state(result; force_x=0.0, force_y=0.0, volume=0.25)
    coarse = getproperty(result, :coarse_F)
    patch = getproperty(result, :patch)
    is_solid = getproperty(result, :is_solid_leaf)
    leaf = zeros(eltype(coarse), 2 * size(coarse, 1), 2 * size(coarse, 2), 9)
    composite_to_leaf_F_2d!(leaf, coarse, patch)
    fields = _leaf_fields(leaf, is_solid; volume=volume,
                          force_x=force_x, force_y=force_y)
    return (; leaf, patch, is_solid, fields)
end

function _leaf_fields(F, is_solid; volume=0.25, force_x=0.0, force_y=0.0)
    ux = fill(NaN, size(F, 1), size(F, 2))
    uy = similar(ux)
    rho = similar(ux)
    speed = similar(ux)
    @inbounds for j in axes(F, 2), i in axes(F, 1)
        is_solid[i, j] && continue
        cell = @view F[i, j, :]
        rho_ij = mass_F(cell) / volume
        mx, my = momentum_F(cell)
        ux_ij = (mx / volume + force_x / 2) / rho_ij
        uy_ij = (my / volume + force_y / 2) / rho_ij
        rho[i, j] = rho_ij
        ux[i, j] = ux_ij
        uy[i, j] = uy_ij
        speed[i, j] = hypot(ux_ij, uy_ij)
    end
    return (; ux, uy, rho, speed)
end

function _patch_leaf_ranges(patch)
    return (2 * first(patch.parent_i_range) - 1):(2 * last(patch.parent_i_range)),
           (2 * first(patch.parent_j_range) - 1):(2 * last(patch.parent_j_range))
end

function _rect_lines!(ax, xmin, xmax, ymin, ymax; color=:black, linewidth=1)
    lines!(ax, [xmin, xmax, xmax, xmin, xmin],
              [ymin, ymin, ymax, ymax, ymin];
           color=color, linewidth=linewidth)
    return ax
end

function _draw_amr_mesh!(ax, Nx::Int, Ny::Int, patch)
    pi = patch.parent_i_range
    pj = patch.parent_j_range
    @inbounds for J in 1:Ny, I in 1:Nx
        if first(pi) <= I <= last(pi) && first(pj) <= J <= last(pj)
            for jf in (2 * J - 1):(2 * J), ifine in (2 * I - 1):(2 * I)
                _rect_lines!(ax, ifine - 0.5, ifine + 0.5,
                             jf - 0.5, jf + 0.5;
                             color=(:dodgerblue4, 0.65), linewidth=0.6)
            end
        else
            _rect_lines!(ax, 2 * I - 1.5, 2 * I + 0.5,
                         2 * J - 1.5, 2 * J + 0.5;
                         color=(:gray25, 0.55), linewidth=0.7)
        end
    end
    pi_leaf, pj_leaf = _patch_leaf_ranges(patch)
    _rect_lines!(ax, first(pi_leaf) - 0.5, last(pi_leaf) + 0.5,
                 first(pj_leaf) - 0.5, last(pj_leaf) + 0.5;
                 color=:black, linewidth=2.5)
    return ax
end

function _draw_uniform_mesh!(ax, nx::Int, ny::Int)
    for i in 1:nx
        _rect_lines!(ax, i - 0.5, i + 0.5, 0.5, ny + 0.5;
                     color=(:gray25, 0.35), linewidth=0.45)
    end
    for j in 1:ny
        _rect_lines!(ax, 0.5, nx + 0.5, j - 0.5, j + 0.5;
                     color=(:gray25, 0.35), linewidth=0.45)
    end
    return ax
end

function _solid_overlay!(ax, is_solid)
    mask = Float64.(is_solid)
    heatmap!(ax, 1:size(mask, 1), 1:size(mask, 2), mask;
             colormap=[RGBAf(0, 0, 0, 0), RGBAf(0.82, 0.05, 0.02, 0.68)],
             colorrange=(0.0, 1.0))
    return ax
end

function _finite_colorrange(A; symmetric=false)
    vals = Float64[x for x in A if isfinite(x)]
    isempty(vals) && return nothing
    if symmetric
        m = maximum(abs, vals)
        m == 0 && (m = 1)
        return (-m, m)
    end
    lo = minimum(vals)
    hi = maximum(vals)
    lo == hi && (hi = lo + max(abs(lo), 1.0) * 1e-12)
    return (lo, hi)
end

function _plot_field!(fig, row, col, title, A; colormap=:viridis,
                      colorrange=nothing)
    ax = Axis(fig[row, col], title=title, aspect=DataAspect(),
              xlabel="x leaf", ylabel="y leaf")
    cr = colorrange === nothing ? _finite_colorrange(A) : colorrange
    if cr === nothing || !all(isfinite, cr)
        cr = (0.0, 1.0)
    end
    Aplot = map(x -> isfinite(x) ? x : cr[1], A)
    kwargs = (; colorrange=cr, colorscale=identity)
    hm = heatmap!(ax, 1:size(A, 1), 1:size(A, 2), Aplot;
                  colormap=colormap, nan_color=:black, kwargs...)
    Colorbar(fig[row, col + 1], hm)
    return ax
end

function _lines_finite_segments!(ax, xs, ys; label=nothing, kwargs...)
    first = nothing
    current_label = label
    n = length(xs)
    @inbounds for k in 1:(n + 1)
        finite = k <= n && isfinite(xs[k]) && isfinite(ys[k])
        if finite && first === nothing
            first = k
        elseif (!finite || k == n + 1) && first !== nothing
            last = k - 1
            if last > first
                if current_label === nothing
                    lines!(ax, xs[first:last], ys[first:last]; kwargs...)
                else
                    lines!(ax, xs[first:last], ys[first:last];
                           label=current_label, kwargs...)
                    current_label = nothing
                end
            end
            first = nothing
        end
    end
    return ax
end

function _case_title(flow::Symbol)
    flow == :bfs && return "BFS"
    flow == :square && return "square obstacle"
    flow == :cylinder && return "cylinder obstacle"
    return String(flow)
end

function _plot_case_fields(flow::Symbol, route_state, cart_state, outdir)
    title = _case_title(flow)
    speed_range = _finite_colorrange(vcat(vec(route_state.fields.speed),
                                          vec(cart_state.fields.speed)))
    rho_range = _finite_colorrange(vcat(vec(route_state.fields.rho),
                                        vec(cart_state.fields.rho)))
    fig = Figure(size=(1650, 900), fontsize=16)

    axm1 = Axis(fig[1, 1], title="$title AMR-D active mesh",
                aspect=DataAspect(), xlabel="x leaf", ylabel="y leaf")
    _draw_amr_mesh!(axm1, div(size(route_state.fields.speed, 1), 2),
                    div(size(route_state.fields.speed, 2), 2),
                    route_state.patch)
    _solid_overlay!(axm1, route_state.is_solid)

    _plot_field!(fig, 1, 2, "AMR-D |u|", route_state.fields.speed;
                 colormap=:viridis, colorrange=speed_range)
    _plot_field!(fig, 1, 4, "AMR-D rho", route_state.fields.rho;
                 colormap=:viridis, colorrange=rho_range)

    axm2 = Axis(fig[2, 1], title="$title leaf Cartesian mesh",
                aspect=DataAspect(), xlabel="x leaf", ylabel="y leaf")
    _draw_uniform_mesh!(axm2, size(cart_state.fields.speed, 1),
                        size(cart_state.fields.speed, 2))
    _solid_overlay!(axm2, cart_state.is_solid)

    _plot_field!(fig, 2, 2, "Cartesian |u|", cart_state.fields.speed;
                 colormap=:viridis, colorrange=speed_range)
    _plot_field!(fig, 2, 4, "Cartesian rho", cart_state.fields.rho;
                 colormap=:viridis, colorrange=rho_range)

    path = joinpath(outdir, "$(flow)_fields.png")
    save(path, fig)
    return path
end

function _plot_case_profiles(flow::Symbol, route_state, cart_state, outdir)
    title = _case_title(flow)
    nx, ny = size(route_state.fields.ux)
    j_mid = cld(ny, 2)
    i_probe = clamp(round(Int, 0.75 * nx), 1, nx)
    x = (collect(1:nx) .- 0.5) ./ nx
    y = (collect(1:ny) .- 0.5) ./ ny

    fig = Figure(size=(1350, 920), fontsize=16)
    ax1 = Axis(fig[1, 1], title="$title centerline ux",
               xlabel="x/Lx", ylabel="ux")
    _lines_finite_segments!(ax1, x, route_state.fields.ux[:, j_mid];
                            label="AMR-D", color=:orangered, linewidth=2.8)
    _lines_finite_segments!(ax1, x, cart_state.fields.ux[:, j_mid];
                            label="leaf Cartesian", color=:black, linewidth=2.4)
    axislegend(ax1, position=:rb)

    ax2 = Axis(fig[1, 2], title="$title centerline rho",
               xlabel="x/Lx", ylabel="rho")
    _lines_finite_segments!(ax2, x, route_state.fields.rho[:, j_mid];
                            label="AMR-D", color=:orangered, linewidth=2.8)
    _lines_finite_segments!(ax2, x, cart_state.fields.rho[:, j_mid];
                            label="leaf Cartesian", color=:black, linewidth=2.4)
    axislegend(ax2, position=:rb)

    ax3 = Axis(fig[2, 1], title=@sprintf("%s ux at x/Lx=%.2f", title, x[i_probe]),
               xlabel="ux", ylabel="y/Ly")
    _lines_finite_segments!(ax3, route_state.fields.ux[i_probe, :], y;
                            label="AMR-D", color=:orangered, linewidth=2.8)
    _lines_finite_segments!(ax3, cart_state.fields.ux[i_probe, :], y;
                            label="leaf Cartesian", color=:black, linewidth=2.4)
    axislegend(ax3, position=:rb)

    ax4 = Axis(fig[2, 2], title=@sprintf("%s rho at x/Lx=%.2f", title, x[i_probe]),
               xlabel="rho", ylabel="y/Ly")
    _lines_finite_segments!(ax4, route_state.fields.rho[i_probe, :], y;
                            label="AMR-D", color=:orangered, linewidth=2.8)
    _lines_finite_segments!(ax4, cart_state.fields.rho[i_probe, :], y;
                            label="leaf Cartesian", color=:black, linewidth=2.4)
    axislegend(ax4, position=:rb)

    Label(fig[3, 1:2],
          "Analytic profile: not available for this obstacle/open-flow debug case",
          tellwidth=false)

    path = joinpath(outdir, "$(flow)_profiles.png")
    save(path, fig)
    return path
end

function _run_bfs_case()
    route = run_conservative_tree_bfs_route_native_2d(; steps=STEPS)
    cart = run_conservative_tree_bfs_macroflow_2d(; steps=STEPS)
    return (; flow=:bfs, route, cart, force_x=0.0, force_y=0.0)
end

function _run_square_case()
    route = run_conservative_tree_square_obstacle_route_native_2d(;
        steps=STEPS)
    cart = run_conservative_tree_square_obstacle_macroflow_2d(;
        steps=STEPS)
    return (; flow=:square, route, cart, force_x=2e-5, force_y=0.0)
end

function _run_cylinder_case()
    route = run_conservative_tree_cylinder_obstacle_route_native_2d(;
        steps=STEPS, avg_window=min(AVG_WINDOW, STEPS))
    cart = run_conservative_tree_cylinder_macroflow_2d(;
        steps=STEPS, avg_window=min(AVG_WINDOW, STEPS))
    return (; flow=:cylinder, route, cart, force_x=2e-5, force_y=0.0)
end

function _summary_row(case, fields_path, profiles_path)
    route = case.route
    cart = case.cart
    ux_route = _safe_get(route, :ux_mean, _safe_get(route, :u_ref))
    ux_cart = _safe_get(cart, :ux_mean, _safe_get(cart, :u_ref))
    uy_route = _safe_get(route, :uy_mean, 0.0)
    uy_cart = _safe_get(cart, :uy_mean, 0.0)
    return (
        flow=case.flow,
        steps=STEPS,
        route_mass_rel=_mass_rel_drift(route),
        cart_mass_rel=_mass_rel_drift(cart),
        route_ux=Float64(ux_route),
        cart_ux=Float64(ux_cart),
        route_uy=Float64(uy_route),
        cart_uy=Float64(uy_cart),
        route_Cd=Float64(_safe_get(route, :Cd)),
        cart_Cd=Float64(_safe_get(cart, :Cd)),
        fields_path=fields_path,
        profiles_path=profiles_path,
    )
end

function _write_summary(rows, outdir)
    csv_path = joinpath(outdir, "summary.csv")
    open(csv_path, "w") do io
        println(io, "flow,steps,route_mass_rel,cart_mass_rel,route_ux,cart_ux,route_uy,cart_uy,route_Cd,cart_Cd,fields_path,profiles_path")
        for r in rows
            @printf(io, "%s,%d,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%s,%s\n",
                    String(r.flow), r.steps, r.route_mass_rel, r.cart_mass_rel,
                    r.route_ux, r.cart_ux, r.route_uy, r.cart_uy,
                    r.route_Cd, r.cart_Cd, r.fields_path, r.profiles_path)
        end
    end

    md_path = joinpath(outdir, "summary.md")
    open(md_path, "w") do io
        println(io, "# AMR-D Complex Flow Debug Plots")
        println(io)
        println(io, "Steps: `$(STEPS)`")
        println(io)
        println(io, "| flow | route mass rel | cart mass rel | route ux | cart ux | route Cd | cart Cd |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|")
        for r in rows
            @printf(io, "| %s | %.3e | %.3e | %.6e | %.6e | %.6e | %.6e |\n",
                    String(r.flow), r.route_mass_rel, r.cart_mass_rel,
                    r.route_ux, r.cart_ux, r.route_Cd, r.cart_Cd)
        end
        println(io)
        println(io, "Generated files:")
        for r in rows
            println(io, "- `$(r.fields_path)`")
            println(io, "- `$(r.profiles_path)`")
        end
    end
    return csv_path, md_path
end

function main()
    cases = (_run_bfs_case(), _run_square_case(), _run_cylinder_case())
    rows = NamedTuple[]
    for case in cases
        route_state = _leaf_state(case.route; force_x=case.force_x,
                                  force_y=case.force_y)
        cart_state = _leaf_state(case.cart; force_x=case.force_x,
                                 force_y=case.force_y)
        fields_path = _plot_case_fields(case.flow, route_state, cart_state,
                                        OUTDIR)
        profiles_path = _plot_case_profiles(case.flow, route_state, cart_state,
                                            OUTDIR)
        push!(rows, _summary_row(case, fields_path, profiles_path))
        println("wrote ", fields_path)
        println("wrote ", profiles_path)
    end
    csv_path, md_path = _write_summary(rows, OUTDIR)
    println("wrote ", csv_path)
    println("wrote ", md_path)
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
