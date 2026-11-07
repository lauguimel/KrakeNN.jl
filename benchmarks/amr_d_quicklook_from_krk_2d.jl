#!/usr/bin/env julia

using CairoMakie
using Dates
using KrakeNN
using Printf

const DEFAULT_QUICKLOOK_CASE_DIR = joinpath(
    dirname(@__DIR__), "benchmarks", "krk", "amr_d_convergence_2d")
const DEFAULT_QUICKLOOK_OUTDIR = joinpath(
    dirname(@__DIR__), "benchmarks", "results", "quicklook",
    "amr_d_krk_2d_" * Dates.format(now(), "yyyymmdd_HHMMSS"))

struct AMRDQuicklookArtifact2D
    case_name::String
    flow::Symbol
    method::Symbol
    status::Symbol
    outdir::String
    status_csv::String
    mesh_csv::String
    mesh_png::String
    fields_csv::String
    fields_png::String
    profiles_csv::String
    profiles_png::String
end

struct AMRDCartesianChannelQuicklook2D{T}
    flow::Symbol
    F::Array{T,3}
    ux_profile::Vector{T}
    analytic_ux_profile::Vector{T}
    l2_error::T
    linf_error::T
    mass_initial::T
    mass_final::T
    mass_drift::T
    steps::Int
    volume::T
    force_x::T
    force_y::T
    max_level::Int
end

function _ql_sanitize_name(name::AbstractString)
    cleaned = replace(String(name), r"[^A-Za-z0-9_.-]+" => "_")
    return isempty(cleaned) ? "case" : cleaned
end

function _ql_parse_case_list(raw::AbstractString; case_dir=DEFAULT_QUICKLOOK_CASE_DIR)
    items = String[]
    for item in split(raw, ",")
        s = strip(item)
        isempty(s) && continue
        path = isabspath(s) ? s : joinpath(case_dir, s)
        endswith(path, ".krk") || (path *= ".krk")
        push!(items, path)
    end
    return items
end

function _ql_env_case_paths()
    raw = strip(get(ENV, "KRK_AMR_D_QUICKLOOK_CASES", ""))
    case_dir = get(ENV, "KRK_AMR_D_QUICKLOOK_DIR", DEFAULT_QUICKLOOK_CASE_DIR)
    if isempty(raw)
        return sort(filter(endswith(".krk"), readdir(case_dir; join=true)))
    end
    return _ql_parse_case_list(raw; case_dir=case_dir)
end

function _ql_steps(setup; steps_override=nothing)
    steps_override !== nothing && return Int(steps_override)
    raw = strip(get(ENV, "KRK_AMR_D_QUICKLOOK_STEPS_OVERRIDE", ""))
    isempty(raw) || return parse(Int, raw)
    return getproperty(setup, :max_steps)
end

function _ql_avg_window(setup, steps::Int; avg_window_override=nothing)
    avg_window_override !== nothing && return min(Int(avg_window_override), steps)
    raw = strip(get(ENV, "KRK_AMR_D_QUICKLOOK_AVG_WINDOW_OVERRIDE", ""))
    isempty(raw) || return min(parse(Int, raw), steps)
    vars = getproperty(setup, :user_vars)
    return min(round(Int, get(vars, :avg_window, max(1, div(steps, 4)))), steps)
end

function _ql_var(setup, name::Symbol, default)
    vars = getproperty(setup, :user_vars)
    return haskey(vars, name) ? vars[name] : default
end

function _ql_eval_expr(expr, default)
    try
        return Float64(evaluate(expr))
    catch
        return Float64(default)
    end
end

function _ql_body_force(setup, name::Symbol, default)
    body = getproperty(getproperty(setup, :physics), :body_force)
    haskey(body, name) || return _ql_var(setup, name, default)
    return _ql_eval_expr(body[name], _ql_var(setup, name, default))
end

function _ql_boundary_value(setup,
                            face::Symbol,
                            bc_type::Symbol,
                            key::Symbol,
                            default)
    for bc in getproperty(setup, :boundaries)
        getproperty(bc, :face) == face || continue
        getproperty(bc, :type) == bc_type || continue
        values = getproperty(bc, :values)
        haskey(values, key) || continue
        return _ql_eval_expr(values[key], default)
    end
    return default
end

function _ql_patch_ranges(setup)
    ranges = conservative_tree_patch_ranges_from_krk_refines_2d(
        getproperty(setup, :domain), getproperty(setup, :refinements))
    length(ranges) == 1 ||
        throw(ArgumentError("quicklook single-patch route needs one Refine block"))
    return only(ranges)
end

function _ql_mass_rel_drift(result)
    mass_initial = getproperty(result, :mass_initial)
    mass_drift = getproperty(result, :mass_drift)
    return abs(mass_drift) / max(abs(mass_initial),
                                 eps(typeof(float(mass_initial))))
end

function _ql_profile_interp(profile::AbstractVector, y::Real)
    n = length(profile)
    n == 0 && return NaN
    n == 1 && return Float64(only(profile))
    pos = clamp(y, 0.0, 1.0) * (n - 1) + 1
    lo = floor(Int, pos)
    hi = ceil(Int, pos)
    lo == hi && return Float64(profile[lo])
    t = pos - lo
    return (1 - t) * Float64(profile[lo]) + t * Float64(profile[hi])
end

function _ql_profile_errors(profile::AbstractVector, reference::AbstractVector)
    n = length(profile)
    n == 0 && return (NaN, NaN)
    sq = 0.0
    linf = 0.0
    count = 0
    for k in 1:n
        value = Float64(profile[k])
        isfinite(value) || continue
        y = n == 1 ? 0.0 : (k - 1) / (n - 1)
        ref = _ql_profile_interp(reference, y)
        isfinite(ref) || continue
        err = value - ref
        sq += err * err
        linf = max(linf, abs(err))
        count += 1
    end
    count == 0 && return (NaN, NaN)
    return sqrt(sq / count), linf
end

function _ql_field_errors(A, B)
    size(A) == size(B) || return (NaN, NaN)
    sq = 0.0
    linf = 0.0
    count = 0
    @inbounds for idx in eachindex(A, B)
        a = Float64(A[idx])
        b = Float64(B[idx])
        isfinite(a) && isfinite(b) || continue
        err = a - b
        sq += err * err
        linf = max(linf, abs(err))
        count += 1
    end
    count == 0 && return (NaN, NaN)
    return sqrt(sq / count), linf
end

function _ql_finite_mean(A)
    total = 0.0
    count = 0
    for value in A
        v = Float64(value)
        isfinite(v) || continue
        total += v
        count += 1
    end
    return count == 0 ? NaN : total / count
end

function _ql_finite_minmax(A)
    vmin = Inf
    vmax = -Inf
    count = 0
    for value in A
        v = Float64(value)
        isfinite(v) || continue
        vmin = min(vmin, v)
        vmax = max(vmax, v)
        count += 1
    end
    return count == 0 ? (NaN, NaN) : (vmin, vmax)
end

function _ql_leaf_fields(F, is_solid; volume=0.25, force_x=0.0, force_y=0.0)
    ux = fill(NaN, size(F, 1), size(F, 2))
    uy = fill(NaN, size(F, 1), size(F, 2))
    rho = fill(NaN, size(F, 1), size(F, 2))
    speed = fill(NaN, size(F, 1), size(F, 2))
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
    return (; rho, ux, uy, speed)
end

function _ql_single_patch_level_map(Nx::Int, Ny::Int, patch)
    level = zeros(Int, 2 * Nx, 2 * Ny)
    for J in patch.parent_j_range, I in patch.parent_i_range
        level[(2 * I - 1):(2 * I), (2 * J - 1):(2 * J)] .= 1
    end
    return level
end

function _ql_state_from_composite_result(result; force_x=0.0, force_y=0.0)
    coarse = getproperty(result, :coarse_F)
    patch = getproperty(result, :patch)
    leaf = zeros(eltype(coarse), 2 * size(coarse, 1), 2 * size(coarse, 2), 9)
    composite_to_leaf_F_2d!(leaf, coarse, patch)
    is_solid = hasproperty(result, :is_solid_leaf) ?
        getproperty(result, :is_solid_leaf) : falses(size(leaf, 1), size(leaf, 2))
    fields = _ql_leaf_fields(leaf, is_solid; volume=0.25,
                             force_x=force_x, force_y=force_y)
    level = _ql_single_patch_level_map(size(coarse, 1), size(coarse, 2), patch)
    return (; fields, is_solid, level, patch, leaf_nx=size(leaf, 1),
            leaf_ny=size(leaf, 2))
end

function _ql_state_from_spec_result(result; force_x=0.0, force_y=0.0,
                                    level_scaled_force::Bool=false)
    spec = getproperty(result, :spec)
    F = getproperty(result, :F)
    leaf_nx = spec.Nx << spec.max_level
    leaf_ny = spec.Ny << spec.max_level
    ux = fill(NaN, leaf_nx, leaf_ny)
    uy = fill(NaN, leaf_nx, leaf_ny)
    rho = fill(NaN, leaf_nx, leaf_ny)
    speed = fill(NaN, leaf_nx, leaf_ny)
    level = fill(-1, leaf_nx, leaf_ny)

    @inbounds for cell_id in spec.active_cells
        cell = spec.cells[cell_id]
        scale = 1 << (spec.max_level - cell.level)
        mass = zero(eltype(F))
        mx = zero(eltype(F))
        my = zero(eltype(F))
        for q in 1:9
            Fq = F[cell_id, q]
            mass += Fq
            mx += d2q9_cx(q) * Fq
            my += d2q9_cy(q) * Fq
        end
        volume = eltype(F)(cell.metrics.volume)
        rho_c = mass / volume
        fx = level_scaled_force ?
             KrakeNN.conservative_tree_leaf_equivalent_force_2d(
                force_x, spec, cell.level) : force_x
        fy = level_scaled_force ?
             KrakeNN.conservative_tree_leaf_equivalent_force_2d(
                force_y, spec, cell.level) : force_y
        ux_c = (mx / volume + fx / 2) / rho_c
        uy_c = (my / volume + fy / 2) / rho_c
        i0 = (cell.i - 1) * scale + 1
        i1 = cell.i * scale
        j0 = (cell.j - 1) * scale + 1
        j1 = cell.j * scale
        rho[i0:i1, j0:j1] .= rho_c
        ux[i0:i1, j0:j1] .= ux_c
        uy[i0:i1, j0:j1] .= uy_c
        speed[i0:i1, j0:j1] .= hypot(ux_c, uy_c)
        level[i0:i1, j0:j1] .= cell.level
    end

    is_solid = hasproperty(result, :is_solid_leaf) ?
        getproperty(result, :is_solid_leaf) : falses(leaf_nx, leaf_ny)
    @inbounds for j in 1:leaf_ny, i in 1:leaf_nx
        if is_solid[i, j]
            rho[i, j] = NaN
            ux[i, j] = NaN
            uy[i, j] = NaN
            speed[i, j] = NaN
        end
    end

    return (; fields=(; rho, ux, uy, speed), is_solid,
            level, patch=nothing, leaf_nx, leaf_ny)
end

function _ql_state_from_cartesian_channel_result(result)
    F = getproperty(result, :F)
    is_solid = falses(size(F, 1), size(F, 2))
    fields = _ql_leaf_fields(
        F, is_solid; volume=getproperty(result, :volume),
        force_x=getproperty(result, :force_x),
        force_y=getproperty(result, :force_y))
    level = fill(getproperty(result, :max_level), size(F, 1), size(F, 2))
    return (; fields, is_solid, level, patch=nothing, leaf_nx=size(F, 1),
            leaf_ny=size(F, 2))
end

function _ql_static_solid_mask(setup, case, leaf_nx::Int, leaf_ny::Int)
    case.geometry == :cylinder || return falses(leaf_nx, leaf_ny)
    vars = getproperty(setup, :user_vars)
    if !(haskey(vars, :cx_leaf) && haskey(vars, :cy_leaf) &&
            haskey(vars, :radius_leaf))
        if haskey(vars, :cx) && haskey(vars, :cy) && haskey(vars, :R)
            domain = getproperty(setup, :domain)
            rx = Float64(vars[:R]) * leaf_nx / Float64(domain.Lx)
            ry = Float64(vars[:R]) * leaf_ny / Float64(domain.Ly)
            vars = merge(vars, Dict(
                :cx_leaf => Float64(vars[:cx]) * leaf_nx / Float64(domain.Lx) + 0.5,
                :cy_leaf => Float64(vars[:cy]) * leaf_ny / Float64(domain.Ly) + 0.5,
                :radius_leaf => 0.5 * (rx + ry)))
        else
            return falses(leaf_nx, leaf_ny)
        end
    end
    return cylinder_solid_mask_leaf_2d(
        leaf_nx, leaf_ny, vars[:cx_leaf], vars[:cy_leaf], vars[:radius_leaf])
end

function _ql_mesh_cells_from_patch(Nx::Int, Ny::Int, patch)
    rows = NamedTuple[]
    for J in 1:Ny, I in 1:Nx
        inside = I in patch.parent_i_range && J in patch.parent_j_range
        if inside
            for jf in (2 * J - 1):(2 * J), ifine in (2 * I - 1):(2 * I)
                push!(rows, (; level=1, i=ifine, j=jf,
                             leaf_i_min=ifine, leaf_i_max=ifine,
                             leaf_j_min=jf, leaf_j_max=jf, active=true))
            end
        else
            push!(rows, (; level=0, i=I, j=J,
                         leaf_i_min=2 * I - 1, leaf_i_max=2 * I,
                         leaf_j_min=2 * J - 1, leaf_j_max=2 * J,
                         active=true))
        end
    end
    return rows
end

function _ql_mesh_cells_from_spec(spec)
    rows = NamedTuple[]
    for cell_id in spec.active_cells
        cell = spec.cells[cell_id]
        scale = 1 << (spec.max_level - cell.level)
        push!(rows, (; level=cell.level, i=cell.i, j=cell.j,
                     leaf_i_min=(cell.i - 1) * scale + 1,
                     leaf_i_max=cell.i * scale,
                     leaf_j_min=(cell.j - 1) * scale + 1,
                     leaf_j_max=cell.j * scale,
                     active=true))
    end
    return rows
end

function _ql_mesh_cells_uniform(leaf_nx::Int, leaf_ny::Int, level::Int)
    rows = NamedTuple[]
    for j in 1:leaf_ny, i in 1:leaf_nx
        push!(rows, (; level, i, j, leaf_i_min=i, leaf_i_max=i,
                     leaf_j_min=j, leaf_j_max=j, active=true))
    end
    return rows
end

function _ql_write_status_csv(path, setup, case)
    open(path, "w") do io
        println(io, "name,flow,geometry,boundary_policy,wall_model,max_level,refine_count,spec_supported,runtime_supported,runtime_status,reason")
        println(io, join((
            case.name,
            case.flow,
            case.geometry,
            case.boundary_policy,
            case.wall_model,
            case.max_level,
            case.refine_count,
            case.spec_supported,
            case.runtime_supported,
            case.runtime_status,
            repr(case.reason),
        ), ","))
    end
    return path
end

function _ql_write_mesh_csv(path, rows)
    open(path, "w") do io
        println(io, "level,i,j,leaf_i_min,leaf_i_max,leaf_j_min,leaf_j_max,active")
        for r in rows
            println(io, join((r.level, r.i, r.j, r.leaf_i_min, r.leaf_i_max,
                              r.leaf_j_min, r.leaf_j_max, r.active), ","))
        end
    end
    return path
end

function _ql_write_fields_csv(path, state)
    open(path, "w") do io
        println(io, "i_leaf,j_leaf,level,is_solid,rho,ux,uy,speed")
        for j in 1:state.leaf_ny, i in 1:state.leaf_nx
            @printf(io, "%d,%d,%d,%d,%.16e,%.16e,%.16e,%.16e\n",
                    i, j, state.level[i, j], state.is_solid[i, j] ? 1 : 0,
                    state.fields.rho[i, j], state.fields.ux[i, j],
                    state.fields.uy[i, j], state.fields.speed[i, j])
        end
    end
    return path
end

function _ql_profile_vectors(result, state)
    profile = hasproperty(result, :ux_profile) ? getproperty(result, :ux_profile) :
        _ql_mean_ux_by_y(state)
    analytic = hasproperty(result, :analytic_ux_profile) ?
        getproperty(result, :analytic_ux_profile) :
        hasproperty(result, :analytic_profile) ?
        getproperty(result, :analytic_profile) :
        fill(NaN, length(profile))
    return Float64.(profile), Float64.(analytic)
end

function _ql_mean_ux_by_y(state)
    profile = fill(NaN, state.leaf_ny)
    for j in 1:state.leaf_ny
        sum_ux = 0.0
        count = 0
        for i in 1:state.leaf_nx
            value = state.fields.ux[i, j]
            isfinite(value) || continue
            sum_ux += value
            count += 1
        end
        count > 0 && (profile[j] = sum_ux / count)
    end
    return profile
end

function _ql_write_profiles_csv(path, result, state)
    profile, analytic = _ql_profile_vectors(result, state)
    nx = state.leaf_nx
    ny = state.leaf_ny
    j_mid = cld(ny, 2)
    i_probe = clamp(round(Int, 0.75 * nx), 1, nx)
    open(path, "w") do io
        println(io, "kind,index,coord,ux,rho,uy,speed,analytic_ux")
        for j in 1:ny
            coord = ny == 1 ? 0.0 : (j - 1) / (ny - 1)
            @printf(io, "mean_y,%d,%.16e,%.16e,NaN,NaN,NaN,%.16e\n",
                    j, coord, profile[j], analytic[j])
        end
        for i in 1:nx
            coord = (i - 0.5) / nx
            @printf(io, "centerline_x,%d,%.16e,%.16e,%.16e,%.16e,%.16e,NaN\n",
                    i, coord, state.fields.ux[i, j_mid],
                    state.fields.rho[i, j_mid], state.fields.uy[i, j_mid],
                    state.fields.speed[i, j_mid])
        end
        for j in 1:ny
            coord = (j - 0.5) / ny
            @printf(io, "vertical_probe,%d,%.16e,%.16e,%.16e,%.16e,%.16e,NaN\n",
                    j, coord, state.fields.ux[i_probe, j],
                    state.fields.rho[i_probe, j], state.fields.uy[i_probe, j],
                    state.fields.speed[i_probe, j])
        end
    end
    return path
end

function _ql_finite_colorrange(A; symmetric=false)
    vals = Float64[x for x in A if isfinite(x)]
    isempty(vals) && return (0.0, 1.0)
    if symmetric
        m = maximum(abs, vals)
        m == 0 && (m = 1.0)
        return (-m, m)
    end
    lo = minimum(vals)
    hi = maximum(vals)
    abs(hi - lo) <= max(abs(lo), abs(hi), 1.0) * 1e-5 &&
        return _ql_safe_colorrange(lo, hi)
    return (lo, hi)
end

function _ql_safe_colorrange(lo::Real, hi::Real)
    lo_f = Float64(lo)
    hi_f = Float64(hi)
    if abs(hi_f - lo_f) <= max(abs(lo_f), abs(hi_f), 1.0) * 1e-5
        center = 0.5 * (lo_f + hi_f)
        pad = max(abs(center), 1.0) * 1e-4
        return (center - pad, center + pad)
    elseif lo_f == hi_f
        pad = max(abs(lo_f), 1.0) * 1e-4
        return (lo_f - pad, hi_f + pad)
    end
    return (lo_f, hi_f)
end

function _ql_rect!(ax, i0, i1, j0, j1; color, linewidth)
    lines!(ax, [i0 - 0.5, i1 + 0.5, i1 + 0.5, i0 - 0.5, i0 - 0.5],
              [j0 - 0.5, j0 - 0.5, j1 + 0.5, j1 + 0.5, j0 - 0.5];
           color=color, linewidth=linewidth)
    return ax
end

function _ql_mesh_segments_by_level(rows, lmin::Int, lmax::Int)
    nlevels = max(lmax - lmin + 1, 0)
    segments = [Point2f[] for _ in 1:nlevels]
    for r in rows
        idx = clamp(r.level - lmin + 1, 1, nlevels)
        pts = segments[idx]
        x0 = Float32(r.leaf_i_min - 0.5)
        x1 = Float32(r.leaf_i_max + 0.5)
        y0 = Float32(r.leaf_j_min - 0.5)
        y1 = Float32(r.leaf_j_max + 0.5)
        push!(pts,
              Point2f(x0, y0), Point2f(x1, y0),
              Point2f(x1, y0), Point2f(x1, y1),
              Point2f(x1, y1), Point2f(x0, y1),
              Point2f(x0, y1), Point2f(x0, y0))
    end
    return segments
end

function _ql_regular_grid_segments(leaf_nx::Int, leaf_ny::Int,
                                   stride::Int)
    pts = Point2f[]
    for i in 1:stride:(leaf_nx + 1)
        x = Float32(i - 0.5)
        push!(pts, Point2f(x, 0.5f0), Point2f(x, Float32(leaf_ny) + 0.5f0))
    end
    for j in 1:stride:(leaf_ny + 1)
        y = Float32(j - 0.5)
        push!(pts, Point2f(0.5f0, y), Point2f(Float32(leaf_nx) + 0.5f0, y))
    end
    push!(pts,
          Point2f(0.5f0, 0.5f0), Point2f(Float32(leaf_nx) + 0.5f0, 0.5f0),
          Point2f(Float32(leaf_nx) + 0.5f0, 0.5f0),
          Point2f(Float32(leaf_nx) + 0.5f0, Float32(leaf_ny) + 0.5f0),
          Point2f(Float32(leaf_nx) + 0.5f0, Float32(leaf_ny) + 0.5f0),
          Point2f(0.5f0, Float32(leaf_ny) + 0.5f0),
          Point2f(0.5f0, Float32(leaf_ny) + 0.5f0), Point2f(0.5f0, 0.5f0))
    return pts
end

function _ql_overlay_mesh_wireframe!(ax, rows; leaf_nx::Int, leaf_ny::Int,
                                     palette=:viridis, alpha=0.90,
                                     linewidth=0.75)
    isempty(rows) && return ax
    levels = [r.level for r in rows]
    lmin = minimum(levels)
    lmax = maximum(levels)

    if lmin == lmax && length(rows) > 8000
        stride = max(1, ceil(Int, max(leaf_nx, leaf_ny) / 64))
        pts = _ql_regular_grid_segments(leaf_nx, leaf_ny, stride)
        linesegments!(ax, pts; color=(:black, 0.28), linewidth=0.55)
        return ax
    end

    segments_by_level = _ql_mesh_segments_by_level(rows, lmin, lmax)
    colors = cgrad(palette, max(lmax - lmin + 1, 2), categorical=true)
    for (idx, pts) in pairs(segments_by_level)
        isempty(pts) && continue
        level = lmin + idx - 1
        lw = linewidth + 0.16 * max(level - lmin, 0)
        linesegments!(ax, pts; color=(colors[idx], alpha), linewidth=lw)
    end
    return ax
end

function _ql_mesh_cells_from_result_state(result, state)
    result isa AMRDCartesianChannelQuicklook2D &&
        return _ql_mesh_cells_uniform(state.leaf_nx, state.leaf_ny,
                                      getproperty(result, :max_level))
    hasproperty(result, :spec) &&
        return _ql_mesh_cells_from_spec(getproperty(result, :spec))
    if hasproperty(result, :coarse_F) && hasproperty(result, :patch)
        coarse = getproperty(result, :coarse_F)
        return _ql_mesh_cells_from_patch(size(coarse, 1), size(coarse, 2),
                                         getproperty(result, :patch))
    end
    return _ql_mesh_cells_uniform(
        state.leaf_nx, state.leaf_ny, maximum(state.level))
end

function _ql_probe_i_from_max_level(level::AbstractMatrix{<:Integer})
    finite_levels = [l for l in level if l >= 0]
    isempty(finite_levels) && return max(1, size(level, 1) ÷ 2)
    lmax = maximum(finite_levels)
    idxs = findall(==(lmax), level)
    isempty(idxs) && return max(1, size(level, 1) ÷ 2)
    imin = minimum(I[1] for I in idxs)
    imax = maximum(I[1] for I in idxs)
    return clamp(round(Int, 0.5 * (imin + imax)), 1, size(level, 1))
end

function _ql_overlay_vertical_probe!(ax, i_probe::Int, leaf_ny::Int)
    lines!(ax, [i_probe, i_probe], [0.5, leaf_ny + 0.5];
           color=(:black, 0.88), linewidth=2.2, linestyle=:dash)
    return ax
end

function _ql_level_color(level::Int)
    palette = (:gray35, :dodgerblue4, :seagreen4, :orange3, :crimson)
    return palette[clamp(level + 1, 1, length(palette))]
end

function _ql_overlay_solid!(ax, is_solid)
    isempty(is_solid) && return ax
    any(is_solid) || return ax
    heatmap!(ax, 1:size(is_solid, 1), 1:size(is_solid, 2),
             Float64.(is_solid);
             colormap=[RGBAf(0, 0, 0, 0), RGBAf(0.8, 0.03, 0.02, 0.65)],
             colorrange=(0.0, 1.0))
    return ax
end

function _ql_mesh_level_grid(rows, leaf_nx::Int, leaf_ny::Int)
    grid = fill(-1, leaf_nx, leaf_ny)
    for r in rows
        grid[r.leaf_i_min:r.leaf_i_max, r.leaf_j_min:r.leaf_j_max] .= r.level
    end
    return grid
end

function _ql_plot_mesh(path, rows; title, leaf_nx, leaf_ny, is_solid=falses(0, 0))
    fig = Figure(size=(1100, 760), fontsize=16)
    ax = Axis(fig[1, 1]; title=title, xlabel="x leaf", ylabel="y leaf",
              aspect=DataAspect())
    if length(rows) > 8000
        levels = _ql_mesh_level_grid(rows, leaf_nx, leaf_ny)
        hm = heatmap!(ax, 1:leaf_nx, 1:leaf_ny, Float64.(levels);
                      colormap=:viridis,
                      colorrange=_ql_safe_colorrange(
                          minimum(levels), maximum(levels)))
        Colorbar(fig[1, 2], hm; label="level")
    else
        for r in rows
            _ql_rect!(ax, r.leaf_i_min, r.leaf_i_max, r.leaf_j_min, r.leaf_j_max;
                      color=(_ql_level_color(r.level), 0.68),
                      linewidth=r.level == 0 ? 0.45 : 0.65)
        end
    end
    _ql_overlay_solid!(ax, is_solid)
    xlims!(ax, 0.5, leaf_nx + 0.5)
    ylims!(ax, 0.5, leaf_ny + 0.5)
    save(path, fig)
    return path
end

function _ql_heatmap!(fig, row, col, title, A; colormap=:viridis,
                      colorrange=nothing)
    ax = Axis(fig[row, col]; title=title, xlabel="x leaf", ylabel="y leaf",
              aspect=DataAspect())
    cr = colorrange === nothing ? _ql_finite_colorrange(A) : colorrange
    Aplot = map(x -> isfinite(x) ? x : cr[1], A)
    hm = heatmap!(ax, 1:size(A, 1), 1:size(A, 2), Aplot;
                  colormap=colormap, colorrange=cr)
    Colorbar(fig[row, col + 1], hm)
    return ax
end

function _ql_plot_fields(path, state; title)
    speed_range = _ql_finite_colorrange(state.fields.speed)
    rho_range = _ql_finite_colorrange(state.fields.rho)
    ux_range = _ql_finite_colorrange(state.fields.ux; symmetric=true)
    fig = Figure(size=(1500, 950), fontsize=16)
    ax1 = _ql_heatmap!(fig, 1, 1, "$title |u|", state.fields.speed;
                       colormap=:viridis, colorrange=speed_range)
    _ql_overlay_solid!(ax1, state.is_solid)
    ax2 = _ql_heatmap!(fig, 1, 3, "$title ux", state.fields.ux;
                       colormap=:balance, colorrange=ux_range)
    _ql_overlay_solid!(ax2, state.is_solid)
    ax3 = _ql_heatmap!(fig, 2, 1, "$title rho", state.fields.rho;
                       colormap=:viridis, colorrange=rho_range)
    _ql_overlay_solid!(ax3, state.is_solid)
    ax4 = _ql_heatmap!(fig, 2, 3, "$title level", Float64.(state.level);
                       colormap=:viridis,
                       colorrange=_ql_safe_colorrange(
                           minimum(state.level), maximum(state.level)))
    _ql_overlay_solid!(ax4, state.is_solid)
    save(path, fig)
    return path
end

function _ql_lines_finite!(ax, xs, ys; label=nothing, kwargs...)
    first_i = nothing
    label_pending = label
    n = min(length(xs), length(ys))
    for k in 1:(n + 1)
        finite = k <= n && isfinite(xs[k]) && isfinite(ys[k])
        if finite && first_i === nothing
            first_i = k
        elseif (!finite || k == n + 1) && first_i !== nothing
            last_i = k - 1
            if last_i >= first_i
                if label_pending === nothing
                    lines!(ax, xs[first_i:last_i], ys[first_i:last_i]; kwargs...)
                else
                    lines!(ax, xs[first_i:last_i], ys[first_i:last_i];
                           label=label_pending, kwargs...)
                    label_pending = nothing
                end
            end
            first_i = nothing
        end
    end
    return ax
end

function _ql_has_finite_pairs(xs, ys)
    n = min(length(xs), length(ys))
    for k in 1:n
        isfinite(xs[k]) && isfinite(ys[k]) && return true
    end
    return false
end

function _ql_plot_profiles(path, result, state; title)
    profile, analytic = _ql_profile_vectors(result, state)
    nx = state.leaf_nx
    ny = state.leaf_ny
    j_mid = cld(ny, 2)
    i_probe = clamp(round(Int, 0.75 * nx), 1, nx)
    x = (collect(1:nx) .- 0.5) ./ nx
    y = (collect(1:ny) .- 0.5) ./ ny
    y_mean = length(profile) == 1 ? [0.0] :
        [(j - 1) / (length(profile) - 1) for j in eachindex(profile)]

    fig = Figure(size=(1350, 920), fontsize=16)
    ax1 = Axis(fig[1, 1]; title="$title mean ux profile",
               xlabel="ux", ylabel="y/Ly")
    _ql_lines_finite!(ax1, profile, y_mean; label="AMR-D",
                      color=:orangered, linewidth=2.8)
    _ql_lines_finite!(ax1, analytic, y_mean; label="steady analytic",
                      color=:black, linestyle=:dash, linewidth=2.2)
    (_ql_has_finite_pairs(profile, y_mean) ||
     _ql_has_finite_pairs(analytic, y_mean)) &&
        axislegend(ax1, position=:rb)

    ax2 = Axis(fig[1, 2]; title="$title centerline ux",
               xlabel="x/Lx", ylabel="ux")
    _ql_lines_finite!(ax2, x, state.fields.ux[:, j_mid];
                      color=:orangered, linewidth=2.5)

    ax3 = Axis(fig[2, 1]; title="$title vertical ux probe",
               xlabel="ux", ylabel="y/Ly")
    _ql_lines_finite!(ax3, state.fields.ux[i_probe, :], y;
                      color=:orangered, linewidth=2.5)

    ax4 = Axis(fig[2, 2]; title="$title centerline rho",
               xlabel="x/Lx", ylabel="rho")
    _ql_lines_finite!(ax4, x, state.fields.rho[:, j_mid];
                      color=:dodgerblue4, linewidth=2.5)
    save(path, fig)
    return path
end

function _ql_array_diff(A, B)
    size(A) == size(B) || return fill(NaN, size(A))
    C = fill(NaN, size(A))
    @inbounds for idx in eachindex(A, B)
        a = A[idx]
        b = B[idx]
        C[idx] = isfinite(a) && isfinite(b) ? a - b : NaN
    end
    return C
end

function _ql_plot_compare_fields(path, amr_state, reference_state; title)
    speed_range = _ql_finite_colorrange(vcat(vec(amr_state.fields.speed),
                                             vec(reference_state.fields.speed)))
    rho_range = _ql_finite_colorrange(vcat(vec(amr_state.fields.rho),
                                           vec(reference_state.fields.rho)))
    ux_diff = _ql_array_diff(amr_state.fields.ux, reference_state.fields.ux)
    rho_diff = _ql_array_diff(amr_state.fields.rho, reference_state.fields.rho)
    ux_diff_range = _ql_finite_colorrange(ux_diff; symmetric=true)
    rho_diff_range = _ql_finite_colorrange(rho_diff; symmetric=true)

    fig = Figure(size=(1900, 980), fontsize=16)
    ax1 = _ql_heatmap!(fig, 1, 1, "$title AMR-D |u|",
                       amr_state.fields.speed; colorrange=speed_range)
    _ql_overlay_solid!(ax1, amr_state.is_solid)
    ax2 = _ql_heatmap!(fig, 1, 3, "$title reference |u|",
                       reference_state.fields.speed; colorrange=speed_range)
    _ql_overlay_solid!(ax2, reference_state.is_solid)
    _ql_heatmap!(fig, 1, 5, "$title ux diff AMR-D - reference",
                 ux_diff; colormap=:balance, colorrange=ux_diff_range)

    ax3 = _ql_heatmap!(fig, 2, 1, "$title AMR-D rho",
                       amr_state.fields.rho; colorrange=rho_range)
    _ql_overlay_solid!(ax3, amr_state.is_solid)
    ax4 = _ql_heatmap!(fig, 2, 3, "$title reference rho",
                       reference_state.fields.rho; colorrange=rho_range)
    _ql_overlay_solid!(ax4, reference_state.is_solid)
    _ql_heatmap!(fig, 2, 5, "$title rho diff AMR-D - reference",
                 rho_diff; colormap=:balance, colorrange=rho_diff_range)
    save(path, fig)
    return path
end

function _ql_plot_compare_profiles(path, amr_result, amr_state,
                                   reference_result, reference_state; title)
    amr_profile, analytic = _ql_profile_vectors(amr_result, amr_state)
    ref_profile, _ = _ql_profile_vectors(reference_result, reference_state)
    nx = min(amr_state.leaf_nx, reference_state.leaf_nx)
    ny = min(amr_state.leaf_ny, reference_state.leaf_ny)
    j_mid = cld(ny, 2)
    i_probe = clamp(round(Int, 0.75 * nx), 1, nx)
    x = (collect(1:nx) .- 0.5) ./ nx
    y = (collect(1:ny) .- 0.5) ./ ny
    y_amr = length(amr_profile) == 1 ? [0.0] :
        [(j - 1) / (length(amr_profile) - 1) for j in eachindex(amr_profile)]
    y_ref = length(ref_profile) == 1 ? [0.0] :
        [(j - 1) / (length(ref_profile) - 1) for j in eachindex(ref_profile)]

    fig = Figure(size=(1500, 980), fontsize=16)
    ax1 = Axis(fig[1, 1]; title="$title mean ux",
               xlabel="ux", ylabel="y/Ly")
    _ql_lines_finite!(ax1, amr_profile, y_amr; label="AMR-D",
                      color=:orangered, linewidth=2.8)
    _ql_lines_finite!(ax1, ref_profile, y_ref;
                      label="classic Cartesian transient",
                      color=:dodgerblue4, linewidth=2.5)
    _ql_lines_finite!(ax1, analytic, y_amr; label="steady analytic",
                      color=:black, linestyle=:dash, linewidth=2.2)
    (_ql_has_finite_pairs(amr_profile, y_amr) ||
     _ql_has_finite_pairs(ref_profile, y_ref) ||
     _ql_has_finite_pairs(analytic, y_amr)) &&
        axislegend(ax1, position=:rb)

    ax2 = Axis(fig[1, 2]; title="$title centerline ux",
               xlabel="x/Lx", ylabel="ux")
    _ql_lines_finite!(ax2, x, amr_state.fields.ux[1:nx, j_mid];
                      label="AMR-D", color=:orangered, linewidth=2.5)
    _ql_lines_finite!(ax2, x, reference_state.fields.ux[1:nx, j_mid];
                      label="classic Cartesian transient", color=:dodgerblue4,
                      linewidth=2.3)
    (_ql_has_finite_pairs(x, amr_state.fields.ux[1:nx, j_mid]) ||
     _ql_has_finite_pairs(x, reference_state.fields.ux[1:nx, j_mid])) &&
        axislegend(ax2, position=:rb)

    ax3 = Axis(fig[2, 1]; title="$title vertical ux probe",
               xlabel="ux", ylabel="y/Ly")
    _ql_lines_finite!(ax3, amr_state.fields.ux[i_probe, 1:ny], y;
                      label="AMR-D", color=:orangered, linewidth=2.5)
    _ql_lines_finite!(ax3, reference_state.fields.ux[i_probe, 1:ny], y;
                      label="classic Cartesian transient", color=:dodgerblue4,
                      linewidth=2.3)
    (_ql_has_finite_pairs(amr_state.fields.ux[i_probe, 1:ny], y) ||
     _ql_has_finite_pairs(reference_state.fields.ux[i_probe, 1:ny], y)) &&
        axislegend(ax3, position=:rb)

    ax4 = Axis(fig[2, 2]; title="$title centerline rho",
               xlabel="x/Lx", ylabel="rho")
    _ql_lines_finite!(ax4, x, amr_state.fields.rho[1:nx, j_mid];
                      label="AMR-D", color=:orangered, linewidth=2.5)
    _ql_lines_finite!(ax4, x, reference_state.fields.rho[1:nx, j_mid];
                      label="classic Cartesian transient", color=:dodgerblue4,
                      linewidth=2.3)
    (_ql_has_finite_pairs(x, amr_state.fields.rho[1:nx, j_mid]) ||
     _ql_has_finite_pairs(x, reference_state.fields.rho[1:nx, j_mid])) &&
        axislegend(ax4, position=:rb)
    save(path, fig)
    return path
end

function _ql_profile_axis(n::Int)
    return n == 1 ? [0.0] : [(j - 1) / (n - 1) for j in 1:n]
end

function _ql_profile_residual(profile::AbstractVector,
                              analytic::AbstractVector)
    residual = fill(NaN, length(profile))
    n = length(profile)
    for k in eachindex(profile)
        y = n == 1 ? 0.0 : (k - 1) / (n - 1)
        ref = _ql_profile_interp(analytic, y)
        value = Float64(profile[k])
        residual[k] = isfinite(value) && isfinite(ref) ? value - ref : NaN
    end
    return residual
end

function _ql_plot_debug_dashboard(path, amr_result, amr_state,
                                  reference_result, reference_state; title,
                                  convergence_rows=nothing)
    ux_range = _ql_finite_colorrange(vcat(vec(amr_state.fields.ux),
                                          vec(reference_state.fields.ux));
                                     symmetric=true)
    rho_range = _ql_finite_colorrange(vcat(vec(amr_state.fields.rho),
                                           vec(reference_state.fields.rho)))
    level_min = min(minimum(amr_state.level), minimum(reference_state.level))
    level_max = max(maximum(amr_state.level), maximum(reference_state.level))
    level_range = _ql_safe_colorrange(level_min, level_max)
    amr_mesh = _ql_mesh_cells_from_result_state(amr_result, amr_state)
    ref_mesh = _ql_mesh_cells_from_result_state(reference_result,
                                               reference_state)

    nx = min(amr_state.leaf_nx, reference_state.leaf_nx)
    ny = min(amr_state.leaf_ny, reference_state.leaf_ny)
    i_probe = clamp(_ql_probe_i_from_max_level(amr_state.level), 1, nx)
    y_probe = (collect(1:ny) .- 0.5) ./ ny

    has_convergence = convergence_rows !== nothing && !isempty(convergence_rows)
    fig = Figure(size=(1900, has_convergence ? 2050 : 1650), fontsize=15)
    Label(fig[0, 1:6], title; fontsize=22, tellwidth=false)

    ax11 = _ql_heatmap!(fig, 1, 1, "Cartesian transient reference mesh",
                        Float64.(reference_state.level);
                        colormap=:viridis, colorrange=level_range)
    _ql_overlay_mesh_wireframe!(ax11, ref_mesh;
                                leaf_nx=reference_state.leaf_nx,
                                leaf_ny=reference_state.leaf_ny,
                                alpha=0.72, linewidth=0.55)
    _ql_overlay_vertical_probe!(ax11, i_probe, reference_state.leaf_ny)
    _ql_overlay_solid!(ax11, reference_state.is_solid)
    ax12 = _ql_heatmap!(fig, 1, 3, "Cartesian transient ux + mesh",
                        reference_state.fields.ux;
                        colormap=:balance, colorrange=ux_range)
    _ql_overlay_mesh_wireframe!(ax12, ref_mesh;
                                leaf_nx=reference_state.leaf_nx,
                                leaf_ny=reference_state.leaf_ny,
                                alpha=0.35, linewidth=0.45)
    _ql_overlay_vertical_probe!(ax12, i_probe, reference_state.leaf_ny)
    _ql_overlay_solid!(ax12, reference_state.is_solid)
    ax13 = _ql_heatmap!(fig, 1, 5, "Cartesian transient rho + mesh",
                        reference_state.fields.rho;
                        colormap=:viridis, colorrange=rho_range)
    _ql_overlay_mesh_wireframe!(ax13, ref_mesh;
                                leaf_nx=reference_state.leaf_nx,
                                leaf_ny=reference_state.leaf_ny,
                                alpha=0.35, linewidth=0.45)
    _ql_overlay_vertical_probe!(ax13, i_probe, reference_state.leaf_ny)
    _ql_overlay_solid!(ax13, reference_state.is_solid)

    ax21 = _ql_heatmap!(fig, 2, 1, "AMR-D mesh",
                        Float64.(amr_state.level);
                        colormap=:viridis, colorrange=level_range)
    _ql_overlay_mesh_wireframe!(ax21, amr_mesh;
                                leaf_nx=amr_state.leaf_nx,
                                leaf_ny=amr_state.leaf_ny,
                                alpha=0.96, linewidth=0.95)
    _ql_overlay_vertical_probe!(ax21, i_probe, amr_state.leaf_ny)
    _ql_overlay_solid!(ax21, amr_state.is_solid)
    ax22 = _ql_heatmap!(fig, 2, 3, "AMR-D ux + mesh",
                        amr_state.fields.ux;
                        colormap=:balance, colorrange=ux_range)
    _ql_overlay_mesh_wireframe!(ax22, amr_mesh;
                                leaf_nx=amr_state.leaf_nx,
                                leaf_ny=amr_state.leaf_ny,
                                alpha=0.92, linewidth=0.82)
    _ql_overlay_vertical_probe!(ax22, i_probe, amr_state.leaf_ny)
    _ql_overlay_solid!(ax22, amr_state.is_solid)
    ax23 = _ql_heatmap!(fig, 2, 5, "AMR-D rho + mesh",
                        amr_state.fields.rho;
                        colormap=:viridis, colorrange=rho_range)
    _ql_overlay_mesh_wireframe!(ax23, amr_mesh;
                                leaf_nx=amr_state.leaf_nx,
                                leaf_ny=amr_state.leaf_ny,
                                alpha=0.92, linewidth=0.82)
    _ql_overlay_vertical_probe!(ax23, i_probe, amr_state.leaf_ny)
    _ql_overlay_solid!(ax23, amr_state.is_solid)

    amr_profile, analytic = _ql_profile_vectors(amr_result, amr_state)
    ref_profile, _ = _ql_profile_vectors(reference_result, reference_state)
    y_amr = _ql_profile_axis(length(amr_profile))
    y_ref = _ql_profile_axis(length(ref_profile))
    y_analytic = _ql_profile_axis(length(analytic))

    ax31 = Axis(fig[3, 1:2];
                title="row-mean ux(y), averaged over all fluid x",
                xlabel="ux", ylabel="y/Ly")
    _ql_lines_finite!(ax31, ref_profile, y_ref;
                      label="classic Cartesian transient",
                      color=:dodgerblue4, linewidth=2.5)
    _ql_lines_finite!(ax31, amr_profile, y_amr; label="AMR-D",
                      color=:orangered, linewidth=2.8)
    _ql_lines_finite!(ax31, analytic, y_analytic; label="steady analytic",
                      color=:black, linestyle=:dash, linewidth=2.2)
    (_ql_has_finite_pairs(ref_profile, y_ref) ||
     _ql_has_finite_pairs(amr_profile, y_amr) ||
     _ql_has_finite_pairs(analytic, y_analytic)) &&
        axislegend(ax31, position=:rb)

    ax32 = Axis(fig[3, 3:4];
                title="vertical ux(y) probe at x leaf $i_probe",
                xlabel="ux", ylabel="y/Ly")
    _ql_lines_finite!(ax32, reference_state.fields.ux[i_probe, 1:ny],
                      y_probe; label="classic Cartesian transient",
                      color=:dodgerblue4, linewidth=2.4)
    _ql_lines_finite!(ax32, amr_state.fields.ux[i_probe, 1:ny],
                      y_probe; label="AMR-D", color=:orangered,
                      linewidth=2.6)
    _ql_lines_finite!(ax32, analytic, y_analytic; label="steady analytic",
                      color=:black, linestyle=:dash, linewidth=2.0)
    (_ql_has_finite_pairs(reference_state.fields.ux[i_probe, 1:ny],
                          y_probe) ||
     _ql_has_finite_pairs(amr_state.fields.ux[i_probe, 1:ny], y_probe) ||
     _ql_has_finite_pairs(analytic, y_analytic)) &&
        axislegend(ax32, position=:rb)

    probe_diff = amr_state.fields.ux[i_probe, 1:ny] .-
        reference_state.fields.ux[i_probe, 1:ny]
    mean_diff = _ql_profile_residual(amr_profile, ref_profile)
    err_range = _ql_finite_colorrange(vcat(probe_diff, mean_diff);
                                      symmetric=true)
    ax33 = Axis(fig[3, 5:6];
                title="AMR-D minus Cartesian profile errors",
                xlabel="ux difference", ylabel="y/Ly")
    _ql_lines_finite!(ax33, mean_diff, y_amr; label="row mean",
                      color=:purple4, linewidth=2.4)
    _ql_lines_finite!(ax33, probe_diff, y_probe; label="vertical probe",
                      color=:orangered, linewidth=2.6)
    xlims!(ax33, err_range...)
    (_ql_has_finite_pairs(mean_diff, y_amr) ||
     _ql_has_finite_pairs(probe_diff, y_probe)) &&
        axislegend(ax33, position=:rb)

    if has_convergence
        steps = [r.step for r in convergence_rows]
        ux_linf = [r.ux_linf_delta for r in convergence_rows]
        ux_limit = [r.ux_delta_limit for r in convergence_rows]
        rho_linf = [r.rho_linf_delta for r in convergence_rows]
        rho_limit = [r.rho_delta_limit for r in convergence_rows]
        mass_drift = [r.mass_rel_drift for r in convergence_rows]

        ax41 = Axis(fig[4, 1:2]; title="temporal ux convergence",
                    xlabel="steps", ylabel="Linf delta ux")
        _ql_lines_finite!(ax41, steps, ux_linf; label="delta ux",
                          color=:dodgerblue4, linewidth=2.4)
        _ql_lines_finite!(ax41, steps, ux_limit; label="limit",
                          color=:black, linestyle=:dash, linewidth=2.0)
        axislegend(ax41, position=:rt)

        ax42 = Axis(fig[4, 3:4]; title="temporal rho convergence",
                    xlabel="steps", ylabel="Linf delta rho")
        _ql_lines_finite!(ax42, steps, rho_linf; label="delta rho",
                          color=:seagreen4, linewidth=2.4)
        _ql_lines_finite!(ax42, steps, rho_limit; label="limit",
                          color=:black, linestyle=:dash, linewidth=2.0)
        axislegend(ax42, position=:rt)

        ax43 = Axis(fig[4, 5:6]; title="mass drift after correction",
                    xlabel="steps", ylabel="relative drift")
        _ql_lines_finite!(ax43, steps, mass_drift; label="mass drift",
                          color=:crimson, linewidth=2.4)
    end

    save(path, fig)
    return path
end

function _ql_run_cartesian_classic_channel(setup, case; steps, T)
    domain = getproperty(setup, :domain)
    ml = case.max_level
    scale = 1 << ml
    reference_steps = Int(steps) * scale
    nx = Int(domain.Nx) * scale
    ny = Int(domain.Ny) * scale
    volume = one(T) / T(scale * scale)
    rho0 = T(_ql_var(setup, :rho0, 1.0))
    omega = T(_ql_var(setup, :omega, 1.0))
    F = zeros(T, nx, ny, 9)
    Ftmp = similar(F)
    fill_equilibrium_integrated_D2Q9!(F, volume, rho0, zero(T), zero(T))
    mass_initial = sum(F)
    force_x = zero(T)
    force_y = zero(T)

    for _ in 1:reference_steps
        if case.flow == :poiseuille
            Fx = T(_ql_body_force(setup, :Fx, _ql_var(setup, :Fx, 1e-6)))
            force_x = Fx
            collide_Guo_integrated_D2Q9!(F, volume, omega, Fx, zero(T))
            stream_periodic_x_wall_y_F_2d!(Ftmp, F)
        elseif case.flow == :couette
            U = T(_ql_boundary_value(
                setup, :north, :velocity, :ux, _ql_var(setup, :U, 1e-3)))
            collide_BGK_integrated_D2Q9!(F, volume, omega)
            stream_periodic_x_moving_wall_y_F_2d!(
                Ftmp, F; u_south=zero(T), u_north=U,
                rho_wall=rho0, volume=volume)
        else
            throw(ArgumentError("classic Cartesian reference supports nested channels only"))
        end
        F, Ftmp = Ftmp, F
    end

    fields = _ql_leaf_fields(F, falses(nx, ny); volume=volume,
                             force_x=force_x, force_y=force_y)
    state = (; fields, is_solid=falses(nx, ny), level=fill(ml, nx, ny),
             patch=nothing, leaf_nx=nx, leaf_ny=ny)
    profile = T.(_ql_mean_ux_by_y(state))
    analytic = case.flow == :poiseuille ?
        T.(poiseuille_analytic_profile_2d(ny, force_x, omega; rho=rho0)) :
        T.(couette_analytic_profile_2d(
            ny, _ql_boundary_value(setup, :north, :velocity, :ux,
                                   _ql_var(setup, :U, 1e-3))))
    l2, linf = _ql_profile_errors(profile, analytic)
    mass_final = sum(F)
    return AMRDCartesianChannelQuicklook2D{T}(
        case.flow, F, profile, analytic, T(l2), T(linf), mass_initial,
        mass_final, mass_final - mass_initial, reference_steps, volume,
        force_x, force_y, ml)
end

function _ql_run_one_level_case(setup, case; steps, avg_window, method, T)
    domain = getproperty(setup, :domain)
    Nx = Int(domain.Nx)
    Ny = Int(domain.Ny)
    rho = _ql_var(setup, :rho0, 1.0)
    omega = _ql_var(setup, :omega, 1.0)
    patch_i, patch_j = method == :leaf_oracle ? (1:Nx, 1:Ny) :
        _ql_patch_ranges(setup)

    if case.flow == :poiseuille
        Fx = _ql_body_force(setup, :Fx, _ql_var(setup, :Fx, 5e-5))
        runner = method == :leaf_oracle ?
            run_conservative_tree_poiseuille_macroflow_2d :
            run_conservative_tree_poiseuille_route_native_2d
        result = runner(; Nx=Nx, Ny=Ny, patch_i_range=patch_i,
                        patch_j_range=patch_j, Fx=Fx, omega=omega,
                        rho=rho, steps=steps, T=T)
        return result, (force_x=Fx, force_y=0.0)
    elseif case.flow == :couette
        U = _ql_boundary_value(setup, :north, :velocity, :ux,
                               _ql_var(setup, :U, 1e-3))
        runner = method == :leaf_oracle ?
            run_conservative_tree_couette_macroflow_2d :
            run_conservative_tree_couette_route_native_2d
        result = runner(; Nx=Nx, Ny=Ny, patch_i_range=patch_i,
                        patch_j_range=patch_j, U=U, omega=omega,
                        rho=rho, steps=steps, T=T)
        return result, (force_x=0.0, force_y=0.0)
    elseif case.flow == :bfs
        u_in = _ql_boundary_value(setup, :west, :velocity, :ux,
                                  _ql_var(setup, :u_in, 0.03))
        rho_out = _ql_boundary_value(setup, :east, :pressure, :rho, 1.0)
        kwargs = (; Nx=Nx, Ny=Ny, patch_i_range=patch_i,
                  patch_j_range=patch_j,
                  step_i_leaf=round(Int, _ql_var(setup, :step_i_leaf, 16)),
                  step_height_leaf=round(Int, _ql_var(setup, :step_height_leaf, 8)),
                  u_in=u_in, omega=omega, rho=rho, steps=steps, T=T)
        result = method == :leaf_oracle ?
            run_conservative_tree_bfs_macroflow_2d(; kwargs...) :
            run_conservative_tree_bfs_route_native_2d(; kwargs...,
                                                      rho_out=rho_out)
        return result, (force_x=0.0, force_y=0.0)
    elseif case.flow == :square
        Fx = _ql_body_force(setup, :Fx, _ql_var(setup, :Fx, 2e-5))
        Fy = _ql_body_force(setup, :Fy, _ql_var(setup, :Fy, 0.0))
        runner = method == :leaf_oracle ?
            run_conservative_tree_square_obstacle_macroflow_2d :
            run_conservative_tree_square_obstacle_route_native_2d
        result = runner(; Nx=Nx, Ny=Ny, patch_i_range=patch_i,
                        patch_j_range=patch_j,
                        obstacle_i_range=round(Int, _ql_var(setup, :obstacle_i0, 22)):
                                         round(Int, _ql_var(setup, :obstacle_i1, 27)),
                        obstacle_j_range=round(Int, _ql_var(setup, :obstacle_j0, 12)):
                                         round(Int, _ql_var(setup, :obstacle_j1, 17)),
                        Fx=Fx, Fy=Fy, omega=omega, rho=rho, steps=steps, T=T)
        return result, (force_x=Fx, force_y=Fy)
    elseif case.flow == :cylinder
        Fx = _ql_body_force(setup, :Fx, _ql_var(setup, :Fx, 2e-5))
        runner = method == :leaf_oracle ?
            run_conservative_tree_cylinder_macroflow_2d :
            run_conservative_tree_cylinder_obstacle_route_native_2d
        result = runner(; Nx=Nx, Ny=Ny, patch_i_range=patch_i,
                        patch_j_range=patch_j,
                        cx_leaf=_ql_var(setup, :cx_leaf, (2 * Nx + 1) / 2),
                        cy_leaf=_ql_var(setup, :cy_leaf, (2 * Ny + 1) / 2),
                        radius_leaf=_ql_var(setup, :radius_leaf, 3.0),
                        Fx=Fx, omega=omega, rho=rho, steps=steps,
                        avg_window=avg_window, T=T)
        return result, (force_x=Fx, force_y=0.0)
    end

    throw(ArgumentError("quicklook does not dispatch flow $(case.flow)"))
end

function _ql_write_summary_csv(path, artifacts)
    open(path, "w") do io
        println(io, "case,flow,method,status,outdir,status_csv,mesh_csv,mesh_png,fields_csv,fields_png,profiles_csv,profiles_png,values_csv,fields_compare_png,profiles_compare_png,debug_dashboard_png")
        for a in artifacts
            values_csv = joinpath(a.outdir, "values.csv")
            fields_compare_png = joinpath(a.outdir, "fields_compare.png")
            profiles_compare_png = joinpath(a.outdir, "profiles_compare.png")
            debug_dashboard_png = joinpath(a.outdir, "debug_dashboard.png")
            println(io, join((a.case_name, a.flow, a.method, a.status, a.outdir,
                              a.status_csv, a.mesh_csv, a.mesh_png,
                              a.fields_csv, a.fields_png,
                              a.profiles_csv, a.profiles_png,
                              isfile(values_csv) ? values_csv : "",
                              isfile(fields_compare_png) ? fields_compare_png : "",
                              isfile(profiles_compare_png) ? profiles_compare_png : "",
                              isfile(debug_dashboard_png) ? debug_dashboard_png : ""), ","))
        end
    end
    return path
end

function _ql_method_values(record, reference)
    profile, analytic = _ql_profile_vectors(record.result, record.state)
    l2_analytic, linf_analytic = _ql_profile_errors(profile, analytic)
    ux_min, ux_max = _ql_finite_minmax(record.state.fields.ux)
    uy_min, uy_max = _ql_finite_minmax(record.state.fields.uy)
    rho_min, rho_max = _ql_finite_minmax(record.state.fields.rho)
    _, speed_max = _ql_finite_minmax(record.state.fields.speed)
    if reference === nothing
        l2_ref = NaN
        linf_ref = NaN
        ux_l2_ref = NaN
        ux_linf_ref = NaN
        rho_l2_ref = NaN
        rho_linf_ref = NaN
    else
        ref_profile, _ = _ql_profile_vectors(reference.result, reference.state)
        l2_ref, linf_ref = _ql_profile_errors(profile, ref_profile)
        ux_l2_ref, ux_linf_ref = _ql_field_errors(
            record.state.fields.ux, reference.state.fields.ux)
        rho_l2_ref, rho_linf_ref = _ql_field_errors(
            record.state.fields.rho, reference.state.fields.rho)
    end
    return (;
        method=record.method,
        steps=getproperty(record.result, :steps),
        mass_rel_drift=_ql_mass_rel_drift(record.result),
        max_raw_mass_rel_drift=hasproperty(record.result, :max_raw_relative_mass_drift) ?
            getproperty(record.result, :max_raw_relative_mass_drift) : NaN,
        ux_mean=_ql_finite_mean(record.state.fields.ux),
        uy_mean=_ql_finite_mean(record.state.fields.uy),
        rho_mean=_ql_finite_mean(record.state.fields.rho),
        ux_min=ux_min,
        ux_max=ux_max,
        uy_min=uy_min,
        uy_max=uy_max,
        rho_min=rho_min,
        rho_max=rho_max,
        speed_max=speed_max,
        l2_profile_vs_analytic=l2_analytic,
        linf_profile_vs_analytic=linf_analytic,
        l2_profile_vs_reference=l2_ref,
        linf_profile_vs_reference=linf_ref,
        l2_ux_field_vs_reference=ux_l2_ref,
        linf_ux_field_vs_reference=ux_linf_ref,
        l2_rho_field_vs_reference=rho_l2_ref,
        linf_rho_field_vs_reference=rho_linf_ref)
end

function _ql_write_values_csv(path, records)
    reference = nothing
    for record in records
        if record.method in (:cartesian_classic, :leaf_oracle)
            reference = record
            break
        end
    end
    open(path, "w") do io
        println(io, "method,steps,mass_rel_drift,max_raw_mass_rel_drift,ux_mean,uy_mean,rho_mean,ux_min,ux_max,uy_min,uy_max,rho_min,rho_max,speed_max,l2_profile_vs_analytic,linf_profile_vs_analytic,l2_profile_vs_reference,linf_profile_vs_reference,l2_ux_field_vs_reference,linf_ux_field_vs_reference,l2_rho_field_vs_reference,linf_rho_field_vs_reference")
        for record in records
            values = _ql_method_values(
                record, record.method in (:cartesian_classic, :leaf_oracle) ?
                nothing : reference)
            @printf(io, "%s,%d,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e\n",
                    String(values.method), values.steps, values.mass_rel_drift,
                    values.max_raw_mass_rel_drift,
                    values.ux_mean, values.uy_mean, values.rho_mean,
                    values.ux_min, values.ux_max, values.uy_min, values.uy_max,
                    values.rho_min, values.rho_max, values.speed_max,
                    values.l2_profile_vs_analytic,
                    values.linf_profile_vs_analytic,
                    values.l2_profile_vs_reference,
                    values.linf_profile_vs_reference,
                    values.l2_ux_field_vs_reference,
                    values.linf_ux_field_vs_reference,
                    values.l2_rho_field_vs_reference,
                    values.linf_rho_field_vs_reference)
        end
    end
    return path
end

function _ql_empty_artifact(setup, case, outdir, status_csv, mesh_csv, mesh_png)
    return AMRDQuicklookArtifact2D(
        case.name, case.flow, :none, case.runtime_status, outdir, status_csv,
        mesh_csv, mesh_png, "", "", "", "")
end

function _ql_artifact(setup, case, method, outdir, status_csv,
                      mesh_csv, mesh_png, fields_csv, fields_png,
                      profiles_csv, profiles_png)
    return AMRDQuicklookArtifact2D(
        case.name, case.flow, method, :ok, outdir, status_csv, mesh_csv,
        mesh_png, fields_csv, fields_png, profiles_csv, profiles_png)
end

function run_amr_d_quicklook_from_krk_2d(paths;
        outdir::AbstractString=DEFAULT_QUICKLOOK_OUTDIR,
        steps_override=nothing,
        avg_window_override=nothing,
        include_reference::Bool=true,
        make_plots::Bool=true,
        T::Type{<:AbstractFloat}=Float64)
    mkpath(outdir)
    artifacts = AMRDQuicklookArtifact2D[]

    for raw_path in paths
        setup = load_kraken(String(raw_path))
        case = conservative_tree_amr_d_case_from_krk_2d(setup)
        case_dir = joinpath(outdir, _ql_sanitize_name(case.name))
        mkpath(case_dir)
        status_csv = _ql_write_status_csv(joinpath(case_dir, "status.csv"),
                                          setup, case)

        mesh_csv = ""
        mesh_png = ""
        if case.spec_supported
            spec = create_conservative_tree_spec_from_krk_2d(setup)
            mesh_rows = _ql_mesh_cells_from_spec(spec)
            leaf_nx = spec.Nx << spec.max_level
            leaf_ny = spec.Ny << spec.max_level
            solid = _ql_static_solid_mask(setup, case, leaf_nx, leaf_ny)
            mesh_csv = _ql_write_mesh_csv(joinpath(case_dir, "mesh_static.csv"),
                                          mesh_rows)
            mesh_png = joinpath(case_dir, "mesh_static.png")
            make_plots && _ql_plot_mesh(
                mesh_png, mesh_rows; title="$(case.name) static AMR mesh",
                leaf_nx=leaf_nx, leaf_ny=leaf_ny, is_solid=solid)
        end

        if !case.runtime_supported
            push!(artifacts, _ql_empty_artifact(
                setup, case, case_dir, status_csv, mesh_csv, mesh_png))
            continue
        end

        steps = _ql_steps(setup; steps_override=steps_override)
        avg = _ql_avg_window(setup, steps; avg_window_override=avg_window_override)
        methods = case.runtime_status == :subcycled_nested_channel && include_reference ?
            (:amr_d, :cartesian_classic) :
            case.max_level <= 1 && include_reference ?
            (:amr_d, :leaf_oracle) : (:amr_d,)
        records = NamedTuple[]

        for method in methods
            result = nothing
            force = (force_x=0.0, force_y=0.0)
            if method == :cartesian_classic
                result = _ql_run_cartesian_classic_channel(setup, case;
                                                           steps=steps, T=T)
                force = (force_x=getproperty(result, :force_x),
                         force_y=getproperty(result, :force_y))
            elseif case.runtime_status in (:subcycled_nested_channel,
                                           :subcycled_nested_solid)
                result = run_conservative_tree_amr_d_case_from_krk_2d(
                    setup; steps_override=steps, T=T)
                force = (force_x=_ql_body_force(setup, :Fx, 0.0),
                         force_y=_ql_body_force(setup, :Fy, 0.0))
            else
                result, force = _ql_run_one_level_case(
                    setup, case; steps=steps, avg_window=avg,
                    method=method == :leaf_oracle ? :leaf_oracle : :amr_d,
                    T=T)
            end

            state = result isa AMRDCartesianChannelQuicklook2D ?
                _ql_state_from_cartesian_channel_result(result) :
                hasproperty(result, :spec) ?
                _ql_state_from_spec_result(result; force_x=force.force_x,
                                           force_y=force.force_y,
                                           level_scaled_force=
                                               case.runtime_status in
                                               (:subcycled_nested_channel,
                                                :subcycled_nested_solid) &&
                                               case.flow != :couette) :
                _ql_state_from_composite_result(result; force_x=force.force_x,
                                                force_y=force.force_y)
            mesh_rows = result isa AMRDCartesianChannelQuicklook2D ?
                _ql_mesh_cells_uniform(state.leaf_nx, state.leaf_ny,
                                       getproperty(result, :max_level)) :
                state.patch === nothing ?
                _ql_mesh_cells_from_spec(getproperty(result, :spec)) :
                _ql_mesh_cells_from_patch(
                    div(state.leaf_nx, 2), div(state.leaf_ny, 2), state.patch)

            prefix = String(method)
            mesh_csv_m = _ql_write_mesh_csv(
                joinpath(case_dir, "mesh_$(prefix).csv"), mesh_rows)
            fields_csv = _ql_write_fields_csv(
                joinpath(case_dir, "fields_$(prefix).csv"), state)
            profiles_csv = _ql_write_profiles_csv(
                joinpath(case_dir, "profiles_$(prefix).csv"), result, state)
            mesh_png_m = joinpath(case_dir, "mesh_$(prefix).png")
            fields_png = joinpath(case_dir, "fields_$(prefix).png")
            profiles_png = joinpath(case_dir, "profiles_$(prefix).png")
            if make_plots
                _ql_plot_mesh(mesh_png_m, mesh_rows;
                              title="$(case.name) $(prefix) mesh",
                              leaf_nx=state.leaf_nx, leaf_ny=state.leaf_ny,
                              is_solid=state.is_solid)
                _ql_plot_fields(fields_png, state;
                                title="$(case.name) $(prefix)")
                _ql_plot_profiles(profiles_png, result, state;
                                  title="$(case.name) $(prefix)")
            end
            push!(records, (; method, result, state))
            push!(artifacts, _ql_artifact(
                setup, case, method, case_dir, status_csv, mesh_csv_m,
                mesh_png_m, fields_csv, fields_png, profiles_csv, profiles_png))
        end

        if !isempty(records)
            _ql_write_values_csv(joinpath(case_dir, "values.csv"), records)
            amr_record = findfirst(record -> record.method == :amr_d, records)
            ref_record = findfirst(record -> record.method in
                                  (:cartesian_classic, :leaf_oracle), records)
            if make_plots && amr_record !== nothing && ref_record !== nothing
                amr = records[amr_record]
                ref = records[ref_record]
                _ql_plot_compare_fields(
                    joinpath(case_dir, "fields_compare.png"),
                    amr.state, ref.state; title=case.name)
                _ql_plot_compare_profiles(
                    joinpath(case_dir, "profiles_compare.png"),
                    amr.result, amr.state, ref.result, ref.state;
                    title=case.name)
                _ql_plot_debug_dashboard(
                    joinpath(case_dir, "debug_dashboard.png"),
                    amr.result, amr.state, ref.result, ref.state;
                    title=case.name)
            end
        end
    end

    _ql_write_summary_csv(joinpath(outdir, "summary.csv"), artifacts)
    return artifacts
end

run_amr_d_quicklook_from_krk_2d(path::AbstractString; kwargs...) =
    run_amr_d_quicklook_from_krk_2d([path]; kwargs...)

function main()
    paths = _ql_env_case_paths()
    outdir = get(ENV, "KRK_AMR_D_QUICKLOOK_OUTDIR", DEFAULT_QUICKLOOK_OUTDIR)
    include_reference = lowercase(get(ENV, "KRK_AMR_D_QUICKLOOK_REFERENCE", "true")) != "false"
    artifacts = run_amr_d_quicklook_from_krk_2d(
        paths; outdir=outdir, include_reference=include_reference)
    println("wrote ", joinpath(outdir, "summary.csv"))
    for artifact in artifacts
        println("wrote ", artifact.outdir)
    end
    return artifacts
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
