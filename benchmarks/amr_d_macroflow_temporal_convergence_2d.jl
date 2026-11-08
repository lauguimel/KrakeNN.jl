#!/usr/bin/env julia

using CairoMakie
using Dates
using KrakeNN
using Printf

include(joinpath(@__DIR__, "amr_d_quicklook_from_krk_2d.jl"))

const TEMP_CASE_DIR = joinpath(@__DIR__, "krk", "amr_d_convergence_2d")
const TEMP_DEFAULT_OUTDIR = joinpath(
    @__DIR__, "results", "quicklook", "amr_d_temporal_convergence_20260507")

const TEMP_DEFAULT_CASES = [
    "poiseuille_xband_scale1.krk",
    "poiseuille_yband_scale1.krk",
    "couette_scale1.krk",
    "bfs_scale1.krk",
    "square_scale1.krk",
    "cylinder_scale1.krk",
    "poiseuille_xband_nested4_debug.krk",
    "poiseuille_yband_nested4_debug.krk",
    "poiseuille_wall_ybands_nested4_debug.krk",
    "couette_yband_nested4_debug.krk",
    "cylinder_nested4_probe.krk",
]

function _temp_env_float(name::AbstractString, default)
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return Float64(default)
    return parse(Float64, raw)
end

function _temp_env_int(name::AbstractString, default)
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return Int(default)
    return parse(Int, raw)
end

function _temp_env_bool(name::AbstractString, default=false)
    raw = lowercase(strip(get(ENV, name, "")))
    isempty(raw) && return Bool(default)
    return raw in ("1", "true", "yes", "on")
end

function _temp_case_paths()
    raw = strip(get(ENV, "KRK_AMR_D_TEMP_CASES", ""))
    names = isempty(raw) ? TEMP_DEFAULT_CASES : split(raw, ",")
    paths = String[]
    for name in names
        s = strip(String(name))
        isempty(s) && continue
        path = isabspath(s) ? s : joinpath(TEMP_CASE_DIR, s)
        endswith(path, ".krk") || (path *= ".krk")
        push!(paths, path)
    end
    return paths
end

function _temp_reference_method(case)
    case.runtime_status == :subcycled_nested_channel && return :cartesian_classic
    case.max_level <= 1 && return :leaf_oracle
    return nothing
end

function _temp_run_case(setup, case; steps::Int, method::Symbol, T=Float64)
    avg = _ql_avg_window(setup, steps)
    force = (force_x=0.0, force_y=0.0)
    if method == :cartesian_classic
        result = _ql_run_cartesian_classic_channel(setup, case; steps=steps, T=T)
        force = (force_x=getproperty(result, :force_x),
                 force_y=getproperty(result, :force_y))
    elseif method == :amr_d && case.runtime_status in
            (:subcycled_nested_channel, :subcycled_nested_solid)
        result = run_conservative_tree_amr_d_case_from_krk_2d(
            setup; steps_override=steps, T=T)
        force = (force_x=_ql_body_force(setup, :Fx, 0.0),
                 force_y=_ql_body_force(setup, :Fy, 0.0))
    else
        result, force = _ql_run_one_level_case(
            setup, case; steps=steps, avg_window=avg,
            method=method == :leaf_oracle ? :leaf_oracle : :amr_d, T=T)
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
    return (; result, state, force)
end

function _temp_finite_linf_delta(A, B)
    size(A) == size(B) || return NaN
    linf = 0.0
    count = 0
    @inbounds for idx in eachindex(A, B)
        a = Float64(A[idx])
        b = Float64(B[idx])
        isfinite(a) && isfinite(b) || continue
        linf = max(linf, abs(a - b))
        count += 1
    end
    return count == 0 ? NaN : linf
end

function _temp_finite_l2_delta(A, B)
    size(A) == size(B) || return NaN
    sq = 0.0
    count = 0
    @inbounds for idx in eachindex(A, B)
        a = Float64(A[idx])
        b = Float64(B[idx])
        isfinite(a) && isfinite(b) || continue
        d = a - b
        sq += d * d
        count += 1
    end
    return count == 0 ? NaN : sqrt(sq / count)
end

function _temp_finite_maxabs(A)
    vmax = 0.0
    count = 0
    for x in A
        v = Float64(x)
        isfinite(v) || continue
        vmax = max(vmax, abs(v))
        count += 1
    end
    return count == 0 ? NaN : vmax
end

function _temp_values_row(case, method, result, state, reference)
    record = (; method, result, state)
    ref_record = reference === nothing ? nothing :
        (; method=reference.method, result=reference.result,
           state=reference.state)
    vals = _ql_method_values(record, ref_record)
    return merge(vals, (case=case.name, flow=case.flow))
end

function _temp_write_values_csv(path, rows)
    open(path, "w") do io
        println(io, "case,flow,method,steps,mass_rel_drift,max_raw_mass_rel_drift,ux_mean,uy_mean,rho_mean,ux_min,ux_max,uy_min,uy_max,rho_min,rho_max,speed_max,l2_profile_vs_analytic,linf_profile_vs_analytic,l2_profile_vs_reference,linf_profile_vs_reference,l2_ux_field_vs_reference,linf_ux_field_vs_reference,l2_rho_field_vs_reference,linf_rho_field_vs_reference")
        for r in rows
            @printf(io, "%s,%s,%s,%d,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e\n",
                    r.case, r.flow, r.method, r.steps, r.mass_rel_drift,
                    r.max_raw_mass_rel_drift, r.ux_mean, r.uy_mean, r.rho_mean,
                    r.ux_min, r.ux_max, r.uy_min, r.uy_max, r.rho_min,
                    r.rho_max, r.speed_max, r.l2_profile_vs_analytic,
                    r.linf_profile_vs_analytic, r.l2_profile_vs_reference,
                    r.linf_profile_vs_reference, r.l2_ux_field_vs_reference,
                    r.linf_ux_field_vs_reference, r.l2_rho_field_vs_reference,
                    r.linf_rho_field_vs_reference)
        end
    end
    return path
end

function _temp_write_convergence_csv(path, rows)
    open(path, "w") do io
        println(io, "case,step,status,ux_linf_delta,ux_l2_delta,rho_linf_delta,rho_l2_delta,ux_delta_limit,rho_delta_limit,ux_maxabs,ux_mean,rho_mean,mass_rel_drift")
        for r in rows
            @printf(io, "%s,%d,%s,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e\n",
                    r.case, r.step, r.status, r.ux_linf_delta,
                    r.ux_l2_delta, r.rho_linf_delta, r.rho_l2_delta,
                    r.ux_delta_limit, r.rho_delta_limit, r.ux_maxabs,
                    r.ux_mean, r.rho_mean, r.mass_rel_drift)
        end
    end
    return path
end

function _temp_existing_case_summary(case, case_dir::AbstractString)
    dashboard_png = joinpath(case_dir, "debug_dashboard.png")
    convergence_csv = joinpath(case_dir, "convergence.csv")
    values_csv = joinpath(case_dir, "values.csv")
    all(isfile, (dashboard_png, convergence_csv, values_csv)) ||
        return nothing

    lines = readlines(convergence_csv)
    length(lines) >= 2 || return nothing
    fields = split(lines[end], ",")
    length(fields) >= 3 || return nothing
    return (;
        case=case.name, flow=case.flow, status=Symbol(fields[3]),
        steps=parse(Int, fields[2]), outdir=case_dir,
        dashboard_png=dashboard_png, convergence_csv=convergence_csv,
        values_csv=values_csv)
end

function _temp_plot_single_dashboard(path, case, amr, convergence_rows)
    state = amr.state
    speed_range = _ql_finite_colorrange(state.fields.speed)
    rho_range = _ql_finite_colorrange(state.fields.rho)
    ux_range = _ql_finite_colorrange(state.fields.ux; symmetric=true)
    level_range = _ql_safe_colorrange(minimum(state.level),
                                      maximum(state.level))
    mesh_rows = _ql_mesh_cells_from_result_state(amr.result, state)
    i_probe = _ql_probe_i_from_max_level(state.level)
    profile, analytic = _ql_profile_vectors(amr.result, state)
    y_profile = _ql_profile_axis(length(profile))
    steps = [r.step for r in convergence_rows]
    ux_delta = [r.ux_linf_delta for r in convergence_rows]

    fig = Figure(size=(1900, 1650), fontsize=15)
    Label(fig[0, 1:6], case.name; fontsize=22, tellwidth=false)
    ax1 = _ql_heatmap!(fig, 1, 1, "AMR-D |u| + mesh",
                       state.fields.speed; colormap=:viridis,
                       colorrange=speed_range)
    _ql_overlay_mesh_wireframe!(ax1, mesh_rows; leaf_nx=state.leaf_nx,
                                leaf_ny=state.leaf_ny, alpha=0.92,
                                linewidth=0.82)
    _ql_overlay_vertical_probe!(ax1, i_probe, state.leaf_ny)
    _ql_overlay_solid!(ax1, state.is_solid)
    ax2 = _ql_heatmap!(fig, 1, 3, "AMR-D ux + mesh",
                       state.fields.ux; colormap=:balance,
                       colorrange=ux_range)
    _ql_overlay_mesh_wireframe!(ax2, mesh_rows; leaf_nx=state.leaf_nx,
                                leaf_ny=state.leaf_ny, alpha=0.92,
                                linewidth=0.82)
    _ql_overlay_vertical_probe!(ax2, i_probe, state.leaf_ny)
    _ql_overlay_solid!(ax2, state.is_solid)
    ax3 = _ql_heatmap!(fig, 1, 5, "AMR-D rho + mesh",
                       state.fields.rho; colormap=:magma,
                       colorrange=rho_range)
    _ql_overlay_mesh_wireframe!(ax3, mesh_rows; leaf_nx=state.leaf_nx,
                                leaf_ny=state.leaf_ny, alpha=0.92,
                                linewidth=0.82)
    _ql_overlay_vertical_probe!(ax3, i_probe, state.leaf_ny)
    _ql_overlay_solid!(ax3, state.is_solid)

    ax4 = Axis(fig[2, 1:2]; title="row-mean ux(y), averaged over fluid x",
               xlabel="ux", ylabel="y/Ly")
    _ql_lines_finite!(ax4, profile, y_profile; label="AMR-D",
                      color=:orangered, linewidth=2.5)
    _ql_lines_finite!(ax4, analytic, _ql_profile_axis(length(analytic));
                      label="steady analytic", color=:black,
                      linestyle=:dash, linewidth=2.0)
    (_ql_has_finite_pairs(profile, y_profile) ||
     _ql_has_finite_pairs(analytic, _ql_profile_axis(length(analytic)))) &&
        axislegend(ax4, position=:rb)

    ax5 = Axis(fig[2, 3:4]; title="temporal ux convergence",
               xlabel="steps", ylabel="Linf delta ux")
    _ql_lines_finite!(ax5, steps, ux_delta; color=:dodgerblue4,
                      linewidth=2.4)
    ax6 = _ql_heatmap!(fig, 2, 5, "AMR-D level + mesh",
                       Float64.(state.level); colormap=:plasma,
                       colorrange=level_range)
    _ql_overlay_mesh_wireframe!(ax6, mesh_rows; leaf_nx=state.leaf_nx,
                                leaf_ny=state.leaf_ny, alpha=0.96,
                                linewidth=0.95)
    _ql_overlay_vertical_probe!(ax6, i_probe, state.leaf_ny)
    _ql_overlay_solid!(ax6, state.is_solid)
    save(path, fig)
    return path
end

function _temp_final_dashboard(path, case, amr, reference, convergence_rows)
    if reference !== nothing
        return _ql_plot_debug_dashboard(
            path, amr.result, amr.state, reference.result, reference.state;
            title=case.name, convergence_rows=convergence_rows)
    end
    return _temp_plot_single_dashboard(path, case, amr, convergence_rows)
end

function run_amr_d_temporal_convergence_2d(paths=_temp_case_paths();
        outdir::AbstractString=get(ENV, "KRK_AMR_D_TEMP_OUTDIR",
                                   TEMP_DEFAULT_OUTDIR),
        max_steps::Int=_temp_env_int("KRK_AMR_D_TEMP_MAX_STEPS", 2560),
        ux_rtol::Float64=_temp_env_float("KRK_AMR_D_TEMP_UX_RTOL", 2e-2),
        ux_atol::Float64=_temp_env_float("KRK_AMR_D_TEMP_UX_ATOL", 2e-6),
        rho_atol::Float64=_temp_env_float("KRK_AMR_D_TEMP_RHO_ATOL", 2e-4),
        skip_existing::Bool=_temp_env_bool("KRK_AMR_D_TEMP_SKIP_EXISTING"),
        T::Type{<:AbstractFloat}=Float64)
    mkpath(outdir)
    summary_rows = NamedTuple[]

    for path in paths
        setup = load_kraken(String(path))
        case = conservative_tree_amr_d_case_from_krk_2d(setup)
        case.runtime_supported ||
            throw(ArgumentError("case $(case.name) is not runtime-supported: $(case.reason)"))
        case_dir = joinpath(outdir, _ql_sanitize_name(case.name))
        mkpath(case_dir)
        if skip_existing
            existing = _temp_existing_case_summary(case, case_dir)
            if existing !== nothing
                push!(summary_rows, existing)
                println("skipping ", case.name, " ", existing.status,
                        " steps=", existing.steps)
                flush(stdout)
                continue
            end
        end
        println("running ", case.name)
        flush(stdout)

        start_steps = max(1, Int(getproperty(setup, :max_steps)))
        final_cap = max(start_steps, max_steps)
        steps = start_steps
        previous = nothing
        convergence_rows = NamedTuple[]
        amr = nothing
        final_status = :not_converged

        while true
            amr = _temp_run_case(setup, case; steps=steps,
                                 method=:amr_d, T=T)
            ux_linf = NaN
            ux_l2 = NaN
            rho_linf = NaN
            rho_l2 = NaN
            ux_maxabs = _temp_finite_maxabs(amr.state.fields.ux)
            ux_limit = max(ux_atol, ux_rtol * max(ux_maxabs, eps(Float64)))
            rho_limit = rho_atol
            if previous !== nothing
                ux_linf = _temp_finite_linf_delta(
                    amr.state.fields.ux, previous.state.fields.ux)
                ux_l2 = _temp_finite_l2_delta(
                    amr.state.fields.ux, previous.state.fields.ux)
                rho_linf = _temp_finite_linf_delta(
                    amr.state.fields.rho, previous.state.fields.rho)
                rho_l2 = _temp_finite_l2_delta(
                    amr.state.fields.rho, previous.state.fields.rho)
                if ux_linf <= ux_limit && rho_linf <= rho_limit
                    final_status = :converged
                end
            end
            push!(convergence_rows, (;
                case=case.name, step=steps, status=final_status,
                ux_linf_delta=ux_linf, ux_l2_delta=ux_l2,
                rho_linf_delta=rho_linf, rho_l2_delta=rho_l2,
                ux_delta_limit=ux_limit, rho_delta_limit=rho_limit,
                ux_maxabs=ux_maxabs,
                ux_mean=_ql_finite_mean(amr.state.fields.ux),
                rho_mean=_ql_finite_mean(amr.state.fields.rho),
                mass_rel_drift=_ql_mass_rel_drift(amr.result)))
            final_status == :converged && break
            steps >= final_cap && (final_status = :max_steps_reached; break)
            previous = amr
            steps = min(2 * steps, final_cap)
        end

        if final_status == :max_steps_reached && !isempty(convergence_rows)
            last_row = convergence_rows[end]
            convergence_rows[end] = merge(last_row, (; status=final_status))
        end

        reference_method = _temp_reference_method(case)
        reference = reference_method === nothing ? nothing :
            merge(_temp_run_case(setup, case; steps=steps,
                                 method=reference_method, T=T),
                  (; method=reference_method,))
        amr_record = merge(amr, (; method=:amr_d,))
        value_rows = [_temp_values_row(case, :amr_d, amr.result,
                                       amr.state, reference)]
        reference !== nothing &&
            push!(value_rows, _temp_values_row(case, reference_method,
                                               reference.result,
                                               reference.state, nothing))

        _temp_write_convergence_csv(
            joinpath(case_dir, "convergence.csv"), convergence_rows)
        _temp_write_values_csv(joinpath(case_dir, "values.csv"), value_rows)
        _temp_final_dashboard(
            joinpath(case_dir, "debug_dashboard.png"), case, amr_record,
            reference, convergence_rows)

        push!(summary_rows, (;
            case=case.name, flow=case.flow, status=final_status,
            steps=steps, outdir=case_dir,
            dashboard_png=joinpath(case_dir, "debug_dashboard.png"),
            convergence_csv=joinpath(case_dir, "convergence.csv"),
            values_csv=joinpath(case_dir, "values.csv")))
        println("finished ", case.name, " ", final_status,
                " steps=", steps)
        flush(stdout)
    end

    open(joinpath(outdir, "summary.csv"), "w") do io
        println(io, "case,flow,status,steps,outdir,dashboard_png,convergence_csv,values_csv")
        for r in summary_rows
            println(io, join((r.case, r.flow, r.status, r.steps, r.outdir,
                              r.dashboard_png, r.convergence_csv,
                              r.values_csv), ","))
        end
    end
    return summary_rows
end

function main()
    rows = run_amr_d_temporal_convergence_2d()
    outdir = get(ENV, "KRK_AMR_D_TEMP_OUTDIR", TEMP_DEFAULT_OUTDIR)
    println("wrote ", joinpath(outdir, "summary.csv"))
    for row in rows
        println(row.case, " ", row.status, " steps=", row.steps)
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
