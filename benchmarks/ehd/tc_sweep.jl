#!/usr/bin/env julia

using Dates
using KernelAbstractions
using KrakeNN
using Printf

function _arg_value(argv, name, default)
    prefix = name * "="
    for arg in argv
        startswith(arg, prefix) && return split(arg, "="; limit=2)[2]
    end
    return default
end

function _parse_grid(s)
    parts = split(lowercase(String(s)), 'x')
    length(parts) == 2 || throw(ArgumentError("grid must be Nx x Ny, e.g. 197x321"))
    return parse(Int, parts[1]), parse(Int, parts[2])
end

function _parse_T_list(s)
    return [parse(Float64, x) for x in split(String(s), ',') if !isempty(strip(x))]
end

function _parse_symbol(s)
    return Symbol(lowercase(String(s)))
end

function _parse_growth_window(s)
    growth_window = parse(Float64, s)
    (0.0 < growth_window <= 1.0) ||
        throw(ArgumentError("--growth-window must be in (0, 1], got $(growth_window)"))
    return growth_window
end

function _cuda_backend()
    # The functional() check lives INSIDE the @eval and the backend is built via
    # invokelatest: referencing CUDA from this function's original world age
    # throws UndefVarError ("binding may be too new").
    try
        @eval Main begin
            using CUDA, CUDSS
            CUDA.functional() || error("CUDA requested but CUDA.functional() is false")
        end
    catch err
        error("CUDA sweep requested but CUDA/CUDSS could not be loaded/functional. Co-install sibling weakdeps as noted in docs/agent/solve-linear-implication.md. Original error: $(sprint(showerror, err))")
    end
    return Base.invokelatest(() -> Main.CUDA.CUDABackend())
end

function _cumulative_log_slope_series(steps, speeds)
    growth = similar(speeds, Float64)
    first_positive = findfirst(>(0.0), speeds)
    if first_positive === nothing
        fill!(growth, 0.0)
        return growth
    end
    s0 = steps[first_positive]
    u0 = max(speeds[first_positive], floatmin(Float64))
    for k in eachindex(speeds)
        if steps[k] == s0
            growth[k] = 0.0
        else
            growth[k] = log(max(speeds[k], floatmin(Float64)) / u0) / (steps[k] - s0)
        end
    end
    return growth
end

function _late_window_log_slope(steps, speeds; growth_window=0.4)
    n = length(speeds)
    n == length(steps) || throw(ArgumentError("steps and speeds must have the same length"))
    n >= 2 || return NaN

    window_n = max(2, ceil(Int, growth_window * n))
    start_idx = max(1, n - window_n + 1)
    x = Float64[]
    y = Float64[]
    for k in start_idx:n
        speed = speeds[k]
        if isfinite(speed) && speed > 0.0 && isfinite(steps[k])
            push!(x, Float64(steps[k]))
            push!(y, log(Float64(speed)))
        end
    end
    # Non-positive samples cannot be logged. If fewer than two valid samples
    # survive, the late-window slope is undefined for this history.
    length(x) >= 2 || return NaN

    xbar = sum(x) / length(x)
    ybar = sum(y) / length(y)
    denom = 0.0
    numer = 0.0
    for k in eachindex(x)
        dx = x[k] - xbar
        denom += dx * dx
        numer += dx * (y[k] - ybar)
    end
    denom > 0.0 || return NaN
    return numer / denom
end

function _write_history(path, steps, speeds; growth_window=0.4)
    cumulative = _cumulative_log_slope_series(steps, speeds)
    growth_late = _late_window_log_slope(steps, speeds; growth_window=growth_window)
    open(path, "w") do io
        println(io, "step,max_abs_u,cumulative_log_slope,growth_rate_late")
        for k in eachindex(steps)
            # The late-window estimator is a final full-history fit, so repeat
            # it on each row; per-row cumulative slope remains the running value.
            @printf(io, "%d,%.12g,%.12g,%.12g\n",
                    steps[k], speeds[k], cumulative[k], growth_late)
        end
    end
    return (cumulative_log_slope=isempty(cumulative) ? 0.0 : cumulative[end],
            growth_rate_late=growth_late)
end

function _write_summary(csv_path, md_path, rows)
    open(csv_path, "w") do io
        println(io, "T,ns_scheme,charge_scheme,phi_scheme,steps,ms_per_step,final_max_abs_u,cumulative_log_slope,growth_rate_late,history_csv")
        for row in rows
            @printf(io, "%.12g,%s,%s,%s,%d,%.12g,%.12g,%.12g,%.12g,%s\n",
                    row.T, row.ns_scheme, row.charge_scheme, row.phi_scheme, row.steps,
                    row.ms_per_step, row.final_max_abs_u, row.cumulative_log_slope,
                    row.growth_rate_late,
                    row.history_csv)
        end
    end
    open(md_path, "w") do io
        println(io, "# EHD Tc Sweep")
        println(io)
        println(io, "| T | NS | charge | phi | steps | ms/step | final max|u| | growth (cumulative log slope) | growth (late, decides Tc) |")
        println(io, "|---:|:---|:---|:---|---:|---:|---:|---:|---:|")
        for row in rows
            @printf(io, "| %.6g | %s | %s | %s | %d | %.6g | %.6g | %.6g | %.6g |\n",
                    row.T, row.ns_scheme, row.charge_scheme, row.phi_scheme, row.steps,
                    row.ms_per_step, row.final_max_abs_u, row.cumulative_log_slope,
                    row.growth_rate_late)
        end
    end
end

function main(argv=ARGS)
    if "--help" in argv || "-h" in argv
        println("usage: julia --project=. benchmarks/ehd/tc_sweep.jl [--smoke] [--gpu] [--grid=197x321] [--T=150,160,163.5,165,170,190] [--cycles=50000] [--ns-scheme=mrt] [--charge-scheme=regularized] [--phi-scheme=lbm|direct] [--phi-substeps=1|auto] [--phi-tol=1e-4] [--phi-max-iter=10000] [--growth-window=0.4]")
        return nothing
    end

    smoke = "--smoke" in argv
    use_gpu = "--gpu" in argv && !smoke

    Nx, Ny = smoke ? (31, 48) : _parse_grid(_arg_value(argv, "--grid", "197x321"))
    T_values = smoke ? [150.0, 220.0] : _parse_T_list(_arg_value(argv, "--T", "150,160,163.5,165,170,190"))
    cycles = parse(Int, _arg_value(argv, "--cycles", smoke ? "500" : "50000"))
    ns_scheme = smoke ? :mrt : _parse_symbol(_arg_value(argv, "--ns-scheme", "mrt"))
    charge_scheme = _parse_symbol(_arg_value(argv, "--charge-scheme", "regularized"))
    phi_scheme = _parse_symbol(_arg_value(argv, "--phi-scheme", "lbm"))
    phi_substeps_arg = _arg_value(argv, "--phi-substeps", "1")
    phi_substeps = lowercase(String(phi_substeps_arg)) == "auto" ? nothing : parse(Int, phi_substeps_arg)
    phi_tol = parse(Float64, _arg_value(argv, "--phi-tol", "1e-4"))
    phi_max_iter = parse(Int, _arg_value(argv, "--phi-max-iter", "10000"))
    history_interval = parse(Int, _arg_value(argv, "--history-interval", smoke ? "10" : "100"))
    growth_window = _parse_growth_window(_arg_value(argv, "--growth-window", "0.4"))
    force_projection = _parse_symbol(_arg_value(argv, "--force-projection", "none"))
    outdir = String(_arg_value(argv, "--output-dir", joinpath(dirname(@__DIR__), "results", "ehd")))
    mkpath(outdir)

    backend = use_gpu ? _cuda_backend() : KernelAbstractions.CPU()
    # Include the process id so independent per-T jobs started in the same
    # second cannot overwrite each other's summary files.
    tag = "$(Dates.format(now(), "yyyymmdd_HHMMSS"))_pid$(getpid())"
    rows = NamedTuple[]
    used_history_names = Set{String}()

    for (case_index, T_ehd) in pairs(T_values)
        result = KrakeNN.run_electroconvection_2d(;
            Nx=Nx, Ny=Ny, T=T_ehd, max_cycles=cycles,
            ns_scheme=ns_scheme, charge_scheme=charge_scheme,
            phi_scheme=phi_scheme, phi_substeps=phi_substeps,
            phi_tol=phi_tol, phi_max_iter=phi_max_iter,
            history_interval=history_interval, force_projection=force_projection,
            backend=backend, FT=Float64)

        T_tag = replace(@sprintf("%.6g", T_ehd), "." => "p", "-" => "m")
        history_name = "tc_sweep_T$(T_tag)_$(ns_scheme)_$(tag).csv"
        if history_name in used_history_names
            history_name = "tc_sweep_T$(T_tag)_$(ns_scheme)_$(tag)_case$(case_index).csv"
        end
        push!(used_history_names, history_name)
        history_path = joinpath(outdir, history_name)
        growth = _write_history(history_path, result.cycle_history, result.umax_history;
                                growth_window=growth_window)
        push!(rows, (T=T_ehd, ns_scheme=String(ns_scheme), charge_scheme=String(charge_scheme),
                     phi_scheme=String(phi_scheme),
                     steps=result.steps, ms_per_step=result.loop_ms_per_step,
                     final_max_abs_u=last(result.umax_history),
                     cumulative_log_slope=growth.cumulative_log_slope,
                     growth_rate_late=growth.growth_rate_late,
                     history_csv=history_name))
        @info "EHD Tc sweep case complete" T=T_ehd ns_scheme steps=result.steps ms_per_step=result.loop_ms_per_step maxu=last(result.umax_history)
    end

    summary_csv = joinpath(outdir, "tc_sweep_summary_$(tag).csv")
    summary_md = joinpath(outdir, "tc_sweep_summary_$(tag).md")
    _write_summary(summary_csv, summary_md, rows)
    println("summary_csv=$(summary_csv)")
    println("summary_md=$(summary_md)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
