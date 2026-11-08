#!/usr/bin/env julia
# Big-sweep cylinder Cd + trace(C) + N1 convergence study for Aqua A100 F64.
#
# Sweeps:  beta x Wi x Re x R x bsd_fraction
# Output:  CSV per case at $KRAKEN_OUTPUT_DIR/cyl_bigsweep_v2_<tag>.csv
#          rolling summary at $KRAKEN_OUTPUT_DIR/SUMMARY.csv
#
# Env-var configurable (defaults match the user-directive 2026-05-18 plan):
#   KRAKEN_BETA_LIST        "0.59"               (Liu 2025/rheoTool match; beta = nu_s/nu_total)
#   KRAKEN_WI_LIST          "0.1,0.3,0.5"
#   KRAKEN_RE_LIST          "0.1,1.0"
#   KRAKEN_R_LIST           "30,50,80"
#   KRAKEN_BSD_LIST         "1.0"                (user directive: use 1.0, not 0.75)
#   KRAKEN_U_MEAN           "0.005"              (controls Re via Re_R = u_mean*R/nu_total)
#   KRAKEN_MAX_STEPS_BASE   "100000"             (scales with R^2 implicitly via dx)
#   KRAKEN_AVG_WINDOW_FRAC  "0.2"                (last 20% averaged for Cd)
#   KRAKEN_BACKEND          "cuda" | "metal" | "cpu"  (auto-selected if not set)
#   KRAKEN_FT               "float64" | "float32"
#   KRAKEN_OUTPUT_DIR       "results/viscoelastic_logfv/cyl_bigsweep_<jobid>"

using KrakeNN
using Printf
using Dates
using KernelAbstractions
using Serialization

function detect_backend()
    req = lowercase(get(ENV, "KRAKEN_BACKEND", "auto"))
    ft  = lowercase(get(ENV, "KRAKEN_FT", "float64"))
    FT  = ft in ("f32", "float32", "single") ? Float32 : Float64

    if (req == "auto" || req == "cuda")
        try
            @eval using CUDA
            cuda_mod = Base.invokelatest(getfield, Main, :CUDA)
            functional = Base.invokelatest(getfield(cuda_mod, :functional))
            if functional
                backend_ctor = Base.invokelatest(getfield, cuda_mod, :CUDABackend)
                return Base.invokelatest(backend_ctor), "cuda", FT
            else
                @warn "CUDA loaded but CUDA.functional() returned false (req=$req); trying next backend."
            end
        catch e
            @warn "CUDA detection failed (req=$req): $(sprint(showerror, e))"
        end
    end
    if (req == "auto" || req == "metal") && Sys.isapple()
        try
            @eval using Metal
            metal_mod = Base.invokelatest(getfield, Main, :Metal)
            functional = Base.invokelatest(getfield(metal_mod, :functional))
            if functional
                backend_ctor = Base.invokelatest(getfield, metal_mod, :MetalBackend)
                return Base.invokelatest(backend_ctor), "metal", FT == Float64 ? Float32 : FT
            else
                @warn "Metal loaded but Metal.functional() returned false (req=$req); trying next backend."
            end
        catch e
            @warn "Metal detection failed (req=$req): $(sprint(showerror, e))"
        end
    end
    if req != "auto" && req != "cpu"
        @warn "Requested KRAKEN_BACKEND=$req but detection failed; falling back to CPU."
    elseif req == "auto"
        @warn "KRAKEN_BACKEND=auto and no GPU backend usable; falling back to CPU."
    end
    return KernelAbstractions.CPU(), "cpu", FT
end

const BACKEND, BACKEND_LABEL, FT = detect_backend()

parse_list(name, default) =
    [parse(Float64, strip(x)) for x in split(get(ENV, name, default), ",")]
parse_int_list(name, default) =
    [parse(Int, strip(x)) for x in split(get(ENV, name, default), ",")]
parse_symbol_list(name, default) =
    [Symbol(strip(x)) for x in split(get(ENV, name, default), ",")]

function parse_bool_token(x)
    s = lowercase(strip(x))
    s in ("1", "true", "t", "yes", "y", "on") && return true
    s in ("0", "false", "f", "no", "n", "off") && return false
    throw(ArgumentError("invalid boolean token: $(x)"))
end

parse_bool_list(name, default) =
    [parse_bool_token(x) for x in split(get(ENV, name, default), ",")]

function zip_equal(name, lists...)
    n = length(first(lists))
    all(l -> length(l) == n, lists) ||
        throw(ArgumentError("$(name) lists must have equal length"))
    return collect(zip(lists...))
end

const BETA_LIST       = parse_list("KRAKEN_BETA_LIST",       "0.59")
const WI_LIST         = parse_list("KRAKEN_WI_LIST",         "0.1,0.3,0.5")
const RE_LIST         = parse_list("KRAKEN_RE_LIST",         "0.1,1.0")
const R_LIST          = parse_int_list("KRAKEN_R_LIST",      "30,50,80")
const BSD_LIST        = parse_list("KRAKEN_BSD_LIST",        "0.0,0.5,1.0")
const L_UP_LIST       = parse_list("KRAKEN_L_UP_LIST",       "15.0")
const L_DOWN_LIST     = parse_list("KRAKEN_L_DOWN_LIST",     "15.0")
const EMBEDDED_GRADIENT_LIST  = parse_bool_list("KRAKEN_EMBEDDED_GRADIENT",  "0")
const EMBEDDED_ADVECTION_LIST = parse_bool_list("KRAKEN_EMBEDDED_ADVECTION", "0")
const EMBEDDED_FORCE_LIST     = parse_bool_list("KRAKEN_EMBEDDED_FORCE",     "0")
const EMBEDDED_DRAG_LIST      = parse_bool_list("KRAKEN_EMBEDDED_DRAG",      "0")
const EMBEDDED_GEOMETRY_LIST  = parse_symbol_list("KRAKEN_EMBEDDED_GEOMETRY", "qwall")
all(g -> g in (:qwall, :circle), EMBEDDED_GEOMETRY_LIST) ||
    throw(ArgumentError("KRAKEN_EMBEDDED_GEOMETRY values must be qwall or circle"))
const ADVECTION_SCHEME = Symbol(lowercase(strip(get(ENV, "KRAKEN_ADVECTION_SCHEME", "rusanov"))))
ADVECTION_SCHEME in (:rusanov, :muscl_superbee, :muscl_superbee_relax) ||
    throw(ArgumentError("KRAKEN_ADVECTION_SCHEME must be rusanov, muscl_superbee, or muscl_superbee_relax"))
# M34: wall BC (halfwayBB default; bouzidi_fl / bouzidi_fl_twopass available)
const WALL_BC = Symbol(lowercase(strip(get(ENV, "KRAKEN_WALL_BC", "halfwayBB"))))
WALL_BC in (:halfwaybb, :halfwayBB, :bouzidi_fl, :bouzidi_fl_twopass) ||
    throw(ArgumentError("KRAKEN_WALL_BC must be halfwayBB, bouzidi_fl, or bouzidi_fl_twopass"))
const WALL_BC_NORMALIZED = WALL_BC === :halfwaybb ? :halfwayBB : WALL_BC
const GEOM_CONFIGS = zip_equal("KRAKEN_L_UP_LIST/KRAKEN_L_DOWN_LIST",
                                L_UP_LIST, L_DOWN_LIST)
const EMBEDDED_CONFIGS = zip_equal("KRAKEN_EMBEDDED_*",
    EMBEDDED_GRADIENT_LIST, EMBEDDED_ADVECTION_LIST, EMBEDDED_FORCE_LIST,
    EMBEDDED_DRAG_LIST, EMBEDDED_GEOMETRY_LIST)
const U_MEAN          = parse(Float64, get(ENV, "KRAKEN_U_MEAN", "0.005"))
const MAX_STEPS_BASE  = parse(Int,     get(ENV, "KRAKEN_MAX_STEPS_BASE", "100000"))
const AVG_WINDOW_FRAC = parse(Float64, get(ENV, "KRAKEN_AVG_WINDOW_FRAC", "0.2"))
const JOB_ID          = get(ENV, "PBS_JOBID", "manual")
const OUTPUT_DIR      = get(ENV, "KRAKEN_OUTPUT_DIR",
                            "results/viscoelastic_logfv/cyl_bigsweep_v2_$(JOB_ID)")
# M29-tau-compare: optional per-case field snapshot (.jls) for field-level
# rheoTool vs KrakeNN diagnostics. Defaults OFF — adds ~5 MB per case at R=30.
const SAVE_FIELDS     = lowercase(get(ENV, "KRAKEN_SAVE_FIELDS", "0")) in
                        ("1", "true", "yes", "on")
const NAN_PROBE        = haskey(ENV, "KRAKEN_NAN_PROBE")
const PROBE_EVERY_N    = parse(Int, get(ENV, "KRAKEN_NAN_PROBE_EVERY", "50"))
const PROBE_BUFFER_LEN = parse(Int, get(ENV, "KRAKEN_NAN_PROBE_BUFFER", "5"))
PROBE_EVERY_N > 0 || throw(ArgumentError("KRAKEN_NAN_PROBE_EVERY must be positive"))
PROBE_BUFFER_LEN >= 0 || throw(ArgumentError("KRAKEN_NAN_PROBE_BUFFER must be non-negative"))

const CSV_COLUMNS = [
    :timestamp, :backend, :FT, :R, :Wi, :Re_R, :beta, :bsd_fraction,
    :L_up, :L_down, :embedded_gradient, :embedded_advection,
    :embedded_force, :embedded_drag, :embedded_geometry,
    :advection_scheme,
    :u_mean, :nu_total, :nu_s, :nu_p, :lambda, :max_steps, :avg_window,
    :polymer_substeps_used, :completed_steps,
    :Cd_kraken, :Cd_s, :Cd_p, :Cd_bsd, :min_det_C, :min_c_eig,
    :u_max_abs, :tau_xx_max_abs, :tau_xy_max_abs, :tau_yy_max_abs,
    :trace_C_max, :trace_C_mean, :N1_max_abs, :N1_mean_abs,
    :first_nonfinite_step, :first_nonfinite_field, :nan_flag,
    :walltime_s, :MLUPS,
]

const NAN_PROBE_COLUMNS = [
    :step,
    :max_abs_psixx, :argmax_i_psixx, :argmax_j_psixx,
    :max_abs_psixy, :argmax_i_psixy, :argmax_j_psixy,
    :max_abs_psiyy, :argmax_i_psiyy, :argmax_j_psiyy,
    :max_tr_tau, :argmax_i_tau, :argmax_j_tau,
    :max_div_u, :argmax_i_div, :argmax_j_div,
]

struct KrakenNanProbeStop <: Exception
    path::String
    step::Int
    field::Symbol
    i::Int
    j::Int
end

Base.showerror(io::IO, err::KrakenNanProbeStop) =
    print(
        io,
        "KrakenNanProbeStop($(err.path), step=$(err.step), field=$(err.field), i=$(err.i), j=$(err.j))",
    )

function csv_cell(x)
    x === nothing && return ""
    if x isa AbstractFloat
        isnan(x) && return "NaN"
        return @sprintf("%.16g", Float64(x))
    end
    s = string(x)
    if any(c in s for c in (',', '"', '\n'))
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function write_nan_probe_header(path)
    open(path, "w") do io
        println(io, join(string.(NAN_PROBE_COLUMNS), ","))
    end
end

function append_nan_probe_row!(path, row)
    open(path, "a") do io
        println(io, join((csv_cell(get(row, c, "")) for c in NAN_PROBE_COLUMNS), ","))
    end
end

function write_csv(path, row; append_summary::Bool=false)
    open(path, "w") do io
        println(io, join(string.(CSV_COLUMNS), ","))
        println(io, join((csv_cell(get(row, c, "")) for c in CSV_COLUMNS), ","))
    end
end

function append_summary!(summary_path, row)
    new_file = !isfile(summary_path)
    open(summary_path, "a") do io
        if new_file
            println(io, join(string.(CSV_COLUMNS), ","))
        end
        println(io, join((csv_cell(get(row, c, "")) for c in CSV_COLUMNS), ","))
    end
end

function max_abs_fluid(field, is_solid)
    found = false; best = 0.0
    for idx in CartesianIndices(is_solid)
        is_solid[idx] && continue
        v = abs(Float64(field[idx]))
        isfinite(v) || return NaN
        best = found ? max(best, v) : v
        found = true
    end
    return found ? best : NaN
end

function max_speed_fluid(ux, uy, is_solid)
    found = false; best = 0.0
    for idx in CartesianIndices(is_solid)
        is_solid[idx] && continue
        v = hypot(Float64(ux[idx]), Float64(uy[idx]))
        isfinite(v) || return NaN
        best = found ? max(best, v) : v
        found = true
    end
    return found ? best : NaN
end

function nan_probe_max_abs_arg(field, is_solid)
    found = false
    best = 0.0
    best_i = 0
    best_j = 0
    @inbounds for j in axes(is_solid, 2), i in axes(is_solid, 1)
        is_solid[i, j] && continue
        v = abs(Float64(field[i, j]))
        isfinite(v) || return (value=NaN, i=0, j=0)
        if !found || v > best
            found = true
            best = v
            best_i = i
            best_j = j
        end
    end
    return found ? (value=best, i=best_i, j=best_j) : (value=NaN, i=0, j=0)
end

function nan_probe_max_trace_tau_arg(tauxx, tauyy, is_solid)
    found = false
    best = -Inf
    best_i = 0
    best_j = 0
    @inbounds for j in axes(is_solid, 2), i in axes(is_solid, 1)
        is_solid[i, j] && continue
        v = Float64(tauxx[i, j]) + Float64(tauyy[i, j])
        isfinite(v) || return (value=NaN, i=0, j=0)
        if !found || v > best
            found = true
            best = v
            best_i = i
            best_j = j
        end
    end
    return found ? (value=best, i=best_i, j=best_j) : (value=NaN, i=0, j=0)
end

function nan_probe_max_div_arg(dudx, dvdy, is_solid)
    found = false
    best = 0.0
    best_i = 0
    best_j = 0
    @inbounds for j in axes(is_solid, 2), i in axes(is_solid, 1)
        is_solid[i, j] && continue
        v = abs(Float64(dudx[i, j]) + Float64(dvdy[i, j]))
        isfinite(v) || return (value=NaN, i=0, j=0)
        if !found || v > best
            found = true
            best = v
            best_i = i
            best_j = j
        end
    end
    return found ? (value=best, i=best_i, j=best_j) : (value=NaN, i=0, j=0)
end

function nan_probe_snapshot(step, fields)
    return (;
        step=Int(step),
        rho=deepcopy(Array(fields.rho)),
        ux=deepcopy(Array(fields.ux)),
        uy=deepcopy(Array(fields.uy)),
        psixx=deepcopy(Array(fields.psixx)),
        psixy=deepcopy(Array(fields.psixy)),
        psiyy=deepcopy(Array(fields.psiyy)),
        tauxx=deepcopy(Array(fields.tauxx)),
        tauxy=deepcopy(Array(fields.tauxy)),
        tauyy=deepcopy(Array(fields.tauyy)),
        fx_poly=deepcopy(Array(fields.fx_poly)),
        fy_poly=deepcopy(Array(fields.fy_poly)),
        fx_total=deepcopy(Array(fields.fx_total)),
        fy_total=deepcopy(Array(fields.fy_total)),
        dudx=deepcopy(Array(fields.dudx)),
        dudy=deepcopy(Array(fields.dudy)),
        dvdx=deepcopy(Array(fields.dvdx)),
        dvdy=deepcopy(Array(fields.dvdy)),
        is_solid=deepcopy(Array(fields.is_solid_h)),
    )
end

function nan_probe_first_nonfinite(fields)
    return KrakeNN._logfv_first_nonfinite_field_2d(
        fields.is_solid_h,
        :rho => fields.rho,
        :ux => fields.ux,
        :uy => fields.uy,
        :psixx => fields.psixx,
        :psixy => fields.psixy,
        :psiyy => fields.psiyy,
        :tauxx => fields.tauxx,
        :tauxy => fields.tauxy,
        :tauyy => fields.tauyy,
        :fx_poly => fields.fx_poly,
        :fy_poly => fields.fy_poly,
        :fx_total => fields.fx_total,
        :fy_total => fields.fy_total,
        :dudx => fields.dudx,
        :dudy => fields.dudy,
        :dvdx => fields.dvdx,
        :dvdy => fields.dvdy,
    )
end

function append_nan_probe_metrics!(path, step, fields)
    is_solid = fields.is_solid_h
    psixx = Array(fields.psixx)
    psixy = Array(fields.psixy)
    psiyy = Array(fields.psiyy)
    tauxx = Array(fields.tauxx)
    tauyy = Array(fields.tauyy)
    dudx = Array(fields.dudx)
    dvdy = Array(fields.dvdy)

    m_xx = nan_probe_max_abs_arg(psixx, is_solid)
    m_xy = nan_probe_max_abs_arg(psixy, is_solid)
    m_yy = nan_probe_max_abs_arg(psiyy, is_solid)
    m_tau = nan_probe_max_trace_tau_arg(tauxx, tauyy, is_solid)
    m_div = nan_probe_max_div_arg(dudx, dvdy, is_solid)

    append_nan_probe_row!(path, Dict{Symbol,Any}(
        :step => Int(step),
        :max_abs_psixx => m_xx.value,
        :argmax_i_psixx => m_xx.i,
        :argmax_j_psixx => m_xx.j,
        :max_abs_psixy => m_xy.value,
        :argmax_i_psixy => m_xy.i,
        :argmax_j_psixy => m_xy.j,
        :max_abs_psiyy => m_yy.value,
        :argmax_i_psiyy => m_yy.i,
        :argmax_j_psiyy => m_yy.j,
        :max_tr_tau => m_tau.value,
        :argmax_i_tau => m_tau.i,
        :argmax_j_tau => m_tau.j,
        :max_div_u => m_div.value,
        :argmax_i_div => m_div.i,
        :argmax_j_div => m_div.j,
    ))
end

function build_nan_probe_callback(case_parameters)
    probe_csv = joinpath(OUTPUT_DIR, "nan_probe.csv")
    event_path = joinpath(OUTPUT_DIR, "nan_event.jls")
    write_nan_probe_header(probe_csv)
    clean_snapshots = NamedTuple[]
    event_ref = Ref{Any}(nothing)

    callback = function (step, fields)
        finite_diag = nan_probe_first_nonfinite(fields)
        if !finite_diag.finite
            event = (;
                first_nonfinite_step=Int(step),
                first_nonfinite_field=finite_diag.field,
                first_nonfinite_i=Int(finite_diag.i),
                first_nonfinite_j=Int(finite_diag.j),
                clean_snapshots=copy(clean_snapshots),
                nan_snapshot=nan_probe_snapshot(step, fields),
                case_parameters,
            )
            serialize(event_path, event)
            event_ref[] = event
            throw(KrakenNanProbeStop(
                event_path, Int(step), finite_diag.field,
                Int(finite_diag.i), Int(finite_diag.j),
            ))
        end

        if step % PROBE_EVERY_N == 0
            append_nan_probe_metrics!(probe_csv, step, fields)
            if PROBE_BUFFER_LEN > 0
                push!(clean_snapshots, nan_probe_snapshot(step, fields))
                while length(clean_snapshots) > PROBE_BUFFER_LEN
                    popfirst!(clean_snapshots)
                end
            end
        end
        return nothing
    end

    return callback, event_ref
end

function c_n1_stats(psixx, psixy, psiyy, tauxx, tauyy, is_solid)
    found = false; bad = false
    trace_max = -Inf; trace_sum = 0.0
    n1_max = 0.0; n1_sum = 0.0; det_min = Inf; n_fluid = 0
    try
        for idx in CartesianIndices(is_solid)
            is_solid[idx] && continue
            cxx, cxy, cyy = KrakeNN.logfv_exp_sym2_2d(
                psixx[idx], psixy[idx], psiyy[idx])
            tr_c = Float64(cxx) + Float64(cyy)
            det_c = Float64(cxx) * Float64(cyy) - Float64(cxy)^2
            n1 = Float64(tauxx[idx]) - Float64(tauyy[idx])
            if !(isfinite(tr_c) && isfinite(det_c) && isfinite(n1))
                bad = true; break
            end
            trace_max = found ? max(trace_max, tr_c) : tr_c
            trace_sum += tr_c
            det_min = found ? min(det_min, det_c) : det_c
            n1_max = found ? max(n1_max, abs(n1)) : abs(n1)
            n1_sum += abs(n1); n_fluid += 1; found = true
        end
    catch err
        err isa DomainError || rethrow()
        bad = true
    end
    bad || !found && return (NaN, NaN, NaN, NaN, NaN, true)
    return (trace_max, trace_sum / n_fluid, n1_max, n1_sum / n_fluid,
            det_min, false)
end

function case_tag(beta, wi, re, R, bsd, L_up, L_down, eg, ea, ef, ed, geom)
    fmt(x) = replace(@sprintf("%.4g", x), "." => "p", "-" => "m")
    b(x) = x ? "1" : "0"
    return "beta$(fmt(beta))_wi$(fmt(wi))_re$(fmt(re))_R$(R)_bsd$(fmt(bsd))" *
           "_Lup$(fmt(L_up))_Ldn$(fmt(L_down))_eg$(b(eg))_ea$(b(ea))" *
           "_ef$(b(ef))_ed$(b(ed))_geom$(geom)"
end

function run_case(beta, wi, re_target, R, bsd, domain_cfg, embedded_cfg, summary_path)
    L_up, L_down = domain_cfg
    embedded_gradient, embedded_advection, embedded_force, embedded_drag,
        embedded_geometry = embedded_cfg
    H = 4 * R
    Nx = ceil(Int, (L_up + L_down) * R)
    Ny = H
    # Re_R = u_mean * R / nu_total  ->  nu_total = u_mean * R / Re_R
    nu_total = U_MEAN * R / re_target
    nu_s = beta * nu_total
    nu_p = (1.0 - beta) * nu_total
    lambda = wi * R / U_MEAN
    max_steps = MAX_STEPS_BASE
    avg_window = max(1, round(Int, max_steps * AVG_WINDOW_FRAC))
    tag = case_tag(beta, wi, re_target, R, bsd, L_up, L_down,
                   embedded_gradient, embedded_advection, embedded_force,
                   embedded_drag, embedded_geometry)
    csv_path = joinpath(OUTPUT_DIR, "cyl_bigsweep_v2_$(tag).csv")
    geom_x_center = Float64(L_up * R)
    geom_y_center = Float64((Ny - 1) / 2)
    max_polymer_substeps = 64
    max_grad_norm_estimate = 4.0 * abs(U_MEAN) / Float64(H)
    subcycle_estimate = KrakeNN.logfv_oldroydb_subcycle_estimate(
        max_grad_norm_estimate,
        Float64(lambda),
        1.0;
        relative_tolerance=0.01,
        max_deformation_increment=0.05,
        max_memory_deformation_increment=0.07,
        min_substeps=1,
        max_substeps=max_polymer_substeps,
    )
    selected_polymer_substeps = subcycle_estimate.recommended
    case_parameters = (;
        tag,
        R=Int(R),
        Wi=Float64(wi),
        Re_R=Float64(re_target),
        beta=Float64(beta),
        bsd_fraction=Float64(bsd),
        L_up=Float64(L_up),
        L_down=Float64(L_down),
        embedded_gradient,
        embedded_advection,
        embedded_force,
        embedded_drag,
        embedded_geometry=string(embedded_geometry),
        advection_scheme=string(ADVECTION_SCHEME),
        Nx=Int(Nx),
        Ny=Int(Ny),
        cylinder_x_lbm=geom_x_center,
        cylinder_y_lbm=geom_y_center,
        radius_lbm=Float64(R),
        u_mean=Float64(U_MEAN),
        nu_total=Float64(nu_total),
        nu_s=Float64(nu_s),
        nu_p=Float64(nu_p),
        lambda=Float64(lambda),
        max_steps=Int(max_steps),
        avg_window=Int(avg_window),
        polymer_substeps_used=Int(selected_polymer_substeps),
        max_polymer_substeps=Int(max_polymer_substeps),
        subcycle_estimate,
        backend=BACKEND_LABEL,
        FT=string(FT),
    )
    step_callback = nothing
    probe_event_ref = Ref{Any}(nothing)
    if NAN_PROBE
        step_callback, probe_event_ref = build_nan_probe_callback(case_parameters)
    end

    row = Dict{Symbol,Any}(
        :timestamp => string(now()),
        :backend => BACKEND_LABEL, :FT => string(FT),
        :R => R, :Wi => wi, :Re_R => re_target, :beta => beta,
        :bsd_fraction => bsd, :u_mean => U_MEAN,
        :L_up => L_up, :L_down => L_down,
        :embedded_gradient => Int(embedded_gradient),
        :embedded_advection => Int(embedded_advection),
        :embedded_force => Int(embedded_force), :embedded_drag => Int(embedded_drag),
        :embedded_geometry => string(embedded_geometry),
        :advection_scheme => string(ADVECTION_SCHEME),
        :nu_total => nu_total, :nu_s => nu_s, :nu_p => nu_p,
        :lambda => lambda, :max_steps => max_steps, :avg_window => avg_window,
    )
    for c in CSV_COLUMNS
        haskey(row, c) || (row[c] = NaN)
    end
    row[:first_nonfinite_field] = "none"
    row[:nan_flag] = true
    row[:polymer_substeps_used] = selected_polymer_substeps

    t0 = time()
    status = :ok
    result = nothing
    try
        result = KrakeNN.run_viscoelastic_logfv_cylinder_coupled_2d(;
            radius=R, H=H, L_up=L_up, L_down=L_down,
            nu_s=FT(nu_s), nu_p=FT(nu_p), lambda=FT(lambda),
            polymer_model=:oldroydb, L_max=FT(10.0),
            u_mean=FT(U_MEAN), Fx_body=FT(0.0),
            bsd_fraction=FT(bsd), polymer_substeps=:auto,
            subcycle_relative_tolerance=FT(0.01),
            max_deformation_increment=FT(0.05),
            max_memory_deformation_increment=FT(0.07),
            max_polymer_substeps=64, max_steps=max_steps,
            avg_window=avg_window, drag_stride=200, diagnostic_stride=0,
            embedded_geometry=embedded_geometry, embedded_gradient=embedded_gradient,
            embedded_advection=embedded_advection, embedded_force=embedded_force,
            embedded_drag=embedded_drag,
            advection_scheme=ADVECTION_SCHEME,
            wall_bc=WALL_BC_NORMALIZED,
            embedded_circle_samples=32, force_boundary_fill=:bc_aware,
            step_callback=step_callback,
            backend=BACKEND, T=FT,
        )
    catch err
        if err isa KrakenNanProbeStop
            status = :nan_probe_complete
            row[:completed_steps] = err.step
            row[:first_nonfinite_step] = err.step
            row[:first_nonfinite_field] = string(err.field)
            row[:nan_flag] = true
        elseif err isa DomainError || err isa BoundsError
            status = :spd_or_bounds_error
        else
            rethrow(err)
        end
    end
    dt_s = time() - t0
    row[:walltime_s] = dt_s

    if status == :ok && result !== nothing
        tr_max, tr_mean, n1_max, n1_mean, det_min, bad = c_n1_stats(
            result.psixx, result.psixy, result.psiyy,
            result.tauxx, result.tauyy, result.is_solid)
        row[:completed_steps] = result.completed_steps
        row[:polymer_substeps_used] = result.polymer_substeps
        row[:Cd_kraken] = Float64(result.Cd)
        row[:Cd_s]      = isdefined(result, :Cd_s) ? Float64(result.Cd_s) : NaN
        row[:Cd_p]      = isdefined(result, :Cd_p) ? Float64(result.Cd_p) : NaN
        row[:Cd_bsd]    = isdefined(result, :Cd_bsd) ? Float64(result.Cd_bsd) : NaN
        row[:min_det_C] = det_min
        row[:min_c_eig] = Float64(result.min_c_eig)
        row[:u_max_abs] = max_speed_fluid(result.ux, result.uy, result.is_solid)
        row[:tau_xx_max_abs] = max_abs_fluid(result.tauxx, result.is_solid)
        row[:tau_xy_max_abs] = max_abs_fluid(result.tauxy, result.is_solid)
        row[:tau_yy_max_abs] = max_abs_fluid(result.tauyy, result.is_solid)
        row[:trace_C_max]  = tr_max
        row[:trace_C_mean] = tr_mean
        row[:N1_max_abs]   = n1_max
        row[:N1_mean_abs]  = n1_mean
        row[:first_nonfinite_step]  = result.first_nonfinite_step
        row[:first_nonfinite_field] = string(result.first_nonfinite_field)
        row[:nan_flag] = bad || result.first_nonfinite_step > 0 ||
            !isfinite(Float64(result.Cd))
        row[:MLUPS] = dt_s > 0 ?
            Float64(Nx) * Float64(Ny) * Float64(result.completed_steps) /
            dt_s / 1e6 : NaN
    elseif status == :nan_probe_complete
        event = probe_event_ref[]
        if event !== nothing
            row[:completed_steps] = event.first_nonfinite_step
            row[:first_nonfinite_step] = event.first_nonfinite_step
            row[:first_nonfinite_field] = string(event.first_nonfinite_field)
        end
        row[:MLUPS] = dt_s > 0 && isfinite(Float64(row[:completed_steps])) ?
            Float64(Nx) * Float64(Ny) * Float64(row[:completed_steps]) /
            dt_s / 1e6 : NaN
    else
        row[:first_nonfinite_step] = 0
        row[:first_nonfinite_field] = "spd_or_bounds"
    end

    write_csv(csv_path, row)
    append_summary!(summary_path, row)

    if SAVE_FIELDS && status == :ok && result !== nothing
        jls_path = joinpath(OUTPUT_DIR, "cyl_bigsweep_v2_$(tag)_fields.jls")
        try
            serialize(jls_path, (;
                tag,
                # case parameters
                R, Wi=wi, Re=re_target, beta, bsd_fraction=bsd,
                L_up, L_down, embedded_gradient, embedded_advection,
                embedded_force, embedded_drag,
                embedded_geometry=string(embedded_geometry),
                advection_scheme=string(ADVECTION_SCHEME),
                u_mean=U_MEAN, nu_total, nu_s, nu_p, lambda,
                # grid (cells are 1..Nx × 1..Ny; LBM node centers at (i, j))
                Nx, Ny, dx=1.0, dy=1.0,
                cylinder_x_lbm=geom_x_center,
                cylinder_y_lbm=geom_y_center,
                radius_lbm=Float64(R),
                # Cd outputs (already in row but redundant for self-contained file)
                Cd_kraken=Float64(get(row, :Cd_kraken, NaN)),
                Cd_s=Float64(get(row, :Cd_s, NaN)),
                Cd_p=Float64(get(row, :Cd_p, NaN)),
                Cd_bsd=Float64(get(row, :Cd_bsd, NaN)),
                # fields (Array, CPU-side already by driver return)
                ux=Array{Float64}(result.ux),
                uy=Array{Float64}(result.uy),
                tauxx=Array{Float64}(result.tauxx),
                tauxy=Array{Float64}(result.tauxy),
                tauyy=Array{Float64}(result.tauyy),
                is_solid=Array{Bool}(result.is_solid),
                # M30 schema extension (2026-05-20): persist LBM density rho
                # so wall-pressure azimuthal profile p(theta) = c_s^2 * rho(theta)
                # is reconstructible from snapshots without re-running.
                # Backward-compatible (Serialization NamedTuple): older
                # analysis scripts that don't ask for :rho ignore this field.
                rho=Array{Float64}(result.rho),
            ))
            @printf("  saved fields to %s (%.1f MB)\n", jls_path,
                    filesize(jls_path) / 1024^2)
        catch err
            @warn "field snapshot failed" tag exception=err
        end
    end

    @printf("[%s] beta=%.2f Wi=%.2f Re=%.2f R=%d bsd=%.2f: Cd=%s min_detC=%s tr_C_max=%s N1_mean=%s dt=%.0fs nan=%s\n",
        string(now()),
        beta, wi, re_target, R, bsd,
        isfinite(Float64(row[:Cd_kraken])) ? @sprintf("%.4f", row[:Cd_kraken]) : "NaN",
        isfinite(Float64(row[:min_det_C]))  ? @sprintf("%.3g", row[:min_det_C]) : "NaN",
        isfinite(Float64(row[:trace_C_max])) ? @sprintf("%.3g", row[:trace_C_max]) : "NaN",
        isfinite(Float64(row[:N1_mean_abs])) ? @sprintf("%.3g", row[:N1_mean_abs]) : "NaN",
        dt_s, string(row[:nan_flag]))
    flush(stdout)
    return row
end

function main()
    println("=== cyl_bigsweep_v2 backend=$(BACKEND_LABEL) FT=$(FT) ===")
    println("beta=$BETA_LIST | Wi=$WI_LIST | Re=$RE_LIST | R=$R_LIST | bsd=$BSD_LIST")
    println("domains=$GEOM_CONFIGS | embedded=$EMBEDDED_CONFIGS")
    println("output_dir=$OUTPUT_DIR")
    mkpath(OUTPUT_DIR)
    summary_path = joinpath(OUTPUT_DIR, "SUMMARY.csv")
    n_total = length(BETA_LIST) * length(WI_LIST) * length(RE_LIST) *
              length(R_LIST) * length(BSD_LIST) *
              length(GEOM_CONFIGS) * length(EMBEDDED_CONFIGS)
    println("$n_total cases total")
    flush(stdout)
    n_done = 0
    for beta in BETA_LIST, wi in WI_LIST, re in RE_LIST,
        R in R_LIST, bsd in BSD_LIST, domain_cfg in GEOM_CONFIGS,
        embedded_cfg in EMBEDDED_CONFIGS
        n_done += 1
        @printf("\n[%d/%d] starting case at %s\n", n_done, n_total, string(now()))
        flush(stdout)
        run_case(beta, wi, re, R, bsd, domain_cfg, embedded_cfg, summary_path)
    end
    println("\n=== cyl_bigsweep_v2 DONE at $(now()) ===")
end

main()
