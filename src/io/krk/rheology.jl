"""
    _warn_ignored_thermal_keys(rs::RheologySetup)

Emit one `@warn` naming the thermal keys (`E_a`, `C1`, `C2`, `T_ref`) of a
`Rheology` block that an isothermal run silently ignores: only the thermal
drivers evaluate the temperature-dependent viscosity. No behaviour change.
"""
function _warn_ignored_thermal_keys(rs::RheologySetup)
    ignored = [k for k in (:E_a, :C1, :C2, :T_ref) if haskey(rs.params, k)]
    isempty(ignored) && return nothing
    @warn "Rheology $(rs.model): keys $(join(ignored, ", ")) are ignored in an isothermal run (no thermal module); the viscosity is not temperature-dependent." ignored_keys=ignored
    return nothing
end

"""
    build_rheology_model(rs::RheologySetup; T=Float64) → AbstractRheology

Instantiate a concrete rheology model from a parsed `RheologySetup`.
"""
function build_rheology_model(rs::RheologySetup; FT=Float64)
    p = rs.params
    g = (k, default) -> FT(get(p, k, default))

    # Thermal coupling
    thermal = if haskey(p, :E_a)
        ArrheniusCoupling(g(:T_ref, 1.0), g(:E_a, 0.0))
    elseif haskey(p, :C1)
        WLFCoupling(g(:T_ref, 1.0), g(:C1, 8.86), g(:C2, 101.6))
    else
        IsothermalCoupling()
    end
    _check_arrhenius_temperatures(thermal; context=".krk Rheology T_ref")

    if rs.model == :newtonian
        return Newtonian(g(:nu, 0.1); thermal=thermal)
    elseif rs.model == :power_law
        return PowerLaw(g(:K, 0.1), g(:n, 1.0);
                        nu_min=g(:nu_min, 1e-6), nu_max=g(:nu_max, 10.0), thermal=thermal)
    elseif rs.model == :carreau
        return CarreauYasuda(g(:eta_0, 1.0), g(:eta_inf, 0.01), g(:lambda, 1.0),
                             g(:a, 2.0), g(:n, 0.5); thermal=thermal)
    elseif rs.model == :cross
        return Cross(g(:eta_0, 1.0), g(:eta_inf, 0.01), g(:K, 1.0), g(:m, 1.0);
                     thermal=thermal)
    elseif rs.model == :bingham
        return Bingham(g(:tau_y, 0.1), g(:mu_p, 0.05);
                       m_reg=g(:m_reg, 1000.0), thermal=thermal)
    elseif rs.model == :herschel_bulkley
        return HerschelBulkley(g(:tau_y, 0.1), g(:K, 0.1), g(:n, 0.5);
                               m_reg=g(:m_reg, 1000.0), thermal=thermal)
    elseif rs.model == :oldroyd_b
        form = haskey(p, :formulation) && p[:formulation] == 0.0 ? StressFormulation() : LogConfFormulation()
        return OldroydB(g(:nu_s, 0.1), g(:nu_p, 0.05), g(:lambda, 1.0);
                        formulation=form, thermal=thermal)
    elseif rs.model == :fene_p
        form = haskey(p, :formulation) && p[:formulation] == 0.0 ? StressFormulation() : LogConfFormulation()
        return FENEP(g(:nu_s, 0.1), g(:nu_p, 0.05), g(:lambda, 1.0), g(:L_max, 100.0);
                     formulation=form, thermal=thermal)
    elseif rs.model == :saramito
        form = haskey(p, :formulation) && p[:formulation] == 0.0 ? StressFormulation() : LogConfFormulation()
        return Saramito(g(:nu_s, 0.1), g(:nu_p, 0.05), g(:lambda, 1.0), g(:tau_y, 0.01);
                        n=g(:n, 1.0), m_reg=g(:m_reg, 1000.0),
                        formulation=form, thermal=thermal)
    else
        throw(ArgumentError("Unimplemented rheology model: $(rs.model)"))
    end
end

"""Parse: Run <N> steps"""
function _parse_run(line::String)
    m = match(r"(\d+)", line)
    m === nothing && throw(ArgumentError("Cannot parse Run: $line"))
    return parse(Int, m.captures[1])
end

"""Parse: Output <format> every <N> [field1, field2, ...] [fps=<N>]"""
function _parse_output(line::String)
    # Format
    fmt_m = match(r"^Output\s+(\w+)", line)
    fmt_m === nothing && throw(ArgumentError("Cannot parse Output format: $line"))
    format = Symbol(fmt_m.captures[1])
    format in (:vtk, :png, :gif) || throw(ArgumentError(
        "Unknown Output format '$format'. Use vtk, png, or gif."))

    # Interval
    int_m = match(r"every\s+(\d+)", line)
    int_m === nothing && throw(ArgumentError("Cannot parse Output interval: $line"))
    interval = parse(Int, int_m.captures[1])

    # Fields: [field1, field2, ...]
    fields_m = match(r"\[([^\]]+)\]", line)
    fields = Symbol[]
    if fields_m !== nothing
        for f in split(fields_m.captures[1], r"[,\s]+")
            s = strip(f)
            isempty(s) || push!(fields, Symbol(s))
        end
    end

    # fps parameter (optional, for gif format)
    fps = 10
    fps_m = match(r"fps\s*=\s*(\d+)", line)
    if fps_m !== nothing
        fps = parse(Int, fps_m.captures[1])
    end

    # Directory (optional, default "output/")
    directory = "output/"

    return OutputSetup(format, interval, fields, directory, fps)
end

"""Parse: Diagnostics every <N> [col1, col2, ...]"""
function _parse_diagnostics(line::String)
    int_m = match(r"every\s+(\d+)", line)
    int_m === nothing && throw(ArgumentError("Cannot parse Diagnostics interval: $line"))
    interval = parse(Int, int_m.captures[1])

    fields_m = match(r"\[([^\]]+)\]", line)
    columns = Symbol[]
    if fields_m !== nothing
        for f in split(fields_m.captures[1], r"[,\s]+")
            s = strip(f)
            isempty(s) || push!(columns, Symbol(s))
        end
    end

    return DiagnosticsSetup(interval, columns)
end

"""Parse STL parameters from stl(file = "...", scale = ..., translate = [...], z_slice = ...)"""
function _parse_stl_params(params_str::AbstractString)
    # Extract file path (quoted string)
    file_m = match(r"""file\s*=\s*"([^"]+)""", params_str)
    file_m === nothing && throw(ArgumentError(
        "STL source requires file parameter: stl(file = \"path.stl\")"))
    file = String(file_m.captures[1])

    # Optional: scale
    scale = 1.0
    scale_m = match(r"scale\s*=\s*([\d.eE+-]+)", params_str)
    scale_m !== nothing && (scale = parse(Float64, scale_m.captures[1]))

    # Optional: translate = [x, y, z]
    translate = (0.0, 0.0, 0.0)
    tr_m = match(r"translate\s*=\s*\[\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\s*\]", params_str)
    if tr_m !== nothing
        translate = (parse(Float64, tr_m.captures[1]),
                     parse(Float64, tr_m.captures[2]),
                     parse(Float64, tr_m.captures[3]))
    end

    # Optional: z_slice (for 2D cross-section)
    z_slice = 0.0
    zs_m = match(r"z_slice\s*=\s*([\d.eE+-]+)", params_str)
    zs_m !== nothing && (z_slice = parse(Float64, zs_m.captures[1]))

    return STLSource(file, scale, translate, z_slice)
end

