# --- Individual statement parsers ---

function _first_word(line::String)
    m = match(r"^(\w+)", line)
    m === nothing && throw(ArgumentError("Cannot parse line: $line"))
    return m.captures[1]
end

"""Parse: Simulation <name> <lattice>"""
function _parse_simulation(line::String)
    tokens = split(line)
    length(tokens) < 3 && throw(ArgumentError("Simulation needs name and lattice: $line"))
    name = String(tokens[2])
    lattice = Symbol(tokens[3])
    lattice in (:D2Q9, :D3Q19) || throw(ArgumentError("Unknown lattice '$lattice'. Use D2Q9 or D3Q19"))
    return name, lattice
end

"""Parse: Domain L = <Lx> x <Ly> N = <Nx> x <Ny>  (values can be expressions/variables)"""
function _parse_domain(line::String, user_vars::Dict{Symbol,Any}=Dict{Symbol,Any}())
    # Extract L = ... x ... and N = ... x ...  (accept variable names and numbers)
    lm = match(r"L\s*=\s*([\w.eE+-]+)\s*x\s*([\w.eE+-]+)(?:\s*x\s*([\w.eE+-]+))?", line)
    nm = match(r"N\s*=\s*([\w.eE+-]+)\s*x\s*([\w.eE+-]+)(?:\s*x\s*([\w.eE+-]+))?", line)

    lm === nothing && throw(ArgumentError("Cannot parse Domain L = ... : $line"))
    nm === nothing && throw(ArgumentError("Cannot parse Domain N = ... : $line"))

    Lx = _eval_domain_value(lm.captures[1], user_vars)
    Ly = _eval_domain_value(lm.captures[2], user_vars)
    Lz = lm.captures[3] !== nothing ? _eval_domain_value(lm.captures[3], user_vars) : 1.0

    Nx = round(Int, _eval_domain_value(nm.captures[1], user_vars))
    Ny = round(Int, _eval_domain_value(nm.captures[2], user_vars))
    Nz = nm.captures[3] !== nothing ? round(Int, _eval_domain_value(nm.captures[3], user_vars)) : 1

    return DomainSetup(Lx, Ly, Lz, Nx, Ny, Nz)
end

"""Evaluate a domain value: either a number literal or a numeric user variable."""
function _eval_domain_value(s::AbstractString, user_vars::Dict{Symbol,Any})
    val = tryparse(Float64, s)
    val !== nothing && return val
    sym = Symbol(s)
    if haskey(user_vars, sym)
        value = user_vars[sym]
        value isa Real && return Float64(value)
        throw(ArgumentError("Define '$s' is symbolic and cannot be used as a numeric Domain value."))
    end
    throw(ArgumentError("Unknown variable '$s' in Domain. Define it with 'Define $s = ...' or pass as kwarg."))
end

"""Parse: Units { length = mm L_ref = ... R_LU = ... Re = ... scaling = ... }"""
function _parse_units_block(line::String, user_vars::Dict{Symbol,Any})
    brace_m = match(r"\{(.+)\}", line)
    brace_m === nothing && throw(ArgumentError("Missing { ... } in Units block: $line"))
    vals = Dict{Symbol,Any}()
    known = (:length, :L_ref, :R_LU, :Re, :scaling, :L_up, :L_down)
    for m in eachmatch(r"(\w+)\s*=\s*([^\s,{}]+)", brace_m.captures[1])
        key = Symbol(m.captures[1])
        key in known || throw(ArgumentError("Unknown Units key '$key'"))
        raw = strip(String(m.captures[2]))
        if key in (:length, :scaling)
            vals[key] = Symbol(raw)
        elseif key === :R_LU
            vals[key] = round(Int, _eval_units_number(raw, user_vars))
        else
            vals[key] = _eval_units_number(raw, user_vars)
        end
    end
    for key in (:length, :L_ref, :R_LU, :Re)
        haskey(vals, key) || throw(ArgumentError("Units block missing required key '$key'"))
    end
    L_ref = Float64(vals[:L_ref])
    R_LU = Int(vals[:R_LU])
    L_ref > 0 || throw(ArgumentError("Units L_ref must be positive"))
    R_LU > 0 || throw(ArgumentError("Units R_LU must be positive"))
    return UnitsSetup(Symbol(vals[:length]), L_ref, R_LU, Float64(vals[:Re]),
                      Symbol(get(vals, :scaling, :auto)),
                      Float64(get(vals, :L_up, NaN)),
                      Float64(get(vals, :L_down, NaN)))
end

function _eval_units_number(s::AbstractString, user_vars::Dict{Symbol,Any})
    val = tryparse(Float64, s)
    val !== nothing && return val
    sym = Symbol(s)
    if haskey(user_vars, sym)
        value = user_vars[sym]
        value isa Real && return Float64(value)
        throw(ArgumentError("Define '$s' is symbolic and cannot be used as a numeric Units value."))
    end
    return Float64(evaluate(parse_kraken_expr(s, user_vars)))
end

"""Parse: Physics <key> = <value> ...  (values can be expressions with user vars)"""
function _parse_physics(line::String, user_vars::Dict{Symbol,Any}=Dict{Symbol,Any}();
                        allow_auto_nu::Bool=false)
    params = Dict{Symbol, Any}()
    # Match all key = value pairs (value can be a number or an expression)
    for m in eachmatch(r"(\w+)\s*=\s*([\w.eE+\-*/()]+)", line)
        key = Symbol(m.captures[1])
        val_str = strip(String(m.captures[2]))
        allow_auto_nu && key === :nu && val_str == "auto" && continue
        params[key] = _parse_numeric_or_symbolic_value(val_str, user_vars)
    end
    return params
end

"""Parse: Define <VAR> = <expression>"""
function _parse_define(line::String)
    m = match(r"^Define\s+(\w+)\s*=\s*(.+)$", line)
    m === nothing && throw(ArgumentError("Cannot parse Define: $line"))
    key = Symbol(m.captures[1])
    raw = strip(m.captures[2])
    val = tryparse(Float64, raw)
    if val === nothing
        _is_symbolic_bareword(raw) || throw(ArgumentError(
            "Define '$key' must be a numeric literal or symbolic bareword, got '$raw'"))
        val = Symbol(raw)
    end
    return key, val
end

_is_symbolic_bareword(s::AbstractString) = occursin(r"^[A-Za-z_]\w*$", s)

function _parse_numeric_or_symbolic_value(val_str::AbstractString,
                                          user_vars::Dict{Symbol,Any})
    val = tryparse(Float64, val_str)
    val !== nothing && return val

    if _is_symbolic_bareword(val_str)
        sym = Symbol(val_str)
        if haskey(user_vars, sym)
            user_val = user_vars[sym]
            user_val isa Real && return Float64(user_val)
            user_val isa Symbol && return user_val
        end
        return sym
    end

    expr = parse_kraken_expr(val_str, user_vars)
    return Float64(evaluate(expr))
end

"""Parse: Sensitivity { qoi = drag wrt = radius }"""
function _parse_sensitivity(line::String)
    brace_m = match(r"\{(.+)\}", line)
    brace_m === nothing && throw(ArgumentError(
        "Missing { ... } in Sensitivity block: $line"))
    content = strip(String(brace_m.captures[1]))

    vals = Dict{Symbol, Symbol}()
    for m in eachmatch(r"(\w+)\s*=\s*([A-Za-z_]\w*)", content)
        key = Symbol(m.captures[1])
        key in (:qoi, :wrt) || throw(ArgumentError(
            "Unknown Sensitivity key '$key'. Supported keys: qoi, wrt."))
        vals[key] = Symbol(m.captures[2])
    end

    for key in (:qoi, :wrt)
        haskey(vals, key) || throw(ArgumentError(
            "Sensitivity block missing required key '$key'"))
    end
    return (; qoi=vals[:qoi], wrt=vals[:wrt])
end

"""Parse: Obstacle <name> [wall(...)] { <condition> } or stl(..., wall=libb)"""
function _parse_obstacle(line::String, user_vars::Dict{Symbol,Any})
    return _parse_geometry_region(line, :obstacle, user_vars)
end

"""Parse: Fluid <name> { <condition> }"""
function _parse_fluid(line::String, user_vars::Dict{Symbol,Any})
    return _parse_geometry_region(line, :fluid, user_vars)
end

"""
Parse: Refine <name> { region = [x0, y0, x1, y1], ratio = 2, parent = <name> }

Adaptive conservative-tree fields are optional and intentionally explicit:
`criterion = gradient(ux) > 0.02`, `update_every = 50`, `pad = 2`,
`max_growth = 1`, `shrink_margin = 1`, `balance = 1`.
"""
function _parse_refine(line::String, user_vars::Dict{Symbol,Any})
    # Extract name (second word)
    after_kw = strip(replace(line, r"^\w+" => ""))
    name_m = match(r"^(\w+)", after_kw)
    name_m === nothing && throw(ArgumentError("Missing name in Refine: $line"))
    name = String(name_m.captures[1])

    # Extract block content inside { ... }
    brace_m = match(r"\{(.+)\}", line)
    brace_m === nothing && throw(ArgumentError("Missing { ... } block in Refine: $line"))
    content = strip(String(brace_m.captures[1]))

    # Parse region = [x0, y0, x1, y1] (2D) or [x0, y0, z0, x1, y1, z1] (3D)
    region_m = match(r"region\s*=\s*\[([^\]]+)\]", content)
    region_m === nothing && throw(ArgumentError("Missing 'region = [...]' in Refine: $line"))
    coords = [_eval_number(strip(s), user_vars) for s in split(region_m.captures[1], ",")]
    if length(coords) == 4
        region = (coords[1], coords[2], coords[3], coords[4])
        region_3d = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        is_3d = false
    elseif length(coords) == 6
        region = (coords[1], coords[2], coords[4], coords[5])  # x/y projection
        region_3d = (coords[1], coords[2], coords[3], coords[4], coords[5], coords[6])
        is_3d = true
    else
        throw(ArgumentError("Refine region must have 4 (2D) or 6 (3D) values: $line"))
    end

    # Parse ratio (default 2)
    ratio_m = match(r"ratio\s*=\s*(\d+)", content)
    ratio = ratio_m !== nothing ? parse(Int, ratio_m.captures[1]) : 2

    # Parse parent (default "" = base grid)
    parent_m = match(r"parent\s*=\s*(\w+)", content)
    parent = parent_m !== nothing ? String(parent_m.captures[1]) : ""

    criterion = _parse_refine_criterion(content, user_vars)

    return RefineSetup(name, region, region_3d, ratio, parent, is_3d, criterion)
end

function _parse_refine_int_option(content::AbstractString,
                                  key::AbstractString,
                                  default::Int,
                                  user_vars::Dict{Symbol,Any})
    m = match(Regex("\\b" * key * "\\s*=\\s*([\\w.eE+\\-*/()]+)"), content)
    m === nothing && return default
    value = round(Int, _eval_number(strip(m.captures[1]), user_vars))
    return value
end

function _parse_refine_criterion(content::AbstractString,
                                 user_vars::Dict{Symbol,Any})
    occursin(r"\bcriterion\s*=", content) || return nothing
    m = match(r"criterion\s*=\s*(\w+)\((\w+)\)\s*(>=|>)\s*([\w.eE+\-*/()]+)", content)
    m === nothing && throw(ArgumentError(
        "Refine criterion must look like `criterion = gradient(ux) > threshold`"))

    indicator = Symbol(m.captures[1])
    indicator in (:gradient,) || throw(ArgumentError(
        "Unsupported Refine criterion indicator '$indicator'"))
    field = Symbol(m.captures[2])
    op = Symbol(m.captures[3])
    threshold = _eval_number(strip(m.captures[4]), user_vars)
    update_every = _parse_refine_int_option(content, "update_every", 1, user_vars)
    pad = _parse_refine_int_option(content, "pad", 0, user_vars)
    max_growth = _parse_refine_int_option(content, "max_growth", typemax(Int), user_vars)
    shrink_margin = _parse_refine_int_option(content, "shrink_margin", 1, user_vars)
    balance = _parse_refine_int_option(content, "balance", 1, user_vars)

    update_every > 0 || throw(ArgumentError("Refine update_every must be positive"))
    pad >= 0 || throw(ArgumentError("Refine pad must be nonnegative"))
    max_growth >= 0 || throw(ArgumentError("Refine max_growth must be nonnegative"))
    shrink_margin >= 0 || throw(ArgumentError("Refine shrink_margin must be nonnegative"))
    balance == 1 || throw(ArgumentError(
        "Only 2:1 AMR balance is supported in .krk Refine blocks (balance = 1)"))

    return RefineCriterionSetup(indicator, field, op, threshold, update_every,
                                pad, max_growth, shrink_margin, balance)
end

"""Evaluate a number string, substituting user variables."""
function _eval_number(s::AbstractString, user_vars::Dict{Symbol,Any})
    # Try direct parse first
    v = tryparse(Float64, s)
    v !== nothing && return v
    # Try user variable
    sym = Symbol(s)
    if haskey(user_vars, sym)
        value = user_vars[sym]
        value isa Real && return Float64(value)
        throw(ArgumentError("Define '$s' is symbolic and cannot be used as a numeric value"))
    end
    throw(ArgumentError("Cannot evaluate '$s' as a number"))
end

function _parse_geometry_region(line::String, kind::Symbol, user_vars::Dict{Symbol,Any})
    # Extract name (second word after keyword)
    after_kw = strip(replace(line, r"^\w+" => ""))
    name_m = match(r"^(\w+)", after_kw)
    name_m === nothing && throw(ArgumentError("Missing name in: $line"))
    name = String(name_m.captures[1])

    # Check for wall(...) with properties
    bc_type = :wall
    bc_values = Dict{Symbol, KrakenExpr}()
    wall_type_m = match(r"\bwall\s*=\s*(\w+)", line)
    if wall_type_m !== nothing
        wall_type = Symbol(wall_type_m.captures[1])
        wall_type in (:wall, :libb) || throw(ArgumentError(
            "Unknown obstacle wall selector '$wall_type'. Expected wall or libb."))
        bc_type = wall_type
    end
    wall_m = match(r"wall\(([^)]+)\)", line)
    if wall_m !== nothing
        for param_m in eachmatch(r"(\w+)\s*=\s*([^,)]+)", wall_m.captures[1])
            k = Symbol(param_m.captures[1])
            v = strip(String(param_m.captures[2]))
            bc_values[k] = parse_kraken_expr(v, user_vars)
        end
    end

    # Check for STL source: stl(file = "...", scale = ..., ...)
    stl_m = match(r"stl\(([^)]+)\)", line)
    if stl_m !== nothing
        stl_source = _parse_stl_params(stl_m.captures[1])
        return GeometryRegion(name, kind, nothing, stl_source, bc_type, bc_values)
    end

    # Otherwise: condition expression in { ... }
    brace_m = match(r"\{(.+)\}", line)
    brace_m === nothing && throw(ArgumentError("Missing { condition } or stl(...) in: $line"))
    condition_str = strip(String(brace_m.captures[1]))
    condition = parse_kraken_expr(condition_str, user_vars)
    return GeometryRegion(name, kind, condition, nothing, bc_type, bc_values)
end

"""
    _resolve_axisym_face(face) -> String

Alias resolution for `Module axisymmetric` cases. The axisymmetric solver
lives on a 2D `(z, r)` mesh where `z` is the streamwise axis (mapped to the
`x` direction of the underlying D2Q9 grid) and `r` is the radial coordinate
(mapped to `y`). Hence the user-facing aliases:

- `z`     → `x`      (streamwise axis, usually periodic)
- `wall`  → `north`  (outer radial wall at r = R, i.e. y = Ly)
- `axis`  → `south`  (axis of symmetry at r = 0,  i.e. y = 0)
"""
_resolve_axisym_face(face::AbstractString) =
    face == "z"    ? "x"     :
    face == "wall" ? "north" :
    face == "axis" ? "south" : String(face)

"""Parse: Boundary <face> <type>(<params>) or Boundary <axis> periodic"""
function _parse_boundary(line::String, user_vars::Dict{Symbol,Any};
                         is_axisym::Bool=false)
    # Remove keyword
    rest = strip(replace(line, r"^Boundary\s+" => ""))

    # In axisymmetric mode, rewrite the first token using the (z, r) aliases
    # before the generic face/type parsing runs.
    if is_axisym
        tok_m = match(r"^(\S+)(\s.*)?$", rest)
        if tok_m !== nothing
            first_tok = String(tok_m.captures[1])
            resolved = _resolve_axisym_face(first_tok)
            if resolved != first_tok
                tail = tok_m.captures[2] === nothing ? "" : String(tok_m.captures[2])
                rest = resolved * tail
            end
        end
    end

    # Check for axis periodic shorthand: "Boundary x periodic"
    axis_m = match(r"^(x|y|z)\s+periodic$", rest)
    if axis_m !== nothing
        axis = axis_m.captures[1]
        if axis == "x"
            return [BoundarySetup(:west, :periodic, Dict{Symbol,KrakenExpr}()),
                    BoundarySetup(:east, :periodic, Dict{Symbol,KrakenExpr}())]
        elseif axis == "y"
            return [BoundarySetup(:south, :periodic, Dict{Symbol,KrakenExpr}()),
                    BoundarySetup(:north, :periodic, Dict{Symbol,KrakenExpr}())]
        end
    end

    # Parse face name
    face_m = match(r"^(\w+)\s+", rest)
    face_m === nothing && throw(ArgumentError("Cannot parse Boundary face: $line"))
    face = Symbol(face_m.captures[1])
    # Allowed faces:
    #   2D (D2Q9):  west (x=0), east (x=Lx), south (y=0), north (y=Ly)
    #   3D (D3Q19): + bottom (z=0), top (z=Lz); front/back kept as legacy aliases
    # Lattice-specific validation (2D rejects top/bottom) happens post-parse
    # in `_validate_faces_vs_lattice` once the lattice symbol is known.
    face in (:north, :south, :east, :west, :front, :back, :top, :bottom) ||
        throw(ArgumentError("Unknown boundary face '$face'. " *
              "Use north/south/east/west (2D) or +top/bottom (3D)"))

    after_face = strip(rest[face_m.offset + length(face_m.match):end])

    values = Dict{Symbol, KrakenExpr}()

    # `symmetry` is accepted by the parser (emitted on the axis face in
    # axisymmetric cases); the axisymmetric kernel enforces the axis
    # condition internally, so non-axisym runners may treat it as a no-op.
    known_bc_types = (:wall, :velocity, :pressure, :periodic, :outflow,
                      :neumann, :symmetry)

    # Check for type(params) format — find matching parentheses
    type_m = match(r"^(\w+)\(", after_face)
    if type_m !== nothing
        bc_type = Symbol(type_m.captures[1])
        if bc_type ∉ known_bc_types
            sug = _suggest_name(String(type_m.captures[1]), known_bc_types)
            msg = "Unknown boundary type '$bc_type'"
            sug !== nothing && (msg *= " (did you mean: $sug?)")
            throw(ArgumentError(msg))
        end
        paren_start = length(type_m.match)
        params_str = _extract_balanced_parens(after_face, paren_start)
        for kv in _split_params(params_str)
            kv_m = match(r"^\s*(\w+)\s*=\s*(.+)$", kv)
            kv_m === nothing && continue
            k = Symbol(kv_m.captures[1])
            v = strip(String(kv_m.captures[2]))
            values[k] = parse_kraken_expr(v, user_vars)
        end
        # Check for additional params after closing paren (e.g. thermal BC)
        close_idx = paren_start + length(params_str) + 1
        if close_idx < length(after_face)
            extra = strip(after_face[nextind(after_face, close_idx):end])
            _parse_kv_pairs!(values, extra, user_vars)
        end
        return [BoundarySetup(face, bc_type, values)]
    end

    # Simple type without parentheses: "wall" or "wall T = 1.0"
    simple_m = match(r"^(\w+)(.*)", after_face)
    if simple_m !== nothing
        bc_type = Symbol(simple_m.captures[1])
        if bc_type ∉ known_bc_types
            sug = _suggest_name(String(simple_m.captures[1]), known_bc_types)
            msg = "Unknown boundary type '$bc_type'"
            sug !== nothing && (msg *= " (did you mean: $sug?)")
            throw(ArgumentError(msg))
        end
        extra = strip(String(simple_m.captures[2]))
        if !isempty(extra)
            _parse_kv_pairs!(values, extra, user_vars)
        end
        return [BoundarySetup(face, bc_type, values)]
    end

    throw(ArgumentError("Cannot parse Boundary: $line"))
end

"""Extract content between balanced parentheses starting at position `start`."""
function _extract_balanced_parens(s::AbstractString, start::Int)
    depth = 1
    i = nextind(s, start)
    while i <= lastindex(s) && depth > 0
        c = s[i]
        if c == '('
            depth += 1
        elseif c == ')'
            depth -= 1
        end
        depth > 0 && (i = nextind(s, i))
    end
    return s[nextind(s, start):prevind(s, i)]
end

"""Split parameter string by top-level commas (respecting nested parens)."""
function _split_params(s::AbstractString)
    parts = String[]
    current = IOBuffer()
    depth = 0
    for c in s
        if c == '('
            depth += 1
            write(current, c)
        elseif c == ')'
            depth -= 1
            write(current, c)
        elseif c == ',' && depth == 0
            push!(parts, strip(String(take!(current))))
        else
            write(current, c)
        end
    end
    rest = strip(String(take!(current)))
    isempty(rest) || push!(parts, rest)
    return parts
end

"""Parse key = value pairs from a string and add to dict."""
function _parse_kv_pairs!(values::Dict{Symbol,KrakenExpr}, s::AbstractString,
                          user_vars::Dict{Symbol,Any})
    for pm in eachmatch(r"(\w+)\s*=\s*(\S+)", s)
        k = Symbol(pm.captures[1])
        v = strip(String(pm.captures[2]))
        values[k] = parse_kraken_expr(v, user_vars)
    end
end

"""Parse: Initial { field = expr ... }"""
function _parse_initial(line::String, user_vars::Dict{Symbol,Any})
    brace_m = match(r"\{(.+)\}", line)
    brace_m === nothing && throw(ArgumentError("Missing { ... } in Initial: $line"))
    content = strip(String(brace_m.captures[1]))

    fields = Dict{Symbol, KrakenExpr}()
    for m in eachmatch(r"(\w+)\s*=\s*([^=]+?)(?=\s+\w+\s*=|$)", content)
        k = Symbol(m.captures[1])
        v = strip(String(m.captures[2]))
        fields[k] = parse_kraken_expr(v, user_vars)
    end

    return InitialSetup(fields)
end

"""Parse: Module <name>"""
function _parse_module(line::String)
    tokens = split(line)
    length(tokens) < 2 && throw(ArgumentError("Module needs a name: $line"))
    return Symbol(tokens[2])
end

const _KRK_KNOWN_BACKENDS = (:cpu, :cuda, :metal)

"""
    _parse_backend(line::String) -> Symbol

Parse: `Backend <cpu|cuda|metal>`

Selects the compute backend for `run_simulation`. Only the backend *name* is
validated here (fail fast at parse time on a typo); whether the named
backend's package is actually functional (a GPU/driver present) is checked
later, at run time, in `_backend_from_krk_symbol` (`src/simulation_runner.jl`)
— it never falls back to CPU silently.
"""
function _parse_backend(line::String)
    tokens = split(line)
    length(tokens) < 2 && throw(ArgumentError("Backend needs a name: $line"))
    name = Symbol(lowercase(tokens[2]))
    name in _KRK_KNOWN_BACKENDS || throw(ArgumentError(
        "Unknown backend '$(tokens[2])' in .krk 'Backend' directive. " *
        "Supported values: $(join(_KRK_KNOWN_BACKENDS, ", "))."))
    return name
end

"""
Parse: Rheology [phase] <model> { key = value ... }
Examples:
    Rheology power_law { K = 0.1  n = 0.5 }
    Rheology liquid  power_law   { K = 0.1  n = 0.5  nu_min = 1e-6 }
    Rheology gas     newtonian   { nu = 0.01 }
"""
function _parse_rheology(line::String, user_vars::Dict{Symbol,Any}=Dict{Symbol,Any}())
    # Extract brace block if present
    params = Dict{Symbol, Float64}()
    brace_m = match(r"\{(.+)\}", line)
    if brace_m !== nothing
        for m in eachmatch(r"(\w+)\s*=\s*([\w.eE+\-*/()]+)", brace_m.captures[1])
            key = Symbol(m.captures[1])
            val_str = strip(String(m.captures[2]))
            val = tryparse(Float64, val_str)
            if val === nothing
                expr = parse_kraken_expr(val_str, user_vars)
                val = Float64(evaluate(expr))
            end
            params[key] = val
        end
    end

    # Parse tokens before the brace: "Rheology [phase] model"
    pre_brace = brace_m !== nothing ? strip(line[1:brace_m.offset-1]) : strip(line)
    tokens = split(pre_brace)

    known_phases = (:liquid, :gas, :default)
    known_models = (:newtonian, :power_law, :carreau, :cross, :bingham,
                    :herschel_bulkley, :oldroyd_b, :fene_p, :saramito,
                    :giesekus, :ptt)

    if length(tokens) >= 3
        phase = Symbol(tokens[2])
        model = Symbol(tokens[3])
        if phase ∉ known_phases
            # Maybe no phase specified, tokens[2] is the model
            model = Symbol(tokens[2])
            phase = :default
        end
    elseif length(tokens) == 2
        phase = :default
        model = Symbol(tokens[2])
    else
        throw(ArgumentError("Rheology needs at least a model name: $line"))
    end

    model ∉ known_models && throw(ArgumentError("Unknown rheology model '$model'. Known: $known_models"))

    return RheologySetup(phase, model, params)
end
