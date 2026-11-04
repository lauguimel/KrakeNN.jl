# --- Parser for .krk simulation config files ---

"""
    DomainSetup

Domain geometry: physical extents and grid resolution.
"""
struct DomainSetup
    Lx::Float64
    Ly::Float64
    Lz::Float64
    Nx::Int
    Ny::Int
    Nz::Int
end

"""
    PhysicsSetup

Physical parameters (nu, Pr, Ra, etc.) and optional body force expressions.
"""
struct PhysicsSetup
    params::Dict{Symbol, Any}
    body_force::Dict{Symbol, KrakenExpr}  # :Fx, :Fy, :Fz

    function PhysicsSetup(params, body_force::Dict{Symbol, KrakenExpr})
        return new(Dict{Symbol, Any}(params), body_force)
    end
end

"""
    UnitsSetup

Physical-to-LU bridge parsed from a `.krk` `Units { ... }` block.
`L_ref` and `R_LU` are the same characteristic length; the STL example uses
the cylinder radius, so `dx_real = L_ref / R_LU` has no hidden factor of two.
"""
struct UnitsSetup
    length::Symbol
    L_ref::Float64
    R_LU::Int
    Re::Float64
    scaling::Symbol
    L_up::Float64
    L_down::Float64
end

"""
    STLSource

Reference to an STL file with optional transform parameters.
"""
struct STLSource
    file::String
    scale::Float64
    translate::NTuple{3, Float64}
    z_slice::Float64   # z-plane for 2D cross-section (default 0.0)
end

"""
    GeometryRegion

A geometry region defining solid obstacles or fluid zones.
Either via a condition expression OR an STL file (mutually exclusive).
"""
struct GeometryRegion
    name::String
    kind::Symbol                        # :obstacle or :fluid
    condition::Union{KrakenExpr, Nothing}  # (x, y [,z]) -> Bool (nothing if STL)
    stl::Union{STLSource, Nothing}         # STL file source (nothing if condition)
    bc_type::Symbol                     # :wall (default)
    bc_values::Dict{Symbol, KrakenExpr}
end

"""
    BoundarySetup

Boundary condition for one face of the domain.
"""
struct BoundarySetup
    face::Symbol          # :north, :south, :east, :west (2D)
    type::Symbol          # :wall, :velocity, :pressure, :periodic
    values::Dict{Symbol, KrakenExpr}
end

"""
    InitialSetup

Initial condition expressions for fields.
"""
struct InitialSetup
    fields::Dict{Symbol, KrakenExpr}
end

"""
    OutputSetup

Output configuration: format, interval, and field names.
"""
struct OutputSetup
    format::Symbol
    interval::Int
    fields::Vector{Symbol}
    directory::String
    fps::Int             # frames per second (used by :gif format, default 10)
end

# Convenience constructor without fps (backward compat)
OutputSetup(format, interval, fields, directory) =
    OutputSetup(format, interval, fields, directory, 10)

"""
    DiagnosticsSetup

Diagnostics logging configuration.
"""
struct DiagnosticsSetup
    interval::Int
    columns::Vector{Symbol}
end

"""
    RefineSetup

Refinement patch specification from a .krk `Refine` block.
"""
struct RefineCriterionSetup
    indicator::Symbol              # :gradient, currently production-facing
    field::Symbol                  # :ux, :uy, :rho, ...
    op::Symbol                     # :> or :>=
    threshold::Float64
    update_every::Int
    pad::Int
    max_growth::Int
    shrink_margin::Int
    balance::Int                   # 1 means adjacent AMR leaves differ by <= 1 level
end

"""
    RefineSetup

Public type or module in the .krk parsing and I/O API.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using Kraken

Kraken.RefineSetup
```
"""
struct RefineSetup
    name::String
    region::NTuple{4, Float64}   # 2D: (x_min, y_min, x_max, y_max)
    region_3d::NTuple{6, Float64}  # 3D: (x_min, y_min, z_min, x_max, y_max, z_max); zeros for 2D
    ratio::Int                   # refinement ratio (default 2)
    parent::String               # parent patch name ("" = base grid)
    is_3d::Bool                  # true when 6 region coords provided
    criterion::Union{RefineCriterionSetup, Nothing}
end

RefineSetup(name, region, region_3d, ratio, parent, is_3d) =
    RefineSetup(name, region, region_3d, ratio, parent, is_3d, nothing)

"""
    RheologySetup

Rheology model specification for a single phase.
The `model` symbol selects the type (`:newtonian`, `:power_law`, `:carreau`,
`:cross`, `:bingham`, `:herschel_bulkley`, `:oldroyd_b`, `:fene_p`, `:saramito`).
The `params` dict holds the model parameters (e.g., `K`, `n`, `nu_min`, `nu_max`).
"""
struct RheologySetup
    phase::Symbol                    # :liquid, :gas, or :default
    model::Symbol                    # :newtonian, :power_law, :carreau, etc.
    params::Dict{Symbol, Float64}
end

"""
    SimulationSetup

Top-level simulation configuration parsed from a .krk file.
"""
struct SimulationSetup
    name::String
    lattice::Symbol
    domain::DomainSetup
    physics::PhysicsSetup
    user_vars::Dict{Symbol, Any}
    regions::Vector{GeometryRegion}
    boundaries::Vector{BoundarySetup}
    initial::Union{InitialSetup, Nothing}
    modules::Vector{Symbol}
    max_steps::Int
    outputs::Vector{OutputSetup}
    diagnostics::Union{DiagnosticsSetup, Nothing}
    refinements::Vector{RefineSetup}
    velocity_field::Union{InitialSetup, Nothing}  # prescribed velocity expressions (ux, uy)
    rheology::Vector{RheologySetup}                # per-phase rheology models
    mesh::Any                                      # Mesh-directive descriptor (body-fitted / Gmsh); `nothing` for the Cartesian path. Consumed by run_simulation (`setup.mesh !== nothing`) + _run_gmsh_slbm_drag. The producing parser is not built yet (KRK-GEO).
    units::Union{UnitsSetup, Nothing}              # Parse-time physical-units descriptor. Runner ignores it; fields above are already raw LU.
    collision::Symbol                              # Generic 2D collision selector; default preserves the historical BGK path.
    wall_bc::Symbol                                # Generic 2D wall selector; default preserves historical halfway bounce-back.
    sensitivity::Union{Nothing,NamedTuple}         # Optional AD request parsed from `Sensitivity { qoi = ..., wrt = ... }`.
    backend::Union{Nothing,Symbol}                 # Optional `Backend <cpu|cuda|metal>` directive; `nothing` = file does not select a backend (caller/default decides). See `_backend_from_krk_symbol` in simulation_runner.jl.

    function SimulationSetup(name, lattice, domain, physics, user_vars, regions,
                             boundaries, initial, modules, max_steps, outputs,
                             diagnostics, refinements, velocity_field, rheology,
                             mesh, units, collision, wall_bc, sensitivity=nothing,
                             backend=nothing)
        return new(name, lattice, domain, physics, Dict{Symbol, Any}(user_vars),
                   regions, boundaries, initial, modules, max_steps, outputs,
                   diagnostics, refinements, velocity_field, rheology, mesh,
                   units, Symbol(collision), Symbol(wall_bc), sensitivity, backend)
    end
end

# Backward-compatible constructors: `mesh`, `units`, `collision`, and `wall_bc`
# default to the historical Cartesian 2D path, so existing 15/16/17-argument
# call sites keep working unchanged.
SimulationSetup(name, lattice, domain, physics, user_vars, regions, boundaries,
                initial, modules, max_steps, outputs, diagnostics, refinements,
                velocity_field, rheology) =
    SimulationSetup(name, lattice, domain, physics, user_vars, regions, boundaries,
                    initial, modules, max_steps, outputs, diagnostics, refinements,
                    velocity_field, rheology, nothing, nothing, :bgk, :halfwaybb)

SimulationSetup(name, lattice, domain, physics, user_vars, regions, boundaries,
                initial, modules, max_steps, outputs, diagnostics, refinements,
                velocity_field, rheology, mesh) =
    SimulationSetup(name, lattice, domain, physics, user_vars, regions, boundaries,
                    initial, modules, max_steps, outputs, diagnostics, refinements,
                    velocity_field, rheology, mesh, nothing, :bgk, :halfwaybb)

SimulationSetup(name, lattice, domain, physics, user_vars, regions, boundaries,
                initial, modules, max_steps, outputs, diagnostics, refinements,
                velocity_field, rheology, mesh, units) =
    SimulationSetup(name, lattice, domain, physics, user_vars, regions, boundaries,
                    initial, modules, max_steps, outputs, diagnostics, refinements,
                    velocity_field, rheology, mesh, units, :bgk, :halfwaybb)

function _numeric_user_vars(user_vars::Dict{Symbol, Any})
    out = Dict{Symbol, Float64}()
    for (key, value) in user_vars
        value isa Real && (out[key] = Float64(value))
    end
    return out
end

parse_kraken_expr(source::AbstractString, user_vars::Dict{Symbol, Any}) =
    parse_kraken_expr(source, _numeric_user_vars(user_vars))

_canonical_knob_symbol(value::Symbol) = Symbol(lowercase(String(value)))

function _physics_symbol_option(params::Dict{Symbol, Any}, key::Symbol,
                                default::Symbol)
    value = get(params, key, default)
    value isa Symbol || throw(ArgumentError(
        "Physics $key must be a symbolic token, got '$value'"))
    return _canonical_knob_symbol(value)
end

# --- Tokenization: strip comments, join multi-line blocks ---

"""
    _preprocess_lines(text::String) -> Vector{String}

Strip comments, remove blank lines, and join multi-line `{ ... }` blocks
into single statements.
"""
function _preprocess_lines(text::String)
    lines = String[]
    raw_lines = split(text, '\n')

    buffer = ""
    brace_depth = 0

    for raw in raw_lines
        # Strip comments (but not inside strings — simple heuristic)
        stripped = _strip_comment(String(raw))
        trimmed = strip(stripped)
        isempty(trimmed) && brace_depth == 0 && continue

        # Count braces
        opens = count(==( '{'), trimmed)
        closes = count(==('}'), trimmed)

        if brace_depth == 0 && opens == 0
            # Simple single-line statement
            push!(lines, String(trimmed))
        else
            # Accumulate multi-line block
            buffer *= " " * String(trimmed)
            brace_depth += opens - closes
            if brace_depth <= 0
                push!(lines, strip(buffer))
                buffer = ""
                brace_depth = 0
            end
        end
    end

    if brace_depth > 0
        throw(ArgumentError("Unclosed brace in .krk file"))
    end

    return lines
end

function _strip_comment(line::String)
    # Simple: strip everything after # that's not preceded by backslash
    idx = findfirst('#', line)
    idx === nothing && return line
    return line[1:prevind(line, idx)]
end

# --- Main parser ---

"""
    load_kraken(filename::String; kwargs...) -> SimulationSetup

Parse a `.krk` file and return a `SimulationSetup` struct.
Keyword arguments override `Define` defaults for parametric studies.

# Example
```julia
setup = load_kraken("examples/cavity.krk")
setup = load_kraken("examples/cavity.krk"; Re=400, N=256)
```
"""
function load_kraken(filename::String; kwargs...)
    text = read(filename, String)
    return parse_kraken(text; kwargs...)
end

"""
    parse_kraken(text::String; kwargs...) -> SimulationSetup

Parse .krk format text into a SimulationSetup struct.
Keyword arguments override `Define` defaults for parametric studies.

# Example
```julia
setup = parse_kraken(text; Re=400, N=256)
```
"""
function parse_kraken(text::String; kwargs...)
    setups = _parse_kraken_internal(text; kwargs...)
    length(setups) == 1 || throw(ArgumentError(
        "parse_kraken: got $(length(setups)) setups (Sweep directive present?). " *
        "Use parse_kraken_sweep for sweeps."))
    return setups[1]
end

"""
    parse_kraken_sweep(text::String; kwargs...) -> Vector{SimulationSetup}

Parse a .krk file containing zero or more `Sweep param = [a, b, c]` directives
and return one `SimulationSetup` per combination of sweep values. If no sweep
is present, returns a single-element vector.
"""
function parse_kraken_sweep(text::String; kwargs...)
    return _parse_kraken_internal(text; kwargs...)
end

"""
    load_kraken_sweep(filename::String; kwargs...) -> Vector{SimulationSetup}

File version of [`parse_kraken_sweep`](@ref).
"""
function load_kraken_sweep(filename::String; kwargs...)
    return parse_kraken_sweep(read(filename, String); kwargs...)
end

function _parse_kraken_internal(text::String; kwargs...)
    lines = _preprocess_lines(text)

    # --- Pre-pass: Preset expansion ---
    expanded = String[]
    for line in lines
        if _first_word(line) == "Preset"
            append!(expanded, _expand_preset(line))
        else
            push!(expanded, line)
        end
    end
    lines = expanded

    # --- Sweep pre-pass: collect sweeps and expand combinations ---
    sweeps = Pair{Symbol, Vector{Float64}}[]
    non_sweep_lines = String[]
    for line in lines
        if _first_word(line) == "Sweep"
            push!(sweeps, _parse_sweep(line))
        else
            push!(non_sweep_lines, line)
        end
    end
    lines = non_sweep_lines

    if !isempty(sweeps)
        setups = SimulationSetup[]
        # Cartesian product
        counters = ones(Int, length(sweeps))
        sizes = [length(s.second) for s in sweeps]
        while true
            sweep_kwargs = Dict{Symbol, Any}(kwargs)
            for (i, sw) in enumerate(sweeps)
                sweep_kwargs[sw.first] = sw.second[counters[i]]
            end
            append!(setups, _parse_kraken_internal_single(lines; sweep_kwargs...))
            # advance
            k = length(counters)
            while k > 0
                counters[k] += 1
                if counters[k] > sizes[k]
                    counters[k] = 1
                    k -= 1
                else
                    break
                end
            end
            k == 0 && break
        end
        return setups
    end

    return _parse_kraken_internal_single(lines; kwargs...)
end

function _parse_kraken_internal_single(lines::Vector{String}; kwargs...)
    # --- First pass: collect Define defaults ---
    user_vars = Dict{Symbol, Any}()
    for line in lines
        _first_word(line) == "Define" || continue
        k, v = _parse_define(line)
        user_vars[k] = v
    end

    # Override with kwargs (highest priority)
    for (k, v) in kwargs
        user_vars[k] = Float64(v)
    end

    units_setup = nothing
    for line in lines
        _first_word(line) == "Units" || continue
        units_setup === nothing ||
            throw(ArgumentError("Only one Units { ... } block is allowed"))
        units_setup = _parse_units_block(line, user_vars)
    end
    units_setup !== nothing && (user_vars[:u_LU] = get(user_vars, :u_LU, 0.0))

    # --- Second pass: parse everything ---
    name = ""
    lattice = :D2Q9
    domain = nothing
    physics_params = Dict{Symbol, Any}()
    body_force = Dict{Symbol, KrakenExpr}()
    regions = GeometryRegion[]
    boundaries = BoundarySetup[]
    initial = nothing
    velocity_field = nothing
    modules = Symbol[]
    # Pre-scan modules so boundary parsing can honour module-specific aliases
    # (e.g. `Module axisymmetric` enables `Boundary z/wall/axis ...`).
    for line in lines
        _first_word(line) == "Module" || continue
        push!(modules, _parse_module(line))
    end
    is_axisym = :axisymmetric in modules
    max_steps = 0
    outputs = OutputSetup[]
    diagnostics = nothing
    refinements = RefineSetup[]
    rheology_setups = RheologySetup[]
    sensitivity = nothing
    backend_setup = nothing
    setup_helpers = Dict{Symbol, Float64}()  # reynolds, rayleigh, prandtl, L_ref, U_ref

    for line in lines
        keyword = _first_word(line)

        if keyword == "Simulation"
            name, lattice = _parse_simulation(line)
        elseif keyword == "Domain"
            domain = _parse_domain(line, user_vars)
        elseif keyword == "Physics"
            merge!(physics_params, _parse_physics(line, user_vars;
                                                  allow_auto_nu=units_setup !== nothing))
        elseif keyword == "Define"
            # Already processed in first pass
        elseif keyword == "Units"
            # Already processed before boundary parsing so `u_LU` is an allowed token.
        elseif keyword == "Obstacle"
            push!(regions, _parse_obstacle(line, user_vars))
        elseif keyword == "Fluid"
            push!(regions, _parse_fluid(line, user_vars))
        elseif keyword == "Boundary"
            append!(boundaries, _parse_boundary(line, user_vars; is_axisym=is_axisym))
        elseif keyword == "Refine"
            push!(refinements, _parse_refine(line, user_vars))
        elseif keyword == "Initial"
            initial = _parse_initial(line, user_vars)
        elseif keyword == "Velocity"
            velocity_field = _parse_initial(line, user_vars)  # same { ux = ... uy = ... } syntax
        elseif keyword == "Module"
            # Already collected in the pre-scan above

        elseif keyword == "Run"
            max_steps = _parse_run(line)
        elseif keyword == "Output"
            push!(outputs, _parse_output(line))
        elseif keyword == "Diagnostics"
            diagnostics = _parse_diagnostics(line)
        elseif keyword == "Rheology"
            push!(rheology_setups, _parse_rheology(line, user_vars))
        elseif keyword == "Setup"
            merge!(setup_helpers, _parse_setup(line, user_vars))
        elseif keyword == "Sensitivity"
            sensitivity === nothing ||
                throw(ArgumentError("Only one Sensitivity { ... } block is allowed"))
            sensitivity = _parse_sensitivity(line)
        elseif keyword == "Backend"
            backend_setup === nothing ||
                throw(ArgumentError("Only one Backend directive is allowed"))
            backend_setup = _parse_backend(line)
        else
            known = ("Simulation", "Domain", "Physics", "Define", "Obstacle",
                     "Fluid", "Boundary", "Refine", "Initial", "Velocity",
                     "Module", "Run", "Output", "Diagnostics", "Rheology",
                     "Setup", "Units", "Preset", "Sweep", "Sensitivity", "Backend")
            suggestion = _suggest_name(keyword, known)
            msg = "Unknown keyword '$keyword' in .krk file"
            if suggestion !== nothing
                msg *= " (did you mean: $suggestion?)"
            end
            throw(ArgumentError(msg))
        end
    end

    domain === nothing && throw(ArgumentError("Missing 'Domain' in .krk file"))
    isempty(name) && throw(ArgumentError("Missing 'Simulation' in .krk file"))
    max_steps == 0 && throw(ArgumentError("Missing 'Run' in .krk file"))

    # --- Apply kwargs overrides to Physics and Domain ---
    param_kwargs = Dict{Symbol,Float64}(k => Float64(v) for (k, v) in kwargs)

    # Override Physics params with matching kwargs
    for (k, v) in param_kwargs
        if haskey(physics_params, k)
            physics_params[k] = v
        end
    end

    # Override Domain with matching kwargs (Nx, Ny, Nz, Lx, Ly, Lz)
    if domain !== nothing
        Nx = haskey(param_kwargs, :Nx) ? round(Int, param_kwargs[:Nx]) : domain.Nx
        Ny = haskey(param_kwargs, :Ny) ? round(Int, param_kwargs[:Ny]) : domain.Ny
        Nz = haskey(param_kwargs, :Nz) ? round(Int, param_kwargs[:Nz]) : domain.Nz
        Lx = get(param_kwargs, :Lx, domain.Lx)
        Ly = get(param_kwargs, :Ly, domain.Ly)
        Lz = get(param_kwargs, :Lz, domain.Lz)
        domain = DomainSetup(Lx, Ly, Lz, Nx, Ny, Nz)
    end

    # Override max_steps with kwarg
    if haskey(param_kwargs, :max_steps)
        max_steps = round(Int, param_kwargs[:max_steps])
    end

    # Parse body force if present in physics_params
    for sym in (:Fx, :Fy, :Fz)
        if haskey(physics_params, sym)
            body_force[sym] = parse_kraken_expr(string(physics_params[sym]), user_vars)
            delete!(physics_params, sym)
        end
    end

    # --- Apply Setup helpers (Reynolds, Rayleigh) ---
    _apply_setup_helpers!(physics_params, setup_helpers, domain, boundaries)

    if units_setup !== nothing
        regions = _apply_units_bridge!(units_setup, lattice, domain, regions, boundaries,
                                       physics_params, user_vars)
    end

    collision = _physics_symbol_option(physics_params, :collision, :bgk)
    wall_bc = _physics_symbol_option(physics_params, :wall_bc, :halfwaybb)
    # TODO(K1-followup): wire advection_scheme dispatch.
    # TODO(K1-followup): wire formulation dispatch.
    # TODO(K1-followup): wire gas_model dispatch.

    physics = PhysicsSetup(physics_params, body_force)

    setup = SimulationSetup(name, lattice, domain, physics, user_vars,
                            regions, boundaries, initial, modules,
                            max_steps, outputs, diagnostics, refinements,
                            velocity_field, rheology_setups, nothing, units_setup,
                            collision, wall_bc, sensitivity, backend_setup)

    # --- Validate face names against lattice dimensionality ---
    _validate_faces_vs_lattice(setup)

    # --- Run sanity checks (no summary at parse time — printed at run time) ---
    sanity_check(setup; verbose=false)

    return [setup]
end
