# ============================================================================
# src/platform/state.jl
# Platform state contract — resumable simulation state, solver-agnostic.
#
# Positioning against the rest of `src/platform/`:
#
# - a *state* is the transient counterpart of the `u` of `residual.jl`: everything a
#   solver must carry from one cycle to the next (populations, force history,
#   counters), and nothing that can be rebuilt from it;
# - [`advance!`](@ref)`(state, n)` is `n` applications of the step `G` whose fixed
#   point `residual.jl` exposes as `R(u, p) = u - G(u, p)`;
# - [`solve`](@ref) stays the one-shot verb: it is `init_state` + `advance!` +
#   [`solution`](@ref), and `solution(state)` returns an [`AbstractSolution`](@ref);
# - the parameters that may change between two segments of a run are declared with
#   the existing `ParameterSpace` ([`updatable_parameters`](@ref)), not a parallel list.
#
# This file defines the in-memory contract only: the [`StateSnapshot`](@ref)
# container, the generic verbs a client implements, and the platform-owned checks
# ([`export_state`](@ref), [`check_compatible`](@ref)). It knows nothing about any
# file format; the on-disk layer lives in `src/io/`.
# ============================================================================

"""
    AbstractSimulationState

Supertype of every resumable solver state. A client (one solver) subtypes it and
implements the verbs of this file: [`init_state`](@ref), [`advance!`](@ref),
[`solution`](@ref), [`snapshot`](@ref), [`restore_state`](@ref), and optionally
[`validate_snapshot`](@ref), [`at_boundary`](@ref), [`update_parameter!`](@ref),
[`updatable_parameters`](@ref), [`migrate`](@ref).
"""
abstract type AbstractSimulationState end

"""
    CheckpointError(msg)

Raised whenever a snapshot or a checkpoint file is refused: invalid content,
non-finite values, solver / schema / identity mismatch, damaged or truncated
file. The message always names the offending key or object.
"""
struct CheckpointError <: Exception
    msg::String
end

Base.showerror(io::IO, e::CheckpointError) = print(io, "CheckpointError: ", e.msg)

"""Element types allowed in `fields` and `series` of a [`StateSnapshot`](@ref)."""
const SNAPSHOT_ELTYPES = (Float32, Float64, Int32, Int64, Bool)

"""
Names of the three configuration classes of a [`StateSnapshot`](@ref); these are
the only scalar dictionaries in which `nothing` is a legal value.
"""
const SNAPSHOT_CONFIG_CLASSES = ("identity", "run_control", "parameters")

"""
    StateSnapshot(; solver, schema_version, cycle, fields, series, scalars,
                    identity, run_control, parameters, derived)

Plain host-side image of a solver state, free of Julia-specific types, so that it
can be written by any storage layer and read from another language.

# Fields
- `solver::String`: client name (e.g. `"ehd_ec"`).
- `schema_version::Int`: version of *this client's* set of keys.
- `cycle::Int`: global cycle counter (counts from the start of the run, not of
  the segment).
- `fields::Dict{String,Array}`: arrays whose shape is part of the contract
  (checked by [`validate_snapshot`](@ref)). Names may contain `/` to express a
  hierarchy (`"block003/f"`).
- `series::Dict{String,Vector}`: append-only histories, free length.
- `scalars::Dict{String,Any}`: carried scalar state (must be finite).
- `identity::Dict{String,Any}`: configuration that defines *which* simulation
  this is; compared key by key on restore ([`check_compatible`](@ref)).
- `run_control::Dict{String,Any}`: configuration that may differ between two
  segments of the same run (horizon, backend); never compared.
- `parameters::Dict{String,Any}`: physical parameters governed by
  [`update_parameter!`](@ref); never compared.
- `derived::Dict{String,Any}`: values recomputable from the configuration, stored
  so that a client can detect drift in the code that derives them.

# Allowed values
Scalars: `String`, `Bool`, `Int`, `Float32`, `Float64`, plus `nothing` in the
three configuration classes only. Array element types: `Float32`, `Float64`,
`Int32`, `Int64`, `Bool`. A `Symbol` must be converted to `String` by the client.
Anything else is rejected by the constructor with a [`CheckpointError`](@ref)
naming the key. Scalar keys must be non-empty, free of `/` and control
characters, and the name `keys` is reserved for storage layers.

The constructor does not copy: [`snapshot`](@ref) must hand over host arrays the
snapshot may own.
"""
struct StateSnapshot
    solver::String
    schema_version::Int
    cycle::Int
    fields::Dict{String,Array}
    series::Dict{String,Vector}
    scalars::Dict{String,Any}
    identity::Dict{String,Any}
    run_control::Dict{String,Any}
    parameters::Dict{String,Any}
    derived::Dict{String,Any}

    function StateSnapshot(solver, schema_version, cycle, fields, series, scalars,
                           identity, run_control, parameters, derived)
        snap = new(String(solver), Int(schema_version), Int(cycle),
                   _as_dict(Array, fields, "fields"), _as_dict(Vector, series, "series"),
                   _as_dict(Any, scalars, "scalars"), _as_dict(Any, identity, "identity"),
                   _as_dict(Any, run_control, "run_control"),
                   _as_dict(Any, parameters, "parameters"), _as_dict(Any, derived, "derived"))
        validate_content(snap)
        return snap
    end
end

function StateSnapshot(; solver, schema_version, cycle,
                       fields=Dict{String,Array}(), series=Dict{String,Vector}(),
                       scalars=Dict{String,Any}(), identity=Dict{String,Any}(),
                       run_control=Dict{String,Any}(), parameters=Dict{String,Any}(),
                       derived=Dict{String,Any}())
    return StateSnapshot(solver, schema_version, cycle, fields, series, scalars,
                         identity, run_control, parameters, derived)
end

function _as_dict(::Type{V}, d, class::String) where {V}
    d isa Dict{String,V} && return d
    out = Dict{String,V}()
    for (k, v) in pairs(d)
        k isa AbstractString ||
            throw(CheckpointError("$class: key $(repr(k)) must be a String, got $(typeof(k))"))
        v isa V ||
            throw(CheckpointError("$class/$k: expected a $(V), got $(typeof(v))"))
        out[String(k)] = v
    end
    return out
end

function _check_array_name(class::String, name::String)
    ok = !isempty(name) && !any(iscntrl, name) &&
         all(seg -> !isempty(seg) && seg != "." && seg != "..", split(name, '/'))
    ok || throw(CheckpointError("$class: invalid name $(repr(name)) (non-empty `/`-separated segments required)"))
    return nothing
end

function _check_scalar_key(class::String, key::String)
    ok = !isempty(key) && !occursin('/', key) && !any(iscntrl, key) && key != "keys"
    ok || throw(CheckpointError("$class: invalid key $(repr(key)) (non-empty, no `/`, no control character, not the reserved name `keys`)"))
    return nothing
end

function _check_arrays(class::String, d::Dict)
    names = sort!(collect(keys(d)))
    for name in names
        _check_array_name(class, name)
        T = eltype(d[name])
        T in SNAPSHOT_ELTYPES ||
            throw(CheckpointError("$class/$name: element type $T is not allowed (allowed: $(join(SNAPSHOT_ELTYPES, ", ")))"))
        ndims(d[name]) >= 1 ||
            throw(CheckpointError("$class/$name: zero-dimensional arrays are not allowed (use `scalars`)"))
    end
    # A name cannot be both a leaf and a directory ("a" and "a/b").
    for name in names, other in names
        startswith(other, name * "/") &&
            throw(CheckpointError("$class: name $(repr(name)) is also used as a prefix by $(repr(other))"))
    end
    return nothing
end

function _check_scalars(class::String, d::Dict{String,Any})
    allow_nothing = class in SNAPSHOT_CONFIG_CLASSES
    for (key, v) in d
        _check_scalar_key(class, key)
        ok = v isa Union{String,Bool,Int,Float32,Float64} || (allow_nothing && v === nothing)
        ok || throw(CheckpointError("$class/$key: value of type $(typeof(v)) is not allowed " *
                                    "(allowed: String, Bool, Int, Float32, Float64" *
                                    (allow_nothing ? ", nothing" : "") * ")"))
    end
    return nothing
end

"""
    validate_content(snap::StateSnapshot) -> snap

Check names, keys, value types and element types of every entry of `snap`. Run by
the constructor, and again by storage layers before writing because the
dictionaries are mutable. Throws a [`CheckpointError`](@ref) naming the key.
"""
function validate_content(snap::StateSnapshot)
    isempty(snap.solver) && throw(CheckpointError("solver: name must not be empty"))
    snap.cycle >= 0 || throw(CheckpointError("cycle: must be >= 0, got $(snap.cycle)"))
    _check_arrays("fields", snap.fields)
    _check_arrays("series", snap.series)
    _check_scalars("scalars", snap.scalars)
    _check_scalars("identity", snap.identity)
    _check_scalars("run_control", snap.run_control)
    _check_scalars("parameters", snap.parameters)
    _check_scalars("derived", snap.derived)
    return snap
end

# Function barrier (the dictionaries hold abstractly typed arrays) and an explicit
# loop (`findfirst` throws on an empty N-d array).
_first_nonfinite(::Array) = nothing
function _first_nonfinite(a::Array{<:AbstractFloat})
    for i in CartesianIndices(a)
        isfinite(a[i]) || return i
    end
    return nothing
end

"""
    check_finite(snap::StateSnapshot) -> snap

Refuse a snapshot whose `fields` or `series` hold a `NaN` or an `Inf`: a diverged
state must never replace the last good restart point. Carried `scalars` and
configuration values are not checked: a convergence indicator legitimately reads
`Inf` before its first sample, and `Inf` is a legitimate run horizon. Divergence
shows in the arrays. Throws a [`CheckpointError`](@ref) naming the key and the
first offending index.
"""
function check_finite(snap::StateSnapshot)
    for (class, d) in (("fields", snap.fields), ("series", snap.series))
        for (name, a) in d
            i = _first_nonfinite(a)
            i === nothing ||
                throw(CheckpointError("$class/$name: non-finite value $(a[i]) at index $(Tuple(i)); snapshot refused"))
        end
    end
    return snap
end

_unimplemented(verb::String, S) =
    error("`$verb` is not implemented for $S: a state-contract client must add a method " *
          "(see docs/platform/07-STATE-CONTRACT.md)")

"""
    init_state(::Type{S}; kwargs...) -> S

Build a fresh state of client type `S <: AbstractSimulationState` at cycle 0 from
the client's configuration keywords. No default: each client adds a method.
"""
init_state(::Type{S}; kwargs...) where {S<:AbstractSimulationState} = _unimplemented("init_state", S)

"""
    advance!(state, n; sample_final=false) -> state

Apply `n` cycles of the solver step to `state`, in place. Sampling of histories is
decided on the **global** cycle counter only, so that `advance!(s, a); advance!(s, b)`
and `advance!(s, a + b)` leave bit-identical states. `sample_final=true` forces a
sample on the last cycle of the call (what a one-shot driver does at its horizon).

A client method MUST start with [`require_boundary`](@ref)`(state, "advance!")`,
before touching anything and whatever `n` is (including `n == 0`): a state whose
previous `advance!` threw mid-cycle holds populations and history at mismatched
time levels, and advancing it again would silently raise [`at_boundary`](@ref)
on garbage. The rejected call leaves `state` untouched.
"""
advance!(state::AbstractSimulationState, n::Integer; sample_final::Bool=false) =
    _unimplemented("advance!", typeof(state))

"""
    solution(state) -> AbstractSolution

The queryable result of `state` at its current cycle, as an
[`AbstractSolution`](@ref): what [`solve`](@ref) would have returned had the run
stopped here. Must not mutate `state`. A client method MUST start with
[`require_boundary`](@ref)`(state, "solution")`: there is no meaningful result
for a state left in the middle of a cycle.
"""
solution(state::AbstractSimulationState) = _unimplemented("solution", typeof(state))

"""
    snapshot(state) -> StateSnapshot

Client hook: copy everything needed to resume `state` into a host-side
[`StateSnapshot`](@ref). Raw: performs no check. Callers use
[`export_state`](@ref), which adds the platform-owned refusals.
"""
snapshot(state::AbstractSimulationState) = _unimplemented("snapshot", typeof(state))

"""
    restore_state(::Type{S}, snap::StateSnapshot; backend, kwargs...) -> S

Client hook: rebuild a state of type `S` from `snap` on `backend`. `kwargs` carry
the run-control values of the new segment. A conforming method calls
[`check_compatible`](@ref) and [`validate_snapshot`](@ref) **before allocating
anything**, so that a refused snapshot costs nothing.
"""
restore_state(::Type{S}, snap::StateSnapshot; kwargs...) where {S<:AbstractSimulationState} =
    _unimplemented("restore_state", S)

"""
    validate_snapshot(::Type{S}, snap::StateSnapshot) -> nothing

Client hook: check that the expected `fields` / `series` are present with the
expected shapes and element types, throwing a [`CheckpointError`](@ref) naming the
key otherwise. Shapes are the client's business (with mesh refinement they are
part of the state), hence a hook rather than a comparison with a fresh
`init_state`. Default: no check.
"""
validate_snapshot(::Type{S}, snap::StateSnapshot) where {S<:AbstractSimulationState} = nothing

"""
    at_boundary(state) -> Bool

`true` when `state` sits exactly between two cycles. A client whose `advance!` can
be interrupted mid-cycle (caught exception) lowers the flag on entry of a cycle and
raises it on exit; [`export_state`](@ref) refuses a state that is not at a
boundary, and the client's [`advance!`](@ref) and [`solution`](@ref) refuse it
through [`require_boundary`](@ref). Default: `true`.
"""
at_boundary(state::AbstractSimulationState) = true

"""
    require_boundary(state, what::AbstractString) -> nothing

Platform-owned guard for the client verbs that read or move a state: throw an
`ArgumentError` when `!at_boundary(state)`, i.e. when a previous `advance!` threw in
the middle of a cycle. `what` names the refusing verb in the message. There is no
repair path: the caller starts from a fresh state or restores a checkpoint.
Unexported; call as `Kraken.require_boundary`. [`export_state`](@ref) performs the
same refusal with a [`CheckpointError`](@ref).
"""
function require_boundary(state::AbstractSimulationState, what::AbstractString)
    at_boundary(state) ||
        throw(ArgumentError("$what: $(nameof(typeof(state))) state is not on a completed cycle boundary " *
                            "(a previous advance! failed mid-cycle); start from a fresh state or restore a checkpoint"))
    return nothing
end

"""
    update_parameter!(state, name::Symbol, value) -> state

Change one physical parameter between two segments of a run. Only the names
declared by [`updatable_parameters`](@ref) are legal; a client method starts with
[`check_updatable`](@ref). No default.
"""
update_parameter!(state::AbstractSimulationState, name::Symbol, value) =
    _unimplemented("update_parameter!", typeof(state))

"""
    updatable_parameters(::Type{S}) -> ParameterSpace

The parameters of client `S` that [`update_parameter!`](@ref) accepts, with their
bounds, expressed with the platform's existing `ParameterSpace` (the same object
`fit` consumes). Default: an empty space, i.e. nothing is updatable.
"""
updatable_parameters(::Type{S}) where {S<:AbstractSimulationState} =
    ParameterSpace(Symbol[], Float64[], Float64[])

"""
    check_updatable(::Type{S}, name::Symbol, value) -> nothing

Platform-owned guard for [`update_parameter!`](@ref): `name` must be a free entry
of `updatable_parameters(S)` and `value` must lie within its bounds. Throws an
`ArgumentError` naming the parameter otherwise.
"""
function check_updatable(::Type{S}, name::Symbol, value) where {S<:AbstractSimulationState}
    ps = updatable_parameters(S)
    i = findfirst(==(name), ps.names)
    (i === nothing || ps.fixed[i]) &&
        throw(ArgumentError("parameter :$name of $S is not updatable (updatable: $(ps.names[.!ps.fixed]))"))
    (value isa Real && ps.lower[i] <= value <= ps.upper[i]) ||
        throw(ArgumentError("parameter :$name = $(repr(value)) is outside its bounds [$(ps.lower[i]), $(ps.upper[i])]"))
    return nothing
end

"""
    export_state(state) -> StateSnapshot

Platform-owned export: refuse a state that is not [`at_boundary`](@ref), call the
client's [`snapshot`](@ref), refuse non-finite values ([`check_finite`](@ref)),
return the snapshot. This is the only sanctioned way to obtain a snapshot that is
going to be stored.
"""
function export_state(state::AbstractSimulationState)
    at_boundary(state) ||
        throw(CheckpointError("at_boundary: $(typeof(state)) was left in the middle of a cycle; export refused"))
    snap = snapshot(state)
    snap isa StateSnapshot ||
        throw(CheckpointError("snapshot: $(typeof(state)) returned a $(typeof(snap)), expected a StateSnapshot"))
    validate_content(snap)
    check_finite(snap)
    return snap
end

"""
    migrate(::Type{S}, snap::StateSnapshot, from_version::Int) -> StateSnapshot

Hook to upgrade a snapshot written with an older `schema_version` of client `S`.
Reject-only for now: the default throws a [`CheckpointError`](@ref). It exists so
that the first schema change is a new method, not a redesign.
"""
function migrate(::Type{S}, snap::StateSnapshot, from_version::Int) where {S<:AbstractSimulationState}
    throw(CheckpointError("schema_version: no migration from version $from_version is defined for $S " *
                          "(solver $(repr(snap.solver)))"))
end

_describe(v) = string(repr(v), "::", typeof(v))

"""
    check_compatible(snap; solver, schema_version, identity, identity_defaults=Dict()) -> StateSnapshot
    check_compatible(::Type{S}, snap; ...) -> StateSnapshot

Decide whether `snap` may be restored into the simulation the caller is
configuring. Compared: the solver name, the client's `schema_version`, and the
`identity` dictionary **key by key** (same type and same value). A key missing
from the snapshot is filled from `identity_defaults` (the value the client used
before the key existed); missing without a default, extra, or different keys are
errors. `run_control` and `parameters` are deliberately **not** compared:
extending a run or restoring after a parameter update must work.

The second form gives the client's [`migrate`](@ref) hook a chance when the schema
versions differ. Returns the (possibly migrated) snapshot; throws a
[`CheckpointError`](@ref) naming the key, the expected and the found value.
"""
function check_compatible(snap::StateSnapshot; solver::AbstractString, schema_version::Integer,
                          identity::AbstractDict,
                          identity_defaults::AbstractDict=Dict{String,Any}())
    snap.solver == solver ||
        throw(CheckpointError("solver: expected $(repr(String(solver))), found $(repr(snap.solver))"))
    snap.schema_version == schema_version ||
        throw(CheckpointError("schema_version: expected $schema_version, found $(snap.schema_version)"))
    for (key, expected) in identity
        found = if haskey(snap.identity, key)
            snap.identity[key]
        elseif haskey(identity_defaults, key)
            identity_defaults[key]
        else
            throw(CheckpointError("identity/$key: expected $(_describe(expected)), key missing from the " *
                                  "snapshot and no default declared"))
        end
        (typeof(found) == typeof(expected) && isequal(found, expected)) ||
            throw(CheckpointError("identity/$key: expected $(_describe(expected)), found $(_describe(found))"))
    end
    for key in keys(snap.identity)
        haskey(identity, key) ||
            throw(CheckpointError("identity/$key: found $(_describe(snap.identity[key])), key not expected by the caller"))
    end
    return snap
end

function check_compatible(::Type{S}, snap::StateSnapshot; schema_version::Integer,
                          kwargs...) where {S<:AbstractSimulationState}
    if snap.schema_version != schema_version
        snap = migrate(S, snap, snap.schema_version)
    end
    return check_compatible(snap; schema_version=schema_version, kwargs...)
end
