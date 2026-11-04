# Reusable conformance suite of the platform state contract
# (src/platform/state.jl + src/io/checkpoint_hdf5.jl).
#
# Every client of the contract — every `S <: AbstractSimulationState` — must pass
# `run_state_contract_suite`. The suite is solver-agnostic: it only speaks the
# generic verbs, and it compares snapshots bit for bit (no tolerance: a restart
# that is "close" hides a missing piece of state).
#
# Usage, from a client's test file:
#
#     include(joinpath(@__DIR__, "..", "platform", "state_contract_suite.jl"))
#     run_state_contract_suite(MyState;
#         make_state     = () -> init_state(MyState; cfg...),
#         restore_kwargs = (; cfg...),
#         interrupt!     = s -> ...,   # leave `s` in the middle of a cycle
#         tmpdir         = mktempdir())

using Test
using Kraken
# Client hooks are deliberately unexported: name them explicitly.
using Kraken: snapshot, validate_snapshot, at_boundary, check_updatable, migrate

_bitwise_equal(a::Array, b::Array) =
    typeof(a) == typeof(b) && size(a) == size(b) && all(i -> a[i] === b[i], eachindex(a))
_bitwise_equal(a, b) = typeof(a) == typeof(b) && a === b

const _SNAPSHOT_DICTS = (:fields, :series, :scalars, :identity, :run_control, :parameters, :derived)

"""
    snapshot_differences(a, b; classes=_SNAPSHOT_DICTS) -> Vector{String}

Names of everything that differs, bit for bit, between two snapshots (empty when
identical). Returned as a list so that a failing `@test isempty(...)` prints the
offending keys.
"""
function snapshot_differences(a::StateSnapshot, b::StateSnapshot; classes=_SNAPSHOT_DICTS)
    diffs = String[]
    a.solver == b.solver || push!(diffs, "solver")
    a.schema_version == b.schema_version || push!(diffs, "schema_version")
    a.cycle == b.cycle || push!(diffs, "cycle")
    for class in classes
        da, db = getfield(a, class), getfield(b, class)
        for key in sort!(collect(union(keys(da), keys(db))))
            same = haskey(da, key) && haskey(db, key) && _bitwise_equal(da[key], db[key])
            same || push!(diffs, "$class/$key")
        end
    end
    return diffs
end

"""
    tampered(snap; kwargs...) -> StateSnapshot

Deep copy of `snap` with some components replaced, e.g.
`tampered(snap; schema_version=2)` or `tampered(snap; identity=other_dict)`.
"""
function tampered(snap::StateSnapshot; kwargs...)
    parts = Dict{Symbol,Any}(name => deepcopy(getfield(snap, name)) for name in fieldnames(StateSnapshot))
    for (name, value) in kwargs
        parts[name] = value
    end
    return StateSnapshot(; parts...)
end

_other_value(v::String) = v * "_x"
_other_value(v::Bool) = !v
_other_value(v::Int) = v + 1
_other_value(v::AbstractFloat) = v == 0 ? one(v) : -v
_other_value(::Nothing) = "was_nothing"

# A small, sign-preserving change: stays inside any reasonable bound.
_nudged(v::AbstractFloat) = v == 0 ? eps(typeof(v)) : v * (1 + one(v) / 1024)
_nudged(v::Int) = v + 1
_nudged(v) = v

function _perturb_first!(a::Array)
    isempty(a) && return false
    a[1] = a[1] isa Bool ? !a[1] : a[1] + one(eltype(a))
    return true
end

function _error_message(f)
    try
        f()
    catch err
        return err isa CheckpointError ? err.msg : "not a CheckpointError: " * sprint(showerror, err)
    end
    return "no error thrown"
end

# The exception `f()` throws, or `nothing`.
function _thrown(f)
    try
        f()
    catch err
        return err
    end
    return nothing
end

# `f()` must throw an ArgumentError whose message names the boundary.
_refused_off_boundary(f) = (err = _thrown(f); err isa ArgumentError && occursin("boundary", err.msg))

"""
    run_state_contract_suite(::Type{S}; make_state, restore_kwargs, interrupt!, tmpdir,
                             n_first=7, n_second=13, negative_controls=true)

Conformance suite of the state contract for client `S`.

- `make_state()`: a fresh, deterministic state at cycle 0 (called several times;
  two calls must give identical states).
- `restore_kwargs`: keywords for `restore_state(S, snap; ...)` /
  `load_checkpoint(S, path; ...)` describing the same simulation as `make_state`.
- `interrupt!(state)`: leaves `state` in the middle of a cycle (typically makes
  `advance!` throw and catches the exception), so that `at_boundary(state)` is false.
- `tmpdir`: an existing directory the suite may write into.
- `n_first`, `n_second`: the split; choose them so that histories are sampled in
  both segments and at least once off the segment boundary.
- `negative_controls`: for every stored field, check that perturbing it before the
  restore changes the final state, i.e. that the comparison has teeth and that the
  field is not dead weight. Disable only with a written reason.
- `parameter_snapshot`: optional `(snapshot, name, value) -> snapshot` callback
  that applies the client's public update verb and exports coherent derived values
  and parameter provenance. Default: change only the parameter dictionary (toy).
"""
function run_state_contract_suite(::Type{S}; make_state, restore_kwargs, interrupt!, tmpdir,
                                  n_first::Int=7, n_second::Int=13,
                                  parameter_snapshot=nothing,
                                  negative_controls::Bool=true) where {S<:AbstractSimulationState}
    n_total = n_first + n_second
    @testset "state contract: $S" begin
        @testset "determinism and global cycle counter" begin
            @test isempty(snapshot_differences(export_state(make_state()), export_state(make_state())))
            s = make_state()
            @test s isa S
            @test at_boundary(s)
            @test export_state(s).cycle == 0
            @test advance!(s, n_first) === s
            @test export_state(s).cycle == n_first
            @test solution(s) isa AbstractSolution
            @test updatable_parameters(S) isa ParameterSpace
        end

        # Reference: one continuous run, and its image at the split point.
        continuous = advance!(make_state(), n_total)
        final = export_state(continuous)
        mid_state = advance!(make_state(), n_first)
        mid = export_state(mid_state)
        path = joinpath(tmpdir, "contract_$(nameof(S)).h5")

        @testset "split advance == continuous advance" begin
            s = advance!(advance!(make_state(), n_first), n_second)
            @test isempty(snapshot_differences(export_state(s), final))
        end

        @testset "zero-step round trip, in memory" begin
            restored = restore_state(S, mid; restore_kwargs...)
            @test restored isa S
            @test isempty(snapshot_differences(export_state(restored), mid))
            # The restored state owns its arrays: advancing it leaves `mid` intact.
            reference = deepcopy(mid)
            advance!(restored, n_second)
            @test isempty(snapshot_differences(mid, reference))
            @test isempty(snapshot_differences(export_state(restored), final))
        end

        @testset "zero-step round trip, via disk" begin
            @test save_checkpoint(path, mid_state) == path
            @test isempty(snapshot_differences(read_checkpoint(path), mid))
            @test checkpoint_info(path).cycle == n_first
            restored = load_checkpoint(S, path; restore_kwargs...)
            @test isempty(snapshot_differences(export_state(restored), mid))
            advance!(restored, n_second)
            @test isempty(snapshot_differences(export_state(restored), final))
            @test !ispath(path * ".tmp")
        end

        if negative_controls
            @testset "negative control: field $name" for name in sort!(collect(keys(mid.fields)))
                bad = tampered(mid)
                if _perturb_first!(bad.fields[name])
                    restored = restore_state(S, bad; restore_kwargs...)
                    advance!(restored, n_second)
                    # Raw snapshot: the perturbed run is allowed to be non-finite.
                    @test !isempty(snapshot_differences(snapshot(restored), final))
                end
            end
        end

        @testset "export refused in the middle of a cycle" begin
            s = advance!(make_state(), n_first)
            interrupt!(s)
            @test !at_boundary(s)
            @test occursin("at_boundary", _error_message(() -> export_state(s)))
            good = read(path)
            @test_throws CheckpointError save_checkpoint(path, s)
            @test read(path) == good
            @test !ispath(path * ".tmp")
        end

        @testset "advance! and solution refused in the middle of a cycle" begin
            s = advance!(make_state(), n_first)
            interrupt!(s)
            @test !at_boundary(s)
            # Raw snapshot (export_state refuses): the image right after the failure.
            broken = snapshot(s)
            @test _refused_off_boundary(() -> advance!(s, 1))
            @test _refused_off_boundary(() -> advance!(s, 0))
            @test _refused_off_boundary(() -> advance!(s, n_second; sample_final=true))
            @test _refused_off_boundary(() -> solution(s))
            # The rejected calls changed nothing: flag, counter, arrays, histories.
            @test !at_boundary(s)
            @test snapshot(s).cycle == broken.cycle
            @test isempty(snapshot_differences(snapshot(s), broken))
            # Recovery is a fresh state or a restore, never the broken one.
            fresh = make_state()
            @test at_boundary(fresh) && advance!(fresh, 1) === fresh
            restored = load_checkpoint(S, path; restore_kwargs...)
            @test at_boundary(restored) && solution(restored) isa AbstractSolution
        end

        @testset "solver and schema mismatch rejected" begin
            msg = _error_message(() -> restore_state(S, tampered(mid; solver=mid.solver * "_other"); restore_kwargs...))
            @test startswith(msg, "solver:")
            msg = _error_message(() -> restore_state(S, tampered(mid; schema_version=mid.schema_version + 1); restore_kwargs...))
            @test startswith(msg, "schema_version:")
        end

        @testset "identity mismatch rejected, key by key" begin
            @test !isempty(mid.identity)
            for key in sort!(collect(keys(mid.identity)))
                ident = deepcopy(mid.identity)
                ident[key] = _other_value(ident[key])
                msg = _error_message(() -> restore_state(S, tampered(mid; identity=ident); restore_kwargs...))
                @test startswith(msg, "identity/$key:")
            end
            ident = deepcopy(mid.identity)
            ident["contract_suite_extra_key"] = 1
            msg = _error_message(() -> restore_state(S, tampered(mid; identity=ident); restore_kwargs...))
            @test startswith(msg, "identity/contract_suite_extra_key:")
            # Same verdict when the mismatch comes from a file.
            other = joinpath(tmpdir, "contract_$(nameof(S))_identity.h5")
            write_checkpoint(other, tampered(mid; identity=ident))
            msg = _error_message(() -> load_checkpoint(S, other; restore_kwargs...))
            @test startswith(msg, "identity/contract_suite_extra_key:")
        end

        @testset "run_control and parameters differences accepted" begin
            run_control = Dict{String,Any}(k => _nudged(v) for (k, v) in mid.run_control)
            run_control["contract_suite_extra_key"] = "ignored"
            restored = restore_state(S, tampered(mid; run_control=run_control); restore_kwargs...)
            differences = snapshot_differences(export_state(restored), mid; classes=(:fields, :series, :scalars))
            @test isempty(differences)
            for key in keys(mid.parameters)
                parameters = deepcopy(mid.parameters)
                parameters[key] = _nudged(parameters[key])
                # Updating only T while retaining old derived values/provenance
                # would create a corrupt checkpoint, not a legitimate change.
                changed = parameter_snapshot === nothing ? tampered(mid; parameters=parameters) :
                          parameter_snapshot(mid, Symbol(key), parameters[key])
                @test changed.parameters[key] == parameters[key]
                @test restore_state(S, changed; restore_kwargs...) isa S
            end
        end
    end
    return nothing
end
