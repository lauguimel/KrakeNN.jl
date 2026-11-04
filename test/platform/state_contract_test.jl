using Test
using Kraken
# Client hooks are deliberately unexported: name them explicitly.
using Kraken: snapshot, validate_snapshot, at_boundary, check_updatable, migrate

include(joinpath(@__DIR__, "state_contract_suite.jl"))

# ---------------------------------------------------------------------------
# Toy client of the state contract. Small enough to read in one go, but with one
# of everything the contract has to carry: two coupled Float arrays, a Bool array,
# two histories sampled on the global cycle, a carried scalar, an updatable
# parameter, a String and a `nothing` configuration value, a derived value.
# (structs are top-level only in Julia, so it lives outside the @testset.)
# ---------------------------------------------------------------------------
mutable struct ToyState{T} <: Kraken.AbstractSimulationState
    a::Matrix{T}
    b::Array{T,3}
    mask::Matrix{Bool}
    hist_cycle::Vector{Int64}
    hist_sum::Vector{T}
    cycle::Int
    last_sum::T                        # carried scalar
    gain::T                            # updatable parameter
    label::String                      # identity (String)
    limiter::Union{Nothing,Float64}    # identity (may be `nothing`)
    history_interval::Int              # identity
    max_cycles::Int                    # run control
    fail_at::Int                       # test device: throw in the middle of this cycle
    boundary::Bool
end

struct ToySolution{T} <: Kraken.AbstractSolution
    a::Matrix{T}
end

const TOY_SCHEMA = 1
toy_coef(nx, ny) = 1.0 / (nx * ny)

function toy_identity(::Type{T}, nx, ny, label, limiter, history_interval) where {T}
    return Dict{String,Any}("FT" => string(T), "nx" => nx, "ny" => ny, "label" => label,
                            "limiter" => limiter, "history_interval" => history_interval)
end

function Kraken.init_state(::Type{ToyState{T}}; nx, ny, label="toy", limiter=nothing,
                           history_interval=3, max_cycles=100, gain=0.25) where {T}
    a = T[sin(0.3i) + cos(0.7j) for i in 1:nx, j in 1:ny]
    b = T[0.1 * cos(0.2i * k) - 0.05j for i in 1:nx, j in 1:ny, k in 1:3]
    mask = Bool[isodd(i + j) for i in 1:nx, j in 1:ny]
    return ToyState{T}(a, b, mask, Int64[], T[], 0, zero(T), T(gain), label, limiter,
                       history_interval, max_cycles, -1, true)
end

function Kraken.advance!(s::ToyState{T}, n::Integer; sample_final::Bool=false) where {T}
    Kraken.require_boundary(s, "advance!")
    nx, ny = size(s.a)
    for step in 1:n
        s.boundary = false
        old = copy(s.a)
        for j in 1:ny, i in 1:nx
            ip, jp = mod1(i + 1, nx), mod1(j + 1, ny)
            s.a[i, j] = T(0.5) * old[i, j] + T(0.25) * (old[ip, j] + old[i, jp]) +
                        s.gain * sin(s.b[i, j, 1]) + T(1e-3) * s.last_sum
        end
        s.cycle + 1 == s.fail_at && error("toy failure in the middle of cycle $(s.fail_at)")
        for k in 1:3, j in 1:ny, i in 1:nx
            s.b[i, j, k] = T(0.9) * s.b[i, j, k] + (s.mask[i, j] ? T(0.1) * s.a[i, j] : zero(T))
        end
        if s.cycle % 5 == 4      # the mask is part of the state: it evolves
            s.mask[mod1(s.cycle, nx), 1] = !s.mask[mod1(s.cycle, nx), 1]
        end
        s.last_sum = sum(s.a) / length(s.a)
        s.cycle += 1
        if s.cycle % s.history_interval == 0 || (sample_final && step == n)
            push!(s.hist_cycle, s.cycle)
            push!(s.hist_sum, s.last_sum)
        end
        s.boundary = true
    end
    return s
end

function Kraken.solution(s::ToyState)
    Kraken.require_boundary(s, "solution")
    return ToySolution(copy(s.a))
end
Kraken.at_boundary(s::ToyState) = s.boundary
Kraken.updatable_parameters(::Type{<:ToyState}) = ParameterSpace([:gain], [0.0], [1.0])

function Kraken.update_parameter!(s::ToyState{T}, name::Symbol, value) where {T}
    check_updatable(typeof(s), name, value)
    s.gain = T(value)
    return s
end

function Kraken.snapshot(s::ToyState{T}) where {T}
    nx, ny = size(s.a)
    return StateSnapshot(solver="toy", schema_version=TOY_SCHEMA, cycle=s.cycle,
        fields=Dict("a" => copy(s.a), "aux/b" => copy(s.b), "aux/mask" => copy(s.mask)),
        series=Dict("cycle" => copy(s.hist_cycle), "sum" => copy(s.hist_sum)),
        scalars=Dict("last_sum" => s.last_sum),
        identity=toy_identity(T, nx, ny, s.label, s.limiter, s.history_interval),
        run_control=Dict("max_cycles" => s.max_cycles, "backend" => "cpu"),
        parameters=Dict("gain" => s.gain),
        derived=Dict("coef" => toy_coef(nx, ny)))
end

function Kraken.validate_snapshot(::Type{ToyState{T}}, snap::StateSnapshot) where {T}
    nx, ny = snap.identity["nx"], snap.identity["ny"]
    expected = ("a" => Array{T,2} => (nx, ny), "aux/b" => Array{T,3} => (nx, ny, 3),
                "aux/mask" => Array{Bool,2} => (nx, ny))
    for (name, (type, dims)) in expected
        haskey(snap.fields, name) || throw(CheckpointError("fields/$name: missing"))
        found = snap.fields[name]
        found isa type ||
            throw(CheckpointError("fields/$name: expected element type $(eltype(type)), found $(eltype(found))"))
        size(found) == dims ||
            throw(CheckpointError("fields/$name: expected size $dims, found $(size(found))"))
    end
    length(snap.series["cycle"]) == length(snap.series["sum"]) ||
        throw(CheckpointError("series/sum: length differs from series/cycle"))
    return nothing
end

function Kraken.restore_state(::Type{ToyState{T}}, snap::StateSnapshot; backend=nothing, nx, ny,
                              label="toy", limiter=nothing, history_interval=3,
                              max_cycles=100) where {T}
    snap = check_compatible(ToyState{T}, snap; solver="toy", schema_version=TOY_SCHEMA,
                            identity=toy_identity(T, nx, ny, label, limiter, history_interval),
                            identity_defaults=Dict{String,Any}("limiter" => nothing))
    validate_snapshot(ToyState{T}, snap)
    snap.derived["coef"] == toy_coef(nx, ny) ||
        throw(CheckpointError("derived/coef: expected $(toy_coef(nx, ny)), found $(snap.derived["coef"])"))
    return ToyState{T}(copy(snap.fields["a"]), copy(snap.fields["aux/b"]), copy(snap.fields["aux/mask"]),
                       copy(snap.series["cycle"]), copy(snap.series["sum"]), snap.cycle,
                       snap.scalars["last_sum"], T(snap.parameters["gain"]), label, limiter,
                       history_interval, max_cycles, -1, true)
end

function toy_interrupt!(s::ToyState)
    s.fail_at = s.cycle + 1
    try
        advance!(s, 1)
    catch
    end
    return s
end

struct BareState <: Kraken.AbstractSimulationState end

# --- raw-byte helpers: damage a file without going through the storage library ---
function byte_offsets(buf::Vector{UInt8}, pattern::Vector{UInt8})
    n = length(pattern)
    return [i for i in 1:(length(buf) - n + 1) if @view(buf[i:i+n-1]) == pattern]
end
value_bytes(x::String) = Vector{UInt8}(x)
value_bytes(x) = collect(reinterpret(UInt8, [x]))

# Flip one bit at every place `value` occurs in `src`; each damaged copy must be
# refused. Returns the number of places (0 would make the test vacuous).
function count_refused_flips(src, value; byte=0, bit=0x01)
    buf = read(src)
    hits = byte_offsets(buf, value_bytes(value))
    refused = 0
    for hit in hits
        bad = copy(buf)
        bad[hit + byte] ⊻= bit
        dst = src * ".damaged"
        write(dst, bad)
        msg = _error_message(() -> read_checkpoint(dst))
        refused += occursin(dst, msg) && !startswith(msg, "no error") && !startswith(msg, "not a")
    end
    return (hits=length(hits), refused=refused)
end

function disk_snapshot()
    a = Float64[0.8123456789 + i + 10j + 100k for i in 1:6, j in 1:5, k in 1:9]
    identity = Dict{String,Any}("id_$i" => 1000.0 + i + 0.123456789 for i in 1:30)  # dense attribute storage
    identity["tau"] = 0.6180339887
    identity["label_sentinel_key"] = "LABELVALUESENTINEL"
    identity["flag"] = true
    identity["limiter"] = nothing
    return StateSnapshot(solver="toy_disk_sentinel", schema_version=3, cycle=123456789,
        fields=Dict("block003/f" => a), series=Dict("h" => Float64[1.5, 2.5]),
        scalars=Dict("carried" => 0.5772156649), identity=identity,
        run_control=Dict("max_cycles" => 987654321),
        parameters=Dict("T" => 2.7182818284), derived=Dict("p" => 1.4142135623))
end

@testset "platform state contract" begin
    tmpdir = mktempdir()

    @testset "toy client, $T" for T in (Float64, Float32)
        cfg = (nx=6, ny=5, label="toy run α", limiter=nothing, history_interval=3)
        run_state_contract_suite(ToyState{T};
            make_state=() -> init_state(ToyState{T}; cfg...), restore_kwargs=cfg,
            interrupt! =toy_interrupt!, tmpdir=mktempdir(tmpdir))
    end

    @testset "sampling follows the global cycle, sample_final is opt-in" begin
        s = advance!(init_state(ToyState{Float64}; nx=4, ny=4), 7)
        @test s.hist_cycle == [3, 6]
        advance!(s, 4; sample_final=true)
        @test s.hist_cycle == [3, 6, 9, 11]
    end

    @testset "updatable parameter" begin
        s = init_state(ToyState{Float64}; nx=4, ny=4)
        @test updatable_parameters(ToyState{Float64}).names == [:gain]
        @test update_parameter!(s, :gain, 0.5).gain == 0.5
        @test_throws ArgumentError update_parameter!(s, :nx, 8)
        @test_throws ArgumentError update_parameter!(s, :gain, 2.0)
        # A parameter update survives the round trip (parameters are not compared).
        restored = restore_state(ToyState{Float64}, export_state(s); nx=4, ny=4)
        @test restored.gain == 0.5
    end

    @testset "defaults for an unimplemented client" begin
        @test_throws ErrorException init_state(BareState)
        @test_throws ErrorException advance!(BareState(), 1)
        @test_throws ErrorException solution(BareState())
        @test_throws ErrorException snapshot(BareState())
        @test_throws ErrorException export_state(BareState())
        @test_throws ErrorException update_parameter!(BareState(), :x, 1.0)
        snap = StateSnapshot(solver="bare", schema_version=1, cycle=0)
        @test_throws ErrorException restore_state(BareState, snap)
        @test at_boundary(BareState())
        @test validate_snapshot(BareState, snap) === nothing
        @test isempty(updatable_parameters(BareState).names)
        @test startswith(_error_message(() -> migrate(BareState, snap, 0)), "schema_version:")
    end

    @testset "snapshot constructor rejects foreign values, naming the key" begin
        make(; kw...) = StateSnapshot(; solver="t", schema_version=1, cycle=0, kw...)
        @test occursin("identity/scheme", _error_message(() -> make(identity=Dict("scheme" => :bgk))))
        @test occursin("scalars/x", _error_message(() -> make(scalars=Dict("x" => nothing))))
        @test occursin("derived/x", _error_message(() -> make(derived=Dict("x" => nothing))))
        @test occursin("scalars/n", _error_message(() -> make(scalars=Dict("n" => Int32(1)))))
        @test occursin("parameters/v", _error_message(() -> make(parameters=Dict("v" => [1.0]))))
        @test occursin("fields/h", _error_message(() -> make(fields=Dict("h" => zeros(Float16, 2)))))
        @test occursin("fields/z", _error_message(() -> make(fields=Dict("z" => zeros(ComplexF64, 2)))))
        @test occursin("fields/r", _error_message(() -> make(fields=Dict("r" => 1:3))))
        @test occursin("fields/0d", _error_message(() -> make(fields=Dict("0d" => fill(1.0)))))
        @test occursin("series/m", _error_message(() -> make(series=Dict("m" => zeros(2, 2)))))
        @test occursin("a/b", _error_message(() -> make(identity=Dict("a/b" => 1))))
        @test occursin("keys", _error_message(() -> make(identity=Dict("keys" => 1))))
        @test occursin("\"a//b\"", _error_message(() -> make(fields=Dict("a//b" => zeros(2)))))
        @test occursin("prefix", _error_message(() -> make(fields=Dict("a" => zeros(2), "a/b" => zeros(2)))))
        @test occursin("cycle", _error_message(() -> make(identity=Dict(:cycle => 1))))
        @test make(identity=Dict("x" => nothing), run_control=Dict("x" => nothing),
                   parameters=Dict("x" => nothing)) isa StateSnapshot
    end

    @testset "check_compatible: defaults, missing keys, types" begin
        snap = StateSnapshot(solver="t", schema_version=1, cycle=0, identity=Dict("nx" => 8))
        ok(identity; kw...) = check_compatible(snap; solver="t", schema_version=1, identity=identity, kw...)
        @test ok(Dict("nx" => 8)) === snap
        @test ok(Dict("nx" => 8, "new" => "bgk"); identity_defaults=Dict("new" => "bgk")) === snap
        msg = _error_message(() -> ok(Dict("nx" => 8, "new" => "bgk")))
        @test startswith(msg, "identity/new:") && occursin("missing", msg)
        msg = _error_message(() -> ok(Dict("nx" => 8, "new" => "mrt"); identity_defaults=Dict("new" => "bgk")))
        @test startswith(msg, "identity/new:") && occursin("\"mrt\"", msg) && occursin("\"bgk\"", msg)
        msg = _error_message(() -> ok(Dict("nx" => 8.0)))
        @test startswith(msg, "identity/nx:") && occursin("Float64", msg) && occursin("Int64", msg)
    end

    @testset "disk layer" begin
        dir = mktempdir(tmpdir)
        path = joinpath(dir, "ck.h5")
        snap = disk_snapshot()
        write_checkpoint(path, snap)

        @testset "layout: superblock with metadata checksums" begin
            @test read(path)[1:8] == UInt8[0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a]
            @test read(path)[9] >= 2       # superblock version; v0 has no metadata checksum
            info = checkpoint_info(path)
            @test info.solver == "toy_disk_sentinel" && info.cycle == 123456789
            @test info.container_version == CHECKPOINT_CONTAINER_VERSION == 1
            @test info.julia_version == string(VERSION) && info.created_unix <= time()
            @test isempty(snapshot_differences(read_checkpoint(path), snap))
        end

        @testset "payload bit flip rejected" begin
            r = count_refused_flips(path, snap.fields["block003/f"][100])
            @test r.hits >= 1 && r.refused == r.hits
            r = count_refused_flips(path, snap.series["h"][2]; byte=6, bit=0x80)
            @test r.hits >= 1 && r.refused == r.hits
        end

        # Fails without libver_bounds=(1.10, 1.10): superblock v0 has no metadata
        # checksum and the flipped attribute reads back silently.
        @testset "attribute bit flip rejected: $label" for (label, value) in (
                "root Int (cycle)" => 123456789,
                "root String (solver)" => "toy_disk_sentinel",
                "identity Float64 (tau)" => 0.6180339887,
                "identity Float64, dense storage (id_17)" => 1000.0 + 17 + 0.123456789,
                "identity String value" => "LABELVALUESENTINEL",
                "key name and key list" => "label_sentinel_key",
                "scalars Float64" => 0.5772156649,
                "run_control Int" => 987654321,
                "parameters Float64" => 2.7182818284,
                "derived Float64" => 1.4142135623)
            r = count_refused_flips(path, value)
            @test r.hits >= 1
            @test r.refused == r.hits
        end
        @test count_refused_flips(path, "label_sentinel_key").hits >= 2   # attribute name + `keys` list

        @testset "truncated file rejected" begin
            buf = read(path)
            for keep in (0, 7, 100, length(buf) ÷ 2, length(buf) - 1)
                cut = joinpath(dir, "truncated.h5")
                write(cut, buf[1:keep])
                msg = _error_message(() -> read_checkpoint(cut))
                @test occursin(cut, msg) && !startswith(msg, "not a")
            end
            @test occursin("no such", _error_message(() -> read_checkpoint(joinpath(dir, "absent.h5"))))
        end

        @testset "unknown container version and unlisted attribute rejected" begin
            # The only place that touches the storage library directly (through
            # Kraken's own binding, not a test dependency): these two files cannot
            # be produced by damaging bytes, their checksums must stay valid.
            H5 = Kraken.HDF5
            future = joinpath(dir, "future.h5")
            cp(path, future)
            H5.h5open(f -> (H5.delete_attribute(f, "container_version"); H5.attributes(f)["container_version"] = 2), future, "r+")
            @test occursin("container_version: expected 1, found 2", _error_message(() -> read_checkpoint(future)))
            stray = joinpath(dir, "stray.h5")
            cp(path, stray)
            H5.h5open(f -> (H5.attributes(f["identity"])["smuggled"] = 1), stray, "r+")
            @test occursin("/identity@smuggled", _error_message(() -> read_checkpoint(stray)))
        end

        @testset "shape and eltype mismatch rejected through the client hook" begin
            cfg = (nx=6, ny=5)
            good = export_state(advance!(init_state(ToyState{Float64}; cfg...), 4))
            file = joinpath(dir, "mismatch.h5")
            for (fields, expected) in (
                    merge(good.fields, Dict("a" => zeros(6, 4))) => "fields/a: expected size (6, 5), found (6, 4)",
                    merge(good.fields, Dict("a" => zeros(Float32, 6, 5))) => "fields/a: expected element type Float64, found Float32",
                    merge(good.fields, Dict("aux/mask" => zeros(Int32, 6, 5))) => "fields/aux/mask: expected element type Bool, found Int32",
                    filter(p -> p.first != "aux/b", good.fields) => "fields/aux/b: missing")
                write_checkpoint(file, tampered(good; fields=Dict{String,Array}(fields)))
                @test _error_message(() -> load_checkpoint(ToyState{Float64}, file; cfg...)) == expected
            end
            # Float32 file into a Float64 simulation: refused on identity before shapes.
            write_checkpoint(file, export_state(init_state(ToyState{Float32}; cfg...)))
            @test startswith(_error_message(() -> load_checkpoint(ToyState{Float64}, file; cfg...)), "identity/FT:")
        end

        @testset "rotation: .prev holds the previous generation, no .tmp left" begin
            rot = joinpath(mktempdir(tmpdir), "rot.h5")
            first_, second, third = (tampered(snap; cycle=c) for c in (1, 2, 3))
            write_checkpoint(rot, first_)
            @test !ispath(rot * ".prev")
            write_checkpoint(rot, second)
            @test read_checkpoint(rot).cycle == 2 && read_checkpoint(rot * ".prev").cycle == 1
            write_checkpoint(rot, third)
            @test read_checkpoint(rot).cycle == 3 && read_checkpoint(rot * ".prev").cycle == 2
            write_checkpoint(rot, first_; keep_previous=false)
            @test read_checkpoint(rot).cycle == 1 && read_checkpoint(rot * ".prev").cycle == 2
            @test sort(readdir(dirname(rot))) == ["rot.h5", "rot.h5.prev"]
        end

        @testset "non-finite state refused, good checkpoint untouched" begin
            keep = joinpath(mktempdir(tmpdir), "keep.h5")
            s = advance!(init_state(ToyState{Float64}; nx=6, ny=5), 4)
            save_checkpoint(keep, s)
            advance!(s, 1)
            save_checkpoint(keep, s)
            before, before_prev = read(keep), read(keep * ".prev")
            s.b[2, 3, 1] = NaN
            msg = _error_message(() -> save_checkpoint(keep, s))
            @test occursin("fields/aux/b", msg) && occursin("(2, 3, 1)", msg)
            for (class, bad) in ("fields" => Dict("a" => [1.0, Inf]), "series" => Dict("s" => Float32[NaN]))
                poisoned = StateSnapshot(; solver="toy", schema_version=1, cycle=9, Symbol(class) => bad)
                @test occursin("$class/$(first(keys(bad)))", _error_message(() -> write_checkpoint(keep, poisoned)))
            end
            @test read(keep) == before && read(keep * ".prev") == before_prev
            @test sort(readdir(dirname(keep))) == ["keep.h5", "keep.h5.prev"]
            # Inf is legitimate in a carried scalar (an indicator before its first sample)...
            unsampled = StateSnapshot(solver="toy", schema_version=1, cycle=0, scalars=Dict("rel_last" => Inf))
            @test read_checkpoint(write_checkpoint(joinpath(dir, "unsampled.h5"), unsampled)).scalars["rel_last"] == Inf
            # ...and in the configuration (an open horizon).
            open_horizon = StateSnapshot(solver="toy", schema_version=1, cycle=0, run_control=Dict("t_end" => Inf))
            @test read_checkpoint(write_checkpoint(joinpath(dir, "inf.h5"), open_horizon)).run_control["t_end"] == Inf
        end

        @testset "bitwise round trip of every allowed type" begin
            tiny32, tiny64 = nextfloat(0.0f0), nextfloat(0.0)
            full = StateSnapshot(solver="types", schema_version=1, cycle=0,
                fields=Dict("f32" => Float32[-0.0f0 tiny32; floatmin(Float32) / 2 -tiny32],
                            "f64" => Float64[-0.0, tiny64, floatmin(Float64) / 2, floatmax(Float64)],
                            "i32" => Int32[typemin(Int32), -1, typemax(Int32)],
                            "i64" => Int64[typemin(Int64), 0, typemax(Int64)],
                            "deep/er/mask" => Bool[true false; false true; true true],
                            "f4d" => reshape(collect(Float32, 1:120), 2, 3, 4, 5),
                            "long" => collect(Float64, 1:200_000),          # several chunks
                            "wide" => reshape(collect(Float64, 1:300_000), 1000, 300),
                            "empty2d" => zeros(Float32, 0, 3)),
                series=Dict("empty_f64" => Float64[], "empty_bool" => Bool[], "empty_i32" => Int32[],
                            "cycles" => Int64[5, 10]),
                scalars=Dict("negzero" => -0.0, "tiny32" => tiny32, "flag" => false, "count" => typemin(Int),
                             "name" => "état 状態"),
                identity=Dict("none" => nothing, "empty" => "", "spaced" => " a b ", "f32" => 1.5f0),
                run_control=Dict("none" => nothing), parameters=Dict("none" => nothing, "T" => 0.1))
            file = write_checkpoint(joinpath(dir, "types.h5"), full)
            back = read_checkpoint(file)
            @test isempty(snapshot_differences(back, full))
            @test signbit(back.fields["f32"][1, 1]) && signbit(back.scalars["negzero"])
            @test back.identity["none"] === nothing && back.identity["empty"] === ""
            @test back.scalars["tiny32"] isa Float32 && back.series["empty_i32"] isa Vector{Int32}
        end
    end
end
