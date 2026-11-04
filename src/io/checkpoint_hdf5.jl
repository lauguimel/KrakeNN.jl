# ============================================================================
# src/io/checkpoint_hdf5.jl
# On-disk layer of the platform state contract (src/platform/state.jl):
# one HDF5 file per checkpoint, flat schema, no Julia-specific type metadata.
# This is the only file of Kraken that talks to HDF5.
#
# Integrity rests on three measured facts (docs/platform/07-STATE-CONTRACT.md):
#   1. payload: every dataset is chunked and carries the Fletcher32 filter;
#   2. metadata: the file is created with libver_bounds = (1.10, 1.10), which
#      selects superblock v3 and checksummed object headers. With the library
#      default (superblock v0) a flipped bit in an attribute reads back silently;
#   3. variable-length strings live in the global heap, which is NOT checksummed
#      even in the 1.10 format: only scalar (fixed-length) strings are written.
# ============================================================================

import HDF5

"""
    CHECKPOINT_CONTAINER_VERSION

Version of the file *layout* (groups, attribute encoding), independent of any
solver. A client's own set of keys is versioned separately by the
`schema_version` of its [`StateSnapshot`](@ref). Files with another container
version are rejected by [`read_checkpoint`](@ref).
"""
const CHECKPOINT_CONTAINER_VERSION = 1

const _CKPT_LIBVER = (v"1.10", v"1.10")
const _CKPT_KEYS_ATTR = "keys"
const _CKPT_ARRAY_GROUPS = ("fields", "series")
const _CKPT_SCALAR_GROUPS = ("scalars", "identity", "run_control", "parameters", "derived")
const _CKPT_VECTOR_CHUNK = 65_536          # elements per chunk of a 1D dataset
const _CKPT_MATRIX_CHUNK_BYTES = 1 << 20   # target chunk size of a 2D dataset
const _CKPT_MAX_CHUNK_BYTES = 1 << 30      # HDF5 caps a chunk at 4 GiB; stay far below

"""
    _checkpoint_chunk(dims, elsize) -> Dims

Chunk shape of a dataset. Three dimensions and more: one slab along the last
dimension (`(Nx, Ny, 1)`, `(Nx, Ny, Nz, 1)`), i.e. one chunk per population.
Matrices: whole columns, grouped up to about 1 MiB. Vectors: at most 65 536
elements. Every extent is at least 1 (empty arrays), and a slab larger than 1 GiB
is halved along its trailing dimensions until it fits.
"""
function _checkpoint_chunk(dims::Dims{N}, elsize::Int) where {N}
    c = collect(max.(dims, 1))
    if N == 1
        c[1] = min(c[1], _CKPT_VECTOR_CHUNK)
    elseif N == 2
        c[2] = clamp(fld(_CKPT_MATRIX_CHUNK_BYTES, c[1] * elsize), 1, c[2])
    else
        c[N] = 1
    end
    while prod(c) * elsize > _CKPT_MAX_CHUNK_BYTES
        i = findlast(>(1), c)
        c[i] = cld(c[i], 2)
    end
    return Tuple(c)
end

function _write_array(root::HDF5.Group, name::String, a::Array)
    segments = String.(split(name, '/'))
    g = root
    for seg in segments[1:end-1]
        g = haskey(g, seg) ? g[seg] : HDF5.create_group(g, seg)
    end
    g[segments[end], chunk=_checkpoint_chunk(size(a), sizeof(eltype(a))), fletcher32=true] = a
    return nothing
end

function _write_scalars(g::HDF5.Group, d::Dict{String,Any})
    names = sort!(collect(keys(d)))
    attrs = HDF5.attributes(g)
    # One newline-joined scalar string, not a string array: see header, fact 3.
    attrs[_CKPT_KEYS_ATTR] = join(names, '\n')
    for name in names
        d[name] === nothing && continue      # `nothing` = listed in `keys`, attribute absent
        attrs[name] = d[name]
    end
    return nothing
end

_short(err) = first(sprint(showerror, err), 400)

"""
    write_checkpoint(path, snap::StateSnapshot; keep_previous=true) -> path

Write `snap` as one HDF5 file at `path`.

The snapshot is validated first and refused if a field or series holds a non-finite value
([`check_finite`](@ref)): a diverged state never replaces a good restart point. The
file is written to `path * ".tmp"` in the same directory, closed, then moved into
place with an atomic `rename`. With `keep_previous=true` an existing `path` is
first renamed to `path * ".prev"`, so one older generation always survives; during
that short window the newest complete checkpoint is the `.prev` file.

# Layout (container version 1)
- root attributes: `container_version`, `schema_version`, `solver`, `cycle`,
  `kraken_version`, `julia_version`, `created_unix`;
- `/fields`, `/series`: one chunked, Fletcher32-checksummed dataset per entry; a
  `/` in a name maps to nested groups (`/fields/block003/f`);
- `/scalars`, `/identity`, `/run_control`, `/parameters`, `/derived`: one attribute
  per entry, plus a `keys` attribute listing every key (newline-separated). A key
  listed in `keys` with no attribute of that name holds `nothing`.

# Reading from Python
`h5py` opens these files directly. Arrays appear with **reversed index order**
(Julia is column-major: a Julia `(Nx, Ny, Q)` array is seen as `(Q, Ny, Nx)`), and
string attributes come back as `bytes` (`.decode()` them; split `keys` on `"\\n"`).
"""
function write_checkpoint(path::AbstractString, snap::StateSnapshot; keep_previous::Bool=true)
    validate_content(snap)
    check_finite(snap)
    tmp = path * ".tmp"
    try
        HDF5.h5open(tmp, "w"; libver_bounds=_CKPT_LIBVER) do file
            attrs = HDF5.attributes(file)
            attrs["container_version"] = CHECKPOINT_CONTAINER_VERSION
            attrs["schema_version"] = snap.schema_version
            attrs["solver"] = snap.solver
            attrs["cycle"] = snap.cycle
            attrs["kraken_version"] = string(something(pkgversion(@__MODULE__), "unknown"))
            attrs["julia_version"] = string(VERSION)
            attrs["created_unix"] = time()
            for class in _CKPT_ARRAY_GROUPS
                g = HDF5.create_group(file, class)
                d = getfield(snap, Symbol(class))
                for name in sort!(collect(keys(d)))
                    _write_array(g, name, d[name])
                end
            end
            for class in _CKPT_SCALAR_GROUPS
                _write_scalars(HDF5.create_group(file, class), getfield(snap, Symbol(class)))
            end
        end
    catch err
        rm(tmp; force=true)
        err isa InterruptException && rethrow()
        throw(CheckpointError("$tmp: write failed, $(repr(String(path))) left untouched: $(_short(err))"))
    end
    # Two renames, never `mv(...; force=true)`: on Julia 1.11 `mv` removes the
    # destination before renaming, leaving a window with no checkpoint at all.
    keep_previous && ispath(path) && Base.rename(path, path * ".prev")
    Base.rename(tmp, path)
    return path
end

# Run `f`; any failure of the storage library becomes a CheckpointError naming
# the file and the object being read.
function _guard(f, path::AbstractString, object::AbstractString)
    try
        return f()
    catch err
        (err isa CheckpointError || err isa InterruptException) && rethrow()
        throw(CheckpointError("$path: cannot read $object (damaged, truncated or foreign file): $(_short(err))"))
    end
end

function _read_arrays!(out::Dict, g::HDF5.Group, prefix::String, path, class::String)
    for name in _guard(() -> keys(g), path, "group /$class/$prefix")
        full = isempty(prefix) ? name : prefix * "/" * name
        object = "/$class/$full"
        obj = _guard(() -> g[name], path, object)
        if obj isa HDF5.Group
            _read_arrays!(out, obj, full, path, class)
        elseif obj isa HDF5.Dataset
            a = _guard(() -> read(obj), path, "dataset $object")
            (a isa Array && eltype(a) in SNAPSHOT_ELTYPES) ||
                throw(CheckpointError("$path: dataset $object has unsupported type $(typeof(a))"))
            class == "series" && ndims(a) != 1 &&
                throw(CheckpointError("$path: dataset $object must be a vector, found size $(size(a))"))
            out[full] = a
        else
            throw(CheckpointError("$path: object $object is neither a group nor a dataset"))
        end
    end
    return out
end

function _read_scalars(file::HDF5.File, class::String, path)
    g = _guard(() -> file[class], path, "group /$class")
    attrs = HDF5.attributes(g)
    joined = _guard(() -> HDF5.read_attribute(g, _CKPT_KEYS_ATTR), path, "attribute /$class@$(_CKPT_KEYS_ATTR)")
    joined isa String ||
        throw(CheckpointError("$path: attribute /$class@$(_CKPT_KEYS_ATTR) must be a string, found $(typeof(joined))"))
    listed = String.(split(joined, '\n'; keepempty=false))
    out = Dict{String,Any}()
    for key in listed
        out[key] = if _guard(() -> haskey(attrs, key), path, "attribute /$class@$key")
            _guard(() -> HDF5.read_attribute(g, key), path, "attribute /$class@$key")
        else
            nothing
        end
    end
    for key in _guard(() -> keys(attrs), path, "attributes of /$class")
        (key == _CKPT_KEYS_ATTR || haskey(out, key)) ||
            throw(CheckpointError("$path: attribute /$class@$key is not listed in /$class@$(_CKPT_KEYS_ATTR)"))
    end
    return out
end

function _read_root(file::HDF5.File, path)
    version = _guard(() -> HDF5.read_attribute(file, "container_version"), path, "attribute /@container_version")
    version == CHECKPOINT_CONTAINER_VERSION ||
        throw(CheckpointError("$path: container_version: expected $(CHECKPOINT_CONTAINER_VERSION), found $(repr(version))"))
    root = Dict{String,Any}()
    for key in ("solver", "schema_version", "cycle", "kraken_version", "julia_version", "created_unix")
        root[key] = _guard(() -> HDF5.read_attribute(file, key), path, "attribute /@$key")
    end
    (root["solver"] isa String && root["schema_version"] isa Int && root["cycle"] isa Int) ||
        throw(CheckpointError("$path: root attributes solver / schema_version / cycle have unexpected types"))
    return root
end

function _open_checkpoint(f, path)
    isfile(path) || throw(CheckpointError("$path: no such checkpoint file"))
    return _guard(() -> HDF5.h5open(f, path, "r"), path, "the file")
end

"""
    read_checkpoint(path) -> StateSnapshot

Read a file written by [`write_checkpoint`](@ref). Every failure is a
[`CheckpointError`](@ref) naming `path` and the failing object: truncated or
foreign file, checksum failure in a dataset (Fletcher32) or in the metadata
(attributes, groups), unknown `container_version`, missing group, unlisted
attribute, unsupported type. Compatibility with the simulation being configured
is *not* judged here; that is [`check_compatible`](@ref), called by the client's
[`restore_state`](@ref).
"""
function read_checkpoint(path::AbstractString)
    return _open_checkpoint(path) do file
        root = _read_root(file, path)
        arrays = map(_CKPT_ARRAY_GROUPS) do class
            g = _guard(() -> file[class], path, "group /$class")
            _read_arrays!(Dict{String,Array}(), g, "", path, class)
        end
        scalars = map(class -> _read_scalars(file, class, path), _CKPT_SCALAR_GROUPS)
        try
            StateSnapshot(root["solver"], root["schema_version"], root["cycle"],
                          arrays[1], Dict{String,Vector}(arrays[2]), scalars...)
        catch err
            err isa CheckpointError || rethrow()
            throw(CheckpointError("$path: invalid content: $(err.msg)"))
        end
    end
end

"""
    checkpoint_info(path) -> NamedTuple

Root attributes of a checkpoint, without reading any array: `container_version`,
`solver`, `schema_version`, `cycle`, `kraken_version`, `julia_version`,
`created_unix` (seconds since the Unix epoch). Same error policy as
[`read_checkpoint`](@ref).
"""
function checkpoint_info(path::AbstractString)
    root = _open_checkpoint(file -> _read_root(file, path), path)
    return (container_version=CHECKPOINT_CONTAINER_VERSION, solver=root["solver"],
            schema_version=root["schema_version"], cycle=root["cycle"],
            kraken_version=root["kraken_version"], julia_version=root["julia_version"],
            created_unix=root["created_unix"])
end

"""
    save_checkpoint(path, state; keep_previous=true) -> path

[`export_state`](@ref) then [`write_checkpoint`](@ref): refuses a state that is
not at a cycle boundary or that holds non-finite values, and leaves any existing
checkpoint at `path` untouched when it refuses.
"""
save_checkpoint(path::AbstractString, state::AbstractSimulationState; kwargs...) =
    write_checkpoint(path, export_state(state); kwargs...)

"""
    load_checkpoint(::Type{S}, path; backend, kwargs...) -> S

[`read_checkpoint`](@ref) then the client's [`restore_state`](@ref), which checks
compatibility and shapes before allocating. `kwargs` are passed to `restore_state`.
"""
load_checkpoint(::Type{S}, path::AbstractString; kwargs...) where {S<:AbstractSimulationState} =
    restore_state(S, read_checkpoint(path); kwargs...)
