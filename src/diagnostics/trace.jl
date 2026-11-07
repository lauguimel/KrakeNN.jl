module Diagnostics

export @trace_enter

const TRACE_ENABLED = haskey(ENV, "KRAKEN_TRACE")
const _TRACE_LOCK = ReentrantLock()

function _json_escape(s::AbstractString)
    io = IOBuffer()
    for c in s
        if c == '"'
            print(io, "\\\"")
        elseif c == '\\'
            print(io, "\\\\")
        elseif c == '\n'
            print(io, "\\n")
        elseif c == '\r'
            print(io, "\\r")
        elseif c == '\t'
            print(io, "\\t")
        elseif c < Char(0x20)
            print(io, "\\u", lpad(string(UInt32(c), base=16), 4, '0'))
        else
            print(io, c)
        end
    end
    return String(take!(io))
end

@inline function _hash_arg!(h::UInt, name::Symbol, value)
    h = hash(name, h)
    h = hash(typeof(value), h)
    if value isa AbstractArray
        h = hash(size(value), h)
        h = hash(eltype(value), h)
    elseif value isa Number || value isa Symbol || value isa Bool
        h = hash(value, h)
    elseif value isa Tuple || value isa NamedTuple
        h = hash(length(value), h)
    end
    return h
end

function _args_hash(locals)
    h = UInt(0x6b72616b656e7472)
    for (name, value) in pairs(locals)
        h = _hash_arg!(h, name, value)
    end
    return lpad(string(UInt64(h), base=16), 16, '0')
end

function trace_enter_impl(kernel::Symbol, locals)
    t_ns = time_ns()
    args_hash = _args_hash(locals)
    path = get(ENV, "KRAKEN_TRACE_FILE", joinpath(".engineer_logs", "trace.jsonl"))
    dir = dirname(path)
    lock(_TRACE_LOCK)
    try
        isempty(dir) || mkpath(dir)
        open(path, "a") do io
            print(io, "{\"t_ns\":", t_ns,
                ",\"kernel\":\"", _json_escape(String(kernel)),
                "\",\"args_hash\":\"", args_hash,
                "\",\"extras\":{}}\n")
        end
    finally
        unlock(_TRACE_LOCK)
    end
    return nothing
end

macro trace_enter(kernel_id)
    id = kernel_id isa QuoteNode ? kernel_id.value : kernel_id
    id isa Symbol || error("@trace_enter expects a literal Symbol, e.g. @trace_enter :kernel_id")
    TRACE_ENABLED || return :(nothing)
    return esc(quote
        KrakeNN.Diagnostics.trace_enter_impl($(QuoteNode(id)), Base.@locals)
        nothing
    end)
end

end # module Diagnostics

using .Diagnostics: @trace_enter
