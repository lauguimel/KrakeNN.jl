# Shared utilities for the KrakeNN benchmark suite.
#
# Every benchmark script produces a Vector{NamedTuple} with a stable
# schema and writes it to a timestamped CSV file in benchmarks/results/.
# The same CSVs are consumed by the documentation pipeline
# (docs/src/benchmarks/*.md and the publication figures) so that
# figures stay in sync with the numbers reported in the paper.

using Dates
using Printf

const BENCH_ROOT    = normpath(joinpath(@__DIR__))
const RESULTS_DIR   = joinpath(BENCH_ROOT, "results")

"""
    timestamp() -> String

Return an ISO-like timestamp suitable for filenames, e.g. `"20260410_142300"`.
"""
timestamp() = Dates.format(now(), "yyyymmdd_HHMMSS")

"""
    results_path(name, hardware_id; ext=".csv") -> String

Build a canonical result file path of the form
`benchmarks/results/<name>_<hardware>_<timestamp>.<ext>`.
"""
function results_path(name::AbstractString, hardware_id::AbstractString; ext::AbstractString=".csv")
    isdir(RESULTS_DIR) || mkpath(RESULTS_DIR)
    fname = "$(name)_$(hardware_id)_$(timestamp())$(ext)"
    return joinpath(RESULTS_DIR, fname)
end

"""
    write_csv(path, rows::Vector{<:NamedTuple})

Write a vector of NamedTuples to a CSV file. The column names are taken
from the first row; every subsequent row must have the same fields.
Numeric values are formatted with `@sprintf("%.6g", ...)`. Strings are
written as-is (no quoting — keep values ASCII and comma-free).

Returns the path for convenience.
"""
function write_csv(path::AbstractString, rows::Vector{<:NamedTuple})
    isempty(rows) && return path
    keys_ = collect(propertynames(rows[1]))
    open(path, "w") do io
        println(io, join(String.(keys_), ","))
        for row in rows
            vals = [_fmt(getfield(row, k)) for k in keys_]
            println(io, join(vals, ","))
        end
    end
    return path
end

_fmt(x::AbstractFloat) = @sprintf("%.6g", x)
_fmt(x::Integer)       = string(x)
_fmt(x::AbstractString) = string(x)
_fmt(x::Symbol)         = string(x)
_fmt(x::Bool)           = x ? "true" : "false"
_fmt(x)                 = string(x)

"""
    parse_args(argv=ARGS) -> NamedTuple

Minimal arg parser for the benchmark scripts. Supports:
  --gpu                         enable GPU benchmarks
  --quick                       run a reduced subset (< 5 min CPU)
  --hardware-id=<key>           label for the hardware.toml section
  --skip-existing               skip a case if a matching CSV already exists
  --output-dir=<path>           override the results directory

Unknown flags are ignored so scripts stay forward-compatible.
"""
function parse_args(argv=ARGS)
    gpu            = "--gpu" in argv
    quick          = "--quick" in argv
    skip_existing  = "--skip-existing" in argv
    hardware_id    = _get_kv(argv, "--hardware-id", "apple_m2")
    output_dir     = _get_kv(argv, "--output-dir", RESULTS_DIR)
    return (; gpu, quick, skip_existing, hardware_id, output_dir)
end

function _get_kv(argv, key, default)
    for a in argv
        if startswith(a, key * "=")
            return String(split(a, "="; limit=2)[2])
        end
    end
    return default
end
