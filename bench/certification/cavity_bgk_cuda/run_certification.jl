# =============================================================================
# E1 certification benchmark — single-phase BGK D2Q9 throughput (Taylor-Green)
# =============================================================================
#
# Measures sustained MLUPS (Mega Lattice Updates Per Second) of the KrakeNN
# Newtonian BGK D2Q9 solver, driving the canonical periodic Taylor-Green vortex
# (`run_taylor_green_2d`, src/drivers/basic.jl:281).
#
# MLUPS formula
# -------------
#     MLUPS = (N * N * measured_steps) / wallclock_s / 1e6
#
# i.e. the number of lattice-cell updates performed during the *measured*
# window, divided by the wall-clock time of that window, scaled to millions.
#
# Warm-up is EXCLUDED from the timing: a separate WARMUP-step call is run first
# (and its result + time discarded) so that JIT compilation, kernel
# specialisation and the first GPU allocation/launch latencies do not pollute
# the steady-state throughput measurement. Only the subsequent STEPS-step call
# is timed and reported.
#
# NOTE: `run_cavity_2d` (lid-driven cavity) is the alternative single-phase
# driver; we use `run_taylor_green_2d` here because its periodic BCs give a
# pure, BC-overhead-free interior-kernel throughput figure.
#
# Configuration is entirely env-driven (see below) so the same script runs
# unchanged on CPU locally and on CUDA/Metal GPUs (Aqua H100/A100).
# =============================================================================

using KrakeNN
using KernelAbstractions

# --- backend resolution ------------------------------------------------------
# Resolve a KernelAbstractions backend from KRK_BENCH_BACKEND ("cpu"/"metal"/
# "cuda"). GPU backends are optional dependencies, so they are loaded lazily and
# instantiated through `Base.invokelatest` (the package may be `using`-ed after
# this module was first parsed).
function resolve_backend(name::AbstractString)
    n = lowercase(strip(name))
    if n == "cpu"
        return CPU()
    elseif n == "cuda"
        @eval Main using CUDA
        return Base.invokelatest(getfield(Main, :CUDABackend))
    elseif n == "metal"
        @eval Main using Metal
        return Base.invokelatest(getfield(Main, :MetalBackend))
    else
        error("KRK_BENCH_BACKEND=$(name) not understood (use cpu/metal/cuda)")
    end
end

# --- env config --------------------------------------------------------------
backend_name = get(ENV, "KRK_BENCH_BACKEND", "cpu")
Ns           = parse.(Int, split(get(ENV, "KRK_BENCH_N", "1024,2048"), ","))
steps        = parse(Int, get(ENV, "KRK_BENCH_STEPS", "1000"))
warmup       = parse(Int, get(ENV, "KRK_BENCH_WARMUP", "100"))
outpath      = get(ENV, "KRK_BENCH_OUT", "benchmarks/results/certification_h100.csv")

# Precision: Float64 by default (this gate is an F64 roofline comparison).
const T = Float64

backend = resolve_backend(backend_name)

# Physical params (irrelevant to throughput, kept stable across N).
const ν  = 0.01
const u0 = 0.05

# --- CSV setup ---------------------------------------------------------------
mkpath(dirname(outpath))
write_header = !isfile(outpath) || filesize(outpath) == 0
io = open(outpath, "a")
if write_header
    println(io, "N,steps,backend,precision,wallclock_s,MLUPS")
    flush(io)
end

println("E1 certification — backend=$(backend_name) precision=$(T) steps=$(steps) warmup=$(warmup)")
println("output -> $(outpath)")

for N in Ns
    println("--- N = $(N) ---")

    # Warm-up: timed separately and discarded (JIT + first-launch latencies).
    twarm = @elapsed run_taylor_green_2d(; N=N, ν=ν, u0=u0,
                                          max_steps=warmup, backend=backend, T=T)
    println("  warm-up ($(warmup) steps) discarded: $(round(twarm, digits=3)) s")

    # Measured window.
    wallclock_s = @elapsed run_taylor_green_2d(; N=N, ν=ν, u0=u0,
                                                max_steps=steps, backend=backend, T=T)

    mlups = (N * N * steps) / wallclock_s / 1e6
    println("  measured ($(steps) steps): $(round(wallclock_s, digits=3)) s  ->  $(round(mlups, digits=1)) MLUPS")

    println(io, "$(N),$(steps),$(backend_name),$(T),$(wallclock_s),$(mlups)")
    flush(io)
end

close(io)
println("done.")
