using Kraken, CUDA, CUDSS, KernelAbstractions, TOML, SHA, Serialization, LinearAlgebra, Dates
BLAS.set_num_threads(1)
CUDA.allowscalar(false)
@assert realpath(pathof(Kraken)) == realpath("source/src/Kraken.jl")
@assert readchomp(`git -C source rev-parse HEAD`) == strip(read("SOURCE_SHA", String))
@assert CUDA.functional() && Base.get_extension(Kraken, :KrakenCUDSSExt) !== nothing
save_toml(path, x) = open(io -> TOML.print(io, x), path, "w")
save_bin(path, x) = open(io -> serialize(io, x), path, "w")
maxabs(a) = isempty(a) ? 0.0 : maximum(abs, a)
distance(a, b) = isempty(a) ? 0.0 : maximum(abs.(a .- b))
function fresh(closure, backend=CUDA.CUDABackend())
    s = Kraken.init_state(Kraken.ECState; Nx=10, Ny=16, C=10., M=10., T=190.,
        Ma_E=.01, alpha=1e-4, charge_scheme=:regularized, phi_scheme=:direct,
        ns_scheme=:mrt, injector_closure=closure, backend, FT=Float64, history_interval=1)
    backend isa CPU || @assert s.poisson_setup isa Kraken.EhdPoissonSetupGPU
    return s
end
function sample(s)
    snap = Kraken.export_state(s)
    d = Dict{String,Any}(k => a for (k,a) in snap.fields)
    d["umax"] = copy(snap.series["umax"])
    @assert all(all(isfinite, a) for a in values(d))
    return d
end
function trajectory(s, n)
    samples = [sample(s)]
    for _ in 1:n
        Kraken.advance!(s, 1)
        push!(samples, sample(s))
    end
    return samples
end
function stage_sample(s)
    KernelAbstractions.synchronize(s.backend)
    Dict(string(k) => Array(getproperty(s,k)) for k in
         (:f_in, :q_f_in, :Fx_prev, :Fy_prev, :qfield, :phi, :Ex, :Ey))
end
stage_diff(a,b) = Dict(k => distance(a[k],b[k]) for k in keys(a))
save_toml("raw/runtime_$(ARGS[1]).toml", Dict("utc"=>string(now(UTC)),
    "julia"=>string(VERSION), "source"=>strip(read("SOURCE_SHA",String)),
    "loaded_path"=>pathof(Kraken), "gpu"=>CUDA.name(CUDA.device()),
    "threads"=>Threads.nthreads(), "job_id"=>get(ENV,"PBS_JOBID","UNKNOWN")))
CUDA.versioninfo()
