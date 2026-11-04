include("common.jl")
tol = Dict{String,Any}("status"=>"FROZEN_BEFORE_RESTART_TESTS", "factor"=>2,
    "repeats"=>8, "cycles"=>20, "utc"=>string(now(UTC)),
    "source"=>strip(read("SOURCE_SHA",String)))
allcycles = Dict{String,Any}()
for closure in (:nee, :moment)
    traces = Any[]
    for repeat in 1:8
        s = fresh(closure)
        save_bin("raw/identity_$(closure)_$(repeat).bin", Kraken.export_state(s))
        trace = trajectory(s,20)
        save_bin("raw/calibration_$(closure)_$(repeat).bin",trace)
        push!(traces,trace)
        println("CALIBRATION ",closure," ",repeat,"/8"); flush(stdout)
    end
    limits = Dict{String,Any}()
    for q in keys(traces[1][1])
        errors = Float64[]; scale = 0.0
        for step in 1:21
            arrays = [t[step][q] for t in traces]
            lo = copy(first(arrays)); hi = copy(lo)
            for a in arrays
                lo = min.(lo,a); hi = max.(hi,a)
                scale = max(scale,maxabs(a))
            end
            push!(errors,maxabs(hi .- lo))
        end
        envelope = maximum(errors)
        gate = 2envelope
        @assert gate <= 1e-12*max(1.0,scale) "Calibration exceeds safety ceiling"
        limits[q] = Dict("envelope_abs"=>envelope,"tolerance_abs"=>gate,"signal_scale"=>scale)
        allcycles["$(closure)_$(q)"] = errors
    end
    tol[string(closure)] = limits
    GC.gc(); CUDA.reclaim()
end
@assert !Kraken._EHD_CUDSS_FALLBACK_WARNED[]
save_toml("raw/calibration_by_cycle.toml",allcycles)
save_toml("raw/tolerances.toml",tol)
open("raw/tolerances.sha256","w") do io
    println(io,bytes2hex(sha256(read("raw/tolerances.toml"))),"  raw/tolerances.toml")
end
println("CALIBRATION_SEALED ",bytes2hex(sha256(read("raw/tolerances.toml")))); flush(stdout)
