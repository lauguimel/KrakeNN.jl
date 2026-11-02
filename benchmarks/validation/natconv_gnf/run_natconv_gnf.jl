using Kraken

"""Prepared CUDA sweep; importing this file does not load CUDA or run a case."""
function run_natconv_gnf_validation(; N=256, max_steps=2_000_000,
        output=joinpath(@__DIR__, "results.csv"), log_path=joinpath(dirname(output), "run.log"))
    N >= 256 || throw(ArgumentError("This GPU validation requires N >= 256"))
    Base.find_package("CUDA") === nothing && error("CUDA must already be installed")
    @eval import CUDA
    return Base.invokelatest(_run_natconv_gnf_cuda; N, max_steps, output, log_path)
end

function _run_natconv_gnf_cuda(; N, max_steps, output, log_path)
    CUDA.functional() || error("A functional CUDA device is required")
    backend = CUDA.CUDABackend()
    nu0, Pr = 0.05, 0.71
    hardware = (; device=CUDA.name(CUDA.device()), Julia=string(VERSION), precision="Float64")
    @info "GNF validation hardware" hardware
    open(output, "w") do io
      open(log_path, "w") do log
        println(log, "GNF validation hardware: ", hardware)
        println(io, "n,Ra,N,steps,Nu,wall_seconds,clamped_fraction,converged,Nu_drift")
        for Ra in (1e4, 1e5), n in (0.6, 0.8, 1.0, 1.2, 1.4, 1.8)
            rheology = cavity_powerlaw(n; nu0, alpha=nu0/Pr, H=N,
                                        nu_min=1e-4, nu_max=1.0)
            started = time()
            result = run_natural_convection_gnf_2d(; N, Ra, Pr, nu0, rheology,
                max_steps, backend, FT=Float64, check_every=5000,
                convergence_rtol=1e-6, min_steps=50000)
            elapsed = time() - started
            metrics = (; n, Ra, N, steps=result.steps, Nu=result.Nu, wall_seconds=elapsed,
                clamped_fraction=result.clamped_fraction, converged=result.converged,
                Nu_drift=result.Nu_drift)
            @info "GNF cavity" metrics
            println(log, "GNF cavity: ", metrics)
            flush(log)
            println(io, join((n, Ra, N, result.steps, result.Nu, elapsed,
                result.clamped_fraction, result.converged, result.Nu_drift), ','))
            flush(io)
        end
      end
    end
    return output
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_natconv_gnf_validation()
end
