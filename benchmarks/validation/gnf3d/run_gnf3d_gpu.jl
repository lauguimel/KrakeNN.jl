#!/usr/bin/env julia
using KrakeNN
using KernelAbstractions
using Printf
using Dates

const _GNF3D_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
isdefined(@__MODULE__, :_gnf3d_reference) ||
    include(joinpath(_GNF3D_ROOT, "test", "analytical", "gnf3d_reference.jl"))

function _gnf3d_options(args)
    smoke = "--smoke" in args
    outdir = joinpath(_GNF3D_ROOT, smoke ? "tmp/m1b/gpu_smoke" :
                                             "benchmarks/validation/gnf3d/results")
    for arg in args
        arg == "--smoke" && continue
        startswith(arg, "--output-dir=") || error("Unknown option: $arg")
        outdir = split(arg, '='; limit=2)[2]
    end
    outdir = abspath(outdir)
    if smoke
        tmp_root = abspath(joinpath(_GNF3D_ROOT, "tmp")) * "/"
        startswith(outdir, tmp_root) || error("Smoke output must be inside $tmp_root")
    end
    return (; smoke, outdir)
end

const _GNF3D_OPTIONS = _gnf3d_options(ARGS)
# This branch is evaluated before main is defined; smoke never imports CUDA.
if !_GNF3D_OPTIONS.smoke
    using CUDA
end

function _gnf3d_log(io, message)
    println(io, message)
    flush(io)
    println(message)
    flush(stdout)
end

function _gnf3d_run_reference(ref, backend, FT, max_steps, tol_ss, check_every)
    return run_gnf_channel_3d(; rheology=ref.rheology, N=ref.N, q_w=ref.q_w,
        F=ref.F, nu_init=ref.nu, u_init=ref.u, u_top=ref.u_top,
        max_steps, tol_ss, check_every, backend, FT)
end

function _gnf3d_bounds(model)
    model isa PowerLaw && return (model.nu_min, model.nu_max)
    return (model.eta_inf, model.eta_0)
end

function _gnf3d_gpu_main(options)
    smoke, outdir = options.smoke, options.outdir
    backend = if smoke
        any(id.name == "CUDA" for id in keys(Base.loaded_modules)) &&
            error("CPU smoke must not load CUDA")
        KernelAbstractions.CPU()
    else
        CUDA.functional() || error("A functional CUDA device is required; use --smoke for CPU IO checks")
        CUDA.CUDABackend()
    end
    mkpath(outdir)
    csv_path, log_path = joinpath(outdir, "gnf3d_order.csv"), joinpath(outdir, "gnf3d.log")
    failures = String[]
    # Identical constitutive laws, cut fractions, and diffusive scaling to the
    # CPU order test. Its four cases pass without smooth-stress companions.
    cases = smoke ? ((:carreau, 0.5, Float64, (4,)),) :
        ((:carreau, 0.5, Float64, (32, 64, 128, 256)),
         (:carreau, 0.3, Float64, (32, 64, 128, 256)),
         # n=0.7 stops at N=128: at N=256 the nearest-centre node needs
         # nu=1.077 > nu_max=1.0, so the unclamped reference would not apply.
         (:power_thin, 0.5, Float64, (32, 64, 128)),
         (:power_thick, 0.5, Float64, (32, 64, 128, 256)),
         (:carreau, 0.5, Float32, (128,)))
    open(log_path, "w") do logio
        _gnf3d_log(logio, "timestamp_utc=$(now(UTC)) julia=$(VERSION) backend=$(typeof(backend)) smoke=$smoke")
        _gnf3d_log(logio, "mode=$(smoke ? "CPU IO smoke only; no numerical validation or performance claim" : "GPU validation")")
        _gnf3d_log(logio, "cuda_loaded=$(any(id.name == "CUDA" for id in keys(Base.loaded_modules)))")
        if !smoke
            CUDA.versioninfo(logio)
            flush(logio)
            _gnf3d_log(logio, "Warm up model/precision kernels before measured runs.")
            for (law, _, FT, _) in cases
                ref = _gnf3d_reference(law, 4, 0.5)
                _gnf3d_run_reference(ref, backend, FT, 2, 0.0, 1)
            end
        end
        open(csv_path, "w") do csvio
            println(csvio, "law,q_w,N,H,FT,err_L2,err_Linf,p_observed,steps,wall_s,MLUPS")
            for (law, q_w, FT, grids) in cases
                previous = nothing
                for N in grids
                    ref = _gnf3d_reference(law, N, q_w)
                    lower, upper = _gnf3d_bounds(ref.rheology)
                    reference_valid = _gnf3d_inside_bounds(ref.nu, lower, upper)
                    # Float32 populations resolve small lattice velocities less
                    # accurately. This tolerance is a convergence diagnostic;
                    # the single Float32 row makes no observed-order claim.
                    tol_ss = FT === Float32 ? 2e-4 : 1e-10
                    max_steps = smoke ? 2 : max(60_000, ceil(Int, 80_000 * (ref.H / 64)^2))
                    check_every = smoke ? 1 : 200
                    KernelAbstractions.synchronize(backend)
                    wall_s = @elapsed out = _gnf3d_run_reference(
                        ref, backend, FT, max_steps, tol_ss, check_every)
                    sp_h, solid_h = Array(out.fields.sp), Array(out.fields.is_solid)
                    nu_h = (inv.(sp_h) .- FT(0.5)) ./ FT(3)
                    fluid_nu = nu_h[.!solid_h]
                    bounds_valid = _gnf3d_inside_bounds(fluid_nu, lower, upper)
                    clamp_active = ref.rheology isa PowerLaw && !bounds_valid
                    finite = all(all(isfinite, Array(a)) for a in
                        (out.fields.f, out.fields.rho, out.fields.ux, out.fields.uy,
                         out.fields.uz, out.fields.sp, out.fields.sm))
                    valid = reference_valid && bounds_valid && finite && out.converged && !smoke
                    err_L2 = valid ? _gnf3d_rel_l2(out.u, ref.u) : NaN
                    err_Linf = valid ? maximum(abs, out.u .- ref.u) / maximum(abs, ref.u) : NaN
                    p = valid && !isnothing(previous) ?
                        log(previous.err / err_L2) / log(ref.H / previous.H) : NaN
                    # End-to-end launched-cell throughput, including setup,
                    # convergence checks and final profile transfer, after JIT warmup.
                    mlups = smoke ? NaN : prod(out.dims) * out.steps / (wall_s * 1e6)
                    label = smoke ? "carreau_smoke" : string(law)
                    println(csvio, join((label, q_w, N, ref.H, FT, err_L2, err_Linf,
                                         p, out.steps, wall_s, mlups), ','))
                    flush(csvio)
                    _gnf3d_log(logio, "law=$law q_w=$q_w N=$N FT=$FT H=$(ref.H) steps=$(out.steps) " *
                        "converged=$(out.converged) tol_ss=$tol_ss check_every=$check_every " *
                        "reference_valid=$reference_valid bounds_valid=$bounds_valid clamp_active=$clamp_active " *
                        "reference_nu_range=$(extrema(ref.nu)) fluid_nu_range=$(extrema(fluid_nu)) finite=$finite")
                    _gnf3d_log(logio, @sprintf("err_L2=%.16e err_Linf=%.16e p_observed=%.10f wall_s=%.6f MLUPS=%.6f",
                                               err_L2, err_Linf, p, wall_s, mlups))
                    if smoke
                        finite || error("CPU smoke produced nonfinite fields")
                    else
                        valid || push!(failures, "$law q_w=$q_w N=$N FT=$FT: invalid reference/bounds, nonfinite, or nonconverged")
                        !isnan(p) && p < 1.8 && push!(failures, "$law q_w=$q_w N=$N: order $p < 1.8")
                    end
                    previous = valid ? (; H=ref.H, err=err_L2) : nothing
                end
            end
        end
        for failure in failures
            _gnf3d_log(logio, "VALIDATION FAILURE: $failure")
        end
        _gnf3d_log(logio, "rows_written=true validation_failures=$(length(failures)) csv=$csv_path")
    end
    isempty(failures) || error("GNF3D validation failed for $(length(failures)) checks; inspect $log_path")
    return nothing
end

_gnf3d_gpu_main(_GNF3D_OPTIONS)
