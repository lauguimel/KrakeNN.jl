#!/usr/bin/env julia
"""
Natural convection GPU benchmark on HPC (CUDA H100).
Refinement study: N ∈ {128, 256, 512}, Ra ∈ {1e3, 1e4, 1e5}, Rc=1.
Properly scaled steps: dt ~ dx² → steps ~ N².

Also runs CPU comparison for N=128 to measure GPU speedup.
"""

using KrakeNN
using KernelAbstractions
using CUDA
using Printf
using DelimitedFiles

const DE_VAHL_DAVIS = Dict(1e3 => 1.118, 1e4 => 2.243, 1e5 => 4.519, 1e6 => 8.800)

# Steps scale as N² (diffusive scaling). Base calibrated at N=128.
function nsteps(N, Ra)
    base_128 = Ra <= 1e3 ? 40_000 : (Ra <= 1e4 ? 80_000 : 160_000)
    return round(Int, base_128 * (N / 128)^2)
end

function run_study()
    println("\n  KrakeNN — Natural Convection GPU Benchmark (CUDA)\n")
    println("  GPU: ", CUDA.name(CUDA.device()))
    println()

    Ra_values = [1e3, 1e4, 1e5]
    N_values  = [128, 256, 512]
    Pr = 0.71
    Rc = 1.0

    results = []

    # --- GPU runs ---
    println("=" ^ 80)
    println("  GPU (CUDA/Float64) Refinement — Pr=$Pr, Rc=$Rc")
    println("=" ^ 80)
    @printf("  %-6s  %-8s  %-10s  %-10s  %-8s  %-10s  %-8s\n",
            "N", "Ra", "Nu (GPU)", "Nu (ref)", "Err %", "Time (s)", "Steps")
    println("  " * "-" ^ 66)

    for Ra in Ra_values
        Nu_ref = DE_VAHL_DAVIS[Ra]
        for N in N_values
            steps = nsteps(N, Ra)

            t = @elapsed begin
                r = run_natural_convection_2d(; N=N, Ra=Ra, Pr=Pr, Rc=Rc,
                                                max_steps=steps,
                                                backend=CUDABackend(),
                                                FT=Float64)
            end
            err = 100 * abs(r.Nu - Nu_ref) / Nu_ref

            @printf("  %-6d  %-8.0e  %-10.4f  %-10.3f  %-8.2f  %-10.1f  %-8d\n",
                    N, Ra, r.Nu, Nu_ref, err, t, steps)
            push!(results, (N=N, Ra=Ra, Nu=r.Nu, Nu_ref=Nu_ref, err=err, time=t,
                           steps=steps, backend="GPU"))
        end
        println()
    end

    # --- CPU comparison at N=128 ---
    println("=" ^ 80)
    println("  CPU (Float64) comparison — N=128, Pr=$Pr, Rc=$Rc")
    println("=" ^ 80)
    @printf("  %-8s  %-10s  %-10s  %-8s\n", "Ra", "Nu (CPU)", "Time (s)", "Speedup")
    println("  " * "-" ^ 42)

    for Ra in Ra_values
        steps = nsteps(128, Ra)

        t_cpu = @elapsed begin
            r_cpu = run_natural_convection_2d(; N=128, Ra=Ra, Pr=Pr, Rc=Rc,
                                                max_steps=steps,
                                                backend=KernelAbstractions.CPU())
        end

        # Find matching GPU result
        gpu_match = filter(x -> x.N == 128 && x.Ra == Ra, results)
        t_gpu = isempty(gpu_match) ? NaN : gpu_match[1].time
        speedup = t_cpu / t_gpu

        @printf("  %-8.0e  %-10.4f  %-10.1f  %-8.1fx\n", Ra, r_cpu.Nu, t_cpu, speedup)
        push!(results, (N=128, Ra=Ra, Nu=r_cpu.Nu, Nu_ref=DE_VAHL_DAVIS[Ra],
                       err=100*abs(r_cpu.Nu-DE_VAHL_DAVIS[Ra])/DE_VAHL_DAVIS[Ra],
                       time=t_cpu, steps=steps, backend="CPU"))
    end

    # --- Save CSV ---
    open("natconv_results.csv", "w") do io
        println(io, "backend,N,Ra,Pr,Rc,Nu,Nu_ref,err_pct,time_s,steps")
        for r in results
            @printf(io, "%s,%d,%.0e,%.2f,%.1f,%.6f,%.3f,%.4f,%.1f,%d\n",
                    r.backend, r.N, r.Ra, Pr, Rc, r.Nu, r.Nu_ref, r.err, r.time, r.steps)
        end
    end
    println("\n  Results saved to natconv_results.csv")
    println()
end

run_study()
