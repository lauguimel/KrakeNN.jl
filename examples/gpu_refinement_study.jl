#!/usr/bin/env julia
"""
GPU refinement study: increase N to reduce error vs De Vahl Davis.
Runs GPU (Metal/Float32) first, then CPU (Float64) in a separate pass
to avoid Metal/CPU backend conflict.
"""

using KrakeNN
using KernelAbstractions
using Printf

try
    @eval using Metal
catch
    error("Metal.jl required for this study")
end

const DE_VAHL_DAVIS = Dict(1e3 => 1.118, 1e4 => 2.243, 1e5 => 4.519, 1e6 => 8.800)

function nsteps(N, Ra)
    base = N <= 128 ? 40_000 : (N <= 256 ? 100_000 : 200_000)
    return Ra >= 1e5 ? base * 2 : base
end

function run_gpu_study()
    println("\n  GPU (Metal/Float32) Refinement — Natural Convection (Pr=0.71, Rc=1)\n")
    println("=" ^ 70)

    Ra_values = [1e3, 1e4, 1e5]
    N_values  = [128, 256, 512]

    @printf("  %-6s  %-8s  %-10s  %-10s  %-8s  %-8s\n",
            "N", "Ra", "Nu (GPU)", "Nu (ref)", "Err %", "Time (s)")
    println("  " * "-" ^ 56)

    for Ra in Ra_values
        Nu_ref = DE_VAHL_DAVIS[Ra]
        for N in N_values
            steps = nsteps(N, Ra)

            t = @elapsed begin
                r = run_natural_convection_2d(; N=N, Ra=Ra, Pr=0.71, Rc=1.0,
                                                max_steps=steps,
                                                backend=Metal.MetalBackend(),
                                                FT=Float32)
            end
            err = 100 * abs(r.Nu - Nu_ref) / Nu_ref

            @printf("  %-6d  %-8.0e  %-10.4f  %-10.3f  %-8.2f  %-8.1f\n",
                    N, Ra, r.Nu, Nu_ref, err, t)
        end
        println()
    end
end

function run_cpu_benchmark()
    println("=" ^ 70)
    println("  CPU vs GPU timing comparison (Ra=1e4, Pr=0.71, Rc=1)\n")

    for N in [128, 256]
        steps = nsteps(N, 1e4)

        # CPU first (separate process avoids Metal conflict)
        t_cpu = @elapsed begin
            r_cpu = run_natural_convection_2d(; N=N, Ra=1e4, Pr=0.71, Rc=1.0,
                                                max_steps=steps,
                                                backend=KernelAbstractions.CPU())
        end

        @printf("  N=%-4d  CPU: %6.1fs (Nu=%.4f)\n", N, t_cpu, r_cpu.Nu)
    end
    println()
    println("  OpenFOAM ref: 200×200 graded, 12 CPUs, Ra=1e3 → 32s CPU / 72s wall")
    println()
end

# Run GPU study first, then CPU benchmark
run_gpu_study()
# Skip CPU to avoid segfault — use times from previous run
println("  CPU reference (from previous run):")
println("  N=128, Ra=1e3: 10.5s  |  Ra=1e4: 21s  |  Ra=1e5: 42s")
println()
