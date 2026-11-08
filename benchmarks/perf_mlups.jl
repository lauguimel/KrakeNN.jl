# MLUPS performance scaling benchmark
# Measures throughput on CPU and GPU (if available)
# Usage: julia --project benchmarks/perf_mlups.jl [--gpu]
using KrakeNN
using Printf
using KernelAbstractions

# Try loading GPU backends at top level
const HAS_CUDA = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

const HAS_METAL = try
    @eval using Metal
    Metal.functional()
catch
    false
end

function run_mlups_benchmark(; gpu=false)
    Ns = [64, 128, 256, 512, 1024, 2048, 4096]
    steps = 200

    println("\n=== MLUPS Performance Scaling ===")

    # CPU benchmark
    println("\n--- CPU ---")
    @printf("  %5s   %10s\n", "N", "MLUPS")
    @printf("  %5s   %10s\n", "-----", "--------")

    cpu_results = benchmark_mlups(; Ns=Ns, steps=steps)
    for (N, mlups) in cpu_results
        @printf("  %5d   %10.1f\n", N, mlups)
    end

    # GPU benchmark
    if gpu
        gpu_backend = nothing
        if HAS_CUDA
            gpu_backend = CUDABackend()
            println("\n--- CUDA GPU ---")
        elseif HAS_METAL
            gpu_backend = MetalBackend()
            println("\n--- Metal GPU ---")
        end

        if !isnothing(gpu_backend)
            gpu_Ns = [64, 128, 256, 512, 1024, 2048, 4096]
            @printf("  %5s   %10s   %8s\n", "N", "MLUPS", "Speedup")
            @printf("  %5s   %10s   %8s\n", "-----", "--------", "-------")

            gpu_results = benchmark_mlups(; Ns=gpu_Ns, steps=steps, backend=gpu_backend)

            for (N, mlups_gpu) in gpu_results
                cpu_mlups = 0.0
                for (Nc, mc) in cpu_results
                    Nc == N && (cpu_mlups = mc; break)
                end
                speedup = cpu_mlups > 0 ? mlups_gpu / cpu_mlups : NaN
                sp_str = isnan(speedup) ? "    -" : @sprintf("%7.1fx", speedup)
                @printf("  %5d   %10.1f   %s\n", N, mlups_gpu, sp_str)
            end
        else
            println("\nNo GPU backend available. Skipping GPU benchmark.")
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    gpu = "--gpu" in ARGS
    run_mlups_benchmark(; gpu=gpu)
end
