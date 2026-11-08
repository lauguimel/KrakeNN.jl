# GPU optimization benchmark: 7 levels from baseline to AA+F32
# Measures MLUPS, bandwidth, and %peak for each optimization level.
#
# Usage:
#   julia --project benchmarks/perf_optimizations.jl           # CPU only
#   julia --project benchmarks/perf_optimizations.jl --gpu     # GPU (auto-detect)
using KrakeNN
using Printf
using KernelAbstractions

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

# --- Helpers ---

function make_arrays(N, T, backend)
    Nx, Ny, Q = N, N, 9
    if backend === nothing
        f_in  = ones(T, Nx, Ny, Q) .* T(1 / Q)
        f_out = similar(f_in)
        ρ  = ones(T, Nx, Ny)
        ux = zeros(T, Nx, Ny)
        uy = zeros(T, Nx, Ny)
        is_solid = falses(Nx, Ny)
    else
        f_cpu = ones(T, Nx, Ny, Q) .* T(1 / Q)
        if HAS_CUDA
            f_in  = CuArray(f_cpu)
            f_out = CUDA.similar(f_in)
            ρ  = CUDA.ones(T, Nx, Ny)
            ux = CUDA.zeros(T, Nx, Ny)
            uy = CUDA.zeros(T, Nx, Ny)
            is_solid = CuArray(falses(Nx, Ny))
        elseif HAS_METAL
            f_in  = MtlArray(f_cpu)
            f_out = Metal.similar(f_in)
            ρ  = MtlArray(ones(T, Nx, Ny))
            ux = MtlArray(zeros(T, Nx, Ny))
            uy = MtlArray(zeros(T, Nx, Ny))
            is_solid = MtlArray(falses(Nx, Ny))
        end
    end
    return f_in, f_out, ρ, ux, uy, is_solid
end

function compute_mlups(N, steps, elapsed)
    N * N * steps / elapsed / 1e6
end

function compute_bw(N, T, steps, elapsed)
    # Each node: read 9 + write 9 populations + read 1 is_solid (bool)
    # + write 3 macroscopic fields (ρ, ux, uy)
    bytes_per_node = (9 + 9) * sizeof(T) + sizeof(Bool) + 3 * sizeof(T)
    total_bytes = N * N * bytes_per_node * steps
    total_bytes / elapsed / 1e9  # GB/s
end

function compute_bw_aa(N, T, steps, elapsed)
    # AA pattern: single buffer, each node reads/writes 9 pops in-place
    # Even: read 9 from neighbors + write 9 local = 18
    # Odd: read 9 local + write 9 to neighbors = 18
    bytes_per_node = 18 * sizeof(T) + sizeof(Bool)
    total_bytes = N * N * bytes_per_node * steps
    total_bytes / elapsed / 1e9
end

# --- Level runners ---

function run_level0(f_in, f_out, ρ, ux, uy, is_solid, N, ω, steps)
    for _ in 1:steps
        stream_2d!(f_out, f_in, N, N)
        collide_2d!(f_out, is_solid, ω)
        compute_macroscopic_2d!(ρ, ux, uy, f_out)
        f_in, f_out = f_out, f_in
    end
    return f_in, f_out
end

function run_level2_nosync(f_in, f_out, ρ, ux, uy, is_solid, N, ω, steps)
    for _ in 1:steps
        stream_2d!(f_out, f_in, N, N; sync=false)
        collide_2d!(f_out, is_solid, ω; sync=false)
        compute_macroscopic_2d!(ρ, ux, uy, f_out; sync=true)
        f_in, f_out = f_out, f_in
    end
    return f_in, f_out
end

function run_level3_fused(f_in, f_out, ρ, ux, uy, is_solid, N, ω, steps)
    for _ in 1:steps
        fused_bgk_step!(f_out, f_in, ρ, ux, uy, is_solid, N, N, ω)
        f_in, f_out = f_out, f_in
    end
    return f_in, f_out
end

function run_level5_aa(f, is_solid, N, ω, steps)
    for s in 1:steps
        if iseven(s)
            aa_even_step!(f, is_solid, N, N, ω)
        else
            aa_odd_step!(f, is_solid, N, N, ω)
        end
    end
    return f
end

# --- Main benchmark ---

function run_optimization_benchmark(; N=1024, steps=500, gpu=false, peak_bw=3350.0)
    backend = nothing
    backend_name = "CPU"

    if gpu
        if HAS_CUDA
            backend = CUDABackend()
            backend_name = "CUDA"
        elseif HAS_METAL
            backend = MetalBackend()
            backend_name = "Metal"
            peak_bw = 400.0  # M3 Max approximate
        else
            println("No GPU backend available.")
            return
        end
    end

    ν = 0.01
    ω64 = 1.0 / (3.0 * ν + 0.5)
    ω32 = Float32(ω64)

    levels = [
        (0, "baseline",      Float64, :standard),
        (1, "float32",        Float32, :standard),
        (2, "nosync",         Float64, :nosync),
        (3, "fused",          Float64, :fused),
        (4, "fused+f32",      Float32, :fused),
        (5, "AA-pattern",     Float64, :aa),
        (6, "AA+f32",         Float32, :aa),
    ]

    # Theoretical peak MLUPS for each precision
    peak_mlups_f64 = peak_bw / (18 * sizeof(Float64)) * 1e3  # GB/s → MLUPS
    peak_mlups_f32 = peak_bw / (18 * sizeof(Float32)) * 1e3

    println("\n=== GPU Optimization Benchmark (N=$N, steps=$steps, $backend_name) ===")
    println("  Peak BW: $(peak_bw) GB/s → $(round(Int, peak_mlups_f64)) MLUPS (f64) / $(round(Int, peak_mlups_f32)) MLUPS (f32)")
    @printf("  %-7s %-18s %-9s %8s %10s %7s %9s %7s\n",
            "Level", "Label", "Prec.", "MLUPS", "BW(GB/s)", "%BW", "%MLUPS", "vs L0")
    @printf("  %-7s %-18s %-9s %8s %10s %7s %9s %7s\n",
            "-----", "---------------", "-------", "------", "--------", "-----", "-------", "-----")

    mlups_l0 = 0.0

    for (level, label, T, mode) in levels
        local elapsed

        ω = T == Float32 ? ω32 : ω64

        if mode == :aa
            f, _, _, _, _, is_solid = make_arrays(N, T, backend)

            # Warmup
            run_level5_aa(f, is_solid, N, ω, 2)
            if gpu
                HAS_CUDA && CUDA.synchronize()
                HAS_METAL && Metal.synchronize(Metal.global_queue(Metal.current_device()))
            end

            # Reset
            f, _, _, _, _, is_solid = make_arrays(N, T, backend)
            if gpu
                HAS_CUDA && CUDA.synchronize()
                HAS_METAL && Metal.synchronize(Metal.global_queue(Metal.current_device()))
            end

            elapsed = @elapsed begin
                run_level5_aa(f, is_solid, N, ω, steps)
            end

            bw = compute_bw_aa(N, T, steps, elapsed)
        else
            f_in, f_out, ρ, ux, uy, is_solid = make_arrays(N, T, backend)

            # Warmup (2 steps)
            if mode == :standard
                run_level0(f_in, f_out, ρ, ux, uy, is_solid, N, ω, 2)
            elseif mode == :nosync
                run_level2_nosync(f_in, f_out, ρ, ux, uy, is_solid, N, ω, 2)
            elseif mode == :fused
                run_level3_fused(f_in, f_out, ρ, ux, uy, is_solid, N, ω, 2)
            end
            if gpu
                HAS_CUDA && CUDA.synchronize()
                HAS_METAL && Metal.synchronize(Metal.global_queue(Metal.current_device()))
            end

            # Reset arrays
            f_in, f_out, ρ, ux, uy, is_solid = make_arrays(N, T, backend)
            if gpu
                HAS_CUDA && CUDA.synchronize()
                HAS_METAL && Metal.synchronize(Metal.global_queue(Metal.current_device()))
            end

            elapsed = @elapsed begin
                if mode == :standard
                    run_level0(f_in, f_out, ρ, ux, uy, is_solid, N, ω, steps)
                elseif mode == :nosync
                    run_level2_nosync(f_in, f_out, ρ, ux, uy, is_solid, N, ω, steps)
                elseif mode == :fused
                    run_level3_fused(f_in, f_out, ρ, ux, uy, is_solid, N, ω, steps)
                end
            end

            bw = compute_bw(N, T, steps, elapsed)
        end

        mlups = compute_mlups(N, steps, elapsed)
        pct_bw = bw / peak_bw * 100
        peak_mlups = T == Float32 ? peak_mlups_f32 : peak_mlups_f64
        pct_mlups = mlups / peak_mlups * 100
        if level == 0
            mlups_l0 = mlups
        end
        speedup = mlups_l0 > 0 ? mlups / mlups_l0 : 1.0

        @printf("  %-7d %-18s %-9s %8.0f %10.1f %6.1f%% %8.1f%% %6.2fx\n",
                level, label, T == Float32 ? "f32" : "f64",
                mlups, bw, pct_bw, pct_mlups, speedup)
    end
    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    gpu = "--gpu" in ARGS
    local bench_N = 1024
    for a in ARGS
        m = match(r"^--N=(\d+)$", a)
        if !isnothing(m)
            bench_N = parse(Int, m.captures[1])
        end
    end
    run_optimization_benchmark(; N=bench_N, gpu=gpu)
end
