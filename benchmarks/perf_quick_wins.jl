# GPU optimization quick-wins benchmark: persistent kernels + workgroup tuning.
#
# Usage:
#   julia --project benchmarks/perf_quick_wins.jl           # CPU
#   julia --project benchmarks/perf_quick_wins.jl --gpu     # GPU
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

function make_f(N, T, backend)
    Nx, Ny, Q = N, N, 9
    f_cpu = ones(T, Nx, Ny, Q) .* T(1 / Q)
    is_cpu = falses(Nx, Ny)
    if backend === nothing
        return f_cpu, is_cpu
    elseif HAS_CUDA
        return CuArray(f_cpu), CuArray(is_cpu)
    elseif HAS_METAL
        return MtlArray(f_cpu), MtlArray(is_cpu)
    end
end

gpu_sync() = begin
    HAS_CUDA && CUDA.synchronize()
    HAS_METAL && Metal.synchronize(Metal.global_queue(Metal.current_device()))
end

compute_mlups(N, steps, elapsed) = N * N * steps / elapsed / 1e6
function compute_bw_aa(N, T, steps, elapsed)
    bytes_per_node = 18 * sizeof(T) + sizeof(Bool)
    N * N * bytes_per_node * steps / elapsed / 1e9
end

# Baseline (Level 6): AA+f32 with per-step host sync
function run_baseline_aa(f, is_solid, N, ω, steps)
    for s in 1:steps
        if iseven(s)
            aa_even_step!(f, is_solid, N, N, ω)
        else
            aa_odd_step!(f, is_solid, N, N, ω)
        end
    end
end

# Persistent: 1 sync per Nt steps
function run_persistent_aa(f, is_solid, N, ω, steps; workgroupsize=nothing)
    persistent_aa_bgk!(f, is_solid, N, N, ω, steps; workgroupsize=workgroupsize)
end

function run_quick_wins(; N=1024, steps=500, gpu=false, peak_bw=3350.0)
    backend = nothing
    backend_name = "CPU"
    if gpu
        if HAS_CUDA
            backend = CUDABackend(); backend_name = "CUDA"
        elseif HAS_METAL
            backend = MetalBackend(); backend_name = "Metal"; peak_bw = 400.0
        else
            println("No GPU backend available."); return
        end
    end

    T = Float32
    ν = 0.01
    ω = Float32(1.0 / (3.0 * ν + 0.5))
    peak_mlups = peak_bw / (18 * sizeof(T)) * 1e3

    levels = [
        (6,  "AA+f32 (baseline)",      :baseline, nothing),
        (7,  "AA+f32 persistent",      :persist,  nothing),
        (8,  "AA+f32 persist wg=64",   :persist,  (8, 8)),
        (9,  "AA+f32 persist wg=256",  :persist,  (16, 16)),
        (10, "AA+f32 persist wg=1024", :persist,  (32, 32)),
    ]

    println("\n=== Quick Wins Benchmark (N=$N, steps=$steps, $backend_name) ===")
    println("  Peak BW: $(peak_bw) GB/s → $(round(Int, peak_mlups)) MLUPS (f32)")
    @printf("  %-7s %-30s %8s %10s %7s %9s %7s\n",
            "Level", "Label", "MLUPS", "BW(GB/s)", "%BW", "%MLUPS", "vs L6")
    @printf("  %-7s %-30s %8s %10s %7s %9s %7s\n",
            "-----", "------------------------------", "------", "--------", "-----", "-------", "-----")

    mlups_l6 = 0.0
    for (level, label, mode, wg) in levels
        f, is_solid = make_f(N, T, backend)

        # Warmup
        if mode == :baseline
            run_baseline_aa(f, is_solid, N, ω, 4)
        else
            run_persistent_aa(f, is_solid, N, ω, 4; workgroupsize=wg)
        end
        gpu && gpu_sync()

        # Reset
        f, is_solid = make_f(N, T, backend)
        gpu && gpu_sync()

        elapsed = @elapsed begin
            if mode == :baseline
                run_baseline_aa(f, is_solid, N, ω, steps)
            else
                run_persistent_aa(f, is_solid, N, ω, steps; workgroupsize=wg)
            end
            gpu && gpu_sync()
        end

        mlups = compute_mlups(N, steps, elapsed)
        bw = compute_bw_aa(N, T, steps, elapsed)
        pct_bw = bw / peak_bw * 100
        pct_mlups = mlups / peak_mlups * 100
        if level == 6
            mlups_l6 = mlups
        end
        speedup = mlups_l6 > 0 ? mlups / mlups_l6 : 1.0

        @printf("  %-7d %-30s %8.0f %10.1f %6.1f%% %8.1f%% %6.2fx\n",
                level, label, mlups, bw, pct_bw, pct_mlups, speedup)
    end
    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    gpu = "--gpu" in ARGS
    local bench_N = 1024
    local bench_steps = 500
    for a in ARGS
        m = match(r"^--N=(\d+)$", a)
        isnothing(m) || (bench_N = parse(Int, m.captures[1]))
        m = match(r"^--steps=(\d+)$", a)
        isnothing(m) || (bench_steps = parse(Int, m.captures[1]))
    end
    run_quick_wins(; N=bench_N, steps=bench_steps, gpu=gpu)
end
