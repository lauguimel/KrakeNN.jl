# C1 GPU receipts: the C1 test files (CPU checks + GPU parity / allocation
# contract) and a timing of the residual kernel at 512² and 1024² in Float64
# and Dual{4} (the C2 chunk size), with CUDA.@allocated per call and the device
# free memory over 10 evaluations (R11 reduced contract).
#
#   KRAKEN_REQUIRE_GPU=1 julia --project=<env> benchmarks/fv_c1/c1_gpu.jl
#
# <env> holds Kraken (dev'd by path), CUDA, ForwardDiff, KernelAbstractions, Test.
using CUDA
using Test, LinearAlgebra, Random, Printf, ForwardDiff
using Kraken

println("CUDA.functional() = ", CUDA.functional())
CUDA.functional() || error("CUDA not functional")
println("GPU: ", CUDA.name(CUDA.device()), "  CUDA.jl ", pkgversion(CUDA), "  Kraken ", pkgversion(Kraken))
const TA = joinpath(@__DIR__, "..", "..", "test", "analytical")

@testset "C1 GPU receipts" begin
    for f in ("incns_fv_steady_residual.jl", "incns_fv_steady_residual_gpu.jl")
        println("\n--- ", f, " ---"); flush(stdout)
        include(joinpath(TA, f))
    end
end

struct C1BenchTag end
function bench(N)
    p = fv_steady_problem(N, N; nu = (x, y) -> 1e-3 * (1 + x * y),
                          boundary_velocity = (s, x, y) -> s === :N ? (1.0, 0.0) : (0.0, 0.0))
    n = fv_nunknowns(p)
    U = randn(MersenneTwister(1), n)
    dp = fv_adapt(CuArray, p)
    dU = CuArray(U); dR = CUDA.zeros(Float64, n)
    D = ForwardDiff.Dual{C1BenchTag,Float64,4}
    dUd = CuArray([D(U[k], ForwardDiff.Partials(ntuple(m -> m == (k % 4) + 1 ? 1.0 : 0.0, 4))) for k in 1:n])
    dRd = similar(dUd)
    # CPU reference at the same size (KA CPU backend, threads = JULIA_NUM_THREADS)
    t_cpu = @elapsed Rc = fv_residual(U, p)
    t_cpu = @elapsed Rc = fv_residual(U, p)
    fv_residual!(dR, dU, dp)
    par = norm(Array(dR) - Rc, Inf) / norm(Rc, Inf)
    for (lab, R_, U_) in (("Float64", dR, dU), ("Dual{4}", dRd, dUd))
        fv_residual!(R_, U_, dp)
        ts = [CUDA.@elapsed(fv_residual!(R_, U_, dp)) for _ in 1:20]
        a = CUDA.@allocated fv_residual!(R_, U_, dp)
        frees = Int[]
        for it in 1:10
            fv_residual!(R_, U_, dp); push!(frees, CUDA.free_memory())
        end
        @printf("  %5d² %-8s residual median %.3f ms (min %.3f) | CUDA.@allocated %d B | free-memory spread over 10 calls %.1f MiB\n",
                N, lab, 1e3 * sort(ts)[10], 1e3 * minimum(ts), a, (maximum(frees) - minimum(frees)) / 2^20)
    end
    @printf("  %5d² CPU residual %.1f ms (%d threads) | GPU vs CPU max rel diff %.1e\n",
            N, 1e3 * t_cpu, Threads.nthreads(), par)
end
println("\n--- residual timing ---")
for N in (512, 1024)
    bench(N)
end
println("DONE")
