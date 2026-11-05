# C2 GPU receipts: the C2 test files (CPU checks + GPU/cuDSS contract), then a
# Newton benchmark of the lid-driven cavity at 512² and 1024² (Re = 100 and
# Re = 1000, from rest), and the pressure-pin row-scale check on cuDSS.
#
#   KRAKEN_REQUIRE_GPU=1 julia --project=<env> benchmarks/fv_c2/c2_gpu.jl
#
# <env> holds Kraken (dev'd by path), CUDA 6, CUDSS 1, ForwardDiff,
# KernelAbstractions, Test. Per Newton iteration it reports the split
# assembly / (re)factorization / solve / line search, the residual history and
# Ghia profile errors; once per case the pattern + colouring set-up time, the
# first factorization (symbolic analysis + numeric) and the analysis cost
# deduced from it (first factorization minus the median refactorization), the
# cuDSS memory estimates and the device memory used; then a 10-iteration hot
# loop with CUDA.@allocated per phase and the device free-memory drift.
using CUDA
println("CUDA.functional() = ", CUDA.functional()); flush(stdout)
CUDA.functional() || error("CUDA not functional")
using CUDSS
using Test, LinearAlgebra, Random, Printf, Statistics
using Kraken
using Kraken.SparseArrays
println("GPU: ", CUDA.name(CUDA.device()), "  CUDA.jl ", pkgversion(CUDA), "  CUDSS.jl ",
        pkgversion(CUDSS), "  Kraken ", pkgversion(Kraken), "  threads ", Threads.nthreads())
const TA = joinpath(@__DIR__, "..", "..", "test", "analytical")
include(joinpath(TA, "fv_newton_helpers.jl"))

const STATUS = Ref(0)
println("\n--- C2 test files ---"); flush(stdout)
try
    @testset "C2 GPU receipts" begin
        for f in ("incns_fv_newton.jl", "incns_fv_newton_gpu.jl")
            println("\n--- ", f, " ---"); flush(stdout)
            include(joinpath(TA, f))
        end
    end
catch err
    println("TESTS FAILED: ", sprint(showerror, err)); STATUS[] = 1
end
flush(stdout)

memory_estimates(F) = try
    v = zeros(Int64, 16)
    cudss_set(F, "memory_estimates", v)
    cudss_get(F, "memory_estimates")
    v
catch
    nothing
end
lu_nnz(F) = try cudss_get(F, "lu_nnz") catch; -1 end

gib(x) = x / 2^30
hostJ(J) = sparse(transpose(SparseMatrixCSC(size(J)..., Array(J.rowPtr), Array(J.colVal), Array(J.nzVal))))

function bench(N, Re)
    println("\n================ cavity N = $(N)², Re = $Re (from rest) ================"); flush(stdout)
    GC.gc(); CUDA.reclaim()
    free0 = CUDA.free_memory()
    p = fv_steady_problem(N, N; nu = 1 / Re, boundary_velocity = _fvc2_lid)
    t_setup = @elapsed s = fv_newton_solver(p; to = CuArray, linsolve = CUDABackendTag())
    n = fv_nunknowns(p)
    @printf("  unknowns %d | nnz %d (%.1f/row) | colours %d | chunk %d -> %d dual passes\n",
            n, s.asm.nnz, s.asm.nnz / n, s.asm.ncolors, fv_jacobian_chunk(s.asm),
            cld(s.asm.ncolors, fv_jacobian_chunk(s.asm)))
    @printf("  set-up (CPU pattern %.2f s, colouring %.2f s, total incl. upload %.2f s)\n",
            s.t_pattern, s.t_coloring, t_setup)
    free_setup = CUDA.free_memory()
    U = CUDA.zeros(Float64, n)
    t_newton = @elapsed st = fv_newton!(s, U; maxit = 40)
    free_newton = CUDA.free_memory()
    @printf("  converged %s in %d iterations, %.2f s wall (n_factorize %d, n_refactorize %d)\n",
            st.converged, st.iterations, t_newton, st.n_factorize, st.n_refactorize)
    println("  it | ‖R‖₂ after  | ‖R_mom‖₂   | ‖R_cont‖₂  | ‖R‖∞      | alpha | assembly ms | (re)factor s | solve ms | line search ms")
    @printf("   0 | %.3e |  %.3e |  %.3e | %.3e |\n", st.history[1], st.mom[1], st.cont[1], st.inf[1])
    for k in 1:st.iterations
        @printf("  %2d | %.3e |  %.3e |  %.3e | %.3e | %5.3f | %10.2f | %11.3f | %8.2f | %8.2f\n",
                k, st.history[k+1], st.mom[k+1], st.cont[k+1], st.inf[k+1], st.alpha[k],
                1e3 * st.t_assembly[k], st.t_factor[k], 1e3 * st.t_solve[k], 1e3 * st.t_linesearch[k])
    end
    tref = st.iterations > 1 ? median(st.t_factor[2:end]) : NaN
    @printf("  first factorization (analysis + numeric) %.2f s | median refactorization %.3f s | analysis (deduced) %.2f s\n",
            st.t_factor[1], tref, st.t_factor[1] - tref)
    @printf("  per Newton iteration (median, it >= 2): assembly %.2f ms | refactorize %.3f s | solve %.2f ms | line search %.2f ms\n",
            1e3 * median(st.t_assembly[2:end]), tref, 1e3 * median(st.t_solve[2:end]),
            1e3 * median(st.t_linesearch[2:end]))
    u, v, _ = fv_fields(p, Array(U))
    eu, ev = _fvc2_ghia_error(u, v, Re)
    @printf("  Ghia (1982) max |Δu| on x=1/2 %.4f, max |Δv| on y=1/2 %.4f (lid units)\n", eu, ev)
    me = memory_estimates(s.cache.factor)
    me === nothing || @printf("  cuDSS memory_estimates: permanent device %.2f GiB, peak device %.2f GiB, peak host %.2f GiB\n",
                              gib(me[1]), gib(me[2]), gib(me[4]))
    @printf("  cuDSS lu_nnz %d (fill ratio %.1f vs nnz(J))\n", lu_nnz(s.cache.factor), lu_nnz(s.cache.factor) / s.asm.nnz)
    @printf("  device memory used: after set-up %.2f GiB | after Newton (factors resident) %.2f GiB\n",
            gib(free0 - free_setup), gib(free0 - free_newton))
    # hot loop: CUDA.@allocated per phase, free memory per iteration
    R = s.R; dU = s.dU
    println("  hot loop | assembly B | refactorize B | solve B | residual+norms B | device free GiB")
    frees = Int[]
    for it in 1:10
        a1 = CUDA.@allocated fv_jacobian!(s, U)
        a2 = CUDA.@allocated lin_refactorize!(s.cache)
        a3 = CUDA.@allocated (lin_solve!(dU, s.cache, R); CUDA.synchronize())
        a4 = CUDA.@allocated (fv_residual!(R, U, s.prob); Kraken._fv_norms(s.norms, R))
        push!(frees, CUDA.free_memory())
        @printf("    %2d | %d | %d | %d | %d | %.3f\n", it, a1, a2, a3, a4, gib(frees[end]))
    end
    @printf("  free-memory spread over the 10 hot iterations: %.1f MiB | CUDA.@allocated of 3 full Newton iterations from a perturbed state: ",
            (maximum(frees) - minimum(frees)) / 2^20)
    U .+= 1e-6 .* CuArray(randn(MersenneTwister(1), n))
    println(CUDA.@allocated(fv_newton!(s, U; maxit = 3)), " B")
    # parity of the GPU Jacobian with the CPU assembly at this size (one state)
    if N <= 512
        Uh = Array(U)
        sc = fv_newton_solver(p)
        tcpu = @elapsed fv_jacobian!(sc, Uh)
        fv_jacobian!(s, U)
        Jg = hostJ(s.J)
        @printf("  Jacobian GPU vs CPU: max |Δ| / max |J| = %.1e (CPU KA assembly %.2f s)\n",
                norm(nonzeros(Jg) - nonzeros(sc.J), Inf) / norm(nonzeros(sc.J), Inf), tcpu)
    end
    flush(stdout)
    s = nothing; U = nothing
    GC.gc(); CUDA.reclaim()
    return nothing
end

# pressure-pin row scale through the cuDSS seam (host matrix route): solve
# J δ = R at a mid-Newton state with the pinned row scaled by 1, by the median
# continuity diagonal, by 1e-6 and 1e6; same δ expected
function pin_scaling(N, Re)
    p = fv_steady_problem(N, N; nu = 1 / Re, boundary_velocity = _fvc2_lid)
    s = fv_newton_solver(p; to = CuArray, linsolve = CUDABackendTag())
    U = CUDA.zeros(Float64, fv_nunknowns(p))
    fv_newton!(s, U; maxit = 2)
    fv_jacobian!(s, U)
    J = hostJ(s.J)
    R = fv_residual(Array(U), p)
    kpin = findfirst(==(FV_CELL_PINNED), vec(p.tag)); rp = 3 * (kpin - 1) + 3
    c = median(abs(J[3k, 3k]) for k in 2:N*N)
    ref = nothing
    for sc in (1.0, c, 1e-6, 1e6)
        D = ones(size(J, 1)); D[rp] = sc
        Js = sparse(Diagonal(D) * J); Rs = D .* R
        cache = lin_factorize(SparseMatrixCSC{Float64,Int}(Js); backend = CUDABackendTag(), spd = false)
        x = Array(lin_solve!(cache, CuArray(Rs)))
        be = norm(Js * x - Rs) / norm(Rs)
        ref === nothing && (ref = x)
        @printf("  pin %4d² Re=%5g scale %-9.3g | backward err %.1e | rel. diff vs scale 1 %.1e\n",
                N, Re, sc, be, norm(x - ref) / norm(ref))
    end
    flush(stdout)
end

for (N, Re) in ((512, 100.0), (1024, 100.0), (512, 1000.0), (1024, 1000.0))
    try
        bench(N, Re)
    catch err
        println("BENCH FAILED N=$N Re=$Re: ", sprint(showerror, err)); STATUS[] = 1
    end
end
println("\n--- pressure-pin row scale on cuDSS ---")
for (N, Re) in ((128, 100.0), (512, 1000.0))
    try
        pin_scaling(N, Re)
    catch err
        println("PIN SCALING FAILED N=$N: ", sprint(showerror, err)); STATUS[] = 1
    end
end
println("DONE status=", STATUS[])
exit(STATUS[])
