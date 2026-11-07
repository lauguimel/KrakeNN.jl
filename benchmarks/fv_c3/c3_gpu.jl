# C3 GPU receipts: the C3 test files (CPU checks + GPU/cuDSS contract), then a
# Newton benchmark of generalised-Newtonian (GNF) flows on cuDSS:
#   - power-law lid-driven cavity, n = 0.5, K = 0.01, ε = 1e-2 (Re_K = 100),
#     from rest, at 256², 512² and 1024²;
#   - Carreau-Yasuda obstacle channel of the LBM GNF adjoint tests
#     (test/analytical/incns_fv_gnf.jl, _fvc3_obstacle) at m = 32 and 64
#     (256×128, 512×256), from rest: flux and iterations under refinement.
# Per case: pattern size (nnz/row), colours, set-up time, per Newton iteration
# the split assembly / (re)factorization / solve / line search and the residual
# history, first factorization (analysis + numeric), cuDSS memory estimates,
# device memory used, then a 10-iteration hot loop with CUDA.@allocated per
# phase and the free-memory drift, and the GPU vs CPU Jacobian at ≤ 256².
#
#   KRAKEN_REQUIRE_GPU=1 julia --project=<env> benchmarks/fv_c3/c3_gpu.jl
#
# <env>: KrakeNN (dev'd by path), CUDA 6, CUDSS 1, ForwardDiff,
# KernelAbstractions, Test (benchmarks/fv_c3/setup_c3.sh).
# Local smoke run without a GPU (CPU backend, small sizes, CUDA-only parts
# skipped): C3_CPU_SMOKE=1 julia --project=<env> benchmarks/fv_c3/c3_gpu.jl
using CUDA
const GPU = CUDA.functional()
println("CUDA.functional() = ", GPU); flush(stdout)
const SMOKE = get(ENV, "C3_CPU_SMOKE", "0") == "1"
GPU || SMOKE || error("CUDA not functional (set C3_CPU_SMOKE=1 for the CPU smoke run)")
using CUDSS
using Test, LinearAlgebra, Random, Printf, Statistics
using KrakeNN
using KrakeNN.SparseArrays
println(GPU ? "GPU: $(CUDA.name(CUDA.device()))" : "CPU smoke run", "  CUDA.jl ", pkgversion(CUDA),
        "  CUDSS.jl ", pkgversion(CUDSS), "  KrakeNN ", pkgversion(KrakeNN), "  threads ", Threads.nthreads())
const TA = joinpath(@__DIR__, "..", "..", "test", "analytical")
include(joinpath(TA, "fv_newton_helpers.jl"))

const STATUS = Ref(0)
const TO = GPU ? CuArray : Array
const LS = GPU ? CUDABackendTag() : CPUBackendTag()

if !SMOKE
    println("\n--- C3 test files ---"); flush(stdout)
    try
        @testset "C3 GPU receipts" begin
            for f in ("incns_fv_gnf.jl", "incns_fv_gnf_gpu.jl")
                println("\n--- ", f, " ---"); flush(stdout)
                include(joinpath(TA, f))
            end
        end
    catch err
        println("TESTS FAILED: ", sprint(showerror, err)); STATUS[] = 1
    end
    flush(stdout)
end

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
free_mem() = GPU ? CUDA.free_memory() : 0
reclaim() = (GC.gc(); GPU && CUDA.reclaim(); nothing)
devzeros(n) = GPU ? CUDA.zeros(Float64, n) : zeros(n)
hostJ(J) = J isa SparseMatrixCSC ? J :
    sparse(transpose(SparseMatrixCSC(size(J)..., Array(J.rowPtr), Array(J.colVal), Array(J.nzVal))))
allocated(f) = GPU ? (CUDA.@allocated f()) : -1

thin_law() = PowerLaw(0.01, 0.5; nu_min = 0.0, nu_max = Inf)
cavity(N) = fv_steady_problem(N, N; rheology = thin_law(), gnf_eps = 1e-2, boundary_velocity = _fvc2_lid)
function obstacle(m)
    nu0 = 0.17; Nx, Ny = 8m, 4m
    solid = [(((i - 0.5) / m - 4)^2 + ((j - 0.5) / m)^2 < 1.5^2) ||
             (((i - 0.5) / m - 4)^2 + ((j - 0.5) / m - 4)^2 < 1.5^2) for i in 1:Nx, j in 1:Ny]
    g = 0.3 * 5 * nu0^2 / m^3
    law = CarreauYasuda(nu0, 0.02 * nu0, 5.0 * m^2 / nu0, 2.0, 0.7)
    return fv_steady_problem(Nx, Ny; h = 1.0, rheology = law, gnf_eps = 0.0, solid,
                             bc = (W = :periodic, E = :periodic, S = :wall, N = :wall),
                             source = (x, y) -> (g, 0.0, 0.0))
end

function bench(name, p; parity = false)
    g = p.grid
    println("\n================ ", name, " (", g.Nx, " x ", g.Ny, ", from rest) ================"); flush(stdout)
    reclaim()
    free0 = free_mem()
    t_setup = @elapsed s = fv_newton_solver(p; to = TO, linsolve = LS)
    n = fv_nunknowns(p)
    @printf("  unknowns %d | nnz %d (%.2f/row) | colours %d | chunk %d -> %d dual passes\n",
            n, s.asm.nnz, s.asm.nnz / n, s.asm.ncolors, fv_jacobian_chunk(s.asm),
            cld(s.asm.ncolors, fv_jacobian_chunk(s.asm)))
    @printf("  set-up (CPU pattern %.2f s, colouring %.2f s, total incl. upload %.2f s)\n",
            s.t_pattern, s.t_coloring, t_setup)
    free_setup = free_mem()
    U = devzeros(n)
    t_newton = @elapsed st = fv_newton!(s, U; maxit = 50)
    free_newton = free_mem()
    @printf("  converged %s in %d iterations, %.2f s wall (n_factorize %d, n_refactorize %d)\n",
            st.converged, st.iterations, t_newton, st.n_factorize, st.n_refactorize)
    st.converged || (STATUS[] = 1)
    println("  it | ‖R‖₂ after  | ‖R_mom‖₂   | ‖R_cont‖₂  | ‖R‖∞      | alpha | assembly ms | (re)factor s | solve ms | line search ms")
    @printf("   0 | %.3e |  %.3e |  %.3e | %.3e |\n", st.history[1], st.mom[1], st.cont[1], st.inf[1])
    for k in 1:st.iterations
        @printf("  %2d | %.3e |  %.3e |  %.3e | %.3e | %5.3f | %10.2f | %11.3f | %8.2f | %8.2f\n",
                k, st.history[k+1], st.mom[k+1], st.cont[k+1], st.inf[k+1], st.alpha[k],
                1e3 * st.t_assembly[k], st.t_factor[k], 1e3 * st.t_solve[k], 1e3 * st.t_linesearch[k])
    end
    if st.iterations > 1
        tref = median(st.t_factor[2:end])
        @printf("  first factorization (analysis + numeric) %.2f s | median refactorization %.3f s | analysis (deduced) %.2f s\n",
                st.t_factor[1], tref, st.t_factor[1] - tref)
        @printf("  per Newton iteration (median, it >= 2): assembly %.2f ms | refactorize %.3f s | solve %.2f ms | line search %.2f ms\n",
                1e3 * median(st.t_assembly[2:end]), tref, 1e3 * median(st.t_solve[2:end]),
                1e3 * median(st.t_linesearch[2:end]))
    end
    u, _, _ = fv_fields(p, Array(U))
    @printf("  flux through x = 0 (sum of u over the first column, x h): %.6e | max u %.6e\n",
            sum(u[1, :]) * g.h, maximum(u))
    if GPU
        me = memory_estimates(s.cache.factor)
        me === nothing || @printf("  cuDSS memory_estimates: permanent device %.2f GiB, peak device %.2f GiB, peak host %.2f GiB\n",
                                  gib(me[1]), gib(me[2]), gib(me[4]))
        @printf("  cuDSS lu_nnz %d (fill ratio %.1f vs nnz(J))\n", lu_nnz(s.cache.factor), lu_nnz(s.cache.factor) / s.asm.nnz)
        @printf("  device memory used: after set-up %.2f GiB | after Newton (factors resident) %.2f GiB\n",
                gib(free0 - free_setup), gib(free0 - free_newton))
    end
    R = s.R; dU = s.dU
    println("  hot loop | assembly B | refactorize B | solve B | residual+norms B | device free GiB")
    frees = Int[]
    for it in 1:10
        a1 = allocated(() -> fv_jacobian!(s, U))
        a2 = allocated(() -> lin_refactorize!(s.cache))
        a3 = allocated(() -> (lin_solve!(dU, s.cache, R); GPU && CUDA.synchronize()))
        a4 = allocated(() -> (fv_residual!(R, U, s.prob); KrakeNN._fv_norms(s.norms, R)))
        push!(frees, free_mem())
        @printf("    %2d | %d | %d | %d | %d | %.3f\n", it, a1, a2, a3, a4, gib(frees[end]))
    end
    U .+= 1e-6 .* TO(randn(MersenneTwister(1), n))
    a5 = allocated(() -> fv_newton!(s, U; maxit = 3))
    @printf("  free-memory spread over the 10 hot iterations: %.1f MiB | allocated by 3 full Newton iterations from a perturbed state: %d B\n",
            (maximum(frees) - minimum(frees)) / 2^20, a5)
    if parity
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
    reclaim()
    return nothing
end

cases = SMOKE ?
    (("power-law cavity n = 0.5, N = 32", () -> cavity(32), true),
     ("CY obstacle channel m = 4", () -> obstacle(4), false)) :
    (("power-law cavity n = 0.5, N = 256", () -> cavity(256), true),
     ("power-law cavity n = 0.5, N = 512", () -> cavity(512), false),
     ("CY obstacle channel m = 32", () -> obstacle(32), false),
     ("CY obstacle channel m = 64", () -> obstacle(64), false),
     ("power-law cavity n = 0.5, N = 1024", () -> cavity(1024), false))
for (name, mk, par) in cases
    try
        bench(name, mk(); parity = par)
    catch err
        println("BENCH FAILED ", name, ": ", sprint(showerror, err)); STATUS[] = 1
    end
end
println("DONE status=", STATUS[])
exit(STATUS[])
