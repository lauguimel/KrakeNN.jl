# C0 cuDSS path — GPU receipts at production sizes (512², 1024²) against CPU
# UMFPACK on the same operators:
#   * timings: cuDSS analysis / factorization (raw CudssSolver phases),
#     lin_factorize, lin_refactorize! (device values and host upload),
#     in-place solve, transposed solve, 4-RHS solve;
#   * correctness: |x_gpu - x_umfpack| / |x_umfpack| for A, Aᵀ, 4 RHS, and
#     after refactorization with new values (lu! == fresh lu);
#   * reduced R11 contract: 10 iterations (fixed precomputed value sets, no
#     random draws, nothing allocated outside the measured calls) with
#     CUDA.@allocated per iteration and device free memory after each.
#
# Operators: the EHD Poisson matrix (production :direct phi operator,
# non-symmetric because of the identity plate rows) and a non-symmetric
# upwind convection-diffusion (Aᵀ ≠ A, so a silently ignored transpose shows).
#
#   julia --project=<env with KrakeNN, CUDA, CUDSS> benchmarks/cudss_c0/c0_cudss_bench.jl
# Env: C0_SIZES (default "512,1024"), C0_OPS (default "ehd,convdiff").
using CUDA, CUDSS
using KrakeNN
using LinearAlgebra, Printf, Statistics
using KrakeNN.SparseArrays

include(joinpath(@__DIR__, "..", "..", "test", "analytical", "linear_solve_c0_helpers.jl"))

CUDA.functional() || error("CUDA not functional")
const SIZES = parse.(Int, split(get(ENV, "C0_SIZES", "512,1024"), ","))
const OPS = Symbol.(split(get(ENV, "C0_OPS", "ehd,convdiff"), ","))
const ext = Base.get_extension(KrakeNN, :KrakeNNCUDSSExt)
ext === nothing && error("KrakeNNCUDSSExt not loaded")
const HAS_T = KrakeNN._cudss_transpose_supported(pkgversion(CUDSS))

println("GPU: ", CUDA.name(CUDA.device()))
println("CUDA.jl ", pkgversion(CUDA), "  CUDSS.jl ", pkgversion(CUDSS),
        "  KrakeNN ", pkgversion(KrakeNN), "  native transpose: ", HAS_T)
flush(stdout)

timed(f) = (CUDA.synchronize(); t = time(); f(); CUDA.synchronize(); time() - t)
median_ms(f; n = 5) = (f(); 1e3 * median([timed(f) for _ in 1:n]))
relerr(x, xref) = norm(Array(x) - xref) / norm(xref)
verdict(e, tol = 1e-10) = e <= tol ? "ok" : "FAILED"
# A2 is ill-conditioned: judge the GPU against UMFPACK's own forward error
# on A2 (one iterative-refinement step), not against a fixed 1e-10.
verdict_ref(e, eref) = verdict(e, max(10eref, 1e-12))
refstep(F, M, x, b) = norm(F \ (b - M * x)) / norm(x)

function build(op::Symbol, N::Int)
    op === :ehd && return KrakeNN._ehd_poisson_matrix(N, N, :neumann)
    op === :convdiff && return _c0_convdiff(N)
    error("unknown operator $op")
end

function run_case(op::Symbol, N::Int)
    A = build(op, N)
    n = size(A, 1)
    A2 = _c0_newvalues(A, 0.3)
    b = collect(range(1.0, 2.0; length = n))
    B = hcat(b, reverse(b), ones(n), sin.(1:n))
    @printf("\n=== %s %d² : n = %d  nnz = %d  symmetric = %s ===\n", op, N, n, nnz(A),
            issymmetric(A))

    # CPU UMFPACK references
    t_cpu = @elapsed Fc = lu(A)
    x_ref = Fc \ b
    xT_ref = transpose(Fc) \ b
    X_ref = Fc \ B
    Fc2 = lu(A2)
    x2_ref = Fc2 \ b
    x2T_ref = transpose(Fc2) \ b
    u2 = refstep(Fc2, A2, x2_ref, b)
    u2T = refstep(transpose(Fc2), transpose(A2), x2T_ref, b)
    Fc = nothing; Fc2 = nothing
    @printf("  CPU UMFPACK lu: %.2f s\n", t_cpu)

    # cuDSS phase split (raw API, separate solver)
    Ad = ext._CUDSS_GPU_MAT(A)
    xd = CUDA.zeros(Float64, n); bd = CuArray(b)
    raw = CudssSolver(Ad, "G", 'F')
    t_an = timed(() -> cudss("analysis", raw, xd, bd))
    t_fa = timed(() -> cudss("factorization", raw, xd, bd))
    raw = nothing; Ad = nothing
    GC.gc(); CUDA.reclaim()
    @printf("  cuDSS raw phases: analysis %.2f s  factorization %.3f s\n", t_an, t_fa)

    # KrakeNN seam
    CUDA.synchronize()
    t_fac = @elapsed begin
        cache = lin_factorize(A; backend = CUDABackendTag(), spd = false)
        CUDA.synchronize()
    end
    dX = CUDA.zeros(Float64, n, 4); dB = CuArray(B)
    t_s = median_ms(() -> lin_solve!(xd, cache, bd))
    e_s = relerr(xd, x_ref)
    t_m = median_ms(() -> lin_solve!(dX, cache, dB))
    e_m = relerr(dX, X_ref)
    @printf("  lin_factorize (analysis+factorization): %.2f s\n", t_fac)
    @printf("  solve        : %8.2f ms  |x - x_umf|/|x_umf| = %.1e  %s\n", t_s, e_s, verdict(e_s))
    @printf("  solve 4 RHS  : %8.2f ms  rel err = %.1e  %s\n", t_m, e_m, verdict(e_m))
    if HAS_T
        t_t = median_ms(() -> lin_solve!(xd, cache, bd; transpose = true))
        e_t = relerr(xd, xT_ref)
        e_wrong = relerr(xd, x_ref)
        @printf("  solve Aᵀ     : %8.2f ms  rel err vs Aᵀ\\b = %.1e  %s  (vs A\\b: %.1e)\n",
                t_t, e_t, verdict(e_t), e_wrong)
    else
        refused = try
            lin_solve!(xd, cache, bd; transpose = true); false
        catch err
            err isa ErrorException
        end
        println("  solve Aᵀ     : refused with an explicit error (CUDSS.jl < 1.0): ",
                refused ? "ok" : "FAILED")
    end

    # refactorization: device values, then host upload; lu! == lu
    v1 = copy(cache.A.nzVal)
    v2 = ext._CUDSS_GPU_MAT(A2).nzVal
    t_rd = median_ms(() -> (copyto!(cache.A.nzVal, v2); lin_refactorize!(cache)); n = 3)
    lin_solve!(xd, cache, bd); e_r = relerr(xd, x2_ref)
    @printf("  refactorize (device values): %8.1f ms  rel err vs lu(A2) = %.1e  (UMFPACK own err %.1e)  %s\n",
            t_rd, e_r, u2, verdict_ref(e_r, u2))
    if HAS_T
        lin_solve!(xd, cache, bd; transpose = true); e_rt = relerr(xd, x2T_ref)
        @printf("  Aᵀ after refactorize        : rel err = %.1e  (UMFPACK own err %.1e)  %s\n",
                e_rt, u2T, verdict_ref(e_rt, u2T))
    end
    copyto!(cache.A.nzVal, v1); lin_refactorize!(cache)
    t_rh = timed(() -> lin_refactorize!(cache, A2))
    lin_solve!(xd, cache, bd); e_rh = relerr(xd, x2_ref)
    @printf("  refactorize (host CSC upload): %6.1f ms  rel err = %.1e  (UMFPACK own err %.1e)  %s\n",
            1e3t_rh, e_rh, u2, verdict_ref(e_rh, u2))

    # reduced R11 contract
    println("  R11 iter | CUDA.@allocated B (refactorize + solve [+ Aᵀ] + 4 RHS) | device free GiB")
    frees = Int[]; allocs = Int[]
    step!(v) = (copyto!(cache.A.nzVal, v); lin_refactorize!(cache);
                lin_solve!(xd, cache, bd);
                HAS_T && lin_solve!(xd, cache, bd; transpose = true);
                lin_solve!(dX, cache, dB); nothing)
    for it in 1:10
        a = CUDA.@allocated step!(isodd(it) ? v2 : v1)
        CUDA.synchronize()
        push!(allocs, a); push!(frees, CUDA.free_memory())
        @printf("  %8d | %d | %.3f\n", it, a, frees[end] / 2^30)
    end
    spread = (maximum(frees[2:end]) - minimum(frees[2:end])) / 2^20
    @printf("  R11 reduced contract: allocations %s, free-memory spread (iters 2-10) = %.1f MiB\n",
            all(==(0), allocs) ? "0 (ok)" : "NONZERO (FAILED)", spread)
    cache = nothing
    GC.gc(); CUDA.reclaim()
    flush(stdout)
end

for op in OPS, N in SIZES
    run_case(op, N)
end
println("\nDONE")
