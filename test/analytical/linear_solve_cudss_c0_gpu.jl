# GPU contract of the extended cuDSS seam (C0), against CPU UMFPACK/CHOLMOD
# on the same operators: refactorization reusing the symbolic analysis
# (`lin_factorize` after a value change == `lin_refactorize!`), in-place and
# multi-RHS solves, native transposed solve (CUDSS.jl >= 1.0) or its explicit
# refusal (older CUDSS.jl), and the reduced allocation contract
# (`CUDA.@allocated == 0` per refactorize + solve iteration on the Kraken side).
#
# Self-gated like poisson_cudss_gpu.jl: skipped when CUDA + CUDSS are not
# loadable or CUDA is not functional. KRAKEN_REQUIRE_GPU=1 turns the skip into
# a failure (GPU job receipts must not pass by skipping).
# Large sizes (512²-1024²), timings and the 10-iteration free-memory
# stability check: benchmarks/cudss_c0/c0_cudss_bench.jl.

using Test
using LinearAlgebra
using Kraken.SparseArrays

const _C0_GPU_OK = (try
    @eval Main using CUDA, CUDSS
    @eval Main CUDA.functional()
catch
    false
end) === true

const _C0_REQUIRE_GPU = get(ENV, "KRAKEN_REQUIRE_GPU", "0") in ("1", "true")

if !isdefined(@__MODULE__, :_c0_convdiff)
    include(joinpath(@__DIR__, "linear_solve_c0_helpers.jl"))
end

# `CUDA.@allocated` must not be macro-expanded where CUDA is absent: define the
# probe through @eval only when the GPU path will run.
if _C0_GPU_OK
    @eval _c0_gpu_allocated(f) = Main.CUDA.@allocated f()
end

@testset "linear-solve seam C0 (cuDSS GPU)" begin
    if !_C0_GPU_OK
        if _C0_REQUIRE_GPU
            @error "KRAKEN_REQUIRE_GPU=1 but CUDA + CUDSS are not functional"
            @test _C0_GPU_OK
        else
            @info "Skipping cuDSS C0 GPU tests (CUDA+CUDSS not functional here)"
            @test_skip "cuDSS C0 GPU contract requires CUDA.functional() + CUDSS"
        end
    else
        CU = Main.CUDA
        CUDSS = Main.CUDSS
        ext = Base.get_extension(Kraken, :KrakenCUDSSExt)
        @test ext !== nothing
        CSR = ext._CUDSS_GPU_MAT
        has_T = Kraken._cudss_transpose_supported(pkgversion(CUDSS))
        @info "cuDSS C0 GPU" CUDA = pkgversion(CU) CUDSS = pkgversion(CUDSS) transpose_native = has_T

        rel(A, x, b) = norm(A * Array(x) - Array(b)) / norm(Array(b))
        n = 48
        N = n^2
        A = _c0_convdiff(n)
        At = sparse(A')
        b = collect(range(1.0, 2.0; length = N))
        B = hcat(b, reverse(b), ones(N), sin.(1:N))
        db = CU.CuArray(b)
        dB = CU.CuArray(B)
        cpu = lin_factorize(A; spd = false)

        @testset "LU: solve / multi-RHS / in-place parity vs UMFPACK" begin
            gc = lin_factorize(A; backend = CUDABackendTag(), spd = false)
            @test gc.work isa ext._CudssWork
            x_ref = lin_solve!(cpu, b)
            @test norm(Array(lin_solve!(gc, b)) - x_ref) / norm(x_ref) < 1e-12   # host b
            @test norm(Array(lin_solve!(gc, db)) - x_ref) / norm(x_ref) < 1e-12  # device b
            X = lin_solve!(gc, dB)
            @test X isa CU.CuMatrix{Float64}
            Xref = lin_solve!(cpu, B)
            @test norm(Array(X) - Xref) / norm(Xref) < 1e-12
            dx = CU.zeros(Float64, N)
            db0 = copy(db)
            @test lin_solve!(dx, gc, db) === dx
            @test Array(db) == Array(db0)                      # b untouched
            @test norm(Array(dx) - x_ref) / norm(x_ref) < 1e-12
            @test_throws ArgumentError lin_solve!(zeros(N), gc, db)   # host x refused
        end

        @testset "LU: transposed solve (native) or explicit refusal" begin
            gc = lin_factorize(A; backend = CUDABackendTag(), spd = false)
            @test lin_supports_transpose(gc) == has_T
            if has_T
                xT_ref = lin_solve!(cpu, b; transpose = true)
                xT = lin_solve!(gc, db; transpose = true)
                @test norm(Array(xT) - xT_ref) / norm(xT_ref) < 1e-12
                @test rel(At, xT, b) < 1e-12
                @test rel(A, xT, b) > 1e-3                  # not A \ b
                XT = similar(dB)
                lin_solve!(XT, gc, dB; transpose = true)
                XTref = lin_solve!(cpu, B; transpose = true)
                @test norm(Array(XT) - XTref) / norm(XTref) < 1e-12
                # solve_mode is restored: a following normal solve is A \ b
                @test rel(A, lin_solve!(gc, db), b) < 1e-12
            else
                @test_throws ErrorException lin_solve!(gc, db; transpose = true)
                @test rel(A, lin_solve!(gc, db), b) < 1e-12
            end
        end

        @testset "refactorization reuses the analysis: lu! == lu" begin
            A2 = _c0_newvalues(A, 0.3)
            fresh = lin_factorize(A2; backend = CUDABackendTag(), spd = false)
            x_fresh = Array(lin_solve!(fresh, db))
            # (1) host CSC with new values
            gc = lin_factorize(A; backend = CUDABackendTag(), spd = false)
            lin_refactorize!(gc, A2)
            @test norm(Array(lin_solve!(gc, db)) - x_fresh) / norm(x_fresh) < 1e-12
            @test rel(A2, lin_solve!(gc, db), b) < 1e-12
            # (2) values written into cache.A.nzVal on device, no argument
            copyto!(gc.A.nzVal, CSR(A).nzVal)
            lin_refactorize!(gc)
            @test rel(A, lin_solve!(gc, db), b) < 1e-12
            # (3) a separate device matrix with the same pattern
            lin_refactorize!(gc, CSR(A2))
            @test norm(Array(lin_solve!(gc, db)) - x_fresh) / norm(x_fresh) < 1e-12
            if has_T
                xT_ref = lin_solve!(lin_factorize(A2; spd = false), b; transpose = true)
                @test norm(Array(lin_solve!(gc, db; transpose = true)) - xT_ref) /
                      norm(xT_ref) < 1e-12
            end
            # pattern change refused
            Abad = A + sparse([1], [N], [1.0], N, N)
            @test_throws ArgumentError lin_refactorize!(gc, Abad)
            @test_throws ArgumentError lin_refactorize!(gc, CSR(Abad))
        end

        @testset "Cholesky (spd) and pinned operators" begin
            S = _c0_spd(n)
            gs = lin_factorize(S; backend = CUDABackendTag(), spd = true)
            xs = lin_solve!(lin_factorize(S; spd = true), b)
            @test norm(Array(lin_solve!(gs, db; transpose = true)) - xs) / norm(xs) < 1e-12
            @test lin_supports_transpose(gs)
            S2 = S + 0.5I
            lin_refactorize!(gs, CSR(S2))
            @test rel(S2, lin_solve!(gs, db), b) < 1e-12

            Ln = _c0_singular(n)
            k0 = 7
            gp = lin_factorize(Ln; backend = CUDABackendTag(), spd = true, pin_k0 = k0)
            cp = lin_factorize(Ln; spd = true, pin_k0 = k0)
            xp_ref = lin_solve!(cp, b)
            xp = Array(lin_solve!(gp, db))
            @test norm(xp - xp_ref) / norm(xp_ref) < 1e-10
            Xp = Array(lin_solve!(gp, dB))
            @test norm(Xp - lin_solve!(cp, B)) / norm(xp_ref) < 1e-10
            @test maximum(abs, Xp[k0, :]) < 1e-12
            Ln2 = copy(Ln); nonzeros(Ln2) .*= 2
            lin_refactorize!(gp, Ln2)
            lin_refactorize!(cp, copy(Ln2))
            @test norm(Array(lin_solve!(gp, db)) - lin_solve!(cp, b)) / norm(xp_ref) < 1e-10
            @test_throws ArgumentError lin_refactorize!(gp)        # device re-pin refused
            @test_throws ArgumentError lin_refactorize!(gp, gp.A)
        end

        @testset "reduced allocation contract (CUDA.@allocated == 0)" begin
            gc = lin_factorize(A; backend = CUDABackendTag(), spd = false)
            gp = lin_factorize(_c0_singular(n); backend = CUDABackendTag(), spd = true,
                               pin_k0 = 3)
            dx = CU.zeros(Float64, N)
            dX = CU.zeros(Float64, N, size(B, 2))
            v1 = copy(gc.A.nzVal)
            v2 = CSR(_c0_newvalues(A, 0.2)).nzVal
            step!(v) = (copyto!(gc.A.nzVal, v); lin_refactorize!(gc);
                        lin_solve!(dx, gc, db);
                        has_T && lin_solve!(dx, gc, db; transpose = true);
                        lin_solve!(dX, gc, dB); lin_solve!(dx, gp, db); nothing)
            step!(v2); step!(v1)                 # warm-up (compilation, dX staging)
            CU.synchronize()
            allocs = [_c0_gpu_allocated(() -> step!(isodd(it) ? v2 : v1)) for it in 1:10]
            @info "cuDSS C0 per-iteration CUDA.@allocated (bytes)" allocs
            @test all(==(0), allocs)
        end
    end
end
