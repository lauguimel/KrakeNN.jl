# CPU contract of the extended linear-solve seam (C0): numeric
# refactorization with the same pattern (`lin_refactorize!`), in-place and
# multiple right-hand-side solves, transposed solves, and the explicit errors
# (pattern change, LDLᵀ symmetry loss, unsupported backends, cuDSS transpose
# version gate). The cuDSS (GPU) side of the same contract is
# test/analytical/linear_solve_cudss_c0_gpu.jl.

using Test
using LinearAlgebra
# SparseArrays through Kraken (not in test/Project.toml, see linear_solve_nonsym.jl).
using Kraken.SparseArrays

if !isdefined(@__MODULE__, :_c0_convdiff)
    include(joinpath(@__DIR__, "linear_solve_c0_helpers.jl"))
end

# A minimal backend with only the allocating vector solve (no refactorization,
# no transpose): exercises the generic fallbacks and their errors.
struct _C0DummyBackend <: Kraken.LinearSolveBackend end
Kraken.lin_solve!(::_C0DummyBackend, cache::LinearSolveCache, b::AbstractVector) =
    cache.factor \ Vector{Float64}(b)

@testset "linear-solve seam C0 (CPU): refactorize / transpose / multi-RHS" begin
    n = 12
    N = n^2
    b = collect(range(1.0, 2.0; length = N))
    B = hcat(b, reverse(b), ones(N))

    @testset "UMFPACK LU (spd=false, non-symmetric)" begin
        A = _c0_convdiff(n)
        @test !issymmetric(A)
        cache = lin_factorize(copy(A); spd = false)
        @test cache.work === nothing                 # 6-arg constructor default
        @test lin_supports_transpose(cache)

        # transposed solve vs the explicit transpose
        xT = lin_solve!(cache, b; transpose = true)
        @test _c0_relres(sparse(A'), xT, b) < 1e-12
        @test _c0_relres(A, xT, b) > 1e-3               # really a different system

        # multi-RHS (allocating + in-place), normal and transposed
        X = lin_solve!(cache, B)
        @test size(X) == size(B)
        @test maximum(_c0_relres(A, X[:, j], B[:, j]) for j in 1:3) < 1e-12
        XT = similar(B)
        lin_solve!(XT, cache, B; transpose = true)
        @test maximum(_c0_relres(sparse(A'), XT[:, j], B[:, j]) for j in 1:3) < 1e-12

        # in-place vector solve leaves b untouched
        x = zeros(N); b0 = copy(b)
        @test lin_solve!(x, cache, b) === x
        @test b == b0
        @test x ≈ lin_solve!(cache, b) rtol = 1e-14

        # refactorize with new values == fresh factorization
        A2 = _c0_newvalues(A, 0.3)
        lin_refactorize!(cache, A2)
        @test nonzeros(cache.A) == nonzeros(A2)      # values copied in place
        x2 = lin_solve!(cache, b)
        x2_ref = lin_factorize(A2; spd = false) |> c -> lin_solve!(c, b)
        @test x2 ≈ x2_ref rtol = 1e-12
        @test _c0_relres(A2, x2, b) < 1e-12
        @test _c0_relres(sparse(A2'), lin_solve!(cache, b; transpose = true), b) < 1e-12

        # values written into cache.A directly, then lin_refactorize!(cache)
        nonzeros(cache.A) .= nonzeros(A)
        lin_refactorize!(cache)
        @test _c0_relres(A, lin_solve!(cache, b), b) < 1e-12

        # errors: pattern change, size mismatch, wrong type
        Abad = A + sparse([1], [N], [1.0], N, N)
        @test_throws ArgumentError lin_refactorize!(cache, Abad)
        @test_throws ArgumentError lin_refactorize!(cache, Matrix(A))
        @test_throws DimensionMismatch lin_solve!(zeros(N + 1), cache, b)
        @test_throws DimensionMismatch lin_solve!(zeros(N - 1), cache, b[1:end-1])
    end

    @testset "CHOLMOD Cholesky (spd=true)" begin
        A = _c0_spd(n)
        cache = lin_factorize(copy(A); spd = true)
        @test lin_supports_transpose(cache)
        @test lin_solve!(cache, b; transpose = true) ≈ lin_solve!(cache, b) rtol = 1e-14
        A2 = A + 0.5I
        lin_refactorize!(cache, A2)
        @test _c0_relres(A2, lin_solve!(cache, b), b) < 1e-12
        X = lin_solve!(cache, B)
        @test maximum(_c0_relres(A2, X[:, j], B[:, j]) for j in 1:3) < 1e-12
    end

    @testset "CHOLMOD LDLᵀ (spd=false, symmetric indefinite)" begin
        S = _c0_spd(n)
        A = sparse(Symmetric(S - 0.5 * maximum(diag(S)) * I))   # indefinite
        cache = lin_factorize(copy(A); spd = false)
        @test cache.factor isa SparseArrays.CHOLMOD.Factor
        A2 = A + 0.1I
        lin_refactorize!(cache, A2)
        @test _c0_relres(A2, lin_solve!(cache, b), b) < 1e-10
        # non-symmetric new values on an LDLᵀ cache: refused, never mis-factorized
        Ans = copy(A2)
        Ans[1, 2] += 1.0                       # existing entry: same pattern
        @test_throws ArgumentError lin_refactorize!(cache, Ans)
    end

    @testset "pinned singular operator (pin_k0 > 0)" begin
        Ln = _c0_singular(n)                    # zero row sums: singular
        k0 = 5
        cache = lin_factorize(copy(Ln); spd = true, pin_k0 = k0)
        x = lin_solve!(cache, b)
        @test abs(x[k0]) < 1e-12
        X = lin_solve!(cache, B)
        @test X[:, 1] ≈ x rtol = 1e-12
        @test maximum(abs, X[k0, :]) < 1e-12
        # refactorize from new unpinned values: re-pinned, equal to fresh
        L2 = _c0_newvalues(Ln, 0.0)
        nonzeros(L2) .*= 2
        lin_refactorize!(cache, L2)
        x2 = lin_solve!(cache, b)
        x2_ref = lin_solve!(lin_factorize(L2; spd = true, pin_k0 = k0), b)
        @test x2 ≈ x2_ref rtol = 1e-12
    end

    @testset "explicit errors and the cuDSS transpose gate" begin
        # cuDSS: CUDSS.jl < 1.0 silently ignores solve_mode='T' -> refuse.
        @test Kraken.CUDSS_TRANSPOSE_MIN_VERSION == v"1.0.0"
        @test !Kraken._cudss_transpose_supported(v"0.6.5")
        @test Kraken._cudss_transpose_supported(v"1.0.0")
        @test Kraken._cudss_check_transpose(v"1.2.0") === nothing
        err = try
            Kraken._cudss_check_transpose(v"0.6.5"); nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("CUDSS.jl >= 1.0.0", err.msg)
        @test occursin("0.6.5", err.msg)

        # A backend without refactorization / transpose support raises.
        A = _c0_convdiff(4)
        dcache = LinearSolveCache(_C0DummyBackend(), lu(A), A, A, 0, false)
        @test !lin_supports_transpose(dcache)
        @test_throws ArgumentError lin_refactorize!(dcache, A)
        @test_throws ArgumentError lin_solve!(dcache, ones(16); transpose = true)
        # in-place / multi-RHS fall back column-wise through the backend solve
        Bd = hcat(ones(16), collect(1.0:16.0))
        @test lin_solve!(dcache, Bd) ≈ Matrix(A) \ Bd rtol = 1e-12
    end
end
