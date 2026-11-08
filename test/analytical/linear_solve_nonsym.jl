# Regression test for the spd=false contract of the linear-solve seam.
#
# BUG HISTORY: the old spd=false path tried `ldlt(Symmetric(A))` FIRST. On a
# genuinely NON-symmetric A this does not throw — CHOLMOD silently factorizes
# the symmetrized (upper-triangle) operator and returns wrong fields, seen as
# O(1e2) residuals on advection-diffusion systems. The fix gates LDLᵀ on
# `issymmetric(A)` and routes non-symmetric operators straight to LU.
#
# This testset pins the contract: `lin_factorize(A; spd=false)` + `lin_solve!`
# must solve BOTH a genuinely non-symmetric system and a symmetric INDEFINITE
# system to direct-solver accuracy (relative residual < 1e-10). It checks the
# residual, NOT which factorization branch fired.

using Test
using LinearAlgebra
# SparseArrays is a KrakeNN dep but NOT in test/Project.toml: a bare
# `using SparseArrays` fails in the Pkg.test sandbox (Julia >= 1.12 strict
# loading, same failure mode as Gmsh/WriteVTK). Load it through KrakeNN.
using KrakeNN.SparseArrays

if !isdefined(@__MODULE__, :lin_factorize)
    include(joinpath(@__DIR__, "..", "..", "src", "solve", "linear_solve.jl"))
end

# 1D advection-diffusion on (0,1), homogeneous Dirichlet ends, first-order
# upwind advection (a > 0). Interior row i:
#   (-nu/h^2 - a/h) x_{i-1} + (2 nu/h^2 + a/h) x_i + (-nu/h^2) x_{i+1} = b_i
# The upwind term loads ONLY the sub-diagonal, so sub != super -> !issymmetric.
function assemble_advdiff_upwind(N::Int; nu::Float64=1.0e-2, a::Float64=1.0)
    h = 1.0 / (N + 1)
    lower = fill(-nu / h^2 - a / h, N - 1)
    diag = fill(2.0 * nu / h^2 + a / h, N)
    upper = fill(-nu / h^2, N - 1)
    return spdiagm(-1 => lower, 0 => diag, 1 => upper)
end

# Symmetric INDEFINITE tridiagonal: eigenvalues 0.5 + 2 cos(k*pi/(N+1)) straddle
# zero, so spd=true (Cholesky) would be wrong but LDLᵀ/LU must still solve it.
function assemble_symmetric_indefinite(N::Int)
    return spdiagm(-1 => ones(N - 1), 0 => fill(0.5, N), 1 => ones(N - 1))
end

@testset "Linear-solve seam spd=false (non-symmetric regression)" begin
    @testset "Genuinely non-symmetric advection-diffusion (upwind)" begin
        N = 100
        A = assemble_advdiff_upwind(N)
        # Guard: the test MUST exercise the non-symmetric path. If an edit to
        # the assembly ever makes A symmetric, the regression coverage is gone.
        @test !issymmetric(A)

        x_exact = [sin(pi * i / (N + 1)) + 0.1 * i for i in 1:N]
        b = A * x_exact

        cache = lin_factorize(A; backend = CPUBackendTag(), spd = false)
        x = lin_solve!(cache, b)

        @test norm(A * x - b) / norm(b) < 1.0e-10
        @test norm(x - x_exact) / norm(x_exact) < 1.0e-8

        # Factorize-once reuse: a SECOND RHS through the same cache must be
        # just as accurate (the seam never re-factorizes).
        b2 = A * reverse(x_exact)
        x2 = lin_solve!(cache, b2)
        @test norm(A * x2 - b2) / norm(b2) < 1.0e-10
    end

    @testset "Symmetric indefinite still solves under spd=false" begin
        N = 100
        A = assemble_symmetric_indefinite(N)
        @test issymmetric(A)
        @test !isposdef(Matrix(A))   # indefinite: Cholesky (spd=true) would fail

        x_exact = [cos(2.0 * pi * i / N) for i in 1:N]
        b = A * x_exact

        cache = lin_factorize(A; backend = CPUBackendTag(), spd = false)
        x = lin_solve!(cache, b)

        @test norm(A * x - b) / norm(b) < 1.0e-10
        @test norm(x - x_exact) / norm(x_exact) < 1.0e-8
    end
end
