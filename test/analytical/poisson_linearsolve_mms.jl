# Analytical @testset for the LinearSolve.jl front-end (ext/KrakeNNLinearSolveExt).
# Requires `using LinearSolve` BEFORE inclusion — runtests.jl guards the LOAD
# (Enzyme-guard pattern), so real ext failures still surface when LinearSolve
# IS present. Covers: Dirichlet MMS convergence, parity vs the CHOLMOD seam AND
# the matrix-free MG path (same discretization), the pinned Neumann gauge, and
# the factorize-once cache-reuse contract of the PoissonLinearSolve tag.

using Test
using LinearAlgebra: norm
using SparseArrays

ls_dirichlet_exact(x, y) = sin(pi * x) * sin(pi * y)
ls_dirichlet_rhs(x, y)   = 2.0 * pi^2 * sin(pi * x) * sin(pi * y)
ls_neumann_exact(x, y)   = cos(pi * x) * cos(pi * y)
ls_neumann_rhs(x, y)     = 2.0 * pi^2 * cos(pi * x) * cos(pi * y)

@testset "Poisson LinearSolve front-end (ext)" begin

    @testset "Dirichlet MMS second-order convergence" begin
        NS = (16, 32, 64)
        errors = Float64[]
        for N in NS
            u = solve_poisson_direct(ls_dirichlet_rhs, N)
            push!(errors, KrakeNN.l2_error(u, ls_dirichlet_exact, N))
        end
        orders = [log2(errors[k-1] / errors[k]) for k in 2:length(errors)]
        mean_order = sum(orders) / length(orders)
        @test 1.8 <= mean_order <= 2.2
        @info "LinearSolve Dirichlet MMS" N=NS errors=errors orders=orders mean_order=mean_order
    end

    @testset "Parity vs CHOLMOD seam and MG at N=32" begin
        N = 32
        u_ls   = solve_poisson_direct(ls_dirichlet_rhs, N)
        u_chol = solve_poisson_dirichlet(N, ls_dirichlet_rhs)
        # Same assembled operator, both direct solvers: agreement to solver eps.
        @test maximum(abs.(u_ls .- u_chol)) <= 1e-10

        u_mg, _, _ = solve_poisson_mg(ls_dirichlet_rhs, N; bc=:dirichlet,
                                      tol=1e-12, maxcycles=100, smoother=:rbgs)
        # Same tolerance class as the MG↔CHOLMOD parity test (poisson_mg_mms.jl).
        @test maximum(abs.(u_ls .- Array(u_mg))) <= 1e-7
        @info "LinearSolve parity" N=N Linf_vs_cholmod=maximum(abs.(u_ls .- u_chol)) Linf_vs_mg=maximum(abs.(u_ls .- Array(u_mg)))
    end

    @testset "Neumann pinned gauge (singular nullspace handled)" begin
        NS = (16, 32, 64)
        errors = Float64[]
        for N in NS
            u = solve_poisson_direct(ls_neumann_rhs, N; bc=:neumann)
            # Pinned-to-zero gauge: compare zero-mean fields (poisson_mg_mms.jl
            # Neumann convention).
            ua = u .- sum(u) / length(u)
            ex = KrakeNN.exact_field(N, ls_neumann_exact)
            ex .-= sum(ex) / length(ex)
            push!(errors, sqrt((1.0 / N)^2 * sum((ua .- ex) .^ 2)))
        end
        orders = [log2(errors[k-1] / errors[k]) for k in 2:length(errors)]
        mean_order = sum(orders) / length(orders)
        @test 1.8 <= mean_order <= 2.2
        @info "LinearSolve Neumann+pin MMS" N=NS errors=errors orders=orders mean_order=mean_order
    end

    @testset "Factorize-once cache reuse (PoissonLinearSolve tag)" begin
        N = 16
        A, b = KrakeNN.assemble_poisson_dirichlet(N, ls_dirichlet_rhs)

        cache = lin_factorize(A; backend=PoissonLinearSolve(), spd=true)
        x1 = lin_solve!(cache, b)
        x2 = lin_solve!(cache, 2.0 .* b)   # fresh RHS, SAME factors
        @test norm(A * x1 .- b) / norm(b) <= 1e-10
        @test maximum(abs.(x2 .- 2.0 .* x1)) <= 1e-10

        # Explicit algorithm passthrough (LU path, spd=false).
        cache_lu = lin_factorize(A; backend=PoissonLinearSolve(alg=UMFPACKFactorization()),
                                 spd=false)
        x_lu = lin_solve!(cache_lu, b)
        @test maximum(abs.(x_lu .- x1)) <= 1e-9
    end
end
