# Analytical @testset for the matrix-free geometric multigrid Poisson V-cycle.
# Covers all exit criteria: MMS 2nd-order convergence, the multigrid hallmark
# (V-cycle count ~ constant vs N), parity vs the assembled CHOLMOD Poisson, and
# the singular Neumann+pin variant.

using Test
using LinearAlgebra: norm

# Auto-skip under `using KrakeNN` (runtests.jl); unexported helpers are
# KrakeNN.-qualified below, so the package-test path never double-includes src.
if !isdefined(@__MODULE__, :solve_poisson_dirichlet)
    include(joinpath(@__DIR__, "..", "..", "src", "solve", "poisson.jl"))     # CHOLMOD reference
end
if !isdefined(@__MODULE__, :solve_poisson_mg)
    include(joinpath(@__DIR__, "..", "..", "src", "solve", "poisson_mg.jl"))  # MG solver
end

mg_dirichlet_exact(x, y) = sin(pi * x) * sin(pi * y)
mg_dirichlet_rhs(x, y)   = 2.0 * pi^2 * sin(pi * x) * sin(pi * y)
mg_neumann_exact(x, y)   = cos(pi * x) * cos(pi * y)
mg_neumann_rhs(x, y)     = 2.0 * pi^2 * cos(pi * x) * cos(pi * y)

@testset "Poisson multigrid (matrix-free V-cycle)" begin

    @testset "Dirichlet MMS second-order convergence" begin
        NS = (16, 32, 64, 128)
        errors = Float64[]
        for N in NS
            u, _, _ = solve_poisson_mg(mg_dirichlet_rhs, N; bc=:dirichlet,
                                       tol=1e-10, maxcycles=60, smoother=:rbgs)
            push!(errors, KrakeNN.l2_error(Array(u), mg_dirichlet_exact, N))
        end
        orders = [log2(errors[k-1] / errors[k]) for k in 2:length(errors)]
        mean_order = sum(orders) / length(orders)
        @test 1.8 <= mean_order <= 2.2
        @info "MG Dirichlet MMS" N=NS errors=errors orders=orders mean_order=mean_order
    end

    @testset "Multigrid hallmark: V-cycle count ~ constant vs N" begin
        NS = (64, 128, 256, 512)
        counts = Int[]
        rhos = Float64[]
        for N in NS
            _, nc, hist = solve_poisson_mg(mg_dirichlet_rhs, N; bc=:dirichlet,
                                           tol=1e-10, maxcycles=60, smoother=:rbgs)
            push!(counts, nc)
            # asymptotic reduction factor: geometric mean of the last 3 ratios
            ratios = [hist[k] / hist[k-1] for k in max(2, length(hist)-2):length(hist)
                      if hist[k-1] > 0]
            push!(rhos, exp(sum(log.(ratios)) / length(ratios)))
        end
        # Hallmark: count must NOT grow like O(N). Spread across a 8x range in N
        # stays within a few cycles, and never blows up.
        @test maximum(counts) - minimum(counts) <= 5
        @test maximum(counts) <= 20
        # Each V-cycle must contract the residual by a healthy factor.
        @test all(r -> r < 0.4, rhos)
        @info "MG hallmark" N=NS cycle_counts=counts reduction_per_cycle=rhos
    end

    @testset "Parity vs assembled CHOLMOD at N=256" begin
        N = 256
        u_mg, _, _ = solve_poisson_mg(mg_dirichlet_rhs, N; bc=:dirichlet,
                                      tol=1e-12, maxcycles=100, smoother=:rbgs)
        u_chol = solve_poisson_dirichlet(N, mg_dirichlet_rhs)
        linf = maximum(abs.(Array(u_mg) .- u_chol))
        @test linf <= 1e-7
        @info "MG vs CHOLMOD parity" N=N Linf=linf
    end

    @testset "Neumann+pin variant converges (singular nullspace handled)" begin
        NS = (16, 32, 64, 128)
        errors = Float64[]
        maxcycles_seen = 0
        for N in NS
            u, nc, _ = solve_poisson_mg(mg_neumann_rhs, N; bc=:neumann,
                                        tol=1e-10, maxcycles=80, smoother=:rbgs)
            maxcycles_seen = max(maxcycles_seen, nc)
            ua = Array(u); ua .-= sum(ua) / length(ua)
            ex = KrakeNN.exact_field(N, mg_neumann_exact); ex .-= sum(ex) / length(ex)
            push!(errors, sqrt((1.0 / N)^2 * sum((ua .- ex) .^ 2)))
        end
        orders = [log2(errors[k-1] / errors[k]) for k in 2:length(errors)]
        mean_order = sum(orders) / length(orders)
        @test 1.8 <= mean_order <= 2.2
        @test maxcycles_seen < 80   # converged before the cap
        @info "MG Neumann+pin MMS" N=NS errors=errors orders=orders mean_order=mean_order maxcycles=maxcycles_seen
    end

    @testset "MG-preconditioned CG converges (bonus)" begin
        N = 128
        _, ni, hist = solve_poisson_mgcg(mg_dirichlet_rhs, N; bc=:dirichlet,
                                         tol=1e-10, maxiters=50, smoother=:rbgs)
        @test hist[end] <= 1e-10
        @test ni < 50
        @info "MG-CG" N=N niters=ni final_relres=hist[end]
    end
end
