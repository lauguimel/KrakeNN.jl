# T3: Newton on the flat-bed ice-stream rectangle (Dirichlet edges, Weertman drag,
# Glen n = 3). Default 24 x 24; KRAKEN_SSA_N=64 for the full-size case.
using Test
if !isdefined(@__MODULE__, :ssa_stream_problem)
    include(joinpath(@__DIR__, "ssa_helpers.jl"))
end

@testset "SSA Newton, Weertman drag, n = 3 (T3)" begin
    N = ssa_test_n()
    p = ssa_stream_problem(N)
    t = @elapsed r = ssa_solve(p)
    res = r.stats.inf[end]
    println("SSA T3 Newton: N = $N, converged = $(r.converged), iterations = $(r.iterations), ",
            "final |R|_inf = $res kPa, |R0|_inf = $(r.stats.inf[1]) kPa, ",
            "min step = $(minimum(r.stats.alpha)), u in $(extrema(r.u)), time = $(round(t, digits = 1)) s")
    @test r.converged
    @test r.iterations <= 20
    @test res <= 1e-8
    @test all(isfinite, r.u) && all(isfinite, r.v)
    @test maximum(r.u) > 10.0                  # the stream actually flows
    # quadratic tail: the last full step takes the residual from > 1e-4 to <= 1e-8
    h = r.stats.history
    @test h[end] < 1e-6 * h[end-2]
    # one more Newton call from the solution does nothing
    again = ssa_newton!(r.solver, r.U)
    @test again.converged && again.iterations == 0
    @test r.solver.n_factorize == 1       # one symbolic analysis, refactorize afterwards
end
