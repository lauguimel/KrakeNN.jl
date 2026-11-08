# T4: learned sliding law F_theta (TinyMLP, 17 weights) in the SSA solve; misfit
# L = 1/2 sum (u - u_obs)^2 with u_obs from the Weertman solution. dL/dtheta from ONE
# transposed solve vs central finite differences of the re-solved misfit.
using Test
if !isdefined(@__MODULE__, :ssa_stream_problem)
    include(joinpath(@__DIR__, "ssa_helpers.jl"))
end

@testset "SSA learned drag adjoint (T4)" begin
    r = ssa_learned_gradient_check()
    println("SSA T4: N = $(ssa_test_n()), weights = $(r.nweights), L(theta0) = $(r.L0), |grad|_inf = $(maximum(abs, r.grad))")
    println("SSA T4: adjoint vs central FD (h = $(r.steps[2])), relative error on weights $(r.chosen): ", r.rel)
    println("SSA T4: V-shape on weight $(r.chosen[1]) for steps $(r.steps): ", r.vshape)
    @test r.L0 > 0
    @test all(isfinite, r.grad)
    @test maximum(r.rel) < 1e-5
    @test r.vshape[2] < r.vshape[1] && r.vshape[2] < r.vshape[3]    # V: truncation left, round-off right
    # gradient at the Weertman solution of a law that is not zero misfit is nonzero;
    # the factorization of the forward solve is reused: no second symbolic analysis
    @test r.solver.n_factorize == 1
end
