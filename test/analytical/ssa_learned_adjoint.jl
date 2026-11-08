# T4: learned sliding law F_theta (TinyMLP, 17 weights) in the SSA solve; misfit
# L = 1/2 sum (u - u_obs)^2 with u_obs from the Weertman solution. dL/dtheta from ONE
# transposed solve vs central finite differences of the re-solved misfit.
using Test
if !isdefined(@__MODULE__, :ssa_stream_problem)
    include(joinpath(@__DIR__, "ssa_helpers.jl"))
end

"""Misfit and device state at weights `theta` (re-solves from the observed state)."""
function ssa_misfit_at!(s, U, Uobs, theta)
    ssa_set_theta!(s, theta)
    copyto!(U, Uobs)
    st = ssa_newton!(s, U; atol = 1e-11, rtol = 1e-14, maxit = 40)
    st.converged || error("SSA Newton did not converge for the FD solve")
    return ssa_misfit(U, Uobs)[1]
end

function ssa_learned_gradient_check(; N = ssa_test_n(), to = Array, linsolve = CPUBackendTag(),
                                      nweights = 17, steps = (1e-2, 1e-5, 1e-10), seed = 1)
    pw = ssa_stream_problem(N)
    obs = ssa_solve(pw; atol = 1e-11, rtol = 1e-14)
    @assert obs.converged
    Uobs = Vector{Float64}(Array(obs.U))
    drag, theta0 = ssa_learned_drag(; seed)
    pl = ssa_with(pw; drag)
    s = ssa_newton_solver(pl; to, linsolve)
    U = to(copy(Uobs)); Uo = to(copy(Uobs))
    L0 = ssa_misfit_at!(s, U, Uo, theta0)
    # state at theta0, gradient by the adjoint
    _, g = ssa_misfit(U, Uo)
    nit = s.n_refactorize
    grad = ssa_theta_gradient(s, U, g)
    # weights to check: the largest gradient entries, spread over the layout
    order = sortperm(abs.(grad); rev = true)
    chosen = order[1:nweights]
    fd(k, h) = begin
        tp = copy(theta0); tm = copy(theta0); tp[k] += h; tm[k] -= h
        Lp = ssa_misfit_at!(s, U, Uo, tp); Lm = ssa_misfit_at!(s, U, Uo, tm)
        (Lp - Lm) / (2h)
    end
    rel = Float64[]
    for k in chosen
        e = abs(fd(k, steps[2]) - grad[k]) / abs(grad[k])
        push!(rel, e)
    end
    kV = chosen[1]
    vshape = [abs(fd(kV, h) - grad[kV]) / abs(grad[kV]) for h in steps]
    return (; L0, grad, chosen, rel, vshape, steps, solver = s, nweights = length(theta0), U)
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
