# Analytical validation: steady SIMPLE body-force plane Poiseuille.
#
# Body-force-driven periodic channel: height H in y, periodic in x, no-slip
# walls (u=v=0) at y=0 and y=H, constant streamwise body force G = -dP/dx.
# Fully-developed analytic solution:
#   u(y) = (G / 2mu) * y * (H - y),  centreline max = G H^2 / (8 mu),
#   v(y) = 0,  flow rate Q = G H^3 / (12 mu).
#
# Asserts the converged x-averaged u-profile matches the analytic parabola to
# <= 1% in the relative L2 norm, plus zero cross-flow and the Q/G relation.

using Test
using KernelAbstractions   # CPU() — no longer inherited once the include below auto-skips

if !isdefined(@__MODULE__, :solve_incns_simple)
    include(joinpath(@__DIR__, "..", "..", "src", "methods", "inc_ns", "simple.jl"))
end

const INCNS_POISEUILLE_RESULTS = Dict{Symbol,Any}()

function incns_poiseuille_case(; nx::Integer = 8, ny::Integer = 64,
                               H::Real = 1.0, mu::Real = 1.0, G::Real = 1.0,
                               tol::Real = 1e-10, maxiter::Integer = 300,
                               relax = (u = 0.7, p = 0.3),
                               scheme::Symbol = :simplec,
                               momentum_advection::Symbol = :linear_upwind,
                               backend = CPU())
    res = solve_incns_simple(; nx, ny, H, mu, G,
                             relax, scheme, momentum_advection, tol, maxiter, backend)

    # Analytic parabola at cell centres.
    uan = [(G / (2mu)) * y * (H - y) for y in res.ycenters]
    umax_an = G * H^2 / (8mu)

    # x-averaged numerical profile (must be x-independent for this flow).
    uprof = vec(sum(res.u; dims = 1)) ./ nx

    l2_rel = sqrt(sum(abs2, uprof .- uan) / ny) / sqrt(sum(abs2, uan) / ny)
    linf_rel = maximum(abs.(uprof .- uan)) / umax_an

    # Flow-rate / driving relation.
    Q_num = sum(uprof) * res.dy
    Q_an = G * H^3 / (12mu)
    q_rel = abs(Q_num - Q_an) / Q_an

    maxv = maximum(abs, res.v)

    return (; res, uprof, uan, umax_an, l2_rel, linf_rel, Q_num, Q_an, q_rel, maxv)
end

@testset "IncNS steady SIMPLE plane Poiseuille" begin
    backend = CPU()
    c = incns_poiseuille_case(; backend)
    INCNS_POISEUILLE_RESULTS[:poiseuille] = c

    # Converged.
    @test c.res.converged
    @test c.res.scheme === :simplec
    @test c.res.momentum_advection === :linear_upwind
    @test c.res.iters <= 50

    # Cross-flow is zero.
    @test c.maxv < 1e-10

    # u-profile within 1% (relative L2 norm) and 1% (relative Linf vs umax).
    @test c.l2_rel <= 0.01
    @test c.linf_rel <= 0.01

    # Flow-rate / driving relation within 1%.
    @test c.q_rel <= 0.01

    # Second-order spatial convergence of the wall-normal resolution.
    coarse = incns_poiseuille_case(; ny = 32, backend)
    fine = incns_poiseuille_case(; ny = 64, backend)
    order = log2(coarse.l2_rel / fine.l2_rel)
    INCNS_POISEUILLE_RESULTS[:order] = order
    @test 1.7 <= order <= 2.3
end

@testset "IncNS SIMPLE legacy scheme parity" begin
    c = incns_poiseuille_case(; ny = 32, scheme = :simple,
                              momentum_advection = :upwind, backend = CPU())
    @test c.res.scheme === :simple
    @test c.res.momentum_advection === :upwind
    @test c.res.converged
    @test c.res.iters == 2
    @test maximum(c.res.u) ≈ 0.12499999999999806 atol=1e-14
    @test sum(c.res.u) ≈ 21.374999999999662 atol=1e-11
    # Round-off of a direct solve: about 4.5e-16 on macOS arm64, 5.1e-16 on
    # Linux x86_64 (different BLAS/SuiteSparse kernels). A bound, not a literal.
    @test c.res.residual_history[end] < 1e-13

    @test_throws ArgumentError incns_poiseuille_case(;
        nx = 4, ny = 8, momentum_advection = :quick)
end
