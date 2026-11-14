# FV wall slip on smooth immersed walls: inclined plane channel (CPU Float64).
#
# Frozen gates (2026-11-12, set before the first run; case in the helper header):
# - Navier slip b = 0 on the immersed walls: residual equal to the no-slip
#   problem (<= 1e-14 relative, random state);
# - Navier slip b = 0.04 (U_s ≈ 0.3 max U) and cubic slip, quadratic ghost +
#   M3: velocity order >= 1.8 between consecutive grids h = 1/16 ... 1/128,
#   L-infinity and RMS, Stokes and Navier-Stokes (the gate of the no-slip
#   inclined channel, incns_fv_immersed_walls_channel.jl);
#   Navier-Stokes runs with convection = :central (2026-11-13: with the
#   default first-order upwind the cubic case reached 1.77 on the finest pair,
#   the O(h) upwind diffusion grows with the wall speed; gate unchanged);
# - the converged M3 pair rows vanish (<= 1e-10).
isdefined(@__MODULE__, :_isl_problem) || include(joinpath(@__DIR__, "incns_fv_slip_helpers.jl"))

function _isl_solve(p, us)
    U = _isl_state(p, us)
    s = fv_newton_solver(p)
    stats = fv_newton!(s, U; atol = 1e-12, rtol = 1e-13, maxit = 15)
    return U, stats
end

@testset "FV wall slip: inclined immersed channel" begin
    @testset "b = 0 equals no-slip" begin
        p0 = _iw_problem(0; conv = true)
        pb = _isl_problem(0; law = FVNavierSlip(0.0), conv = true)
        U = _iw_exact_state(p0) .+ 0.05 .* randn(MersenneTwister(3), fv_nunknowns(p0))
        R0, Rb = fv_residual(U, p0), fv_residual(U, pb)
        @test norm(Rb - R0, Inf) <= 1e-14 * norm(R0, Inf)
        @test_throws ArgumentError _isl_problem(0; closure = :c1, mass = :M2)
    end

    for (name, law) in (("Navier b=0.04", FVNavierSlip(0.04)), ("cubic", _SlipCubic(0.02, 0.002))),
            conv in (false, true)
        @testset "$name slip, conv=$conv: velocity order" begin
            us = _isl_us(law)
            errors = []
            for r in 0:3
                p = _isl_problem(r; law, conv, convection = conv ? :central : :upwind)
                U, stats = _isl_solve(p, us)
                @test stats.converged
                npair = KrakeNN._fv_npairs(p.immersed)
                R = fv_residual(U, p)
                @test norm(R[end-npair+1:end], Inf) <= 1e-10
                e = _isl_errors(p, U, us)
                push!(errors, e)
                @info "slip inclined channel" name conv r h = p.grid.h us linf = e.inf rms = e.l2 iterations = stats.iterations
            end
            for r in 1:3, metric in (:inf, :l2)
                order = _iw_order(getproperty(errors[r], metric), getproperty(errors[r+1], metric))
                @info "slip inclined channel order" name conv grids = (r - 1, r) metric order
                @test order >= 1.8
            end
        end
    end
end
