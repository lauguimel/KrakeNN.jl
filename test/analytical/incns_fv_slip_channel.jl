# FV wall slip, grid-aligned walls: plane channel (CPU Float64).
#
# Frozen gates (2026-11-12, set before the first run; cases in the helper header):
# - slip = nothing and slip walls on no face give a bit-identical residual
#   to the default problem (random state);
# - b = 0 Navier slip gives the no-slip solution (<= 1e-12 relative);
# - Navier and cubic slip, Stokes and Navier-Stokes: the FV solution equals
#   u_noslip + G h²/(8ν) + f(G H/2) to 1e-10 relative (discrete exactness of
#   the slip velocity), hence second order (order >= 1.8, Ny = 8 ... 64);
# - Herschel-Bulkley (Papanastasiou) channel with Navier slip vs the
#   regularised 1D reference + b G H/2: relative L2 error <= 2 % and
#   order >= 1.8 between Ny = 32, 64, 128.
isdefined(@__MODULE__, :_sl_channel) || include(joinpath(@__DIR__, "incns_fv_slip_helpers.jl"))

@testset "FV wall slip: plane channel" begin
    @testset "no-slip path unchanged" begin
        p0 = fv_steady_problem(5, 9; h = 1 / 9, nu = 0.3, conv = true,
            bc = (W = :wall, E = :wall, S = :wall, N = :wall),
            boundary_velocity = (side, x, y) -> (side === :N ? 1.0 : 0.0, 0.0))
        pn = fv_steady_problem(5, 9; h = 1 / 9, nu = 0.3, conv = true,
            bc = (W = :wall, E = :wall, S = :wall, N = :wall),
            boundary_velocity = (side, x, y) -> (side === :N ? 1.0 : 0.0, 0.0), slip = nothing)
        pe = fv_steady_problem(5, 9; h = 1 / 9, nu = 0.3, conv = true,
            bc = (W = :wall, E = :wall, S = :wall, N = :wall),
            boundary_velocity = (side, x, y) -> (side === :N ? 1.0 : 0.0, 0.0),
            slip = FVSlipWalls(FVNavierSlip(0.3)))
        @test pn.slip === nothing && p0.slip === nothing
        U = randn(MersenneTwister(5), fv_nunknowns(p0))
        R0 = fv_residual(U, p0)
        @test fv_residual(U, pn) == R0
        @test fv_residual(U, pe) == R0
        @test fv_jacobian_pattern(pe) == fv_jacobian_pattern(p0)
        @test_throws ArgumentError FVSlipWalls(FVNavierSlip(0.1); sides = (:X,))
        @test_throws ArgumentError _sl_channel(8; sides = (:W,))     # periodic side
    end

    @testset "b = 0 equals no-slip" begin
        a = solve_incns_fv_newton(; problem = _sl_channel(16; law = FVNavierSlip(0.0)), maxit = 20)
        b = solve_incns_fv_newton(; problem = _sl_channel(16; slip = nothing), maxit = 20)
        @test a.converged && b.converged
        @test norm(a.u - b.u, Inf) <= 1e-12 * norm(b.u, Inf)
    end

    for (name, law) in (("Navier b=0.1", FVNavierSlip(0.1)), ("cubic", _SlipCubic(0.05, 0.4))), conv in (false, true)
        @testset "$name slip, conv=$conv: discrete exactness and order" begin
            s = fv_slip_velocity(law, _sl_tauw())
            errs = Float64[]
            for Ny in (8, 16, 32, 64)
                p = _sl_channel(Ny; law, conv)
                r = solve_incns_fv_newton(; problem = p, maxit = 30)
                @test r.converged
                h = p.grid.h
                y = [(j - 0.5) * h for j in 1:Ny]
                ue = _sl_noslip_u.(y) .+ s
                shift = _SL_G * h^2 / (8 * _SL_NU)
                @test maximum(abs, r.u[1, :] .- (ue .+ shift)) <= 1e-10 * maximum(abs, ue)
                @test maximum(abs, r.v) <= 1e-12 * maximum(abs, ue)
                push!(errs, maximum(abs, r.u[1, :] .- ue))
                @info "slip channel" name conv Ny err = last(errs) slip = s
            end
            for k in 1:3
                @test log2(errs[k] / errs[k+1]) >= 1.8
            end
        end
    end

    @testset "Herschel-Bulkley channel with Navier slip" begin
        law = HerschelBulkley(0.2, 0.1, 0.5; m_reg = 100.0)
        eps, b = 1e-2, 0.1
        errs = Float64[]
        for Ny in (32, 64, 128)
            p = _sl_channel(Ny; law = FVNavierSlip(b), rheology = law, eps)
            r = solve_incns_fv_newton(; problem = p, maxit = 60)
            @test r.converged
            y = [(j - 0.5) * p.grid.h for j in 1:Ny]
            ur = [_sl_gnf_noslip_u(law, eps, yy) for yy in y] .+ b * _sl_tauw()
            e = norm(r.u[1, :] - ur) / norm(ur)
            push!(errs, e)
            @info "slip HB channel" Ny e iterations = r.iterations
            @test e <= 0.02
        end
        for k in 1:2
            @test log2(errs[k] / errs[k+1]) >= 1.8
        end
    end
end
