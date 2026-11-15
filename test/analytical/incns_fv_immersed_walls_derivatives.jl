# Generic immersed walls: circle equivalence, dual Jacobian and adjoint (CPU Float64).
#
# Frozen gates (2026-11-12):
# - FVCircleWall lowers to the same tags, cut fractions and wall velocities as
#   fv_immersed_circle (<= 1e-14), and the same residual (<= 1e-13 relative);
# - coloured dual Jacobian = dense ForwardDiff Jacobian <= 1e-12 * max|J|, and
#   every nonzero of the dense Jacobian lies in fv_jacobian_pattern (same
#   contract as incns_fv_energy_open_bc_test.jl), energy active, every closure;
# - central FD of the residual along a random direction: J*v - FD <= 1e-6 * |J*v|;
# - adjoint gradient of a law parameter vs central full-resolve FD:
#   best relative <= 1e-6 (T4 contract of incns_fv_thermal_adjoint_test.jl),
#   adjoint residual |J' lambda - g_U| <= 1e-10 |g_U|.
isdefined(@__MODULE__, :_iw_problem) || include(joinpath(@__DIR__, "incns_fv_immersed_walls_helpers.jl"))

const _IWD_TW = 468.15

function _iwd_dense_jacobian(p, U)
    D = ForwardDiff.Dual{KrakeNN.FVJacobianTag,Float64,1}
    X = D.(U); J = zeros(length(U), length(U)); R = similar(X)
    for col in eachindex(U)
        X[col] = D(U[col], ForwardDiff.Partials((1.0,)))
        fv_residual!(R, X, p)
        for row in eachindex(U)
            J[row, col] = ForwardDiff.partials(R[row])[1]
        end
        X[col] = D(U[col])
    end
    return J
end

function _iwd_law_problem(theta; r=0)
    law = LearnedViscosity(TinyMLP{2,2,1}(; shift=(0.0, _IWD_TW), scale=(1.0, 2.0));
                           nu_ref=0.3, monotone=false)
    energy = FVEnergy(; alpha=0.2, sigma=0.4, left=_IWD_TW, right=_IWD_TW + 0.5,
        solid=(k, x, y) -> _IWD_TW + (k == 1 ? 1.0 : -0.5) * sin(pi * x / 2))
    viscosity = FVLawViscosity(law, theta, (FVStateTemperature(),); eps=0.02)
    return _iw_problem(r; conv=true, energy, viscosity)
end

function _iwd_solve!(s, U)
    stats = fv_newton!(s, U; maxit=30)
    for _ in 1:2
        fv_newton_finalize!(s, U)
        fv_residual!(s.R, U, s.prob)
        lin_solve!(s.dU, s.cache, s.R)
        U .-= s.dU
    end
    return stats
end

@testset "FV immersed walls: circle path and derivatives" begin
    @testset "FVCircleWall equals the analytic circle" begin
        for (closure, mass) in ((:quadratic, :M3), (:linear, :M1), (:c1, :M2))
            ref = fv_immersed_circle(24, 20, 11.3, 9.7, 5.3; h=1.0, closure, mass, pressure=:linear)
            gen = fv_immersed_walls(24, 20; walls=(FVCircleWall(11.3, 9.7, 5.3),), h=1.0,
                                    closure, mass, pressure=:linear)
            @test gen.tag == ref.tag
            @test maximum(abs, gen.qcut - ref.qcut) <= 1e-14
            @test maximum(abs, gen.wvel - ref.wvel) <= 1e-14
            @test gen.closure == ref.closure
            kw = (; h=1.0, nu=0.3, conv=true, wall_flux=0.0,
                  bc=(W=:inlet, E=:outlet, S=:wall, N=:wall),
                  boundary_velocity=(side, x, y) -> (side === :W ? 0.2 : 0.0, 0.0))
            pr = fv_steady_problem(24, 20; immersed=ref, kw...)
            pg = fv_steady_problem(24, 20; immersed=gen, kw...)
            @test fv_nunknowns(pr) == fv_nunknowns(pg)
            U = 0.1randn(MersenneTwister(7), fv_nunknowns(pr))
            Rr, Rg = fv_residual(U, pr), fv_residual(U, pg)
            @test norm(Rr - Rg, Inf) <= 1e-13 * max(norm(Rr, Inf), 1)
        end
    end

    @testset "dual Jacobian, energy active, closure=$closure mass=$mass" for (closure, mass) in
            ((:quadratic, :M3), (:linear, :M3), (:quadratic, :M1), (:c1, :M2))
        p = _iw_problem(0; conv=true, thermal=true, closure, mass)
        rng = MersenneTwister(2026)
        U = _iw_exact_state(p) .+ 0.05 .* randn(rng, fv_nunknowns(p))
        S = Matrix(fv_jacobian_pattern(p))
        J = Matrix(fv_jacobian!(fv_newton_solver(p), U))
        Jref = _iwd_dense_jacobian(p, U)
        scale = max(1.0, maximum(abs, Jref))
        @test isempty(findall((.!iszero.(Jref)) .& .!S))
        @test maximum(abs, J - Jref) <= 1e-12 * scale
        v = randn(rng, length(U)); v ./= norm(v, Inf)
        hstep = 1e-6
        fd = (fv_residual(U + hstep * v, p) - fv_residual(U - hstep * v, p)) / (2hstep)
        Jv = J * v
        @test norm(Jv - fd, Inf) <= 1e-6 * max(norm(Jv, Inf), 1)
        @info "immersed walls Jacobian" closure mass unknowns=length(U) dual_vs_dense=maximum(abs, J - Jref) / scale fd=norm(Jv - fd, Inf) / norm(Jv, Inf)
    end

    @testset "dual Jacobian, state-temperature law viscosity" begin
        theta = [0.3, -0.2, 0.4, -0.3, 0.1, -0.15, 0.2, -0.15, 0.04]
        p = _iwd_law_problem(theta)
        rng = MersenneTwister(99)
        U = fv_state(p, (x, y) -> _iw_velocity(x, y)[1], (x, y) -> _iw_velocity(x, y)[2],
                     (x, y) -> 0.0, (x, y) -> _IWD_TW + 0.3sin(3x + 2y))
        U .+= 0.02 .* randn(rng, length(U))
        S = Matrix(fv_jacobian_pattern(p))
        J = Matrix(fv_jacobian!(fv_newton_solver(p), U))
        Jref = _iwd_dense_jacobian(p, U)
        scale = max(1.0, maximum(abs, Jref))
        @test isempty(findall((.!iszero.(Jref)) .& .!S))
        @test maximum(abs, J - Jref) <= 1e-12 * scale
    end

    @testset "adjoint law gradient vs full-resolve FD" begin
        theta = [0.3, -0.2, 0.4, -0.3, 0.1, -0.15, 0.2, -0.15, 0.04]
        p = _iwd_law_problem(theta)
        s = fv_newton_solver(p)
        U = fv_state(p, (x, y) -> _iw_velocity(x, y)[1], (x, y) -> _iw_velocity(x, y)[2],
                     (x, y) -> 0.0, _IWD_TW)
        @test _iwd_solve!(s, U).converged
        map = FVLawParameters(length(theta)); ws = fv_adjoint_workspace(s, map)
        objectives = FVObjective[fv_mean_temperature(p; T_wall=_IWD_TW, target=0.25),
                                 fv_temperature_l2(p, _IWD_TW + 0.25; weights=0.7)]
        gradients = Vector{Float64}[]
        for obj in objectives
            r = fv_steady_gradient(s, U, theta, obj; params=map, ws)
            d = fv_steady_gradient(s, U, theta, obj; params=map, ws, law_gradient=:duals, finalize=false)
            @test norm(r.grad - d.grad) <= 1e-12 * max(norm(r.grad), eps())
            @test norm(transpose(s.J) * r.λ - r.gU) <= 1e-10 * max(norm(r.gU), eps())
            push!(gradients, r.grad)
        end
        sfd = fv_newton_solver(p)
        rng = MersenneTwister(4018)
        for (label, direction) in (("theta[1]", Float64.(1:length(theta) .== 1)),
                                   ("random", normalize(randn(rng, length(theta)))))
            exact = [dot(g, direction) for g in gradients]
            best = fill(Inf, length(objectives))
            for hstep in (1e-3, 3e-4, 1e-4)
                values = zeros(length(objectives), 2)
                for (side, sgn) in enumerate((-1.0, 1.0))
                    fv_set_params!(sfd, map, theta .+ sgn * hstep .* direction)
                    V = copy(U)
                    @test _iwd_solve!(sfd, V).converged
                    values[:, side] = [fv_objective(obj, V, sfd.prob) for obj in objectives]
                end
                fd = (values[:, 2] - values[:, 1]) / (2hstep)
                rel = abs.(fd - exact) ./ max.(abs.(exact), eps())
                best = min.(best, rel)
                @info "immersed walls adjoint FD" label hstep exact=Tuple(exact) fd=Tuple(fd) relative=Tuple(rel)
            end
            fv_set_params!(sfd, map, theta)
            @test all(best .<= 1e-6)
        end
    end
end
