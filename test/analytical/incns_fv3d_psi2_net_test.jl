# Learned second normal stress coefficient (NeuralPsi2Law, a TinyMLP for the
# dimensionless g(Wi), Psi2 = -eta_p0 lambda g(lambda gdot)) in the 3D
# Reiner-Rivlin slice, CPU only, Float64:
#   (a) law: value against the hand formula, g = 0 at law_init, theta duals vs
#       centred FD of the value;
#   (b) adjoint gradient of a velocity misfit (axial + secondary, relative
#       weights, the objective of benchmarks/ve_duct/slice_learn.jl) through
#       fv_steady_adjoint with FVReinerRivlinParameters, against centred FD of
#       the converged objective, along single network weights and random
#       directions, at a trained-like point and at the g = 0 start;
#   (c) the parameter-dual cotangent does not depend on the chunk size.
using Test
using LinearAlgebra
using Random
using ForwardDiff
using KrakeNN

const _PN_BC = (W = :periodic, E = :periodic, S = :wall, N = :wall, B = :wall, T = :wall)
const _PN_NEWTON = (; atol = 1e-14, rtol = 1e-15, maxit = 60)

_pn_slice(N, visc) =
    fv_steady_problem(1, N, N; h = 1 / N, bc = _PN_BC, viscosity = visc,
                      source = (x, y, z) -> (57.0, 0.0, 0.0, 0.0), u_mean_target = 1.0)

function _pn_solve!(s, U)
    st = fv_newton!(s, U; _PN_NEWTON...)
    st.converged || st.stagnated || error("Newton did not converge (residual $(last(st.inf)))")
    return st
end

# misfit weights and data on the full unknown vector: velocities only
function _pn_misfit_setup(p, u, v, w; wax = 1.0)
    g = p.grid
    n = fv_nunknowns(p)
    W = zeros(n); D = zeros(n)
    sa = sum(abs2, u); ss = sum(abs2, v) + sum(abs2, w)
    for k in 1:g.Nz, j in 1:g.Ny
        a = fv_index(g, 1, j, k, 1)
        W[a] = wax / sa; D[a] = u[j, k]
        W[a + 1] = 1 / ss; D[a + 1] = v[j, k]
        W[a + 2] = 1 / ss; D[a + 2] = w[j, k]
    end
    return W, D
end
_pn_J(W, D, U) = sum(W .* (U .- D) .^ 2) / 2

@testset "3D FV Reiner-Rivlin: learned Psi2 (NeuralPsi2Law)" begin
    @testset "(a) law value, start, duals" begin
        mlp = TinyMLP{1,3,1}(; shift = (0.0,), scale = (2.0,))
        law = NeuralPsi2Law(mlp; wi0 = 1e-2)
        @test law_ninputs(law) == 1 && law_nparams(law) == 2 + 10
        rng = MersenneTwister(3)
        θ0 = law_init(law; rng, eta_p0 = 1.5, lambda = 0.6)
        @test θ0[1:2] == [1.5, 0.6]
        @test all(gd -> law_value(law, (gd,), θ0) == 0, (1e-4, 0.3, 7.0))     # g = 0 at the start
        θ = [1.5; 0.6; randn(rng, 10)]
        for gd in (1e-3, 0.5, 9.0)
            wi = 0.6 * gd
            z = log(wi + 1e-2) / 2
            h = [tanh(θ[2 + i] * z + θ[2 + 3 + i]) for i in 1:3]
            gref = dot(θ[9:11], h) + θ[12]
            @test neural_psi2_g(law, wi, θ) ≈ gref atol = 1e-13   # sum of O(1) terms
            @test law_value(law, (gd,), θ) ≈ -1.5 * 0.6 * gref atol = 1e-13
            gr = ForwardDiff.gradient(t -> law_value(law, (gd,), t), θ)
            for q in (1, 2, 4, 9, 12)
                e = 1e-6; d = Float64.(1:12 .== q)
                fd = (law_value(law, (gd,), θ .+ e .* d) - law_value(law, (gd,), θ .- e .* d)) / (2e)
                @test gr[q] ≈ fd rtol = 1e-7 atol = 1e-10
            end
        end
        @test_throws ArgumentError NeuralPsi2Law(mlp; wi0 = 0.0)
    end

    @testset "(b) adjoint of the velocity misfit vs centred FD" begin
        N = 8; lam = 0.6; alpha = 0.4
        eta = FVLawViscosity(GiesekusViscosityLaw(), [1.0, 1.0, lam, alpha]; eps = 1e-4)
        # synthetic "truth": closed-form Giesekus Psi2
        ptrue = _pn_slice(N, FVReinerRivlinViscosity(eta, FVLawViscosity(GiesekusPsi2Law(), [1.0, lam, alpha];
                                                                         eps = 1e-4)))
        strue = fv_newton_solver(ptrue)
        Ut = KrakeNN._fv_initial_force!(zeros(fv_nunknowns(ptrue)), ptrue)
        _pn_solve!(strue, Ut)
        ut, vt, wt, _ = fv_fields(ptrue, Ut)
        @test maximum(hypot.(vt[1, :, :], wt[1, :, :])) > 1e-5
        law = NeuralPsi2Law(TinyMLP{1,3,1}(; shift = (0.0,), scale = (2.0,)))
        rng = MersenneTwister(11)
        θnet = law_init(law; rng, eta_p0 = 1.0, lambda = lam)
        θtr = copy(θnet); θtr[end] = 0.3; θtr[(end - 3):(end - 1)] .= 0.05 .* randn(rng, 3)  # g ~ 0.3
        npsi = law_nparams(law)
        map = FVReinerRivlinParameters(4, npsi)
        W, D = _pn_misfit_setup(ptrue, ut[1, :, :], vt[1, :, :], wt[1, :, :])
        for (label, θpsi) in (("trained-like point", θtr), ("g = 0 start", θnet))
            @testset "$label" begin
                θ = [1.0, 1.0, lam, alpha, θpsi...]
                p = _pn_slice(N, FVReinerRivlinViscosity(eta, FVLawViscosity(law, copy(θpsi); eps = 1e-4)))
                s = fv_newton_solver(p)
                U0 = KrakeNN._fv_initial_force!(zeros(fv_nunknowns(p)), p)
                U = copy(U0)
                st = _pn_solve!(s, U)
                @test last(st.inf) < 1e-11
                gU = W .* (U .- D)
                ws = fv_adjoint_workspace(s, map; chunk = 4)
                grad, stats = fv_steady_adjoint(s, U, gU; params = map, θ, ws)
                @test stats.residual_inf <= 1e-12
                @test length(grad) == 4 + npsi && all(isfinite, grad)
                sfd = fv_newton_solver(p)
                function J_at(θp)
                    fv_set_params!(sfd, map, θp)
                    Up = copy(U0)
                    _pn_solve!(sfd, Up)
                    return _pn_J(W, D, Up)
                end
                @test J_at(θ) ≈ _pn_J(W, D, U) rtol = 1e-10
                P = 4 + npsi
                dnet(q) = Float64.(1:P .== 4 + 2 + q)          # q-th network weight
                rdir = [zeros(6); randn(rng, npsi - 2)]
                dirs = ((dnet(1), "hidden weight 1"), (dnet(4), "hidden bias 1"),
                        (dnet(8), "output weight 2"), (dnet(npsi - 2), "output bias"),
                        (rdir, "random network direction"), (Float64.(1:P .== 2), "eta_p0 of eta"))
                for (d, name) in dirs
                    gdd = dot(grad, d)
                    errs = Float64[]
                    for step in (1e-3, 1e-4)
                        ε = step / norm(d, Inf)
                        push!(errs, abs((J_at(θ .+ ε .* d) - J_at(θ .- ε .* d)) / (2ε) - gdd))
                    end
                    @info "NeuralPsi2Law adjoint vs FD" label name gdd err = minimum(errs)
                    @test minimum(errs) <= 1e-9 + 1e-5 * abs(gdd)
                end
                # the misfit must see the output layer at the start
                @test abs(grad[end]) > 1e-6
                if label == "g = 0 start"
                    # output weights zero: the hidden layer has no first-order effect
                    @test maximum(abs, grad[7:(6 + 6)]) <= 1e-12 * maximum(abs, grad)
                end
                # (c) chunk size does not change the cotangent
                ws2 = fv_adjoint_workspace(s, map; chunk = 8)
                grad2, _ = fv_steady_adjoint(s, U, gU; params = map, θ, ws = ws2)
                @test grad2 ≈ grad rtol = 1e-12 atol = 1e-15
            end
        end
    end
end
