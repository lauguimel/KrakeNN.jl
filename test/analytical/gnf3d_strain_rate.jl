using Test
using Random
using LinearAlgebra
using KrakeNN

# Frozen Float64 gates derive from the D3Q19 fourth-order isotropy identity:
# sum_q w_q c_a c_b c_c c_d = (delta_ab delta_cd + delta_ac delta_bd +
# delta_ad delta_bc)/9. The construction below therefore gives
# Pi_neq = -(2rho*tau/3)S and gamma = sqrt(2S:S) for traceless symmetric S.
# Relative tolerance 1e-12 allows only floating-point summation/subtraction error.
function _gnf3d_strain_lattice()
    lattice = D3Q19()
    return (KrakeNN.velocities_x(lattice), KrakeNN.velocities_y(lattice),
            KrakeNN.velocities_z(lattice)), KrakeNN.weights(lattice)
end

@testset "GNF model conversion before kernel launch" begin
    models = (Newtonian(0.08), PowerLaw(0.08, 0.7; nu_min=0.01, nu_max=0.4),
              CarreauYasuda(0.3, 0.01, 12.0, 2.0, 0.6), Cross(0.3, 0.01, 12.0, 0.6))
    for FT in (Float32, Float64), model in models
        converted = KrakeNN._gnf3d_rheology(FT, model)
        @test isbitstype(typeof(converted))
        @test converted.thermal isa IsothermalCoupling
        for name in fieldnames(typeof(model))
            name === :thermal && continue
            @test getfield(converted, name) isa FT
            @test getfield(converted, name) == FT(getfield(model, name))
        end
        @test effective_viscosity(converted, FT(0.02)) isa FT
    end
    thermal = ArrheniusCoupling(300.0, 10.0)
    for model in (Newtonian(0.08; thermal), PowerLaw(0.08, 0.7; thermal),
                  CarreauYasuda(0.3, 0.01, 12.0, 2.0, 0.6; thermal),
                  Cross(0.3, 0.01, 12.0, 0.6; thermal), Bingham(0.01, 0.08))
        @test_throws ArgumentError KrakeNN._gnf3d_rheology(Float32, model)
    end
end

function _gnf3d_strain_equilibrium(rho, u)
    c, w = _gnf3d_strain_lattice()
    return [w[q] * rho * (1 + 3sum(c[a][q] * u[a] for a in 1:3) +
            4.5sum(c[a][q] * u[a] for a in 1:3)^2 - 1.5sum(abs2, u)) for q in 1:19]
end

function _gnf3d_strain_populations(S, rho, tau, u)
    c, w = _gnf3d_strain_lattice()
    feq = _gnf3d_strain_equilibrium(rho, u)
    fneq = [-3w[q] * rho * tau * sum(
        (c[a][q] * c[b][q] - (a == b) / 3) * S[a,b]
        for a in 1:3, b in 1:3) for q in 1:19]
    return feq + fneq, feq
end

function _gnf3d_strain_moments(fneq)
    c, _ = _gnf3d_strain_lattice()
    return [sum(fneq[q] * c[a][q] * c[b][q] for q in 1:19)
            for a in 1:3, b in 1:3]
end

function _gnf3d_strain_from_tensor(Pi, rho, tau)
    return KrakeNN.strain_rate_from_pi_3d(
        Pi[1,1], Pi[2,2], Pi[3,3], Pi[1,2], Pi[1,3], Pi[2,3], rho, tau)
end

@testset "GNF D3Q19 strain-rate identities (CPU Float64)" begin
    rng = MersenneTwister(0x3d19)
    strains = Matrix{Float64}[]
    for (a, b) in ((1, 2), (1, 3), (2, 3))
        S = zeros(3, 3)
        S[a,b] = S[b,a] = 0.007
        push!(strains, S)
    end
    push!(strains, Matrix(Diagonal([0.012, -0.006, -0.006])))
    for _ in 1:8
        A = randn(rng, 3, 3)
        S = 0.005 .* (A + A')
        S -= (tr(S) / 3) * I
        push!(strains, S)
    end
    for S in strains, rho in (0.73, 1.0, 1.61), tau in (0.61, 0.95, 1.7),
            u in ((0.0, 0.0, 0.0), (0.027, -0.019, 0.013))
        f, feq = _gnf3d_strain_populations(S, rho, tau, u)
        Pi = _gnf3d_strain_moments(f - feq)
        expected = sqrt(2sum(abs2, S))
        @test isapprox(Pi, -(2rho * tau / 3) .* S; rtol=1e-12)
        @test isapprox(strain_rate_magnitude_3d(f..., feq..., rho, tau), expected;
                       rtol=1e-12)
        @test isapprox(_gnf3d_strain_from_tensor(Pi, rho, tau), expected; rtol=1e-12)
    end
end

@testset "GNF Guo half-force stress correction (CPU Float64)" begin
    # Homogeneous f=feq(rho,u) and u_star=u+F/(2rho) give the exact residual
    # Pi_neq + (F*u_star' + u_star*F')/2 = F*F'/(4rho).
    # Thus gamma = 3sqrt(2)|F|^2/(8rho^2*tau), with no term linear in F*u.
    # The 1e-7 relative gate and 2e-16 absolute moment gate account for
    # cancellation of O(1) equilibrium terms down to O(1e-8) moments.
    for rho in (0.73, 1.61), tau in (0.61, 1.7),
            u in ([0.0, 0.0, 0.0], [0.027, -0.019, 0.013])
        rates = Float64[]
        for amplitude in (1e-3, 5e-4, 2.5e-4)
            F = amplitude .* [1.0, -0.7, 0.4]
            u_star = u + F / (2rho)
            Pi_neq = _gnf3d_strain_moments(_gnf3d_strain_equilibrium(rho, u) -
                                         _gnf3d_strain_equilibrium(rho, u_star))
            Pi = Pi_neq + (F * u_star' + u_star * F') / 2
            expected_pi = F * F' / (4rho)
            expected_rate = 3sqrt(2) * sum(abs2, F) / (8rho^2 * tau)
            rate = _gnf3d_strain_from_tensor(Pi, rho, tau)
            @test isapprox(Pi, expected_pi; rtol=1e-7, atol=2e-16)
            @test isapprox(rate, expected_rate; rtol=1e-7)
            push!(rates, rate)
        end
        @test isapprox(rates[1] / rates[2], 4; rtol=1e-7)
        @test isapprox(rates[2] / rates[3], 4; rtol=1e-7)
    end
end
