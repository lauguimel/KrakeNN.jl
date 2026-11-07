using Test
using Kraken
using ForwardDiff
using Random
using LinearAlgebra

# Frozen L1 gates: independent ForwardDiff derivatives agree to 1e-12 in
# relative norm; classical values reproduce gnf_viscosity to 1e-14 relative.
# L1b initialization uses the analytical tanh residual bound around the
# normalized-slope baseline; it is nu_ref only for identity shear normalization.
# Derivative, classical, positivity, monotonicity and runtime gates are unchanged.
function _law_check_derivatives(law, x, theta; seed=-0.7)
    gp = ForwardDiff.gradient(p -> law_value(law, x, p), theta)
    gx = ForwardDiff.gradient(z -> law_value(law, Tuple(z), theta), collect(x))
    gtheta = zeros(eltype(theta), length(theta))
    dx = law_pullback!(gtheta, law, x, theta, seed)
    @test norm(gtheta - seed * gp) <= 1e-12 * max(norm(seed * gp), eps())
    @test norm(collect(dx) - seed * gx) <= 1e-12 * max(norm(seed * gx), eps())
    dx2 = law_pullback!(gtheta, law, x, theta, 0.3)
    @test norm(gtheta - (seed + 0.3) * gp) <= 1e-12 * max(norm(gp), eps())
    @test norm(collect(dx2) - 0.3 * gx) <= 1e-12 * max(norm(0.3 * gx), eps())
end

function _law_check_runtime(law, x, theta)
    gtheta = zeros(eltype(theta), length(theta))
    @test (@inferred law_value(law, x, theta)) isa eltype(theta)
    @test (@inferred law_pullback!(gtheta, law, x, theta, one(eltype(theta)))) isa typeof(x)
    @test (@allocated law_value(law, x, theta)) == 0
    @test (@allocated law_pullback!(gtheta, law, x, theta, one(eltype(theta)))) == 0
end

# Return only weight entries: every hidden bias and the output bias is excluded.
function _law_weight_indices(::TinyMLP{N,H,L}, base=0) where {N,H,L}
    indices = Int[]
    offset = base
    for k in 1:L
        count = H*(k == 1 ? N : H)
        append!(indices, offset+1:offset+count)
        offset += count+H
    end
    append!(indices, offset+1:offset+H)
    return indices
end

# Dependency-free full-batch Adam; the production handwritten pullback supplies
# the gradient of half the mean squared log-viscosity error.
function _law_fit_adam!(theta, law, inputs, targets, steps)
    gradient = zeros(length(theta))
    first_moment = zero(gradient)
    second_moment = zero(gradient)
    for step in 1:steps
        fill!(gradient, 0)
        for (x, target) in zip(inputs, targets)
            nu = law_value(law, x, theta)
            law_pullback!(gradient, law, x, theta,
                          (log(nu)-target)/(length(inputs)*nu))
        end
        for j in eachindex(theta)
            first_moment[j] = 0.9*first_moment[j] + 0.1*gradient[j]
            second_moment[j] = 0.999*second_moment[j] + 0.001*gradient[j]^2
            theta[j] -= 0.01*(first_moment[j]/(1-0.9^step))/
                        (sqrt(second_moment[j]/(1-0.999^step))+1e-8)
        end
    end
    return theta
end

# Analytical Carreau-Yasuda viscosity times exp(E*(1/T-1/T0)):
# eta0=2, eta_inf=0.2, lambda=1.4, a=1.8, n=0.55, E=1.2, T0=1.
_law_fit_target(gamma, temperature) =
    log(0.2 + (2.0-0.2)*(1+(1.4*gamma)^1.8)^((0.55-1)/1.8)) +
    1.2*(1/temperature-1)

struct _LawPolynomial <: AbstractScalarLaw end
Kraken.law_nparams(::_LawPolynomial) = 2
Kraken.law_ninputs(::_LawPolynomial) = 2
Kraken.law_value(::_LawPolynomial, x::Tuple, theta::AbstractVector) =
    theta[1] * x[1]^2 + theta[2] * x[1] * x[2]

_law_rebuild(m::PowerLaw, p) =
    PowerLaw(p[1], p[2]; nu_min=p[3], nu_max=p[4], thermal=m.thermal)
_law_rebuild(m::CarreauYasuda, p) =
    CarreauYasuda(p[1], p[2], p[3], p[4], p[5]; thermal=m.thermal)

@testset "L1 scalar laws" begin
    rng = MersenneTwister(807)
    mlp = TinyMLP{3,3,2}(; shift=(0.2, 300.0, 0.4), scale=(1.7, 30.0, 0.7))
    laws = (mlp, LearnedViscosity(mlp; nu_ref=0.8),
            LearnedViscosity(mlp; nu_ref=0.8, monotone=false))
    gradient_laws = laws

    @testset "dimensions, constructors and initialization" begin
        @test law_ninputs(mlp) == 3
        @test law_nparams(mlp) == 28
        @test law_nparams(laws[2]) == 28 + 25 + 1
        @test law_nparams(laws[3]) == 28
        @test mlp isa AbstractClosure
        @test isbitstype(typeof(mlp))
        @test isbitstype(typeof(laws[2]))
        @test_throws ArgumentError TinyMLP{3,3,2}(; scale=(0.0, 1.0, 1.0))
        @test_throws ArgumentError LearnedViscosity(TinyMLP{1,1,1}(; scale=(-1.0,)))
        @test_throws ArgumentError LearnedViscosity(mlp; epsilon=0.0)
        @test_throws ArgumentError LearnedViscosity(mlp; nu_ref=0.0)
        for law in laws
            @test length(law_init(law)) == law_nparams(law)
            @test law_init(law; rng=MersenneTwister(41)) ==
                  law_init(law; rng=MersenneTwister(41))
        end
        @test maximum(abs, law_init(mlp; rng=MersenneTwister(41))) < 1
        @test laws[2].field_mlp.shift == Base.tail(mlp.shift)
        @test laws[2].field_mlp.scale == Base.tail(mlp.scale)
        @test laws[3].field_mlp === nothing
        for law in (laws[2], laws[3], LearnedViscosity(mlp; nu_ref=0.8, epsilon=0.9999),
                    LearnedViscosity(TinyMLP{1,3,2}(); nu_ref=0.8))
            init_rng = MersenneTwister(9)
            theta = law_init(law; rng=MersenneTwister(9))
            raw_h = law_init(law.mlp; rng=init_rng)
            nh = law_nparams(law.mlp)
            ho = Kraken._mlp_output_offset(law.mlp)
            width = Kraken._mlp_width(law.mlp)
            @test theta[1:ho] == raw_h[1:ho]
            @test theta[nh] == 0
            if law.monotone
                @test all(isapprox.(Kraken._law_softplus.(theta[ho+1:ho+width]), 0.01;
                                   rtol=8eps(), atol=0))
                @test isapprox(law.epsilon + Kraken._law_softplus(theta[end]), 1; rtol=8eps())
                bound = width*0.01
                if law.field_mlp === nothing
                    @test law_nparams(law) == nh + 1
                else
                    raw_g = law_init(law.field_mlp; rng=init_rng)
                    go = Kraken._mlp_output_offset(law.field_mlp)
                    @test theta[nh+1:nh+go] == raw_g[1:go]
                    @test theta[nh+go+1:nh+go+width] == 0.2 .* raw_g[go+1:go+width]
                    @test theta[end-1] == 0
                    bound += sum(abs, theta[nh+go+1:nh+go+width])
                end
            else
                @test theta[ho+1:ho+width] == 0.2 .* raw_h[ho+1:ho+width]
                bound = sum(abs, theta[ho+1:ho+width])
            end
            # |tanh| <= 1 gives this bound without a fitted numerical gate.
            # The 64eps allowance covers only floating-point evaluation.
            for gamma in 10.0 .^ range(-4, 4; length=41),
                t in (270.0, 300.0, 330.0), c in (0.0, 1.0)
                x = law_ninputs(law) == 1 ? (gamma,) : (gamma, t, c)
                u1 = (log(gamma)-law.mlp.shift[1])/law.mlp.scale[1]
                baseline = law.nu_ref * exp(law.monotone ? u1-log(gamma) : 0.0)
                law_ninputs(law) == 1 && @test baseline == law.nu_ref
                ratio = law_value(law, x, theta)/baseline
                @test exp(-bound)*(1-64eps()) <= ratio <= exp(bound)*(1+64eps())
            end
        end
    end

    @testset "manual reverse, dual inputs and accumulation" begin
        for law in gradient_laws, trial in 1:5
            theta = 0.8 .* randn(rng, law_nparams(law))
            x = (exp(randn(rng)), 300 + 20 * randn(rng), rand(rng))
            _law_check_derivatives(law, x, theta)
        end
        _law_check_derivatives(_LawPolynomial(), (0.7, -1.2), [0.4, -0.8])
        _law_check_runtime(_LawPolynomial(), (0.7, -1.2), [0.4, -0.8])
        asymmetric = TinyMLP{2,3,1}(; shift=(0.2, 0.3), scale=(2.0, 0.5))
        for trial in 1:3
            _law_check_derivatives(asymmetric, Tuple(randn(rng, 2)),
                                   0.8 .* randn(rng, law_nparams(asymmetric)))
        end
        _law_check_runtime(asymmetric, (0.7, -1.2), law_init(asymmetric; rng=rng))
        single = LearnedViscosity(TinyMLP{1,3,2}(; shift=(0.3,), scale=(1.7,)))
        @test single.field_mlp === nothing
        @test law_nparams(single) == law_nparams(single.mlp)+1
        _law_check_derivatives(single, (0.7,), randn(rng, law_nparams(single)))
        _law_check_runtime(single, (0.7,), law_init(single; rng=rng))
        for law in gradient_laws
            _law_check_runtime(law, (0.7, 305.0, 0.2), law_init(law; rng=rng))
        end
    end

    @testset "default conditioning and short synthetic fit" begin
        # Log shear uses identity normalization; temperature maps [0.8,1.2]
        # onto [-1,1]. The same seeded default initialization is used for both
        # checks, without changing weights before the conditioning measurement.
        law = LearnedViscosity(TinyMLP{2,8,1}(; shift=(0.0, 1.0), scale=(1.0, 0.2)))
        theta = law_init(law; rng=MersenneTwister(807))
        nh = law_nparams(law.mlp)
        ng = law_nparams(law.field_mlp)
        hweights = _law_weight_indices(law.mlp)
        gweights = _law_weight_indices(law.field_mlp, nh)
        gradient = zeros(length(theta))
        minima = fill(Inf, 4)
        for gamma in 10.0 .^ range(-2, 2; length=21),
            temperature in range(0.8, 1.2; length=9)
            x = (gamma, temperature)
            fill!(gradient, 0)
            law_pullback!(gradient, law, x, theta, inv(law_value(law, x, theta)))
            # Euclidean norms of dlog(nu)/dtheta, per point, for the entire
            # h/g blocks and for their weights alone (exclude ALL biases).
            norms = (norm(gradient[1:nh]), norm(gradient[nh+1:nh+ng]),
                     norm(gradient[hweights]), norm(gradient[gweights]))
            for i in 1:4
                @test norms[i] >= 1e-3 # L1b conditioning acceptance gate.
                minima[i] = min(minima[i], norms[i])
            end
        end
        @info "L1b initialization conditioning" min_h=minima[1] min_g=minima[2] min_h_weights=minima[3] min_g_weights=minima[4]

        # Fit is chosen over a spectral surrogate: it exercises actual
        # parameter updates with both shear and thermal dependence. 600 Adam
        # steps and the 0.05 relative RMS log(nu) gate are frozen L1b criteria.
        inputs = [(g, t) for g in 10.0 .^ range(-2, 2; length=21)
                         for t in range(0.8, 1.2; length=5)]
        targets = [_law_fit_target(x...) for x in inputs]
        _law_fit_adam!(copy(theta), law, inputs, targets, 1) # Compilation warmup.
        elapsed = @elapsed _law_fit_adam!(theta, law, inputs, targets, 600)
        validation = [(g, t) for g in 10.0 .^ range(-2, 2; length=30)
                             for t in range(0.8, 1.2; length=9)]
        reference = [_law_fit_target(x...) for x in validation]
        predicted = [log(law_value(law, x, theta)) for x in validation]
        relative_rms = norm(predicted-reference)/norm(reference)
        @test relative_rms < 0.05
        @test elapsed < 10 # Warmed CPU scalar fit only; no flow simulation.
        @info "L1b synthetic fit" steps=600 relative_rms_lognu=relative_rms warmed_seconds=elapsed
    end

    @testset "classical parameterization and legacy parity" begin
        models = (PowerLaw(0.8, 0.65; nu_min=0.1, nu_max=4.0,
                           thermal=ArrheniusCoupling(300.0, 1200.0)),
                  CarreauYasuda(1.2, 0.04, 2.0, 1.6, 0.55))
        for model in models
            law = ClassicalLaw(model)
            theta = law_init(law)
            @test theta == [getfield(model, i) for i in 1:(fieldcount(typeof(model)) - 1)]
            @test law_ninputs(law) == 1
            @test law_nparams(law) == length(theta)
            # Rebuild the legacy model after perturbing theta: the adapter must
            # use the supplied parameters, not the ones captured at construction.
            for trial in 1:3
                p = theta .* (0.9 .+ 0.2 * rand(rng, length(theta)))
                reference = _law_rebuild(model, p)
                for g in (1e-4, 0.2, 1.0, 5.0, 1e4)
                    @test isapprox(law_value(law, (g,), p),
                                   Kraken.gnf_viscosity(reference, g^2 / 2); rtol=1e-14, atol=0)
                    _law_check_derivatives(law, (g,), p)
                end
            end
            _law_check_runtime(law, (0.7,), theta)
        end
    end

    @testset "positive viscosity and increasing stress" begin
        law = LearnedViscosity(mlp; nu_ref=0.8)
        gamma = 10.0 .^ range(-4, 4; length=41)
        for trial in 1:5
            theta = 3 .* randn(rng, law_nparams(law))
            trial == 1 && (theta[end] = -20.0)
            for t in (270.0, 300.0, 330.0), c in (0.0, 1.0)
                viscosity = [law_value(law, (g, t, c), theta) for g in gamma]
                stress = gamma .* viscosity
                slopes = [ForwardDiff.derivative(g -> g * law_value(law, (g, t, c), theta), g)
                          for g in gamma]
                @test all(isfinite, viscosity) && all(>(0), viscosity)
                @test all(>(0), diff(stress))
                @test all(isfinite, slopes) && all(>(0), slopes)
            end
        end
        # A stress slope below one in log coordinates permits shear thinning.
        thin = LearnedViscosity(TinyMLP{1,1,1}(); epsilon=0.01)
        theta = law_init(thin; rng=rng)
        theta[end] = log(expm1(0.5 - 0.01))
        @test law_value(thin, (10.0,), theta) < law_value(thin, (0.1,), theta)
        @test ForwardDiff.derivative(g -> g * law_value(thin, (g,), theta), 1.0) > 0

        # Temperature and concentration columns remain unconstrained, so both
        # can lower viscosity while the shear-rate stress remains monotone.
        fields = LearnedViscosity(TinyMLP{3,1,1}())
        # h: 6 parameters; g: 5 parameters; raw slope: last.
        pfields = [0.0, -1.0, -1.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        @test ForwardDiff.derivative(t -> law_value(fields, (1.0, t, 0.0), pfields), 0.0) < 0
        @test ForwardDiff.derivative(c -> law_value(fields, (1.0, 0.0, c), pfields), 0.0) < 0

        # Isolate g: h has zero field weights. Unrestricted g output weights
        # permit either sign of the temperature and concentration effects.
        pfields[2:3] .= 0
        pfields[7:11] .= [1.0, -1.0, 0.0, 1.0, 0.0]
        for sign in (-1.0, 1.0)
            pfields[10] = sign
            @test sign*ForwardDiff.derivative(t -> law_value(fields, (1.0, t, 0.0), pfields), 0.0) > 0
            @test sign*ForwardDiff.derivative(c -> law_value(fields, (1.0, 0.0, c), pfields), 0.0) < 0
            _law_check_derivatives(fields, (0.8, 0.2, -0.3), pfields)
            without_g = copy(pfields)
            without_g[10] = 0
            for gamma in (0.2, 5.0)
                x = (gamma, 0.2, -0.3)
                @test isapprox(log(law_value(fields, x, pfields)/law_value(fields, x, without_g)),
                               sign*tanh(0.5); rtol=1e-12, atol=0)
            end
        end

        # Explicit unrestricted law: log(nu) = -2tanh(log(gamma)). Its stress
        # decreases at gamma=1, so monotone=false must not enforce monotonicity.
        free = LearnedViscosity(TinyMLP{1,1,1}(); monotone=false)
        p = [1.0, 0.0, -2.0, 0.0]
        _law_check_derivatives(free, (0.7,), p)
        _law_check_runtime(free, (0.7,), p)
        @test ForwardDiff.derivative(g -> g * law_value(free, (g,), p), 1.0) < 0
        @test 1.1 * law_value(free, (1.1,), p) < 0.9 * law_value(free, (0.9,), p)
    end
end
