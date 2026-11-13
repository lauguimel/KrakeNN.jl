using Test, KrakeNN, ForwardDiff, Random, LinearAlgebra

# DenseNet gates: bitwise/1e-14 equality with TinyMLP on TinyMLP's layout,
# handwritten pullback vs ForwardDiff (<= 1e-12) on mixed architectures,
# Dual inputs, zero allocation, load-time guards.
@testset "DenseNet" begin
    @testset "equals TinyMLP" begin
        for (nin, H, L) in ((1, 3, 1), (2, 4, 2), (3, 5, 3))
            shift = ntuple(i -> 0.1i, nin); scale = ntuple(i -> 1.0 + 0.3i, nin)
            tiny = TinyMLP{nin,H,L}(; shift, scale)
            dn = DenseNet(nin, ntuple(_ -> H, L); activation=:tanh, shift, scale)
            @test law_nparams(dn) == law_nparams(tiny)
            theta = law_init(tiny; rng=MersenneTwister(1)); theta[end] = 0.3
            for x in (ntuple(i -> 0.4i, nin), ntuple(i -> -1.3 + 0.2i, nin))
                @test law_value(dn, x, theta) == law_value(tiny, x, theta)
                g1 = zeros(length(theta)); g2 = zeros(length(theta))
                dx1 = law_pullback!(g1, dn, x, theta, -0.7)
                dx2 = law_pullback!(g2, tiny, x, theta, -0.7)
                @test norm(g1 - g2) <= 1e-14 * max(norm(g2), 1)
                @test norm(collect(dx1) .- collect(dx2)) <= 1e-14 * max(norm(collect(dx2)), 1)
            end
        end
    end

    archs = (
        (2, (5, 4), (:tanh, :softplus), :identity),
        (3, (6, 5, 4), (:relu, :sigmoid, :tanh), :exp),
        (1, (7,), :softplus, :softplus),
    )
    @testset "pullback vs ForwardDiff: arch $i" for (i, (nin, hidden, act, out)) in enumerate(archs)
        dn = DenseNet(nin, hidden; activation=act, output=out,
                      shift=ntuple(k -> 0.05k, nin), scale=ntuple(k -> 1.0 + 0.2k, nin))
        theta = law_init(dn; rng=MersenneTwister(10 + i))
        theta .+= 0.1 .* randn(MersenneTwister(20 + i), length(theta))  # nonzero biases
        x = ntuple(k -> 0.3k - 0.4, nin)
        gp = ForwardDiff.gradient(p -> law_value(dn, x, p), theta)
        gx = ForwardDiff.gradient(z -> law_value(dn, Tuple(z), theta), collect(x))
        gt = zeros(length(theta))
        dx = law_pullback!(gt, dn, x, theta, -0.7)
        @test norm(gt - (-0.7) * gp) <= 1e-12 * max(norm(gp), 1e-3)
        @test norm(collect(dx) - (-0.7) * gx) <= 1e-12 * max(norm(gx), 1e-3)
        law_pullback!(gt, dn, x, theta, 0.3)  # accumulates
        @test norm(gt - (-0.4) * gp) <= 1e-12 * max(norm(gp), 1e-3)
        # Dual inputs and Float32 propagate.
        d = ForwardDiff.Dual(x[1], 1.0)
        v = law_value(dn, (d, Base.tail(x)...), theta)
        @test v isa ForwardDiff.Dual
        @test isapprox(ForwardDiff.partials(v)[1], gx[1]; rtol=1e-12)
        @test law_value(DenseNet(nin, hidden; activation=act, output=out, shift=ntuple(_ -> 0f0, nin), scale=ntuple(_ -> 1f0, nin)),
                         Float32.(x), Float32.(theta)) isa Float32
        @test (@inferred law_value(dn, x, theta)) isa Float64
        g = zeros(length(theta))
        @test (@inferred law_pullback!(g, dn, x, theta, 1.0)) isa typeof(x)
        @test (@allocated law_value(dn, x, theta)) == 0
        @test (@allocated law_pullback!(g, dn, x, theta, 1.0)) == 0
    end

    @testset "guards" begin
        @test_throws ArgumentError DenseNet(2, (4, 4, 4, 4, 4))
        @test_throws ArgumentError DenseNet(2, (65,))
        @test DenseNet(2, (4, 4, 4, 4, 4); max_hidden_layers=5) isa DenseNet
        @test DenseNet(2, (128,); max_width=128) isa DenseNet
        @test_throws ArgumentError DenseNet(2, (4,); activation=:gelu)
        @test_throws ArgumentError DenseNet(2, (4,); output=:tanh)
        @test_throws ArgumentError DenseNet(2, (4, 4); activation=(:tanh,))
        @test isbitstype(typeof(DenseNet(2, (4, 4))))
    end
end
