# Plane-PIV observation operator (src/platform/piv_observation.jl), CPU Float64.
# Gates: exactness on linear fields (round-off), adjoint identity (round-off), gradient vs
# central finite differences, and a square-duct Poiseuille profile (analytical series; the
# sampled field is analytical, no 3D LBM run) sampled on a z-normal mid-plane.

using Test
using KrakeNN
using LinearAlgebra: dot
using Random: MersenneTwister

@testset "plane PIV 3D observation operator" begin
    N = (14, 11, 9)
    h = (0.5, 0.25, 0.4)
    ctr(d, i) = (i - 0.5) * h[d]
    a0 = (0.3, -0.7, 1.1); b = (0.9, -1.3, 0.6)
    # component c: a0[c] + b . x
    u = ntuple(c -> [a0[c] + b[1] * ctr(1, i) + b[2] * ctr(2, j) + b[3] * ctr(3, k)
                     for i in 1:N[1], j in 1:N[2], k in 1:N[3]], 3)
    exact(c, x, y, z) = a0[c] + b[1] * x + b[2] * y + b[3] * z
    L = (N[1] * h[1], N[2] * h[2], N[3] * h[3])

    @testset "linear field exact, any thickness, off-centre planes" begin
        for (nrm, pos, xs, ys, sig) in (
                (:x, 2.77, 1.0:0.37:2.5, 0.9:0.41:2.0, 0.0),
                (:x, 2.77, 1.0:0.37:2.5, 0.9:0.41:2.0, 0.3),
                (:y, 1.33, 1.2:0.5:5.0, 0.8:0.3:2.2, 0.0),
                (:y, 1.33, 1.2:0.5:5.0, 0.8:0.3:2.2, 0.15),
                (:z, 1.71, 1.1:0.6:5.5, 0.6:0.33:2.0, 0.0),
                (:z, 1.71, 1.1:0.6:5.5, 0.6:0.33:2.0, 0.25))
            pl = PIVPlane(; normal=nrm, position=pos, xs=xs, ys=ys, window=(0.3, 0.2),
                          thickness=sig, stereo=true)
            op = piv_operator(pl, u[1]; h=h)
            s = piv_apply(op, u)
            @test length(s) == 3
            comps = nrm == :x ? (2, 3, 1) : nrm == :y ? (1, 3, 2) : (1, 2, 3)
            err = 0.0
            for (c, comp) in enumerate(comps), (p, xa) in enumerate(xs), (q, yb) in enumerate(ys)
                xyz = nrm == :x ? (pos, xa, yb) : nrm == :y ? (xa, pos, yb) : (xa, yb, pos)
                err = max(err, abs(s[c][p, q] - exact(comp, xyz...)))
            end
            @test err < 1e-12
        end
    end

    @testset "adjoint identity" begin
        rng = MersenneTwister(7)
        for nrm in (:x, :y, :z), sig in (0.0, 0.3), stereo in (false, true), usemask in (false, true)
            pl = PIVPlane(; normal=nrm, position=1.6, xs=0.8:0.45:3.0, ys=0.7:0.35:1.9,
                          window=(0.4, 0.3), thickness=sig, stereo=stereo)
            f = ntuple(_ -> randn(rng, N...), 3)
            op = piv_operator(pl, f[1]; h=h)
            s = piv_apply(op, f)
            g = ntuple(c -> randn(rng, size(s[c])...), length(s))
            if usemask   # a mask zeroes cotangent entries (misfit path); here weight both sides
                mk = ntuple(c -> Float64.(rand(rng, size(s[c])...) .> 0.3), length(s))
                g = ntuple(c -> mk[c] .* g[c], length(s))
            end
            ft = piv_apply_adjoint(op, g; template=f[1])
            lhs = sum(c -> dot(s[c], g[c]), 1:length(s))
            rhs = sum(d -> dot(f[d], ft[d]), 1:3)
            @test abs(lhs - rhs) < 1e-12 * max(1, abs(lhs))
        end
    end

    @testset "misfit gradient vs central finite differences" begin
        rng = MersenneTwister(11)
        for nrm in (:x, :y, :z)
            pl = PIVPlane(; normal=nrm, position=1.55, xs=0.8:0.45:3.0, ys=0.7:0.35:1.9,
                          window=(0.4, 0.3), thickness=0.2, stereo=true)
            f = ntuple(_ -> randn(rng, N...), 3)
            op = piv_operator(pl, f[1]; h=h)
            s = piv_apply(op, f)
            data = ntuple(c -> randn(rng, size(s[c])...), 3)
            mask = ntuple(c -> Float64.(rand(rng, size(s[c])...) .> 0.25), 3)
            data[1][1, 1] = NaN; mask[1][1, 1] = 0.0   # missing vector
            Lv, gr = piv_misfit(op, f, data; mask=mask)
            @test isfinite(Lv)
            ε = 1e-5
            for _ in 1:6
                d = rand(rng, 1:3); I = CartesianIndex(rand(rng, 1:N[1]), rand(rng, 1:N[2]), rand(rng, 1:N[3]))
                fp = map(copy, f); fm = map(copy, f)
                fp[d][I] += ε; fm[d][I] -= ε
                fd = (piv_misfit(op, fp, data; mask=mask)[1] - piv_misfit(op, fm, data; mask=mask)[1]) / 2ε
                @test abs(fd - gr[d][I]) < 1e-7 * max(1, abs(fd))
            end
        end
    end

    @testset "type stability" begin
        pl = PIVPlane(; normal=:z, position=1.0, xs=1.0:0.5:3.0, ys=1.0:0.5:2.0, thickness=0.2)
        op = piv_operator(pl, u[1]; h=h)
        @inferred piv_apply(op, u)
        data = piv_apply(op, u)
        @inferred piv_misfit(op, u, data)
        @inferred piv_apply_adjoint(op, data; template=u[1])
    end

    @testset "square duct (analytical series) on a z-normal mid-plane" begin
        # flow along z in a square duct of side 2a, uz(x, y) = series; ux = uy = 0.
        a = 1.0; G = 1.0; μ = 1.0
        function duct(x, y)   # x, y in [-a, a]
            acc = 0.0
            for n in 1:2:199
                acc += (-1)^((n - 1) ÷ 2) * (1 - cosh(n * π * y / (2a)) / cosh(n * π * a / (2a))) *
                     cos(n * π * x / (2a)) / n^3
            end
            return 16a^2 * G / (μ * π^3) * acc
        end
        n = 64; hx = 2a / n
        xc = [(i - 0.5) * hx - a for i in 1:n]
        nz = 6
        uz = [duct(xc[i], xc[j]) for i in 1:n, j in 1:n, k in 1:nz]
        zero3 = zero(uz)
        xs = range(-0.8, 0.8; length=9) .+ a   # physical coordinates measured from the wall
        pl = PIVPlane(; normal=:z, position=0.31 * hx * nz, xs=xs, ys=xs, thickness=0.5hx * nz,
                      stereo=true)
        op = piv_operator(pl, uz; h=(hx, hx, hx))
        s = piv_apply(op, (zero3, zero3, uz))
        ref = [duct(x - a, y - a) for x in xs, y in xs]
        @test maximum(abs.(s[1])) == 0 && maximum(abs.(s[2])) == 0
        @test maximum(abs.(s[3] .- ref)) / maximum(ref) < 2e-3
            end
end
