# Geometry-only C5 checks; no flow solve or convergence campaign.
# Roundoff gates use 512 eps(T): the reference and FV quadratics operate on
# coordinates up to 32 grid units with different origin/scale conversions.
using Test
using KrakeNN
using ForwardDiff

function _c5_geometry_error(f)
    try
        f()
        return ""
    catch e
        e isa ArgumentError || rethrow()
        return sprint(showerror, e)
    end
end

@testset "C5 immersed analytic geometry" begin
    lbm_directions = (2, 4, 3, 5) # FV E,W,N,S -> D2Q9 E,W,N,S
    for T in (Float32, Float64), h0 in (1, 0.2)
        h = T(h0)
        cx, cy, R = T(16.13) * h, T(15.87) * h, T(5.21) * h
        b = KrakeNN.fv_immersed_circle(32, 32, cx, cy, R; h)
        ref, solid = KrakeNN.precompute_q_wall_cylinder(32, 32, cx / h - T(0.5),
                                                       cy / h - T(0.5), R / h; FT=T)
        @test eltype(b.qcut) == T
        @test size(b.qcut) == (4, 32, 32)
        @test size(b.wvel) == (2, 4, 32, 32)
        @test (b.tag .== FV_CELL_SOLID) == solid
        @test all(iszero, b.wvel)
        for d in 1:4
            @test maximum(abs, b.qcut[d, :, :] .- ref[:, :, lbm_directions[d]]) <= 512eps(T)
        end
        Ri, Ro = T(3.21) * h, T(9.21) * h
        a = KrakeNN.fv_immersed_annulus(32, 32, cx, cy, Ri, Ro; h,
            wall_velocity=(wall, x, y) -> wall == 1 ? (-y, x) : (x, -y))
        aref, asolid, inner = KrakeNN.precompute_q_wall_annulus(32, 32,
            cx / h - T(0.5), cy / h - T(0.5), Ri / h, Ro / h; FT=T)
        @test (a.tag .== FV_CELL_SOLID) == asolid
        for d in 1:4
            @test maximum(abs, a.qcut[d, :, :] .- aref[:, :, lbm_directions[d]]) <= 512eps(T)
        end
        velocity_error = zero(T)
        for j in 1:32, i in 1:32, d in 1:4
            q = a.qcut[d, i, j]
            q == 0 && continue
            x = (T(i) - T(0.5) + q * ((d == 1) - (d == 2))) * h
            y = (T(j) - T(0.5) + q * ((d == 3) - (d == 4))) * h
            uv = inner[i, j, lbm_directions[d]] ? (-y, x) : (x, -y)
            velocity_error = max(velocity_error, abs(a.wvel[1, d, i, j] - uv[1]),
                                  abs(a.wvel[2, d, i, j] - uv[2]))
        end
        @test velocity_error <= 512eps(T) * 32h
    end

    @testset "disjoint union and frozen membership" begin
        circles = ((10.13, 15.87, 3.21), (23.13, 15.87, 3.21))
        u = KrakeNN.fv_immersed_boundary(36, 32; circles)
        a = KrakeNN.fv_immersed_circle(36, 32, circles[1]...)
        b = KrakeNN.fv_immersed_circle(36, 32, circles[2]...)
        @test (u.tag .== FV_CELL_SOLID) == ((a.tag .== FV_CELL_SOLID) .| (b.tag .== FV_CELL_SOLID))
        @test u.qcut == a.qcut + b.qcut
        frozen = copy(u.tag)
        frozen[1, 1] = FV_CELL_PINNED
        same = KrakeNN.fv_immersed_boundary(36, 32; circles, frozen_tag=frozen)
        @test same.tag == u.tag
        @test same.qcut == u.qcut
        changed = ((10.63, 15.87, 3.21), circles[2])
        message = _c5_geometry_error(() -> KrakeNN.fv_immersed_boundary(36, 32;
            circles=changed, frozen_tag=frozen))
        @test occursin("frozen fluid/solid tag changed", message)
        @test occursin("cells (", message)
        @test_throws DimensionMismatch KrakeNN.fv_immersed_boundary(36, 32;
            circles, frozen_tag=zeros(UInt8, 2, 2))
    end

    @testset "dual geometry and independent dual wall speed" begin
        for h in (1.0, 0.2)
            cx, cy, R = 16.13h, 15.87h, 5.21h
            b = KrakeNN.fv_immersed_circle(32, 32, cx, cy, R; h)
            db = KrakeNN.fv_immersed_circle(32, 32, cx, cy, ForwardDiff.Dual(R, 1.0);
                                         h, frozen_tag=b.tag)
            ref = KrakeNN.dq_wall_dR_cylinder(32, 32, cx / h - 0.5, cy / h - 0.5, R / h)
            @test db.tag == b.tag
            @test ForwardDiff.value.(db.qcut) == b.qcut
            for d in 1:4
                @test maximum(abs, ForwardDiff.partials.(db.qcut[d, :, :], 1) .-
                                   ref[:, :, lbm_directions[d]] ./ h) <= 2e-12 / h
            end
        end
        speed = ForwardDiff.Dual(2.0, 1.0)
        b = KrakeNN.fv_immersed_circle(12, 12, 6.13, 5.87, 2.21;
            wall_velocity=(wall, x, y) -> (speed * y, -speed * x))
        @test eltype(b.qcut) == Float64
        @test eltype(b.wvel) <: ForwardDiff.Dual
        @test any(!iszero, ForwardDiff.partials.(b.wvel, 1))
        host = fv_adapt(Array, b)
        @test host.qcut == b.qcut
        @test host.wvel == b.wvel
        @test host.tag == b.tag
        @test host.h == b.h
        @test host.closure == b.closure
    end

    @testset "solid endpoint q=1 is retained and differentiated" begin
        # Binary-scaled 3-4-5 triangle gives an exact non-tangent centre hit.
        cx, cy, R = 12.8125, 12.25, 2.8125
        b = KrakeNN.fv_immersed_circle(28, 28, cx, cy, R)
        @test b.tag[15, 15] == FV_CELL_SOLID
        @test b.qcut[2, 16, 15] == 1
        for seed in (-1.0, 1.0)
            d = KrakeNN.fv_immersed_circle(28, 28, cx, cy, ForwardDiff.Dual(R, seed);
                                         frozen_tag=b.tag)
            @test d.tag == b.tag
            @test ForwardDiff.value(d.qcut[2, 16, 15]) == 1
            @test ForwardDiff.partials(d.qcut[2, 16, 15], 1) ≈ -seed * 5 / 3 atol=4eps()
        end
        msg = _c5_geometry_error(() -> KrakeNN.fv_immersed_circle(28, 28, cx, cy, R - 1e-7;
                                                                 frozen_tag=b.tag))
        @test occursin("frozen fluid/solid tag changed", msg)
        # A newly fluid centre can be arbitrarily close to the wall. The
        # quadratic root keeps this small positive q without a numerical floor.
        smallR = R - 2.0^-32
        tiny = KrakeNN.fv_immersed_circle(28, 28, cx, cy, smallR)
        @test 0 < tiny.qcut[2, 15, 15] < 1e-8
        tinydual = KrakeNN.fv_immersed_circle(28, 28, cx, cy, ForwardDiff.Dual(smallR, 1.0);
                                            frozen_tag=tiny.tag)
        @test ForwardDiff.partials(tinydual.qcut[2, 15, 15], 1) ≈
              -smallR / sqrt(smallR^2 - (14.5 - cy)^2) atol=8eps()
    end

    @testset "resolution contract rejects and identifies cells" begin
        checks = (
            ("radius", () -> KrakeNN.fv_immersed_circle(32, 32, 16.13, 15.87, 1.9)),
            ("domain gap", () -> KrakeNN.fv_immersed_circle(32, 32, 4.0, 15.87, 2.21)),
            ("walls 1/2 gap", () -> KrakeNN.fv_immersed_boundary(32, 32;
                circles=((12.13, 15.87, 3.21), (20.13, 15.87, 3.21)))),
            ("walls 1/2 gap", () -> KrakeNN.fv_immersed_annulus(32, 32, 16.13, 15.87, 3.21, 5.21)),
            ("domain gap", () -> KrakeNN.fv_immersed_annulus(32, 32, 16.13, 15.87, 3.21, 13.21)),
            ("boundary intersections", () -> KrakeNN.fv_immersed_circle(32, 32, 16.0, 16.48, 10.021)),
            ("boundary intersections", () -> KrakeNN.fv_immersed_circle(32, 32, 16.0, 16.5, 5.0)),
            ("opposite cell", () -> KrakeNN.fv_immersed_annulus(36, 36, 16.5, 16.5, 3.21, 10.01)),
            ("transversality", () -> KrakeNN.fv_immersed_circle(32, 32, 16.13, 15.87, 5.21;
                                                            min_transversality=0.99)),
        )
        for (expected, f) in checks
            message = _c5_geometry_error(f)
            @test occursin(expected, message)
            @test occursin("cells (", message)
        end
        @test_throws ArgumentError KrakeNN.fv_immersed_circle(32, 32, 16.13, 15.87, 5.21;
                                                            min_transversality=-1)
        @test_throws ArgumentError KrakeNN.fv_immersed_circle(32, 32, 16.13, 15.87, 5.21;
                                                            min_transversality=1)
    end
end
