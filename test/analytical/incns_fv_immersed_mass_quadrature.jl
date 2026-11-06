# Regression: an implicitly inherited M3 flux keeps its quadrature estimate;
# an explicitly prescribed physical target supersedes the numerical estimate.
# Harmonic 128 aliases to a constant on 128 angular samples and integrates
# to roundoff on 256 samples, giving a nonzero error estimate of 2pi*radius.
using Test, Kraken

@testset "M3 inherited physical flux retains quadrature uncertainty" begin
    cx, cy, radius = 6.13, 5.87, 2.21
    function velocity(wall,x,y)
        harmonic = cos(128atan(y-cy,x-cx))
        return ((x-cx)*harmonic/radius,(y-cy)*harmonic/radius)
    end
    ib = fv_immersed_circle(12,12,cx,cy,radius;h=1.0,wall_velocity=velocity)
    p = @test_logs (:warn,r"M3 physical wall_flux uses circle quadrature") fv_steady_problem(
        12,12;h=1.0,immersed=ib,conv=false)
    target, uncertainty = copy(p.immersed.mass.Q), copy(p.immersed.mass.quadrature_error)
    @test only(uncertainty) ≈ 2pi*radius rtol=1e-11
    @test abs(only(target)) <= 1e-11

    rebuilt = fv_steady_problem(12,12;h=1.0,immersed=p.immersed,conv=false)
    @test rebuilt.immersed.mass.Q == target
    @test rebuilt.immersed.mass.quadrature_error == uncertainty

    prescribed = fv_steady_problem(12,12;h=1.0,immersed=p.immersed,conv=false,wall_flux=0.0)
    @test all(iszero,prescribed.immersed.mass.Q)
    @test all(iszero,prescribed.immersed.mass.quadrature_error)

    adapted = fv_adapt(Array,rebuilt)
    @test adapted.immersed.mass.Q == target
    @test adapted.immersed.mass.quadrature_error == uncertainty
    @test adapted.immersed.mass.quadrature_error !== rebuilt.immersed.mass.quadrature_error
end
