using Test
using KrakeNN

const U = KrakeNN.Units

_geom_stab(; q_wall_dist=nothing) =
    (type=:cylinder_2d, blockage=0.5, L_up=15.0, L_down=15.0,
     q_wall_dist=q_wall_dist)

_bc_stab(; wall_bc=:halfwayBB) =
    (inlet=:velocity_parabolic, outlet=:zou_he_pressure,
     north_wall=:halfwayBB, south_wall=:halfwayBB, wall_bc=wall_bc)

function _compile_stab(; wall_bc=:halfwayBB, geom=_geom_stab(), tau_target=0.95,
                       strict=false)
    return U.compile(; physics=:viscoelastic, Re=1.0, Wi=1.0, beta=0.59,
                     R_LU=30, scaling=:diffusive, tau_target=tau_target,
                     geometry=geom, bc=_bc_stab(; wall_bc=wall_bc),
                     strict=strict)
end

@testset "Units stability registries" begin
    @testset "HalfwayBB predicate" begin
        bad = _compile_stab(; wall_bc=:halfwayBB, tau_target=1.6)
        @test :halfway_tau_above_ceiling in U.issue_codes(bad.warnings)
        good = _compile_stab(; wall_bc=:halfwayBB, tau_target=0.95, strict=true)
        @test isempty(good.warnings)
    end

    @testset "Bouzidi-FL predicate and q-wall warning" begin
        low_tau = _compile_stab(; wall_bc=:bouzidi_fl, tau_target=0.55)
        @test :bouzidi_tau_floor in U.issue_codes(low_tau.warnings)
        qwarn = _compile_stab(; wall_bc=:bouzidi_fl,
                              geom=_geom_stab(; q_wall_dist=[0.02, 0.5, 0.8]))
        @test :q_wall_near_cliff in U.issue_codes(qwarn.warnings)
    end

    @testset "custom wall extension seam" begin
        struct CustomWallBC <: U.AbstractWallBC end
        U.register_stability!(CustomWallBC, U.ViscoelasticSpec,
                              (units, geom) -> [U.warn_issue(:custom_wall_seen,
                                  "custom wall predicate was called")])
        plan = _compile_stab(; wall_bc=:custom_wall)
        @test :custom_wall_seen in U.issue_codes(plan.warnings)
    end
end
