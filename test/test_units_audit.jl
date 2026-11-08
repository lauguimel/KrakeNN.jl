using Test
using KrakeNN

const U = KrakeNN.Units

_geom_audit() = (type=:cylinder_2d, blockage=0.5, L_up=15.0, L_down=15.0)
_bc_audit() = (inlet=:velocity_parabolic, outlet=:zou_he_pressure,
               north_wall=:halfwayBB, south_wall=:halfwayBB, wall_bc=:halfwayBB)

@testset "Units audit" begin
    @testset "M59-B U-shape detection" begin
        raw = (R_LU=50, u_mean=0.005, nu_s=0.1475, nu_p=0.1025,
               lambda=10000.0, max_steps=300_000)
        plan = U.audit(raw; physics=:viscoelastic, geometry=_geom_audit(),
                       bc=_bc_audit())
        @test plan.units.tau_hydro ≈ 1.25 atol=1e-12 rtol=0
        @test :tau_above_magic in U.issue_codes(plan.warnings)
    end

    @testset "M48 toggle audit" begin
        raw = (R_LU=30, u_mean=0.005, nu_s=0.0885, nu_p=0.0615,
               lambda=6000.0, max_steps=180_000,
               embedded_gradient=true)
        plan = U.audit(raw; physics=:viscoelastic, geometry=_geom_audit(),
                       bc=_bc_audit())
        @test :m48_toggle in U.issue_codes(plan.warnings)
    end

    @testset "compile audit issue symmetry" begin
        compiled = U.compile(; physics=:viscoelastic, Re=1.0, Wi=1.0,
                             beta=0.59, R_LU=50, scaling=:acoustic,
                             geometry=_geom_audit(), bc=_bc_audit(),
                             strict=false)
        audited = U.audit(U.driver_kwargs(compiled); physics=:viscoelastic,
                          geometry=_geom_audit(), bc=_bc_audit())
        @test U.issue_codes(audited.warnings) == U.issue_codes(compiled.warnings)
    end
end
