using Test
using KrakeNN

const U = KrakeNN.Units

_geom(; q_wall_dist=nothing) =
    (type=:cylinder_2d, blockage=0.5, L_up=15.0, L_down=15.0,
     q_wall_dist=q_wall_dist)

_bc(; wall_bc=:halfwayBB) =
    (inlet=:velocity_parabolic, outlet=:zou_he_pressure,
     north_wall=:halfwayBB, south_wall=:halfwayBB, wall_bc=wall_bc)

function _ve_plan(; R=30, scaling=:diffusive, tau_target=0.95, T=Float64,
                  strict=true, kwargs...)
    return U.compile(; physics=:viscoelastic, Re=1.0, Wi=1.0, beta=0.59,
                     bsd_fraction=1.0, model=:oldroyd_b, R_LU=R,
                     scaling=scaling, tau_target=tau_target,
                     geometry=_geom(), bc=_bc(), T=T, strict=strict,
                     kwargs...)
end

function _assert_units_close(a, b; atol=1e-12)
    @test a.tau_hydro ≈ b.tau_hydro atol=atol rtol=0
    @test a.nu_total_LU ≈ b.nu_total_LU atol=atol rtol=0
    @test a.u_LU ≈ b.u_LU atol=atol rtol=0
    @test a.R_LU == b.R_LU
    @test a.Ma ≈ b.Ma atol=atol rtol=0
    @test a.scaling == b.scaling
    @test a.max_steps == b.max_steps
    @test isequal(a.nu_s_LU, b.nu_s_LU)
    @test isequal(a.nu_p_LU, b.nu_p_LU)
    @test isequal(a.lambda_LU, b.lambda_LU)
end

@testset "Units forward and reverse" begin
    @testset "M48/M61 parity anchors" begin
        for R in (10, 30, 50)
            acoustic = _ve_plan(; R=R, scaling=:acoustic, strict=false)
            @test acoustic.units.u_LU ≈ 0.005 atol=1e-12 rtol=0
            @test acoustic.units.nu_total_LU ≈ 0.005 * R atol=1e-12 rtol=0
            @test acoustic.units.tau_hydro ≈ 0.5 + 3 * 0.005 * R atol=1e-12 rtol=0
            @test acoustic.units.lambda_LU ≈ R / 0.005 atol=1e-12 rtol=0

            diffusive = _ve_plan(; R=R, scaling=:diffusive)
            @test diffusive.units.tau_hydro ≈ 0.95 atol=1e-12 rtol=0
            @test diffusive.units.nu_total_LU ≈ 0.15 atol=1e-12 rtol=0
            @test diffusive.units.u_LU ≈ 0.15 / R atol=1e-12 rtol=0
            @test diffusive.units.lambda_LU ≈ R^2 / 0.15 atol=1e-12 rtol=0
        end
    end

    @testset "round trip identity" begin
        for R in (10, 30, 50), scaling in (:acoustic, :diffusive)
            plan = _ve_plan(; R=R, scaling=scaling, strict=false)
            rt = U.audit(U.driver_kwargs(plan); physics=:viscoelastic,
                         geometry=_geom(), bc=_bc(), strict=false)
            @test rt.physics_spec.Re ≈ plan.physics_spec.Re atol=1e-12 rtol=0
            @test rt.physics_spec.Wi ≈ plan.physics_spec.Wi atol=1e-12 rtol=0
            @test rt.physics_spec.beta ≈ plan.physics_spec.beta atol=1e-12 rtol=0
            @test rt.physics_spec.bsd_fraction ≈ plan.physics_spec.bsd_fraction atol=1e-12 rtol=0
            @test rt.physics_spec.model == plan.physics_spec.model
            _assert_units_close(rt.units, plan.units)
            @test rt.bc == plan.bc
        end
    end

    @testset "strict and lenient validation" begin
        @test_throws U.PlanValidationError _ve_plan(; tau_target=1.6)
        lenient = _ve_plan(; tau_target=1.6, strict=false)
        @test :tau_above_trt_window in U.issue_codes(lenient.warnings)
    end

    @testset "auto scaling" begin
        sweep = U.compile(; physics=:viscoelastic, Re=1.0, Wi=1.0, beta=0.59,
                          R_LU=30, sweep_R=(10, 30, 50), scaling=:auto,
                          geometry=_geom(), bc=_bc())
        single = U.compile(; physics=:viscoelastic, Re=1.0, Wi=1.0, beta=0.59,
                           R_LU=30, scaling=:auto, geometry=_geom(), bc=_bc())
        @test sweep.units.scaling == :diffusive
        @test single.units.scaling == :acoustic
    end

    @testset "Float32 floor" begin
        @test_throws U.PlanValidationError _ve_plan(; tau_target=0.55, T=Float32)
    end

    @testset "BSD-aware tau shift" begin
        bsd1 = U.compile(; physics=:viscoelastic, Re=1.0, Wi=1.0, beta=0.59,
                         bsd_fraction=1.0, R_LU=30, scaling=:acoustic,
                         geometry=_geom(), bc=_bc())
        bsd05 = U.compile(; physics=:viscoelastic, Re=1.0, Wi=1.0, beta=0.59,
                          bsd_fraction=0.5, R_LU=30, scaling=:acoustic,
                          geometry=_geom(), bc=_bc())
        @test bsd1.units.tau_hydro ≈ 0.95 atol=1e-12 rtol=0
        @test bsd05.units.tau_hydro ≈ 0.5 + 3 * (0.59 + 0.5 * 0.41) * 0.15 atol=1e-12 rtol=0
        @test bsd05.units.tau_hydro < bsd1.units.tau_hydro
    end

    @testset "max_steps polymer floor" begin
        plan = _ve_plan(; R=30, scaling=:diffusive)
        @test plan.units.max_steps >= ceil(Int, 5 * plan.units.lambda_LU)
    end

    @testset "thermal stub seam" begin
        before = _ve_plan(; R=30, scaling=:diffusive)
        @test U.PHYSICS_REGISTRY[:thermal_boussinesq] === U.ThermalBoussinesqSpec
        spec = U.ThermalBoussinesqSpec{Float64}(1.0, 1.0, 1.0)
        @test_throws U.NotImplementedError U._compile_with_spec(spec)
        after = _ve_plan(; R=30, scaling=:diffusive)
        _assert_units_close(after.units, before.units)
        @test U.issue_codes(after.warnings) == U.issue_codes(before.warnings)
    end
end
