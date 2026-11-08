using Test
using KrakeNN

const UTHERM = KrakeNN.Units

_thermal_geom() = (type=:channel, blockage=0.0, L_up=15.0, L_down=15.0)

_thermal_bc(; north_wall=:temperature_dirichlet,
            south_wall=:temperature_dirichlet,
            wall_bc=:halfwayBB) =
    (inlet=:velocity_parabolic, outlet=:zou_he_pressure,
     north_wall=north_wall, south_wall=south_wall, wall_bc=wall_bc)

function _thermal_plan(; Re=1.0, Pr=0.71, Ra=1e5, R=128,
                       strict=false, kwargs...)
    return UTHERM.compile(; physics=:thermal_boussinesq, Re=Re, Pr=Pr,
                          Ra=Ra, R_LU=R, scaling=:diffusive,
                          tau_target=0.95, geometry=_thermal_geom(),
                          bc=_thermal_bc(), strict=strict, kwargs...)
end

function _ve_m4_plan(R)
    return UTHERM.compile(; physics=:viscoelastic, Re=1.0, Wi=1.0,
                          beta=0.59, bsd_fraction=1.0,
                          model=:oldroyd_b, R_LU=R, scaling=:diffusive,
                          tau_target=0.95, geometry=_thermal_geom(),
                          bc=(inlet=:velocity_parabolic,
                              outlet=:zou_he_pressure,
                              north_wall=:halfwayBB,
                              south_wall=:halfwayBB,
                              wall_bc=:halfwayBB))
end

function _m4_reference_units(R)
    refs = Dict(
        10 => UTHERM.LBMUnits{Float64}(
            0.95, 0.15, 0.015, 10, 0.025980762113533156,
            :diffusive, 20000, 0.0885, 0.0615, 666.6666666666667,
            NaN, NaN, NaN, NaN, NaN),
        30 => UTHERM.LBMUnits{Float64}(
            0.95, 0.15, 0.005, 30, 0.008660254037844387,
            :diffusive, 180000, 0.0885, 0.0615, 6000.0,
            NaN, NaN, NaN, NaN, NaN),
        50 => UTHERM.LBMUnits{Float64}(
            0.95, 0.15, 0.003, 50, 0.005196152422706632,
            :diffusive, 500000, 0.0885, 0.0615, 16666.666666666668,
            NaN, NaN, NaN, NaN, NaN),
    )
    return refs[R]
end

function _assert_units_isequal(actual, expected)
    for field in fieldnames(typeof(actual))
        @test isequal(getfield(actual, field), getfield(expected, field))
    end
end

@testset "Units Thermal-Boussinesq" begin
    @testset "forward thermal closure" begin
        plan = _thermal_plan(; Re=1e3, Pr=0.71, Ra=1e5, R=128)
        @test plan isa UTHERM.SimulationPlan{Float64}
        @test plan.physics_spec isa UTHERM.ThermalBoussinesqSpec{Float64}
        units = plan.units
        @test units.alpha_LU ≈ units.nu_total_LU / plan.physics_spec.Pr atol=1e-12 rtol=0
        reconstructed_Ra = units.beta_thermal_LU * units.R_LU^3 /
                           (units.nu_total_LU * units.alpha_LU)
        @test reconstructed_Ra ≈ plan.physics_spec.Ra rtol=1e-10
        @test isnan(units.nu_s_LU)
        @test isnan(units.nu_p_LU)
        @test isnan(units.lambda_LU)

        raw = (R_LU=units.R_LU, u_LU=units.u_LU,
               tau_hydro=units.tau_hydro, nu_total_LU=units.nu_total_LU,
               alpha_LU=units.alpha_LU,
               beta_thermal_LU=units.beta_thermal_LU,
               scaling=units.scaling, max_steps=units.max_steps)
        audited = UTHERM.audit(raw; physics=:thermal_boussinesq,
                               geometry=_thermal_geom(), bc=_thermal_bc(),
                               strict=false)
        @test audited.physics_spec.Re ≈ plan.physics_spec.Re atol=1e-12 rtol=0
        @test audited.physics_spec.Pr ≈ plan.physics_spec.Pr atol=1e-12 rtol=0
        @test audited.physics_spec.Ra ≈ plan.physics_spec.Ra rtol=1e-10
    end

    @testset "thermal registries" begin
        @test haskey(UTHERM.STABILITY_REGISTRY,
                     (UTHERM.HalfwayBB, UTHERM.ThermalBoussinesqSpec))
        unstable = _thermal_plan(; Re=1.0, Pr=100.0, Ra=1e5, R=128)
        @test :thermal_tau_below_floor in UTHERM.issue_codes(unstable.warnings)

        combo = (:velocity_parabolic, :zou_he_pressure,
                 :temperature_dirichlet, :temperature_dirichlet)
        @test UTHERM.BC_COMPATIBILITY[combo] == :ok
        @test !(:bc_combo_unknown in UTHERM.issue_codes(unstable.warnings))
        @test !(:bc_combo_incompatible in UTHERM.issue_codes(unstable.warnings))
    end

    @testset "VE zero-edit proof" begin
        thermal_probe = _thermal_plan(; Re=1e3, Pr=0.71, Ra=1e5, R=128)
        @test thermal_probe.units.alpha_LU > 0
        for R in (10, 30, 50)
            before = _ve_m4_plan(R)
            after = _ve_m4_plan(R)
            reference = _m4_reference_units(R)
            _assert_units_isequal(before.units, reference)
            _assert_units_isequal(after.units, before.units)
            @test UTHERM.issue_codes(before.warnings) == Symbol[]
            @test UTHERM.issue_codes(after.warnings) == Symbol[]
        end
    end
end
