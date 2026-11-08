using Test
using KrakeNN

const U = KrakeNN.Units

function _assert_same_plan_units(a, b)
    @test a.units.tau_hydro ≈ b.units.tau_hydro atol=1e-12 rtol=0
    @test a.units.nu_total_LU ≈ b.units.nu_total_LU atol=1e-12 rtol=0
    @test a.units.u_LU ≈ b.units.u_LU atol=1e-12 rtol=0
    @test a.units.R_LU == b.units.R_LU
    @test a.units.scaling == b.units.scaling
    @test a.physics_spec.Re ≈ b.physics_spec.Re atol=1e-12 rtol=0
    @test a.physics_spec.Wi ≈ b.physics_spec.Wi atol=1e-12 rtol=0
    @test a.physics_spec.beta ≈ b.physics_spec.beta atol=1e-12 rtol=0
end

@testset "Units .krk binding" begin
    path = joinpath(@__DIR__, "..", "benchmarks", "krk", "units",
                    "m61_diffusive_reproduction.krk")
    plans = U.load_units_krk(path)

    @testset "mega-block parse" begin
        @test haskey(plans, :mega_m61)
        mega = plans[:mega_m61]
        @test mega.units.tau_hydro ≈ 0.95 atol=1e-12 rtol=0
        @test mega.units.nu_total_LU ≈ 0.15 atol=1e-12 rtol=0
    end

    @testset "cross-reference equivalence" begin
        @test haskey(plans, :cross_m61)
        _assert_same_plan_units(plans[:mega_m61], plans[:cross_m61])
    end

    @testset "planner-owned override warning" begin
        @test :planner_override in U.issue_codes(plans[:cross_m61].warnings)
        @test plans[:cross_m61].units.u_LU ≈ 0.005 atol=1e-12 rtol=0
    end
end
