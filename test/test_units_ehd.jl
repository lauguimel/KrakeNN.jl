using Test
using KrakeNN

const UEHD = KrakeNN.Units

function _ehd_reference_params()
    return (
        H=320.0,
        cs=0.5773502691896258,
        K=1.8475208614068028,
        nu=0.9723794007404226,
        tau=3.4171382022212677,
        omega=0.2926425391135666,
        eps=341.3333333333334,
        q_inj=0.03333333333333335,
        D=0.0001847520861406803,
        tau_U=1.4,
        nu_U=0.3,
        omega_U=0.7142857142857143,
        tau_q=0.5005542562584221,
        omega_q=1.997785429845048,
        dt_star=1.804219591217581e-5,
    )
end

@testset "Units EHD electroconvection" begin
    T_ehd = 190.0
    C = 10.0
    M = 10.0
    alpha = 1e-4
    Ma_E = 0.01
    spec = UEHD.EHDSpec{Float64}(T_ehd, C, M, alpha, Ma_E)

    @test UEHD.PHYSICS_REGISTRY[:ehd_ec] === UEHD.EHDSpec
    @test UEHD._PHYSICS_KW[:ehd_ec] == Set{Symbol}((:T, :T_ehd, :C, :M, :alpha, :Ma_E))

    p = UEHD.ehd_ec_lattice_params(spec, 321, 1.0, 0.3; FT=Float64)
    @test p.T_check ≈ T_ehd atol=1e-12 rtol=0
    @test p.C_check ≈ C atol=1e-12 rtol=0
    @test p.M_check ≈ M atol=1e-12 rtol=0
    @test p.alpha_check ≈ alpha atol=1e-12 rtol=0

    ref = _ehd_reference_params()
    for name in propertynames(ref)
        @test getproperty(p, name) ≈ getproperty(ref, name) atol=1e-12 rtol=0
    end

    wrapped = KrakeNN._ehd_ec_lattice_params(321, C, M, T_ehd, Ma_E, alpha, 1.0, 0.3; FT=Float64)
    for name in propertynames(p)
        @test isequal(getproperty(wrapped, name), getproperty(p, name))
    end
end
