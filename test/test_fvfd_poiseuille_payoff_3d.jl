using Printf
using Test
using Kraken
using KernelAbstractions

const FT = Float64

function signed_shear(profile::AbstractVector{<:Real})
    Ny = length(profile)
    Ny >= 3 || error("need at least three y rows to compute a centered shear profile")
    gamma = Vector{FT}(undef, Ny)
    gamma[1] = FT(profile[2] - profile[1])
    for j in 2:(Ny - 1)
        gamma[j] = FT((profile[j + 1] - profile[j - 1]) / 2)
    end
    gamma[Ny] = FT(profile[Ny] - profile[Ny - 1])
    return gamma
end

function analytic_velocity_profile(Ny::Int, Fx::Real, nu_total::Real)
    [FT(Fx / (2 * nu_total) * (j - 0.5) * (Ny + 0.5 - j)) for j in 1:Ny]
end

function analytic_shear_profile(Ny::Int, Fx::Real, nu_total::Real)
    [FT(Fx / (2 * nu_total) * (Ny + 1 - 2j)) for j in 1:Ny]
end

function y_profile(res, prof_name::Symbol, field_name::Symbol)
    if prof_name in propertynames(res)
        return FT.(getproperty(res, prof_name))
    end
    field = getproperty(res, field_name)
    Nx, Ny, Nz = size(field)
    return [sum(@view field[:, j, :]) / (Nx * Nz) for j in 1:Ny]
end

function ref_profiles(gamma::AbstractVector{<:Real}, lambda::Real)
    Wi = FT(lambda) .* FT.(gamma)
    Cxy = Wi
    Cxx = one(FT) .+ 2 .* Wi.^2
    N1 = Cxx .- one(FT)
    return (Cxy=Cxy, Cxx=Cxx, N1=N1)
end

function err(value::Real, reference::Real; abs_floor::Real=1e-3)
    return abs(reference) <= abs_floor ? abs(value - reference) :
           abs(value - reference) / abs(reference)
end

function collect_report(label::AbstractString, res, lambda::Real, Fx::Real, nu_total::Real)
    profile = FT.(res.profile)
    Ny = length(profile)
    Cxy = y_profile(res, :Cxy_prof, :C_xy)
    Cxx = y_profile(res, :Cxx_prof, :C_xx)
    Cyy = y_profile(res, :Cyy_prof, :C_yy)
    gamma_meas = signed_shear(profile)
    meas_ref = ref_profiles(gamma_meas, lambda)
    analytic_ref = ref_profiles(analytic_shear_profile(Ny, Fx, nu_total), lambda)
    u_ref = analytic_velocity_profile(Ny, Fx, nu_total)

    return (
        label=label,
        profile=profile,
        Cxy=Cxy,
        Cxx=Cxx,
        N1=Cxx .- Cyy,
        gamma_meas=gamma_meas,
        meas_ref=meas_ref,
        analytic_ref=analytic_ref,
        Cxy_err=[err(Cxy[j], meas_ref.Cxy[j]) for j in 1:Ny],
        Cxx_err=[err(Cxx[j], meas_ref.Cxx[j]) for j in 1:Ny],
        Cxy_err_an=[err(Cxy[j], analytic_ref.Cxy[j]) for j in 1:Ny],
        Cxx_err_an=[err(Cxx[j], analytic_ref.Cxx[j]) for j in 1:Ny],
        u_ratio=maximum(profile) / maximum(u_ref),
    )
end

function print_station_table(reports, stations)
    println("Station table: measured local-shear reference; *_an_err uses analytic parabola shear.")
    @printf("%-10s %4s %-4s %13s %13s %11s %11s %13s %13s %11s %11s %13s %13s\n",
            "station", "j", "path", "Cxy", "Cxy_ref", "Cxy_err", "Cxy_an_err",
            "Cxx", "Cxx_ref", "Cxx_err", "Cxx_an_err", "N1", "N1_ref")
    for (station, j) in stations
        for report in reports
            @printf("%-10s %4d %-4s % .6e % .6e %11.3e %11.3e % .6e % .6e %11.3e %11.3e % .6e % .6e\n",
                    station, j, report.label,
                    report.Cxy[j], report.meas_ref.Cxy[j], report.Cxy_err[j], report.Cxy_err_an[j],
                    report.Cxx[j], report.meas_ref.Cxx[j], report.Cxx_err[j], report.Cxx_err_an[j],
                    report.N1[j], report.meas_ref.N1[j])
        end
    end
end

function run_payoff_case(; Ny::Int=32, max_steps::Int=40_000)
    backend = CPU()
    Nx, Nz = 6, 6
    nu_total = 0.1
    beta = 0.5
    nu_s = beta * nu_total
    nu_p = (1 - beta) * nu_total
    Fx = 1.5e-5

    gamma_wall = Fx / (2 * nu_total) * (Ny - 1)
    Wi_wall_target = 0.5
    lambda = Wi_wall_target / gamma_wall

    println("M-VE3D-FVFD-M4 setup: Nx=$Nx Ny=$Ny Nz=$Nz Fx=$Fx nu_total=$nu_total beta=$beta lambda=$lambda Wi_wall=$Wi_wall_target steps=$max_steps")

    fv = Kraken.run_viscoelastic_fvfd_poiseuille_3d(;
        Nx, Ny, Nz, Fx, ν_s=nu_s, ν_p=nu_p, lambda,
        max_steps, backend, FT,
        advection_scheme=:muscl_superbee,
    )
    lbm = Kraken.run_conformation_poiseuille_libb_3d(;
        Nx, Ny, Nz, Fx, ν_s=nu_s, ν_p=nu_p, lambda,
        tau_plus=1.0, max_steps, backend, FT,
    )

    fv_report = collect_report("FV", fv, lambda, Fx, nu_total)
    lbm_report = collect_report("LBM", lbm, lambda, Fx, nu_total)

    near_j = 4
    mid_j = Ny ÷ 4
    centre_j = Ny ÷ 2 + 1
    stations = [("near-wall", near_j), ("mid", mid_j), ("centre", centre_j)]
    print_station_table((fv_report, lbm_report), stations)

    fv_near = fv_report.Cxy_err[near_j]
    lbm_near = lbm_report.Cxy_err[near_j]
    cure_ratio = fv_near / lbm_near
    @printf("Near-wall Cxy error: FV=%.6e LBM=%.6e FV/LBM=%.6e\n",
            fv_near, lbm_near, cure_ratio)
    @printf("Velocity peak u-ratio vs parabola: FV=%.8f LBM=%.8f\n",
            fv_report.u_ratio, lbm_report.u_ratio)
    @printf("FV substeps: last=%d max=%d completed=%d\n",
            fv.last_n_sub, fv.max_substeps_observed, fv.completed_steps)

    @testset "P1 FV bulk conformation accuracy" begin
        for j in (mid_j, centre_j)
            @test fv_report.Cxy_err[j] < 0.01
            @test fv_report.Cxx_err[j] < 0.01
        end
    end

    @testset "P2 FV cures near-wall Cxy diffusion" begin
        @test fv_near < lbm_near / 3
        @test fv_near < 0.05
    end

    return (fv=fv_report, lbm=lbm_report, cure_ratio=cure_ratio,
            setup=(Nx=Nx, Ny=Ny, Nz=Nz, Fx=Fx, nu_total=nu_total,
                   beta=beta, lambda=lambda, Wi_wall=Wi_wall_target,
                   max_steps=max_steps))
end

@testset "FVFD payoff for curved Poiseuille conformation in 3D" begin
    result = run_payoff_case(Ny=32, max_steps=40_000)
    @test isfinite(result.cure_ratio)
end
