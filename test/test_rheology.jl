using Test
using KrakeNN

@testset "Rheology models" begin

    @testset "Effective viscosity functions" begin
        # Newtonian: constant
        m = Newtonian(0.1)
        @test effective_viscosity(m, 0.0) ≈ 0.1
        @test effective_viscosity(m, 100.0) ≈ 0.1

        # Power-law: η = K · γ̇^(n-1)
        m = PowerLaw(0.1, 0.5)  # shear-thinning
        @test effective_viscosity(m, 1.0) ≈ 0.1  # K · 1^(n-1) = K
        @test effective_viscosity(m, 4.0) ≈ 0.1 * 4.0^(-0.5)  # K · 4^(0.5-1)

        m = PowerLaw(0.1, 1.5)  # shear-thickening
        @test effective_viscosity(m, 1.0) ≈ 0.1
        @test effective_viscosity(m, 4.0) ≈ 0.1 * 4.0^(0.5)

        # Carreau-Yasuda (a=2 → standard Carreau)
        m = CarreauYasuda(1.0, 0.01, 1.0, 2.0, 0.5)
        @test effective_viscosity(m, 0.0) ≈ 1.0   # zero-shear = eta_0
        @test effective_viscosity(m, 1e10) < 0.02  # high-shear → eta_inf

        # Cross
        m = Cross(1.0, 0.01, 1.0, 1.0)
        @test effective_viscosity(m, 0.0) ≈ 1.0   # zero-shear
        @test effective_viscosity(m, 1e10) < 0.02  # high-shear

        # Bingham: η = τ_y·(1-exp(-m·γ̇))/γ̇ + μ_p
        m = Bingham(0.1, 0.05; m_reg=1000.0)
        # At high shear rate: τ_y/γ̇ → 0, so η → μ_p
        @test effective_viscosity(m, 1000.0) ≈ 0.05 atol=0.01
        # At low shear rate: effective viscosity should be large
        @test effective_viscosity(m, 1e-5) > 10.0

        # Herschel-Bulkley: η = τ_y·(...)/γ̇ + K·γ̇^(n-1)
        m = HerschelBulkley(0.1, 0.1, 0.5; m_reg=1000.0)
        @test effective_viscosity(m, 1.0) ≈ 0.1 + 0.1  atol=0.01  # τ_y≈τ_y, K·1^(n-1)=K
    end

    @testset "Thermal coupling" begin
        # Arrhenius: a_T = exp(E_a · (1/T - 1/T_ref))
        tc = ArrheniusCoupling(1.0, 5.0)
        @test thermal_shift_factor(tc, 1.0) ≈ 1.0  # at T_ref
        @test thermal_shift_factor(tc, 2.0) < 1.0   # hotter → lower viscosity

        # WLF
        tc = WLFCoupling(1.0, 8.86, 101.6)
        @test thermal_shift_factor(tc, 1.0) ≈ 1.0  # at T_ref

        # Isothermal → always 1
        @test thermal_shift_factor(IsothermalCoupling(), 42.0) ≈ 1.0

        # Power-law with thermal coupling
        tc = ArrheniusCoupling(1.0, 5.0)
        m = PowerLaw(0.1, 0.5; thermal=tc)
        ν_cold = effective_viscosity_thermal(m, 1.0, 0.5)  # colder
        ν_ref  = effective_viscosity_thermal(m, 1.0, 1.0)  # at T_ref
        ν_hot  = effective_viscosity_thermal(m, 1.0, 2.0)  # hotter
        @test ν_cold > ν_ref > ν_hot
    end

    @testset "Power-law Poiseuille flow" begin
        # Analytical solution for power-law Poiseuille with body force:
        #   u(y) = n/(n+1) · (Fx/K)^(1/n) · [(H/2)^((n+1)/n) - |y - H/2|^((n+1)/n)]
        # where H is the channel width (with half-way bounce-back: H = Ny)

        Nx, Ny = 4, 32
        K = 0.1
        n_pl = 0.7  # mild shear-thinning
        Fx_val = 1e-4
        max_steps = 50000

        # Analytical solution (half-way BB: effective channel = Ny, walls at 0.5 and Ny+0.5)
        H = Float64(Ny)
        u_analytical = zeros(Ny)
        for j in 1:Ny
            y = j - 0.5  # node position (half-way bounce-back)
            dist = abs(y - H/2)
            u_analytical[j] = n_pl/(n_pl+1) * (Fx_val/K)^(1/n_pl) *
                              ((H/2)^((n_pl+1)/n_pl) - dist^((n_pl+1)/n_pl))
        end
        u_max_ana = maximum(u_analytical)
        @info "Expected power-law u_max = $(round(u_max_ana, digits=6))"

        # Initialize LBM arrays
        f_in  = zeros(Float64, Nx, Ny, 9)
        f_out = zeros(Float64, Nx, Ny, 9)
        is_solid = falses(Nx, Ny)

        # Estimate initial viscosity from expected center shear rate
        γ_est = u_max_ana / (H/4)
        ν_est = K * max(γ_est, 1e-8)^(n_pl - 1)
        tau_init = 3 * ν_est + 0.5
        tau_field = fill(tau_init, Nx, Ny)

        ux = zeros(Float64, Nx, Ny)
        uy = zeros(Float64, Nx, Ny)
        ρ  = ones(Float64, Nx, Ny)
        Fx_arr = fill(Float64(Fx_val), Nx, Ny)
        Fy_arr = zeros(Float64, Nx, Ny)

        # Initialize equilibrium at rest
        for j in 1:Ny, i in 1:Nx, q in 1:9
            f_in[i,j,q] = equilibrium(D2Q9(), 1.0, 0.0, 0.0, q)
        end
        copy!(f_out, f_in)

        rheology = PowerLaw(K, n_pl; nu_min=1e-5, nu_max=5.0)

        for step in 1:max_steps
            stream_periodic_x_wall_y_2d!(f_out, f_in, Nx, Ny)
            collide_rheology_guo_2d!(f_out, is_solid, rheology, tau_field, Fx_arr, Fy_arr)
            f_in, f_out = f_out, f_in
        end

        # Compute macroscopic (scalar Fx for this kernel)
        compute_macroscopic_forced_2d!(ρ, ux, uy, f_in, Fx_val, 0.0)

        # Compare at center x-slice
        u_num = ux[2, :]
        u_max_num = maximum(u_num)

        # Relative error on interior points (skip wall-adjacent nodes)
        errors = abs.(u_num[3:end-2] .- u_analytical[3:end-2]) ./ u_max_ana
        max_err = maximum(errors)

        @info "Power-law Poiseuille (n=$n_pl): L∞ rel error = $(round(max_err, digits=4)), u_max_num=$(round(u_max_num, digits=6)), u_max_ana=$(round(u_max_ana, digits=6))"
        @test max_err < 0.10  # 10% tolerance (non-Newtonian + half-way BB)
    end

    @testset "Thermal shift factors: WLF base 10 and Arrhenius set-up check" begin
        # WLF with the published (base-10) constants: a_T = 10^(-C1 dT / (C2 + dT)).
        tc = WLFCoupling(300.0, 8.86, 101.6)
        expected = 10.0^(-8.86 * 20.0 / (101.6 + 20.0))
        @test thermal_shift_factor(tc, 320.0) ≈ expected rtol=1e-12
        @test thermal_shift_factor(tc, 320.0) ≈ 0.0349 atol=5e-5   # was exp(): 0.233
        @test thermal_shift_factor(tc, 300.0) == 1.0
        @test thermal_shift_factor(tc, 280.0) > 1.0                 # colder -> stiffer

        # Arrhenius: absolute temperatures only, refused at set-up time.
        arr = ArrheniusCoupling(300.0, 10.0)
        @test thermal_shift_factor(arr, 300.0) == 1.0
        check = KrakeNN._check_arrhenius_temperatures
        @test check(arr, 290.0, 310.0) === nothing
        @test_throws ArgumentError check(arr, 0.0)
        @test_throws ArgumentError check(arr, 300.0, -1.0)
        @test_throws ArgumentError check(ArrheniusCoupling(0.0, 10.0))
        @test check(WLFCoupling(1.0, 8.86, 101.6), 0.0) === nothing   # WLF has no such constraint
        @test check(IsothermalCoupling(), -5.0) === nothing

        # Driver entry point: the default walls (T_hot=1, T_cold=0) give T = 0.
        model = PowerLaw(0.05, 0.8; thermal=ArrheniusCoupling(1.5, 0.2))
        @test_throws ArgumentError run_natural_convection_gnf_2d(; N=4, rheology=model, max_steps=1)
        @test_throws ArgumentError run_natural_convection_gnf_2d(; N=4, rheology=model,
                                       max_steps=1, T_hot=2.0, T_cold=-1.0)
        ok = run_natural_convection_gnf_2d(; N=4, rheology=model, max_steps=1,
                                           T_hot=2.0, T_cold=1.0)
        @test all(isfinite, ok.Temp)

        # .krk loader entry point.
        bad = KrakeNN.RheologySetup(:default, :power_law, Dict(:K => 0.1, :n => 0.5, :E_a => 5.0, :T_ref => 0.0))
        @test_throws ArgumentError build_rheology_model(bad)
        good = KrakeNN.RheologySetup(:default, :power_law, Dict(:K => 0.1, :n => 0.5, :E_a => 5.0, :T_ref => 300.0))
        @test build_rheology_model(good).thermal isa ArrheniusCoupling

        # Isothermal .krk route: thermal keys are ignored, one warning names them.
        @test_logs (:warn, r"E_a, T_ref.*ignored") KrakeNN._warn_ignored_thermal_keys(good)
        wlf = KrakeNN.RheologySetup(:default, :carreau, Dict(:C1 => 8.86, :C2 => 101.6))
        @test_logs (:warn, r"C1, C2.*ignored") KrakeNN._warn_ignored_thermal_keys(wlf)
        plain = KrakeNN.RheologySetup(:default, :power_law, Dict(:K => 0.1, :n => 0.5))
        @test_logs KrakeNN._warn_ignored_thermal_keys(plain)
    end
end
