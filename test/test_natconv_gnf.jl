using Test, Kraken

@testset "GNF natural convection CPU Float64" begin
    started = time()
    nu0, Pr = 0.05, 0.71
    @testset "Reference scales and parameter types" begin
        scales = gnf_reference_scales(; Ra=1e3, Pr, nu0, H=64.0, T_hot=1.0, T_cold=0.0)
        @test scales.alpha == nu0 / Pr
        @test scales.beta_g == 1e3 * nu0 * scales.alpha / 64.0^3
        model = cavity_powerlaw(0.8; nu0, alpha=scales.alpha, H=64)
        @test effective_viscosity(model, scales.alpha/64^2) ≈ nu0 rtol=10eps(Float64)
        mixed = PowerLaw(0.05f0, 1f0; nu_min=0.001f0, nu_max=1f0,
                         thermal=ArrheniusCoupling(1.0, 0.2))
        @test_throws ArgumentError run_natural_convection_gnf_2d(; N=4,
            rheology=mixed, max_steps=1, FT=Float32)
    end

    @testset "Newtonian limit and De Vahl Davis" begin
        reference = run_natural_convection_2d(; N=64, Ra=1e3, Pr, max_steps=30000)
        result = run_natural_convection_gnf_2d(; N=64, Ra=1e3, Pr, nu0,
            rheology=PowerLaw(nu0, 1.0), max_steps=30000)
        gap = abs(result.Nu - reference.Nu) / abs(reference.Nu)
        @test gap < 0.01 # Required Newtonian-limit agreement: identical physical problem.
        # De Vahl Davis (1983), DOI 10.1002/fld.1650030305, Nu=1.118;
        # same frozen 10% N64 gate as test_thermal.jl (docs/refs.bib: devahldavis1983).
        @test abs(result.Nu - 1.118) / 1.118 < 0.10
        @test all(isfinite, result.Temp)
        @test all(isfinite, result.ux) && all(isfinite, result.uy)
        @test maximum(abs.(result.ν_field .- nu0)) < 10eps(Float64)
        @test result.ν_field === result.nu_field
        @test result.clamped_fraction == 0.0
        @info "GNF n=1" Nu=result.Nu Newtonian_Nu=reference.Nu relative_gap=gap steps=result.steps
    end

    @testset "Carreau-Yasuda cavity API" begin
        model = CarreauYasuda(0.08, 0.01, 10.0, 2.0, 0.6)
        result = run_natural_convection_gnf_2d(; N=8, Ra=1e3, Pr,
            rheology=model, max_steps=5)
        @test all(isfinite, result.Temp) && all(isfinite, result.ν_field)
        @test all(model.eta_inf .<= result.ν_field .<= model.eta_0)
        @test size(result.ν_field) == (8, 8) && result.rheology == model
    end

    @testset "QUALITATIVE literature trend (no quantitative power-law claim)" begin
        # alpha/H^2 is the characteristic shear rate of Turan et al. (2011).
        # Pr = 0.71 is outside their power-law data (Pr >= 10), so this check
        # remains qualitative: Nu increases with shear thinning.
        results = map((0.8, 1.0, 1.2)) do n
            model = cavity_powerlaw(n; nu0, alpha=nu0/Pr, H=32,
                                     nu_min=1e-4, nu_max=1.0)
            result = run_natural_convection_gnf_2d(; N=32, Ra=1e4, Pr, nu0,
                rheology=model, max_steps=30000, check_every=1000,
                convergence_rtol=1e-5, min_steps=5000)
            # Nu stationarity: three successive 1000-step changes below 1e-5.
            # This is a numerical stopping gate, not a literature error bar.
            @info "GNF qualitative case" n Nu=result.Nu steps=result.steps drift=result.Nu_drift clamped_fraction=result.clamped_fraction
            @test result.converged
            @test all(isfinite, result.Temp) && all(isfinite, result.ux) && all(isfinite, result.uy)
            @test all(isfinite, result.ν_field) && all(isfinite, result.tau_field)
            @test all(model.nu_min .<= result.ν_field .<= model.nu_max)
            @test result.clamped_fraction < 0.01 # Fewer than 1% cutoff nodes: constitutive regime.
            result
        end
        @test results[1].Nu > results[2].Nu > results[3].Nu
        @info "QUALITATIVE Nu(0.8/1/1.2)" Nu=map(r -> r.Nu, results)
        # Quantitative comparison: see the Turan et al. (2011) testset below.
    end

    @testset "Turan et al. (2011) power-law point, CPU" begin
        Pr, nu0, N, Ra = 10.0, 0.3, 48, 1e3
        model = cavity_powerlaw(0.6; nu0, alpha=nu0/Pr, H=N,
                                 nu_min=1e-3, nu_max=2.0)
        result = run_natural_convection_gnf_2d(; N, Ra, Pr, nu0, rheology=model,
            max_steps=60000, check_every=1000, convergence_rtol=1e-5, min_steps=5000)
        @test result.converged
        @test result.clamped_fraction == 0.0
        @test all(isfinite, result.Temp)
        @test all(isfinite, result.ux)
        @test all(isfinite, result.uy)
        # Turan et al. (2011), Fig. 9a (p. 1059): Nu = 2.07 at Ra = 1e3,
        # Pr = 10, n = 0.6; doi:10.1016/j.jnnfm.2011.06.003.
        # Frozen gate from the reference's own uncertainty: figure reading
        # +/-0.07 (3.4%) plus mesh-M2 uncertainty <= 0.35% (Table 2), rounded
        # up to 4%. Measured 2026-11-02 on an Apple-silicon Mac, CPU Float64:
        # Nu = 2.124 (+2.6%).
        @test abs(result.Nu - 2.07) / 2.07 < 0.04
        @info "GNF Turan point" Nu=result.Nu steps=result.steps clamped_fraction=result.clamped_fraction
    end

    @testset "Parsed thermal rheology reaches run_simulation" begin
        setup = parse_kraken("""
        Simulation natural_convection_gnf D2Q9
        Domain L = 1 x 1 N = 8 x 8
        Physics nu = 0.04 Ra = 1000 Pr = 0.71
        Module thermal
        Rheology power_law { K = 0.04 n = 0.9 nu_min = 0.01 nu_max = 1.0 }
        Boundary west wall
        Boundary east wall
        Boundary south wall
        Boundary north wall
        Run 5 steps
        """)
        result = run_simulation(setup; T=Float64)
        @test result.rheology isa PowerLaw
        @test result.rheology.n == 0.9
        @test result.nu0 == 0.04 && result.Ra == 1000 && result.Pr == 0.71
        @test result.steps == 5 && size(result.ν_field) == (8, 8)
        @test all(isfinite, result.Temp)
    end
    @info "GNF test runtime" seconds=time()-started
end
