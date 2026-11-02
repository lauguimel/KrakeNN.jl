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
        # Reference-viscosity Ra/Pr and the provisional diffusive shear scale.
        # This checks only the qualitative increase of Nu with shear thinning.
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
        # TODO: Turan, Sachdeva, Chakraborty, Poole, JNNFM 166 (2011), 1049-1063,
        # DOI 10.1016/j.jnnfm.2011.06.003. Tabulated Nu and the paper's power-law
        # Ra/Pr definitions are unavailable; quantitative validation awaits both.
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
