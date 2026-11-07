using Test
using KrakeNN

@testset "Fused TRT 2D" begin

    @testset "trt_rates — magic Λ=3/16 gives canonical values" begin
        # Convention (post-2026-04-15 label fix, Ginzburg/Krüger):
        #   s_plus  rate on symmetric (even) mode, fixes viscosity.
        #   s_minus rate on antisymmetric (odd) mode, set by Λ.
        # At ν = 1/6 (τ = 1.0): s_plus = 1/(3·1/6 + 0.5) = 1.0.
        # Λ = 3/16 ⇒ s_minus = 1/((3/16)/(0.5) + 0.5) = 8/7.
        sp, sm = trt_rates(1/6; Λ=3/16)
        @test sp ≈ 1.0 atol=1e-12
        @test sm ≈ 8/7 atol=1e-12
    end

    @testset "trt_rates — magic Λ reproduces BGK when Λ = (1/s−1/2)²" begin
        ν = 0.01
        s = 1.0 / (3ν + 0.5)
        Λ_bgk = (1/s - 0.5)^2
        sp, sm = trt_rates(ν; Λ=Λ_bgk)
        @test sp ≈ sm atol=1e-12
        @test sm ≈ s atol=1e-12
    end

    @testset "TRT with s_plus = s_minus reproduces BGK exactly" begin
        # Pick Λ such that s_plus = s_minus ⇒ TRT becomes BGK.
        N = 16
        ν = 0.02
        s = 1.0 / (3ν + 0.5)
        Λ_bgk = (1/s - 0.5)^2   # makes s_plus = s_minus = s

        f_in  = zeros(Float64, N, N, 9)
        ρ = ones(N, N); ux = zeros(N, N); uy = zeros(N, N)
        is_solid = zeros(Bool, N, N)
        for j in 1:N, i in 1:N
            u_init = 0.01 * sin(2π * (i - 1) / N)
            for q in 1:9
                f_in[i, j, q] = KrakeNN.equilibrium(D2Q9(), 1.0, u_init, 0.0, q)
            end
        end

        f_trt = copy(f_in); f_trt_out = similar(f_trt)
        f_bgk = copy(f_in); f_bgk_out = similar(f_bgk)
        ρ_t = copy(ρ); ux_t = copy(ux); uy_t = copy(uy)
        ρ_b = copy(ρ); ux_b = copy(ux); uy_b = copy(uy)

        for _ in 1:50
            fused_trt_step!(f_trt_out, f_trt, ρ_t, ux_t, uy_t, is_solid, N, N, ν; Λ=Λ_bgk)
            fused_bgk_step!(f_bgk_out, f_bgk, ρ_b, ux_b, uy_b, is_solid, N, N, s)
            f_trt, f_trt_out = f_trt_out, f_trt
            f_bgk, f_bgk_out = f_bgk_out, f_bgk
        end

        @test maximum(abs.(f_trt .- f_bgk)) < 1e-12
        @test maximum(abs.(ux_t .- ux_b)) < 1e-12
        @test maximum(abs.(ρ_t .- ρ_b)) < 1e-12
    end

    @testset "TRT equilibrium is stationary" begin
        N = 16
        ν = 0.02
        f_in = zeros(Float64, N, N, 9)
        for j in 1:N, i in 1:N, q in 1:9
            f_in[i, j, q] = KrakeNN.equilibrium(D2Q9(), 1.0, 0.0, 0.0, q)
        end
        f_out = similar(f_in)
        ρ = ones(N, N); ux = zeros(N, N); uy = zeros(N, N)
        is_solid = zeros(Bool, N, N)
        f0 = copy(f_in)
        for _ in 1:100
            fused_trt_step!(f_out, f_in, ρ, ux, uy, is_solid, N, N, ν)
            f_in, f_out = f_out, f_in
        end
        @test maximum(abs.(f_in .- f0)) < 1e-12
        @test maximum(abs.(ux)) < 1e-12
        @test maximum(abs.(ρ .- 1.0)) < 1e-12
    end

    @testset "TRT is stable and dissipates kinetic energy" begin
        # Walled domain (halfway BB at edges of fused_trt), random
        # initial disturbance. We only check: no NaN, KE is bounded,
        # and KE decreases. Quantitative accuracy against analytical
        # is deferred to the LI-BB + wall-bounded tests.
        N = 48
        ν = 0.02
        u0 = 0.01
        f_in = zeros(Float64, N, N, 9)
        ρ = ones(N, N); ux = zeros(N, N); uy = zeros(N, N)
        is_solid = zeros(Bool, N, N)
        k = 2π / N
        for j in 1:N, i in 1:N
            u_ij = -u0 * cos(k*(i-1)) * sin(k*(j-1))
            v_ij =  u0 * sin(k*(i-1)) * cos(k*(j-1))
            ux[i, j] = u_ij; uy[i, j] = v_ij
            for q in 1:9
                f_in[i, j, q] = KrakeNN.equilibrium(D2Q9(), 1.0, u_ij, v_ij, q)
            end
        end
        f_out = similar(f_in)
        KE0 = sum(ux .^ 2 .+ uy .^ 2)
        for _ in 1:500
            fused_trt_step!(f_out, f_in, ρ, ux, uy, is_solid, N, N, ν)
            f_in, f_out = f_out, f_in
        end
        KE_final = sum(ux .^ 2 .+ uy .^ 2)
        @info "TRT KE decay: initial $(round(KE0, digits=4)) → final $(round(KE_final, digits=4))"
        @test all(isfinite.(ux))
        @test all(isfinite.(uy))
        @test KE_final < KE0
        @test KE_final > 0   # not complete decay
    end

end
