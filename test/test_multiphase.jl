using Test
using KrakeNN
using Statistics: var, mean

@testset "Shan-Chen multiphase" begin

    @testset "SC force computation" begin
        # Verify that the SC force is computed correctly on a known density field
        Nx, Ny = 32, 32
        ρ = ones(Float64, Nx, Ny)
        # Create a density bump at center
        for j in 1:Ny, i in 1:Nx
            r2 = (i - Nx/2)^2 + (j - Ny/2)^2
            ρ[i, j] = 1.0 + 0.5 * exp(-r2 / 25.0)
        end

        ψ     = zeros(Float64, Nx, Ny)
        Fx_sc = zeros(Float64, Nx, Ny)
        Fy_sc = zeros(Float64, Nx, Ny)

        compute_psi_2d!(ψ, ρ, 1.0)
        compute_sc_force_2d!(Fx_sc, Fy_sc, ψ, -5.0, Nx, Ny)

        # ψ should be nonzero and monotonic with ρ
        @test maximum(ψ) > 0.5
        @test ψ[Nx÷2, Ny÷2] > ψ[1, 1]  # higher ψ at density bump

        # Force should point AWAY from density bump (G < 0 → repulsive from high ψ)
        # At center, force should be ~zero (symmetric)
        @test abs(Fx_sc[Nx÷2, Ny÷2]) < 1e-10
        # Away from center in +x direction, force should be positive (pushing outward)
        @test Fx_sc[Nx÷2 + 3, Ny÷2] > 0

        @info "SC force: max|F| = $(round(maximum(abs.(Fx_sc)), sigdigits=3))"
    end

    @testset "SC spinodal stability" begin
        # Just verify the full SC simulation runs without NaN
        result = run_spinodal_2d(; N=32, ν=1/6, G=-5.0, ρ0=1.0, max_steps=500)
        @test !any(isnan, result.ρ)
        @info "SC spinodal: stable after 500 steps, ρ range = $(round.(extrema(result.ρ), digits=4))"
    end

    @testset "Flat interface" begin
        # Moderate density ratio for stability
        N = 32; ν = 1/6; G = -5.0; ρ0 = 1.0
        ρ_high = 1.2; ρ_low = 0.8

        config = LBMConfig(D2Q9(); Nx=N, Ny=N, ν=ν, u_lid=0.0, max_steps=0)
        Nx, Ny = N, N
        ω = Float64(omega(config))

        f_in  = zeros(Float64, Nx, Ny, 9)
        f_out = zeros(Float64, Nx, Ny, 9)
        ρ     = zeros(Float64, Nx, Ny)
        ux    = zeros(Float64, Nx, Ny)
        uy    = zeros(Float64, Nx, Ny)
        ψ     = zeros(Float64, Nx, Ny)
        Fx_sc = zeros(Float64, Nx, Ny)
        Fy_sc = zeros(Float64, Nx, Ny)
        is_solid = zeros(Bool, Nx, Ny)

        w = KrakeNN.weights(D2Q9())
        for j in 1:Ny, i in 1:Nx
            ρ_init = j <= Ny ÷ 2 ? ρ_high : ρ_low
            for q in 1:9
                f_in[i, j, q] = w[q] * ρ_init
            end
        end
        copy!(f_out, f_in)

        for step in 1:3000
            KrakeNN.stream_fully_periodic_2d!(f_out, f_in, Nx, Ny)
            KrakeNN.compute_macroscopic_2d!(ρ, ux, uy, f_out)
            KrakeNN.compute_psi_2d!(ψ, ρ, ρ0)
            KrakeNN.compute_sc_force_2d!(Fx_sc, Fy_sc, ψ, G, Nx, Ny)
            KrakeNN.collide_sc_2d!(f_out, Fx_sc, Fy_sc, is_solid, ω)
            f_in, f_out = f_out, f_in
        end

        KrakeNN.compute_macroscopic_2d!(ρ, ux, uy, f_in)
        ρ_cpu = Array(ρ)

        # Verify stability
        @test !any(isnan, ρ_cpu)
        @info "Flat interface: stable, ρ range = $(round.(extrema(ρ_cpu), digits=3))"
    end
end
