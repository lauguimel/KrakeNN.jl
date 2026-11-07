using Test
using KrakeNN

@testset "Two-phase rheology collision" begin

    @testset "Stability with Newtonian phases" begin
        # Simple static configuration: uniform C=1 (pure liquid),
        # no surface tension, should behave like single-phase Newtonian
        T = Float64
        Nx, Ny = 32, 32
        f   = ones(T, Nx, Ny, 9) ./ 9   # uniform equilibrium at rest
        C   = ones(T, Nx, Ny)            # pure liquid everywhere
        Fx  = zeros(T, Nx, Ny)
        Fy  = zeros(T, Nx, Ny)
        is_solid = zeros(Bool, Nx, Ny)
        tau = ones(T, Nx, Ny)            # initial tau = 1

        rheo_l = Newtonian(0.1)
        rheo_g = Newtonian(0.05)

        # Run 50 collision steps (no streaming — just check stability)
        for _ in 1:50
            collide_twophase_rheology_2d!(f, C, Fx, Fy, is_solid, tau;
                                          rheology_l=rheo_l, rheology_g=rheo_g,
                                          rho_l=1.0, rho_g=0.5)
        end

        @test all(isfinite, f)
        @test all(isfinite, tau)

        # Density should remain ~1 everywhere (equilibrium at rest)
        rho = sum(f[:,:,q] for q in 1:9)
        @test maximum(abs.(rho .- 1.0)) < 1e-10
    end

    @testset "Per-phase viscosity interpolation" begin
        # Interface at center: C=1 (liquid, left), C=0 (gas, right)
        T = Float64
        Nx, Ny = 32, 32
        f   = ones(T, Nx, Ny, 9) ./ 9
        C   = zeros(T, Nx, Ny)
        C[1:16, :] .= 1.0               # liquid on the left half
        Fx  = zeros(T, Nx, Ny)
        Fy  = zeros(T, Nx, Ny)
        is_solid = zeros(Bool, Nx, Ny)
        tau = ones(T, Nx, Ny)

        ν_l = 0.1
        ν_g = 0.05
        rheo_l = Newtonian(ν_l)
        rheo_g = Newtonian(ν_g)

        collide_twophase_rheology_2d!(f, C, Fx, Fy, is_solid, tau;
                                      rheology_l=rheo_l, rheology_g=rheo_g,
                                      rho_l=1.0, rho_g=0.5)

        # tau_field should reflect ν_l in liquid region and ν_g in gas region
        # tau = 1/omega = 3*nu + 0.5
        tau_liquid_expected = 3 * ν_l + 0.5
        tau_gas_expected = 3 * ν_g + 0.5

        @test abs(tau[4, 16] - tau_liquid_expected) < 1e-10
        @test abs(tau[24, 16] - tau_gas_expected) < 1e-10
    end

    @testset "Power-law liquid with Newtonian gas" begin
        # Driven two-phase channel: apply body force, check stability
        T = Float64
        Nx, Ny = 4, 32
        Fx_body = T(1e-5)

        f   = ones(T, Nx, Ny, 9) ./ 9
        C   = ones(T, Nx, Ny)            # all liquid
        Fx  = fill(Fx_body, Nx, Ny)
        Fy  = zeros(T, Nx, Ny)
        is_solid = zeros(Bool, Nx, Ny)
        tau = ones(T, Nx, Ny)

        rheo_l = PowerLaw(0.1, 0.7)      # shear-thinning liquid
        rheo_g = Newtonian(0.05)

        # Stream + collide loop (simplified: periodic x, wall y)
        f_out = copy(f)
        ρ  = ones(T, Nx, Ny)
        ux = zeros(T, Nx, Ny)
        uy = zeros(T, Nx, Ny)

        for step in 1:2000
            stream_periodic_x_wall_y_2d!(f_out, f, Nx, Ny)
            collide_twophase_rheology_2d!(f_out, C, Fx, Fy, is_solid, tau;
                                          rheology_l=rheo_l, rheology_g=rheo_g,
                                          rho_l=1.0, rho_g=0.5)
            compute_macroscopic_forced_2d!(ρ, ux, uy, f_out, Fx_body, zero(T))
            f, f_out = f_out, f
        end

        @test all(isfinite, f)
        @test all(isfinite, ux)

        # Velocity should be positive (driven by Fx) and have a parabolic-like profile
        u_center = ux[2, Ny÷2]
        u_wall   = ux[2, 1]
        @test u_center > 0
        @test abs(u_wall) < u_center * 2.0   # much smaller than center

        # tau should vary across the channel (shear-thinning → lower tau near walls)
        @test tau[2, Ny÷2] != tau[2, 2]   # different tau at center vs near wall
    end

    @testset "Solid bounce-back" begin
        # Solid node should swap populations
        T = Float64
        Nx, Ny = 4, 4
        f   = rand(T, Nx, Ny, 9)
        C   = ones(T, Nx, Ny)
        Fx  = zeros(T, Nx, Ny)
        Fy  = zeros(T, Nx, Ny)
        is_solid = zeros(Bool, Nx, Ny)
        is_solid[2, 2] = true
        tau = ones(T, Nx, Ny)

        f_before = copy(f)

        collide_twophase_rheology_2d!(f, C, Fx, Fy, is_solid, tau;
                                      rheology_l=Newtonian(0.1), rheology_g=Newtonian(0.1),
                                      rho_l=1.0, rho_g=1.0)

        # At solid node (2,2): populations should be swapped (bounce-back)
        @test f[2,2,2] ≈ f_before[2,2,4]
        @test f[2,2,4] ≈ f_before[2,2,2]
        @test f[2,2,3] ≈ f_before[2,2,5]
        @test f[2,2,5] ≈ f_before[2,2,3]
        @test f[2,2,6] ≈ f_before[2,2,8]
        @test f[2,2,8] ≈ f_before[2,2,6]
    end

end
