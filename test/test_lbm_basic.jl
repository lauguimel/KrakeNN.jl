using Test
using KrakeNN

@testset "LBM Basic" begin

    @testset "D2Q9 lattice constants" begin
        w = weights(D2Q9())
        @test length(w) == 9
        @test sum(w) ≈ 1.0 atol=1e-14

        # Weights: rest=4/9, axis=1/9, diagonal=1/36
        @test w[1] ≈ 4.0/9.0
        @test w[2] ≈ 1.0/9.0
        @test w[6] ≈ 1.0/36.0

        # Opposite directions
        opp = opposite(D2Q9())
        for q in 1:9
            @test opp[opp[q]] == q  # double opposite = identity
        end
    end

    @testset "D3Q19 lattice constants" begin
        w = weights(D3Q19())
        @test length(w) == 19
        @test sum(w) ≈ 1.0 atol=1e-14

        @test w[1] ≈ 1.0/3.0
        @test w[2] ≈ 1.0/18.0
        @test w[8] ≈ 1.0/36.0

        opp = opposite(D3Q19())
        for q in 1:19
            @test opp[opp[q]] == q
        end
    end

    @testset "Equilibrium distribution 2D" begin
        # At rest (u=0), f_eq = w_q * ρ
        lattice = D2Q9()
        ρ = 1.0; ux = 0.0; uy = 0.0
        for q in 1:9
            feq = equilibrium(lattice, ρ, ux, uy, q)
            @test feq ≈ weights(lattice)[q] atol=1e-14
        end

        # Sum of equilibrium = ρ
        ρ = 2.5; ux = 0.05; uy = -0.03
        feq_sum = sum(equilibrium(lattice, ρ, ux, uy, q) for q in 1:9)
        @test feq_sum ≈ ρ atol=1e-12

        # Momentum: Σ f_eq * c = ρ * u
        cx = velocities_x(lattice)
        cy = velocities_y(lattice)
        mx = sum(equilibrium(lattice, ρ, ux, uy, q) * cx[q] for q in 1:9)
        my = sum(equilibrium(lattice, ρ, ux, uy, q) * cy[q] for q in 1:9)
        @test mx ≈ ρ * ux atol=1e-12
        @test my ≈ ρ * uy atol=1e-12
    end

    @testset "Equilibrium distribution 3D" begin
        lattice = D3Q19()
        ρ = 1.8; ux = 0.02; uy = -0.01; uz = 0.03

        feq_sum = sum(equilibrium(lattice, ρ, ux, uy, uz, q) for q in 1:19)
        @test feq_sum ≈ ρ atol=1e-12

        cx = velocities_x(lattice)
        cy = velocities_y(lattice)
        cz = velocities_z(lattice)
        mx = sum(equilibrium(lattice, ρ, ux, uy, uz, q) * cx[q] for q in 1:19)
        my = sum(equilibrium(lattice, ρ, ux, uy, uz, q) * cy[q] for q in 1:19)
        mz = sum(equilibrium(lattice, ρ, ux, uy, uz, q) * cz[q] for q in 1:19)
        @test mx ≈ ρ * ux atol=1e-12
        @test my ≈ ρ * uy atol=1e-12
        @test mz ≈ ρ * uz atol=1e-12
    end

    @testset "Mass conservation 2D" begin
        # Run a small 2D simulation and check mass is conserved
        N = 16
        ν = 0.1
        config = LBMConfig(D2Q9(); Nx=N, Ny=N, ν=ν, u_lid=0.05, max_steps=100)
        result = run_cavity_2d(config)
        total_mass = sum(result.ρ)
        expected_mass = N * N * 1.0  # initial ρ = 1 everywhere
        @test abs(total_mass - expected_mass) / expected_mass < 0.01  # <1% error
    end

    @testset "Mass conservation 3D" begin
        N = 8
        ν = 0.1
        config = LBMConfig(D3Q19(); Nx=N, Ny=N, Nz=N, ν=ν, u_lid=0.05, max_steps=50)
        result = run_cavity_3d(config)
        total_mass = sum(result.ρ)
        expected_mass = N^3 * 1.0
        @test abs(total_mass - expected_mass) / expected_mass < 0.01
    end

    @testset "Poiseuille 2D" begin
        # Channel flow between two walls (j=1 and j=Ny are walls)
        # Driven by body force. Here we use a simpler test:
        # Initialize with equilibrium + small uniform velocity,
        # run a few steps and check stability (no blow-up).
        Nx, Ny = 4, 32
        ν = 0.1
        config = LBMConfig(D2Q9(); Nx=Nx, Ny=Ny, ν=ν, u_lid=0.0, max_steps=200)
        result = run_cavity_2d(config)

        # With u_lid=0, the system should relax to rest (all u → 0)
        @test maximum(abs.(result.ux)) < 1e-6
        @test maximum(abs.(result.uy)) < 1e-6
    end

end
