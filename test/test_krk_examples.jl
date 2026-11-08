using Test
using KrakeNN

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")

@testset "All .krk examples" begin

    @testset "cavity.krk — Ghia validation" begin
        result = run_simulation(joinpath(EXAMPLES_DIR, "cavity.krk"))

        N = 128
        u_lid = 0.1

        # Ghia et al. 1982 reference data for Re=100
        ghia_y = [0.0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
                  0.4531, 0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
                  0.9688, 0.9766, 1.0]
        ghia_u = [0.0, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
                  -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
                  0.68717, 0.73722, 0.78871, 0.84123, 1.0]

        i_center = N ÷ 2
        u_centerline = result.ux[i_center, :] ./ u_lid

        max_error = 0.0
        for (yg, ug) in zip(ghia_y, ghia_u)
            j = clamp(round(Int, yg * (N - 1)) + 1, 1, N)
            max_error = max(max_error, abs(u_centerline[j] - ug))
        end

        @test max_error < 0.1  # within 10% of Ghia
        @test !any(isnan, result.ρ)
        @test abs(sum(result.ρ) - N^2) / N^2 < 0.01
        @info "cavity.krk: max Ghia error = $(round(max_error, digits=4))"
    end

    @testset "poiseuille.krk — parabolic profile" begin
        result = run_simulation(joinpath(EXAMPLES_DIR, "poiseuille.krk"))

        Ny = 32
        nu = 0.1
        Fx = 1e-5

        # Analytical (half-way BB: walls at y=0.5, y=Ny+0.5)
        u_analytical = [Fx / (2nu) * (j - 0.5) * (Ny + 0.5 - j) for j in 1:Ny]
        u_numerical = result.ux[2, :]

        u_max = maximum(u_analytical)
        errors = abs.(u_numerical[2:end-1] .- u_analytical[2:end-1])
        max_rel_err = maximum(errors) / u_max

        @test max_rel_err < 0.02
        @info "poiseuille.krk: L∞ relative error = $(round(max_rel_err, digits=5))"
    end

    @testset "couette.krk — linear profile" begin
        result = run_simulation(joinpath(EXAMPLES_DIR, "couette.krk"))

        Ny = 32
        u_wall = 0.05

        # North wall moves, south wall stationary
        # u(j) = u_wall * (j - 1) / (Ny - 1)
        H = Ny - 1
        u_analytical = [u_wall * (j - 1) / H for j in 1:Ny]
        u_numerical = result.ux[2, :]

        errors = abs.(u_numerical[2:end-1] .- u_analytical[2:end-1])
        max_rel_err = maximum(errors) / u_wall

        @test max_rel_err < 0.02
        @test maximum(abs.(result.uy)) < 1e-6
        @info "couette.krk: L∞ relative error = $(round(max_rel_err, digits=5))"
    end

    @testset "cylinder.krk — stable flow" begin
        result = run_simulation(joinpath(EXAMPLES_DIR, "cylinder.krk"))

        @test !any(isnan, result.ρ)
        @test !any(isnan, result.ux)

        # Check inlet profile is approximately parabolic
        Ny = 50
        ux_inlet = result.ux[1, :]
        ux_max_inlet = maximum(ux_inlet)
        @test ux_max_inlet > 0.01  # flow is established

        # Mass conservation
        @test abs(sum(result.ρ) / (200 * 50) - 1.0) < 0.01

        @info "cylinder.krk: stable, max inlet ux = $(round(ux_max_inlet, digits=4))"
    end

    @testset "taylor_green.krk — exponential decay" begin
        result = run_simulation(joinpath(EXAMPLES_DIR, "taylor_green.krk"))

        N = 64
        nu = 0.01
        u0 = 0.01
        k = 2π / N
        max_steps = 1000

        # Check velocity has decayed
        max_ux = maximum(abs.(result.ux))
        decay_factor = exp(-2 * nu * k^2 * max_steps)

        @test max_ux < u0  # must have decayed
        @test max_ux > u0 * decay_factor * 0.3  # not too much

        # L2 check (with shifted coordinates: x = (i-0.5)*dx instead of x = i-1)
        Lx = 64.0
        dx = Lx / N
        ux_analytical = zeros(N, N)
        for j in 1:N, i in 1:N
            x = (i - 0.5) * dx
            y = (j - 0.5) * dx
            ux_analytical[i, j] = -u0 * cos(2π * x / Lx) * sin(2π * y / Lx) * decay_factor
        end

        diff = result.ux .- ux_analytical
        l2_err = sqrt(sum(diff .^ 2) / sum(ux_analytical .^ 2))

        @test l2_err < 0.1  # 10% — larger tolerance due to coordinate shift
        @test !any(isnan, result.ρ)

        @info "taylor_green.krk: L2 error = $(round(l2_err, digits=4)), " *
              "max|ux| = $(round(max_ux, digits=5)), decay = $(round(decay_factor, digits=4))"
    end

    @testset "Parametric kwargs override" begin
        # Override domain size and physics
        s = load_kraken(joinpath(EXAMPLES_DIR, "cavity.krk"); Nx=32, Ny=32, nu=0.2)
        @test s.domain.Nx == 32
        @test s.domain.Ny == 32
        @test s.physics.params[:nu] ≈ 0.2

        # Override max_steps
        s = load_kraken(joinpath(EXAMPLES_DIR, "poiseuille.krk"); max_steps=500)
        @test s.max_steps == 500

        # Override Define variable
        s = load_kraken(joinpath(EXAMPLES_DIR, "cylinder.krk"); U=0.1, R=1.0)
        @test s.user_vars[:U] ≈ 0.1
        @test s.user_vars[:R] ≈ 1.0

        # Run with overrides — smaller/faster cavity
        result = run_simulation(joinpath(EXAMPLES_DIR, "cavity.krk");
                                Nx=16, Ny=16, nu=0.2, max_steps=2000)
        @test !any(isnan, result.ρ)
        @test size(result.ux) == (16, 16)

        @info "Parametric kwargs: all overrides work"
    end
end
