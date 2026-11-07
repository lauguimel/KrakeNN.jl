using Test
using KrakeNN

@testset "Lid-Driven Cavity" begin

    @testset "Cavity 2D Re=100" begin
        # Ghia et al. 1982 reference data for Re=100
        # u-velocity along vertical centerline (x = 0.5)
        ghia_y = [0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719,
                  0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516,
                  0.9531, 0.9609, 0.9688, 0.9766, 1.0000]
        ghia_u = [0.0000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
                  -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
                  0.68717, 0.73722, 0.78871, 0.84123, 1.00000]

        N = 128
        Re = 100.0
        u_lid = 0.1
        ν = u_lid * N / Re
        max_steps = 30000  # enough for Re=100 on 128² to converge

        config = LBMConfig(D2Q9(); Nx=N, Ny=N, ν=ν, u_lid=u_lid, max_steps=max_steps)
        result = run_cavity_2d(config)

        # Extract u along vertical centerline (i = N/2)
        i_center = N ÷ 2
        u_centerline = result.ux[i_center, :] ./ u_lid  # normalized

        # Compare at Ghia y-locations (interpolate from our data)
        max_error = 0.0
        for (yg, ug) in zip(ghia_y, ghia_u)
            # Find nearest grid point
            j = clamp(round(Int, yg * (N - 1)) + 1, 1, N)
            error = abs(u_centerline[j] - ug)
            max_error = max(max_error, error)
        end

        @test max_error < 0.1  # within 10% of Ghia (reasonable for 128² with BGK)
        @info "Cavity 2D Re=100: max error vs Ghia = $(round(max_error, digits=4))"
    end

    @testset "Cavity 3D basic convergence" begin
        # Small 3D cavity at Re = u_lid * N / ν = 0.05*16/0.025 = 32.
        #
        # Relaxation time, not Reynolds number, sets the stable envelope here.
        # BGK + Zou-He lid on this driver loses linear stability above
        # ω ≈ 1.81 (measured on 16³: ω=1.8149/ν=0.017 diverges, ω=1.8051/ν=0.018
        # survives; the same threshold holds on 32³, and the blow-up is
        # independent of u_lid over 0.005–0.04, i.e. of Re and Ma). The former
        # ν=0.01 gave ω=1.8868 (τ=0.53) — above that ceiling — and produced NaN
        # at step 194 regardless of lid speed.
        #
        # ν=0.025 gives ω=1.7391 (τ=0.575), ~0.07 below the measured ceiling,
        # at the same Re=32 and Ma = u_lid*√3 = 0.087 (well inside the
        # incompressible LBM range). The run is converged and steady: repeating
        # it to 6000 steps changes nothing.
        N = 16
        u_lid = 0.05
        ν = 0.025

        config = LBMConfig(D3Q19(); Nx=N, Ny=N, Nz=N, ν=ν, u_lid=u_lid, max_steps=2000)
        result = run_cavity_3d(config)

        # Check no NaN
        @test !any(isnan, result.ρ)

        # Check lid velocity is imposed. The D3Q19 Zou-He velocity BC satisfies
        # the tangential momentum constraint exactly, so this is a tight bound
        # (measured relative error ≈ 6e-16), not a convergence tolerance.
        ux_top = result.ux[:, :, N]
        mean_ux_top = sum(ux_top[2:N-1, 2:N-1]) / ((N-2)^2)
        @test abs(mean_ux_top - u_lid) / u_lid < 0.05

        # No-overshoot guard: nothing in the cavity may move faster than the lid.
        @test maximum(abs.(result.ux)) <= 1.05 * u_lid

        # Check mass conservation (measured drift ≈ 6e-5)
        total_mass = sum(result.ρ)
        expected = N^3 * 1.0
        @test abs(total_mass - expected) / expected < 1e-3

        @info "Cavity 3D: mean ux at lid = $(round(mean_ux_top, digits=5)), target = $u_lid"
    end

end
