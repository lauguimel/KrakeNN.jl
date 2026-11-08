using Test
using KrakeNN

@testset "Prescribed-velocity VOF advection" begin

    @testset "Rigid rotation of circle — mass conservation" begin
        N = 64
        R = 15.0
        cx, cy = N / 2, N / 2
        angular_vel = 2π / (4 * N)  # one full rotation in 4*N steps

        velocity_fn(x, y, t) = (-(y - cy) * angular_vel, (x - cx) * angular_vel)
        init_fn(x, y) = 0.5 * (1 - tanh((sqrt((x - cx)^2 + (y - cy)^2) - R) / 2))

        max_steps = 4 * N  # one full rotation
        result = run_advection_2d(; Nx=N, Ny=N, max_steps=max_steps,
                                   velocity_fn=velocity_fn, init_C_fn=init_fn)

        # Mass conservation: total C should be preserved
        mass_initial = result.mass_history[1]
        mass_final = result.mass_history[end]
        rel_mass_err = abs(mass_final - mass_initial) / mass_initial
        @test rel_mass_err < 0.05  # <5% mass loss
        @info "Circle rotation: mass conservation error = $(round(rel_mass_err*100, digits=2))%"
    end

    @testset "Rigid rotation of circle — shape error" begin
        N = 64
        R = 15.0
        cx, cy = N / 2, N / 2
        angular_vel = 2π / (4 * N)

        velocity_fn(x, y, t) = (-(y - cy) * angular_vel, (x - cx) * angular_vel)
        init_fn(x, y) = 0.5 * (1 - tanh((sqrt((x - cx)^2 + (y - cy)^2) - R) / 2))

        max_steps = 4 * N
        result = run_advection_2d(; Nx=N, Ny=N, max_steps=max_steps,
                                   velocity_fn=velocity_fn, init_C_fn=init_fn)

        # Shape error: L1 difference between initial and final C
        L1_err = sum(abs.(result.C .- result.C0)) / sum(result.C0)
        @test L1_err < 0.5  # first-order upwind on 64² has significant diffusion
        @info "Circle rotation: L1 shape error = $(round(L1_err, digits=4))"
    end

    @testset "Zalesak disk — notched disk rotation" begin
        N = 100
        R = 15.0
        cx, cy = 50.0, 75.0
        slot_w = 5.0
        slot_h = 25.0
        angular_vel = 2π / (N * π)  # approx one rotation in ~314 steps

        function zalesak_init(x, y)
            r = sqrt((x - cx)^2 + (y - cy)^2)
            disk = 0.5 * (1 - tanh((r - R) / 2))
            # Cut slot
            in_slot = abs(x - cx) < slot_w / 2 && y < cy + slot_h / 2 && y > cy - R
            return in_slot ? 0.0 : disk
        end

        velocity_fn(x, y, t) = (-(y - 50.0) * angular_vel, (x - 50.0) * angular_vel)

        max_steps = round(Int, 2π / angular_vel)  # one full rotation
        result = run_advection_2d(; Nx=N, Ny=N, max_steps=max_steps,
                                   velocity_fn=velocity_fn, init_C_fn=zalesak_init)

        mass_initial = result.mass_history[1]
        mass_final = result.mass_history[end]
        rel_mass_err = abs(mass_final - mass_initial) / mass_initial
        @test rel_mass_err < 0.1  # <10% mass loss
        @info "Zalesak disk: mass error = $(round(rel_mass_err*100, digits=2))%"
    end

    @testset "Reversed vortex — time-dependent velocity" begin
        N = 64
        T_period = 8.0  # full period for reversal
        R = 0.15 * N
        cx, cy = 0.5 * N, 0.75 * N

        function vortex_velocity(x, y, t)
            xn = x / N  # normalize to [0,1]
            yn = y / N
            tn = t / T_period
            scale = cos(π * tn) * 0.5  # reverses at t = T/2
            vx = -sin(π * xn) * cos(π * yn) * scale
            vy =  cos(π * xn) * sin(π * yn) * scale
            return (vx, vy)
        end

        init_fn(x, y) = 0.5 * (1 - tanh((sqrt((x - cx)^2 + (y - cy)^2) - R) / 2))

        max_steps = round(Int, T_period)
        result = run_advection_2d(; Nx=N, Ny=N, max_steps=max_steps,
                                   velocity_fn=vortex_velocity, init_C_fn=init_fn)

        # After full reversal, shape should approximately recover
        L1_err = sum(abs.(result.C .- result.C0)) / sum(result.C0)
        @test L1_err < 0.5  # tolerance for first-order on 64²
        @info "Reversed vortex: L1 recovery error = $(round(L1_err, digits=4))"
    end

end
