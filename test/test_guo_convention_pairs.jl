using Test
using KrakeNN

@testset "Guo convention pairs (isothermal 2D)" begin
    # Periodic-box analytical: u_phys(N) = gx * N from rest.
    nx, ny, steps = 32, 32, 500
    gx, gy = 1e-5, 0.0
    tol = 50 * eps(Float64) * steps

    function periodic_box_mean_ux(readout!)
        config = LBMConfig(D2Q9(); Nx=nx, Ny=ny, ν=0.1, u_lid=0.0, max_steps=steps)
        state = initialize_2d(config)
        f_in, f_out = state.f_in, state.f_out
        ρ, ux, uy = state.ρ, state.ux, state.uy
        is_solid = state.is_solid
        ω = omega(config)

        for _ in 1:steps
            stream_fully_periodic_2d!(f_out, f_in, nx, ny)
            collide_guo_2d!(f_out, is_solid, ω, gx, gy)
            readout!(ρ, ux, uy, f_out)
            f_in, f_out = f_out, f_in
        end

        ux_cpu = Array(ux)
        return sum(ux_cpu) / length(ux_cpu)
    end

    @testset "collide_guo_2d! + compute_macroscopic_2d! (production)" begin
        mean_ux = periodic_box_mean_ux() do ρ, ux, uy, f
            compute_macroscopic_2d!(ρ, ux, uy, f)
        end

        @test abs(mean_ux - gx * steps) <= tol
    end

    @testset "broken pair regression sentinel" begin
        mean_ux = periodic_box_mean_ux() do ρ, ux, uy, f
            compute_macroscopic_forced_2d!(ρ, ux, uy, f, gx, gy)
        end

        @test abs((mean_ux - gx * steps) - gx / 2) <= tol
    end
end
