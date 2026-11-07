using Test
using KrakeNN

@testset "MRT collision" begin

    @testset "MRT cavity 2D Re=100" begin
        # Same as BGK cavity but with MRT — should be at least as accurate
        N = 64
        Re = 100.0
        u_lid = 0.1
        ν = u_lid * N / Re

        config = LBMConfig(D2Q9(); Nx=N, Ny=N, ν=ν, u_lid=u_lid, max_steps=15000)
        state = initialize_2d(config)
        f_in, f_out = state.f_in, state.f_out
        ρ, ux, uy = state.ρ, state.ux, state.uy
        is_solid = state.is_solid

        for step in 1:15000
            stream_2d!(f_out, f_in, N, N)
            apply_zou_he_north_2d!(f_out, u_lid, N, N)
            collide_mrt_2d!(f_out, is_solid, ν)
            compute_macroscopic_2d!(ρ, ux, uy, f_out)
            f_in, f_out = f_out, f_in
        end

        ux_cpu = Array(ux)

        # Ghia reference at x=0.5 centerline
        ghia_y = [0.0, 0.0547, 0.0625, 0.5, 0.9766, 1.0]
        ghia_u = [0.0, -0.03717, -0.04192, -0.20581, 0.84123, 1.0]

        i_center = N ÷ 2
        u_centerline = ux_cpu[i_center, :] ./ u_lid

        max_error = 0.0
        for (yg, ug) in zip(ghia_y, ghia_u)
            j = clamp(round(Int, yg * (N - 1)) + 1, 1, N)
            max_error = max(max_error, abs(u_centerline[j] - ug))
        end

        @test max_error < 0.15  # MRT on 64² (coarser than BGK 128² test)
        @test !any(isnan, ux_cpu)
        @info "MRT cavity Re=100 (64²): max error vs Ghia = $(round(max_error, digits=3))"
    end
end
