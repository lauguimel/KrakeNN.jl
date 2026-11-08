using Test
using KrakeNN
using KernelAbstractions

@testset "3D Cavity Zou-He all faces" begin

    @testset "Cavity 3D lid-driven (production path) — no overshoot" begin
        # Mirrors the production run_cavity_3d driver: stream built-in bounce-back
        # for the 5 static walls + Zou-He moving lid on top. (Deliberately NOT
        # apply_bounce_back_walls_3d!, which no driver uses and which over-determines
        # the 4 top edges when combined with the lid Zou-He → divergence.)
        # N=32, Re=100 (ω≈1.68): inside the BGK stable envelope. The old N=16/Re=32
        # config sits at ω≈1.89 (near the BGK limit of 2) on a coarse grid — outside
        # the stable envelope for the stream-BB cavity; it only produced a finite
        # *overshooting* (≈1.96·u_lid, unphysical) result before the BC fix.
        N = 32
        u_lid = 0.1
        ν = 0.032  # Re = u_lid * N / ν = 100
        max_steps = 2000

        config = LBMConfig(D3Q19(); Nx=N, Ny=N, Nz=N, ν=ν, u_lid=u_lid, max_steps=max_steps)
        res = run_cavity_3d(config)

        # No NaN anywhere
        @test !any(isnan, res.ρ)
        @test !any(isnan, res.ux)
        @test !any(isnan, res.uy)
        @test !any(isnan, res.uz)

        # Mass conservation (coarse grid tolerance)
        ρ_mean = sum(res.ρ) / length(res.ρ)
        @test abs(ρ_mean - 1.0) < 0.05

        # Lid imposes u_lid (mean over interior of top face)
        ux_top = res.ux[2:N-1, 2:N-1, N]
        mean_ux_top = sum(ux_top) / length(ux_top)
        @test abs(mean_ux_top - u_lid) / u_lid < 0.1  # was 0.6 pre-fix (overshoot)

        # NO OVERSHOOT regression guard: the D3Q19 top Zou-He must not over-impose
        # lid momentum. Pre-fix this reached ~1.5*u_lid because the transverse-
        # momentum correction omitted the wall-parallel diagonal pops (parallel[6:9]).
        max_u = maximum(abs.(res.ux))
        @test max_u <= 1.05 * u_lid

        @info "Cavity 3D (production): ρ_mean=$(round(ρ_mean, digits=5)), mean_ux_top=$(round(mean_ux_top, digits=5)), max|ux|/u_lid=$(round(max_u/u_lid, digits=4))"
    end

    @testset "Zou-He velocity all 6 faces — mass conservation" begin
        # Small box: impose zero velocity on all 6 faces via Zou-He
        # Should maintain equilibrium (no flow, ρ≈1 everywhere)
        N = 8
        ν = 0.1
        max_steps = 500

        config = LBMConfig(D3Q19(); Nx=N, Ny=N, Nz=N, ν=ν, u_lid=0.0, max_steps=max_steps)
        state = initialize_3d(config, Float64)
        f_in, f_out = state.f_in, state.f_out
        ρ = state.ρ
        ux, uy, uz = state.ux, state.uy, state.uz
        is_solid = state.is_solid
        ω = Float64(omega(config))

        for step in 1:max_steps
            stream_3d!(f_out, f_in, N, N, N)

            # All 6 faces: Zou-He velocity with zero velocity
            apply_zou_he_top_3d!(f_out, 0.0, N, N, N)
            apply_zou_he_bottom_3d!(f_out, 0.0, 0.0, N, N)
            apply_zou_he_west_3d!(f_out, 0.0, 0.0, 0.0, N, N)
            apply_zou_he_east_3d!(f_out, 0.0, 0.0, 0.0, N, N, N)
            apply_zou_he_south_3d!(f_out, 0.0, 0.0, 0.0, N, N)
            apply_zou_he_north_3d!(f_out, 0.0, 0.0, 0.0, N, N, N)

            collide_3d!(f_out, is_solid, ω)
            compute_macroscopic_3d!(ρ, ux, uy, uz, f_out)
            f_in, f_out = f_out, f_in
        end

        ρ_cpu = Array(ρ)
        ux_cpu = Array(ux)
        uy_cpu = Array(uy)
        uz_cpu = Array(uz)

        # Should stay at equilibrium: ρ≈1, u≈0
        @test !any(isnan, ρ_cpu)
        @test abs(sum(ρ_cpu) / length(ρ_cpu) - 1.0) < 0.001
        @test maximum(abs.(ux_cpu)) < 1e-10
        @test maximum(abs.(uy_cpu)) < 1e-10
        @test maximum(abs.(uz_cpu)) < 1e-10

        @info "Zou-He 6 faces zero vel: ρ_mean=$(round(sum(ρ_cpu)/length(ρ_cpu), digits=8)), max|u|=$(round(maximum(abs.(ux_cpu)), sigdigits=3))"
    end

    @testset "Zou-He pressure outlet 3D — no NaN" begin
        # Quick test: west inlet + east pressure outlet
        N = 8
        ν = 0.1
        u_in = 0.01
        max_steps = 200

        config = LBMConfig(D3Q19(); Nx=N, Ny=N, Nz=N, ν=ν, u_lid=0.0, max_steps=max_steps)
        state = initialize_3d(config, Float64)
        f_in, f_out = state.f_in, state.f_out
        ρ = state.ρ
        ux, uy, uz = state.ux, state.uy, state.uz
        is_solid = state.is_solid
        ω = Float64(omega(config))

        for step in 1:max_steps
            stream_3d!(f_out, f_in, N, N, N)

            # West: velocity inlet
            apply_zou_he_west_3d!(f_out, u_in, 0.0, 0.0, N, N)
            # East: pressure outlet
            apply_zou_he_pressure_east_3d!(f_out, N, N, N; ρ_out=1.0)
            # Top/bottom: Zou-He zero velocity
            apply_zou_he_top_3d!(f_out, 0.0, N, N, N)
            apply_zou_he_bottom_3d!(f_out, 0.0, 0.0, N, N)
            # South/north: Zou-He zero velocity
            apply_zou_he_south_3d!(f_out, 0.0, 0.0, 0.0, N, N)
            apply_zou_he_north_3d!(f_out, 0.0, 0.0, 0.0, N, N, N)

            collide_3d!(f_out, is_solid, ω)
            compute_macroscopic_3d!(ρ, ux, uy, uz, f_out)
            f_in, f_out = f_out, f_in
        end

        ρ_cpu = Array(ρ)
        @test !any(isnan, ρ_cpu)
        @test !any(isinf, ρ_cpu)

        @info "Pressure outlet 3D: ρ range=[$(round(minimum(ρ_cpu), digits=5)), $(round(maximum(ρ_cpu), digits=5))]"
    end
end
