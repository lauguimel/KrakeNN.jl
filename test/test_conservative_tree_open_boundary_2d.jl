using Test
using KrakeNN

function _cell_rho_ux_2d(Fcell, volume)
    rho = mass_F(Fcell) / volume
    ux = (momentum_F(Fcell)[1] / volume) / rho
    return rho, ux
end

@testset "Conservative tree open boundary 2D" begin
    @testset "composite Zou-He west covers fine inlet and coarse gaps" begin
        nx, ny = 7, 6
        volume_coarse = 1.0
        volume_fine = 0.25
        u_in = 0.035
        patch = create_conservative_tree_patch_2d(1:2, 2:5)
        coarse = zeros(Float64, nx, ny, 9)
        fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, 1.0, 0.0, 0.0)
        fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, 1.0, 0.0, 0.0)

        apply_composite_zou_he_west_F_2d!(
            coarse, patch, u_in, volume_coarse, volume_fine)

        for J in (1, 6)
            rho, ux = _cell_rho_ux_2d(@view(coarse[1, J, :]), volume_coarse)
            @test isfinite(rho)
            @test isapprox(ux, u_in; atol=1e-14, rtol=0)
        end
        for jf in axes(patch.fine_F, 2)
            rho, ux = _cell_rho_ux_2d(@view(patch.fine_F[1, jf, :]), volume_fine)
            @test isfinite(rho)
            @test isapprox(ux, u_in; atol=1e-14, rtol=0)
        end
        for J in patch.parent_j_range
            il, jl = 1, J - first(patch.parent_j_range) + 1
            shadow = @view patch.coarse_shadow_F[il, jl, :]
            children = @view patch.fine_F[1:2, (2 * jl - 1):(2 * jl), :]
            expected = zeros(Float64, 9)
            coalesce_F_2d!(expected, children)
            @test isapprox(shadow, expected; atol=1e-14, rtol=0)
        end
    end

    @testset "composite Zou-He east covers fine outlet and coarse gaps" begin
        nx, ny = 7, 6
        volume_coarse = 1.0
        volume_fine = 0.25
        rho_out = 1.07
        patch = create_conservative_tree_patch_2d(6:7, 2:5)
        coarse = zeros(Float64, nx, ny, 9)
        fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, 1.0, 0.02, 0.0)
        fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, 1.0, 0.02, 0.0)

        apply_composite_zou_he_pressure_east_F_2d!(
            coarse, patch, volume_coarse, volume_fine; rho_out=rho_out)

        for J in (1, 6)
            rho, ux = _cell_rho_ux_2d(@view(coarse[nx, J, :]), volume_coarse)
            @test isfinite(ux)
            @test isapprox(rho, rho_out; atol=1e-14, rtol=0)
        end
        for jf in axes(patch.fine_F, 2)
            rho, ux = _cell_rho_ux_2d(@view(patch.fine_F[end, jf, :]), volume_fine)
            @test isfinite(ux)
            @test isapprox(rho, rho_out; atol=1e-14, rtol=0)
        end
    end

    @testset "open route step imposes fine inlet and outlet moments" begin
        nx, ny = 6, 5
        volume_coarse = 1.0
        volume_fine = 0.25
        u_in = 0.025
        rho_out = 1.03
        patch = create_conservative_tree_patch_2d(1:nx, 2:4)
        patch_next = create_conservative_tree_patch_2d(1:nx, 2:4)
        topology = create_conservative_tree_topology_2d(nx, ny, patch)
        coarse = zeros(Float64, nx, ny, 9)
        coarse_next = similar(coarse)
        fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, 1.0, u_in, 0.0)
        fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, 1.0, u_in, 0.0)

        stream_composite_routes_zou_he_x_wall_y_F_2d!(
            coarse_next, patch_next, coarse, patch, topology;
            u_in=u_in, rho_out=rho_out,
            volume_coarse=volume_coarse, volume_fine=volume_fine)

        for J in (1, ny)
            rho_w, ux_w = _cell_rho_ux_2d(@view(coarse_next[1, J, :]), volume_coarse)
            rho_e, ux_e = _cell_rho_ux_2d(@view(coarse_next[nx, J, :]), volume_coarse)
            @test isfinite(rho_w)
            @test isfinite(ux_e)
            @test isapprox(ux_w, u_in; atol=1e-14, rtol=0)
            @test isapprox(rho_e, rho_out; atol=1e-14, rtol=0)
        end
        for jf in axes(patch_next.fine_F, 2)
            rho_w, ux_w = _cell_rho_ux_2d(@view(patch_next.fine_F[1, jf, :]), volume_fine)
            rho_e, ux_e = _cell_rho_ux_2d(@view(patch_next.fine_F[end, jf, :]), volume_fine)
            @test isfinite(rho_w)
            @test isfinite(ux_e)
            @test isapprox(ux_w, u_in; atol=1e-14, rtol=0)
            @test isapprox(rho_e, rho_out; atol=1e-14, rtol=0)
        end
        @test active_mass_F(coarse_next, patch_next) > 0
    end

    @testset "solid-aware composite Zou-He skips step cells" begin
        nx, ny = 8, 6
        volume_coarse = 1.0
        volume_fine = 0.25
        u_in = 0.02
        patch = create_conservative_tree_patch_2d(1:4, 1:4)
        coarse = zeros(Float64, nx, ny, 9)
        fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, 1.0, 0.0, 0.0)
        fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, 1.0, 0.0, 0.0)
        is_solid = falses(2 * nx, 2 * ny)
        is_solid[1:6, 1:4] .= true
        solid_before = copy(@view(patch.fine_F[1, 1:4, :]))

        apply_composite_zou_he_west_F_2d!(
            coarse, patch, is_solid, u_in, volume_coarse, volume_fine)

        @test patch.fine_F[1, 1:4, :] == solid_before
        for jf in 5:8
            rho, ux = _cell_rho_ux_2d(@view(patch.fine_F[1, jf, :]), volume_fine)
            @test isfinite(rho)
            @test isapprox(ux, u_in; atol=1e-14, rtol=0)
        end
        bad_solid = copy(is_solid)
        bad_solid[2 * nx - 1, 2 * ny - 1] = true
        @test_throws ArgumentError apply_composite_zou_he_west_F_2d!(
            coarse, patch, bad_solid, u_in, volume_coarse, volume_fine)
    end

    @testset "open route solid step nominal smoke remains finite" begin
        result = run_conservative_tree_bfs_route_native_2d()

        @test result.flow == :bfs_route_native
        @test result.steps == 240
        @test isfinite(result.mass_final)
        @test abs(result.mass_drift) / result.mass_initial < 0.2
        @test result.ux_mean > 0.005
        @test abs(result.uy_mean) < 0.01
        @test sum(result.is_solid_leaf) > 0
    end

    @testset "local BFS route tracks leaf open-solid oracle" begin
        route = run_conservative_tree_bfs_route_native_2d(; steps=240)
        oracle = run_conservative_tree_bfs_macroflow_2d(; steps=240)

        route_rel = abs(route.mass_drift) / route.mass_initial
        oracle_rel = abs(oracle.mass_drift) / oracle.mass_initial

        @test route.steps == oracle.steps
        @test abs(route_rel - oracle_rel) < 0.01
        @test abs(route.ux_mean - oracle.ux_mean) < 2e-3
        @test abs(route.uy_mean - oracle.uy_mean) < 1e-3
    end

    @testset "full-patch BFS route matches leaf open-solid oracle" begin
        route = run_conservative_tree_bfs_route_native_2d(
            ; patch_i_range=1:28, patch_j_range=1:14, steps=160)
        oracle = run_conservative_tree_bfs_macroflow_2d(
            ; patch_i_range=1:28, patch_j_range=1:14, steps=160)

        @test route.steps == oracle.steps
        @test isapprox(route.mass_initial, oracle.mass_initial; atol=1e-12, rtol=0)
        @test isapprox(route.mass_final, oracle.mass_final; atol=1e-10, rtol=0)
        @test isapprox(route.ux_mean, oracle.ux_mean; atol=1e-12, rtol=0)
        @test isapprox(route.uy_mean, oracle.uy_mean; atol=1e-12, rtol=0)
        @test isapprox(route.mass_drift, oracle.mass_drift; atol=1e-10, rtol=0)
    end

    @testset "open channel local patch mass ledger exposes coarse fine drift" begin
        local_patch = KrakeNN.run_conservative_tree_open_channel_mass_ledger_2d(
            ; steps=160)
        full_patch = KrakeNN.run_conservative_tree_open_channel_mass_ledger_2d(
            ; patch_i_range=1:18, patch_j_range=1:10, steps=160)

        local_rel = abs(local_patch.mass_drift) / local_patch.mass_initial
        full_rel = abs(full_patch.mass_drift) / full_patch.mass_initial

        @test length(local_patch.mass_history) == local_patch.steps + 1
        @test length(local_patch.ux_history) == local_patch.steps + 1
        @test all(isfinite, local_patch.mass_history)
        @test all(isfinite, local_patch.ux_history)
        @test local_patch.ux_history[end] > 0
        @test full_rel < 0.02
        @test local_rel < 0.02
    end

    @testset "open channel rest state survives local leaf-equivalent interface" begin
        local_patch = KrakeNN.run_conservative_tree_open_channel_mass_ledger_2d(
            ; u_in=0.0, steps=160)
        full_patch = KrakeNN.run_conservative_tree_open_channel_mass_ledger_2d(
            ; u_in=0.0, patch_i_range=1:18, patch_j_range=1:10, steps=160)

        local_rel = abs(local_patch.mass_drift) / local_patch.mass_initial
        full_rel = abs(full_patch.mass_drift) / full_patch.mass_initial

        @test full_rel < 1e-12
        @test abs(full_patch.ux_history[end]) < 1e-12
        @test local_rel < 1e-12
        @test abs(local_patch.ux_history[end]) < 1e-12
    end

    @testset "short open channel with inlet-spanning patch remains finite" begin
        nx, ny = 8, 6
        volume_coarse = 1.0
        volume_fine = 0.25
        u_in = 0.015
        rho_out = 1.0
        omega = 1.0
        patch = create_conservative_tree_patch_2d(1:4, 2:5)
        patch_next = create_conservative_tree_patch_2d(1:4, 2:5)
        topology = create_conservative_tree_topology_2d(nx, ny, patch)
        coarse = zeros(Float64, nx, ny, 9)
        coarse_next = similar(coarse)
        fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, 1.0, u_in, 0.0)
        fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, 1.0, u_in, 0.0)
        apply_composite_zou_he_west_F_2d!(
            coarse, patch, u_in, volume_coarse, volume_fine)
        apply_composite_zou_he_pressure_east_F_2d!(
            coarse, patch, volume_coarse, volume_fine; rho_out=rho_out)
        mass_initial = active_mass_F(coarse, patch)

        for _ in 1:10
            collide_BGK_composite_F_2d!(
                coarse, patch, volume_coarse, volume_fine, omega, omega)
            stream_composite_routes_zou_he_x_wall_y_F_2d!(
                coarse_next, patch_next, coarse, patch, topology;
                u_in=u_in, rho_out=rho_out,
                volume_coarse=volume_coarse, volume_fine=volume_fine)
            coarse, coarse_next = coarse_next, coarse
            patch, patch_next = patch_next, patch
        end

        mass_final = active_mass_F(coarse, patch)
        @test isfinite(mass_final)
        @test abs(mass_final - mass_initial) / mass_initial < 0.08
        for jf in axes(patch.fine_F, 2)
            _, ux = _cell_rho_ux_2d(@view(patch.fine_F[1, jf, :]), volume_fine)
            @test isapprox(ux, u_in; atol=1e-14, rtol=0)
        end
    end
end
