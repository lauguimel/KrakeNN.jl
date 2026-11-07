using Test
using KrakeNN

function _stream_zero_boundary_packets_3d!(coarse_F, patch, topology)
    for idx in topology.boundary_links
        link = topology.links[idx]
        cell = topology.cells[link.src]
        if cell.level == 0
            coarse_F[cell.i, cell.j, cell.k, link.q] = 0
        else
            i0 = 2 * first(patch.parent_i_range) - 1
            j0 = 2 * first(patch.parent_j_range) - 1
            k0 = 2 * first(patch.parent_k_range) - 1
            patch.fine_F[cell.i - i0 + 1,
                         cell.j - j0 + 1,
                         cell.k - k0 + 1,
                         link.q] = 0
        end
    end
    return coarse_F, patch
end

@testset "Conservative tree route streaming 3D" begin
    Nx, Ny, Nz = 7, 8, 6
    patch_template = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
    topology = create_conservative_tree_topology_3d(Nx, Ny, Nz, patch_template)

    @testset "direct same-level route moves one packet" begin
        patch_in = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        patch_out = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        coarse_in = zeros(Float64, Nx, Ny, Nz, 19)
        coarse_out = similar(coarse_in)
        coarse_in[1, 4, 2, 2] = 3.25

        stream_composite_routes_interior_F_3d!(
            coarse_out, patch_out, coarse_in, patch_in, topology)

        @test coarse_out[2, 4, 2, 2] == 3.25
        @test isapprox(active_mass_F_3d(coarse_out, patch_out), 3.25; atol=1e-14, rtol=0)
    end

    @testset "coarse to fine face route splits one packet" begin
        patch_in = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        patch_out = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        coarse_in = zeros(Float64, Nx, Ny, Nz, 19)
        coarse_out = similar(coarse_in)
        coarse_in[2, 4, 2, 2] = 4.0

        stream_composite_routes_interior_F_3d!(
            coarse_out, patch_out, coarse_in, patch_in, topology)

        @test patch_out.fine_F[1, 1, 1, 2] == 1.0
        @test patch_out.fine_F[1, 2, 1, 2] == 1.0
        @test patch_out.fine_F[1, 1, 2, 2] == 1.0
        @test patch_out.fine_F[1, 2, 2, 2] == 1.0
        @test patch_out.coarse_shadow_F[1, 1, 1, 2] == 4.0
        @test isapprox(active_mass_F_3d(coarse_out, patch_out), 4.0; atol=1e-14, rtol=0)
    end

    @testset "coarse to fine edge route splits one packet" begin
        patch_in = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        patch_out = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        coarse_in = zeros(Float64, Nx, Ny, Nz, 19)
        coarse_out = similar(coarse_in)
        coarse_in[2, 3, 2, 8] = 5.0

        stream_composite_routes_interior_F_3d!(
            coarse_out, patch_out, coarse_in, patch_in, topology)

        @test patch_out.fine_F[1, 1, 1, 8] == 2.5
        @test patch_out.fine_F[1, 1, 2, 8] == 2.5
        @test patch_out.coarse_shadow_F[1, 1, 1, 8] == 5.0
        @test isapprox(active_mass_F_3d(coarse_out, patch_out), 5.0; atol=1e-14, rtol=0)
    end

    @testset "fine to coarse face routes coalesce by accumulation" begin
        patch_in = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        patch_out = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        coarse_in = zeros(Float64, Nx, Ny, Nz, 19)
        coarse_out = similar(coarse_in)
        patch_in.fine_F[1, 1, 1, 3] = 1.25
        patch_in.fine_F[1, 2, 1, 3] = 2.0
        patch_in.fine_F[1, 1, 2, 3] = 0.5
        patch_in.fine_F[1, 2, 2, 3] = 0.25

        stream_composite_routes_interior_F_3d!(
            coarse_out, patch_out, coarse_in, patch_in, topology)

        @test coarse_out[2, 4, 2, 3] == 4.0
        @test isapprox(active_mass_F_3d(coarse_out, patch_out), 4.0; atol=1e-14, rtol=0)
    end

    @testset "fine to coarse edge routes coalesce by accumulation" begin
        patch_in = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        patch_out = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        coarse_in = zeros(Float64, Nx, Ny, Nz, 19)
        coarse_out = similar(coarse_in)
        patch_in.fine_F[1, 1, 1, 11] = 3.0
        patch_in.fine_F[1, 1, 2, 11] = 4.0

        stream_composite_routes_interior_F_3d!(
            coarse_out, patch_out, coarse_in, patch_in, topology)

        @test coarse_out[2, 3, 2, 11] == 7.0
        @test isapprox(active_mass_F_3d(coarse_out, patch_out), 7.0; atol=1e-14, rtol=0)
    end

    @testset "all non-boundary routes conserve active population sums" begin
        patch_in = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        patch_out = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        coarse_in = zeros(Float64, Nx, Ny, Nz, 19)
        coarse_out = similar(coarse_in)

        for q in 1:19, k in axes(coarse_in, 3), j in axes(coarse_in, 2), i in axes(coarse_in, 1)
            if !(i in patch_in.parent_i_range &&
                 j in patch_in.parent_j_range &&
                 k in patch_in.parent_k_range)
                coarse_in[i, j, k, q] = q + i / 32 + j / 64 + k / 128
            end
        end
        for q in 1:19, k in axes(patch_in.fine_F, 3), j in axes(patch_in.fine_F, 2), i in axes(patch_in.fine_F, 1)
            patch_in.fine_F[i, j, k, q] = q / 2 + i / 128 + j / 256 + k / 512
        end
        _stream_zero_boundary_packets_3d!(coarse_in, patch_in, topology)
        pop0 = active_population_sums_F_3d(coarse_in, patch_in)
        moments0 = collect(active_moments_F_3d(coarse_in, patch_in))

        stream_composite_routes_interior_F_3d!(
            coarse_out, patch_out, coarse_in, patch_in, topology)

        @test isapprox(active_population_sums_F_3d(coarse_out, patch_out), pop0;
                       atol=1e-11, rtol=0)
        @test isapprox(collect(active_moments_F_3d(coarse_out, patch_out)), moments0;
                       atol=1e-11, rtol=0)
    end

    @testset "periodic x wraps coarse boundary packets" begin
        patch_in = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        patch_out = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        coarse_in = zeros(Float64, Nx, Ny, Nz, 19)
        coarse_out = similar(coarse_in)
        coarse_in[1, 4, 2, 3] = 2.5
        coarse_in[Nx, 4, 2, 2] = 3.5

        stream_composite_routes_periodic_x_F_3d!(
            coarse_out, patch_out, coarse_in, patch_in, topology)

        @test coarse_out[Nx, 4, 2, 3] == 2.5
        @test coarse_out[1, 4, 2, 2] == 3.5
        @test isapprox(active_mass_F_3d(coarse_out, patch_out), 6.0; atol=1e-14, rtol=0)
    end

    @testset "periodic x wraps coarse boundary packets into fine patch" begin
        patch_in = create_conservative_tree_patch_3d(1:2, 3:4, 2:3)
        patch_out = create_conservative_tree_patch_3d(1:2, 3:4, 2:3)
        topology_x = create_conservative_tree_topology_3d(Nx, Ny, Nz, patch_in)
        coarse_in = zeros(Float64, Nx, Ny, Nz, 19)
        coarse_out = similar(coarse_in)
        coarse_in[Nx, 3, 2, 2] = 4.0

        stream_composite_routes_periodic_x_F_3d!(
            coarse_out, patch_out, coarse_in, patch_in, topology_x)

        @test patch_out.fine_F[1, 1, 1, 2] == 1.0
        @test patch_out.fine_F[1, 2, 1, 2] == 1.0
        @test patch_out.fine_F[1, 1, 2, 2] == 1.0
        @test patch_out.fine_F[1, 2, 2, 2] == 1.0
        @test isapprox(active_mass_F_3d(coarse_out, patch_out), 4.0; atol=1e-14, rtol=0)
    end

    @testset "periodic x wraps fine boundary packets to coarse cells" begin
        patch_in = create_conservative_tree_patch_3d(1:2, 3:4, 2:3)
        patch_out = create_conservative_tree_patch_3d(1:2, 3:4, 2:3)
        topology_x = create_conservative_tree_topology_3d(Nx, Ny, Nz, patch_in)
        coarse_in = zeros(Float64, Nx, Ny, Nz, 19)
        coarse_out = similar(coarse_in)
        patch_in.fine_F[1, 1, 1, 3] = 2.75

        stream_composite_routes_periodic_x_F_3d!(
            coarse_out, patch_out, coarse_in, patch_in, topology_x)

        @test coarse_out[Nx, 3, 2, 3] == 2.75
        @test isapprox(active_mass_F_3d(coarse_out, patch_out), 2.75; atol=1e-14, rtol=0)
    end

    @testset "periodic x wall yz bounces coarse wall packets" begin
        patch_in = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        patch_out = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        coarse_in = zeros(Float64, Nx, Ny, Nz, 19)
        coarse_out = similar(coarse_in)
        coarse_in[3, 1, 2, 5] = 2.0

        stream_composite_routes_periodic_x_wall_yz_F_3d!(
            coarse_out, patch_out, coarse_in, patch_in, topology)

        @test coarse_out[3, 1, 2, 4] == 2.0
        @test isapprox(active_mass_F_3d(coarse_out, patch_out), 2.0; atol=1e-14, rtol=0)
    end

    @testset "periodic x wall yz bounces fine wall packets" begin
        patch_in = create_conservative_tree_patch_3d(3:4, 4:5, 1:2)
        patch_out = create_conservative_tree_patch_3d(3:4, 4:5, 1:2)
        topology_z = create_conservative_tree_topology_3d(Nx, Ny, Nz, patch_in)
        coarse_in = zeros(Float64, Nx, Ny, Nz, 19)
        coarse_out = similar(coarse_in)
        patch_in.fine_F[1, 1, 1, 7] = 3.0

        stream_composite_routes_periodic_x_wall_yz_F_3d!(
            coarse_out, patch_out, coarse_in, patch_in, topology_z)

        @test patch_out.fine_F[1, 1, 1, 6] == 3.0
        @test isapprox(active_mass_F_3d(coarse_out, patch_out), 3.0; atol=1e-14, rtol=0)
    end

    @testset "periodic x wall yz keeps x wrapping and prioritizes walls at corners" begin
        patch_in = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        patch_out = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        coarse_in = zeros(Float64, Nx, Ny, Nz, 19)
        coarse_out = similar(coarse_in)
        coarse_in[1, 4, 2, 3] = 1.25
        coarse_in[1, 1, 2, 11] = 4.5

        stream_composite_routes_periodic_x_wall_yz_F_3d!(
            coarse_out, patch_out, coarse_in, patch_in, topology)

        @test coarse_out[Nx, 4, 2, 3] == 1.25
        @test coarse_out[1, 1, 2, 8] == 4.5
        @test isapprox(active_mass_F_3d(coarse_out, patch_out), 5.75; atol=1e-14, rtol=0)
    end

    @testset "minimal 3D fixed-patch transport plus BGK collision conserves mass" begin
        patch = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        patch_next = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        coarse = zeros(Float64, Nx, Ny, Nz, 19)
        coarse_next = similar(coarse)

        for k in axes(coarse, 3), j in axes(coarse, 2), i in axes(coarse, 1)
            if !(i in patch.parent_i_range &&
                 j in patch.parent_j_range &&
                 k in patch.parent_k_range)
                fill_equilibrium_integrated_D3Q19!(
                    @view(coarse[i, j, k, :]), 1.0, 1.0, 0.0, 0.0, 0.0)
            end
        end
        fill_equilibrium_integrated_D3Q19!(patch.fine_F, 0.125, 1.0, 0.0, 0.0, 0.0)
        coalesce_patch_to_shadow_F_3d!(patch)
        mass0 = active_mass_F_3d(coarse, patch)

        for _ in 1:8
            collide_BGK_composite_F_3d!(coarse, patch, 1.0, 0.125, 1.0, 1.0)
            stream_composite_routes_periodic_x_wall_yz_F_3d!(
                coarse_next, patch_next, coarse, patch, topology)
            coarse, coarse_next = coarse_next, coarse
            patch, patch_next = patch_next, patch
        end

        @test isapprox(active_mass_F_3d(coarse, patch), mass0; atol=1e-10, rtol=0)
    end

    @testset "periodic x wall yz solid cube conserves 3D mass" begin
        patch = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        patch_next = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        topology_solid = create_conservative_tree_topology_3d(Nx, Ny, Nz, patch)
        coarse = zeros(Float64, Nx, Ny, Nz, 19)
        coarse_next = similar(coarse)
        is_solid = falses(2 * Nx, 2 * Ny, 2 * Nz)
        is_solid[6:7, 8:9, 4:5] .= true

        for k in axes(coarse, 3), j in axes(coarse, 2), i in axes(coarse, 1)
            if !(i in patch.parent_i_range &&
                 j in patch.parent_j_range &&
                 k in patch.parent_k_range)
                fill_equilibrium_integrated_D3Q19!(
                    @view(coarse[i, j, k, :]), 1.0, 1.0, 0.0, 0.0, 0.0)
            end
        end
        fill_equilibrium_integrated_D3Q19!(patch.fine_F, 0.125, 1.0, 0.0, 0.0, 0.0)
        patch.fine_F[2:3, 2:3, 2:3, :] .= 0
        coalesce_patch_to_shadow_F_3d!(patch)
        mass0 = active_mass_F_3d(coarse, patch)

        for _ in 1:12
            KrakeNN.collide_Guo_composite_solid_F_3d!(
                coarse, patch, topology_solid, is_solid,
                1.0, 0.125, 1.0, 1.0, 0.0, 0.0, 0.0)
            KrakeNN.stream_composite_routes_periodic_x_wall_yz_solid_F_3d!(
                coarse_next, patch_next, coarse, patch, topology_solid, is_solid)
            coarse, coarse_next = coarse_next, coarse
            patch, patch_next = patch_next, patch
        end

        @test isapprox(active_mass_F_3d(coarse, patch), mass0; atol=1e-10, rtol=0)
        @test all(isfinite, active_momentum_F_3d(coarse, patch))
    end

    @testset "route-native 3D fixed-patch forced channel accelerates conservatively" begin
        result = KrakeNN.run_conservative_tree_poiseuille_route_native_3d(; steps=80)

        @test result.flow == :poiseuille_route_native_3d
        @test isfinite(result.ux_mean)
        @test result.ux_mean > 0
        @test abs(result.uy_mean) < 1e-10
        @test abs(result.uz_mean) < 1e-10
        @test isapprox(result.mass_final, result.mass_initial; atol=1e-10, rtol=0)
        @test result.relative_mass_drift < 1e-12
        @test size(result.ux_profile) == (16, 12)
        @test size(result.oracle_ux_profile) == size(result.ux_profile)
        @test isfinite(result.l2_error)
        @test isfinite(result.linf_error)
        @test result.linf_error < 1e-3
    end

    @testset "route-native 3D full-domain patch matches dense leaf oracle" begin
        result = KrakeNN.run_conservative_tree_poiseuille_route_native_3d(;
            Nx=4, Ny=4, Nz=3,
            patch_strategy=:full,
            steps=20)

        @test result.l2_error < 1e-14
        @test result.linf_error < 1e-14
        @test result.relative_mass_drift < 1e-12
    end

    @testset "route-native 3D cross-section buffered patch reduces dense-oracle gap" begin
        result = KrakeNN.run_conservative_tree_poiseuille_route_native_3d(;
            steps=80, patch_strategy=:cross_section_buffered)
        relative_linf = result.linf_error / maximum(abs.(result.oracle_ux_profile))

        @test relative_linf < 0.30
        @test result.relative_mass_drift < 1e-12
    end

    @testset "route-native 3D fixed-patch forced channel is Float32-clean" begin
        result = KrakeNN.run_conservative_tree_poiseuille_route_native_3d(;
            steps=4, T=Float32)

        @test result.ux_mean > 0f0
        @test isfinite(result.relative_mass_drift)
    end

    @testset "layout checks catch mismatched patch" begin
        patch_in = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        wrong_patch = create_conservative_tree_patch_3d(4:5, 4:5, 2:3)
        coarse_in = zeros(Float64, Nx, Ny, Nz, 19)
        coarse_out = similar(coarse_in)
        @test_throws ArgumentError stream_composite_routes_interior_F_3d!(
            coarse_out, wrong_patch, coarse_in, patch_in, topology)
    end
end
