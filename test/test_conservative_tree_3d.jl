using Test
using KrakeNN

@inline function _test_face_normal_3d(face)
    if face == :west
        return (-1, 0, 0)
    elseif face == :east
        return (1, 0, 0)
    elseif face == :south
        return (0, -1, 0)
    elseif face == :north
        return (0, 1, 0)
    elseif face == :bottom
        return (0, 0, -1)
    elseif face == :top
        return (0, 0, 1)
    else
        error("bad face")
    end
end

@inline function _test_face_children_3d(face)
    if face == :west
        return ((1, 1, 1), (1, 2, 1), (1, 1, 2), (1, 2, 2))
    elseif face == :east
        return ((2, 1, 1), (2, 2, 1), (2, 1, 2), (2, 2, 2))
    elseif face == :south
        return ((1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2))
    elseif face == :north
        return ((1, 2, 1), (2, 2, 1), (1, 2, 2), (2, 2, 2))
    elseif face == :bottom
        return ((1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1))
    else
        return ((1, 1, 2), (2, 1, 2), (1, 2, 2), (2, 2, 2))
    end
end

@inline function _test_enters_face_3d(q, face)
    nx, ny, nz = _test_face_normal_3d(face)
    return d3q19_cx(q) * nx + d3q19_cy(q) * ny + d3q19_cz(q) * nz < 0
end

@inline function _test_leaves_face_3d(q, face)
    nx, ny, nz = _test_face_normal_3d(face)
    return d3q19_cx(q) * nx + d3q19_cy(q) * ny + d3q19_cz(q) * nz > 0
end

@inline function _test_edge_children_3d(edge)
    if edge == :southwest
        return ((1, 1, 1), (1, 1, 2))
    elseif edge == :southeast
        return ((2, 1, 1), (2, 1, 2))
    elseif edge == :northwest
        return ((1, 2, 1), (1, 2, 2))
    elseif edge == :northeast
        return ((2, 2, 1), (2, 2, 2))
    elseif edge == :bottomwest
        return ((1, 1, 1), (1, 2, 1))
    elseif edge == :bottomeast
        return ((2, 1, 1), (2, 2, 1))
    elseif edge == :topwest
        return ((1, 1, 2), (1, 2, 2))
    elseif edge == :topeast
        return ((2, 1, 2), (2, 2, 2))
    elseif edge == :bottomsouth
        return ((1, 1, 1), (2, 1, 1))
    elseif edge == :bottomnorth
        return ((1, 2, 1), (2, 2, 1))
    elseif edge == :topsouth
        return ((1, 1, 2), (2, 1, 2))
    else
        return ((1, 2, 2), (2, 2, 2))
    end
end

@testset "Conservative tree primitives 3D" begin
    @testset "D3Q19 integer accessors" begin
        cxs = velocities_x(D3Q19())
        cys = velocities_y(D3Q19())
        czs = velocities_z(D3Q19())
        opp = opposite(D3Q19())

        for q in 1:19
            @test d3q19_cx(q) == cxs[q]
            @test d3q19_cy(q) == cys[q]
            @test d3q19_cz(q) == czs[q]
            @test d3q19_opposite(q) == opp[q]
            @test d3q19_opposite(d3q19_opposite(q)) == q
        end
        @test_throws ArgumentError d3q19_cx(0)
        @test_throws ArgumentError d3q19_cy(20)
        @test_throws ArgumentError d3q19_cz(20)
        @test_throws ArgumentError d3q19_opposite(20)
    end

    @testset "coalesce preserves per-population sums" begin
        Fc = reshape(collect(1.0:152.0), 2, 2, 2, 19)
        Fp = zeros(Float64, 19)

        coalesce_F_3d!(Fp, Fc)

        for q in 1:19
            @test Fp[q] == sum(Fc[:, :, :, q])
        end
    end

    @testset "coalesce preserves mass and momentum" begin
        Fc = zeros(Float64, 2, 2, 2, 19)
        for q in 1:19, iz in 1:2, iy in 1:2, ix in 1:2
            Fc[ix, iy, iz, q] = 0.001 * (100q + 10ix + 3iy + iz)
        end

        Fp = zeros(Float64, 19)
        coalesce_F_3d!(Fp, Fc)

        child_mass = sum(Fc)
        child_mx = sum(d3q19_cx(q) * Fc[ix, iy, iz, q]
                       for ix in 1:2, iy in 1:2, iz in 1:2, q in 1:19)
        child_my = sum(d3q19_cy(q) * Fc[ix, iy, iz, q]
                       for ix in 1:2, iy in 1:2, iz in 1:2, q in 1:19)
        child_mz = sum(d3q19_cz(q) * Fc[ix, iy, iz, q]
                       for ix in 1:2, iy in 1:2, iz in 1:2, q in 1:19)

        m, mx, my, mz = moments_F_3d(Fp)
        @test isapprox(m, child_mass; atol=1e-14, rtol=0)
        @test isapprox(mx, child_mx; atol=1e-14, rtol=0)
        @test isapprox(my, child_my; atol=1e-14, rtol=0)
        @test isapprox(mz, child_mz; atol=1e-14, rtol=0)
    end

    @testset "uniform explosion conserves parent" begin
        Fp = [0.5 + 0.01q for q in 1:19]
        Fc = zeros(Float64, 2, 2, 2, 19)

        explode_uniform_F_3d!(Fc, Fp)

        Fback = zeros(Float64, 19)
        coalesce_F_3d!(Fback, Fc)

        @test isapprox(Fback, Fp; atol=1e-14, rtol=0)
        @test isapprox(collect(moments_F_3d(Fback)),
                       collect(moments_F_3d(Fp)); atol=1e-14, rtol=0)
    end

    @testset "equilibrium integrated volume moments" begin
        Fcell = zeros(Float64, 19)
        volume = 0.125
        rho = 1.03
        ux, uy, uz = 0.02, -0.015, 0.01

        fill_equilibrium_integrated_D3Q19!(Fcell, volume, rho, ux, uy, uz)

        m, mx, my, mz = moments_F_3d(Fcell)
        @test isapprox(m, volume * rho; atol=1e-14, rtol=0)
        @test isapprox(mx, volume * rho * ux; atol=1e-14, rtol=0)
        @test isapprox(my, volume * rho * uy; atol=1e-14, rtol=0)
        @test isapprox(mz, volume * rho * uz; atol=1e-14, rtol=0)
    end

    @testset "BGK integrated collision conserves D3Q19 moments" begin
        F = zeros(Float64, 19)
        volume = 0.125
        fill_equilibrium_integrated_D3Q19!(F, volume, 1.0, 0.02, -0.01, 0.015)
        F[2] += 1e-4
        F[3] += 1e-4
        before = collect(moments_F_3d(F))

        collide_BGK_integrated_D3Q19!(F, volume, 1.1)

        @test isapprox(collect(moments_F_3d(F)), before; atol=1e-14, rtol=0)
    end

    @testset "BGK omega one projects to D3Q19 equilibrium at conserved moments" begin
        F = [0.02 + q / 2000 for q in 1:19]
        volume = 1.0
        before = moments_F_3d(F)

        collide_BGK_integrated_D3Q19!(F, volume, 1.0)

        m, mx, my, mz = before
        rho = m / volume
        ux = mx / m
        uy = my / m
        uz = mz / m
        expected = [volume * equilibrium(D3Q19(), rho, ux, uy, uz, q) for q in 1:19]
        @test isapprox(F, expected; atol=1e-14, rtol=0)
    end

    @testset "Guo integrated collision conserves mass and drives momentum" begin
        F = zeros(Float64, 19)
        fill_equilibrium_integrated_D3Q19!(F, 1.0, 1.0, 0.0, 0.0, 0.0)
        mass0 = mass_F_3d(F)

        collide_Guo_integrated_D3Q19!(F, 1.0, 1.0, 1e-4, 0.0, -5e-5)

        @test isapprox(mass_F_3d(F), mass0; atol=1e-14, rtol=0)
        @test momentum_F_3d(F)[1] > 0
        @test abs(momentum_F_3d(F)[2]) < 1e-14
        @test momentum_F_3d(F)[3] < 0
    end

    @testset "grid BGK collision conserves global D3Q19 moments" begin
        F = zeros(Float64, 3, 2, 2, 19)
        for q in 1:19, k in axes(F, 3), j in axes(F, 2), i in axes(F, 1)
            F[i, j, k, q] = 0.05 + q / 512 + i / 1024 + j / 2048 + k / 4096
        end
        before = collect(moments_F_3d(F))

        collide_BGK_integrated_D3Q19!(F, 0.125, 1.2)

        @test isapprox(collect(moments_F_3d(F)), before; atol=1e-12, rtol=0)
    end

    @testset "parent-child mapping" begin
        @test conservative_tree_parent_index_3d(1, 1, 1) == (1, 1, 1, 1, 1, 1)
        @test conservative_tree_parent_index_3d(2, 1, 1) == (1, 1, 1, 2, 1, 1)
        @test conservative_tree_parent_index_3d(1, 2, 1) == (1, 1, 1, 1, 2, 1)
        @test conservative_tree_parent_index_3d(1, 1, 2) == (1, 1, 1, 1, 1, 2)
        @test conservative_tree_parent_index_3d(3, 4, 5) == (2, 2, 3, 1, 2, 1)
        @test_throws ArgumentError conservative_tree_parent_index_3d(0, 1, 1)
        @test_throws ArgumentError conservative_tree_parent_index_3d(1, 0, 1)
        @test_throws ArgumentError conservative_tree_parent_index_3d(1, 1, 0)
    end

    @testset "face interface split and coalesce cover all oriented populations" begin
        faces = (:west, :east, :south, :north, :bottom, :top)

        for face in faces
            for q in 1:19
                Fc = zeros(Float64, 2, 2, 2, 19)
                packet = 10.0 + q / 10

                if _test_enters_face_3d(q, face)
                    split_coarse_to_fine_face_F_3d!(Fc, packet, q, face)
                    @test isapprox(sum(Fc[:, :, :, q]), packet; atol=1e-14, rtol=0)
                    @test isapprox(mass_F_3d(Fc), packet; atol=1e-14, rtol=0)
                    @test isapprox(momentum_F_3d(Fc)[1], d3q19_cx(q) * packet; atol=1e-14, rtol=0)
                    @test isapprox(momentum_F_3d(Fc)[2], d3q19_cy(q) * packet; atol=1e-14, rtol=0)
                    @test isapprox(momentum_F_3d(Fc)[3], d3q19_cz(q) * packet; atol=1e-14, rtol=0)
                    @test count(!iszero, Fc[:, :, :, q]) == 4
                else
                    @test_throws ArgumentError split_coarse_to_fine_face_F_3d!(
                        Fc, packet, q, face)
                end

                Fc .= 0
                for iz in 1:2, iy in 1:2, ix in 1:2
                    Fc[ix, iy, iz, q] = 100.0q + 10ix + 2iy + iz
                end

                if _test_leaves_face_3d(q, face)
                    expected = sum(Fc[ix, iy, iz, q]
                                   for (ix, iy, iz) in _test_face_children_3d(face))
                    @test coalesce_fine_to_coarse_face_F_3d(Fc, q, face) == expected
                else
                    @test_throws ArgumentError coalesce_fine_to_coarse_face_F_3d(
                        Fc, q, face)
                end
            end
        end
        @test_throws ArgumentError split_coarse_to_fine_face_F_3d!(
            zeros(Float64, 2, 2, 2, 19), 1.0, 2, :badface)
    end

    @testset "edge interface split and coalesce cover D3Q19 edge populations" begin
        entering = Dict(
            :southwest => 8,
            :southeast => 9,
            :northwest => 10,
            :northeast => 11,
            :bottomwest => 12,
            :bottomeast => 13,
            :topwest => 14,
            :topeast => 15,
            :bottomsouth => 16,
            :bottomnorth => 17,
            :topsouth => 18,
            :topnorth => 19,
        )
        leaving = Dict(
            :northeast => 8,
            :northwest => 9,
            :southeast => 10,
            :southwest => 11,
            :topeast => 12,
            :topwest => 13,
            :bottomeast => 14,
            :bottomwest => 15,
            :topnorth => 16,
            :topsouth => 17,
            :bottomnorth => 18,
            :bottomsouth => 19,
        )

        for edge in keys(entering)
            q = entering[edge]
            packet = 4.0 + q / 10
            Fc = zeros(Float64, 2, 2, 2, 19)
            split_coarse_to_fine_edge_F_3d!(Fc, packet, q, edge)

            @test isapprox(sum(Fc[:, :, :, q]), packet; atol=1e-14, rtol=0)
            @test isapprox(mass_F_3d(Fc), packet; atol=1e-14, rtol=0)
            @test isapprox(momentum_F_3d(Fc)[1], d3q19_cx(q) * packet; atol=1e-14, rtol=0)
            @test isapprox(momentum_F_3d(Fc)[2], d3q19_cy(q) * packet; atol=1e-14, rtol=0)
            @test isapprox(momentum_F_3d(Fc)[3], d3q19_cz(q) * packet; atol=1e-14, rtol=0)
            @test count(!iszero, Fc[:, :, :, q]) == 2

            for q_bad in setdiff(1:19, (q,))
                @test_throws ArgumentError split_coarse_to_fine_edge_F_3d!(
                    zeros(Float64, 2, 2, 2, 19), 1.0, q_bad, edge)
            end

            q_leave = leaving[edge]
            Fc .= 0
            for iz in 1:2, iy in 1:2, ix in 1:2
                Fc[ix, iy, iz, q_leave] = 10.0q_leave + ix + 2iy + 4iz
            end
            expected = sum(Fc[ix, iy, iz, q_leave]
                           for (ix, iy, iz) in _test_edge_children_3d(edge))
            @test coalesce_fine_to_coarse_edge_F_3d(Fc, q_leave, edge) == expected

            for q_bad in setdiff(1:19, (q_leave,))
                @test_throws ArgumentError coalesce_fine_to_coarse_edge_F_3d(
                    Fc, q_bad, edge)
            end
        end
        @test_throws ArgumentError split_coarse_to_fine_edge_F_3d!(
            zeros(Float64, 2, 2, 2, 19), 1.0, 8, :badedge)
    end

    @testset "D3Q19 corner route set is empty" begin
        corners = (:bottomsouthwest, :bottomsoutheast, :bottomnorthwest, :bottomnortheast,
                   :topsouthwest, :topsoutheast, :topnorthwest, :topnortheast)
        for corner in corners, q in 1:19
            @test_throws ArgumentError split_coarse_to_fine_corner_F_3d!(
                zeros(Float64, 2, 2, 2, 19), 1.0, q, corner)
            @test_throws ArgumentError coalesce_fine_to_coarse_corner_F_3d(
                zeros(Float64, 2, 2, 2, 19), q, corner)
        end
        @test_throws ArgumentError split_coarse_to_fine_corner_F_3d!(
            zeros(Float64, 2, 2, 2, 19), 1.0, 8, :badcorner)
    end
end
