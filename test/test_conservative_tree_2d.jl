using Test
using KrakeNN

# Conservation is checked on sums over the whole grid, so the achievable
# agreement scales with the sum's magnitude: a pure absolute tolerance asks for
# more than the arithmetic can give. The Poiseuille canary sums to ~313 and
# drifted 30 ulps (5.4e-15 relative), tripping atol=1e-12 — which is ~18 ulps
# there. Keep the absolute floor for near-zero sums and add a relative band;
# a genuine conservation break is orders of magnitude larger than this.
function _assert_per_q_sums_equal(A, B; atol=1e-14, rtol=1e-13)
    for q in 1:9
        @test isapprox(sum(A[:, :, q]), sum(B[:, :, q]); atol=atol, rtol=rtol)
    end
end

function _mean_ux_by_y(F; volume=1.0, force_x=0.0)
    profile = zeros(Float64, size(F, 2))
    for j in axes(F, 2)
        ux_sum = 0.0
        for i in axes(F, 1)
            cell = @view F[i, j, :]
            ux_sum += (momentum_F(cell)[1] / volume + force_x / 2) / (mass_F(cell) / volume)
        end
        profile[j] = ux_sum / size(F, 1)
    end
    return profile
end

function _composite_mean_ux_by_leaf_y(coarse_F, patch; volume_leaf=0.25, force_x=0.0)
    leaf = zeros(Float64, 2 * size(coarse_F, 1), 2 * size(coarse_F, 2), 9)
    composite_to_leaf_F_2d!(leaf, coarse_F, patch)
    return _mean_ux_by_y(leaf; volume=volume_leaf, force_x=force_x)
end

function _run_cartesian_leaf_couette(nx, ny; U=0.04, steps=3000, omega=1.0)
    F = zeros(Float64, nx, ny, 9)
    Fnext = similar(F)
    fill_equilibrium_integrated_D2Q9!(F, 0.25, 1.0, 0.0, 0.0)
    mass0 = mass_F(F)

    for _ in 1:steps
        collide_BGK_integrated_D2Q9!(F, 0.25, omega)
        stream_periodic_x_moving_wall_y_F_2d!(Fnext, F; u_north=U, volume=0.25)
        F, Fnext = Fnext, F
    end

    return (F=F, ux_profile=_mean_ux_by_y(F; volume=0.25), mass_initial=mass0)
end

function _run_cartesian_leaf_poiseuille(nx, ny; Fx=5e-5, steps=5000, omega=1.0)
    F = zeros(Float64, nx, ny, 9)
    Fnext = similar(F)
    fill_equilibrium_integrated_D2Q9!(F, 0.25, 1.0, 0.0, 0.0)
    mass0 = mass_F(F)

    for _ in 1:steps
        collide_Guo_integrated_D2Q9!(F, 0.25, omega, Fx, 0.0)
        stream_periodic_x_wall_y_F_2d!(Fnext, F)
        F, Fnext = Fnext, F
    end

    return (F=F, ux_profile=_mean_ux_by_y(F; volume=0.25, force_x=Fx),
            mass_initial=mass0)
end

@testset "Poiseuille analytic profile uses halfway walls" begin
    ny = 8
    Fx = 1e-6
    omega = 1.0
    ν = (1 / omega - 0.5) / 3
    profile = poiseuille_analytic_profile_2d(ny, Fx, omega)

    @test profile[1] ≈ Fx / (2ν) * 0.5 * (ny - 0.5)
    @test profile[end] ≈ profile[1]
    @test profile[2] > profile[1]
    @test profile ≈ reverse(profile)
end

function _fluid_mass_F(F, is_solid)
    total = 0.0
    for q in 1:9, j in axes(F, 2), i in axes(F, 1)
        is_solid[i, j] && continue
        total += F[i, j, q]
    end
    return total
end

function _fluid_mean_velocity_F(F, is_solid; volume=0.25, force_x=0.0, force_y=0.0)
    ux_sum = 0.0
    uy_sum = 0.0
    n_fluid = 0
    for j in axes(F, 2), i in axes(F, 1)
        is_solid[i, j] && continue
        cell = @view F[i, j, :]
        rho = mass_F(cell) / volume
        p = momentum_F(cell)
        ux_sum += (p[1] / volume + force_x / 2) / rho
        uy_sum += (p[2] / volume + force_y / 2) / rho
        n_fluid += 1
    end
    return ux_sum / n_fluid, uy_sum / n_fluid
end

function _raw_second_moments_F(Fcell)
    xx = 0.0
    xy = 0.0
    yy = 0.0
    for q in 1:9
        cx = d2q9_cx(q)
        cy = d2q9_cy(q)
        xx += cx * cx * Fcell[q]
        xy += cx * cy * Fcell[q]
        yy += cy * cy * Fcell[q]
    end
    return xx, xy, yy
end

function _aggregate_raw_second_moments_F(F)
    xx = 0.0
    xy = 0.0
    yy = 0.0
    for j in axes(F, 2), i in axes(F, 1)
        s = _raw_second_moments_F(@view F[i, j, :])
        xx += s[1]
        xy += s[2]
        yy += s[3]
    end
    return xx, xy, yy
end

function _noneq_second_moments_F(Fcell; volume=0.25)
    raw = _raw_second_moments_F(Fcell)
    m = mass_F(Fcell)
    p = momentum_F(Fcell)
    rho = m / volume
    ux = p[1] / m
    uy = p[2] / m
    return (raw[1] - volume * rho * (1 / 3 + ux^2),
            raw[2] - volume * rho * ux * uy,
            raw[3] - volume * rho * (1 / 3 + uy^2))
end

function _fill_equilibrium_with_pxy!(Fcell, volume, rho, ux, uy, pxy)
    fill_equilibrium_integrated_D2Q9!(Fcell, volume, rho, ux, uy)
    delta = pxy / 4
    Fcell[6] += delta
    Fcell[7] -= delta
    Fcell[8] += delta
    Fcell[9] -= delta
    minimum(Fcell) > 0 || error("test perturbation made a negative population")
    return Fcell
end

function _run_cartesian_leaf_square_obstacle(is_solid; Fx=2e-5, Fy=0.0,
                                             steps=1200, omega=1.0)
    F = zeros(Float64, size(is_solid, 1), size(is_solid, 2), 9)
    Fnext = similar(F)
    fill_equilibrium_integrated_D2Q9!(F, 0.25, 1.0, 0.0, 0.0)
    mass0 = _fluid_mass_F(F, is_solid)

    for _ in 1:steps
        collide_Guo_integrated_D2Q9!(F, is_solid, 0.25, omega, Fx, Fy)
        stream_periodic_x_wall_y_solid_F_2d!(Fnext, F, is_solid)
        F, Fnext = Fnext, F
    end

    ux, uy = _fluid_mean_velocity_F(F, is_solid; force_x=Fx, force_y=Fy)
    return (F=F, ux_mean=ux, uy_mean=uy, mass_initial=mass0,
            mass_final=_fluid_mass_F(F, is_solid))
end

function _run_conservative_tree_square_obstacle_drag(; Nx=24, Ny=14,
                                                     patch_i_range=9:16,
                                                     patch_j_range=4:11,
                                                     obstacle_i_range=21:28,
                                                     obstacle_j_range=11:18,
                                                     Fx=2e-5, Fy=0.0,
                                                     steps=1200, avg_window=300,
                                                     omega=1.0)
    coarse = zeros(Float64, Nx, Ny, 9)
    coarse_next = similar(coarse)
    patch = create_conservative_tree_patch_2d(patch_i_range, patch_j_range)
    patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range)
    leaf = zeros(Float64, 2 * Nx, 2 * Ny, 9)
    leaf_next = similar(leaf)
    is_solid = square_solid_mask_leaf_2d(2 * Nx, 2 * Ny,
                                         obstacle_i_range, obstacle_j_range)

    fill_equilibrium_integrated_D2Q9!(coarse, 1.0, 1.0, 0.0, 0.0)
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, 0.25, 1.0, 0.0, 0.0)
    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass0 = _fluid_mass_F(leaf, is_solid)
    Fx_sum = 0.0
    Fy_sum = 0.0
    n_avg = 0

    for step in 1:steps
        composite_to_leaf_F_2d!(leaf, coarse, patch)
        collide_Guo_integrated_D2Q9!(leaf, is_solid, 0.25, omega, Fx, Fy)
        stream_periodic_x_wall_y_solid_F_2d!(leaf_next, leaf, is_solid)
        if step > steps - avg_window
            drag = compute_drag_mea_solid_F_2d(leaf, leaf_next, is_solid)
            Fx_sum += drag.Fx
            Fy_sum += drag.Fy
            n_avg += 1
        end
        leaf_to_composite_F_2d!(coarse_next, patch_next, leaf_next)
        coarse, coarse_next = coarse_next, coarse
        patch, patch_next = patch_next, patch
    end

    composite_to_leaf_F_2d!(leaf, coarse, patch)
    ux, uy = _fluid_mean_velocity_F(leaf, is_solid; force_x=Fx, force_y=Fy)
    return (leaf=leaf, coarse_F=coarse, patch=patch, is_solid=is_solid,
            Fx_drag=Fx_sum / n_avg, Fy_drag=Fy_sum / n_avg,
            ux_mean=ux, uy_mean=uy, mass_initial=mass0,
            mass_final=_fluid_mass_F(leaf, is_solid))
end

function _run_coarse_cartesian_square_obstacle_drag(; Nx=24, Ny=14,
                                                   obstacle_i_range=11:14,
                                                   obstacle_j_range=6:9,
                                                   Fx=2e-5, Fy=0.0,
                                                   steps=1200, avg_window=300,
                                                   omega=1.0)
    F = zeros(Float64, Nx, Ny, 9)
    Fnext = similar(F)
    is_solid = square_solid_mask_leaf_2d(Nx, Ny, obstacle_i_range, obstacle_j_range)
    fill_equilibrium_integrated_D2Q9!(F, 1.0, 1.0, 0.0, 0.0)
    mass0 = _fluid_mass_F(F, is_solid)
    Fx_sum = 0.0
    Fy_sum = 0.0
    n_avg = 0

    for step in 1:steps
        collide_Guo_integrated_D2Q9!(F, is_solid, 1.0, omega, Fx, Fy)
        stream_periodic_x_wall_y_solid_F_2d!(Fnext, F, is_solid)
        if step > steps - avg_window
            drag = compute_drag_mea_solid_F_2d(F, Fnext, is_solid)
            Fx_sum += drag.Fx
            Fy_sum += drag.Fy
            n_avg += 1
        end
        F, Fnext = Fnext, F
    end

    ux, uy = _fluid_mean_velocity_F(F, is_solid; volume=1.0, force_x=Fx, force_y=Fy)
    return (F=F, is_solid=is_solid, Fx_drag=Fx_sum / n_avg,
            Fy_drag=Fy_sum / n_avg, ux_mean=ux, uy_mean=uy,
            mass_initial=mass0, mass_final=_fluid_mass_F(F, is_solid))
end

function _run_cartesian_leaf_cylinder_force(is_solid; Fx=2e-5, Fy=0.0,
                                            steps=1200, avg_window=300,
                                            omega=1.0, radius_leaf=3.0)
    F = zeros(Float64, size(is_solid, 1), size(is_solid, 2), 9)
    Fnext = similar(F)
    fill_equilibrium_integrated_D2Q9!(F, 0.25, 1.0, 0.0, 0.0)
    mass0 = _fluid_mass_F(F, is_solid)
    Fx_sum = 0.0
    Fy_sum = 0.0
    n_avg = 0

    for step in 1:steps
        collide_Guo_integrated_D2Q9!(F, is_solid, 0.25, omega, Fx, Fy)
        stream_periodic_x_wall_y_solid_F_2d!(Fnext, F, is_solid)
        if step > steps - avg_window
            drag = compute_drag_mea_solid_F_2d(F, Fnext, is_solid)
            Fx_sum += drag.Fx
            Fy_sum += drag.Fy
            n_avg += 1
        end
        F, Fnext = Fnext, F
    end

    ux, uy = _fluid_mean_velocity_F(F, is_solid; force_x=Fx, force_y=Fy)
    Fx_drag = Fx_sum / n_avg
    Fy_drag = Fy_sum / n_avg
    Cd = 2 * Fx_drag / (max(abs(ux), eps(Float64))^2 * 2 * radius_leaf)
    return (F=F, ux_mean=ux, uy_mean=uy, Fx_drag=Fx_drag, Fy_drag=Fy_drag,
            Cd=Cd, mass_initial=mass0, mass_final=_fluid_mass_F(F, is_solid))
end

function _run_cartesian_leaf_cylinder_channel(is_solid; u_in=0.03, steps=2000,
                                              avg_window=500, omega=1.0,
                                              radius_leaf=3.0)
    F = zeros(Float64, size(is_solid, 1), size(is_solid, 2), 9)
    Fnext = similar(F)
    fill_equilibrium_integrated_D2Q9!(F, 0.25, 1.0, u_in, 0.0)
    mass0 = _fluid_mass_F(F, is_solid)
    Fx_sum = 0.0
    Fy_sum = 0.0
    n_avg = 0

    for step in 1:steps
        stream_bounceback_xy_solid_F_2d!(Fnext, F, is_solid)
        apply_zou_he_west_F_2d!(Fnext, u_in, 0.25, is_solid)
        apply_zou_he_pressure_east_F_2d!(Fnext, 0.25, is_solid; rho_out=1.0)
        if step > steps - avg_window
            drag = compute_drag_mea_solid_F_2d(F, Fnext, is_solid)
            Fx_sum += drag.Fx
            Fy_sum += drag.Fy
            n_avg += 1
        end
        collide_BGK_integrated_D2Q9!(Fnext, is_solid, 0.25, omega)
        F, Fnext = Fnext, F
    end

    ux, uy = _fluid_mean_velocity_F(F, is_solid)
    Fx_drag = Fx_sum / n_avg
    Fy_drag = Fy_sum / n_avg
    Cd = 2 * Fx_drag / (u_in^2 * 2 * radius_leaf)
    return (F=F, ux_mean=ux, uy_mean=uy, Fx_drag=Fx_drag, Fy_drag=Fy_drag,
            Cd=Cd, mass_initial=mass0, mass_final=_fluid_mass_F(F, is_solid))
end

function _run_cartesian_leaf_bfs(is_solid; u_in=0.03, steps=800, omega=1.0)
    F = zeros(Float64, size(is_solid, 1), size(is_solid, 2), 9)
    Fnext = similar(F)
    fill_equilibrium_integrated_D2Q9!(F, 0.25, 1.0, u_in, 0.0)
    mass0 = _fluid_mass_F(F, is_solid)

    for _ in 1:steps
        stream_bounceback_xy_solid_F_2d!(Fnext, F, is_solid)
        apply_zou_he_west_F_2d!(Fnext, u_in, 0.25, is_solid)
        apply_zou_he_pressure_east_F_2d!(Fnext, 0.25, is_solid; rho_out=1.0)
        collide_BGK_integrated_D2Q9!(Fnext, is_solid, 0.25, omega)
        F, Fnext = Fnext, F
    end

    ux, uy = _fluid_mean_velocity_F(F, is_solid)
    return (F=F, ux_mean=ux, uy_mean=uy, mass_initial=mass0,
            mass_final=_fluid_mass_F(F, is_solid))
end

@testset "Conservative tree 2D" begin
    @testset "D2Q9 directions" begin
        @test d2q9_cx(1) == 0
        @test d2q9_cy(1) == 0
        @test d2q9_cx(2) == 1
        @test d2q9_cy(3) == 1
        @test d2q9_cx(4) == -1
        @test d2q9_cy(5) == -1
        @test d2q9_cx(6) == 1
        @test d2q9_cy(9) == -1
        @test d2q9_opposite(1) == 1
        @test d2q9_opposite(2) == 4
        @test d2q9_opposite(6) == 8
        @test d2q9_opposite(d2q9_opposite(9)) == 9
        @test_throws ArgumentError d2q9_cx(0)
        @test_throws ArgumentError d2q9_cy(10)
        @test_throws ArgumentError d2q9_opposite(10)
    end

    @testset "coalesce preserves per-population sums" begin
        Fc = reshape(collect(1.0:36.0), 2, 2, 9)
        Fp = zeros(Float64, 9)

        coalesce_F_2d!(Fp, Fc)

        for q in 1:9
            @test Fp[q] == sum(Fc[:, :, q])
        end
    end

    @testset "coalesce preserves mass and momentum" begin
        Fc = zeros(Float64, 2, 2, 9)
        for q in 1:9, iy in 1:2, ix in 1:2
            Fc[ix, iy, q] = 0.01 * (10q + 2ix + iy)
        end

        Fp = zeros(Float64, 9)
        coalesce_F_2d!(Fp, Fc)

        child_mass = sum(Fc)
        child_mx = sum(KrakeNN.d2q9_cx(q) * Fc[ix, iy, q]
                       for ix in 1:2, iy in 1:2, q in 1:9)
        child_my = sum(KrakeNN.d2q9_cy(q) * Fc[ix, iy, q]
                       for ix in 1:2, iy in 1:2, q in 1:9)

        m, mx, my = moments_F(Fp)
        @test isapprox(m, child_mass; atol=1e-14, rtol=0)
        @test isapprox(mx, child_mx; atol=1e-14, rtol=0)
        @test isapprox(my, child_my; atol=1e-14, rtol=0)
    end

    @testset "uniform explosion conserves parent" begin
        Fp = [0.44, 0.12, 0.08, 0.09, 0.11, 0.03, 0.02, 0.025, 0.035]
        Fc = zeros(Float64, 2, 2, 9)

        explode_uniform_F_2d!(Fc, Fp)

        Fback = zeros(Float64, 9)
        coalesce_F_2d!(Fback, Fc)

        @test isapprox(Fback, Fp; atol=1e-14, rtol=0)
        @test isapprox(collect(moments_F(Fback)), collect(moments_F(Fp)); atol=1e-14, rtol=0)
    end

    @testset "uniform projection loses local velocity and stress moments" begin
        Fc = zeros(Float64, 2, 2, 9)
        for iy in 1:2, ix in 1:2
            ux = iy == 1 ? 0.01 : 0.03
            pxy = iy == 1 ? -2e-4 : 2e-4
            _fill_equilibrium_with_pxy!(@view(Fc[ix, iy, :]), 0.25, 1.0, ux, 0.0, pxy)
        end

        Fp = zeros(Float64, 9)
        Fb = similar(Fc)
        before_moments = collect(moments_F(Fc))
        before_raw2 = collect(_aggregate_raw_second_moments_F(Fc))
        before_ux = [momentum_F(@view(Fc[ix, iy, :]))[1] /
                     mass_F(@view(Fc[ix, iy, :])) for ix in 1:2, iy in 1:2]
        before_pxy = [_noneq_second_moments_F(@view(Fc[ix, iy, :]))[2]
                      for ix in 1:2, iy in 1:2]

        coalesce_F_2d!(Fp, Fc)
        explode_uniform_F_2d!(Fb, Fp)

        after_ux = [momentum_F(@view(Fb[ix, iy, :]))[1] /
                    mass_F(@view(Fb[ix, iy, :])) for ix in 1:2, iy in 1:2]
        after_pxy = [_noneq_second_moments_F(@view(Fb[ix, iy, :]))[2]
                     for ix in 1:2, iy in 1:2]

        @test isapprox(collect(moments_F(Fb)), before_moments; atol=1e-14, rtol=0)
        @test isapprox(collect(_aggregate_raw_second_moments_F(Fb)),
                       before_raw2; atol=1e-14, rtol=0)
        @test maximum(abs.(after_ux .- before_ux)) > 0.009
        @test maximum(abs.(after_pxy .- before_pxy)) > 1.9e-4
    end

    @testset "eq/neq reconstruction preserves integrated macro moments" begin
        Fsrc = zeros(Float64, 9)
        _fill_equilibrium_with_pxy!(Fsrc, 1.0, 1.02, 0.04, -0.015, 2e-4)
        Fdst = zeros(Float64, 9)

        reconstruct_integrated_D2Q9_eq_neq!(
            Fdst, 0.25, Fsrc, 1.0; alpha=0.5)

        rho, ux, uy = macrostate_integrated_D2Q9(Fsrc, 1.0)
        @test isapprox(mass_F(Fdst), 0.25 * rho; atol=1e-14, rtol=0)
        @test isapprox(momentum_F(Fdst)[1], 0.25 * rho * ux; atol=1e-14, rtol=0)
        @test isapprox(momentum_F(Fdst)[2], 0.25 * rho * uy; atol=1e-14, rtol=0)
        @test isapprox(_noneq_second_moments_F(Fdst; volume=0.25)[2],
                       0.5 * 0.25 * _noneq_second_moments_F(Fsrc; volume=1.0)[2];
                       atol=1e-14, rtol=0)
    end

    @testset "eq/neq reconstruction roundtrips alpha-one child split" begin
        Fp = zeros(Float64, 9)
        _fill_equilibrium_with_pxy!(Fp, 1.0, 0.98, -0.02, 0.01, -1.5e-4)
        Fc = zeros(Float64, 2, 2, 9)
        for iy in 1:2, ix in 1:2
            reconstruct_integrated_D2Q9_eq_neq!(
                @view(Fc[ix, iy, :]), 0.25, Fp, 1.0; alpha=1.0)
        end

        Fback = zeros(Float64, 9)
        coalesce_F_2d!(Fback, Fc)

        @test isapprox(Fback, Fp; atol=1e-14, rtol=0)
        @test isapprox(collect(moments_F(Fback)), collect(moments_F(Fp));
                       atol=1e-14, rtol=0)
    end

    @testset "eq/neq reconstruction compiles for Float32" begin
        Fsrc = zeros(Float32, 9)
        fill_equilibrium_integrated_D2Q9!(Fsrc, 1.0f0, 1.0f0, 0.02f0, 0.0f0)
        Fdst = zeros(Float32, 9)

        reconstruct_integrated_D2Q9_eq_neq!(
            Fdst, 0.25f0, Fsrc, 1.0f0; alpha=1.0f0)

        @test eltype(Fdst) == Float32
        @test isapprox(Fdst, Fsrc ./ 4; atol=1f-6, rtol=0)
    end

    @testset "eq/neq route packet scales only non-equilibrium part" begin
        Fsrc = zeros(Float64, 9)
        _fill_equilibrium_with_pxy!(Fsrc, 1.0, 1.0, 0.03, 0.0, 2e-4)
        q = 6
        weight = 0.25
        rho, ux, uy = macrostate_integrated_D2Q9(Fsrc, 1.0)
        feq = equilibrium(D2Q9(), rho, ux, uy, q)

        raw = reconstructed_integrated_D2Q9_packet(
            Fsrc, 1.0, q, weight; alpha=1.0)
        relaxed = reconstructed_integrated_D2Q9_packet(
            Fsrc, 1.0, q, weight; alpha=0.25)

        @test isapprox(raw, weight * Fsrc[q]; atol=1e-14, rtol=0)
        @test isapprox(relaxed,
                       weight * (feq + 0.25 * (Fsrc[q] - feq));
                       atol=1e-14, rtol=0)
    end

    @testset "parent-child mapping" begin
        @test conservative_tree_parent_index(1, 1) == (1, 1, 1, 1)
        @test conservative_tree_parent_index(2, 1) == (1, 1, 2, 1)
        @test conservative_tree_parent_index(1, 2) == (1, 1, 1, 2)
        @test conservative_tree_parent_index(2, 2) == (1, 1, 2, 2)
        @test conservative_tree_parent_index(3, 4) == (2, 2, 1, 2)
        @test_throws ArgumentError conservative_tree_parent_index(0, 1)
        @test_throws ArgumentError conservative_tree_parent_index(1, 0)
    end

    @testset "vertical coarse to fine split conserves F" begin
        Fc = zeros(Float64, 2, 2, 9)
        split_coarse_to_fine_vertical_F_2d!(Fc, 3.7, 2)
        @test Fc[1, 1, 2] == 1.85
        @test Fc[1, 2, 2] == 1.85
        @test isapprox(sum(Fc[:, :, 2]), 3.7; atol=1e-14, rtol=0)
        @test isapprox(sum(Fc), 3.7; atol=1e-14, rtol=0)

        Fc .= 0
        split_coarse_to_fine_vertical_F_2d!(Fc, 2.6, 4)
        @test Fc[2, 1, 4] == 1.3
        @test Fc[2, 2, 4] == 1.3
        @test isapprox(sum(Fc[:, :, 4]), 2.6; atol=1e-14, rtol=0)

        Fc .= 0
        split_coarse_to_fine_vertical_F_2d!(Fc, 1.4, 6)
        @test isapprox(sum(Fc[:, :, 6]), 1.4; atol=1e-14, rtol=0)
        @test_throws ArgumentError split_coarse_to_fine_vertical_F_2d!(Fc, 1.0, 3)
    end

    @testset "vertical fine to coarse coalesces interface children" begin
        Fc = zeros(Float64, 2, 2, 9)
        Fc[1, 1, 4] = 1.2
        Fc[1, 2, 4] = 2.3
        Fc[2, 1, 4] = 99.0
        @test coalesce_fine_to_coarse_vertical_F(Fc, 4) == 3.5

        Fc .= 0
        Fc[2, 1, 2] = 0.4
        Fc[2, 2, 2] = 0.6
        Fc[1, 1, 2] = 99.0
        @test coalesce_fine_to_coarse_vertical_F(Fc, 2) == 1.0
        @test_throws ArgumentError coalesce_fine_to_coarse_vertical_F(Fc, 1)
    end

    @testset "face interface split covers all entering orientations" begin
        entering = Dict(
            :west => (2, 6, 9),
            :east => (4, 7, 8),
            :south => (3, 6, 7),
            :north => (5, 8, 9),
        )

        for face in (:west, :east, :south, :north)
            for q in entering[face]
                Fc = zeros(Float64, 2, 2, 9)
                packet = 10.0 + q / 10

                split_coarse_to_fine_face_F_2d!(Fc, packet, q, face)

                @test isapprox(sum(Fc[:, :, q]), packet; atol=1e-14, rtol=0)
                @test isapprox(sum(Fc), packet; atol=1e-14, rtol=0)
                @test isapprox(mass_F(Fc), packet; atol=1e-14, rtol=0)
                @test isapprox(momentum_F(Fc)[1], d2q9_cx(q) * packet; atol=1e-14, rtol=0)
                @test isapprox(momentum_F(Fc)[2], d2q9_cy(q) * packet; atol=1e-14, rtol=0)
            end

            for q in setdiff(1:9, entering[face])
                @test_throws ArgumentError split_coarse_to_fine_face_F_2d!(
                    zeros(Float64, 2, 2, 9), 1.0, q, face)
            end
        end
        @test_throws ArgumentError split_coarse_to_fine_face_F_2d!(
            zeros(Float64, 2, 2, 9), 1.0, 2, :badface)
    end

    @testset "face interface coalesce covers all leaving orientations" begin
        leaving = Dict(
            :west => (4, 7, 8),
            :east => (2, 6, 9),
            :south => (5, 8, 9),
            :north => (3, 6, 7),
        )

        for face in (:west, :east, :south, :north)
            for q in leaving[face]
                Fc = zeros(Float64, 2, 2, 9)
                for iy in 1:2, ix in 1:2
                    Fc[ix, iy, q] = 100.0 * q + 10.0 * ix + iy
                end

                expected = if face == :west
                    Fc[1, 1, q] + Fc[1, 2, q]
                elseif face == :east
                    Fc[2, 1, q] + Fc[2, 2, q]
                elseif face == :south
                    Fc[1, 1, q] + Fc[2, 1, q]
                else
                    Fc[1, 2, q] + Fc[2, 2, q]
                end

                @test coalesce_fine_to_coarse_face_F(Fc, q, face) == expected
            end

            for q in setdiff(1:9, leaving[face])
                @test_throws ArgumentError coalesce_fine_to_coarse_face_F(
                    zeros(Float64, 2, 2, 9), q, face)
            end
        end
        @test_throws ArgumentError coalesce_fine_to_coarse_face_F(
            zeros(Float64, 2, 2, 9), 2, :badface)
    end

    @testset "corner interface split and coalesce cover diagonal orientations" begin
        entering = Dict(
            :southwest => 6,
            :southeast => 7,
            :northeast => 8,
            :northwest => 9,
        )
        leaving = Dict(
            :northeast => 6,
            :northwest => 7,
            :southwest => 8,
            :southeast => 9,
        )

        for corner in (:southwest, :southeast, :northwest, :northeast)
            q = entering[corner]
            Fc = zeros(Float64, 2, 2, 9)
            packet = 2.0 + q / 8
            split_coarse_to_fine_corner_F_2d!(Fc, packet, q, corner)
            @test isapprox(sum(Fc[:, :, q]), packet; atol=1e-14, rtol=0)
            @test isapprox(mass_F(Fc), packet; atol=1e-14, rtol=0)
            @test isapprox(momentum_F(Fc)[1], d2q9_cx(q) * packet; atol=1e-14, rtol=0)
            @test isapprox(momentum_F(Fc)[2], d2q9_cy(q) * packet; atol=1e-14, rtol=0)

            for q_bad in setdiff(1:9, (q,))
                @test_throws ArgumentError split_coarse_to_fine_corner_F_2d!(
                    zeros(Float64, 2, 2, 9), 1.0, q_bad, corner)
            end

            q = leaving[corner]
            Fc .= 0
            expected_ix = corner in (:southwest, :northwest) ? 1 : 2
            expected_iy = corner in (:southwest, :southeast) ? 1 : 2
            Fc[expected_ix, expected_iy, q] = packet
            @test coalesce_fine_to_coarse_corner_F(Fc, q, corner) == packet
        end

        @test_throws ArgumentError split_coarse_to_fine_corner_F_2d!(
            zeros(Float64, 2, 2, 9), 1.0, 6, :badcorner)
        @test_throws ArgumentError coalesce_fine_to_coarse_corner_F(
            zeros(Float64, 2, 2, 9), 6, :badcorner)
    end

    @testset "coarse to fine full patch boundary transfer is conservative" begin
        patch = create_conservative_tree_patch_2d(3:6, 4:7)
        coarse_src = zeros(Float64, 9, 10, 9)
        for q in 1:9, j in axes(coarse_src, 2), i in axes(coarse_src, 1)
            coarse_src[i, j, q] = q + i / 64 + j / 128 + i * j / 4096
        end

        fine_dest = zeros(size(patch.fine_F))
        expected = zeros(size(patch.fine_F))
        for J in patch.parent_j_range, I in patch.parent_i_range, q in 1:9
            cx = d2q9_cx(q)
            cy = d2q9_cy(q)
            cx == 0 && cy == 0 && continue

            isrc = I - cx
            jsrc = J - cy
            checkbounds(Bool, coarse_src, isrc, jsrc, q) || continue
            (isrc in patch.parent_i_range && jsrc in patch.parent_j_range) && continue

            di = isrc < first(patch.parent_i_range) ? -1 :
                 isrc > last(patch.parent_i_range) ? 1 : 0
            dj = jsrc < first(patch.parent_j_range) ? -1 :
                 jsrc > last(patch.parent_j_range) ? 1 : 0
            il = I - first(patch.parent_i_range) + 1
            jl = J - first(patch.parent_j_range) + 1
            i0 = 2 * il - 1
            j0 = 2 * jl - 1
            packet = coarse_src[isrc, jsrc, q]

            if di != 0 && dj != 0
                ix = di < 0 ? i0 : i0 + 1
                iy = dj < 0 ? j0 : j0 + 1
                expected[ix, iy, q] += packet
            elseif di < 0
                expected[i0, j0, q] += packet / 2
                expected[i0, j0+1, q] += packet / 2
            elseif di > 0
                expected[i0+1, j0, q] += packet / 2
                expected[i0+1, j0+1, q] += packet / 2
            elseif dj < 0
                expected[i0, j0, q] += packet / 2
                expected[i0+1, j0, q] += packet / 2
            elseif dj > 0
                expected[i0, j0+1, q] += packet / 2
                expected[i0+1, j0+1, q] += packet / 2
            end
        end

        coarse_to_fine_patch_boundary_F_2d!(fine_dest, coarse_src, patch)

        @test isapprox(fine_dest, expected; atol=1e-14, rtol=0)
        _assert_per_q_sums_equal(fine_dest, expected)
        @test isapprox(collect(moments_F(fine_dest)),
                       collect(moments_F(expected)); atol=1e-14, rtol=0)
    end

    @testset "fine to coarse full patch boundary transfer is conservative" begin
        patch = create_conservative_tree_patch_2d(3:6, 4:7)
        fine_src = zeros(size(patch.fine_F))
        for q in 1:9, j in axes(fine_src, 2), i in axes(fine_src, 1)
            fine_src[i, j, q] = q / 2 + i / 128 + j / 256 + i * j / 8192
        end

        coarse_dest = zeros(Float64, 9, 10, 9)
        expected = zeros(size(coarse_dest))
        for J in patch.parent_j_range, I in patch.parent_i_range, q in 1:9
            cx = d2q9_cx(q)
            cy = d2q9_cy(q)
            cx == 0 && cy == 0 && continue

            idst = I + cx
            jdst = J + cy
            checkbounds(Bool, coarse_dest, idst, jdst, q) || continue
            (idst in patch.parent_i_range && jdst in patch.parent_j_range) && continue

            di = idst < first(patch.parent_i_range) ? -1 :
                 idst > last(patch.parent_i_range) ? 1 : 0
            dj = jdst < first(patch.parent_j_range) ? -1 :
                 jdst > last(patch.parent_j_range) ? 1 : 0
            il = I - first(patch.parent_i_range) + 1
            jl = J - first(patch.parent_j_range) + 1
            i0 = 2 * il - 1
            j0 = 2 * jl - 1

            if di != 0 && dj != 0
                ix = di < 0 ? i0 : i0 + 1
                iy = dj < 0 ? j0 : j0 + 1
                expected[idst, jdst, q] += fine_src[ix, iy, q]
            elseif di < 0
                expected[idst, jdst, q] += fine_src[i0, j0, q] + fine_src[i0, j0+1, q]
            elseif di > 0
                expected[idst, jdst, q] += fine_src[i0+1, j0, q] + fine_src[i0+1, j0+1, q]
            elseif dj < 0
                expected[idst, jdst, q] += fine_src[i0, j0, q] + fine_src[i0+1, j0, q]
            elseif dj > 0
                expected[idst, jdst, q] += fine_src[i0, j0+1, q] + fine_src[i0+1, j0+1, q]
            end
        end

        fine_to_coarse_patch_boundary_F_2d!(coarse_dest, fine_src, patch)

        @test isapprox(coarse_dest, expected; atol=1e-14, rtol=0)
        _assert_per_q_sums_equal(coarse_dest, expected)
        @test isapprox(collect(moments_F(coarse_dest)),
                       collect(moments_F(expected)); atol=1e-14, rtol=0)
    end

    @testset "composite active moments skip inactive parent ledger" begin
        patch = create_conservative_tree_patch_2d(3:5, 4:6)
        coarse_F = zeros(Float64, 8, 9, 9)
        for q in 1:9, j in axes(coarse_F, 2), i in axes(coarse_F, 1)
            coarse_F[i, j, q] = q + i / 32 + j / 64
        end
        for q in 1:9, j in axes(patch.fine_F, 2), i in axes(patch.fine_F, 1)
            patch.fine_F[i, j, q] = q / 8 + i / 128 + j / 256
        end

        expected_mass = sum(coarse_F[i, j, q]
                            for q in 1:9, j in axes(coarse_F, 2), i in axes(coarse_F, 1)
                            if !(i in patch.parent_i_range && j in patch.parent_j_range)) +
                        sum(patch.fine_F)
        expected_mx = sum(d2q9_cx(q) * coarse_F[i, j, q]
                          for q in 1:9, j in axes(coarse_F, 2), i in axes(coarse_F, 1)
                          if !(i in patch.parent_i_range && j in patch.parent_j_range)) +
                      momentum_F(patch.fine_F)[1]
        expected_my = sum(d2q9_cy(q) * coarse_F[i, j, q]
                          for q in 1:9, j in axes(coarse_F, 2), i in axes(coarse_F, 1)
                          if !(i in patch.parent_i_range && j in patch.parent_j_range)) +
                      momentum_F(patch.fine_F)[2]
        expected_pop = [sum(coarse_F[i, j, q]
                            for j in axes(coarse_F, 2), i in axes(coarse_F, 1)
                            if !(i in patch.parent_i_range && j in patch.parent_j_range)) +
                        sum(patch.fine_F[:, :, q]) for q in 1:9]

        @test isapprox(active_population_sums_F(coarse_F, patch), expected_pop; atol=1e-12, rtol=0)
        @test isapprox(active_mass_F(coarse_F, patch), expected_mass; atol=1e-12, rtol=0)
        @test isapprox(active_momentum_F(coarse_F, patch)[1], expected_mx; atol=1e-12, rtol=0)
        @test isapprox(active_momentum_F(coarse_F, patch)[2], expected_my; atol=1e-12, rtol=0)
        @test isapprox(collect(active_moments_F(coarse_F, patch)),
                       [expected_mass, expected_mx, expected_my]; atol=1e-12, rtol=0)

        coarse_F[4, 5, :] .+= 1000
        @test isapprox(active_population_sums_F(coarse_F, patch), expected_pop; atol=1e-12, rtol=0)
        @test isapprox(collect(active_moments_F(coarse_F, patch)),
                       [expected_mass, expected_mx, expected_my]; atol=1e-12, rtol=0)
    end

    @testset "composite leaf expansion round trip preserves active populations" begin
        patch = create_conservative_tree_patch_2d(2:4, 3:5)
        coarse_F = zeros(Float64, 6, 7, 9)
        for q in 1:9, j in axes(coarse_F, 2), i in axes(coarse_F, 1)
            coarse_F[i, j, q] = q + i / 16 + j / 32
        end
        coarse_F[3, 4, :] .+= 1000
        for q in 1:9, j in axes(patch.fine_F, 2), i in axes(patch.fine_F, 1)
            patch.fine_F[i, j, q] = q / 4 + i / 64 + j / 128
        end
        pop0 = active_population_sums_F(coarse_F, patch)
        moments0 = collect(active_moments_F(coarse_F, patch))

        leaf = zeros(Float64, 2 * size(coarse_F, 1), 2 * size(coarse_F, 2), 9)
        composite_to_leaf_F_2d!(leaf, coarse_F, patch)

        @test isapprox([sum(leaf[:, :, q]) for q in 1:9], pop0; atol=1e-12, rtol=0)
        @test leaf[2 * 3 - 1, 2 * 4 - 1, 1] == patch.fine_F[3, 3, 1]

        coarse_back = zeros(size(coarse_F))
        patch_back = create_conservative_tree_patch_2d(2:4, 3:5)
        leaf_to_composite_F_2d!(coarse_back, patch_back, leaf)

        @test isapprox(active_population_sums_F(coarse_back, patch_back), pop0; atol=1e-12, rtol=0)
        @test isapprox(collect(active_moments_F(coarse_back, patch_back)), moments0; atol=1e-12, rtol=0)
        @test all(iszero, coarse_back[2:4, 3:5, :])
    end

    @testset "composite leaf stream conserves and crosses coarse fine interface" begin
        patch_in = create_conservative_tree_patch_2d(3:4, 3:4)
        patch_out = create_conservative_tree_patch_2d(3:4, 3:4)
        coarse_in = zeros(Float64, 6, 6, 9)
        coarse_out = zeros(size(coarse_in))

        coarse_in[2, 3, 2] = 4.0
        pop0 = active_population_sums_F(coarse_in, patch_in)

        stream_composite_fully_periodic_leaf_F_2d!(coarse_out, patch_out,
                                                   coarse_in, patch_in)

        @test isapprox(active_population_sums_F(coarse_out, patch_out), pop0; atol=1e-14, rtol=0)
        @test isapprox(coarse_out[2, 3, 2], 2.0; atol=1e-14, rtol=0)
        @test isapprox(sum(patch_out.fine_F[:, :, 2]), 2.0; atol=1e-14, rtol=0)
        @test all(iszero, coarse_out[3:4, 3:4, :])
        @test isapprox(collect(active_moments_F(coarse_out, patch_out)),
                       collect(active_moments_F(coarse_in, patch_in)); atol=1e-14, rtol=0)
    end

    @testset "composite leaf stream preserves analytic active totals" begin
        patch_in = create_conservative_tree_patch_2d(3:5, 4:6)
        patch_out = create_conservative_tree_patch_2d(3:5, 4:6)
        coarse_in = zeros(Float64, 8, 9, 9)
        coarse_out = zeros(size(coarse_in))
        for q in 1:9, j in axes(coarse_in, 2), i in axes(coarse_in, 1)
            coarse_in[i, j, q] = q / 2 + i / 32 + j / 64
        end
        for q in 1:9, j in axes(patch_in.fine_F, 2), i in axes(patch_in.fine_F, 1)
            patch_in.fine_F[i, j, q] = q / 8 + i / 128 + j / 256
        end
        pop0 = active_population_sums_F(coarse_in, patch_in)
        moments0 = collect(active_moments_F(coarse_in, patch_in))

        stream_composite_fully_periodic_leaf_F_2d!(coarse_out, patch_out,
                                                   coarse_in, patch_in)

        @test isapprox(active_population_sums_F(coarse_out, patch_out), pop0; atol=1e-12, rtol=0)
        @test isapprox(collect(active_moments_F(coarse_out, patch_out)), moments0; atol=1e-12, rtol=0)
    end

    @testset "composite BGK collision preserves active moments" begin
        patch = create_conservative_tree_patch_2d(3:5, 4:6)
        coarse_F = zeros(Float64, 8, 9, 9)
        for q in 1:9, j in axes(coarse_F, 2), i in axes(coarse_F, 1)
            coarse_F[i, j, q] = 0.2 + q / 64 + i / 256 + j / 512
        end
        for q in 1:9, j in axes(patch.fine_F, 2), i in axes(patch.fine_F, 1)
            patch.fine_F[i, j, q] = 0.05 + q / 256 + i / 1024 + j / 2048
        end
        before = collect(active_moments_F(coarse_F, patch))

        collide_BGK_composite_F_2d!(coarse_F, patch, 1.0, 0.25, 1.1, 1.1)

        @test isapprox(collect(active_moments_F(coarse_F, patch)), before; atol=1e-11, rtol=0)
    end

    @testset "composite stationary wall rest state remains invariant" begin
        patch_in = create_conservative_tree_patch_2d(4:6, 3:5)
        patch_out = create_conservative_tree_patch_2d(4:6, 3:5)
        coarse_in = zeros(Float64, 9, 7, 9)
        coarse_out = zeros(size(coarse_in))
        fill_equilibrium_integrated_D2Q9!(coarse_in, 1.0, 1.0, 0.0, 0.0)
        fill_equilibrium_integrated_D2Q9!(patch_in.fine_F, 0.25, 1.0, 0.0, 0.0)
        pop0 = active_population_sums_F(coarse_in, patch_in)

        stream_composite_periodic_x_wall_y_leaf_F_2d!(coarse_out, patch_out,
                                                      coarse_in, patch_in)
        collide_BGK_composite_F_2d!(coarse_out, patch_out, 1.0, 0.25, 1.2, 1.2)

        @test isapprox(active_population_sums_F(coarse_out, patch_out), pop0; atol=1e-13, rtol=0)
        @test isapprox(collect(active_moments_F(coarse_out, patch_out)),
                       collect(active_moments_F(coarse_in, patch_in)); atol=1e-13, rtol=0)
    end

    @testset "composite Couette canary develops positive shear" begin
        nx, ny = 16, 12
        patch = create_conservative_tree_patch_2d(6:11, 4:9)
        patch_next = create_conservative_tree_patch_2d(6:11, 4:9)
        coarse = zeros(Float64, nx, ny, 9)
        coarse_next = zeros(size(coarse))
        fill_equilibrium_integrated_D2Q9!(coarse, 1.0, 1.0, 0.0, 0.0)
        fill_equilibrium_integrated_D2Q9!(patch.fine_F, 0.25, 1.0, 0.0, 0.0)
        mass0 = active_mass_F(coarse, patch)
        U = 0.04

        for _ in 1:700
            collide_BGK_composite_F_2d!(coarse, patch, 1.0, 0.25, 1.0, 1.0)
            stream_composite_periodic_x_moving_wall_y_leaf_F_2d!(
                coarse_next, patch_next, coarse, patch; u_north=U, volume_leaf=0.25)
            coarse, coarse_next = coarse_next, coarse
            patch, patch_next = patch_next, patch
        end

        profile = _composite_mean_ux_by_leaf_y(coarse, patch; volume_leaf=0.25)
        y = collect(0:length(profile)-1)
        ymean = sum(y) / length(y)
        pmean = sum(profile) / length(profile)
        slope = sum((y[k] - ymean) * (profile[k] - pmean) for k in eachindex(y)) /
                sum((yk - ymean)^2 for yk in y)

        @test isapprox(active_mass_F(coarse, patch), mass0; atol=1e-9, rtol=0)
        @test slope > 0
        @test profile[end] > profile[1] + 0.006
        @test profile[end] > 0.01
    end

    @testset "composite Poiseuille canary develops centerline maximum" begin
        nx, ny = 18, 14
        patch = create_conservative_tree_patch_2d(7:12, 5:10)
        patch_next = create_conservative_tree_patch_2d(7:12, 5:10)
        coarse = zeros(Float64, nx, ny, 9)
        coarse_next = zeros(size(coarse))
        fill_equilibrium_integrated_D2Q9!(coarse, 1.0, 1.0, 0.0, 0.0)
        fill_equilibrium_integrated_D2Q9!(patch.fine_F, 0.25, 1.0, 0.0, 0.0)
        mass0 = active_mass_F(coarse, patch)
        Fx = 5e-5

        for _ in 1:1400
            collide_Guo_composite_F_2d!(coarse, patch, 1.0, 0.25, 1.0, 1.0, Fx, 0.0)
            stream_composite_periodic_x_wall_y_leaf_F_2d!(coarse_next, patch_next,
                                                          coarse, patch)
            coarse, coarse_next = coarse_next, coarse
            patch, patch_next = patch_next, patch
        end

        profile = _composite_mean_ux_by_leaf_y(coarse, patch; volume_leaf=0.25, force_x=Fx)
        center = div(length(profile), 2)
        left = profile[2:center]
        right = reverse(profile[center+1:end-1])
        ncmp = min(length(left), length(right))

        @test isapprox(active_mass_F(coarse, patch), mass0; atol=1e-8, rtol=0)
        @test profile[center] > 0.002
        @test profile[center] > profile[2] + 0.001
        @test profile[center] > profile[end-1] + 0.001
        @test maximum(abs.(left[1:ncmp] .- right[1:ncmp])) < 0.003
    end

    @testset "macroflow runner Couette is quantitatively functional" begin
        result = run_conservative_tree_couette_macroflow_2d()

        @test result.flow == :couette
        @test result.steps == 3000
        @test abs(result.mass_drift) < 1e-8
        @test result.l2_error < 1.0e-3
        @test result.linf_error < 2.0e-3
        @test result.ux_profile[end] > result.ux_profile[1] + 0.03
        @test result.patch isa ConservativeTreePatch2D
    end

    @testset "macroflow Couette matches equivalent Cartesian leaf run" begin
        result = run_conservative_tree_couette_macroflow_2d()
        cart = _run_cartesian_leaf_couette(2 * size(result.coarse_F, 1),
                                           2 * size(result.coarse_F, 2);
                                           steps=result.steps)
        diff = result.ux_profile .- cart.ux_profile

        @test abs(mass_F(cart.F) - cart.mass_initial) < 1e-8
        @test abs(result.mass_drift) < 1e-8
        @test sqrt(sum(diff .^ 2) / length(diff)) < 1.2e-3
        @test maximum(abs.(diff)) < 2.0e-3
    end

    @testset "macroflow runner Poiseuille is functionally parabolic" begin
        result = run_conservative_tree_poiseuille_macroflow_2d()
        center = div(length(result.ux_profile), 2)
        left = result.ux_profile[2:center]
        right = reverse(result.ux_profile[center+1:end-1])
        ncmp = min(length(left), length(right))

        @test result.flow == :poiseuille
        @test result.steps == 5000
        @test abs(result.mass_drift) < 1e-8
        @test result.l2_error < 1.0e-2
        @test result.linf_error < 1.5e-2
        @test result.ux_profile[center] > 0.01
        @test result.ux_profile[center] > result.ux_profile[2] + 0.005
        @test result.ux_profile[center] > result.ux_profile[end-1] + 0.005
        @test maximum(abs.(left[1:ncmp] .- right[1:ncmp])) < 0.003
    end

    @testset "macroflow Poiseuille matches equivalent Cartesian leaf run" begin
        result = run_conservative_tree_poiseuille_macroflow_2d()
        cart = _run_cartesian_leaf_poiseuille(2 * size(result.coarse_F, 1),
                                              2 * size(result.coarse_F, 2);
                                              steps=result.steps)
        diff = result.ux_profile .- cart.ux_profile

        @test abs(mass_F(cart.F) - cart.mass_initial) < 1e-8
        @test abs(result.mass_drift) < 1e-8
        @test sqrt(sum(diff .^ 2) / length(diff)) < 1.2e-2
        @test maximum(abs.(diff)) < 1.6e-2
    end

    @testset "macroflow Poiseuille analytic with full x and y refinement bands" begin
        full = run_conservative_tree_poiseuille_macroflow_2d(
            ; Nx=24, Ny=16, patch_i_range=1:24, patch_j_range=1:16,
            steps=5000)
        vertical = run_conservative_tree_poiseuille_macroflow_2d(
            ; Nx=24, Ny=16, patch_i_range=11:14, patch_j_range=1:16,
            steps=5000)
        horizontal = run_conservative_tree_poiseuille_macroflow_2d(
            ; Nx=24, Ny=16, patch_i_range=1:24, patch_j_range=7:10,
            steps=5000)

        @test abs(full.mass_drift) < 1e-8
        @test abs(vertical.mass_drift) < 1e-8
        @test abs(horizontal.mass_drift) < 1e-8
        @test full.l2_error < 3.0e-3
        @test full.linf_error < 3.0e-3
        @test vertical.l2_error < 3.0e-3
        @test vertical.linf_error < 1.7e-2
        @test horizontal.l2_error < 3.0e-3
        @test horizontal.linf_error < 1.7e-2
        @test vertical.ux_profile[div(length(vertical.ux_profile), 2)] >
              vertical.ux_profile[2] + 0.01
        @test horizontal.ux_profile[div(length(horizontal.ux_profile), 2)] >
              horizontal.ux_profile[2] + 0.01
    end

    @testset "integrated BGK collision conserves mass and momentum" begin
        F0 = [0.44, 0.12, 0.08, 0.09, 0.11, 0.03, 0.02, 0.025, 0.035]
        F1 = copy(F0)

        collide_BGK_integrated_D2Q9!(F1, 1.0, 1.2)

        @test isapprox(mass_F(F1), mass_F(F0); atol=1e-14, rtol=0)
        @test isapprox(momentum_F(F1)[1], momentum_F(F0)[1]; atol=1e-14, rtol=0)
        @test isapprox(momentum_F(F1)[2], momentum_F(F0)[2]; atol=1e-14, rtol=0)
    end

    @testset "integrated equilibrium fill recovers imposed moments" begin
        F = zeros(Float64, 9)
        fill_equilibrium_integrated_D2Q9!(F, 0.25, 1.2, 0.05, -0.025)

        @test isapprox(mass_F(F), 0.25 * 1.2; atol=1e-14, rtol=0)
        @test isapprox(momentum_F(F)[1], 0.25 * 1.2 * 0.05; atol=1e-14, rtol=0)
        @test isapprox(momentum_F(F)[2], 0.25 * 1.2 * -0.025; atol=1e-14, rtol=0)

        Fgrid = zeros(Float64, 4, 5, 9)
        fill_equilibrium_integrated_D2Q9!(Fgrid, 0.25,
            (i, j) -> 1.0 + i / 64,
            (i, j) -> j / 256,
            (i, j) -> -i / 512)
        expected_mass = sum(0.25 * (1.0 + i / 64) for j in 1:5, i in 1:4)
        expected_mx = sum(0.25 * (1.0 + i / 64) * (j / 256) for j in 1:5, i in 1:4)
        expected_my = sum(0.25 * (1.0 + i / 64) * (-i / 512) for j in 1:5, i in 1:4)
        @test isapprox(mass_F(Fgrid), expected_mass; atol=1e-13, rtol=0)
        @test isapprox(momentum_F(Fgrid)[1], expected_mx; atol=1e-13, rtol=0)
        @test isapprox(momentum_F(Fgrid)[2], expected_my; atol=1e-13, rtol=0)
    end

    @testset "periodic integrated streaming is conservative" begin
        Fin = zeros(Float64, 4, 5, 9)
        for q in 1:9, j in axes(Fin, 2), i in axes(Fin, 1)
            Fin[i, j, q] = q + i / 64 + j / 32 + i * j / 1024
        end
        Fin[1, 1, 6] = 99.0
        Fout = similar(Fin)

        stream_fully_periodic_F_2d!(Fout, Fin)

        @test Fout[2, 2, 6] == 99.0
        _assert_per_q_sums_equal(Fout, Fin)
        @test isapprox(collect(moments_F(Fout)), collect(moments_F(Fin)); atol=1e-14, rtol=0)
        @test_throws ArgumentError stream_fully_periodic_F_2d!(Fin, Fin)
    end

    @testset "stationary wall streaming conserves mass on leaf grid" begin
        Fin = zeros(Float64, 5, 6, 9)
        for q in 1:9, j in axes(Fin, 2), i in axes(Fin, 1)
            Fin[i, j, q] = q / 16 + i / 128 + j / 256 + i * j / 2048
        end
        Fout = similar(Fin)

        stream_periodic_x_wall_y_F_2d!(Fout, Fin)

        @test isapprox(mass_F(Fout), mass_F(Fin); atol=1e-13, rtol=0)
        @test Fout[1, 1, 3] == Fin[1, 1, 5]
        @test Fout[1, end, 5] == Fin[1, end, 3]
        @test_throws ArgumentError stream_periodic_x_wall_y_F_2d!(Fin, Fin)
    end

    @testset "stationary wall rest state remains invariant" begin
        Fin = zeros(Float64, 8, 7, 9)
        Ftmp = similar(Fin)
        fill_equilibrium_integrated_D2Q9!(Fin, 1.0, 1.0, 0.0, 0.0)

        stream_periodic_x_wall_y_F_2d!(Ftmp, Fin)
        collide_BGK_integrated_D2Q9!(Ftmp, 1.0, 1.2)

        @test isapprox(Ftmp, Fin; atol=1e-14, rtol=0)
        @test isapprox(collect(moments_F(Ftmp)), collect(moments_F(Fin)); atol=1e-14, rtol=0)
    end

    @testset "moving wall streaming preserves mass and injects wall momentum" begin
        Fin = zeros(Float64, 10, 8, 9)
        Fout = similar(Fin)
        fill_equilibrium_integrated_D2Q9!(Fin, 1.0, 1.0, 0.0, 0.0)

        stream_periodic_x_moving_wall_y_F_2d!(Fout, Fin; u_north=0.05)

        @test isapprox(mass_F(Fout), mass_F(Fin); rtol=1e-12)
        @test momentum_F(Fout)[1] > momentum_F(Fin)[1]
        @test_throws ArgumentError stream_periodic_x_moving_wall_y_F_2d!(Fin, Fin)
    end

    @testset "solid cylinder leaf mask and rest state are stable" begin
        mask = cylinder_solid_mask_leaf_2d(16, 12, 8.5, 6.5, 2.5)
        @test sum(mask) > 0
        @test mask[8, 6]
        @test !mask[1, 1]

        Fin = zeros(Float64, 16, 12, 9)
        Fout = similar(Fin)
        fill_equilibrium_integrated_D2Q9!(Fin, 1.0, 1.0, 0.0, 0.0)

        stream_periodic_x_wall_y_solid_F_2d!(Fout, Fin, mask)
        drag = compute_drag_mea_solid_F_2d(Fin, Fout, mask)
        collide_BGK_integrated_D2Q9!(Fout, mask, 1.0, 1.1)

        @test isapprox(Fout, Fin; atol=1e-14, rtol=0)
        @test abs(drag.Fx) < 1e-14
        @test abs(drag.Fy) < 1e-14
        @test_throws ArgumentError stream_periodic_x_wall_y_solid_F_2d!(Fin, Fin, mask)
    end

    @testset "square and BFS masks preserve rest under open channel BCs" begin
        square = square_solid_mask_leaf_2d(18, 12, 7:10, 5:8)
        bfs = backward_facing_step_solid_mask_leaf_2d(18, 12, 5, 4)

        @test sum(square) == 16
        @test square[7, 5]
        @test !square[6, 5]
        @test sum(bfs) == 20
        @test bfs[5, 4]
        @test !bfs[6, 4]
        @test_throws ArgumentError square_solid_mask_leaf_2d(8, 8, 0:2, 3:4)
        @test_throws ArgumentError backward_facing_step_solid_mask_leaf_2d(8, 8, 8, 3)

        for mask in (square, bfs)
            Fin = zeros(Float64, size(mask, 1), size(mask, 2), 9)
            Fout = similar(Fin)
            fill_equilibrium_integrated_D2Q9!(Fin, 1.0, 1.0, 0.0, 0.0)

            stream_bounceback_xy_solid_F_2d!(Fout, Fin, mask)
            apply_zou_he_west_F_2d!(Fout, 0.0, 1.0, mask)
            apply_zou_he_pressure_east_F_2d!(Fout, 1.0, mask; rho_out=1.0)
            collide_BGK_integrated_D2Q9!(Fout, mask, 1.0, 1.1)

            @test isapprox(Fout, Fin; atol=1e-14, rtol=0)
        end
    end

    @testset "integrated Zou-He channel BC canaries" begin
        F = zeros(Float64, 6, 5, 9)
        fill_equilibrium_integrated_D2Q9!(F, 0.25, 1.0, 0.0, 0.0)

        apply_zou_he_west_F_2d!(F, 0.04, 0.25)
        west = @view F[1, 3, :]
        @test isapprox(momentum_F(west)[1] / mass_F(west), 0.04; atol=1e-14, rtol=0)
        @test isapprox(momentum_F(west)[2] / mass_F(west), 0.0; atol=1e-14, rtol=0)

        apply_zou_he_pressure_east_F_2d!(F, 0.25; rho_out=1.0)
        east = @view F[end, 3, :]
        @test isapprox(mass_F(east), 0.25; atol=1e-14, rtol=0)
        @test isapprox(momentum_F(east)[2], 0.0; atol=1e-14, rtol=0)

        mask = backward_facing_step_solid_mask_leaf_2d(6, 5, 1, 2)
        before_solid = copy(@view F[1, 1, :])
        apply_zou_he_west_F_2d!(F, 0.03, 0.25, mask)
        @test isapprox(@view(F[1, 1, :]), before_solid; atol=1e-14, rtol=0)
    end

    @testset "macroflow square obstacle matches Cartesian canary" begin
        result = run_conservative_tree_square_obstacle_macroflow_2d()
        cart = _run_cartesian_leaf_square_obstacle(result.is_solid_leaf;
                                                   steps=result.steps)

        @test result.flow == :square_obstacle
        @test result.steps == 1200
        @test abs(result.mass_drift) < 1e-8
        @test abs(cart.mass_final - cart.mass_initial) < 1e-8
        @test result.ux_mean > 0
        @test cart.ux_mean > result.ux_mean
        @test abs(result.ux_mean - cart.ux_mean) < 1.0e-3
        @test abs(result.uy_mean - cart.uy_mean) < 1e-12
    end

    @testset "square obstacle drag with enclosing refinement vs coarse Cartesian" begin
        refined = _run_conservative_tree_square_obstacle_drag()
        coarse = _run_coarse_cartesian_square_obstacle_drag()

        obstacle_parents_i = 11:14
        obstacle_parents_j = 6:9
        @test first(refined.patch.parent_i_range) <= first(obstacle_parents_i)
        @test last(refined.patch.parent_i_range) >= last(obstacle_parents_i)
        @test first(refined.patch.parent_j_range) <= first(obstacle_parents_j)
        @test last(refined.patch.parent_j_range) >= last(obstacle_parents_j)
        @test abs(refined.mass_final - refined.mass_initial) < 1e-8
        @test abs(coarse.mass_final - coarse.mass_initial) < 1e-8
        @test refined.Fx_drag > 0
        @test coarse.Fx_drag > 0
        @test abs(refined.Fy_drag / refined.Fx_drag) < 1e-12
        @test abs(coarse.Fy_drag / coarse.Fx_drag) < 1e-12
        @test abs(refined.Fx_drag - coarse.Fx_drag) / coarse.Fx_drag < 0.15
        @test refined.ux_mean > coarse.ux_mean
    end

    @testset "macroflow BFS matches Cartesian canary" begin
        result = run_conservative_tree_bfs_macroflow_2d()
        cart = _run_cartesian_leaf_bfs(result.is_solid_leaf; steps=result.steps)

        @test result.flow == :bfs
        @test result.steps == 800
        @test result.ux_mean > 0.015
        @test abs(result.ux_mean - cart.ux_mean) < 5e-4
        @test abs(result.uy_mean - cart.uy_mean) < 5e-5
        @test abs(result.mass_final - cart.mass_final) < 2.1
    end

    @testset "macroflow cylinder matches Cartesian canary" begin
        result = run_conservative_tree_cylinder_macroflow_2d()
        cart = _run_cartesian_leaf_cylinder_force(result.is_solid_leaf;
                                                  steps=result.steps,
                                                  avg_window=result.avg_window)

        @test result.steps == 1200
        @test result.avg_window == 300
        @test abs(result.mass_drift) < 1e-8
        @test abs(cart.mass_final - cart.mass_initial) < 1e-8
        @test result.u_ref > 0
        @test cart.ux_mean > result.u_ref
        @test abs(result.u_ref - cart.ux_mean) < 1.1e-3
        @test abs(result.Fx_drag - cart.Fx_drag) < 6e-4
        @test abs(result.Fy_drag - cart.Fy_drag) < 3e-6
        @test isfinite(result.Cd)
        @test result.Cd > 0
    end

    @testset "macroflow cylinder channel full-patch matches Cartesian" begin
        result = run_conservative_tree_cylinder_channel_macroflow_2d(
            ; patch_i_range=1:24, patch_j_range=1:14, Re=5.0,
            steps=500, avg_window=100)
        cart = _run_cartesian_leaf_cylinder_channel(result.is_solid_leaf;
                                                    steps=result.steps,
                                                    avg_window=result.avg_window,
                                                    omega=result.omega)

        @test result.steps == 500
        @test result.avg_window == 100
        @test isapprox(result.Cd, cart.Cd; atol=1e-14, rtol=0)
        @test isapprox(result.Fx_drag, cart.Fx_drag; atol=1e-14, rtol=0)
        @test isapprox(result.Fy_drag, cart.Fy_drag; atol=1e-14, rtol=0)
        @test isapprox(result.ux_mean, cart.ux_mean; atol=1e-14, rtol=0)
        @test isapprox(result.mass_final, cart.mass_final; atol=1e-14, rtol=0)
    end

    @testset "macroflow cylinder channel convergence canary" begin
        low = run_conservative_tree_cylinder_channel_macroflow_2d(
            ; Re=5.0, steps=2000, avg_window=500)
        mid = run_conservative_tree_cylinder_channel_macroflow_2d(
            ; Nx=32, Ny=18, patch_i_range=5:12, patch_j_range=5:14,
            cx_leaf=16.0, cy_leaf=18.0, radius_leaf=4.0,
            Re=5.0, steps=2000, avg_window=500)
        high = run_conservative_tree_cylinder_channel_macroflow_2d(
            ; Nx=40, Ny=22, patch_i_range=7:15, patch_j_range=6:17,
            cx_leaf=20.0, cy_leaf=22.0, radius_leaf=5.0,
            Re=5.0, steps=2000, avg_window=500)

        @test all(isfinite, (low.Cd, mid.Cd, high.Cd))
        @test 3.0 < low.Cd < 10.0
        @test 3.0 < mid.Cd < 10.0
        @test 3.0 < high.Cd < 10.0
        @test abs(high.Cd - mid.Cd) < abs(mid.Cd - low.Cd)
        @test abs(high.Cd - mid.Cd) / high.Cd < 0.12
        @test abs(high.ux_mean - mid.ux_mean) < 1.2e-3
        @test abs(low.Fy_drag / low.Fx_drag) < 0.08
        @test abs(mid.Fy_drag / mid.Fx_drag) < 0.08
        @test abs(high.Fy_drag / high.Fx_drag) < 0.08
    end

    @testset "minimal integrated Couette canary develops positive shear" begin
        nx, ny = 24, 18
        U = 0.05
        omega = 1.0
        F = zeros(Float64, nx, ny, 9)
        Fnext = similar(F)
        fill_equilibrium_integrated_D2Q9!(F, 1.0, 1.0, 0.0, 0.0)
        mass0 = mass_F(F)

        for _ in 1:1400
            collide_BGK_integrated_D2Q9!(F, 1.0, omega)
            stream_periodic_x_moving_wall_y_F_2d!(Fnext, F; u_north=U)
            F, Fnext = Fnext, F
        end

        profile = _mean_ux_by_y(F)
        y = collect(0:ny-1)
        ymean = sum(y) / length(y)
        pmean = sum(profile) / length(profile)
        slope_num = sum((y[j] - ymean) * (profile[j] - pmean) for j in eachindex(y))
        slope_den = sum((yj - ymean)^2 for yj in y)
        slope = slope_num / slope_den

        @test isapprox(mass_F(F), mass0; atol=1e-10, rtol=0)
        @test slope > 0
        @test profile[end] > profile[1] + 0.015
        @test profile[end] > 0.02
        @test maximum(abs, diff(profile)) < 0.02

        patch = create_conservative_tree_patch_2d(1:div(nx, 2), 1:div(ny, 2))
        patch.fine_F .= F
        coalesce_patch_to_shadow_F_2d!(patch)
        _assert_per_q_sums_equal(patch.coarse_shadow_F, F; atol=1e-12)
        @test isapprox(collect(moments_F(patch.coarse_shadow_F)),
                       collect(moments_F(F)); rtol=1e-12)
    end

    @testset "grid BGK collision conserves global moments" begin
        F = zeros(Float64, 4, 3, 9)
        for q in 1:9, j in axes(F, 2), i in axes(F, 1)
            F[i, j, q] = 0.2 + q / 64 + i / 256 + j / 512
        end
        before = collect(moments_F(F))

        collide_BGK_integrated_D2Q9!(F, 0.25, 1.1)

        @test isapprox(collect(moments_F(F)), before; atol=1e-12, rtol=0)
    end

    @testset "integrated Guo collision conserves mass and drives momentum" begin
        F = zeros(Float64, 9)
        fill_equilibrium_integrated_D2Q9!(F, 1.0, 1.0, 0.0, 0.0)
        mass0 = mass_F(F)

        collide_Guo_integrated_D2Q9!(F, 1.0, 1.0, 1e-4, 0.0)

        @test isapprox(mass_F(F), mass0; atol=1e-14, rtol=0)
        @test momentum_F(F)[1] > 0
        @test abs(momentum_F(F)[2]) < 1e-14
    end

    @testset "minimal integrated Poiseuille canary develops centerline maximum" begin
        nx, ny = 32, 22
        omega = 1.0
        Fx = 5e-5
        F = zeros(Float64, nx, ny, 9)
        Fnext = similar(F)
        fill_equilibrium_integrated_D2Q9!(F, 1.0, 1.0, 0.0, 0.0)
        mass0 = mass_F(F)

        for _ in 1:2600
            collide_Guo_integrated_D2Q9!(F, 1.0, omega, Fx, 0.0)
            stream_periodic_x_wall_y_F_2d!(Fnext, F)
            F, Fnext = Fnext, F
        end

        profile = _mean_ux_by_y(F; force_x=Fx)
        center = div(ny, 2)
        left = profile[2:center]
        right = reverse(profile[center+1:ny-1])
        ncmp = min(length(left), length(right))

        @test isapprox(mass_F(F), mass0; atol=1e-8, rtol=0)
        @test profile[center] > 0.003
        @test profile[center] > profile[2] + 0.002
        @test profile[center] > profile[end-1] + 0.002
        @test maximum(abs.(left[1:ncmp] .- right[1:ncmp])) < 0.002

        patch = create_conservative_tree_patch_2d(1:div(nx, 2), 1:div(ny, 2))
        patch.fine_F .= F
        coalesce_patch_to_shadow_F_2d!(patch)
        _assert_per_q_sums_equal(patch.coarse_shadow_F, F; atol=1e-12)
        @test isapprox(collect(moments_F(patch.coarse_shadow_F)),
                       collect(moments_F(F)); rtol=1e-12)
    end

    @testset "experimental patch allocation" begin
        patch = create_conservative_tree_patch_2d(3:5, 7:8; T=Float32)

        @test patch.parent_i_range == 3:5
        @test patch.parent_j_range == 7:8
        @test patch.ratio == 2
        @test size(patch.fine_F) == (6, 4, 9)
        @test size(patch.coarse_shadow_F) == (3, 2, 9)
        @test eltype(patch.fine_F) == Float32
        @test all(iszero, patch.fine_F)
        @test all(iszero, patch.coarse_shadow_F)
        @test_throws ArgumentError create_conservative_tree_patch_2d(1:2, 1:2; ratio=3)
    end

    @testset "analytic fine patch coalesces to parent ledger" begin
        patch = create_conservative_tree_patch_2d(3:5, 7:8)
        for q in 1:9, j in axes(patch.fine_F, 2), i in axes(patch.fine_F, 1)
            patch.fine_F[i, j, q] = q / 4 + i / 64 + j / 32 + i * j / 1024
        end
        fine_before = copy(patch.fine_F)

        coalesce_patch_to_shadow_F_2d!(patch)

        for q in 1:9, jp in axes(patch.coarse_shadow_F, 2), ip in axes(patch.coarse_shadow_F, 1)
            i0 = 2 * ip - 1
            j0 = 2 * jp - 1
            expected = sum(fine_before[i0:i0+1, j0:j0+1, q])
            @test isapprox(patch.coarse_shadow_F[ip, jp, q], expected; atol=1e-14, rtol=0)
        end

        for q in 1:9
            @test isapprox(sum(patch.coarse_shadow_F[:, :, q]), sum(fine_before[:, :, q]);
                           atol=1e-14, rtol=0)
        end
        @test isapprox(collect(moments_F(patch.coarse_shadow_F)),
                       collect(moments_F(fine_before)); atol=1e-14, rtol=0)
    end

    @testset "analytic parent ledger explodes to fine patch conservatively" begin
        patch = create_conservative_tree_patch_2d(2:4, 5:7)
        for q in 1:9, jp in axes(patch.coarse_shadow_F, 2), ip in axes(patch.coarse_shadow_F, 1)
            patch.coarse_shadow_F[ip, jp, q] = 1 + q / 2 + ip / 32 + jp / 16
        end
        shadow_before = copy(patch.coarse_shadow_F)

        explode_shadow_to_patch_uniform_F_2d!(patch)

        Fback = zeros(Float64, 9)
        for jp in axes(shadow_before, 2), ip in axes(shadow_before, 1)
            i0 = 2 * ip - 1
            j0 = 2 * jp - 1
            for q in 1:9
                @test patch.fine_F[i0, j0, q] == shadow_before[ip, jp, q] / 4
                @test patch.fine_F[i0+1, j0, q] == shadow_before[ip, jp, q] / 4
                @test patch.fine_F[i0, j0+1, q] == shadow_before[ip, jp, q] / 4
                @test patch.fine_F[i0+1, j0+1, q] == shadow_before[ip, jp, q] / 4
            end

            coalesce_F_2d!(Fback, @view patch.fine_F[i0:i0+1, j0:j0+1, :])
            @test isapprox(Fback, vec(shadow_before[ip, jp, :]); atol=1e-14, rtol=0)
        end

        for q in 1:9
            @test isapprox(sum(patch.fine_F[:, :, q]), sum(shadow_before[:, :, q]);
                           atol=1e-14, rtol=0)
        end
        @test isapprox(collect(moments_F(patch.fine_F)),
                       collect(moments_F(shadow_before)); atol=1e-14, rtol=0)
    end

    @testset "couette imposed flow survives fine-parent ledger transfer" begin
        patch = create_conservative_tree_patch_2d(1:4, 1:5)
        vol_fine = 0.25
        ny = size(patch.fine_F, 2)
        U = 0.0625

        fill_equilibrium_integrated_D2Q9!(patch.fine_F, vol_fine,
            1.0,
            (i, j) -> U * (j - 1) / (ny - 1),
            0.0)
        fine_before = copy(patch.fine_F)
        expected_mx = sum(vol_fine * U * (j - 1) / (ny - 1)
                          for j in axes(patch.fine_F, 2), i in axes(patch.fine_F, 1))

        coalesce_patch_to_shadow_F_2d!(patch)

        _assert_per_q_sums_equal(patch.coarse_shadow_F, fine_before; atol=1e-13)
        @test isapprox(mass_F(patch.coarse_shadow_F), mass_F(fine_before); atol=1e-13, rtol=0)
        @test isapprox(momentum_F(patch.coarse_shadow_F)[1], expected_mx; atol=1e-13, rtol=0)
        @test isapprox(momentum_F(patch.coarse_shadow_F)[2], 0.0; atol=1e-13, rtol=0)
    end

    @testset "poiseuille imposed flow survives parent-fine round trip totals" begin
        patch = create_conservative_tree_patch_2d(1:5, 1:4)
        vol_fine = 0.25
        ny = size(patch.fine_F, 2)
        Umax = 0.08

        fill_equilibrium_integrated_D2Q9!(patch.fine_F, vol_fine,
            (i, j) -> 1.0 + j / 512,
            (i, j) -> begin
                y = (j - 0.5) / ny
                Umax * 4 * y * (1 - y)
            end,
            0.0)
        fine_before = copy(patch.fine_F)

        coalesce_patch_to_shadow_F_2d!(patch)
        shadow_before = copy(patch.coarse_shadow_F)
        patch.fine_F .= 0
        explode_shadow_to_patch_uniform_F_2d!(patch)

        _assert_per_q_sums_equal(patch.fine_F, shadow_before; atol=1e-13)
        @test isapprox(collect(moments_F(patch.fine_F)),
                       collect(moments_F(fine_before)); rtol=1e-12)

        coalesce_patch_to_shadow_F_2d!(patch)
        @test isapprox(patch.coarse_shadow_F, shadow_before; atol=1e-14, rtol=0)
    end
end
