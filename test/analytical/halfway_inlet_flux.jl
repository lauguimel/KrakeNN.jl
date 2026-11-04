# Discrete flow rate of the parabolic inlets between half-way walls (fix for
# v0.7). The channel walls sit half a spacing outside the first and last open
# rows, so the parabola must vanish there, and its row sum (the discrete flow
# rate) must equal H * u_mean exactly. Before the fix the zeros were put on
# rows 1 and H, which made the delivered flow rate low by 1/(H - 1).
#
# All gates are analytical identities (round-off only), CPU Float64 unless
# stated. Runtime: a few seconds.

using Test
using Kraken

@testset "Inlet flow rate (half-way walls)" begin
    rel(a, b) = abs(a - b) / abs(b)
    # Quadratic through rows k0, k0+1, k0+2 of `u`, evaluated at the real
    # row coordinate `k` (Lagrange form). The profile is a quadratic in the
    # row index, so this extrapolates it exactly up to round-off.
    function quad_at(u, k0, k)
        a, b, c = u[k0], u[k0 + 1], u[k0 + 2]
        t = k - k0
        return a * (t - 1) * (t - 2) / 2 - b * t * (t - 2) + c * t * (t - 1) / 2
    end

    @testset "half-way Poiseuille profile: flux and wall zeros" begin
        for H in (2, 3, 5, 8, 20, 33, 80, 101, 200, 401), u_mean in (1e-3, 0.02, 1.0)
            u = Kraken._halfway_poiseuille_profile(Float64, H, u_mean)
            @test length(u) == H
            # Discrete flow rate equals H * u_mean (gate 1e-12, round-off).
            @test rel(sum(u), H * u_mean) <= 1e-12
            # Symmetric and positive on every row (no zero on the end rows).
            @test all(>(0), u)
            @test maximum(abs.(u .- reverse(u))) <= 1e-15 * u_mean * H
            if H >= 3
                # Zero on both half-way walls, k = 1/2 and k = H + 1/2.
                @test abs(quad_at(u, 1, 0.5)) <= 1e-12 * u_mean
                @test abs(quad_at(u, H - 2, H + 0.5)) <= 1e-12 * u_mean
            end
        end
        # Float32 output: same values rounded once.
        u32 = Kraken._halfway_poiseuille_profile(Float32, 40, 0.02)
        u64 = Kraken._halfway_poiseuille_profile(Float64, 40, 0.02)
        @test eltype(u32) === Float32
        @test u32 == Float32.(u64)
        @test rel(Float64(sum(u32)), 40 * 0.02) <= 1e-6
        @test_throws ArgumentError Kraken._halfway_poiseuille_profile(Float64, 0, 1.0)
    end

    @testset "cylinder LI-BB inlet (ad_inlet_profile, same helper as run_cylinder_libb_2d)" begin
        u_in = 0.04
        for Ny in (16, 41, 80, 161)
            u = Kraken.ad_inlet_profile(Ny, u_in, :parabolic)
            # u_ref = (2/3) u_in is the mean used in Cd = 2 Fx / (u_ref^2 D).
            @test rel(sum(u), Ny * (2 / 3) * u_in) <= 1e-12
            @test u == Kraken._halfway_poiseuille_profile(Float64, Ny, (2 / 3) * u_in)
            @test abs(quad_at(u, 1, 0.5)) <= 1e-12 * u_in
            @test abs(quad_at(u, Ny - 2, Ny + 0.5)) <= 1e-12 * u_in
        end
        @test Kraken.ad_inlet_profile(10, u_in, :uniform) == fill(u_in, 10)
    end

    @testset "parabolic_face_profile_2d on step and obstacle channels" begin
        u_mean = 0.01
        geoms = (
            backward_facing_step_geometry_2d(; H_in=8, L_up=2, L_down=3),
            backward_facing_step_geometry_2d(; H_in=17, L_up=2, L_down=3),
            contraction_step_geometry_2d(; H_out=6, β_c=4, L_up=2, L_down=3),
            contraction_step_geometry_2d(; H_out=10, β_c=2, L_up=2, L_down=3),
            square_obstacle_channel_geometry_2d(; H=24, side=6, L_up=2, L_down=3),
            Kraken._logfv_cylinder_channel_geometry_2d(; radius=6),
        )
        for geom in geoms, face in (:west, :east)
            open = face === :west ? geom.inlet_open : geom.outlet_open
            i_face = face === :west ? 1 : geom.Nx
            H = length(open)
            u = parabolic_face_profile_2d(geom; face=face, mean_velocity=u_mean)
            @test length(u) == geom.Ny
            @test rel(sum(u[open]), H * u_mean) <= 1e-12
            @test all(iszero, u[setdiff(1:geom.Ny, open)])
            @test all(>(0), u[open])
            @test abs(quad_at(u, first(open), first(open) - 0.5)) <= 1e-12 * u_mean
            @test abs(quad_at(u, last(open) - 2, last(open) + 0.5)) <= 1e-12 * u_mean
            # Premise of the construction: the opening is bounded by half-way
            # walls (q = 1/2 on the links leaving its end rows, south = 5,
            # north = 3 in the D2Q9 ordering), either the domain edge or a
            # solid neighbour. The cylinder channel leaves its domain-edge
            # walls to the kernel's half-way bounce-back, outside q_wall.
            if geom.name !== :cylinder
                @test geom.q_wall[i_face, first(open), 5] == 0.5
                @test geom.q_wall[i_face, last(open), 3] == 0.5
            end
        end
        # Float32 path converts the Float64 values once.
        g = backward_facing_step_geometry_2d(; H_in=8, L_up=2, L_down=3)
        u32 = parabolic_face_profile_2d(g; mean_velocity=0.01, FT=Float32)
        @test eltype(u32) === Float32
        @test u32 == Float32.(parabolic_face_profile_2d(g; mean_velocity=0.01))
    end

    @testset "AD viscoelastic channel profile" begin
        for Ny in (24, 31, 80)
            u = Kraken.ad_ve_poiseuille_profile(Ny, 2e-4)
            @test rel(sum(u), Ny * 2e-4) <= 1e-12
            g = square_obstacle_channel_geometry_2d(; H=Ny, side=iseven(Ny) ? 6 : 5,
                                                    L_up=2, L_down=3)
            @test u == parabolic_face_profile_2d(g; mean_velocity=2e-4)
        end
    end
end
