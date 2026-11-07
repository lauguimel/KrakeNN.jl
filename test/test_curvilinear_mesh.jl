using Test
using KrakeNN

# ===========================================================================
# Curvilinear mesh + metric computation tests (Week 1, v0.2 SLBM path).
#
# Validates: ForwardDiff-based metric computation against analytical
# derivatives, generator behavior, mesh validation guards.
# ===========================================================================

@testset "Curvilinear mesh" begin

    @testset "build_mesh — identity mapping (Cartesian sanity)" begin
        # Identity: X = ξ, Y = η. Metric should be the identity, J = 1.
        mesh = KrakeNN.build_mesh((ξ, η) -> (ξ, η);
                                  Nξ=8, Nη=6,
                                  type=:custom, FT=Float64)
        @test mesh.Nξ == 8
        @test mesh.Nη == 6
        @test all(mesh.dXdξ .≈ 1.0)
        @test all(mesh.dYdη .≈ 1.0)
        @test all(abs.(mesh.dXdη) .< 1e-14)
        @test all(abs.(mesh.dYdξ) .< 1e-14)
        @test all(mesh.J .≈ 1.0)
    end

    @testset "build_mesh — affine scaling (analytical match)" begin
        # X = 3ξ + 1, Y = 5η + 2. Expected: dXdξ=3, dYdη=5, J=15.
        mesh = KrakeNN.build_mesh((ξ, η) -> (3ξ + 1, 5η + 2);
                                  Nξ=4, Nη=4, FT=Float64)
        @test all(mesh.dXdξ .≈ 3.0)
        @test all(mesh.dYdη .≈ 5.0)
        @test all(mesh.J .≈ 15.0)
        @test mesh.X[1, 1] ≈ 1.0
        @test mesh.X[end, 1] ≈ 4.0  # ξ=1 → X = 3·1 + 1
        @test mesh.Y[1, end] ≈ 7.0  # η=1 → Y = 5·1 + 2
    end

    @testset "build_mesh — non-orthogonal mapping (analytical match)" begin
        # X = ξ + 0.3η, Y = 0.2ξ + η. Cross-derivatives non-zero.
        mesh = KrakeNN.build_mesh((ξ, η) -> (ξ + 0.3η, 0.2ξ + η);
                                  Nξ=5, Nη=5, FT=Float64)
        @test all(mesh.dXdξ .≈ 1.0)
        @test all(mesh.dXdη .≈ 0.3)
        @test all(mesh.dYdξ .≈ 0.2)
        @test all(mesh.dYdη .≈ 1.0)
        @test all(mesh.J .≈ 1.0 - 0.3 * 0.2)  # = 0.94
    end

    @testset "polar_mesh — basic geometry" begin
        m = polar_mesh(; cx=0.0, cy=0.0,
                        r_inner=1.0, r_outer=3.0,
                        n_theta=64, n_r=16,
                        FT=Float64)
        @test m.type == :polar
        @test m.periodic_ξ == true
        @test m.periodic_η == false

        # Inner ring: every (i, 1) sits on r = r_inner = 1
        for i in 1:m.Nξ
            r = sqrt(m.X[i, 1]^2 + m.Y[i, 1]^2)
            @test r ≈ 1.0 atol=1e-12
        end
        # Outer ring: every (i, end) sits on r = r_outer = 3
        for i in 1:m.Nξ
            r = sqrt(m.X[i, end]^2 + m.Y[i, end]^2)
            @test r ≈ 3.0 atol=1e-12
        end
    end

    @testset "polar_mesh — analytical metric at one node" begin
        # For X = r·cos(θ), Y = r·sin(θ) with r linear in η,
        # at ξ=0, θ=0: dX/dξ = -2π·r·sin(0) = 0; dY/dξ = 2π·r·cos(0) = 2π·r.
        # dr/dη = (r_outer − r_inner) = Δr (uniform); dX/dη = Δr·cos(0) = Δr.
        m = polar_mesh(; r_inner=1.0, r_outer=3.0,
                        n_theta=128, n_r=32, FT=Float64)
        # Inner-wall, θ=0 node
        i, j = 1, 1
        Δr = 3.0 - 1.0
        r_at = 1.0
        @test m.dXdξ[i, j] ≈ 0.0 atol=1e-10
        @test m.dYdξ[i, j] ≈ 2π * r_at atol=1e-10
        @test m.dXdη[i, j] ≈ Δr atol=1e-10
        @test m.dYdη[i, j] ≈ 0.0 atol=1e-10
        # J = dXdξ · dYdη − dXdη · dYdξ = 0 − Δr · 2π·r = −2π·r·Δr
        # Sign: ξ runs CCW so positive J convention requires we check |J|.
        # Our convention: J should be > 0; if not, the radial direction
        # is reversed. Confirm: J = 2π·r·Δr in absolute value.
        @test abs(m.J[i, j]) ≈ 2π * r_at * Δr atol=1e-10
    end

    @testset "polar_mesh — Jacobian non-degenerate (no folds)" begin
        # Polar (θ, r) → (X, Y) is left-handed in physical space, so J<0
        # everywhere. What matters is |J| > 0 with consistent sign.
        m = polar_mesh(; r_inner=0.5, r_outer=5.0,
                        n_theta=256, n_r=64,
                        r_stretch=2.0, FT=Float64)
        @test minimum(abs.(m.J)) > 0
        @test minimum(m.J) < 0  # consistent (negative) sign
        @test maximum(m.J) < 0
    end

    @testset "polar_mesh — radial stretching concentrates near inner wall" begin
        m = polar_mesh(; r_inner=1.0, r_outer=10.0,
                        n_theta=8, n_r=33,
                        r_stretch=3.0, FT=Float64)
        # At ξ=0, radii along the radial line:
        rs = [sqrt(m.X[1, j]^2 + m.Y[1, j]^2) for j in 1:m.Nη]
        # Distance from inner wall to first interior node should be much
        # smaller than from the second-to-last to the outer wall.
        Δr_inner = rs[2] - rs[1]
        Δr_outer = rs[end] - rs[end-1]
        @test Δr_inner < 0.5 * Δr_outer
    end

    @testset "stretched_box_mesh — uniform reduces to Cartesian" begin
        m = stretched_box_mesh(; x_min=0.0, x_max=2.0,
                                 y_min=0.0, y_max=1.0,
                                 Nx=11, Ny=6, FT=Float64)
        Δx = 2.0 / 10
        Δy = 1.0 / 5
        @test all(m.dXdξ .≈ 2.0)   # X = 2ξ → dX/dξ = 2
        @test all(m.dYdη .≈ 1.0)   # Y = η → dY/dη = 1
        @test all(m.J .≈ 2.0)       # = (x_max-x_min) · (y_max-y_min)
        # Node spacing in physical x: Δx
        @test m.X[2, 1] - m.X[1, 1] ≈ Δx
        @test m.Y[1, 2] - m.Y[1, 1] ≈ Δy
    end

    @testset "stretched_box_mesh — :both stretching is symmetric" begin
        m = stretched_box_mesh(; x_min=0.0, x_max=1.0,
                                 y_min=0.0, y_max=1.0,
                                 Nx=21, Ny=21,
                                 y_stretch=2.0, y_stretch_dir=:both,
                                 FT=Float64)
        ys = m.Y[1, :]
        # Symmetric about the midline 0.5
        for j in 1:11
            @test ys[j] + ys[end - j + 1] ≈ 1.0 atol=1e-12
        end
        # Spacing tighter at the walls than at the middle
        Δy_wall = ys[2] - ys[1]
        Δy_mid  = ys[12] - ys[11]
        @test Δy_wall < Δy_mid
    end

    @testset "stretched_box_mesh — :left vs :right" begin
        ml = stretched_box_mesh(; x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0,
                                  Nx=11, Ny=4,
                                  x_stretch=2.0, x_stretch_dir=:left)
        mr = stretched_box_mesh(; x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0,
                                  Nx=11, Ny=4,
                                  x_stretch=2.0, x_stretch_dir=:right)
        # :left clusters near x=0, :right clusters near x=1.
        @test ml.X[2, 1] - ml.X[1, 1] < mr.X[2, 1] - mr.X[1, 1]
        @test mr.X[end, 1] - mr.X[end - 1, 1] < ml.X[end, 1] - ml.X[end - 1, 1]
    end

    @testset "Float32 path (Metal-style)" begin
        m = polar_mesh(; r_inner=1.0f0, r_outer=4.0f0,
                        n_theta=32, n_r=8, FT=Float32)
        @test eltype(m.X) === Float32
        @test eltype(m.J) === Float32
        @test minimum(abs.(m.J)) > 0.0f0
    end

    @testset "validate_mesh — rejects folded mapping" begin
        # Fold: X = ξ², Y = η. dX/dξ = 0 at ξ=0 → J = 0 there.
        @test_throws ErrorException KrakeNN.build_mesh((ξ, η) -> (ξ^2, η);
                                                      Nξ=4, Nη=4, FT=Float64)
    end

    @testset "cell_area — integral over polar mesh ≈ annulus area" begin
        r_in, r_out = 1.0, 3.0
        m = polar_mesh(; r_inner=r_in, r_outer=r_out,
                        n_theta=512, n_r=128, FT=Float64)
        total = 0.0
        for j in 1:m.Nη, i in 1:m.Nξ
            total += abs(cell_area(m, i, j))
        end
        annulus = π * (r_out^2 - r_in^2)
        # Midpoint-rule on non-uniform polar — convergence is ~1/N²
        @test isapprox(total, annulus; rtol=1e-2)
    end

    @testset "domain_extent" begin
        m = stretched_box_mesh(; x_min=-2.0, x_max=5.0, y_min=0.0, y_max=3.0,
                                 Nx=8, Ny=8)
        xmin, xmax, ymin, ymax = domain_extent(m)
        @test xmin ≈ -2.0
        @test xmax ≈ 5.0
        @test ymin ≈ 0.0
        @test ymax ≈ 3.0
    end

end
