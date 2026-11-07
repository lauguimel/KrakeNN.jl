using Test
using KrakeNN

# ===========================================================================
# Taylor-Couette flow on a polar O-grid (Week 3.2 validation).
#
# Two concentric cylinders: inner at R_i rotating at Ω_i (CCW), outer
# at R_o stationary. Analytical steady-state azimuthal velocity:
#
#   u_θ(r) = A · r + B / r
#   A = -Ω_i · R_i² / (R_o² − R_i²)
#   B =  Ω_i · R_i² · R_o² / (R_o² − R_i²)
#
# This is the first real physical test of the full curvilinear SLBM
# stack: polar mesh metric + semi-Lagrangian streaming + BGK collision
# + moving-wall Ladd bounce-back (inner cylinder).
# ===========================================================================

function run_taylor_couette(; R_i::Float64=8.0, R_o::Float64=16.0,
                              n_theta::Int=64, n_r::Int=17,
                              ν::Float64=0.1, u_wall::Float64=0.01,
                              steps::Int=5000,
                              interp::Symbol=:bilinear)
    mesh = polar_mesh(; cx=0.0, cy=0.0,
                        r_inner=R_i, r_outer=R_o,
                        n_theta=n_theta, n_r=n_r,
                        FT=Float64)
    geom = build_slbm_geometry(mesh)

    Ω_i = u_wall / R_i
    Nξ, Nη = mesh.Nξ, mesh.Nη

    is_solid = zeros(Bool, Nξ, Nη)
    uw_x = zeros(Float64, Nξ, Nη)
    uw_y = zeros(Float64, Nξ, Nη)

    # Inner cylinder: j=1, rotating at Ω_i
    for i in 1:Nξ
        is_solid[i, 1] = true
        uw_x[i, 1] = -Ω_i * mesh.Y[i, 1]
        uw_y[i, 1] =  Ω_i * mesh.X[i, 1]
    end
    # Outer cylinder: j=Nη, stationary
    for i in 1:Nξ
        is_solid[i, Nη] = true
        # uw_x, uw_y already zero
    end

    f_in = zeros(Float64, Nξ, Nη, 9)
    f_out = similar(f_in)
    ρ = ones(Nξ, Nη); ux = zeros(Nξ, Nη); uy = zeros(Nξ, Nη)
    for j in 1:Nη, i in 1:Nξ, q in 1:9
        f_in[i, j, q] = KrakeNN.equilibrium(D2Q9(), 1.0, 0.0, 0.0, q)
    end

    ω = 1.0 / (3ν + 0.5)
    for _ in 1:steps
        slbm_bgk_moving_step!(f_out, f_in, ρ, ux, uy, is_solid,
                               uw_x, uw_y, geom, ω; interp=interp)
        f_in, f_out = f_out, f_in
    end

    return (mesh=mesh, ρ=ρ, ux=ux, uy=uy, Ω_i=Ω_i)
end

@testset "SLBM Taylor-Couette" begin

    @testset "Analytical u_θ(r) match (Re ≈ 2, half-cell corrected)" begin
        # Halfway bounce-back places the *effective* wall at the midpoint
        # between solid and fluid nodes, so the effective cylinder radii
        # are shifted by Δr/2. Using R_eff in the analytical eliminates
        # the systematic offset; the remaining error is discretisation +
        # interpolation.
        R_i, R_o = 8.0, 16.0
        n_theta, n_r = 128, 33
        Δr = (R_o - R_i) / (n_r - 1)
        R_i_eff = R_i + Δr / 2
        R_o_eff = R_o - Δr / 2
        u_wall = 0.01
        out = run_taylor_couette(; R_i=R_i, R_o=R_o,
                                   n_theta=n_theta, n_r=n_r,
                                   ν=0.1, u_wall=u_wall,
                                   steps=10000)

        Ω_eff = u_wall / R_i_eff
        A = -Ω_eff * R_i_eff^2 / (R_o_eff^2 - R_i_eff^2)
        B =  Ω_eff * R_i_eff^2 * R_o_eff^2 / (R_o_eff^2 - R_i_eff^2)
        u_θ_ana(r) = A * r + B / r

        mesh = out.mesh
        errors = Float64[]
        refs = Float64[]
        for j in 2:mesh.Nη - 1
            X = mesh.X[1, j]; Y = mesh.Y[1, j]
            r = sqrt(X^2 + Y^2)
            ut_num = (-Y * out.ux[1, j] + X * out.uy[1, j]) / r
            ut_ref = u_θ_ana(r)
            push!(errors, abs(ut_num - ut_ref))
            push!(refs, abs(ut_ref))
        end
        L2_rel = sqrt(sum(errors .^ 2) / sum(refs .^ 2))
        Linf_rel = maximum(errors) / maximum(refs)
        @info "Taylor-Couette (128×33, half-cell corrected): L2 = $(round(L2_rel, digits=4)), L∞ = $(round(Linf_rel, digits=4))"

        @test all(isfinite.(out.ux))
        @test all(isfinite.(out.uy))
        # Current state: the polar mesh has non-uniform cell sizes
        # (radial Δr = 0.25 vs angular arc 0.39–0.78), so the effective
        # LBM viscosity varies across the mesh. Per-cell τ rescaling
        # (Krämer 2017 §4) would reduce the error further; shipped as
        # Week 4+ follow-up. These tolerances reflect the un-corrected
        # baseline that already validates the full pipeline qualitatively.
        @test L2_rel < 0.25
        @test Linf_rel < 0.20

        # Sign: CCW inner wall ⇒ u_θ > 0 everywhere in the gap
        for j in 2:mesh.Nη - 1
            X = mesh.X[1, j]; Y = mesh.Y[1, j]
            r = sqrt(X^2 + Y^2)
            ut_num = (-Y * out.ux[1, j] + X * out.uy[1, j]) / r
            @test ut_num > 0
        end
    end

    @testset "Biquadratic stable with near-wall fallback" begin
        # Biquadratic Lagrange weights are signed and can amplify
        # near-wall discontinuities (bounce-back populations). The
        # biquadratic_f helper falls back to bilinear when any of the
        # 9 stencil cells is solid, which keeps the step stable.
        # On Taylor-Couette biquadratic does NOT beat bilinear because
        # the profile is smooth through the whole gap — the gain from
        # higher-order interpolation applies in the bulk, where the
        # field is nearly uniform in θ, so the benefit is small while
        # the two near-wall rows still use bilinear. For problems with
        # sharp features far from walls (shear layers, vortex cores)
        # biquadratic helps measurably — see Taylor-Green stretched.
        R_i, R_o = 8.0, 16.0
        out = run_taylor_couette(; R_i=R_i, R_o=R_o,
                                   n_theta=128, n_r=33,
                                   ν=0.1, u_wall=0.01,
                                   steps=10000, interp=:biquadratic)
        @test all(isfinite.(out.ux))
        @test all(isfinite.(out.uy))
        @test maximum(sqrt.(out.ux .^ 2 .+ out.uy .^ 2)) < 0.02  # bounded
    end

    @testset "Zero rotation → zero flow (sanity)" begin
        out = run_taylor_couette(; R_i=8.0, R_o=16.0,
                                   n_theta=32, n_r=13,
                                   ν=0.1, u_wall=0.0,
                                   steps=500)
        @test all(abs.(out.ux) .< 1e-8)
        @test all(abs.(out.uy) .< 1e-8)
    end

end
