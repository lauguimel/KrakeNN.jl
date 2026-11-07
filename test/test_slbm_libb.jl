using Test
using KrakeNN

# ===========================================================================
# SLBM + LI-BB Taylor-Couette validation (WP1).
#
# Same geometry as test_slbm_taylor_couette.jl but using the new
# slbm_trt_libb_step! kernel with sub-cell q_wall computed in physical
# space. The LI-BB pre-phase + TRT Λ=3/16 should give better accuracy
# than plain Ladd BB, especially for the wall position.
# ===========================================================================

function run_taylor_couette_libb(; R_i::Float64=8.0, R_o::Float64=16.0,
                                   n_theta::Int=64, n_r::Int=17,
                                   ν::Float64=0.1, u_wall::Float64=0.01,
                                   steps::Int=5000)
    mesh = polar_mesh(; cx=0.0, cy=0.0,
                        r_inner=R_i, r_outer=R_o,
                        n_theta=n_theta, n_r=n_r,
                        FT=Float64)
    geom = build_slbm_geometry(mesh)

    Ω_i = u_wall / R_i
    Nξ, Nη = mesh.Nξ, mesh.Nη

    is_solid = zeros(Bool, Nξ, Nη)
    for i in 1:Nξ
        is_solid[i, 1] = true
        is_solid[i, Nη] = true
    end

    # Precompute q_wall for inner cylinder (rotating) in physical space
    q_wall_inner, uwx_inner, uwy_inner = precompute_q_wall_slbm_cylinder_2d(
        mesh, is_solid, 0.0, 0.0, R_i; omega_inner=Ω_i)

    # Precompute q_wall for outer cylinder (stationary)
    q_wall_outer, uwx_outer, uwy_outer = precompute_q_wall_slbm_cylinder_2d(
        mesh, is_solid, 0.0, 0.0, R_o; omega_inner=0.0)

    # Merge: a link can be cut by either wall
    q_wall    = q_wall_inner .+ q_wall_outer
    uw_link_x = uwx_inner .+ uwx_outer
    uw_link_y = uwy_inner .+ uwy_outer

    f_in = zeros(Float64, Nξ, Nη, 9)
    f_out = similar(f_in)
    ρ = ones(Nξ, Nη); ux = zeros(Nξ, Nη); uy = zeros(Nξ, Nη)
    for j in 1:Nη, i in 1:Nξ, q in 1:9
        f_in[i, j, q] = KrakeNN.equilibrium(D2Q9(), 1.0, 0.0, 0.0, q)
    end

    for _ in 1:steps
        slbm_trt_libb_step!(f_out, f_in, ρ, ux, uy, is_solid,
                             q_wall, uw_link_x, uw_link_y,
                             geom, ν)
        f_in, f_out = f_out, f_in
    end

    return (mesh=mesh, ρ=ρ, ux=ux, uy=uy, Ω_i=Ω_i)
end

@testset "SLBM + LI-BB Taylor-Couette" begin

    @testset "LI-BB vs plain BB improvement" begin
        R_i, R_o = 8.0, 16.0
        n_theta, n_r = 128, 33
        u_wall = 0.01

        out = run_taylor_couette_libb(; R_i=R_i, R_o=R_o,
                                        n_theta=n_theta, n_r=n_r,
                                        ν=0.1, u_wall=u_wall,
                                        steps=10000)

        # With LI-BB, the effective wall positions are determined by the
        # physical-space q_w, not by a half-cell offset. Use the true
        # cylinder radii for the analytical reference.
        Ω_eff = out.Ω_i
        A = -Ω_eff * R_i^2 / (R_o^2 - R_i^2)
        B =  Ω_eff * R_i^2 * R_o^2 / (R_o^2 - R_i^2)
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
        @info "SLBM+LI-BB Taylor-Couette (128×33): L2 = $(round(L2_rel, digits=4)), L∞ = $(round(Linf_rel, digits=4))"

        @test all(isfinite.(out.ux))
        @test all(isfinite.(out.uy))
        # LI-BB should beat plain BB (which was L2 < 0.25)
        @test L2_rel < 0.15
        @test Linf_rel < 0.12
    end

    @testset "Stable and finite (coarse)" begin
        out = run_taylor_couette_libb(; R_i=8.0, R_o=16.0,
                                        n_theta=32, n_r=13,
                                        ν=0.1, u_wall=0.01,
                                        steps=2000)
        @test all(isfinite.(out.ux))
        @test all(isfinite.(out.uy))
        @test all(isfinite.(out.ρ))
    end

    @testset "Zero rotation → zero flow" begin
        out = run_taylor_couette_libb(; R_i=8.0, R_o=16.0,
                                        n_theta=32, n_r=13,
                                        ν=0.1, u_wall=0.0,
                                        steps=500)
        @test all(abs.(out.ux) .< 1e-8)
        @test all(abs.(out.uy) .< 1e-8)
    end
end
