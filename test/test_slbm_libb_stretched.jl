using Test
using KrakeNN
using Statistics: mean

# ===========================================================================
# SLBM + LI-BB on stretched Cartesian mesh with embedded cylinder (WP1).
#
# This is the correct use-case for SLBM + IBB: the mesh is NOT body-fitted.
# The cylinder cuts through cells at arbitrary positions, giving q_w ∈ (0,1).
# The stretching concentrates cells near the cylinder for boundary-layer
# resolution, while LI-BB handles the sub-cell wall position.
# ===========================================================================

function setup_stretched_cylinder(; Nx::Int=81, Ny::Int=41,
                                    Lx::Float64=10.0, Ly::Float64=5.0,
                                    cx::Float64=2.5, cy::Float64=2.5,
                                    R::Float64=0.5,
                                    x_stretch::Float64=0.0,
                                    y_stretch::Float64=0.0)
    mesh = stretched_box_mesh(; x_min=0.0, x_max=Lx, y_min=0.0, y_max=Ly,
                                Nx=Nx, Ny=Ny,
                                x_stretch=x_stretch, y_stretch=y_stretch,
                                x_stretch_dir=:both, y_stretch_dir=:both,
                                FT=Float64)

    Nξ, Nη = mesh.Nξ, mesh.Nη
    is_solid = zeros(Bool, Nξ, Nη)

    @inbounds for j in 1:Nη, i in 1:Nξ
        dx = mesh.X[i, j] - cx
        dy = mesh.Y[i, j] - cy
        if dx * dx + dy * dy ≤ R * R
            is_solid[i, j] = true
        end
    end

    q_wall, uw_link_x, uw_link_y = precompute_q_wall_slbm_cylinder_2d(
        mesh, is_solid, cx, cy, R; omega_inner=0.0)

    return (; mesh, is_solid, q_wall, uw_link_x, uw_link_y)
end

@testset "SLBM + LI-BB stretched cylinder" begin

    @testset "q_wall values are in (0, 1) on non-body-fitted mesh" begin
        s = setup_stretched_cylinder(; Nx=81, Ny=41, x_stretch=1.5, y_stretch=1.5)
        qw = s.q_wall

        n_cut = sum(qw .> 0)
        @test n_cut > 0

        cut_values = qw[qw .> 0]
        @info "Stretched cylinder: $(n_cut) cut links, q_w range = [$(round(minimum(cut_values), digits=3)), $(round(maximum(cut_values), digits=3))]"

        # On a non-body-fitted mesh, q_w should NOT be exactly 1.0 for most links
        n_degenerate = sum(cut_values .≥ 0.999)
        frac_degenerate = n_degenerate / n_cut
        @info "  Degenerate (q_w ≥ 0.999): $(n_degenerate)/$(n_cut) = $(round(100*frac_degenerate, digits=1))%"
        @test frac_degenerate < 0.3

        # q_w should span a range, not be concentrated at 0.5
        @test minimum(cut_values) < 0.4
        @test maximum(cut_values) > 0.6
    end

    @testset "Kernel stability (no BCs, quiescent)" begin
        s = setup_stretched_cylinder(; Nx=61, Ny=31, x_stretch=1.0, y_stretch=1.0)
        mesh = s.mesh
        Nξ, Nη = mesh.Nξ, mesh.Nη
        geom = build_slbm_geometry(mesh)

        f_in = zeros(Float64, Nξ, Nη, 9)
        f_out = similar(f_in)
        ρ = ones(Nξ, Nη); ux = zeros(Nξ, Nη); uy = zeros(Nξ, Nη)
        for j in 1:Nη, i in 1:Nξ, q in 1:9
            f_in[i, j, q] = KrakeNN.equilibrium(D2Q9(), 1.0, 0.0, 0.0, q)
        end

        ν = 0.1
        for _ in 1:500
            slbm_trt_libb_step!(f_out, f_in, ρ, ux, uy, s.is_solid,
                                 s.q_wall, s.uw_link_x, s.uw_link_y,
                                 geom, ν)
            f_in, f_out = f_out, f_in
        end

        @test all(isfinite.(ρ))
        @test all(isfinite.(ux))
        @test all(isfinite.(uy))
        # Quiescent: velocities should stay near zero
        max_u = maximum(sqrt.(ux .^ 2 .+ uy .^ 2))
        @info "Quiescent stability: max |u| = $(round(max_u, sigdigits=3)) after 500 steps"
        @test max_u < 0.01
    end

    @testset "Stretched vs uniform: different q_w distribution" begin
        s_uniform = setup_stretched_cylinder(; Nx=61, Ny=31,
                                               x_stretch=0.0, y_stretch=0.0)
        s_stretched = setup_stretched_cylinder(; Nx=61, Ny=31,
                                                  x_stretch=2.0, y_stretch=2.0)

        n_cut_uniform = sum(s_uniform.q_wall .> 0)
        n_cut_stretched = sum(s_stretched.q_wall .> 0)
        @info "Cut links: uniform=$n_cut_uniform, stretched=$n_cut_stretched"

        # Both should have cut links (cylinder cuts through mesh cells)
        @test n_cut_uniform > 0
        @test n_cut_stretched > 0
        # Stretching changes the q_w distribution (different cell sizes near body)
        qw_uni = s_uniform.q_wall[s_uniform.q_wall .> 0]
        qw_str = s_stretched.q_wall[s_stretched.q_wall .> 0]
        @test abs(mean(qw_uni) - mean(qw_str)) > 0.01 || n_cut_uniform != n_cut_stretched
    end
end
