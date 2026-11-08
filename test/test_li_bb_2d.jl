using Test
using KrakeNN

@testset "LI-BB (interpolated bounce-back) + TRT" begin

    @testset "precompute_q_wall_cylinder — analytical geometry" begin
        # Cylinder at (15.5, 15.5) radius 4.0 on 32×32 grid.
        Nx = Ny = 32; cx = cy = 15.5; R = 4.0
        qw, is_solid = precompute_q_wall_cylinder(Nx, Ny, cx, cy, R)
        @test size(qw) == (Nx, Ny, 9)
        @test size(is_solid) == (Nx, Ny)

        # At (1,1) — far from cylinder: no cuts anywhere on the 8 links
        @test all(qw[1, 1, 2:9] .== 0.0)
        @test !is_solid[1, 1]

        # At the cylinder centre's nearest lattice node (16, 16) — it's
        # inside the cylinder (distance 0.707 < R=4)
        @test is_solid[16, 16]

        # A node right next to the cylinder should have at least one cut
        # link on its side of the cylinder
        fluid_adjacent = findall(i -> !is_solid[i[1], i[2]] &&
                                        any(qw[i[1], i[2], 2:9] .> 0),
                                  CartesianIndices((Nx, Ny)))
        @test !isempty(fluid_adjacent)

        # All q_wall values must be in (0, 1]
        cut_vals = filter(v -> v > 0, qw)
        @test all(0 .< cut_vals .≤ 1)

        # Invariance check: link (i,j,q=2) cut at q ⇔ link (i+1,j,q=4)
        # cut at (1−q) (when both ends are fluid). Verify on axis links.
        for j in 1:Ny, i in 1:Nx-1
            q_e = qw[i,   j, 2]   # east link out of (i, j)
            q_w = qw[i+1, j, 4]   # west link out of (i+1, j)
            if q_e > 0 && q_w > 0 && !is_solid[i, j] && !is_solid[i+1, j]
                @test q_e + q_w ≈ 1.0 atol=1e-10
            end
        end
    end

    @testset "Zero flow stays zero with cylinder obstacle" begin
        Nx = Ny = 32
        ν = 0.1
        qw, is_solid = precompute_q_wall_cylinder(Nx, Ny, 15.5, 15.5, 4.0)
        uw_x, uw_y = wall_velocity_rotating_cylinder(qw, 15.5, 15.5, 0.0)

        f_in = zeros(Float64, Nx, Ny, 9)
        for j in 1:Ny, i in 1:Nx, q in 1:9
            f_in[i, j, q] = KrakeNN.equilibrium(D2Q9(), 1.0, 0.0, 0.0, q)
        end
        f_out = similar(f_in)
        ρ = ones(Nx, Ny); ux = zeros(Nx, Ny); uy = zeros(Nx, Ny)

        for _ in 1:200
            fused_trt_libb_step!(f_out, f_in, ρ, ux, uy, is_solid,
                                  qw, uw_x, uw_y, Nx, Ny, ν)
            f_in, f_out = f_out, f_in
        end
        # Zero initial flow + stationary wall ⇒ nothing should develop
        fluid = .!is_solid
        @test maximum(abs.(ux[fluid])) < 1e-10
        @test maximum(abs.(uy[fluid])) < 1e-10
        @test maximum(abs.(ρ[fluid] .- 1.0)) < 1e-10
    end

    @testset "Rotating cylinder drives a swirl (sanity)" begin
        # Small box with a rotating cylinder at the centre. The flow
        # should develop a CCW swirl around the cylinder and decay
        # toward the walls. Not an analytical benchmark — a smoke test
        # that the moving-wall correction produces positive rotation.
        Nx = Ny = 48
        cx = cy = 23.5; R = 4.0
        ν = 0.1
        Ω = 0.002   # low-Mach rotation rate

        qw, is_solid = precompute_q_wall_cylinder(Nx, Ny, cx, cy, R)
        uw_x, uw_y = wall_velocity_rotating_cylinder(qw, cx, cy, Ω)

        f_in = zeros(Float64, Nx, Ny, 9)
        for j in 1:Ny, i in 1:Nx, q in 1:9
            f_in[i, j, q] = KrakeNN.equilibrium(D2Q9(), 1.0, 0.0, 0.0, q)
        end
        f_out = similar(f_in)
        ρ = ones(Nx, Ny); ux = zeros(Nx, Ny); uy = zeros(Nx, Ny)

        for _ in 1:2000
            fused_trt_libb_step!(f_out, f_in, ρ, ux, uy, is_solid,
                                  qw, uw_x, uw_y, Nx, Ny, ν)
            f_in, f_out = f_out, f_in
        end

        # Sample a ring at r ≈ R + 2 and check the azimuthal velocity
        # (u_θ = (−y·u_x + x·u_y)/r) is positive
        ut_samples = Float64[]
        for θ in 0:π/8:2π
            xq = cx + (R + 2) * cos(θ)
            yq = cy + (R + 2) * sin(θ)
            i = round(Int, xq) + 1; j = round(Int, yq) + 1
            if 1 ≤ i ≤ Nx && 1 ≤ j ≤ Ny && !is_solid[i, j]
                X = Float64(i - 1) - cx; Y = Float64(j - 1) - cy
                r = sqrt(X^2 + Y^2)
                push!(ut_samples, (-Y * ux[i, j] + X * uy[i, j]) / r)
            end
        end

        mean_ut = sum(ut_samples) / length(ut_samples)
        max_u = maximum(sqrt.(ux .^ 2 .+ uy .^ 2))
        @info "Rotating cylinder: mean u_θ ≈ $(round(mean_ut, digits=6)), max |u| = $(round(max_u, digits=6))"
        @test all(isfinite.(ux))
        @test mean_ut > 0
        @test max_u < 0.02   # bounded, not diverging (u_wall = Ω·R = 0.008)
    end

end
