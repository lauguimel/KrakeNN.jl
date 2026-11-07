using Test
using KrakeNN

@testset "Grid Refinement" begin

    @testset "Rescaled omega" begin
        # omega_coarse = 1.0 (tau=1) -> tau_fine = 2*(1-0.5)+0.5 = 1.5 -> omega_fine = 2/3
        @test rescaled_omega(1.0, 2) ≈ 2.0 / 3.0
        # omega_coarse = 0.5 (tau=2) -> tau_fine = 2*(2-0.5)+0.5 = 3.5 -> omega_fine = 2/7
        @test rescaled_omega(0.5, 2) ≈ 2.0 / 7.0
        # ratio=1 should give same omega
        @test rescaled_omega(0.8, 1) ≈ 0.8
    end

    @testset "Rescaling factors" begin
        ω_c = 1.0
        ω_f = rescaled_omega(ω_c, 2)
        α_c2f = rescaling_factor_c2f(ω_c, ω_f, 2)
        α_f2c = rescaling_factor_f2c(ω_c, ω_f, 2)
        # Round-trip: α_c2f * α_f2c = 1
        @test α_c2f * α_f2c ≈ 1.0
    end

    @testset "Create patch" begin
        Nx_base, Ny_base = 20, 20
        dx_base = 0.05  # Lx = 1.0
        ω_base = 1.0
        region = (0.2, 0.2, 0.8, 0.8)  # center of domain

        patch = create_patch("center", 1, 2, region,
                            Nx_base, Ny_base, dx_base, ω_base)

        # Check dimensions: region covers 12 coarse cells in each direction -> 24 fine + 4 ghost = 28
        @test patch.Nx_inner == 24  # (floor(0.2/0.05)+1=5 to ceil(0.8/0.05)=16) -> 12 * 2
        @test patch.Ny_inner == 24
        @test patch.Nx == patch.Nx_inner + 2 * patch.n_ghost
        @test patch.Ny == patch.Ny_inner + 2 * patch.n_ghost
        @test patch.n_ghost == 2
        @test patch.dx ≈ dx_base / 2
        @test patch.omega ≈ rescaled_omega(ω_base, 2)
        @test patch.level == 1
        @test patch.ratio == 2

        # f arrays should be initialized to equilibrium
        f_cpu = Array(patch.f_in)
        # Weight for rest population (q=1)
        @test f_cpu[5, 5, 1] ≈ 4.0 / 9.0 atol=1e-10

        @info "Patch created: $(patch.Nx)×$(patch.Ny) (inner $(patch.Nx_inner)×$(patch.Ny_inner))"
    end

    @testset "RefinedDomain construction" begin
        Nx_base, Ny_base = 40, 40
        dx_base = 0.025
        ω_base = 1.0

        patch1 = create_patch("wake", 1, 2, (0.2, 0.1, 0.8, 0.9),
                              Nx_base, Ny_base, dx_base, ω_base)

        domain = create_refined_domain(Nx_base, Ny_base, dx_base, ω_base,
                                       [patch1])

        @test length(domain.patches) == 1
        @test domain.base_Nx == 40
        @test domain.base_Ny == 40
        @test domain.parent_of[1] == 0  # parent is base grid
    end

    @testset "Prolongation ghost fill (uniform field)" begin
        # Test that prolongation preserves a uniform field exactly
        Nx_c, Ny_c = 20, 20
        f_c = zeros(Float64, Nx_c, Ny_c, 9)
        rho_c = ones(Float64, Nx_c, Ny_c)
        ux_c = fill(0.01, Nx_c, Ny_c)
        uy_c = zeros(Float64, Nx_c, Ny_c)

        # Fill coarse f with equilibrium
        for j in 1:Ny_c, i in 1:Nx_c
            for q in 1:9
                f_c[i, j, q] = equilibrium(D2Q9(), 1.0, 0.01, 0.0, q)
            end
        end

        ω_c = 1.0
        ω_f = rescaled_omega(ω_c, 2)
        region = (0.2, 0.2, 0.6, 0.6)
        patch = create_patch("test", 1, 2, region, Nx_c, Ny_c, 0.05, ω_c)

        # Fill ghost
        prolongate_f_rescaled_2d!(
            patch.f_in, f_c, rho_c, ux_c, uy_c,
            2, patch.Nx_inner, patch.Ny_inner,
            patch.n_ghost, first(patch.parent_i_range), first(patch.parent_j_range),
            Nx_c, Ny_c, ω_c, ω_f)

        # Check ghost cells: should be equilibrium with rho=1, ux=0.01, uy=0
        f_fine = Array(patch.f_in)
        for q in 1:9
            feq = equilibrium(D2Q9(), 1.0, 0.01, 0.0, q)
            # Ghost corners
            @test f_fine[1, 1, q] ≈ feq atol=1e-8
            @test f_fine[2, 2, q] ≈ feq atol=1e-8
        end

        @info "Prolongation: uniform field preserved in ghost cells"
    end

    @testset "Restriction round-trip" begin
        # Create a patch with uniform equilibrium everywhere
        Nx_c, Ny_c = 20, 20
        dx_c = 0.05
        ω_c = 1.0
        ω_f = rescaled_omega(ω_c, 2)
        region = (0.2, 0.2, 0.6, 0.6)
        patch = create_patch("test", 1, 2, region, Nx_c, Ny_c, dx_c, ω_c)

        # Set uniform equilibrium on fine patch (rho=1.0, ux=0.02, uy=0.0)
        f_fine = Array(patch.f_in)
        rho_f = Array(patch.rho)
        ux_f = Array(patch.ux)
        uy_f = Array(patch.uy)
        for j in 1:patch.Ny, i in 1:patch.Nx
            rho_f[i, j] = 1.0
            ux_f[i, j] = 0.02
            uy_f[i, j] = 0.0
            for q in 1:9
                f_fine[i, j, q] = equilibrium(D2Q9(), 1.0, 0.02, 0.0, q)
            end
        end
        copyto!(patch.f_in, f_fine)
        copyto!(patch.rho, rho_f)
        copyto!(patch.ux, ux_f)
        copyto!(patch.uy, uy_f)

        # Coarse arrays
        f_c = zeros(Float64, Nx_c, Ny_c, 9)
        rho_c = ones(Float64, Nx_c, Ny_c)
        ux_c = zeros(Float64, Nx_c, Ny_c)
        uy_c = zeros(Float64, Nx_c, Ny_c)

        # Restrict fine -> coarse
        Nx_overlap = length(patch.parent_i_range)
        Ny_overlap = length(patch.parent_j_range)
        restrict_f_rescaled_2d!(
            f_c, rho_c, ux_c, uy_c,
            patch.f_in, patch.rho, patch.ux, patch.uy,
            2, patch.n_ghost,
            first(patch.parent_i_range), first(patch.parent_j_range),
            Nx_overlap, Ny_overlap,
            ω_c, ω_f)

        # Check restricted macroscopic: should match fine values
        i_mid = first(patch.parent_i_range) + Nx_overlap ÷ 2
        j_mid = first(patch.parent_j_range) + Ny_overlap ÷ 2
        @test rho_c[i_mid, j_mid] ≈ 1.0 atol=1e-10
        @test ux_c[i_mid, j_mid] ≈ 0.02 atol=1e-10
        @test uy_c[i_mid, j_mid] ≈ 0.0 atol=1e-10

        # Check restricted f: should be equilibrium (since fine f_neq = 0)
        for q in 1:9
            feq = equilibrium(D2Q9(), 1.0, 0.02, 0.0, q)
            @test f_c[i_mid, j_mid, q] ≈ feq atol=1e-10
        end

        @info "Restriction: uniform field round-trip OK"
    end

    @testset ".krk Refine parsing" begin
        setup = parse_kraken("""
            Simulation refine_test D2Q9
            Domain L = 2.0 x 1.0  N = 40 x 20
            Physics nu = 0.1

            Refine wake { region = [0.5, 0.2, 1.5, 0.8], ratio = 2 }
            Refine bl   { region = [0.3, 0.3, 0.7, 0.7], ratio = 2, parent = wake }

            Boundary north wall
            Boundary south wall
            Boundary east  wall
            Boundary west  wall

            Run 100 steps
        """)

        @test length(setup.refinements) == 2
        @test setup.refinements[1].name == "wake"
        @test setup.refinements[1].region == (0.5, 0.2, 1.5, 0.8)
        @test setup.refinements[1].ratio == 2
        @test setup.refinements[1].parent == ""
        @test setup.refinements[2].name == "bl"
        @test setup.refinements[2].parent == "wake"

        @info "Refine parsing: 2 blocks parsed correctly"
    end

    @testset "Sub-cycling time step (uniform flow)" begin
        # Test that sub-cycling with a patch doesn't corrupt a uniform flow
        Nx, Ny = 30, 30
        dx = 1.0 / Nx
        ν = 0.1
        ω = Float64(1.0 / (3.0 * ν + 0.5))

        # Initialize base grid
        f_in  = zeros(Float64, Nx, Ny, 9)
        f_out = zeros(Float64, Nx, Ny, 9)
        ρ     = ones(Float64, Nx, Ny)
        ux    = zeros(Float64, Nx, Ny)
        uy    = zeros(Float64, Nx, Ny)
        is_solid = zeros(Bool, Nx, Ny)

        for j in 1:Ny, i in 1:Nx
            for q in 1:9
                f_in[i, j, q] = equilibrium(D2Q9(), 1.0, 0.0, 0.0, q)
            end
        end
        copyto!(f_out, f_in)

        # Create patch in center
        patch = create_patch("center", 1, 2,
                            (0.2, 0.2, 0.8, 0.8),
                            Nx, Ny, dx, ω)

        domain = create_refined_domain(Nx, Ny, dx, ω, [patch])

        # Run 10 coarse steps
        for step in 1:10
            f_in, f_out = advance_refined_step!(
                domain, f_in, f_out, ρ, ux, uy, is_solid;
                stream_fn=stream_2d!,
                collide_fn=(f, is_s) -> collide_2d!(f, is_s, ω),
                macro_fn=compute_macroscopic_2d!)
        end

        # Uniform flow at rest should remain at rest
        @test maximum(abs.(ux)) < 1e-10
        @test maximum(abs.(uy)) < 1e-10
        @test maximum(abs.(ρ .- 1.0)) < 1e-10

        @info "Sub-cycling: uniform flow stable after 10 coarse steps"
    end

    # ================================================================
    # 3D refinement tests
    # ================================================================

    @testset "3D: Create patch" begin
        Nx, Ny, Nz = 16, 16, 16
        dx = 1.0 / Nx
        ω = 1.0
        region = (0.25, 0.25, 0.25, 0.75, 0.75, 0.75)

        patch = create_patch_3d("center", 1, 2, region,
                                Nx, Ny, Nz, dx, ω, Float64)

        @test patch.Nx_inner == 16   # 8 coarse cells × ratio 2
        @test patch.Ny_inner == 16
        @test patch.Nz_inner == 16
        @test patch.Nx == patch.Nx_inner + 2 * patch.n_ghost
        @test patch.dx ≈ dx / 2
        @test patch.omega ≈ rescaled_omega(ω, 2)

        @info "3D patch created: $(patch.Nx)×$(patch.Ny)×$(patch.Nz)"
    end

    @testset "3D: Sub-cycling (uniform flow)" begin
        Nx, Ny, Nz = 16, 16, 16
        dx = 1.0 / Nx
        ν = 0.1
        ω = Float64(1.0 / (3.0 * ν + 0.5))

        f_in  = zeros(Float64, Nx, Ny, Nz, 19)
        f_out = zeros(Float64, Nx, Ny, Nz, 19)
        ρ  = ones(Float64, Nx, Ny, Nz)
        ux = zeros(Float64, Nx, Ny, Nz)
        uy = zeros(Float64, Nx, Ny, Nz)
        uz = zeros(Float64, Nx, Ny, Nz)
        is_solid = zeros(Bool, Nx, Ny, Nz)

        w = weights(D3Q19())
        for q in 1:19
            f_in[:, :, :, q] .= w[q]
        end
        copyto!(f_out, f_in)

        patch = create_patch_3d("center", 1, 2,
                                (0.2, 0.2, 0.2, 0.8, 0.8, 0.8),
                                Nx, Ny, Nz, dx, ω, Float64)
        domain = create_refined_domain_3d(Nx, Ny, Nz, dx, ω, [patch])

        for step in 1:5
            f_in, f_out = advance_refined_step_3d!(
                domain, f_in, f_out, ρ, ux, uy, uz, is_solid;
                stream_fn=stream_3d!,
                collide_fn=(f, is_s) -> collide_3d!(f, is_s, ω),
                macro_fn=compute_macroscopic_3d!)
        end

        @test maximum(abs.(ux)) < 1e-10
        @test maximum(abs.(uy)) < 1e-10
        @test maximum(abs.(uz)) < 1e-10
        @test maximum(abs.(ρ .- 1.0)) < 1e-10

        @info "3D sub-cycling: uniform flow stable after 5 coarse steps"
    end

    @testset ".krk Refine 3D parsing" begin
        setup = parse_kraken("""
            Simulation test3d D3Q19
            Domain L = 1.0 x 1.0 x 1.0  N = 16 x 16 x 16
            Physics nu = 0.1
            Boundary west wall
            Boundary east wall
            Boundary south wall
            Boundary north wall
            Boundary bottom wall
            Boundary top wall
            Refine center { region = [0.2, 0.2, 0.2, 0.8, 0.8, 0.8], ratio = 2 }
            Run 10 steps
        """)
        @test length(setup.refinements) == 1
        @test setup.refinements[1].is_3d
        @test setup.refinements[1].region_3d == (0.2, 0.2, 0.2, 0.8, 0.8, 0.8)
        @test setup.refinements[1].ratio == 2

        @info "3D Refine parsing: OK"
    end

    @testset ".krk 2D refined run (isothermal)" begin
        result = run_simulation(parse_kraken("""
            Simulation cavity_ref D2Q9
            Domain L = 1.0 x 1.0  N = 32 x 32
            Physics nu = 0.05
            Boundary north velocity(ux = 0.1, uy = 0)
            Boundary south wall
            Boundary east wall
            Boundary west wall
            Refine corner { region = [0.5, 0.5, 1.0, 1.0], ratio = 2 }
            Run 50 steps
        """))
        @test !any(isnan, result.ρ)
        @test haskey(result, :ux)

        @info ".krk 2D refined isothermal: OK"
    end

    @testset ".krk 3D refined run (isothermal)" begin
        result = run_simulation(parse_kraken("""
            Simulation cavity3d_ref D3Q19
            Domain L = 1.0 x 1.0 x 1.0  N = 12 x 12 x 12
            Physics nu = 0.1
            Boundary west wall
            Boundary east wall
            Boundary south wall
            Boundary north wall
            Boundary bottom wall
            Boundary top wall
            Refine center { region = [0.25, 0.25, 0.25, 0.75, 0.75, 0.75], ratio = 2 }
            Run 20 steps
        """))
        @test !any(isnan, result.ρ)
        @test haskey(result, :uz)
        @test size(result.ρ) == (12, 12, 12)

        @info ".krk 3D refined isothermal: OK"
    end

    @testset ".krk 2D refined thermal run" begin
        result = run_simulation(parse_kraken("""
            Simulation natconv_ref D2Q9
            Domain L = 1.0 x 1.0  N = 16 x 16
            Physics nu = 0.1  Pr = 0.71  Ra = 100
            Module thermal
            Boundary west  wall(T = 1.0)
            Boundary east  wall(T = 0.0)
            Boundary south wall
            Boundary north wall
            Refine hot { region = [0.0, 0.0, 0.4, 1.0], ratio = 2 }
            Run 50 steps
        """))
        @test !any(isnan, result.ρ)
        @test haskey(result, :Temp)
        @test !any(isnan, result.Temp)

        @info ".krk 2D refined thermal: OK"
    end

    @testset "Sanity check: refined thermal tau" begin
        setup = parse_kraken("""
            Simulation sanity_test D2Q9
            Domain L = 1.0 x 1.0  N = 32 x 32
            Physics nu = 0.01  Pr = 0.71  Ra = 1000
            Module thermal
            Boundary west wall(T = 1.0)
            Boundary east wall(T = 0.0)
            Boundary south wall
            Boundary north wall
            Refine near_wall { region = [0.0, 0.0, 0.3, 1.0], ratio = 4 }
            Run 100 steps
        """)
        issues = KrakeNN.sanity_check(setup; verbose=false)
        # ratio=4 with nu=0.01 → tau_base = 0.53, tau_fine = 4*(0.03)+0.5 = 0.62
        # thermal: alpha = 0.01/0.71 ≈ 0.0141, tau_T = 3*0.0141+0.5 = 0.5423
        # tau_T_fine = 4*(0.0423)+0.5 = 0.669 — both should be fine
        @test !isempty(setup.refinements)
        # No errors expected for this configuration
        @test !any(i -> i.level === :error, issues)

        @info "Sanity check refined thermal: OK"
    end
end
