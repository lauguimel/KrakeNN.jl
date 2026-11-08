using Test
using KrakeNN
using KernelAbstractions: @index, @Const, @kernel, get_backend

# ===========================================================================
# SLBM kernel sanity tests (Week 2, v0.2 curvilinear LBM path).
#
# Core invariants:
#   1. On an isotropic Cartesian mesh, SLBM departures are integer and
#      bilinear interpolation returns the exact neighbor value — SLBM
#      reduces to pull-stream.
#   2. Equilibrium initial state remains equilibrium after any number
#      of SLBM + BGK steps (mass, momentum preserved).
#   3. On a periodic mesh, total ρ is conserved to machine precision.
# ===========================================================================

@testset "SLBM kernel" begin

    @testset "build_slbm_geometry — Cartesian isotropic gives integer shifts" begin
        # Physical domain [0, Nx-1] × [0, Ny-1] gives dX/dξ = Nx-1,
        # dY/dη = Ny-1, J = (Nx-1)·(Ny-1), dx_ref = 1. Then Δi = −c_qx
        # and Δj = −c_qy exactly.
        Nx, Ny = 8, 8
        mesh = stretched_box_mesh(; x_min=0.0, x_max=Float64(Nx - 1),
                                    y_min=0.0, y_max=Float64(Ny - 1),
                                    Nx=Nx, Ny=Ny)
        @test mesh.dx_ref ≈ 1.0
        geom = build_slbm_geometry(mesh)

        cx = KrakeNN.velocities_x(D2Q9())
        cy = KrakeNN.velocities_y(D2Q9())
        for j in 2:Ny-1, i in 2:Nx-1
            for q in 1:9
                @test geom.i_dep[i, j, q] ≈ Float64(i - cx[q]) atol=1e-12
                @test geom.j_dep[i, j, q] ≈ Float64(j - cy[q]) atol=1e-12
            end
        end
    end

    @testset "bilinear_f — integer input returns exact neighbor" begin
        Nx, Ny = 16, 16
        # Seed f_in with direction-distinguishable values
        f = zeros(Float64, Nx, Ny, 9)
        for j in 1:Ny, i in 1:Nx, q in 1:9
            f[i, j, q] = 100.0 * q + i + 0.1 * j
        end
        for j in 4:Ny-3, i in 4:Nx-3, q in 1:9
            @test KrakeNN.bilinear_f(f, Float64(i), Float64(j), q,
                                     Nx, Ny, false, false) ≈ f[i, j, q]
        end
    end

    @testset "bilinear_f — fractional input matches analytical" begin
        Nx, Ny = 12, 12
        f = zeros(Float64, Nx, Ny, 9)
        # Smooth linear field per direction
        for j in 1:Ny, i in 1:Nx, q in 1:9
            f[i, j, q] = 0.5 * i + 0.25 * j + q
        end
        # Bilinear interpolation of a linear field reproduces it exactly
        for j in 3:Ny-3, i in 3:Nx-3, q in 1:9
            ifl = Float64(i) + 0.3
            jfl = Float64(j) + 0.7
            expected = 0.5 * ifl + 0.25 * jfl + q
            @test KrakeNN.bilinear_f(f, ifl, jfl, q, Nx, Ny, false, false) ≈ expected
        end
    end

    @testset "SLBM at equilibrium — no evolution (any mesh)" begin
        # ρ=1, u=0 equilibrium must remain stationary under SLBM + BGK.
        Nx, Ny = 12, 12
        mesh = stretched_box_mesh(; x_min=0.0, x_max=Float64(Nx - 1),
                                    y_min=0.0, y_max=Float64(Ny - 1),
                                    Nx=Nx, Ny=Ny)
        geom = build_slbm_geometry(mesh)

        f_in = zeros(Float64, Nx, Ny, 9)
        f_out = similar(f_in)
        ρ = ones(Float64, Nx, Ny)
        ux = zeros(Float64, Nx, Ny)
        uy = zeros(Float64, Nx, Ny)
        is_solid = zeros(Bool, Nx, Ny)
        for j in 1:Ny, i in 1:Nx, q in 1:9
            f_in[i, j, q] = KrakeNN.equilibrium(D2Q9(), 1.0, 0.0, 0.0, q)
        end
        f0 = copy(f_in)

        for _ in 1:20
            slbm_bgk_step!(f_out, f_in, ρ, ux, uy, is_solid, geom, 1.0)
            f_in, f_out = f_out, f_in
        end

        # Interior must remain at initial equilibrium (walls may drift due
        # to clamp vs. bounce-back mismatch — that's handled separately)
        for j in 3:Ny-2, i in 3:Nx-2, q in 1:9
            @test f_in[i, j, q] ≈ f0[i, j, q] atol=1e-12
        end
        @test all(abs.(ux[3:end-2, 3:end-2]) .< 1e-12)
        @test all(abs.(uy[3:end-2, 3:end-2]) .< 1e-12)
        @test all(abs.(ρ[3:end-2, 3:end-2] .- 1.0) .< 1e-12)
    end

    @testset "SLBM reproduces fused_bgk on isotropic Cartesian interior" begin
        # Compare 1 SLBM step against 1 fused_bgk step starting from the
        # same non-trivial initial condition. Wall populations differ
        # (clamp vs. halfway bounce-back) — compare the interior only.
        Nx, Ny = 24, 24
        mesh = stretched_box_mesh(; x_min=0.0, x_max=Float64(Nx - 1),
                                    y_min=0.0, y_max=Float64(Ny - 1),
                                    Nx=Nx, Ny=Ny)
        geom = build_slbm_geometry(mesh)

        # Seed a Gaussian density bump so the populations are non-trivial
        f_slbm = zeros(Float64, Nx, Ny, 9)
        f_ref  = zeros(Float64, Nx, Ny, 9)
        ρ = zeros(Float64, Nx, Ny); ux = zeros(Nx, Ny); uy = zeros(Nx, Ny)
        is_solid = zeros(Bool, Nx, Ny)
        cx0, cy0 = (Nx + 1) / 2, (Ny + 1) / 2
        for j in 1:Ny, i in 1:Nx
            ρ_ij = 1.0 + 0.05 * exp(-((i - cx0)^2 + (j - cy0)^2) / 8.0)
            for q in 1:9
                feq = KrakeNN.equilibrium(D2Q9(), ρ_ij, 0.0, 0.0, q)
                f_slbm[i, j, q] = feq
                f_ref[i, j, q]  = feq
            end
        end

        f_slbm_out = similar(f_slbm)
        f_ref_out  = similar(f_ref)
        ρ_s = zeros(Nx, Ny); ux_s = zeros(Nx, Ny); uy_s = zeros(Nx, Ny)
        ρ_r = zeros(Nx, Ny); ux_r = zeros(Nx, Ny); uy_r = zeros(Nx, Ny)

        slbm_bgk_step!(f_slbm_out, f_slbm, ρ_s, ux_s, uy_s, is_solid, geom, 1.0)
        fused_bgk_step!(f_ref_out, f_ref, ρ_r, ux_r, uy_r, is_solid, Nx, Ny, 1.0)

        # Interior cells away from any boundary-clamp artifact
        for j in 3:Ny-2, i in 3:Nx-2, q in 1:9
            @test f_slbm_out[i, j, q] ≈ f_ref_out[i, j, q] atol=1e-12
        end
        @test ρ_s[3:end-2, 3:end-2] ≈ ρ_r[3:end-2, 3:end-2] atol=1e-12
    end

    @testset "SLBM periodic — total mass conserved exactly" begin
        # Fully periodic mesh: no wall clamp, so mass must be conserved
        # to machine precision even near the seams.
        Nx, Ny = 16, 16
        # Build a periodic Cartesian mesh manually via build_mesh
        mesh = KrakeNN.build_mesh((ξ, η) -> (ξ * Nx, η * Ny);
                                  Nξ=Nx, Nη=Ny,
                                  periodic_ξ=true, periodic_η=true,
                                  type=:custom)
        @test mesh.dx_ref ≈ 1.0
        geom = build_slbm_geometry(mesh)

        # Seed with a non-uniform density field
        f_in = zeros(Float64, Nx, Ny, 9)
        for j in 1:Ny, i in 1:Nx
            ρ0 = 1.0 + 0.1 * sin(2π * (i - 1) / Nx) * cos(2π * (j - 1) / Ny)
            for q in 1:9
                f_in[i, j, q] = KrakeNN.equilibrium(D2Q9(), ρ0, 0.0, 0.0, q)
            end
        end

        f_out = similar(f_in)
        ρ = zeros(Nx, Ny); ux = zeros(Nx, Ny); uy = zeros(Nx, Ny)
        is_solid = zeros(Bool, Nx, Ny)
        M0 = sum(f_in)

        for _ in 1:50
            slbm_bgk_step!(f_out, f_in, ρ, ux, uy, is_solid, geom, 1.0)
            f_in, f_out = f_out, f_in
        end

        @test sum(f_in) ≈ M0 atol=1e-10
    end

    @testset "SLBM + MRT — equilibrium stationary" begin
        Nx, Ny = 12, 12
        mesh = stretched_box_mesh(; x_min=0.0, x_max=Float64(Nx - 1),
                                    y_min=0.0, y_max=Float64(Ny - 1),
                                    Nx=Nx, Ny=Ny)
        geom = build_slbm_geometry(mesh)

        f_in = zeros(Float64, Nx, Ny, 9)
        f_out = similar(f_in)
        ρ = ones(Nx, Ny); ux = zeros(Nx, Ny); uy = zeros(Nx, Ny)
        is_solid = zeros(Bool, Nx, Ny)
        for j in 1:Ny, i in 1:Nx, q in 1:9
            f_in[i, j, q] = KrakeNN.equilibrium(D2Q9(), 1.0, 0.0, 0.0, q)
        end
        f0 = copy(f_in)

        ν = 0.1
        for _ in 1:20
            slbm_mrt_step!(f_out, f_in, ρ, ux, uy, is_solid, geom, ν)
            f_in, f_out = f_out, f_in
        end

        for j in 3:Ny-2, i in 3:Nx-2, q in 1:9
            @test f_in[i, j, q] ≈ f0[i, j, q] atol=1e-12
        end
        @test all(abs.(ρ[3:end-2, 3:end-2] .- 1.0) .< 1e-12)
    end

    @testset "SLBM + MRT periodic — mass conservation" begin
        Nx, Ny = 16, 16
        mesh = KrakeNN.build_mesh((ξ, η) -> (ξ * Nx, η * Ny);
                                  Nξ=Nx, Nη=Ny,
                                  periodic_ξ=true, periodic_η=true)
        geom = build_slbm_geometry(mesh)

        f_in = zeros(Float64, Nx, Ny, 9)
        for j in 1:Ny, i in 1:Nx
            ρ0 = 1.0 + 0.1 * sin(2π * (i - 1) / Nx) * cos(2π * (j - 1) / Ny)
            for q in 1:9
                f_in[i, j, q] = KrakeNN.equilibrium(D2Q9(), ρ0, 0.0, 0.0, q)
            end
        end
        f_out = similar(f_in)
        ρ = zeros(Nx, Ny); ux = zeros(Nx, Ny); uy = zeros(Nx, Ny)
        is_solid = zeros(Bool, Nx, Ny)
        M0 = sum(f_in)

        ν = 0.1
        for _ in 1:50
            slbm_mrt_step!(f_out, f_in, ρ, ux, uy, is_solid, geom, ν)
            f_in, f_out = f_out, f_in
        end
        @test sum(f_in) ≈ M0 atol=1e-10
    end

    @testset "Float32 kernel executes without type errors" begin
        Nx, Ny = 12, 12
        mesh = stretched_box_mesh(; x_min=0.0f0, x_max=Float32(Nx - 1),
                                    y_min=0.0f0, y_max=Float32(Ny - 1),
                                    Nx=Nx, Ny=Ny, FT=Float32)
        geom = build_slbm_geometry(mesh)
        @test eltype(geom.i_dep) === Float32

        f_in = zeros(Float32, Nx, Ny, 9)
        f_out = similar(f_in)
        ρ = ones(Float32, Nx, Ny); ux = zeros(Float32, Nx, Ny); uy = zeros(Float32, Nx, Ny)
        is_solid = zeros(Bool, Nx, Ny)
        for j in 1:Ny, i in 1:Nx, q in 1:9
            f_in[i, j, q] = KrakeNN.equilibrium(D2Q9(), 1.0f0, 0.0f0, 0.0f0, q)
        end

        slbm_bgk_step!(f_out, f_in, ρ, ux, uy, is_solid, geom, 1.0f0)
        @test eltype(f_out) === Float32
        @test all(isfinite.(f_out))
    end

end
