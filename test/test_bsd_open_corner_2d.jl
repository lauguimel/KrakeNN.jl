using Test
using Kraken

# Both-sides-diffusion (BSD) correction of the 2D viscoelastic log-FV drivers:
# the x second derivative at the first/last column of an open x end is taken as
# zero (assumes d2u/dx2 = 0 there). The one-sided stencil used before has the
# wrong sign on the period-2 mode of the outlet corner cells and made runs with
# nu_s << nu_p (beta <= 0.1) diverge.

@testset "BSD open-end corner (2D log-FV)" begin

    @testset "Helper: zero at open ends only" begin
        Nx, Ny = 6, 4
        ux = [Float64(i)^2 for i in 1:Nx, j in 1:Ny]   # d2u/dx2 = 2
        uy = zeros(Nx, Ny)
        solid = falses(Nx, Ny)
        zeros2() = zeros(Nx, Ny)
        zeta_nu_p = 0.3
        force_x(bc) = begin
            fx = zeros2(); fy = zeros2()
            Kraken.fvfd_bsd_force_2d!(fx, fy, zeros2(), zeros2(), ux, uy, solid,
                1.0, zeta_nu_p, 1.0, 1.0, bc)
            fx
        end
        # Open x ends: interior columns see the curvature, end columns do not.
        fx = force_x(Kraken.fvfd_openx_wally_bcspec_2d())
        @test all(fx[2:Nx-1, 2:Ny-1] .≈ -2 * zeta_nu_p)
        @test all(fx[1, :] .== 0)
        @test all(fx[Nx, :] .== 0)
        # Wall x ends and periodic x: unchanged one-sided / wrapped stencils.
        fx_wall = force_x(Kraken.fvfd_wallxwally_bcspec_2d())
        @test all(fx_wall[1, 2:Ny-1] .≈ -2 * zeta_nu_p)
        @test all(fx_wall[Nx, 2:Ny-1] .≈ -2 * zeta_nu_p)
        periodic = Kraken.FVFDDomainBC2D(; west=:periodic, east=:periodic,
            south=:wall, north=:wall)
        ux_p = [sin(2pi * (i - 1) / Nx) for i in 1:Nx, j in 1:Ny]
        fx_p = zeros2(); fy_p = zeros2()
        Kraken.fvfd_bsd_force_2d!(fx_p, fy_p, zeros2(), zeros2(), ux_p, uy, solid,
            1.0, zeta_nu_p, 1.0, 1.0, periodic)
        lap = (circshift(ux_p, (-1, 0)) .- 2 .* ux_p .+ circshift(ux_p, (1, 0)))
        @test all(isapprox.(fx_p[:, 2:Ny-1], -zeta_nu_p .* lap[:, 2:Ny-1]; atol=1e-14))
    end

    @testset "Stability: cylinder, beta = 0.01, Wi = 0.5" begin
        # Before the change the run went NaN at step 300.
        R = 4.0; um = 0.003; nu0 = 0.15; beta = 0.01
        r = Kraken.run_viscoelastic_logfv_cylinder_coupled_2d(;
            radius=R, H=16, L_up=3, L_down=4,
            nu_s=beta * nu0, nu_p=(1 - beta) * nu0, lambda=0.5 * R / um,
            u_mean=um, bsd_fraction=1.0, max_steps=3000,
            wall_bc=:halfwayBB, diagnostic_stride=100)
        @test r.Nx == 28
        @test r.first_nonfinite_step in (nothing, 0)
        @test all(isfinite, r.ux) && all(isfinite, r.uy)
        @test maximum(abs, r.uy) < 1e-2
    end

    @testset "Accuracy: open-x Oldroyd-B Poiseuille, beta = 0.59" begin
        # Straight channel (the cylinder geometry without the cylinder), Ny = 16,
        # Nx = 4 Ny, Wi = 0.5 on the wall shear rate. The fixed and unfixed
        # stencils agree to <1 % here (the corner term matters only at small
        # beta), so this pins the error level of the established case.
        Ny = 16; Nx = 4Ny; beta = 0.59; nu0 = 0.15; um = 0.003
        FT = Float64
        hm = fill(false, Ny); hm[2:(Ny-1)] .= true
        geom = Kraken.StepChannelGeometry2D{FT,Array{FT,3},Matrix{Bool},Vector{Bool}}(
            :straight, Nx, Ny, 1, 1:Ny, 1:Ny, Ny, Ny, Ny, zeros(FT, Nx, Ny, 9),
            Matrix{Bool}(falses(Nx, Ny)), hm, copy(hm), fill(true, Ny), fill(true, Ny))
        r = Kraken._run_viscoelastic_logfv_step_channel_coupled_2d(geom;
            shear_length=Ny, nu_s=beta * nu0, nu_p=(1 - beta) * nu0,
            lambda=0.5 * Ny / (6um), u_mean=um, Fx_body=0.0, bsd_fraction=1.0,
            max_steps=20000, advection_scheme=:muscl_superbee, wall_bc=:halfwayBB)
        shape(j) = (y = (j - 0.5) / Ny; 6y * (1 - y))
        dshape(j) = (y = (j - 0.5) / Ny; 6 * (1 - 2y) / Ny)
        nu_p = (1 - beta) * nu0
        eu = nu = et = nt = 0.0
        for i in (Nx ÷ 2):Nx
            U = sum(r.ux[i, :]) / sum(shape, 1:Ny)
            for j in 1:Ny
                ue = U * shape(j); te = nu_p * U * dshape(j)
                eu += (r.ux[i, j] - ue)^2; nu += ue^2
                et += (r.tauxy[i, j] - te)^2; nt += te^2
            end
        end
        @test all(isfinite, r.ux)
        @test sqrt(eu / nu) < 1.4e-2    # measured 1.22e-2
        @test sqrt(et / nt) < 6.0e-2    # measured 5.28e-2
    end
end
