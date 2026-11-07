using Test
using KrakeNN
using KernelAbstractions

@testset "FVFD log-conformation planar Poiseuille — 3D Oldroyd-B canary" begin
    backend = CPU()
    FT = Float64

    Nx, Ny, Nz = 6, 32, 6
    ν_total = 0.1
    beta = 0.5
    ν_s = beta * ν_total
    ν_p = (1 - beta) * ν_total
    Fx = 1.5e-5
    γ̇_wall = Fx / (2 * ν_total) * (Ny - 1)
    Wi_wall_target = 1.0
    λ = Wi_wall_target / γ̇_wall
    max_steps = 10_000

    res = KrakeNN.run_viscoelastic_fvfd_poiseuille_3d(;
        Nx, Ny, Nz, Fx, ν_s, ν_p, lambda=λ,
        max_steps, backend, FT,
        advection_scheme=:muscl_superbee,
    )

    finite_state = all(isfinite, res.ux) &&
                   all(isfinite, res.psi_xx) &&
                   all(isfinite, res.psi_xy) &&
                   all(isfinite, res.psi_xz) &&
                   all(isfinite, res.psi_yy) &&
                   all(isfinite, res.psi_yz) &&
                   all(isfinite, res.psi_zz) &&
                   all(isfinite, res.tau_p_xx) &&
                   all(isfinite, res.tau_p_xy) &&
                   all(isfinite, res.tau_p_xz) &&
                   all(isfinite, res.tau_p_yy) &&
                   all(isfinite, res.tau_p_yz) &&
                   all(isfinite, res.tau_p_zz)

    yshape = [(j - 0.5) * (Ny + 0.5 - j) for j in 1:Ny]
    fit_slope = sum(res.profile .* yshape) / sum(abs2, yshape)
    ν_eff = Fx / (2 * fit_slope)
    rel_ν_err = abs(ν_eff - ν_total) / ν_total
    u_newtonian = [Fx / (2 * ν_total) * yshape[j] for j in 1:Ny]
    ratio_at(j) = res.profile[j] / u_newtonian[j]

    @testset "C1 NaN-free stability" begin
        @test max_steps >= 10_000
        @test res.completed_steps == max_steps
        @test finite_state
    end

    @testset "C2 coupling self-consistency" begin
        @test ν_total ≈ ν_s / beta
        @test rel_ν_err < 0.01
    end

    jc = Ny ÷ 2 + 1
    println("C1 NaN-free steps=$(res.completed_steps) finite=$(finite_state)")
    println("C2 nu_eff=$(ν_eff) nu_total=$(ν_total) rel_err_percent=$(100 * rel_ν_err)")
    println("profile ratios j=4:$(ratio_at(4)) j=8:$(ratio_at(8)) j=$(jc):$(ratio_at(jc)) j=$(Ny-3):$(ratio_at(Ny - 3))")
    println("centreline Cxx=$(res.Cxx_prof[jc]) Cxy=$(res.Cxy_prof[jc]) max_substeps=$(res.max_substeps_observed)")

    @testset "C3 no spurious wall-normal polymer stress" begin
        # Without u_y = 0 at the half-way wall in the gradient, tau_yy grew
        # exponentially (about 1e2 after 4000 steps) in this channel.
        res3 = KrakeNN.run_viscoelastic_fvfd_poiseuille_3d(;
            Nx=6, Ny=12, Nz=6, Fx=1.0e-5, ν_s=0.05, ν_p=0.05, lambda=5.0,
            max_steps=4000, backend, FT,
        )
        ratio = maximum(abs, res3.tau_p_yy) / maximum(abs, res3.tau_p_xy)
        println("C3 max|tau_yy|/max|tau_xy| = $(ratio)")
        @test all(isfinite, res3.tau_p_yy)
        @test ratio <= 1.0e-10
    end
end
