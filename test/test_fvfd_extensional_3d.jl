using Test
using Kraken
using KernelAbstractions

function center_stagnation_mean(A)
    Nx, Ny, Nz = size(A)
    i1 = max(1, Nx ÷ 2)
    i2 = min(Nx, Nx ÷ 2 + 1)
    j1 = max(1, Ny ÷ 2)
    j2 = min(Ny, Ny ÷ 2 + 1)
    return sum(@view A[i1:i2, j1:j2, :]) / ((i2 - i1 + 1) * (j2 - j1 + 1) * Nz)
end

function finite_extensional_state(res)
    return all(isfinite, res.ρ) &&
           all(isfinite, res.ux) &&
           all(isfinite, res.uy) &&
           all(isfinite, res.uz) &&
           all(isfinite, res.psi_xx) &&
           all(isfinite, res.psi_xy) &&
           all(isfinite, res.psi_xz) &&
           all(isfinite, res.psi_yy) &&
           all(isfinite, res.psi_yz) &&
           all(isfinite, res.psi_zz) &&
           all(isfinite, res.C_xx) &&
           all(isfinite, res.C_yy) &&
           all(isfinite, res.C_zz) &&
           all(isfinite, res.tau_p_xx) &&
           all(isfinite, res.tau_p_xy) &&
           all(isfinite, res.tau_p_xz) &&
           all(isfinite, res.tau_p_yy) &&
           all(isfinite, res.tau_p_yz) &&
           all(isfinite, res.tau_p_zz)
end

function planar_extension_metrics(res, lambda_epsilon, epsilon_dot)
    Cxx_ref = 1 / (1 - 2 * lambda_epsilon)
    Cyy_ref = 1 / (1 + 2 * lambda_epsilon)
    Czz_ref = 1.0
    Cxx_meas = center_stagnation_mean(res.C_xx)
    Cyy_meas = center_stagnation_mean(res.C_yy)
    Czz_meas = center_stagnation_mean(res.C_zz)
    Cxy_meas = center_stagnation_mean(res.C_xy)
    rel_Cxx = abs(Cxx_meas - Cxx_ref) / Cxx_ref
    rel_Cyy = abs(Cyy_meas - Cyy_ref) / Cyy_ref
    abs_Czz = abs(Czz_meas - Czz_ref)
    grad_x = center_stagnation_mean(res.duxdx)
    grad_y = center_stagnation_mean(res.duydy)
    return (;
        Cxx_ref, Cyy_ref, Czz_ref,
        Cxx_meas, Cyy_meas, Czz_meas, Cxy_meas,
        rel_Cxx, rel_Cyy, abs_Czz,
        grad_x, grad_y,
        rel_grad_x=abs(grad_x - epsilon_dot) / epsilon_dot,
        rel_grad_y=abs(grad_y + epsilon_dot) / epsilon_dot,
    )
end

@testset "FVFD 3D planar-extension Oldroyd-B canary" begin
    backend = CPU()
    FT = Float64

    Nx, Ny, Nz = 24, 24, 6
    lambda = 50.0
    epsilon_dot = 0.005
    lambda_epsilon = lambda * epsilon_dot
    @test 2 * lambda_epsilon < 1

    nu_total = 0.1
    beta = 0.5
    nu_s = beta * nu_total
    nu_p = (1 - beta) * nu_total
    max_steps_imposed = 1_000
    max_steps_coupled = 1_500

    @testset "E1 NaN-free imposed-velocity run" begin
        res = Kraken.run_viscoelastic_fvfd_extensional_3d(;
            Nx, Ny, Nz,
            epsilon_dot,
            ν_s=nu_s,
            ν_p=nu_p,
            lambda,
            max_steps=max_steps_imposed,
            backend,
            FT,
            advection_scheme=:muscl_superbee,
            velocity_mode=:imposed,
        )
        finite_state = finite_extensional_state(res)

        @test res.completed_steps == max_steps_imposed
        @test res.velocity_mode === :imposed
        @test res.open_x_gradient_supported
        @test res.open_y_gradient_supported
        @test finite_state

        m = planar_extension_metrics(res, lambda_epsilon, epsilon_dot)
        @test m.rel_Cxx <= 0.01
        @test m.rel_Cyy <= 0.01
        @test m.abs_Czz <= 0.01
        @test abs(m.Cxy_meas) <= 0.01
        @test abs(m.grad_x - epsilon_dot) <= 1e-12
        @test abs(m.grad_y + epsilon_dot) <= 1e-12

        println("E1 NaN-free steps=$(res.completed_steps) finite=$(finite_state) mode=$(res.velocity_mode)")
        println(
            "E1 lambda_epsilon=$(lambda_epsilon) ",
            "Cxx=$(m.Cxx_meas) ref=$(m.Cxx_ref) rel_err_percent=$(100 * m.rel_Cxx) ",
            "Cyy=$(m.Cyy_meas) ref=$(m.Cyy_ref) rel_err_percent=$(100 * m.rel_Cyy)",
        )
        println(
            "E1 Czz=$(m.Czz_meas) abs_err=$(m.abs_Czz) Cxy=$(m.Cxy_meas) ",
            "grad_x=$(m.grad_x) grad_y=$(m.grad_y) max_substeps=$(res.max_substeps_observed)",
        )
    end

    @testset "E2 coupled all-face Zou-He planar-extension fixed point" begin
        res = Kraken.run_viscoelastic_fvfd_extensional_3d(;
            Nx, Ny, Nz,
            epsilon_dot,
            ν_s=nu_s,
            ν_p=nu_p,
            lambda,
            max_steps=max_steps_coupled,
            backend,
            FT,
            advection_scheme=:muscl_superbee,
            velocity_mode=:coupled,
        )
        finite_state = finite_extensional_state(res)
        m = planar_extension_metrics(res, lambda_epsilon, epsilon_dot)

        @test res.completed_steps == max_steps_coupled
        @test res.velocity_mode === :coupled
        @test res.bc_config === :openxy_zh_velocity
        @test finite_state
        # RESOLVED: the prior ~14.5% strain deficit / ~7% C_xx-C_yy error was the
        # z-periodicity mismatch — the LBM solvent's PullHalfwayBB_3D bounced the
        # k=1/k=Nz z-faces as no-slip walls while the FVFD polymer side was fully
        # z-periodic. Threading periodic_z=true into the Guo solvent step (z-wrap
        # variant of PullHalfwayBB_3D) closes it: no calibration.
        # Measured 2026-10-08 (CPU, Float64, Julia 1.11.9), after the TRT Guo
        # forcing fix (issue #67: odd part of the Guo source scaled by
        # 1 - s_minus/2): grad_x = 0.0050362 (+0.72 % over ε̇ = 0.005),
        # grad_y = -0.0050172 (+0.34 %), C_xx rel-err 0.18 %, C_yy rel-err 0.11 %
        # against the NOMINAL-rate fixed point. Against the fixed point at the
        # measured rates, 1/(1 − 2λ·grad), C_xx is -0.54 % and C_yy +0.0001 %.
        # Before the Guo fix (2026-09-24, after the periodic-z MUSCL fix #54):
        # grad_x = 0.0050704 (+1.41 %), C_xx 0.83 %, C_yy 0.29 % nominal; the
        # collision injected g·F per step, g = 1 + (s_minus - s_plus)/2 ≈ 0.52 at
        # ν_s = 0.05. Before #54 the nominal errors read 0.06 % / 0.22 %: the
        # first-order z fallback damped C_xx and happened to offset the
        # strain-rate excess of the coupled flow.
        @test m.rel_Cxx <= 0.01
        @test m.rel_Cyy <= 0.01
        @test m.abs_Czz <= 0.01
        @test abs(m.Cxy_meas) <= 0.01

        println(
            "E2 coupled BC=$(res.bc_config) faces=west/east/south/north ZouHeVelocity ",
            "z=periodic epsilon_dot=$(epsilon_dot) lambda=$(lambda) Ny=$(Ny) max_steps=$(max_steps_coupled)",
        )
        println(
            "E2 coupled Cxx=$(m.Cxx_meas) ref=$(m.Cxx_ref) rel_err_percent=$(100 * m.rel_Cxx) ",
            "Cyy=$(m.Cyy_meas) ref=$(m.Cyy_ref) rel_err_percent=$(100 * m.rel_Cyy)",
        )
        println(
            "E2 coupled Czz=$(m.Czz_meas) abs_err=$(m.abs_Czz) Cxy=$(m.Cxy_meas) ",
            "grad_x=$(m.grad_x) grad_y=$(m.grad_y) ",
            "rel_grad_x=$(m.rel_grad_x) rel_grad_y=$(m.rel_grad_y) ",
            "max_substeps=$(res.max_substeps_observed)",
        )
    end
end

println("EXIT=0")
