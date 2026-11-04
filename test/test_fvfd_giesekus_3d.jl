using Test
using KernelAbstractions
using Kraken

# ---------------------------------------------------------------------
# Giesekus 3D log-conformation CONSTITUTIVE canary.
#
# Scope: the constitutive source ONLY (no coupled driver / stress / .krk),
# exactly like the first FENE-P step. The Giesekus relaxation in the
# eigenframe of C (eigenvalues c_i) is
#
#     (c_i − 1)·(1 + α·(c_i − 1)),
#
# i.e. Oldroyd-B's (c_i − 1) times the quadratic mobility factor
# (1 + α(c_i − 1)). Verification strategy:
#   G0  OB-limit safety net: α = 0 ≡ LogConfOldroydB — bit-identical
#       (the factor is exactly 1, so the integrator visits the same
#       floating-point trajectory). Checked against both the dedicated OB
#       constitutive step AND the OB analytical uniform-shear fixed point.
#   G1  finite-α signature: α > 0 gives shear-thinning — strictly reduced
#       steady C_xx and N1 = C_xx − C_yy vs OB at the same Wi, bounded,
#       NaN-free.
#   G2  self-verifying: the integrated fixed point satisfies the steady
#       Giesekus simple-shear equations to ≤ 1e-6 (independent residual
#       cross-check, no hand-derived closed form).
#
# NOTE: every binding lives INSIDE the @testset local scope — nothing is
# declared at top level / `const`, so the file leaks no globals into Main
# (a top-level `const` collides with other suites' globals in-suite).
# ---------------------------------------------------------------------

@testset "FVFD 3D Giesekus log-conformation constitutive step" begin

    cell_3d = (1, 1, 1)
    field3 = value -> fill(Float64(value), cell_3d)

    psi_state = (; xx=0.0, xy=0.0, xz=0.0, yy=0.0, yz=0.0, zz=0.0) -> (
        field3(xx), field3(xy), field3(xz),
        field3(yy), field3(yz), field3(zz),
    )

    velocity_gradient = (;
        duxdx=0.0, duxdy=0.0, duxdz=0.0,
        duydx=0.0, duydy=0.0, duydz=0.0,
        duzdx=0.0, duzdy=0.0, duzdz=0.0,
    ) -> (
        field3(duxdx), field3(duxdy), field3(duxdz),
        field3(duydx), field3(duydy), field3(duydz),
        field3(duzdx), field3(duzdy), field3(duzdz),
    )

    psi_values = psi -> ntuple(c -> psi[c][1, 1, 1], 6)
    conformation_from_psi = psi -> Kraken.mat_exp_sym3x3(psi_values(psi)...)

    # One Giesekus RK2 constitutive step; returns the max-abs Ψ increment.
    apply_giesekus_step! = function (psi, grad, lambda, dt, alpha, n_sub)
        before = psi_values(psi)
        out = ntuple(c -> similar(psi[c]), 6)
        Kraken.logfv_constitutive_step_log_giesekus_3d!(
            out..., psi..., grad...,
            lambda, dt, alpha, n_sub; sync=true,
        )
        for c in 1:6
            psi[c] .= out[c]
        end
        after = psi_values(psi)
        return maximum(abs(after[c] - before[c]) for c in 1:6)
    end

    # One Oldroyd-B RK2 constitutive step (the dedicated OB kernel).
    apply_ob_step! = function (psi, grad, lambda, dt, n_sub)
        before = psi_values(psi)
        out = ntuple(c -> similar(psi[c]), 6)
        Kraken.logfv_constitutive_step_log_3d!(
            out..., psi..., grad...,
            lambda, dt, n_sub; sync=true,
        )
        for c in 1:6
            psi[c] .= out[c]
        end
        after = psi_values(psi)
        return maximum(abs(after[c] - before[c]) for c in 1:6)
    end

    # Integrate uniform-shear constitutive ODE to its fixed point.
    run_giesekus_uniform_shear = function (Wi, alpha; lambda=1.0, dt=0.02)
        gamma_dot = Wi / lambda
        psi = psi_state()
        grad = velocity_gradient(; duxdy=gamma_dot)
        max_grad_norm = Kraken.logfv_max_grad_norm_3d(grad...)
        n_sub = 8 * Kraken.logfv_recommended_oldroydb_substeps_3d(max_grad_norm, lambda, dt)
        max_steps = ceil(Int, 200lambda / dt)
        delta = Inf
        steps = 0
        while steps < max_steps && delta >= 1e-13
            delta = apply_giesekus_step!(psi, grad, lambda, dt, alpha, n_sub)
            steps += 1
        end
        C = conformation_from_psi(psi)
        return (C=C, trC=C[1] + C[4] + C[6], n_sub=n_sub, steps=steps, delta=delta)
    end

    run_ob_uniform_shear = function (Wi; lambda=1.0, dt=0.02)
        gamma_dot = Wi / lambda
        psi = psi_state()
        grad = velocity_gradient(; duxdy=gamma_dot)
        max_grad_norm = Kraken.logfv_max_grad_norm_3d(grad...)
        n_sub = 8 * Kraken.logfv_recommended_oldroydb_substeps_3d(max_grad_norm, lambda, dt)
        max_steps = ceil(Int, 200lambda / dt)
        delta = Inf
        steps = 0
        while steps < max_steps && delta >= 1e-13
            delta = apply_ob_step!(psi, grad, lambda, dt, n_sub)
            steps += 1
        end
        C = conformation_from_psi(psi)
        return (C=C, trC=C[1] + C[4] + C[6], n_sub=n_sub, steps=steps, delta=delta)
    end

    @testset "spec: α validation + accessor" begin
        m = LogConfGiesekus(G=0.5, λ=2.0, α=0.2)
        @test polymer_mobility(m) == 0.2
        @test uses_log_conformation(m)
        # OB / FENE-P report zero mobility (OB-path branch).
        @test polymer_mobility(LogConfOldroydB(G=0.5, λ=2.0)) == 0.0
        @test polymer_mobility(LogConfFENEP(G=0.5, λ=2.0, Lmax2=50.0)) == 0.0
        @test_throws ArgumentError LogConfGiesekus(G=0.5, λ=2.0, α=-0.1)
        @test_throws ArgumentError LogConfGiesekus(G=0.5, λ=2.0, α=0.6)
    end

    @testset "G0 α=0 ≡ Oldroyd-B bit-identical" begin
        # Per-step bit-identity over a non-trivial sheared trajectory.
        dt = 0.02
        lambda = 1.0
        n_sub = 4
        grad = velocity_gradient(; duxdy=1.3)
        psi_g = psi_state()
        psi_o = psi_state()
        for _ in 1:50
            apply_giesekus_step!(psi_g, grad, lambda, dt, 0.0, n_sub)
            apply_ob_step!(psi_o, grad, lambda, dt, n_sub)
        end
        vg = psi_values(psi_g)
        vo = psi_values(psi_o)
        max_byte_diff = maximum(abs(vg[c] - vo[c]) for c in 1:6)
        @test max_byte_diff == 0.0          # byte-identical
        @test all(vg[c] === vo[c] for c in 1:6)
        println("G0 α=0 vs OB: max |ΔΨ| = $(max_byte_diff) (byte-identical)")

        # And the integrated fixed point matches the OB analytical shear.
        for Wi in (0.5, 1.0, 2.0)
            rg = run_giesekus_uniform_shear(Wi, 0.0)
            target_cxy = Wi
            target_cxx = 1.0 + 2.0 * Wi^2
            @test abs(rg.C[2] - target_cxy) / target_cxy <= 1e-3
            @test abs(rg.C[1] - target_cxx) / target_cxx <= 1e-3
        end
    end

    @testset "G1 finite-α shear-thinning (reduced C_xx / N1 vs OB)" begin
        Wi = 2.0
        alpha = 0.2
        ob = run_ob_uniform_shear(Wi)
        gk = run_giesekus_uniform_shear(Wi, alpha)
        cxx_ob, cxy_ob, cyy_ob = ob.C[1], ob.C[2], ob.C[4]
        cxx_gk, cxy_gk, cyy_gk = gk.C[1], gk.C[2], gk.C[4]
        n1_ob = cxx_ob - cyy_ob
        n1_gk = cxx_gk - cyy_gk
        println(
            "G1 Wi=$(Wi) α=$(alpha): Cxx $(cxx_gk) < OB $(cxx_ob) ? ",
            "Cxy $(cxy_gk) < OB $(cxy_ob) ? N1 $(n1_gk) < OB $(n1_ob) ? ",
            "trC=$(gk.trC) steps=$(gk.steps) delta=$(gk.delta)",
        )
        @test all(isfinite, gk.C)
        @test isfinite(gk.trC)
        # Giesekus mobility relaxes stretched modes faster → reduced stretch.
        @test cxx_gk < cxx_ob
        @test cxy_gk < cxy_ob          # shear-thinning of the shear component
        @test n1_gk < n1_ob            # reduced first normal-stress difference
        # Still a genuine elastic response above equilibrium, and bounded.
        @test cxx_gk > 1.0
        @test gk.trC < ob.trC
        # Off-shear-plane components stay at equilibrium / zero.
        @test abs(gk.C[3]) <= 1e-9     # C_xz
        @test abs(gk.C[5]) <= 1e-9     # C_yz
        @test abs(gk.C[6] - 1.0) <= 1e-9   # C_zz = 1
    end

    @testset "G2 fixed point satisfies steady Giesekus shear equations" begin
        # Independent residual cross-check (no hand-derived closed form):
        # at steady state dC/dt = 0 the conformation obeys
        #   2γ̇ Cxy − (1/λ)[(Cxx−1) + α((Cxx−1)² + Cxy²)] = 0
        #    γ̇ Cyy − (1/λ)[Cxy   + α Cxy ((Cxx−1)+(Cyy−1))] = 0
        #              (1/λ)[(Cyy−1) + α((Cyy−1)² + Cxy²)] = 0
        lambda = 1.0
        for (Wi, alpha) in ((1.0, 0.2), (2.0, 0.3), (0.5, 0.1))
            gamma_dot = Wi / lambda
            r = run_giesekus_uniform_shear(Wi, alpha; lambda=lambda)
            Cxx, Cxy, _, Cyy, _, _ = r.C
            ax = Cxx - 1.0
            ay = Cyy - 1.0
            res_xx = 2 * gamma_dot * Cxy - (1 / lambda) * (ax + alpha * (ax^2 + Cxy^2))
            res_xy = gamma_dot * Cyy - (1 / lambda) * (Cxy + alpha * Cxy * (ax + ay))
            res_yy = (1 / lambda) * (ay + alpha * (ay^2 + Cxy^2))
            r_max = max(abs(res_xx), abs(res_xy), abs(res_yy))
            println("G2 Wi=$(Wi) α=$(alpha): steady residual max = $(r_max)")
            @test r_max <= 1e-6
        end
    end
end

println("EXIT=0")
