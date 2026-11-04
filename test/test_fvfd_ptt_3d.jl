using Test
using KernelAbstractions
using Kraken

# ---------------------------------------------------------------------
# Phan-Thien–Tanner (PTT) 3D log-conformation CONSTITUTIVE canary.
#
# Scope: the constitutive source ONLY (no coupled driver / stress / .krk),
# exactly like the first Giesekus / FENE-P step. The (upper-convected,
# ξ=0) PTT relaxation in the eigenframe of C (eigenvalues c_i) is
#
#     Y(trC)·(c_i − 1),
#
# i.e. Oldroyd-B's (c_i − 1) times the SCALAR trace multiplier
#
#     Y(trC) = 1 + ε·(trC − 3)          (linear PTT)
#     Y(trC) = exp( ε·(trC − 3) )       (exponential PTT).
#
# Unlike Giesekus, Y is the SAME scalar for every eigenvalue (depends only
# on tr C). Verification strategy:
#   G0  OB-limit safety net: ε = 0 ≡ LogConfOldroydB — bit-identical (the
#       multiplier is exactly 1, so the integrator visits the same
#       floating-point trajectory). Checked against both the dedicated OB
#       constitutive step AND the OB analytical uniform-shear fixed point.
#   G1  finite-ε signature: ε > 0 gives shear-thinning — strictly reduced
#       steady C_xx and N1 = C_xx − C_yy vs OB at the same Wi, bounded,
#       NaN-free (linear AND exponential variants).
#   G2  self-verifying: the integrated fixed point satisfies the steady
#       linear-PTT simple-shear equations to ≤ 1e-6 (independent residual
#       cross-check, no hand-derived closed form).
#
# NOTE: every binding lives INSIDE the @testset local scope — nothing is
# declared at top level / `const`, so the file leaks no globals into Main
# (a top-level `const` collides with other suites' globals in-suite).
# ---------------------------------------------------------------------

@testset "FVFD 3D PTT log-conformation constitutive step" begin

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

    # One PTT RK2 constitutive step; returns the max-abs Ψ increment.
    apply_ptt_step! = function (psi, grad, lambda, dt, epsilon, n_sub; variant=:linear)
        before = psi_values(psi)
        out = ntuple(c -> similar(psi[c]), 6)
        Kraken.logfv_constitutive_step_log_ptt_3d!(
            out..., psi..., grad...,
            lambda, dt, epsilon, n_sub; variant=variant, sync=true,
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
    run_ptt_uniform_shear = function (Wi, epsilon; variant=:linear, lambda=1.0, dt=0.02)
        gamma_dot = Wi / lambda
        psi = psi_state()
        grad = velocity_gradient(; duxdy=gamma_dot)
        max_grad_norm = Kraken.logfv_max_grad_norm_3d(grad...)
        n_sub = 8 * Kraken.logfv_recommended_oldroydb_substeps_3d(max_grad_norm, lambda, dt)
        max_steps = ceil(Int, 200lambda / dt)
        delta = Inf
        steps = 0
        while steps < max_steps && delta >= 1e-13
            delta = apply_ptt_step!(psi, grad, lambda, dt, epsilon, n_sub; variant=variant)
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

    @testset "spec: ε validation + accessor" begin
        m = LogConfPTT(G=0.5, λ=2.0, ε=0.25)
        @test polymer_ptt_epsilon(m) == 0.25
        @test polymer_ptt_variant(m) == :linear
        @test uses_log_conformation(m)
        me = LogConfPTT(G=0.5, λ=2.0, ε=0.1, variant=:exponential)
        @test polymer_ptt_variant(me) == :exponential
        # OB / FENE-P / Giesekus report zero ε (OB-path branch).
        @test polymer_ptt_epsilon(LogConfOldroydB(G=0.5, λ=2.0)) == 0.0
        @test polymer_ptt_epsilon(LogConfFENEP(G=0.5, λ=2.0, Lmax2=50.0)) == 0.0
        @test polymer_ptt_epsilon(LogConfGiesekus(G=0.5, λ=2.0, α=0.2)) == 0.0
        @test_throws ArgumentError LogConfPTT(G=0.5, λ=2.0, ε=-0.1)
        @test_throws ArgumentError LogConfPTT(G=0.5, λ=2.0, ε=0.1, variant=:bogus)
    end

    @testset "G0 ε=0 ≡ Oldroyd-B bit-identical" begin
        # Per-step bit-identity over a non-trivial sheared trajectory,
        # for BOTH the linear and exponential variants (Y=1 at ε=0 in both).
        dt = 0.02
        lambda = 1.0
        n_sub = 4
        grad = velocity_gradient(; duxdy=1.3)
        for variant in (:linear, :exponential)
            psi_p = psi_state()
            psi_o = psi_state()
            for _ in 1:50
                apply_ptt_step!(psi_p, grad, lambda, dt, 0.0, n_sub; variant=variant)
                apply_ob_step!(psi_o, grad, lambda, dt, n_sub)
            end
            vp = psi_values(psi_p)
            vo = psi_values(psi_o)
            max_byte_diff = maximum(abs(vp[c] - vo[c]) for c in 1:6)
            @test max_byte_diff == 0.0          # byte-identical
            @test all(vp[c] === vo[c] for c in 1:6)
            println("G0 ε=0 ($(variant)) vs OB: max |ΔΨ| = $(max_byte_diff) (byte-identical)")
        end

        # And the integrated fixed point matches the OB analytical shear.
        for Wi in (0.5, 1.0, 2.0)
            rp = run_ptt_uniform_shear(Wi, 0.0)
            target_cxy = Wi
            target_cxx = 1.0 + 2.0 * Wi^2
            @test abs(rp.C[2] - target_cxy) / target_cxy <= 1e-3
            @test abs(rp.C[1] - target_cxx) / target_cxx <= 1e-3
        end
    end

    @testset "G1 finite-ε shear-thinning (reduced C_xx / N1 vs OB)" begin
        Wi = 2.0
        epsilon = 0.25
        ob = run_ob_uniform_shear(Wi)
        for variant in (:linear, :exponential)
            pt = run_ptt_uniform_shear(Wi, epsilon; variant=variant)
            cxx_ob, cxy_ob, cyy_ob = ob.C[1], ob.C[2], ob.C[4]
            cxx_pt, cxy_pt, cyy_pt = pt.C[1], pt.C[2], pt.C[4]
            n1_ob = cxx_ob - cyy_ob
            n1_pt = cxx_pt - cyy_pt
            println(
                "G1 Wi=$(Wi) ε=$(epsilon) ($(variant)): Cxx $(cxx_pt) < OB $(cxx_ob) ? ",
                "Cxy $(cxy_pt) < OB $(cxy_ob) ? N1 $(n1_pt) < OB $(n1_ob) ? ",
                "trC=$(pt.trC) steps=$(pt.steps) delta=$(pt.delta)",
            )
            @test all(isfinite, pt.C)
            @test isfinite(pt.trC)
            # PTT trace multiplier relaxes stretched modes faster → reduced stretch.
            @test cxx_pt < cxx_ob
            @test cxy_pt < cxy_ob          # shear-thinning of the shear component
            @test n1_pt < n1_ob            # reduced first normal-stress difference
            # Still a genuine elastic response above equilibrium, and bounded.
            @test cxx_pt > 1.0
            @test pt.trC < ob.trC
            # Off-shear-plane components stay at equilibrium / zero.
            @test abs(pt.C[3]) <= 1e-9     # C_xz
            @test abs(pt.C[5]) <= 1e-9     # C_yz
            @test abs(pt.C[6] - 1.0) <= 1e-9   # C_zz = 1
        end
    end

    @testset "G2 fixed point satisfies steady linear-PTT shear equations" begin
        # Independent residual cross-check (no hand-derived closed form):
        # at steady state dC/dt = 0 the conformation obeys, with the linear
        # PTT scalar multiplier Y = 1 + ε(trC − 3) and trC = Cxx+Cyy+Czz,
        #   2γ̇ Cxy − (Y/λ)(Cxx−1) = 0
        #    γ̇ Cyy − (Y/λ) Cxy    = 0
        #             (Y/λ)(Cyy−1) = 0  → Cyy = 1 (since Y > 0)
        lambda = 1.0
        for (Wi, epsilon) in ((1.0, 0.2), (2.0, 0.3), (0.5, 0.1))
            gamma_dot = Wi / lambda
            r = run_ptt_uniform_shear(Wi, epsilon; variant=:linear, lambda=lambda)
            Cxx, Cxy, _, Cyy, _, Czz = r.C
            trC = Cxx + Cyy + Czz
            Y = 1.0 + epsilon * (trC - 3.0)
            res_xx = 2 * gamma_dot * Cxy - (Y / lambda) * (Cxx - 1.0)
            res_xy = gamma_dot * Cyy - (Y / lambda) * Cxy
            res_yy = (Y / lambda) * (Cyy - 1.0)
            r_max = max(abs(res_xx), abs(res_xy), abs(res_yy))
            println("G2 Wi=$(Wi) ε=$(epsilon): steady residual max = $(r_max)")
            @test r_max <= 1e-6
        end
    end
end

println("EXIT=0")
