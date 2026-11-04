using Test
using Kraken
using KernelAbstractions

# ==========================================================================
# 3D viscoelastic Oldroyd-B planar Couette — first canonical analytical canary.
#
# Steady simple shear in a small periodic box: flow in x, shear in y (moving
# no-slip walls on the y-faces via half-way bounce-back + Ladd momentum
# correction → uniform γ̇), neutral & periodic z, periodic x. This exercises ALL
# six conformation components and the uniquely-3D zero-second-normal-stress
# invariant that 2D cannot reach.
#
# Closed-form steady Oldroyd-B targets (Wi ≡ λγ̇), in the bulk:
#   C_xy = Wi,  C_xx = 1 + 2 Wi²,  C_yy = C_zz = 1,  C_xz = C_yz = 0
#   u_x(y) linear,  τ_xy = η_total·γ̇,  N1 = 2 η_p λ γ̇²,  N2 = τ_yy − τ_zz = 0.
#
# Two tiers of assertions, with documented tolerances:
#
#   (A) UNIQUELY-3D INVARIANTS (machine precision). C_yy = C_zz = 1 and
#       C_xz = C_yz = 0 and N2 = 0 are exact symmetries of Oldroyd-B simple
#       shear; the 6-component tensor solver reproduces them to ~1e-13. These
#       are the headline 3D checks (the off-diagonal / z-coupling that 2D never
#       sees) and get a tight 1e-6 band.
#
#   (B) CONSTITUTIVE SELF-CONSISTENCY (≤ 2 %, N1 ≤ 4 %). The direct-C VE-LBM at
#       finite β induces a small (~2.5 %) profile curvature, so the realized
#       centre-line shear rate γ̇_meas drifts a few % below the imposed U/Ny.
#       The rigorous constitutive check therefore compares the conformation /
#       stresses against the LOCAL Weissenberg number Wi_local = λ·γ̇_meas (what
#       the conformation kernel actually advects). This isolates the Oldroyd-B
#       constitutive solver from the momentum-coupling discretisation. The
#       first-normal-stress difference N1 is a small difference of two stresses
#       (∝ C_xx − 1) and is intrinsically the most sensitive, so it carries a
#       documented 4 % band; C_xy / C_xx / τ_xy hold to ≤ 2 %.
#
#   We ALSO report (no gate) the gap against the imposed Wi so the BC/coupling
#   curvature is transparent rather than hidden.
#
# Fast: CPU Float64, 6×24×6 box, 25 000 steps (converged — identical at 20 k and
# 45 k), ~10 s.
# ==========================================================================

@testset "Viscoelastic planar Couette — 3D Oldroyd-B canary" begin

    backend = CPU()
    FT = Float64

    Nx, Ny, Nz = 6, 24, 6
    U_top   = 0.02
    ν_total = 0.1
    beta    = 0.5
    ν_s = beta * ν_total
    ν_p = (1 - beta) * ν_total
    H   = Ny                       # half-way-BB effective gap (walls at ±½ cell)
    γ̇_imposed = U_top / H
    Wi_target = 0.4                # trustworthy direct-C range (NOT noisy low Wi)
    λ = Wi_target / γ̇_imposed
    max_steps = 25_000

    # --- Units sanity: stability + steady-state-time estimate -------------
    # τ_s in the safe TRT window, Ma small, and confirm the new units helpers
    # run clean for this config. (The advective term in the estimator inflates
    # n_steps for a *periodic* box — the physical basis here is viscous
    # diffusion t_diff = H²/ν_total ≈ 23 k; the flow is empirically converged by
    # 20 k, verified offline.)
    plan = Kraken.Units.compile(; physics = :viscoelastic,
                                  geometry = (; type = :couette, blockage = 0.0),
                                  bc = (; wall_bc = :halfwayBB),
                                  Re = U_top * H / ν_total, Wi = Wi_target,
                                  beta = beta, R_LU = H, u_LU = U_top,
                                  nu_s = ν_s, nu_p = ν_p, lambda = λ,
                                  strict = false)
    @test plan.units.tau_hydro > 0.55
    @test plan.units.Ma < 0.05
    est = Kraken.Units.estimate_steady_state(plan)
    @test est.exists                      # Couette admits a steady state
    @test est.t_diff > 0 && isfinite(est.t_diff)
    # No FATAL/blocking issues for this well-resolved config.
    @test isempty(Kraken.Units.blocking_issues(plan.warnings))

    res = run_conformation_couette_libb_3d(; Nx = Nx, Ny = Ny, Nz = Nz,
                                             U = U_top, ν_s = ν_s, ν_p = ν_p,
                                             lambda = λ, tau_plus = 1.0,
                                             max_steps = max_steps,
                                             backend = backend, FT = FT)

    rel(a, b) = abs(a - b) / abs(b)

    # --- NaN-free & physical sanity --------------------------------------
    @test !any(isnan, res.ux)
    @test !any(isnan, res.uy)
    @test !any(isnan, res.uz)
    @test !any(isnan, res.C_xx)
    @test !any(isnan, res.C_xy)
    @test res.beta ≈ beta atol = 1e-12

    # --- Velocity profile: linear, correct magnitude ---------------------
    prof = res.profile
    @test issorted(prof)                              # monotone increasing
    @test prof[end] - prof[1] ≈ U_top * (Ny - 1) / Ny rtol = 0.03
    # Deviation from the ideal linear profile u = γ̇·(j−½) stays < 1 % of U.
    lin = [γ̇_imposed * (j - 0.5) for j in 1:Ny]
    @test maximum(abs.(prof .- lin)) / U_top < 0.01
    # Realized centre-line shear rate within a few % of the imposed value
    # (the residual is the documented finite-β momentum-coupling curvature).
    @test rel(res.gamma_dot_meas, res.gamma_dot) < 0.04

    Wl = res.Wi_local                                  # = λ·γ̇_meas

    # --- (A) Uniquely-3D invariants (machine precision) ------------------
    @test res.Cyy_c ≈ 1.0 atol = 1e-6
    @test res.Czz_c ≈ 1.0 atol = 1e-6
    @test abs(res.Cxz_c) < 1e-6
    @test abs(res.Cyz_c) < 1e-6
    # N2 = τ_yy − τ_zz ≡ 0 for Oldroyd-B (the falsifiable 3D check that C_zz
    # never departs from identity). Compare against the polymer-stress scale.
    @test abs(res.N2_c) < 1e-6 * abs(res.N1_c)

    # --- (B) Constitutive self-consistency vs Wi_local (≤ 2 %, N1 ≤ 4 %) --
    # NOTE (KRK-VE-3D): the momentum coupling is the validated 2D Guo ∇·τ_p
    # body force (no standalone re-relaxed Hermite source), AND the 3D
    # conformation kernel now uses a wall-aware (one-sided 2nd-order at j=1,Ny)
    # velocity-gradient stencil mirroring the validated 2D `_wall_aware_dy_2d`.
    # Defect #2 (the naive clamped wall gradient returned HALF the true shear at
    # the wall rows, depressing the BULK Cxy/N1 via TRT artificial diffusion) is
    # CLOSED for uniform simple shear: Cxy and N1 now match to roundoff
    # (Cxy_rel ≈ 5e-13, N1_rel ≈ 1e-12). All four constitutive checks pass.
    @test rel(res.Cxy_c, Wl) < 0.02
    @test rel(res.Cxx_c, 1 + 2 * Wl^2) < 0.02
    @test rel(res.tau_xy_c, res.eta_total * res.gamma_dot_meas) < 0.02
    N1_target = 2 * res.eta_p * res.lambda * res.gamma_dot_meas^2
    @test rel(res.N1_c, N1_target) < 0.04

    # --- Report (measured vs analytical, both references) ----------------
    @info "VE 3D planar-Couette canary — setup" Nx Ny Nz U_top γ̇_imposed=res.gamma_dot Wi_imposed=res.Wi beta λ max_steps t_diff=est.t_diff t_poly=est.t_poly
    @info("VE 3D planar-Couette canary — measured vs analytical (centre cell)",
          gamma_dot_meas = res.gamma_dot_meas,
          Wi_local       = Wl,
          Cxy            = res.Cxy_c, Cxy_target_local = Wl,
          Cxy_rel_local  = rel(res.Cxy_c, Wl),
          Cxy_rel_imposed = rel(res.Cxy_c, res.Wi),
          Cxx            = res.Cxx_c, Cxx_target_local = 1 + 2Wl^2,
          Cxx_rel_local  = rel(res.Cxx_c, 1 + 2Wl^2),
          Cyy = res.Cyy_c, Czz = res.Czz_c, Cxz = res.Cxz_c, Cyz = res.Cyz_c,
          tau_xy = res.tau_xy_c, tau_xy_target = res.eta_total * res.gamma_dot_meas,
          tau_xy_rel = rel(res.tau_xy_c, res.eta_total * res.gamma_dot_meas),
          N1 = res.N1_c, N1_target = N1_target, N1_rel = rel(res.N1_c, N1_target),
          N2 = res.N2_c)
end
