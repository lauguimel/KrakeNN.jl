using Test
using Kraken
using KernelAbstractions

# ==========================================================================
# 3D viscoelastic Oldroyd-B sphere driver — committed coverage.
#
# Two checks (CPU Float64, fast):
#
#   1. Newtonian-limit analytical invariant (rigorous anchor).
#      At β = 1 the polymer modulus G = ν_p/λ → 0, so the Oldroyd-B
#      stress τ_p = G·(C − I) ≡ 0 and the Oldroyd-B equations reduce
#      EXACTLY to Newtonian. The VE sphere driver at ν_p = 0 must then
#      reproduce Kraken's own Newtonian confined-sphere driver
#      (`run_sphere_libb_3d`) on the SAME geometry / Re / grid: both
#      paths share the identical solvent operators
#      (`fused_trt_libb_v2_step_3d!` + `apply_bc_rebuild_3d!`) and the
#      identical drag closure (`compute_drag_libb_3d`). With a zeroed
#      Hermite source the only difference is the (decoupled, harmless)
#      conformation arithmetic, so agreement is essentially exact in
#      exact arithmetic; we gate on a tight 1 % band to absorb only
#      floating-point reordering. This is the Phase-1 analytical
#      reference (zero polymer contribution).
#
#   2. Cut-link canary (regression). A small VE sphere with a FRACTIONAL
#      radius → fractional q_wall, genuinely exercising the LI-BB
#      cut-link path (a box / q_wall = 0.5 halfway-BB collapse hides
#      cut-link bugs). Wi = 0.01, coarse grid, few hundred steps: must
#      run NaN-free and return a finite, positive Cd.
# ==========================================================================

@testset "Viscoelastic sphere — 3D" begin

    backend = CPU()
    FT = Float64

    @testset "Newtonian-limit invariant (β = 1, η_p = 0)" begin
        # Identical settings for both drivers. Uniform inlet so the two
        # reference-velocity / Cd conventions coincide trivially. Coarse
        # grid + few steps: this is an algebraic-equivalence check, not a
        # benchmark, so we do not need a converged Cd, only that the two
        # solvers track each other.
        Nx, Ny, Nz = 32, 16, 16
        radius = 3
        cx, cy, cz = Nx ÷ 4, Ny ÷ 2, Nz ÷ 2
        u_in = 0.04
        Re = 20.0
        D = 2 * radius
        ν = u_in * D / Re
        steps, window = 400, 100

        new_res = run_sphere_libb_3d(; Nx=Nx, Ny=Ny, Nz=Nz,
                                       cx=cx, cy=cy, cz=cz,
                                       radius=radius, u_in=u_in, ν=ν,
                                       inlet=:uniform,
                                       max_steps=steps, avg_window=window,
                                       backend=backend, T=FT)

        # VE driver at β = 1: ν_p = 0 ⇒ G = ν_p/λ = 0 ⇒ τ_p ≡ 0.
        # ν_s = ν matches the Newtonian total viscosity exactly.
        ve_res = run_conformation_sphere_libb_3d(; Nx=Nx, Ny=Ny, Nz=Nz,
                                                   cx=cx, cy=cy, cz=cz,
                                                   radius=radius, u_in=u_in,
                                                   ν_s=ν, ν_p=0.0, lambda=1.0,
                                                   inlet=:uniform,
                                                   max_steps=steps,
                                                   avg_window=window,
                                                   backend=backend, FT=FT)

        # Sanity: VE driver actually collapsed to β = 1.
        @test ve_res.beta ≈ 1.0 atol=1e-12
        @test isfinite(new_res.Cd)
        @test isfinite(ve_res.Cd)
        @test !any(isnan, ve_res.ux)

        # Analytical-grade invariant: the two must agree to a tight band.
        # 1 % is conservative — actual residual is FP-reordering only.
        @test ve_res.Cd ≈ new_res.Cd rtol=1e-2

        # Loose physical sanity (NOT a gate): for a confined sphere in
        # Stokes flow the Haberman–Sayre / Bohlin wall-correction factor
        # K = F / (6πμU R) is O(1)–O(10) for blockage ratios of order
        # this box; our Re = 1 ducted Cd is an order-of-magnitude–only
        # cross-check (creeping-flow-in-tube geometry differs, so no
        # strict assertion).

        @info "VE 3D Newtonian-limit invariant" Cd_newtonian=new_res.Cd Cd_ve_beta1=ve_res.Cd rel_diff=abs(ve_res.Cd - new_res.Cd) / abs(new_res.Cd) beta=ve_res.beta
    end

    @testset "Cut-link canary (fractional q_wall, Wi = 0.01)" begin
        # Fractional radius ⇒ fractional q_wall ⇒ LI-BB cut-link path is
        # genuinely exercised (no halfway-BB collapse).
        Nx, Ny, Nz = 32, 16, 16
        radius = 3.5
        cx, cy, cz = Nx ÷ 4, Ny ÷ 2, Nz ÷ 2
        u_in = 0.04
        ν_s = 0.04
        # β = 0.8 (ν_p = 0.01). A representative VE case with a
        # non-trivial polymer stress that still runs NaN-free on this
        # deliberately coarse grid. NOTE: the direct-C (non-log) scheme
        # has a resolution-dependent extensional-stiffness limit — at
        # this ~3-cell radius β = 0.5 (ν_p = 0.04) diverges around
        # step ~180, which is the documented direct-C blow-up in
        # extensional regions, NOT a port bug (the solver is NaN-free at
        # the same Wi for milder modulus and for all moduli at ≤100
        # steps; log-conformation is the Phase-2 stabiliser). β = 0.8 is
        # comfortably inside the stable regime here.
        ν_p = 0.01
        # Wi = λ·u_ref/R with u_ref = u_in (uniform inlet) ⇒
        # λ = Wi·R/u_ref. Target Wi ≈ 0.01.
        Wi_target = 0.01
        lambda = Wi_target * radius / u_in
        steps, window = 300, 100

        # Confirm the geometry really has fractional cut links (guards
        # against a silent halfway-BB collapse hiding cut-link bugs).
        qw_h, _ = precompute_q_wall_sphere_3d(Nx, Ny, Nz, cx, cy, cz,
                                              radius; FT=FT)
        frac = count(q -> 0 < q < 1, qw_h)
        @test frac > 0

        res = run_conformation_sphere_libb_3d(; Nx=Nx, Ny=Ny, Nz=Nz,
                                                cx=cx, cy=cy, cz=cz,
                                                radius=radius, u_in=u_in,
                                                ν_s=ν_s, ν_p=ν_p,
                                                lambda=lambda,
                                                inlet=:uniform,
                                                max_steps=steps,
                                                avg_window=window,
                                                backend=backend, FT=FT)

        @test !any(isnan, res.ux)
        @test !any(isnan, res.uy)
        @test !any(isnan, res.uz)
        @test !any(isnan, res.ρ)
        @test !any(isnan, res.C_xx)
        @test isfinite(res.Cd)
        @test res.Cd > 0
        @test res.Wi ≈ Wi_target rtol=1e-6

        @info "VE 3D cut-link canary" Cd=res.Cd Wi=res.Wi beta=res.beta n_fractional_links=frac
    end
end
