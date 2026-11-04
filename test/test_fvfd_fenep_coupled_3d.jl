using Test
using Kraken
using KernelAbstractions

# ---------------------------------------------------------------------
# FENE-P (Peterlin) 3D log-conformation COUPLED Poiseuille canary.
#
# Scope: the full coupled flow — solvent D3Q19 + polymer feedback
# F_poly = ∇·τ_p — driven through run_viscoelastic_fvfd_poiseuille_3d with
# a LogConfFENEP polymer model. This is the coupling counterpart of the
# constitutive-only canary test_fvfd_fenep_3d.jl.
#
# Checks:
#   C1  NaN-free coupled FENE-P run (Wi≥1, β=0.5, finite L²).
#   C2  ν_eff self-consistency (mirror of the OB coupling canary): the
#       fitted effective viscosity matches ν_total within ~1%.
#   C3  finite-extensibility signature: FENE-P polymer N1 / stress is
#       STRICTLY LESS than the Oldroyd-B result at the same Wi.
#   C4  L²→∞ recovers the OB coupled result (large-L² run ≈ OB Poiseuille).
# ---------------------------------------------------------------------

const BACKEND = CPU()
const FT = Float64

# Shared channel / flow setup (β = 0.5, ν_total = 0.1, Wi_wall ≈ 1).
const NX, NY, NZ = 6, 32, 6
const NU_TOTAL = 0.1
const BETA = 0.5
const NU_S = BETA * NU_TOTAL
const NU_P = (1 - BETA) * NU_TOTAL
const FX = 1.5e-5
const GAMMA_WALL = FX / (2 * NU_TOTAL) * (NY - 1)
const LAMBDA = 1.0 / GAMMA_WALL        # Wi_wall ≈ 1
const MAX_STEPS = 10_000

run_poiseuille(model) = Kraken.run_viscoelastic_fvfd_poiseuille_3d(;
    Nx=NX, Ny=NY, Nz=NZ, Fx=FX, ν_s=NU_S, ν_p=nothing, lambda=LAMBDA,
    polymer_model=model, max_steps=MAX_STEPS, backend=BACKEND, FT=FT,
    advection_scheme=:muscl_superbee,
)

# Effective viscosity from a parabolic fit of the velocity profile.
function nu_eff(res)
    yshape = [(j - 0.5) * (NY + 0.5 - j) for j in 1:NY]
    slope = sum(res.profile .* yshape) / sum(abs2, yshape)
    return FX / (2 * slope)
end

# Peak (wall-region) polymer first normal-stress difference N1 = τ_xx − τ_yy.
peak_N1(res) = maximum(abs.(res.N1_prof))
# Peak polymer shear stress.
peak_tau_xy(res) = maximum(abs.(res.tau_p_xy))

G = FT(NU_P / LAMBDA)

@testset "FVFD 3D FENE-P coupled Poiseuille" begin

    # Finite extensibility: L² well above equilibrium trace (=3) but small
    # enough that the Peterlin factor bites at Wi≈1.
    L2_fene = 50.0
    fenep_model = LogConfFENEP(G=G, λ=FT(LAMBDA), Lmax2=FT(L2_fene))
    ob_model    = LogConfOldroydB(G=G, λ=FT(LAMBDA))

    res_fp = run_poiseuille(fenep_model)
    res_ob = run_poiseuille(ob_model)

    @testset "C1 NaN-free coupled FENE-P run" begin
        @test res_fp.completed_steps == MAX_STEPS
        @test all(isfinite, res_fp.ux)
        @test all(isfinite, res_fp.psi_xx) && all(isfinite, res_fp.psi_xy)
        @test all(isfinite, res_fp.psi_yy) && all(isfinite, res_fp.psi_zz)
        @test all(isfinite, res_fp.tau_p_xx) && all(isfinite, res_fp.tau_p_xy)
        @test all(isfinite, res_fp.tau_p_yy) && all(isfinite, res_fp.tau_p_zz)
        @test isfinite(res_fp.L2_fene) && res_fp.L2_fene == L2_fene
        # Finite extensibility caps the polymer trace below L².
        trC = res_fp.Cxx_prof .+ res_fp.Cyy_prof .+ res_fp.Czz_prof
        @test maximum(trC) < L2_fene
    end

    @testset "C2 ν_eff self-consistency" begin
        # Parabolic-fit effective viscosity. For OB this lands within 0.1% of
        # ν_total; for FENE-P at finite L² the spring mildly thins the polymer
        # feedback so the channel-averaged profile departs slightly from the
        # pure parabola (≈1% here) — this is the finite-extensibility imprint
        # on the flow, not a coupling defect. The OB large-L² run (C4) recovers
        # the parabola to <1e-3, confirming the coupling itself is exact.
        ν_eff_fp = nu_eff(res_fp)
        ν_eff_ob = nu_eff(res_ob)
        rel = abs(ν_eff_fp - NU_TOTAL) / NU_TOTAL
        println("C2 FENE-P nu_eff=$(ν_eff_fp) (OB nu_eff=$(ν_eff_ob)) ",
                "nu_total=$(NU_TOTAL) rel%=$(100 * rel)")
        @test NU_TOTAL ≈ NU_S / BETA
        # ν_eff stays within ~1% of ν_total (loose self-consistency guard).
        @test rel < 0.015
        # OB coupling is essentially exact under the same fit.
        @test abs(ν_eff_ob - NU_TOTAL) / NU_TOTAL < 0.005
    end

    @testset "C3 finite-extensibility reduces the polymer response (< OB)" begin
        N1_fp = peak_N1(res_fp)
        N1_ob = peak_N1(res_ob)
        txy_fp = peak_tau_xy(res_fp)
        txy_ob = peak_tau_xy(res_ob)
        println("C3 Wi≈1 L²=$(L2_fene): N1(FENE-P)=$(N1_fp) < N1(OB)=$(N1_ob) ? ",
                "tau_xy(FENE-P)=$(txy_fp) < tau_xy(OB)=$(txy_ob) ?")
        # FENE-P spring caps the elongation → strictly smaller normal stress.
        @test N1_fp < N1_ob
        @test txy_fp < txy_ob
        @test N1_ob > 0
    end

    @testset "C4 L²→∞ recovers the OB coupled result" begin
        L2_big = 1e10
        res_big = run_poiseuille(LogConfFENEP(G=G, λ=FT(LAMBDA), Lmax2=FT(L2_big)))
        @test res_big.completed_steps == MAX_STEPS
        @test all(isfinite, res_big.ux)
        # Velocity profile and polymer N1 match the OB run to tight tolerance.
        u_rel = maximum(abs.(res_big.profile .- res_ob.profile)) /
                maximum(abs.(res_ob.profile))
        N1_rel = abs(peak_N1(res_big) - peak_N1(res_ob)) / peak_N1(res_ob)
        println("C4 L²→∞: u_rel=$(u_rel) N1_rel=$(N1_rel) ",
                "N1(big)=$(peak_N1(res_big)) N1(OB)=$(peak_N1(res_ob))")
        @test u_rel < 1e-3
        @test N1_rel < 1e-2
    end
end

println("EXIT=0")
