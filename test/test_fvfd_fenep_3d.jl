using Test
using KernelAbstractions
using Kraken

# ---------------------------------------------------------------------
# FENE-P (Peterlin) 3D log-conformation CONSTITUTIVE canary.
#
# Scope: the constitutive source ONLY (no coupled driver / stress / .krk).
# The correctness check is the Oldroyd-B (OB) limit: the Peterlin factor
# f = (L²−3)/(L²−trC) → 1 as L²→∞, so FENE-P → OB exactly. We verify this
# numerically against the OB analytical uniform-shear fixed point, then
# confirm that a finite L² caps trC < L² and reduces the shear response.
# ---------------------------------------------------------------------

const CELL_3D = (1, 1, 1)

field3(value) = fill(Float64(value), CELL_3D)

function psi_state(; xx=0.0, xy=0.0, xz=0.0, yy=0.0, yz=0.0, zz=0.0)
    return (
        field3(xx), field3(xy), field3(xz),
        field3(yy), field3(yz), field3(zz),
    )
end

function velocity_gradient(;
    duxdx=0.0, duxdy=0.0, duxdz=0.0,
    duydx=0.0, duydy=0.0, duydz=0.0,
    duzdx=0.0, duzdy=0.0, duzdz=0.0,
)
    return (
        field3(duxdx), field3(duxdy), field3(duxdz),
        field3(duydx), field3(duydy), field3(duydz),
        field3(duzdx), field3(duzdy), field3(duzdz),
    )
end

psi_values(psi) = ntuple(c -> psi[c][1, 1, 1], 6)
conformation_from_psi(psi) = Kraken.mat_exp_sym3x3(psi_values(psi)...)

function apply_fenep_step!(psi, grad, lambda, dt, L2_fene, n_sub)
    before = psi_values(psi)
    out = ntuple(c -> similar(psi[c]), 6)
    Kraken.logfv_constitutive_step_log_fenep_3d!(
        out...,
        psi...,
        grad...,
        lambda, dt, L2_fene, n_sub;
        sync=true,
    )
    for c in 1:6
        psi[c] .= out[c]
    end
    after = psi_values(psi)
    return maximum(abs(after[c] - before[c]) for c in 1:6)
end

# Integrate the FENE-P uniform-shear constitutive ODE to its fixed point.
function run_fenep_uniform_shear(Wi, L2_fene; lambda=1.0, dt=0.02)
    gamma_dot = Wi / lambda
    psi = psi_state()
    grad = velocity_gradient(duxdy=gamma_dot)
    max_grad_norm = Kraken.logfv_max_grad_norm_3d(grad...)
    # Stronger FENE-P relaxation near the wall → use a generous substep
    # count (OB estimator + headroom) so the RK2 step stays accurate.
    n_sub = 8 * Kraken.logfv_recommended_oldroydb_substeps_3d(max_grad_norm, lambda, dt)
    max_steps = ceil(Int, 200lambda / dt)
    t = 0.0
    delta = Inf
    steps = 0
    while steps < max_steps && delta >= 1e-13
        delta = apply_fenep_step!(psi, grad, lambda, dt, L2_fene, n_sub)
        steps += 1
        t += dt
    end
    C = conformation_from_psi(psi)
    return (C=C, trC=C[1] + C[4] + C[6], n_sub=n_sub, steps=steps, delta=delta)
end

@testset "FVFD 3D FENE-P log-conformation constitutive step" begin

    @testset "F1 NaN-free FENE-P uniform shear" begin
        res = run_fenep_uniform_shear(1.5, 50.0)
        cxx, cxy, cxz, cyy, cyz, czz = res.C
        @test all(isfinite, res.C)
        @test isfinite(res.trC)
        println(
            "F1 finite-L Wi=1.5 L2=50 n_sub=$(res.n_sub) steps=$(res.steps) ",
            "Cxx=$(cxx) Cxy=$(cxy) trC=$(res.trC) delta=$(res.delta)",
        )
        # Sanity: off-shear-plane components stay at equilibrium.
        @test abs(cxz) <= 1e-9
        @test abs(cyz) <= 1e-9
    end

    @testset "F2 OB limit (L²→∞)" begin
        # As L²→∞ the Peterlin factor → 1, recovering Oldroyd-B exactly.
        L2_big = 1e8
        for Wi in (0.5, 1.0, 2.0)
            res = run_fenep_uniform_shear(Wi, L2_big)
            cxx, cxy = res.C[1], res.C[2]
            target_cxy = Wi
            target_cxx = 1.0 + 2.0 * Wi^2
            err_cxy = abs(cxy - target_cxy) / target_cxy
            err_cxx = abs(cxx - target_cxx) / target_cxx
            println(
                "F2 OB-limit Wi=$(Wi) L2=$(L2_big) Cxy=$(cxy) (target $(target_cxy)) ",
                "Cxx=$(cxx) (target $(target_cxx)) err_Cxy=$(err_cxy) err_Cxx=$(err_cxx)",
            )
            @test err_cxy <= 1e-3
            @test err_cxx <= 1e-3
        end
    end

    @testset "F3 finite extensibility caps trC and reduces response" begin
        # Strong shear so the finite spring saturates well below the OB value.
        Wi = 3.0
        L2_fene = 50.0
        ob = run_fenep_uniform_shear(Wi, 1e8)        # OB reference (f≈1)
        fp = run_fenep_uniform_shear(Wi, L2_fene)    # finite extensibility
        cxx_ob = ob.C[1]
        cxx_fp = fp.C[1]
        println(
            "F3 Wi=$(Wi) L2=$(L2_fene): trC(FENE-P)=$(fp.trC) < L²=$(L2_fene) ? ",
            "Cxx(FENE-P)=$(cxx_fp) < Cxx(OB)=$(cxx_ob) ?",
        )
        # Finite extensibility: the trace is strictly capped below L².
        @test fp.trC < L2_fene
        @test isfinite(fp.trC)
        # Finite extensibility reduces the elongational/shear response.
        @test cxx_fp < cxx_ob
        # OB reference must match its analytical value (consistency guard).
        @test abs(cxx_ob - (1.0 + 2.0 * Wi^2)) / (1.0 + 2.0 * Wi^2) <= 1e-3
    end
end

println("EXIT=0")
