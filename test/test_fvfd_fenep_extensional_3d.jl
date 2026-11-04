using Test
using Kraken
using KernelAbstractions

# ---------------------------------------------------------------------
# FENE-P (Peterlin) 3D log-conformation planar-EXTENSION canary.
#
# Scope: the FENE-P coupling inside run_viscoelastic_fvfd_extensional_3d
# (velocity_mode=:imposed). Mirror of the Poiseuille FENE-P canary
# (test_fvfd_fenep_coupled_3d.jl) but on the planar-extension driver.
#
# Verification strategy (avoids a closed-form transcendental finite-L
# extensional fixed point — mirror of the session-3 FENE-P approach):
#   G1  OB-limit: LogConfFENEP with L²→very large ≡ the Oldroyd-B
#       extensional result (C_xx≈2.0, C_yy≈2/3) to ~1e-7. The OB path is
#       bit-identical when f≡1 (L²=Inf), so the large-L² run matches the
#       LogConfOldroydB run to round-off.
#   G2  finite-L²: a finite L² gives a BOUNDED, REDUCED stretch:
#       C_xx < 2 (OB value), trC < L² (Peterlin cap), NaN-free.
#   G3  .krk extensional_fenep.krk runs via name-dispatch and produces the
#       FENE-P extensional result.
#
# NOTE: every binding lives INSIDE the @testset local scope — nothing is
# declared at top level / `const`, so the file leaks no globals into Main
# (a top-level `const G` collides with other suites' globals in-suite).
# ---------------------------------------------------------------------

@testset "FVFD 3D FENE-P planar-extension" begin

    backend = CPU()
    FT = Float64

    # Shared planar-extension setup (β = 0.5, ν_total = 0.1, 2λε̇ = 0.5 < 1).
    Nx, Ny, Nz = 24, 24, 6
    ν_total = 0.1
    β = 0.5
    ν_s = β * ν_total
    ν_p = (1 - β) * ν_total
    λ = 50.0
    ε̇ = 0.005
    λε̇ = λ * ε̇                                  # = 0.25
    max_steps = 1_000
    Gmod = FT(ν_p / λ)

    # Oldroyd-B planar-extension fixed point.
    Cxx_ref = 1 / (1 - 2 * λε̇)                   # = 2.0
    Cyy_ref = 1 / (1 + 2 * λε̇)                   # = 2/3

    center_mean = function (A)
        nx, ny, nz = size(A)
        i1 = max(1, nx ÷ 2)
        i2 = min(nx, nx ÷ 2 + 1)
        j1 = max(1, ny ÷ 2)
        j2 = min(ny, ny ÷ 2 + 1)
        return sum(@view A[i1:i2, j1:j2, :]) / ((i2 - i1 + 1) * (j2 - j1 + 1) * nz)
    end

    run_ext = model -> Kraken.run_viscoelastic_fvfd_extensional_3d(;
        Nx=Nx, Ny=Ny, Nz=Nz, epsilon_dot=ε̇,
        ν_s=ν_s, ν_p=nothing, lambda=λ,
        polymer_model=model, max_steps=max_steps, backend=backend, FT=FT,
        advection_scheme=:muscl_superbee, velocity_mode=:imposed,
    )

    finite_all = res ->
        all(isfinite, res.ux) && all(isfinite, res.uy) && all(isfinite, res.uz) &&
        all(isfinite, res.C_xx) && all(isfinite, res.C_yy) && all(isfinite, res.C_zz) &&
        all(isfinite, res.psi_xx) && all(isfinite, res.psi_yy) && all(isfinite, res.psi_zz) &&
        all(isfinite, res.tau_p_xx) && all(isfinite, res.tau_p_yy) && all(isfinite, res.tau_p_zz)

    ob_model = LogConfOldroydB(G=Gmod, λ=FT(λ))
    res_ob = run_ext(ob_model)

    @testset "G0 OB reference recovers the analytical fixed point" begin
        @test res_ob.completed_steps == max_steps
        @test finite_all(res_ob)
        cxx = center_mean(res_ob.C_xx)
        cyy = center_mean(res_ob.C_yy)
        @test abs(cxx - Cxx_ref) / Cxx_ref <= 0.01
        @test abs(cyy - Cyy_ref) / Cyy_ref <= 0.01
        println("G0 OB Cxx=$(cxx) (ref $(Cxx_ref)) Cyy=$(cyy) (ref $(Cyy_ref))")
    end

    @testset "G1 OB-limit (L²→∞) ≡ Oldroyd-B to ~1e-7" begin
        L2_big = 1e8
        res_big = run_ext(LogConfFENEP(G=Gmod, λ=FT(λ), Lmax2=FT(L2_big)))
        @test res_big.completed_steps == max_steps
        @test finite_all(res_big)
        @test isfinite(res_big.L2_fene) && res_big.L2_fene == L2_big
        # Field-wise match to the OB run (Peterlin f≈1 over the whole box).
        cxx_rel = maximum(abs.(res_big.C_xx .- res_ob.C_xx)) /
                  maximum(abs.(res_ob.C_xx))
        cyy_rel = maximum(abs.(res_big.C_yy .- res_ob.C_yy)) /
                  maximum(abs.(res_ob.C_yy))
        cxx_c = center_mean(res_big.C_xx)
        cyy_c = center_mean(res_big.C_yy)
        println("G1 L²=$(L2_big): Cxx_rel=$(cxx_rel) Cyy_rel=$(cyy_rel) ",
                "Cxx=$(cxx_c) (OB-ref $(Cxx_ref)) Cyy=$(cyy_c) (OB-ref $(Cyy_ref))")
        @test cxx_rel <= 1e-7
        @test cyy_rel <= 1e-7
    end

    @testset "G2 finite-L² gives bounded, reduced stretch (< OB)" begin
        L2_fene = 50.0
        res_fp = run_ext(LogConfFENEP(G=Gmod, λ=FT(λ), Lmax2=FT(L2_fene)))
        @test res_fp.completed_steps == max_steps
        @test finite_all(res_fp)
        @test isfinite(res_fp.L2_fene) && res_fp.L2_fene == L2_fene

        cxx_fp = center_mean(res_fp.C_xx)
        cxx_ob = center_mean(res_ob.C_xx)
        trC = res_fp.C_xx .+ res_fp.C_yy .+ res_fp.C_zz
        max_trC = maximum(trC)
        println("G2 L²=$(L2_fene): Cxx(FENE-P)=$(cxx_fp) < Cxx(OB)=$(cxx_ob)≈$(Cxx_ref) ? ",
                "max(trC)=$(max_trC) < L²=$(L2_fene) ?")
        # Finite extensibility caps the polymer trace strictly below L².
        @test max_trC < L2_fene
        # Finite spring reduces the extensional stretch relative to OB (=2.0).
        @test cxx_fp < cxx_ob
        @test cxx_fp < Cxx_ref
        # Still a genuine stretch above equilibrium (C_xx = 1).
        @test cxx_fp > 1.0
    end

    @testset "G3 .krk extensional_fenep dispatches and runs" begin
        root = normpath(joinpath(@__DIR__, ".."))
        case_path = joinpath(root, "benchmarks", "krk", "viscoelastic",
                             "extensional_fenep.krk")
        cd(root) do
            setup = load_kraken(case_path)
            @test setup.lattice == :D3Q19
            @test :viscoelastic in setup.modules
            @test setup.max_steps == 1000

            r = run_simulation(case_path)
            @test r.velocity_mode === :imposed
            @test r.completed_steps == 1000
            @test all(isfinite, r.C_xx) && all(isfinite, r.C_yy) && all(isfinite, r.C_zz)
            @test isfinite(r.L2_fene) && r.L2_fene == 50.0

            cxx = center_mean(r.C_xx)
            trC = r.C_xx .+ r.C_yy .+ r.C_zz
            # FENE-P signature: bounded, reduced stretch.
            @test maximum(trC) < r.L2_fene
            @test cxx < Cxx_ref
            @test cxx > 1.0
            println("G3 KRK FENE-P extensional: Cxx=$(cxx) (< OB $(Cxx_ref)) ",
                    "max(trC)=$(maximum(trC)) < L²=$(r.L2_fene)")
        end
    end
end

println("EXIT=0")
