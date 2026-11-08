# T5 smoke: the three stages of benchmarks/fv_thermal/e5_compare.jl on a small
# synthetic E5-like cell (CPU, Float64). Only checks that every stage runs,
# converges and writes its outputs; no FV-vs-LBM threshold exists here (plan R6:
# thresholds come after the first GPU measurement).
#
# Synthetic case, 24 x 8 periodic cell with one 2 x 2 solid cylinder. Constants
# are the 300x100 E5 constants rescaled by f = 100/8 = 12.5 (Fx*f^2, s/f: same
# velocity, H smaller by f), except alpha and sigma, which are set to
# alpha/f*1e6 and sigma/f*1e4. Reason, measured 2026-11-08: with the true E5
# lattice coefficients (Pe = 800) the thermal Newton gate (scaled residual
# <= 1e-10, T ~ 468 K) is below the round-off floor and the line search fails as
# soon as one solid cell is present; this smoke therefore runs in a diffusive
# regime where the pipeline can be exercised. "Export" fields come from a periodic stream function on
# dyadic values, so the face velocities are exactly divergence-free in floating
# point; they are not an LBM solution.
# Run: sh tmp/thermal_julia.sh --project=. test/analytical/incns_fv_e5_compare_smoke_test.jl
using Test, Kraken, LinearAlgebra, Statistics, TOML

ENV["KRAKEN_E5_BACKEND"] = "cpu"
include(joinpath(@__DIR__, "..", "..", "benchmarks", "fv_thermal", "e5_compare.jl"))

function _e5s_case(; Nx = 24, Ny = 8)
    f = 100 / Ny
    consts = Dict{String,Any}("Nx" => Nx, "Ny" => Ny, "T_wall" => 468.15, "nuref" => 0.5,
        "eta_ref" => 4554.815680501838, "nu_per_eta" => 0.00010977392611964295,
        "s" => 1.0e7 / f, "alpha" => 6.25e-6 / f * 1e6, "sigma" => 45548.15680501838 / f * 1e4,
        "Fx" => 2.5950063712000005e-8 * f^2)
    theta = [2000.0, 0.4, 0.035, 0.3, 0.006, 1.5, 465.15, 2.5]
    solid = falses(Nx, Ny); solid[11:12, 4:5] .= true
    # stream function at cell NE corners, dyadic, periodic in i, 0 at the bottom, Q at the top
    q(x) = round(x * 2^24) / 2^24
    psi = [q(5e-5 * (3(j / Ny)^2 - 2(j / Ny)^3) * Ny + 1e-5 * sin(2pi * i / Nx) * sin(pi * j / Ny)) for i in 1:Nx, j in 0:Ny]
    psi[:, end] .= psi[1, end]
    psi[10:12, 4:6] .= psi[10, 4]   # corners (i=10:12, j=3:5) of the 2x2 solid: no flux through its faces
    uf = [psi[i, j+1] - psi[i, j] for i in 1:Nx, j in 1:Ny]
    vf = [-(psi[i, j+1] - psi[mod1(i - 1, Nx), j+1]) for i in 1:Nx, j in 1:Ny]
    uf[solid] .= 0; vf[solid] .= 0
    ux = [(uf[mod1(i - 1, Nx), j] + uf[i, j]) / 2 for i in 1:Nx, j in 1:Ny]
    uy = [((j > 1 ? vf[i, j-1] : 0.0) + vf[i, j]) / 2 for i in 1:Nx, j in 1:Ny]
    ux[solid] .= 0; uy[solid] .= 0
    Tw = consts["T_wall"]
    T = [Tw + 20 * (1 - (j - 0.5) / Ny) * (solid[i, j] ? 0 : 1) + 2 * (j == 1 && !solid[i, j]) for i in 1:Nx, j in 1:Ny]
    I1 = [((ux[mod1(i + 1, Nx), j] - ux[mod1(i - 1, Nx), j]) / 2)^2 + ((j < Ny ? uy[i, j+1] : 0.0) - (j > 1 ? uy[i, j-1] : 0.0))^2 / 4 +
          ((j < Ny ? ux[i, j+1] : 0.0) - (j > 1 ? ux[i, j-1] : 0.0))^2 / 8 for i in 1:Nx, j in 1:Ny]
    I1[solid] .= 0
    law = E5ThermalLaw(theta; T_wall = Tw, nu_ref = consts["nuref"], shear_scale = consts["s"])
    nu = [solid[i, j] ? 0.0 : Kraken.law_value(law, (sqrt(2 * I1[i, j]), T[i, j]), theta) for i in 1:Nx, j in 1:Ny]
    return E5Compare.Case("synthetic", Nx, Ny, consts, theta, ux, uy, uf, vf, T, solid, nu, I1,
                          nothing, nothing, nothing, Dict{String,Any}())
end

@testset "E5 comparison driver smoke (CPU, synthetic 24x8)" begin
    cs = _e5s_case(); outdir = mktempdir()
    @testset "export check" begin
        chk = E5Compare.check_export(cs)
        @test chk["n_fluid_cells"] == 24 * 8 - 4
        @test chk["max_abs_face_divergence_over_max_uf"] <= 1e-12
    end
    @testset "stage 1: isothermal flow, prescribed T" begin
        r = E5Compare.stage1(cs; outdir)
        @test r["converged"]
        @test isfile(joinpath(outdir, "stage1_fields.csv"))
        @test isfinite(r["flow_rate_ratio_fv_over_lbm"]) && r["flow_rate_ratio_fv_over_lbm"] > 0
    end
    @testset "stage 2: heat only, prescribed faces and source" begin
        r = E5Compare.stage2(cs; outdir)
        @test r["converged"]
        @test isfile(joinpath(outdir, "stage2_fields.csv"))
        @test isfinite(r["m_fv"]) && r["m_fv"] > 0
        # zero source and constant T: the solve must return T_wall exactly
        z = E5Compare.solve_heat_only(cs, zeros(cs.Nx, cs.Ny), cs.uf, cs.vf)
        @test z.stats.converged
        @test all(z.T .== cs.consts["T_wall"])
    end
    @testset "stage 3: full coupling, continuation, adjoint, FD" begin
        r = E5Compare.stage3(cs; outdir, fd = true)
        @test r["converged"]
        @test isfile(joinpath(outdir, "stage3_fields.csv"))
        @test length(r["dm_dtheta_fv"]) == 8 && all(isfinite, r["dm_dtheta_fv"])
        @test r["m_from_adjoint_objective"] ≈ r["m_fv"] rtol = 1e-10
        for name in ("eta0", "beta")
            @test r["finite_difference"][name]["converged"]
            @test isfinite(r["finite_difference"][name]["relative_difference"])
        end
        @info "E5 smoke" steps = r["continuation_steps"] newton = r["newton_iterations_total"] fd_eta0 = r["finite_difference"]["eta0"]["relative_difference"] fd_beta = r["finite_difference"]["beta"]["relative_difference"]
    end
    @testset "summary is TOML-serialisable" begin
        path = joinpath(outdir, "summary.toml")
        E5Compare.write_summary(path, Dict{String,Any}("export_check" => E5Compare.check_export(cs),
            "stage1" => E5Compare.stage1(cs)))
        @test haskey(TOML.parsefile(path), "stage1")
    end
    @testset "thresholds table" begin
        # the measured 600x200 stage-3 values of 2026-11-08 pass; the 300x100 ones are written with applies = false
        st = Dict{String,Any}("m_ratio_fv_over_lbm" => 1.0138840568996894,
            "flow_rate_ratio_fv_over_lbm" => 1.0135247710843704,
            "dm_dtheta_norm_rel_diff" => 0.024374570980317412,
            "dm_dtheta_componentwise_rel_diff" => [0.0128, 0.0083, 0.0455, 0.0142, 0.0242, 0.0122, 0.0133, 0.0145],
            "dm_dtheta_sign_agree" => fill(true, 8))
        t = E5Compare.thresholds("600x200", st)
        @test t["applies"] && t["all_pass"]
        @test t["gradient_worst_component_rel_diff"]["pass"] && t["gradient_signs_agree"]["pass"]
        bad = merge(st, Dict{String,Any}("m_ratio_fv_over_lbm" => 1.031))
        @test !E5Compare.thresholds("600x200", bad)["all_pass"]
        t3 = E5Compare.thresholds("300x100", bad)
        @test !t3["applies"] && t3["evaluated"] && !haskey(t3, "all_pass") && !haskey(t3["m_ratio_deviation"], "pass")
        t0 = E5Compare.thresholds("600x200", nothing)   # stage 3 not run: limits only
        @test !t0["evaluated"] && !haskey(t0["m_ratio_deviation"], "value") && !haskey(t0, "all_pass")
        path = joinpath(mktempdir(), "t.toml")
        E5Compare.write_summary(path, Dict{String,Any}("thresholds" => t))
        @test TOML.parsefile(path)["thresholds"]["all_pass"] === true
    end
end
