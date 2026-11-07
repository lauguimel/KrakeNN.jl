using Test
using KrakeNN
using Random

@testset "platform residual Phase 2a (Enzyme-free)" begin

    @testset "LBMGeomParams construction" begin
        fwd = KrakeNN.ad_forward_solve(; Nx=16, Ny=8, cx=4.0, cy=4.0,
                                        radius=2.0, u_in=0.05, nu=0.05,
                                        inlet=:parabolic, tol=1e-10,
                                        max_steps=50_000)
        p = LBMGeomParams(fwd.q_wall, fwd.is_solid, fwd.u_profile,
                          fwd.rho_out, fwd.s_plus, fwd.s_minus, fwd.Nx, fwd.Ny)
        @test p.Nx == 16
        @test p.Ny == 8
        @test size(p.q_wall) == (16, 8, 9)
        @test size(p.is_solid) == (16, 8)
    end

    @testset "residual returns same size as input" begin
        fwd = KrakeNN.ad_forward_solve(; Nx=16, Ny=8, cx=4.0, cy=4.0,
                                        radius=2.0, u_in=0.05, nu=0.05,
                                        inlet=:parabolic, tol=1e-10,
                                        max_steps=50_000)
        p = LBMGeomParams(fwd.q_wall, fwd.is_solid, fwd.u_profile,
                          fwd.rho_out, fwd.s_plus, fwd.s_minus, fwd.Nx, fwd.Ny)
        R = residual(nothing, LBM(), fwd.f_star, p)
        @test size(R) == size(fwd.f_star)
        @test eltype(R) == Float64
    end

    @testset "residual at fixed point < 1e-10" begin
        fwd = KrakeNN.ad_forward_solve(; Nx=16, Ny=8, cx=4.0, cy=4.0,
                                        radius=2.0, u_in=0.05, nu=0.05,
                                        inlet=:parabolic, tol=1e-10,
                                        max_steps=50_000)
        @test fwd.converged
        p = LBMGeomParams(fwd.q_wall, fwd.is_solid, fwd.u_profile,
                          fwd.rho_out, fwd.s_plus, fwd.s_minus, fwd.Nx, fwd.Ny)
        R = residual(nothing, LBM(), fwd.f_star, p)
        rel = sqrt(sum(abs2, R)) / sqrt(sum(abs2, fwd.f_star))
        @test rel < 1e-10
    end

    @testset "LBMThermalParams construction" begin
        fwd = KrakeNN.ad_thermal_forward_solve(; N=8, Ra=1e3, Pr=0.71,
                                                tol=1e-10, max_steps=100_000)
        @test fwd.converged
        p = LBMThermalParams(fwd.q_wall, fwd.params, fwd.Nx, fwd.Ny)
        @test p.Nx == 8
        @test p.Ny == 8
        @test size(p.q_wall) == size(fwd.q_wall)
    end

    @testset "thermal residual at fixed point" begin
        fwd = KrakeNN.ad_thermal_forward_solve(; N=8, Ra=1e3, Pr=0.71,
                                                tol=1e-10, max_steps=100_000)
        @test fwd.converged
        p = LBMThermalParams(fwd.q_wall, fwd.params, fwd.Nx, fwd.Ny)
        R = residual(nothing, LBM(), fwd.w_star, p)
        @test length(R) == length(fwd.w_star)
        rel = sqrt(sum(abs2, R)) / sqrt(sum(abs2, fwd.w_star))
        @test rel < 1e-9
    end

    @testset "LBMVEParams construction" begin
        Nx, Ny = 12, 12
        s_plus, s_minus = KrakeNN.ad_ve_trt_rates(0.08)
        p_ve = KrakeNN.ADVECoupledParams(Nx, Ny, 0.5, 0.05, 4, 0.04, 0.08,
                                        2e-4, s_plus, s_minus)
        geom = KrakeNN.ad_ve_build_geom(Nx, Ny, 6.35, 5.65, 2.5;
                                       samples=8, u_mean=2e-4)
        p = LBMVEParams(geom.g, geom.q_wall, geom.u_profile, p_ve)
        @test p.p.Nx == Nx
        @test p.p.Ny == Ny
    end

    @testset "SteadyResidual in LBM capabilities" begin
        @test SteadyResidual in capabilities(LBM())
    end

    @testset "capabilities introspection (updated)" begin
        caps = capabilities(LBM())
        @test ForwardSolve in caps
        @test GPUExecution in caps
        @test SteadyAdjoint in caps
        @test SteadyResidual in caps
    end
end

@testset "platform residual Phase 2b-1 (Enzyme-free)" begin

    @testset "LBMScalarParams construction from LBMGeomParams + ν" begin
        fwd = KrakeNN.ad_forward_solve(; Nx=16, Ny=8, cx=4.0, cy=4.0,
                                        radius=2.0, u_in=0.05, nu=0.05,
                                        inlet=:parabolic, tol=1e-10,
                                        max_steps=50_000)
        geom = LBMGeomParams(fwd.q_wall, fwd.is_solid, fwd.u_profile,
                              fwd.rho_out, fwd.s_plus, fwd.s_minus, fwd.Nx, fwd.Ny)
        ν = 0.05
        p = LBMScalarParams(geom, ν)
        @test p.ν == ν
        @test p.Nx == 16
        @test p.Ny == 8
        s_p, s_m = KrakeNN.ad_trt_rates_inline(ν)
        @test p.s_plus ≈ s_p
        @test p.s_minus ≈ s_m
    end

    @testset "LBMScalarParams residual parity vs LBMGeomParams (bit-exact)" begin
        fwd = KrakeNN.ad_forward_solve(; Nx=16, Ny=8, cx=4.0, cy=4.0,
                                        radius=2.0, u_in=0.05, nu=0.05,
                                        inlet=:parabolic, tol=1e-10,
                                        max_steps=50_000)
        geom = LBMGeomParams(fwd.q_wall, fwd.is_solid, fwd.u_profile,
                              fwd.rho_out, fwd.s_plus, fwd.s_minus, fwd.Nx, fwd.Ny)
        p_scalar = LBMScalarParams(geom, 0.05)
        R_geom = residual(nothing, LBM(), fwd.f_star, geom)
        R_scalar = residual(nothing, LBM(), fwd.f_star, p_scalar)
        @test maximum(abs.(R_scalar .- R_geom)) == 0.0
    end

    @testset "LBMScalarParams residual at fixed point < 1e-10" begin
        fwd = KrakeNN.ad_forward_solve(; Nx=16, Ny=8, cx=4.0, cy=4.0,
                                        radius=2.0, u_in=0.05, nu=0.05,
                                        inlet=:parabolic, tol=1e-10,
                                        max_steps=50_000)
        @test fwd.converged
        geom = LBMGeomParams(fwd.q_wall, fwd.is_solid, fwd.u_profile,
                              fwd.rho_out, fwd.s_plus, fwd.s_minus, fwd.Nx, fwd.Ny)
        p = LBMScalarParams(geom, 0.05)
        R = residual(nothing, LBM(), fwd.f_star, p)
        rel = sqrt(sum(abs2, R)) / sqrt(sum(abs2, fwd.f_star))
        @test rel < 1e-10
    end

    @testset "ENZYME_FREE_OK: LBMScalarParams exported, extension check" begin
        @test isdefined(KrakeNN, :LBMScalarParams)
        if Base.get_extension(KrakeNN, :KrakeNNADExt) === nothing
            println("ENZYME_FREE_OK: extension=nothing, LBMScalarParams defined")
        end
    end
end

if Base.get_extension(KrakeNN, :KrakeNNADExt) !== nothing
    using LinearAlgebra

    @testset "adjoint_vjp bit-exact vs internal (Phase 2a, Enzyme)" begin

        @testset "Newtonian VJP parity" begin
            fwd = KrakeNN.ad_forward_solve(; Nx=48, Ny=16, cx=12.0, cy=8.0,
                                            radius=3.75, u_in=0.05, nu=0.05,
                                            inlet=:parabolic, tol=1e-12,
                                            max_steps=120_000)
            @test fwd.converged
            p = LBMGeomParams(fwd.q_wall, fwd.is_solid, fwd.u_profile,
                              fwd.rho_out, fwd.s_plus, fwd.s_minus, fwd.Nx, fwd.Ny)
            rng = Random.MersenneTwister(0xdeadbeef)
            v = rand(rng, size(fwd.f_star)...)

            internal = v .- KrakeNN._ad_vjp_GtT(fwd.f_star, v, fwd.q_wall,
                              fwd.is_solid, fwd.u_profile, fwd.rho_out,
                              fwd.s_plus, fwd.s_minus, fwd.Nx, fwd.Ny)
            exposed = adjoint_vjp(nothing, LBM(), fwd.f_star, p, v)

            delta = norm(exposed .- internal) / norm(internal)
            @test delta == 0.0
        end

        @testset "Thermal VJP parity" begin
            fwd = KrakeNN.ad_thermal_forward_solve(; N=16, Ra=1e3, Pr=0.71,
                                                    tol=1e-11, max_steps=450_000)
            @test fwd.converged
            p = LBMThermalParams(fwd.q_wall, fwd.params, fwd.Nx, fwd.Ny)
            rng = Random.MersenneTwister(0xcafe1234)
            v = rand(rng, length(fwd.w_star))

            internal = v .- KrakeNN._ad_thermal_vjp_GtT(fwd.w_star, v,
                              fwd.q_wall, fwd.q_wall, fwd.params)
            exposed = adjoint_vjp(nothing, LBM(), fwd.w_star, p, v)

            delta = norm(exposed .- internal) / norm(internal)
            @test delta == 0.0
        end

        @testset "VE VJP parity" begin
            Nx, Ny = 24, 24
            s_plus, s_minus = KrakeNN.ad_ve_trt_rates(0.08)
            p_ve = KrakeNN.ADVECoupledParams(Nx, Ny, 0.5, 0.05, 4,
                                            0.02/0.5, 0.08, 2e-4,
                                            s_plus, s_minus)
            geom = KrakeNN.ad_ve_build_geom(Nx, Ny, 12.35, 11.65, 5.13;
                                           samples=16, u_mean=2e-4)
            w0 = KrakeNN.ad_ve_initial_state(geom.g, Nx, Ny, 0.05)
            fwd = KrakeNN.ad_ve_forward_solve(w0, geom, p_ve; fwd_tol=1e-13)
            @test fwd.converged

            p = LBMVEParams(geom.g, geom.q_wall, geom.u_profile, p_ve)
            rng = Random.MersenneTwister(0xabcd5678)
            v = rand(rng, length(fwd.w_star))

            internal = v .- KrakeNN._ad_ve_vjp_GtT(fwd.w_star, v, geom.g,
                              geom.q_wall, geom.u_profile, p_ve)
            exposed = adjoint_vjp(nothing, LBM(), fwd.w_star, p, v)

            delta = norm(exposed .- internal) / norm(internal)
            @test delta == 0.0
        end
    end
end

if Base.get_extension(KrakeNN, :KrakeNNADExt) !== nothing
    using LinearAlgebra

    @testset "Phase 2b-1 Enzyme-gated" begin

        @testset "LBMScalarParams adjoint_vjp parity vs LBMGeomParams" begin
            fwd = KrakeNN.ad_forward_solve(; Nx=32, Ny=16, cx=100.0, cy=8.0,
                                            radius=0.001, u_in=0.05, nu=0.05,
                                            inlet=:parabolic, tol=1e-12,
                                            max_steps=200_000)
            @test fwd.converged
            geom = LBMGeomParams(fwd.q_wall, fwd.is_solid, fwd.u_profile,
                                  fwd.rho_out, fwd.s_plus, fwd.s_minus, fwd.Nx, fwd.Ny)
            p_scalar = LBMScalarParams(geom, 0.05)
            rng = Random.MersenneTwister(0xdeadbeef)
            v = rand(rng, size(fwd.f_star)...)
            vjp_geom = adjoint_vjp(nothing, LBM(), fwd.f_star, geom, v)
            vjp_scalar = adjoint_vjp(nothing, LBM(), fwd.f_star, p_scalar, v)
            @test norm(vjp_scalar .- vjp_geom) == 0.0
        end

        @testset "ν gradient check: adjoint dL/dν vs central-FD (rel < 1e-2)" begin
            function ux_profile_from_f(f, x_obs::Int, Ny::Int)
                ux = zeros(Float64, Ny)
                @inbounds for j in 1:Ny
                    rho = sum(@view f[x_obs, j, :])
                    ux[j] = (f[x_obs, j, 2] - f[x_obs, j, 4] +
                             f[x_obs, j, 6] - f[x_obs, j, 7] -
                             f[x_obs, j, 8] + f[x_obs, j, 9]) / rho
                end
                return ux
            end

            ν_true = 0.05
            ν0 = 0.06
            Nx, Ny = 32, 16

            fwd_true = KrakeNN.ad_forward_solve(; Nx=Nx, Ny=Ny, cx=100.0, cy=8.0,
                                                 radius=0.001, u_in=0.05, nu=ν_true,
                                                 inlet=:parabolic, tol=1e-12,
                                                 max_steps=200_000)
            @test fwd_true.converged

            x_obs = Nx ÷ 2
            data_ux = ux_profile_from_f(fwd_true.f_star, x_obs, Ny)

            fwd0 = KrakeNN.ad_forward_solve(; Nx=Nx, Ny=Ny, cx=100.0, cy=8.0,
                                             radius=0.001, u_in=0.05, nu=ν0,
                                             inlet=:parabolic, tol=1e-12,
                                             max_steps=200_000)
            @test fwd0.converged

            pred_ux = ux_profile_from_f(fwd0.f_star, x_obs, Ny)
            L0 = 0.5 * sum(abs2, pred_ux .- data_ux)
            @test isfinite(L0)

            geom0 = LBMGeomParams(fwd0.q_wall, fwd0.is_solid, fwd0.u_profile,
                                   fwd0.rho_out, fwd0.s_plus, fwd0.s_minus, Nx, Ny)
            p0 = LBMScalarParams(geom0, ν0)

            cx_q = [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0]
            dLdf = zeros(size(fwd0.f_star))
            for j in 1:Ny
                rho_xj = sum(@view fwd0.f_star[x_obs, j, :])
                ux_xj = pred_ux[j]
                res_j = pred_ux[j] - data_ux[j]
                for q in 1:9
                    dLdf[x_obs, j, q] = res_j * (cx_q[q] - ux_xj) / rho_xj
                end
            end

            apply_GtT = v -> KrakeNN._ad_vjp_GtT(fwd0.f_star, v, fwd0.q_wall,
                                                 fwd0.is_solid, fwd0.u_profile,
                                                 fwd0.rho_out, fwd0.s_plus,
                                                 fwd0.s_minus, Nx, Ny)
            adj = KrakeNN.gmres_adjoint(apply_GtT, dLdf;
                                       tol=1e-11, restart=240, max_restarts=20)
            @test adj.converged

            dLdν_adj = KrakeNN._ad_pvjp_nu(fwd0.f_star, adj.lambda, p0)

            h = 1e-4 * ν0
            fwd_p = KrakeNN.ad_forward_solve(; Nx=Nx, Ny=Ny, cx=100.0, cy=8.0,
                                              radius=0.001, u_in=0.05, nu=ν0+h,
                                              inlet=:parabolic, tol=1e-12,
                                              max_steps=200_000)
            fwd_m = KrakeNN.ad_forward_solve(; Nx=Nx, Ny=Ny, cx=100.0, cy=8.0,
                                              radius=0.001, u_in=0.05, nu=ν0-h,
                                              inlet=:parabolic, tol=1e-12,
                                              max_steps=200_000)
            @test fwd_p.converged
            @test fwd_m.converged

            pred_p = ux_profile_from_f(fwd_p.f_star, x_obs, Ny)
            pred_m = ux_profile_from_f(fwd_m.f_star, x_obs, Ny)
            Lp = 0.5 * sum(abs2, pred_p .- data_ux)
            Lm = 0.5 * sum(abs2, pred_m .- data_ux)
            dLdν_fd = (Lp - Lm) / (2h)

            rel = abs(dLdν_adj - dLdν_fd) / max(abs(dLdν_fd), 1e-15)
            @info "ν gradient check" dLdν_adj dLdν_fd rel h
            @test rel < 1e-2
        end
    end
end

# ============================================================================
# M-P2c-1 tests — LBMFieldParams + field VJP (ad_step_nufield! + _ad_pvjp_nufield)
# ============================================================================

@testset "platform residual Phase 2c-1 (Enzyme-free)" begin

    @testset "LBMFieldParams construction from LBMGeomParams + nu_field" begin
        fwd = KrakeNN.ad_forward_solve(; Nx=16, Ny=8, cx=4.0, cy=4.0,
                                        radius=2.0, u_in=0.05, nu=0.05,
                                        inlet=:parabolic, tol=1e-10,
                                        max_steps=50_000)
        @test fwd.converged
        geom = LBMGeomParams(fwd.q_wall, fwd.is_solid, fwd.u_profile,
                              fwd.rho_out, fwd.s_plus, fwd.s_minus, fwd.Nx, fwd.Ny)
        nu_field = fill(0.05, 8)
        p = LBMFieldParams(geom, nu_field)
        @test p.Nx == 16
        @test p.Ny == 8
        @test length(p.nu_field) == 8
        @test length(p.s_plus_field) == 8
        @test length(p.s_minus_field) == 8
        for j in 1:8
            s_p, s_m = KrakeNN.ad_trt_rates_inline(nu_field[j])
            @test p.s_plus_field[j] ≈ s_p
            @test p.s_minus_field[j] ≈ s_m
        end
    end

    @testset "LBMFieldParams: wrong nu_field length throws" begin
        fwd = KrakeNN.ad_forward_solve(; Nx=16, Ny=8, cx=4.0, cy=4.0,
                                        radius=2.0, u_in=0.05, nu=0.05,
                                        inlet=:parabolic, tol=1e-10,
                                        max_steps=50_000)
        geom = LBMGeomParams(fwd.q_wall, fwd.is_solid, fwd.u_profile,
                              fwd.rho_out, fwd.s_plus, fwd.s_minus, fwd.Nx, fwd.Ny)
        @test_throws ArgumentError LBMFieldParams(geom, fill(0.05, 5))
    end

    @testset "CONSISTENCY: ad_step_nufield! with uniform fill(ν,Ny) == ad_step! (bit-exact)" begin
        fwd = KrakeNN.ad_forward_solve(; Nx=16, Ny=8, cx=4.0, cy=4.0,
                                        radius=2.0, u_in=0.05, nu=0.05,
                                        inlet=:parabolic, tol=1e-10,
                                        max_steps=50_000)
        @test fwd.converged
        nu0 = 0.05
        s_plus, s_minus = KrakeNN.ad_trt_rates_inline(nu0)
        nu_field = fill(nu0, 8)

        out_scalar = similar(fwd.f_star)
        KrakeNN.ad_step!(out_scalar, fwd.f_star, fwd.q_wall, fwd.is_solid,
                         fwd.u_profile, fwd.rho_out, s_plus, s_minus, 16, 8)

        out_field = similar(fwd.f_star)
        KrakeNN.ad_step_nufield!(out_field, fwd.f_star, fwd.q_wall, fwd.is_solid,
                                 fwd.u_profile, fwd.rho_out, nu_field, 16, 8)

        @test maximum(abs.(out_field .- out_scalar)) == 0.0
    end

    @testset "residual dispatch parity: LBMFieldParams(uniform) == LBMScalarParams" begin
        fwd = KrakeNN.ad_forward_solve(; Nx=16, Ny=8, cx=4.0, cy=4.0,
                                        radius=2.0, u_in=0.05, nu=0.05,
                                        inlet=:parabolic, tol=1e-10,
                                        max_steps=50_000)
        geom = LBMGeomParams(fwd.q_wall, fwd.is_solid, fwd.u_profile,
                              fwd.rho_out, fwd.s_plus, fwd.s_minus, fwd.Nx, fwd.Ny)
        p_scalar = LBMScalarParams(geom, 0.05)
        p_field  = LBMFieldParams(geom, fill(0.05, 8))
        R_scalar = residual(nothing, LBM(), fwd.f_star, p_scalar)
        R_field  = residual(nothing, LBM(), fwd.f_star, p_field)
        @test maximum(abs.(R_field .- R_scalar)) == 0.0
    end

    @testset "ENZYME_FREE_OK: LBMFieldParams exported" begin
        @test isdefined(KrakeNN, :LBMFieldParams)
    end

end

if Base.get_extension(KrakeNN, :KrakeNNADExt) !== nothing
    using LinearAlgebra

    @testset "Phase 2c-1 Enzyme-gated" begin

        @testset "C-1a: _ad_pvjp_nufield gradient probe (k=5, rel < 1e-2)" begin
            Nx, Ny = 16, 8
            nu0 = 0.05
            nu_field = fill(nu0, Ny)

            fwd = KrakeNN.ad_forward_solve(; Nx=Nx, Ny=Ny, cx=4.0, cy=4.0,
                                            radius=2.0, u_in=0.05, nu=nu0,
                                            inlet=:parabolic, tol=1e-12,
                                            max_steps=200_000)
            @test fwd.converged

            rng = Random.MersenneTwister(123)
            lambda = randn(rng, size(fwd.f_star)...)

            dnu_adj = KrakeNN._ad_pvjp_nufield(fwd.f_star, lambda, nu_field,
                                               fwd.q_wall, fwd.is_solid,
                                               fwd.u_profile, fwd.rho_out, Nx, Ny)

            @test length(dnu_adj) == Ny

            h = 1e-5
            probes = [2, 3, 5, 6, 8]
            for j in probes
                nuf_p = copy(nu_field); nuf_p[j] += h
                nuf_m = copy(nu_field); nuf_m[j] -= h
                out_p = similar(fwd.f_star)
                out_m = similar(fwd.f_star)
                KrakeNN.ad_step_nufield!(out_p, fwd.f_star, fwd.q_wall, fwd.is_solid,
                                         fwd.u_profile, fwd.rho_out, nuf_p, Nx, Ny)
                KrakeNN.ad_step_nufield!(out_m, fwd.f_star, fwd.q_wall, fwd.is_solid,
                                         fwd.u_profile, fwd.rho_out, nuf_m, Nx, Ny)
                dnu_fd_j = dot(lambda, (out_p .- out_m) ./ (2h))
                rel = abs(dnu_adj[j] - dnu_fd_j) / max(abs(dnu_fd_j), 1e-15)
                @info "C-1a probe j=$j" dnu_adj=dnu_adj[j] dnu_fd=dnu_fd_j rel=rel h=h
                @test rel < 1e-2
            end
        end

        @testset "C-1b: _ad_vjp_GtT_nufield state VJP probe (k=3, rel < 1e-2)" begin
            Nx, Ny = 16, 8
            nu0 = 0.05
            nu_field = fill(nu0, Ny)

            fwd = KrakeNN.ad_forward_solve(; Nx=Nx, Ny=Ny, cx=4.0, cy=4.0,
                                            radius=2.0, u_in=0.05, nu=nu0,
                                            inlet=:parabolic, tol=1e-12,
                                            max_steps=200_000)
            @test fwd.converged

            rng = Random.MersenneTwister(456)
            v = randn(rng, size(fwd.f_star)...)

            df_adj = KrakeNN._ad_vjp_GtT_nufield(fwd.f_star, v, fwd.q_wall, fwd.is_solid,
                                                  fwd.u_profile, fwd.rho_out,
                                                  nu_field, Nx, Ny)
            @test size(df_adj) == size(fwd.f_star)

            # FD on state: dG^T v ≈ [G(f+h*e_k, nu_field) - G(f-h*e_k, nu_field)] / (2h)
            # contracted with v: dot(v, delta) / (2h). Compare with df_adj[k].
            h = 1e-5
            # Probe 3 random linear indices
            n_total = length(fwd.f_star)
            rng2 = Random.MersenneTwister(789)
            probe_idx = rand(rng2, 1:n_total, 3)

            for k in probe_idx
                f_p = copy(fwd.f_star); f_p[k] += h
                f_m = copy(fwd.f_star); f_m[k] -= h
                out_p = similar(fwd.f_star)
                out_m = similar(fwd.f_star)
                KrakeNN.ad_step_nufield!(out_p, f_p, fwd.q_wall, fwd.is_solid,
                                         fwd.u_profile, fwd.rho_out, nu_field, Nx, Ny)
                KrakeNN.ad_step_nufield!(out_m, f_m, fwd.q_wall, fwd.is_solid,
                                         fwd.u_profile, fwd.rho_out, nu_field, Nx, Ny)
                dGk_fd = dot(v, (out_p .- out_m) ./ (2h))
                rel = abs(df_adj[k] - dGk_fd) / max(abs(dGk_fd), 1e-15)
                @info "C-1b probe k=$k" adj=df_adj[k] fd=dGk_fd rel=rel
                @test rel < 1e-2
            end
        end

    end
end
