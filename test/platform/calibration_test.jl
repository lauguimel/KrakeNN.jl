using Test
using KrakeNN
using Random
using LinearAlgebra

const _CAL_ENZYME_OK = try
    @eval Main using Enzyme
    Base.get_extension(KrakeNN, :KrakeNNADExt) !== nothing
catch
    false
end

@testset "platform calibration Phase 2b (Enzyme-free)" begin

    @testset "ParameterSpace to_flat / from_flat round-trip (scalar ν)" begin
        ps = ParameterSpace([:ν], [0.01], [0.5])
        p0 = (ν=0.05,)
        x = KrakeNN.to_flat(ps, p0)
        @test length(x) == 1
        @test x[1] ≈ 0.05
        p_rt = KrakeNN.from_flat(ps, x, p0)
        @test p_rt[:ν] ≈ 0.05
    end

    @testset "ParameterSpace log-scale round-trip" begin
        ps = ParameterSpace([:ν], [0.001], [1.0]; log_scale=BitVector([true]))
        p0 = (ν=0.05,)
        x = KrakeNN.to_flat(ps, p0)
        @test x[1] ≈ log(0.05)
        p_rt = KrakeNN.from_flat(ps, x, p0)
        @test p_rt[:ν] ≈ 0.05 atol=1e-14
    end

    @testset "ParameterSpace fixed parameter excluded from flat" begin
        ps = ParameterSpace([:ν, :radius], [0.01, 0.1], [0.5, 10.0];
                            fixed=BitVector([false, true]))
        p0 = (ν=0.05, radius=3.0)
        x = KrakeNN.to_flat(ps, p0)
        @test length(x) == 1
        @test x[1] ≈ 0.05
        p_rt = KrakeNN.from_flat(ps, x, p0)
        @test p_rt[:ν] ≈ 0.05
        @test p_rt[:radius] ≈ 3.0
    end

    @testset "ParameterSpace project! clips bounds" begin
        ps = ParameterSpace([:ν], [0.01], [0.5])
        v = [0.8]
        KrakeNN.project!(ps, v)
        @test v[1] ≈ 0.5
        v2 = [0.001]
        KrakeNN.project!(ps, v2)
        @test v2[1] ≈ 0.01
    end

    @testset "ParameterSpace accepts field {ν_i} (smoke, Ny=8)" begin
        Ny = 8
        names = [Symbol("ν_$i") for i in 1:Ny]
        ps = ParameterSpace(names, fill(0.01, Ny), fill(0.5, Ny))
        @test KrakeNN.n_free(ps) == Ny
        p0 = NamedTuple{Tuple(names)}(Tuple(fill(0.05, Ny)))
        x = KrakeNN.to_flat(ps, p0)
        @test length(x) == Ny
        p_rt = KrakeNN.from_flat(ps, x, p0)
        @test all(p_rt[n] ≈ 0.05 for n in names)
    end

    @testset "loss returns 0 when predictions == data" begin
        obs = LineProfile(:ux, [(1, 1), (1, 2)])
        pred = Prediction(obs, [0.1, 0.2])
        data = [(observable=obs, value=[0.1, 0.2])]
        L = loss([pred], data)
        @test L ≈ 0.0
    end

    @testset "loss returns correct value when predictions ≠ data" begin
        obs = LineProfile(:ux, [(1, 1), (1, 2)])
        pred = Prediction(obs, [0.1, 0.3])
        data = [(observable=obs, value=[0.1, 0.2])]
        L = loss([pred], data)
        @test L ≈ 0.5 * (0.0^2 + 0.1^2)
    end

    @testset "CalibResult fields accessible" begin
        cr = CalibResult((ν=0.05,), 1.0, [2.0, 1.5, 1.0],
                         [0.5, 0.3, 0.1], 3, true, "test")
        @test cr.p_opt[:ν] ≈ 0.05
        @test cr.loss_final ≈ 1.0
        @test length(cr.loss_trace) == 3
        @test cr.converged == true
    end

    @testset "ENZYME_FREE_OK: exports present" begin
        @test isdefined(KrakeNN, :ParameterSpace)
        @test isdefined(KrakeNN, :loss)
        @test isdefined(KrakeNN, :fit)
        @test isdefined(KrakeNN, :CalibResult)
        println("ENZYME_FREE_OK: ParameterSpace/loss/fit/CalibResult all defined")
    end
end

if _CAL_ENZYME_OK
    @testset "M-P2b-2 Enzyme-gated" begin

        # Use a geometry with a real cylinder so that ν is identifiable from ux
        # (no-obstacle channel: ux insensitive to ν, gradient unreliable)
        _CAL_NX = 16
        _CAL_NY = 8
        _CAL_GEOM = (Nx=_CAL_NX, Ny=_CAL_NY, cx=4.0, cy=4.0,
                     radius=2.0, u_in=0.05, rho_out=1.0, inlet=:parabolic)

        function _ux_profile_at_midx(f_star, Nx, Ny)
            x_obs = Nx ÷ 2
            ux = zeros(Float64, Ny)
            cx_q = [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0]
            for j in 1:Ny
                rho = sum(@view f_star[x_obs, j, :])
                ux[j] = sum(cx_q[q] * f_star[x_obs, j, q] for q in 1:9) / rho
            end
            return ux
        end

        @testset "4b TWIN EXPERIMENT: ν recovery |ν_fit - ν_true|/ν_true < 5%" begin
            ν_true = 0.05
            ν0 = 0.07
            # σ=5e-4: recovery precision scales with noise floor (L_noise ∝ σ²,
            # δν ∝ σ); σ=1e-3 hits L_noise≈4e-6 before the gradient is useful
            # at ν_true=0.05. σ=5e-4 reduces the noise floor 4× so rel_err≲2.5%.
            σ = 5e-4

            fwd_true = KrakeNN.ad_forward_solve(; _CAL_GEOM..., nu=ν_true,
                                                tol=1e-12, max_steps=200_000)
            @test fwd_true.converged

            x_obs = _CAL_NX ÷ 2
            obs = LineProfile(:ux, [(x_obs, j) for j in 1:_CAL_NY])
            ux_true = _ux_profile_at_midx(fwd_true.f_star, _CAL_NX, _CAL_NY)

            rng = Random.MersenneTwister(42)
            ux_noisy = ux_true .+ σ .* randn(rng, _CAL_NY)
            data = [(observable=obs, value=ux_noisy)]

            ps = ParameterSpace([:ν], [0.01], [0.2])
            p0 = (ν=ν0,)
            cr = fit(_CAL_GEOM, LBM(), data, p0, ps;
                     observables=[obs],
                     max_iter=100,
                     step_size=0.5)

            ν_fit = cr.p_opt[:ν]
            rel_err = abs(ν_fit - ν_true) / ν_true
            @info "4b TWIN EXPERIMENT" ν_true ν0 ν_fit rel_err n_iter=cr.n_iter loss_final=cr.loss_final
            @test rel_err < 0.05
            @test cr.n_iter < 100
            @test cr.loss_final <= cr.loss_trace[1]
        end

        @testset "4c geometry parity: direct chain vs steady_shape_sensitivity" begin
            # Note: steady_shape_sensitivity uses ρ_out not rho_out; pass geom without rho_out
            geom_c3 = (Nx=48, Ny=16, cx=12.0, cy=8.0, radius=3.75,
                       u_in=0.05, inlet=:parabolic)
            ν_geom = 0.05

            ssa = KrakeNN.steady_shape_sensitivity(; qoi=:drag, wrt=:radius,
                                                    geom_c3..., nu=ν_geom,
                                                    tol=1e-12,
                                                    max_steps=120_000)
            @test ssa.gradient !== nothing

            fwd = KrakeNN.ad_forward_solve(; geom_c3..., nu=ν_geom,
                                            tol=1e-12, max_steps=120_000)
            @test fwd.converged

            rhs = KrakeNN._ad_dJdf(fwd.f_star, fwd.q_wall, fwd.u_ref,
                                  fwd.D, fwd.Nx, fwd.Ny)
            apply_GtT = v -> KrakeNN._ad_vjp_GtT(fwd.f_star, v, fwd.q_wall,
                                                 fwd.is_solid, fwd.u_profile,
                                                 fwd.rho_out, fwd.s_plus,
                                                 fwd.s_minus, fwd.Nx, fwd.Ny)
            adj = KrakeNN.gmres_adjoint(apply_GtT, rhs;
                                       tol=1e-11, restart=240, max_restarts=20)
            @test adj.converged

            dq_dR = KrakeNN.dq_wall_dR_cylinder(fwd.Nx, fwd.Ny, fwd.cx, fwd.cy,
                                                fwd.radius; FT=Float64)
            qwall_terms = KrakeNN._ad_dqwall_terms(fwd.f_star, adj.lambda,
                                                   fwd.q_wall, fwd.is_solid,
                                                   fwd.u_profile, fwd.rho_out,
                                                   fwd.s_plus, fwd.s_minus,
                                                   fwd.Nx, fwd.Ny, fwd.u_ref,
                                                   fwd.D, dq_dR)
            terms = KrakeNN.ad_assemble_radius_terms(fwd.Cd, fwd.radius, dq_dR,
                                                     qwall_terms.explicit,
                                                     qwall_terms.implicit)
            rel_parity = abs(terms.gradient - ssa.gradient) / max(abs(ssa.gradient), 1e-15)
            @info "4c geometry parity" grad_direct=terms.gradient ssa_gradient=ssa.gradient rel=rel_parity
            @test rel_parity < 1e-6
        end

        @testset "4d field-param smoke: ParameterSpace{ν_i} Ny=8" begin
            Ny = 8
            names = [Symbol("ν_$i") for i in 1:Ny]
            ps = ParameterSpace(names, fill(0.01, Ny), fill(0.5, Ny))
            p0 = NamedTuple{Tuple(names)}(Tuple(fill(0.05, Ny)))
            @test KrakeNN.n_free(ps) == Ny
            x = KrakeNN.to_flat(ps, p0)
            @test length(x) == Ny
            p_rt = KrakeNN.from_flat(ps, x, p0)
            @test all(p_rt[n] ≈ 0.05 for n in names)
            println("4d field-param smoke: ParameterSpace{ν_i} Ny=8 round-trip OK")
        end
    end
else
    @info "Skipping M-P2b-2 Enzyme-gated calibration tests (Enzyme extension not loadable)"
end
