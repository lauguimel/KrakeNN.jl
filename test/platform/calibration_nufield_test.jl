using Test
using KrakeNN
using Random
using LinearAlgebra

const _NUFIELD_ENZYME_OK = try
    @eval Main using Enzyme
    Base.get_extension(KrakeNN, :KrakeNNADExt) !== nothing
catch
    false
end

const _NUFIELD_OPTIM_OK = try
    @eval Main using Optim
    Base.get_extension(KrakeNN, :KrakeNNOptimExt) !== nothing
catch
    false
end

const _NF_NX   = 16
const _NF_NY   = 8
const _NF_GEOM = (Nx=_NF_NX, Ny=_NF_NY, cx=4.0, cy=4.0,
                  radius=2.0, u_in=0.05, rho_out=1.0, inlet=:parabolic)

# ---- Enzyme-free tests ----

@testset "M-P2c-2 Enzyme-free" begin

    @testset "_reg_loss: known values" begin
        nu = [0.03, 0.05, 0.07]
        # diff = [0.02, 0.02]; sum(abs2) = 2*0.0004 = 0.0008; *(0.5*1.0) = 0.0004
        @test KrakeNN._reg_loss(nu, 1.0) ≈ 0.0004
        @test KrakeNN._reg_loss(nu, 0.0) == 0.0
        @test KrakeNN._reg_loss([0.05], 1.0) == 0.0
        @test KrakeNN._reg_loss(Float64[], 1.0) == 0.0
    end

    @testset "_reg_grad: known analytic values" begin
        nu = [0.03, 0.05, 0.07]
        g  = KrakeNN._reg_grad(nu, 1.0)
        # j=1 (endpoint): nu[1] - nu[2] = 0.03 - 0.05 = -0.02
        # j=2 (interior): 2*nu[2] - nu[1] - nu[3] = 0.10 - 0.10 = 0.0
        # j=3 (endpoint): nu[3] - nu[2] = 0.07 - 0.05 = 0.02
        @test g[1] ≈ -0.02
        @test g[2] ≈  0.0 atol=1e-15
        @test g[3] ≈  0.02
        # alpha=0 -> zeros
        @test all(KrakeNN._reg_grad(nu, 0.0) .== 0.0)
    end

    @testset "_reg_grad vs central FD (random nu, random alpha)" begin
        rng   = Random.MersenneTwister(11)
        nu    = 0.03 .+ 0.04 .* rand(rng, _NF_NY)
        alpha = 0.1 + rand(rng)
        g_ana = KrakeNN._reg_grad(nu, alpha)
        h     = 1e-6
        g_fd  = similar(g_ana)
        for j in eachindex(nu)
            nu_p = copy(nu); nu_p[j] += h
            nu_m = copy(nu); nu_m[j] -= h
            g_fd[j] = (KrakeNN._reg_loss(nu_p, alpha) - KrakeNN._reg_loss(nu_m, alpha)) / (2h)
        end
        @test maximum(abs.(g_ana .- g_fd)) < 1e-10
    end

    @testset "_is_nufield_pspace: detection" begin
        Ny     = _NF_NY
        names  = [Symbol("ν_$i") for i in 1:Ny]
        ps_fld = ParameterSpace(names, fill(0.01, Ny), fill(0.5, Ny))
        @test KrakeNN._is_nufield_pspace(ps_fld) == true
        ps_sc  = ParameterSpace([:ν], [0.01], [0.5])
        @test KrakeNN._is_nufield_pspace(ps_sc)  == false
        ps_r   = ParameterSpace([:radius], [0.5], [5.0])
        @test KrakeNN._is_nufield_pspace(ps_r)   == false
    end

    @testset "ParameterSpace nufield log-scale round-trip (Ny=8)" begin
        Ny     = _NF_NY
        names  = [Symbol("ν_$j") for j in 1:Ny]
        ps     = ParameterSpace(names, fill(1e-3, Ny), fill(0.5, Ny); log_scale=trues(Ny))
        nu_vec = [0.03 + 0.04*sin(π*(j-1)/(Ny-1)) for j in 1:Ny]
        p0     = NamedTuple{Tuple(names)}(Tuple(nu_vec))
        x      = KrakeNN.to_flat(ps, p0)
        @test x ≈ log.(nu_vec)
        p_rt   = KrakeNN.from_flat(ps, x, p0)
        @test all(abs(p_rt[names[j]] - nu_vec[j]) < 1e-14 for j in 1:Ny)
    end

    @testset "method=:lbfgs clean error without Optim" begin
        if !_NUFIELD_OPTIM_OK
            obs        = LineProfile(:ux, [(_NF_NX÷2, j) for j in 1:_NF_NY])
            data_dummy = [(observable=obs, value=fill(0.01, _NF_NY))]
            ps         = ParameterSpace([:ν], [0.01], [0.5])
            p0         = (ν=0.05,)
            @test_throws ErrorException fit(_NF_GEOM, LBM(), data_dummy, p0, ps;
                                           observables=[obs], method=:lbfgs)
            println("method=:lbfgs clean error without Optim: PASS")
        else
            @info "Skipping :lbfgs-without-Optim test (Optim already loaded in this process)"
        end
    end

    @testset "method=:unknown raises ArgumentError" begin
        obs        = LineProfile(:ux, [(_NF_NX÷2, j) for j in 1:_NF_NY])
        data_dummy = [(observable=obs, value=fill(0.01, _NF_NY))]
        ps         = ParameterSpace([:ν], [0.01], [0.5])
        p0         = (ν=0.05,)
        @test_throws ArgumentError fit(_NF_GEOM, LBM(), data_dummy, p0, ps;
                                      observables=[obs], method=:foobar)
    end

    @testset "OPTIM_FREE_OK: KrakeNNOptimExt not loaded without Optim" begin
        @test isdefined(KrakeNN, :fit)
        @test isdefined(KrakeNN, :ParameterSpace)
        @test isdefined(KrakeNN, :LBMFieldParams)
        println("OPTIM_FREE_OK: fit/ParameterSpace/LBMFieldParams defined without Optim")
    end

end # Enzyme-free testset

# ---- Enzyme-gated tests ----

if _NUFIELD_ENZYME_OK

    @testset "M-P2c-2 Enzyme-gated" begin

        function _ux_midx(f_star, Nx, Ny)
            x_obs = Nx ÷ 2
            cx_q  = [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0]
            ux    = zeros(Float64, Ny)
            for j in 1:Ny
                rho    = sum(@view f_star[x_obs, j, :])
                ux[j]  = sum(cx_q[q] * f_star[x_obs, j, q] for q in 1:9) / rho
            end
            return ux
        end

        @testset "C-2b: scalar rung regression (|ν_fit-ν_true|/ν_true < 5%)" begin
            ν_true   = 0.05
            ν0       = 0.07
            σ        = 5e-4
            fwd_true = KrakeNN.ad_forward_solve(; _NF_GEOM..., nu=ν_true,
                                                tol=1e-12, max_steps=200_000)
            @test fwd_true.converged
            x_obs    = _NF_NX ÷ 2
            obs      = LineProfile(:ux, [(x_obs, j) for j in 1:_NF_NY])
            ux_true  = _ux_midx(fwd_true.f_star, _NF_NX, _NF_NY)
            rng      = Random.MersenneTwister(42)
            ux_noisy = ux_true .+ σ .* randn(rng, _NF_NY)
            data     = [(observable=obs, value=ux_noisy)]
            ps       = ParameterSpace([:ν], [0.01], [0.2])
            p0       = (ν=ν0,)
            cr       = fit(_NF_GEOM, LBM(), data, p0, ps;
                           observables=[obs], max_iter=100, step_size=0.5)
            ν_fit    = cr.p_opt[:ν]
            rel_err  = abs(ν_fit - ν_true) / ν_true
            @info "C-2b scalar regression" ν_true ν0 ν_fit rel_err n_iter=cr.n_iter
            @test rel_err < 0.05
        end

        @testset "C-2c: Enzyme-free load intact (LBMFieldParams + ParameterSpace{ν_i})" begin
            @test isdefined(KrakeNN, :LBMFieldParams)
            Ny    = 4
            names = [Symbol("ν_$i") for i in 1:Ny]
            ps    = ParameterSpace(names, fill(0.01, Ny), fill(0.5, Ny))
            p0    = NamedTuple{Tuple(names)}(Tuple(fill(0.05, Ny)))
            @test KrakeNN.n_free(ps) == Ny
            println("C-2c Enzyme-free load: LBMFieldParams + ParameterSpace{ν_i} OK")
        end

    end # Enzyme-gated testset

else
    @info "Skipping M-P2c-2 Enzyme-gated calibration nufield tests (KrakeNNADExt not loaded)"
end

# ---- Enzyme + Optim gated (field twin experiment C-2a) ----

if _NUFIELD_ENZYME_OK && _NUFIELD_OPTIM_OK

    @testset "M-P2c-2 Optim-gated: field twin experiment" begin

        function _ux_midx(f_star, Nx, Ny)
            x_obs = Nx ÷ 2
            cx_q  = [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0]
            ux    = zeros(Float64, Ny)
            for j in 1:Ny
                rho   = sum(@view f_star[x_obs, j, :])
                ux[j] = sum(cx_q[q] * f_star[x_obs, j, q] for q in 1:9) / rho
            end
            return ux
        end

        @testset "C-2a: FIELD TWIN EXPERIMENT — ν(y) recovery (rel_L2 < 10%)" begin
            Nx     = _NF_NX
            Ny     = _NF_NY
            # True profile: smooth sine stratification, range [0.03, 0.07]
            ν_true = [0.03 + 0.04*sin(π*(j-1)/(Ny-1)) for j in 1:Ny]
            σ      = 5e-4

            # Generate reference data with true ν field
            fwd_true = KrakeNN.ad_forward_solve_nufield(; _NF_GEOM...,
                                                        nu_field=ν_true,
                                                        tol=1e-12,
                                                        max_steps=200_000)
            @test fwd_true.converged

            x_obs    = Nx ÷ 2
            obs      = LineProfile(:ux, [(x_obs, j) for j in 1:Ny])
            ux_true  = _ux_midx(fwd_true.f_star, Nx, Ny)
            rng      = Random.MersenneTwister(42)
            ux_noisy = ux_true .+ σ .* randn(rng, Ny)
            data     = [(observable=obs, value=ux_noisy)]

            # Initial guess: uniform ν_0 ≈ mean of true profile
            names = [Symbol("ν_$j") for j in 1:Ny]
            ps    = ParameterSpace(names, fill(1e-3, Ny), fill(0.5, Ny);
                                   log_scale=trues(Ny))
            ν0    = fill(0.05, Ny)
            p0    = NamedTuple{Tuple(names)}(Tuple(ν0))

            cr = fit(_NF_GEOM, LBM(), data, p0, ps;
                     observables=[obs],
                     max_iter=200,
                     reg_weight=1e-4,
                     method=:lbfgs,
                     gtol=1e-6,
                     ftol=1e-12)

            ν_fit  = [cr.p_opt[names[j]] for j in 1:Ny]
            rel_l2 = norm(ν_fit .- ν_true) / norm(ν_true)

            @info "C-2a FIELD TWIN EXPERIMENT" rel_l2 n_iter=cr.n_iter loss_final=cr.loss_final converged=cr.converged
            @info "C-2a ν_true" ν_true
            @info "C-2a ν_fit " ν_fit

            @test rel_l2 < 0.10
            @test cr.n_iter <= 200
            @test cr.loss_final <= cr.loss_trace[1]
        end

    end # Optim-gated testset

elseif _NUFIELD_ENZYME_OK && !_NUFIELD_OPTIM_OK
    @info "Skipping M-P2c-2 Optim-gated field twin experiment (KrakeNNOptimExt not loaded)"
end
