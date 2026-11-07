using Test
using KrakeNN
using KernelAbstractions

# Steady stop options of three drivers (cylinder LI-BB, 3D FVFD Oldroyd-B channel,
# 2D log-FV coupled cylinder). Off by default: with the default keywords the
# drivers run exactly `max_steps` steps and compute the same values as before.
# Gates are structural (bit-identity, stop before the cap, converged flag,
# last change below the tolerance), not fitted numbers.

@testset "Steady stop options" begin

    @testset "cylinder LI-BB 2D" begin
        kw = (; Nx=60, Ny=24, radius=4, u_in=0.04, ν=0.04, inlet=:uniform,
                avg_window=100)
        base = run_cylinder_libb_2d(; kw..., max_steps=1500)

        @testset "default keywords: nothing changes" begin
            @test base.converged == false
            @test base.stop_step == 1500
            @test isempty(base.cd_history)
            # Sampling without ever meeting the rule must not perturb the state.
            sampled = run_cylinder_libb_2d(; kw..., max_steps=1500,
                                             stop=:cd_change, tol=0.0, check_every=50)
            @test sampled.converged == false
            @test length(sampled.cd_history) == 30
            @test sampled.ρ == base.ρ
            @test sampled.ux == base.ux
            @test sampled.uy == base.uy
            @test sampled.Cd == base.Cd
            @test sampled.Fx == base.Fx
        end

        @testset "stop on: stops before the cap" begin
            r = run_cylinder_libb_2d(; kw..., max_steps=20_000,
                                       stop=:cd_change, tol=1e-6, check_every=100)
            @test r.converged
            @test r.stop_step < 20_000
            @test r.stop_step == r.cd_history[end][1]
            cd_new, cd_old = r.cd_history[end][2], r.cd_history[end-1][2]
            @test abs(cd_new - cd_old) <= 1e-6 * abs(cd_new)
            # No window average exists: the reported value is the last sample.
            @test r.Cd == cd_new
        end

        @testset "stop on: cannot settle within the cap" begin
            r = run_cylinder_libb_2d(; kw..., max_steps=300,
                                       stop=:cd_change, tol=1e-12, check_every=100)
            @test !r.converged
            @test r.stop_step == 300
            @test length(r.cd_history) == 3
        end

        @test_throws ArgumentError run_cylinder_libb_2d(; kw..., max_steps=10, stop=:bogus)
    end

    @testset "3D FVFD Oldroyd-B Poiseuille" begin
        kw = (; Nx=4, Ny=12, Nz=4, Fx=1e-5, ν_s=0.05, ν_p=0.05, lambda=5.0)
        base = KrakeNN.run_viscoelastic_fvfd_poiseuille_3d(; kw..., max_steps=600)

        @testset "default keywords: nothing changes" begin
            @test base.converged == false
            @test base.stop_step == 600
            @test isempty(base.change_history)
            sampled = KrakeNN.run_viscoelastic_fvfd_poiseuille_3d(;
                kw..., max_steps=600, stop=:change, tol=0.0, check_every=50)
            @test sampled.converged == false
            @test length(sampled.change_history) == 11
            @test sampled.ux == base.ux
            @test sampled.tau_p_xy == base.tau_p_xy
            @test sampled.psi_xx == base.psi_xx
            @test sampled.profile == base.profile
        end

        @testset "stop on: stops before the cap" begin
            # tol = 1e-5: on this small grid the change falls to ~1e-6 near step 1500
            # and then grows again (round-off mode, present with the option off too),
            # so a tighter tolerance is never met.
            r = KrakeNN.run_viscoelastic_fvfd_poiseuille_3d(; kw...,
                max_steps=20_000, stop=:change, tol=1e-5, check_every=100)
            @test r.converged
            @test r.stop_step < 20_000
            @test r.stop_step == r.completed_steps
            _, r_u, r_tau = r.change_history[end]
            @test r_u <= 1e-5 && r_tau <= 1e-5
        end

        @testset "stop on: cannot settle within the cap" begin
            r = KrakeNN.run_viscoelastic_fvfd_poiseuille_3d(; kw...,
                max_steps=300, stop=:change, tol=1e-14, check_every=100)
            @test !r.converged
            @test r.stop_step == 300
        end

        @test_throws ArgumentError KrakeNN.run_viscoelastic_fvfd_poiseuille_3d(;
            kw..., max_steps=10, stop=:bogus)
    end

    @testset "2D log-FV coupled cylinder" begin
        kw = (; radius=3, L_up=3, L_down=4, nu_s=0.04, nu_p=0.04, lambda=2.0,
                u_mean=0.02, avg_window=50)
        base = run_viscoelastic_logfv_cylinder_coupled_2d(; kw..., max_steps=400)

        @testset "default keywords: nothing changes" begin
            @test base.converged == false
            @test base.stop_reason === :max_steps
            @test base.stop_step == 400
            @test isempty(base.cd_history)
            @test isnan(base.Cd_last)
            # A callback that never returns `true`, and the sampling without the
            # rule met, must not perturb the state.
            ncalls = Ref(0)
            cb(step, fields) = (ncalls[] += 1; nothing)
            sampled = run_viscoelastic_logfv_cylinder_coupled_2d(; kw...,
                max_steps=400, step_callback=cb,
                stop=:cd_change, tol=0.0, check_every=40)
            @test ncalls[] == 400
            @test sampled.converged == false
            @test length(sampled.cd_history) == 10
            @test sampled.ux == base.ux
            @test sampled.uy == base.uy
            @test sampled.tauxy == base.tauxy
            @test sampled.Cd == base.Cd
            @test sampled.n_drag_samples == base.n_drag_samples
        end

        @testset "callback returning true stops the run" begin
            cb(step, fields) = step >= 123
            r = run_viscoelastic_logfv_cylinder_coupled_2d(; kw..., max_steps=400,
                                                             step_callback=cb)
            @test r.stop_step == 123
            @test r.completed_steps == 123
            @test r.converged
            @test r.stop_reason === :callback
        end

        @testset "stop on: stops before the cap" begin
            r = run_viscoelastic_logfv_cylinder_coupled_2d(; kw..., max_steps=20_000,
                stop=:cd_change, tol=1e-4, check_every=100)
            @test r.converged
            @test r.stop_reason === :rule
            @test r.stop_step < 20_000
            @test r.stop_step == r.cd_history[end][1]
            cd_new, cd_old = r.cd_history[end][2], r.cd_history[end-1][2]
            @test abs(cd_new - cd_old) <= 1e-4 * abs(cd_new)
            @test r.Cd_last == cd_new
            # Stopped before the averaging window: Cd is the last sample, as in the
            # LI-BB cylinder driver.
            @test r.n_drag_samples == 0
            @test r.Cd ≈ cd_new rtol=1e-12
            @test r.Cd_s + r.Cd_p - r.Cd_bsd ≈ r.Cd rtol=1e-12
        end

        @testset "stop on: cannot settle within the cap" begin
            r = run_viscoelastic_logfv_cylinder_coupled_2d(; kw..., max_steps=300,
                stop=:cd_change, tol=1e-14, check_every=100)
            @test !r.converged
            @test r.stop_reason === :max_steps
            @test r.stop_step == 300
            @test length(r.cd_history) == 3
        end

        @test_throws ArgumentError run_viscoelastic_logfv_cylinder_coupled_2d(;
            kw..., max_steps=10, stop=:bogus)
    end
end
