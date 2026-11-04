using Test
using Kraken

const U = Kraken.Units

_geom_ss(; type=:cylinder_2d, blockage=0.5, L_up=15.0, L_down=15.0,
         q_wall_dist=nothing) =
    (type=type, blockage=blockage, L_up=L_up, L_down=L_down,
     q_wall_dist=q_wall_dist)

_bc_ss(; wall_bc=:halfwayBB) =
    (inlet=:velocity_parabolic, outlet=:zou_he_pressure,
     north_wall=:halfwayBB, south_wall=:halfwayBB, wall_bc=wall_bc)

function _ve_plan_ss(; R=30, scaling=:diffusive, tau_target=0.95, Wi=1.0,
                     geom=_geom_ss(), strict=false, kwargs...)
    return U.compile(; physics=:viscoelastic, Re=1.0, Wi=Wi, beta=0.59,
                     bsd_fraction=1.0, model=:oldroyd_b, R_LU=R,
                     scaling=scaling, tau_target=tau_target,
                     geometry=geom, bc=_bc_ss(), strict=strict, kwargs...)
end

@testset "Units steady-state estimate" begin
    @testset "estimator returns dominant timescale and ordering" begin
        plan = _ve_plan_ss(; R=30, scaling=:diffusive)
        est = U.estimate_steady_state(plan)
        # all three timescales finite and positive for a steady confined flow
        @test isfinite(est.t_diff) && est.t_diff > 0
        @test isfinite(est.t_adv) && est.t_adv > 0
        @test isfinite(est.t_poly) && est.t_poly > 0
        # t_ss is the max of the three; n_steps_ss = ceil(t_ss)
        @test est.t_ss_lu ≈ max(est.t_diff, est.t_adv, est.t_poly) atol=0 rtol=1e-12
        @test est.n_steps_ss == ceil(Int, est.t_ss_lu)
        @test est.basis in (:diffusive, :advective, :polymeric)
        @test est.exists
        # accessor and explicit-arg form agree
        est2 = U.estimate_steady_state(plan.units, plan.geometry,
                                       plan.physics_spec)
        @test est.t_ss_lu == est2.t_ss_lu
    end

    @testset "polymer-dominated ordering at high Wi" begin
        # large Wi -> large lambda -> polymeric timescale dominates
        plan = _ve_plan_ss(; R=10, scaling=:diffusive, Wi=50.0)
        est = U.estimate_steady_state(plan)
        @test est.t_poly >= est.t_diff
        @test est.basis === :polymeric
    end

    @testset "too-short max_steps triggers warning" begin
        # A diffusion-limited config: the viscous timescale L^2/nu exceeds the
        # advective+polymeric floor that `_max_steps` already accounts for, so
        # the estimator catches convergence the old heuristic missed.
        geom = _geom_ss(; type=:channel, blockage=0.99, L_up=0.5, L_down=0.5)
        plan = U.compile(; physics=:viscoelastic, Re=5.0, Wi=0.01, beta=0.59,
                         R_LU=10, scaling=:acoustic, u_target=0.03,
                         geometry=geom, bc=_bc_ss(), strict=false)
        est = U.estimate_steady_state(plan)
        @test est.basis === :diffusive
        @test plan.units.max_steps < est.n_steps_ss
        @test :max_steps_below_steady_state in U.issue_codes(plan.warnings)
    end

    @testset "adequate max_steps is clean of the steady-state warning" begin
        plan = _ve_plan_ss(; R=30, scaling=:diffusive)
        @test !(:max_steps_below_steady_state in U.issue_codes(plan.warnings))
    end

    @testset "inherently transient geometry is flagged" begin
        plan = _ve_plan_ss(; R=30, scaling=:diffusive,
                           geom=_geom_ss(; type=:taylor_green))
        est = U.estimate_steady_state(plan)
        @test !est.exists
        @test :steady_state_not_expected in U.issue_codes(plan.warnings)
    end
end

@testset "Units viscoelastic parameter-stability predicate" begin
    @testset "lambda < cell triggers the low-Wi artifact warning" begin
        # acoustic scaling at small Wi -> lambda_LU = Wi*R/u, tiny lambda
        bad = U.compile(; physics=:viscoelastic, Re=1.0, Wi=1.0e-3, beta=0.59,
                        R_LU=30, scaling=:acoustic, u_target=0.05,
                        geometry=_geom_ss(), bc=_bc_ss(), strict=false)
        codes = U.issue_codes(bad.warnings)
        @test (:lambda_below_cell in codes) || (:lambda_near_cell in codes)
    end

    @testset "healthy config is clean of the artifact warning" begin
        good = _ve_plan_ss(; R=30, scaling=:diffusive, Wi=1.0)
        codes = U.issue_codes(good.warnings)
        @test !(:lambda_below_cell in codes)
        @test !(:lambda_near_cell in codes)
    end

    @testset "high Wi flags SPD / positivity headroom" begin
        # large Wi -> lambda*gamma_dot >= 1 -> direct-C SPD warning
        plan = _ve_plan_ss(; R=10, scaling=:diffusive, Wi=200.0)
        codes = U.issue_codes(plan.warnings)
        @test :direct_c_spd_headroom in codes
        @test :polymer_cfl_high in codes
    end

    @testset "predicate fires for VE wall BCs but not Newtonian" begin
        newt = U.compile(; physics=:newtonian, Re=1.0, R_LU=30,
                         scaling=:diffusive, geometry=_geom_ss(),
                         bc=_bc_ss(), strict=false)
        codes = U.issue_codes(newt.warnings)
        @test !(:lambda_below_cell in codes)
        @test !(:direct_c_spd_headroom in codes)
    end
end
