using Test
using KrakeNN

@testset "Conservative tree adaptation plan 2D" begin
    @testset "policy pads clamps and limits growth" begin
        policy = ConservativeTreeAdaptationPolicy2D(
            pad_parent=1, min_i_cells=5, min_j_cells=3,
            max_growth=2, shrink_margin=1)

        plan = conservative_tree_adaptation_plan_2d(
            20, 16, 8:10, 6:8, 9:9, 7:7;
            policy=policy, reason=:unit)

        @test plan.current_i_range == 8:10
        @test plan.current_j_range == 6:8
        @test plan.requested_i_range == 9:9
        @test plan.requested_j_range == 7:7
        @test plan.i_range == 7:11
        @test plan.j_range == 6:8
        @test plan.reason == :unit
        @test plan.changed

        limited = conservative_tree_adaptation_plan_2d(
            20, 16, 10:12, 8:10, 1:20, 1:16;
            policy=ConservativeTreeAdaptationPolicy2D(max_growth=2, shrink_margin=0))
        @test limited.i_range == 8:14
        @test limited.j_range == 6:12

        clamped = conservative_tree_adaptation_plan_2d(
            8, 6, 2:4, 2:4, -2:2, 5:20;
            policy=ConservativeTreeAdaptationPolicy2D(shrink_margin=0))
        @test clamped.i_range == 1:2
        @test clamped.j_range == 5:6
    end

    @testset "hysteresis keeps near shrink stable" begin
        policy = ConservativeTreeAdaptationPolicy2D(shrink_margin=2)
        near = conservative_tree_adaptation_plan_2d(
            20, 16, 4:10, 3:8, 5:9, 4:7; policy=policy)
        deep = conservative_tree_adaptation_plan_2d(
            20, 16, 4:10, 3:8, 6:8, 5:6; policy=policy)

        @test !near.changed
        @test near.i_range == 4:10
        @test near.j_range == 3:8
        @test deep.changed
        @test deep.i_range == 6:8
        @test deep.j_range == 5:6

        @test_throws ArgumentError ConservativeTreeAdaptationPolicy2D(pad_parent=-1)
        @test_throws ArgumentError ConservativeTreeAdaptationPolicy2D(min_i_cells=0)
        @test_throws ArgumentError conservative_tree_adaptation_plan_2d(
            8, 8, 0:2, 2:4, 3:4, 3:4)
    end

    @testset "indicator and krk proposals feed plans" begin
        indicator = zeros(Float64, 12, 10)
        indicator[7:8, 4] .= 0.7
        indicator[4, 5] = -0.9

        plan = conservative_tree_indicator_adaptation_plan_2d(
            indicator, 3:5, 3:5; threshold=0.5,
            policy=ConservativeTreeAdaptationPolicy2D(
                pad_parent=1, shrink_margin=0),
            reason=:scalar)

        @test plan.i_range == 3:9
        @test plan.j_range == 3:6
        @test plan.reason == :scalar
        @test plan.changed

        setup = parse_kraken("""
        Simulation adapt_krk D2Q9
        Domain L = 8 x 4  N = 32 x 16
        Physics nu = 0.1
        Refine fineA { region = [1.0, 0.5, 2.0, 1.5], ratio = 2 }
        Refine fineB { region = [4.0, 2.0, 5.0, 3.0], ratio = 2 }
        Boundary west periodic
        Boundary east periodic
        Boundary south wall
        Boundary north wall
        Run 10 steps
        """)
        proposals = conservative_tree_patch_proposals_from_krk_2d(setup)
        proposal_plan = conservative_tree_adaptation_plan_from_proposal_2d(
            setup.domain.Nx, setup.domain.Ny, 4:6, 3:5, proposals[1];
            policy=ConservativeTreeAdaptationPolicy2D(shrink_margin=0))

        @test length(proposals) == 2
        @test proposals[1].name == "fineA"
        @test proposals[1].i_range == 5:8
        @test proposals[1].j_range == 3:6
        @test proposals[1].reason == :krk_refine
        @test proposal_plan.i_range == 5:8
        @test proposal_plan.j_range == 3:6
        @test proposal_plan.reason == :krk_refine

        adaptive_setup = parse_kraken("""
        Simulation adapt_criterion D2Q9
        Domain L = 8 x 4  N = 16 x 12
        Physics nu = 0.1
        Refine wake {
            region = [2.0, 1.0, 4.0, 2.0],
            ratio = 2,
            criterion = gradient(ux) > 0.2,
            pad = 2,
            max_growth = 1,
            shrink_margin = 0,
            balance = 1
        }
        Boundary west periodic
        Boundary east periodic
        Boundary south wall
        Boundary north wall
        Run 10 steps
        """)
        criterion_refine = adaptive_setup.refinements[1]
        criterion = criterion_refine.criterion
        criterion_policy =
            conservative_tree_adaptation_policy_from_krk_refine_2d(criterion_refine)
        criterion_indicator = zeros(Float64, 16, 12)
        criterion_indicator[10:11, 5] .= 0.3
        criterion_plan = conservative_tree_indicator_adaptation_plan_from_krk_2d(
            criterion_indicator, 9:10, 5:5, criterion_refine)

        @test criterion !== nothing
        @test criterion.indicator == :gradient
        @test criterion.field == :ux
        @test criterion.op == Symbol(">")
        @test criterion.threshold ≈ 0.2
        @test criterion.update_every == 1
        @test criterion.balance == 1
        @test criterion_policy.pad_parent == 2
        @test criterion_policy.max_growth == 1
        @test criterion_policy.shrink_margin == 0
        @test criterion_plan.requested_i_range == 10:11
        @test criterion_plan.requested_j_range == 5:5
        @test criterion_plan.i_range == 8:11
        @test criterion_plan.j_range == 4:6
        @test criterion_plan.reason == :krk_refine_criterion
        @test_throws ArgumentError parse_kraken(
            """
            Simulation bad_balance D2Q9
            Domain L = 8 x 4  N = 16 x 12
            Physics nu = 0.1
            Refine wake {
                region = [2.0, 1.0, 4.0, 2.0],
                ratio = 2,
                criterion = gradient(ux) > 0.2,
                balance = 2
            }
            Boundary west periodic
            Boundary east periodic
            Boundary south wall
            Boundary north wall
            Run 10 steps
            """)
    end

    @testset "plan application regrids conservatively" begin
        nx, ny = 9, 8
        patch = create_conservative_tree_patch_2d(3:5, 3:5)
        coarse = zeros(Float64, nx, ny, 9)
        for q in 1:9, j in axes(coarse, 2), i in axes(coarse, 1)
            if !(i in patch.parent_i_range && j in patch.parent_j_range)
                coarse[i, j, q] = 0.11 + q / 23 + i / 37 + j / 41 + i * j / 4096
            end
        end
        for q in 1:9, j in axes(patch.fine_F, 2), i in axes(patch.fine_F, 1)
            patch.fine_F[i, j, q] = 0.04 + q / 61 + i / 73 + j / 89 + i * j / 8192
        end
        pop0 = active_population_sums_F(coarse, patch)

        plan = conservative_tree_adaptation_plan_2d(
            nx, ny, patch.parent_i_range, patch.parent_j_range, 2:7, 2:6;
            policy=ConservativeTreeAdaptationPolicy2D(shrink_margin=0),
            reason=:test_regrid)
        adapted = adapt_conservative_tree_patch_with_plan_2d(coarse, patch, plan)

        @test adapted.changed
        @test adapted.patch.parent_i_range == 2:7
        @test adapted.patch.parent_j_range == 2:6
        @test isapprox(active_population_sums_F(adapted.coarse_F, adapted.patch),
                       pop0; atol=1e-11, rtol=0)

        noop_plan = conservative_tree_adaptation_plan_2d(
            nx, ny, adapted.patch.parent_i_range, adapted.patch.parent_j_range,
            adapted.patch.parent_i_range, adapted.patch.parent_j_range;
            policy=ConservativeTreeAdaptationPolicy2D(shrink_margin=0))
        noop = adapt_conservative_tree_patch_with_plan_2d(
            adapted.coarse_F, adapted.patch, noop_plan)
        @test !noop.changed
        @test noop.patch === adapted.patch
        @test noop.coarse_F === adapted.coarse_F

        stale_plan = conservative_tree_adaptation_plan_2d(
            nx, ny, 1:2, 1:2, 2:3, 2:3;
            policy=ConservativeTreeAdaptationPolicy2D(shrink_margin=0))
        @test_throws ArgumentError adapt_conservative_tree_patch_with_plan_2d(
            coarse, patch, stale_plan)
    end
end
