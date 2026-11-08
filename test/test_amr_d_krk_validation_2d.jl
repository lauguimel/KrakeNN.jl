using Test
using KrakeNN

@testset "AMR D .krk validation helpers" begin
    convergence_dir = joinpath(dirname(@__DIR__), "benchmarks", "krk",
                               "amr_d_convergence_2d")
    showoff_dir = joinpath(dirname(@__DIR__), "benchmarks", "krk",
                           "amr_d_showoff_2d")

    cases = Dict(
        basename(path) => conservative_tree_amr_d_case_from_krk_2d(path)
        for path in sort(filter(endswith(".krk"), readdir(convergence_dir; join=true))))

    @test cases["poiseuille_xband_scale1.krk"].flow == :poiseuille
    @test cases["poiseuille_xband_scale1.krk"].boundary_policy == :periodic_x_wall_y
    @test cases["poiseuille_xband_scale1.krk"].runtime_supported

    @test cases["couette_scale1.krk"].flow == :couette
    @test cases["couette_scale1.krk"].boundary_policy == :periodic_x_moving_wall_y
    @test cases["couette_scale1.krk"].runtime_supported

    @test cases["bfs_scale1.krk"].flow == :bfs
    @test cases["bfs_scale1.krk"].boundary_policy == :open_x_wall_y
    @test cases["bfs_scale1.krk"].runtime_supported
    result_b = run_conservative_tree_amr_d_case_from_krk_2d(
        joinpath(convergence_dir, "bfs_scale1.krk"); steps_override=2)
    @test result_b.flow == :bfs_route_native
    @test result_b.steps == 2
    @test isfinite(result_b.mass_drift)

    @test cases["square_scale1.krk"].flow == :square
    @test cases["square_scale1.krk"].wall_model == :halfway_bounceback_mask
    @test cases["square_scale1.krk"].runtime_supported
    result_s = run_conservative_tree_amr_d_case_from_krk_2d(
        joinpath(convergence_dir, "square_scale1.krk"); steps_override=2)
    @test result_s.flow == :square_obstacle_route_native
    @test result_s.steps == 2
    @test isfinite(result_s.mass_drift)

    @test cases["cylinder_scale1.krk"].flow == :cylinder
    @test cases["cylinder_scale1.krk"].wall_model == :halfway_bounceback_mask
    @test cases["cylinder_scale1.krk"].runtime_supported
    result_y = run_conservative_tree_amr_d_case_from_krk_2d(
        joinpath(convergence_dir, "cylinder_scale1.krk"); steps_override=2)
    @test result_y.steps == 2
    @test isfinite(result_y.Cd)
    @test isfinite(result_y.mass_drift)

    nested_cylinder = cases["cylinder_nested4_probe.krk"]
    @test nested_cylinder.spec_supported
    @test nested_cylinder.max_level == 4
    @test nested_cylinder.runtime_supported
    @test nested_cylinder.runtime_status == :subcycled_nested_solid
    result_ny = run_conservative_tree_amr_d_case_from_krk_2d(
        joinpath(convergence_dir, "cylinder_nested4_probe.krk");
        steps_override=2)
    @test result_ny.flow == :cylinder_obstacle_subcycled
    @test result_ny.max_level == 4
    @test result_ny.steps == 2
    @test isfinite(result_ny.relative_mass_drift)

    nested_poiseuille = cases["poiseuille_nested4_channel.krk"]
    @test nested_poiseuille.max_level == 4
    @test nested_poiseuille.runtime_status == :subcycled_nested_channel
    result_p = run_conservative_tree_amr_d_case_from_krk_2d(
        joinpath(convergence_dir, "poiseuille_nested4_channel.krk");
        steps_override=2)
    @test result_p.max_level == 4
    @test result_p.steps == 2
    @test isfinite(result_p.relative_mass_drift)

    nested_couette = cases["couette_nested4_channel.krk"]
    @test nested_couette.max_level == 4
    @test nested_couette.runtime_status == :subcycled_nested_channel
    result_c = run_conservative_tree_amr_d_case_from_krk_2d(
        joinpath(convergence_dir, "couette_nested4_channel.krk");
        steps_override=2)
    @test result_c.max_level == 4
    @test result_c.steps == 2
    @test isfinite(result_c.relative_mass_drift)

    @test cases["poiseuille_xband_nested4_debug.krk"].max_level == 4
    @test cases["poiseuille_xband_nested4_debug.krk"].runtime_status ==
          :subcycled_nested_channel
    @test cases["poiseuille_yband_nested4_debug.krk"].max_level == 4
    @test cases["poiseuille_yband_nested4_debug.krk"].runtime_status ==
          :subcycled_nested_channel
    @test cases["poiseuille_wall_ybands_nested4_debug.krk"].max_level == 4
    @test cases["poiseuille_wall_ybands_nested4_debug.krk"].runtime_status ==
          :subcycled_nested_channel
    wall_ybands_spec = create_conservative_tree_spec_from_krk_2d(load_kraken(
        joinpath(convergence_dir,
                 "poiseuille_wall_ybands_nested4_debug.krk")))
    south_j = 1
    north_j = wall_ybands_spec.Ny << wall_ybands_spec.max_level
    @test any(wall_ybands_spec.cells[id].level ==
              wall_ybands_spec.max_level &&
              wall_ybands_spec.cells[id].j == south_j
              for id in wall_ybands_spec.active_cells)
    @test any(wall_ybands_spec.cells[id].level ==
              wall_ybands_spec.max_level &&
              wall_ybands_spec.cells[id].j == north_j
              for id in wall_ybands_spec.active_cells)
    @test cases["couette_yband_nested4_debug.krk"].max_level == 4
    @test cases["couette_yband_nested4_debug.krk"].runtime_status ==
          :subcycled_nested_channel
    couette_debug_spec = create_conservative_tree_spec_from_krk_2d(load_kraken(
        joinpath(convergence_dir, "couette_yband_nested4_debug.krk")))
    north_j = couette_debug_spec.Ny << couette_debug_spec.max_level
    @test any(couette_debug_spec.cells[id].level ==
              couette_debug_spec.max_level &&
              couette_debug_spec.cells[id].j == north_j
              for id in couette_debug_spec.active_cells)

    lift_probe = cases["cylinder_lift_nested4_probe.krk"]
    @test lift_probe.flow == :cylinder_lift
    @test lift_probe.max_level == 4
    @test lift_probe.runtime_supported == false
    @test lift_probe.runtime_status == :nested_obstacle_runtime_pending

    showoff = conservative_tree_amr_d_case_from_krk_2d(
        joinpath(showoff_dir, "cylinder_lift_re100_long_channel.krk"))
    @test showoff.flow == :cylinder_lift
    @test showoff.boundary_policy == :open_x_wall_y
    @test showoff.max_level == 3
    @test :lift in showoff.diagnostics
    @test showoff.runtime_supported == false
    @test showoff.runtime_status == :nested_obstacle_runtime_pending

    support = Dict(row.feature => row for row in conservative_tree_amr_d_support_matrix_2d())
    @test support[:periodic_x_wall_y].nested
    @test support[:periodic_x_moving_wall_y].nested
    @test support[:open_x_wall_y].single_patch
    @test support[:open_x_wall_y].nested == false
    @test support[:halfway_bounceback_solid_mask].nested
    @test support[:ibb].wall_model == :unsupported
    @test support[:libb].wall_model == :unsupported
end
