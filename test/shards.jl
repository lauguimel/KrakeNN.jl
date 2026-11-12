# Test sharding: split the suite over N processes (CI jobs) that run in
# parallel. Pure Julia, no package dependency, so test/shard_check.jl can load
# it with a bare `julia`.
#
#   KRAKEN_TEST_SHARD=2/6 julia --project -e 'using Pkg; Pkg.test()'
#
# runs the 2nd of 6 shards. Unset, the whole suite runs as before.
#
# Assignment: greedy longest-first over COST_SECONDS (each file goes to the
# currently lightest shard, ties broken by name, so it is deterministic and the
# same on every machine). A file missing from the table is never dropped: it
# falls into shard 1 + fnv1a(path) mod N. test/shard_check.jl (run in CI)
# checks that every include in runtests.jl goes through the selector and that
# the shards partition the file list.
module KrakenTestShards

export parse_shard, assignment, shard_of

# Seconds per test file, Julia 1.12 on ubuntu-latest, CI run 43037632029
# (2026-11-11, KRAKEN_SKIP_AD=true, KRAKEN_SKIP_VE3D=true). Read from the log
# timestamps of each file's @info lines, so a file's figure includes its own
# first-call compilation and is approximate to ~10 s. Files that print nothing
# share the gap before the next file that does. Refresh the table when the
# shards drift apart by more than a few minutes.
const COST_SECONDS = Dict{String,Int}(
    "analytical/ehd_onset_2d.jl" => 426,
    "test_natconv_gnf.jl" => 213,
    "analytical/gnf3d_parity.jl" => 186,
    "test_krk_examples.jl" => 164,
    "test_cylinder.jl" => 152,
    "analytical/thermal_adiabatic_walls.jl" => 152,
    "test_cylinder_libb.jl" => 131,
    "test_slbm_taylor_green.jl" => 114,
    "test_cavity.jl" => 103,
    "analytical/incns_fv_e5_compare_smoke_test.jl" => 95,
    "analytical/incns_fv_immersed.jl" => 93,
    "analytical/incns_fv_adjoint_gpu.jl" => 85,
    "analytical/lbm_law_test.jl" => 85,
    "analytical/incns_fv_aligned_c1.jl" => 84,
    "analytical/incns_fv_gnf.jl" => 74,
    "analytical/incns_fv_law_test.jl" => 79,
    "test_axisymmetric.jl" => 77,
    "analytical/incns_fv_immersed_mass.jl" => 67,
    "test_thermal_3d_krk.jl" => 65,
    "analytical/incns_fv_energy_test.jl" => 60,
    "test_viscoelastic.jl" => 29,
    "analytical/gnf3d_duct_obstacle.jl" => 59,
    "analytical/incns_fv_densenet_test.jl" => 52,
    "test_rheology.jl" => 51,
    "test_sphere_libb.jl" => 50,
    "test_benchmark.jl" => 48,
    "analytical/incns_fv_thermal_adjoint_test.jl" => 48,
    "analytical/incns_fv_immersed_mms.jl" => 43,
    "platform/residual_vjp_test.jl" => 48,
    "analytical/incns_fv_law_contraction_test.jl" => 47,
    "test_fused_trt_2d.jl" => 46,
    "analytical/incns_fv_immersed_m3_convergence.jl" => 45,
    "analytical/gnf3d_channel_order.jl" => 44,
    "test_fvfd_logconf_3d.jl" => 43,
    "test_taylor_couette_libb.jl" => 43,
    "analytical/incns_fv_jacobian_pattern_test.jl" => 41,
    "analytical/incns_fv_adjoint.jl" => 35,
    "test_cavity_3d.jl" => 37,
    "test_refinement.jl" => 37,
    "test_viscoelastic_extensional_krk.jl" => 37,
    "ad/gnf_adjoint_test.jl" => 37,
    "analytical/incns_fv_thermal_coupling_test.jl" => 36,
    "test_slbm_taylor_couette.jl" => 34,
    "analytical/lbm3d_law_test.jl" => 31,
    "analytical/TH-002-route.jl" => 33,
    "analytical/halfway_inlet_flux.jl" => 33,
    "analytical/hot_wall_nusselt_average.jl" => 33,
    "test_stl.jl" => 32,
    "analytical/incns_manifold.jl" => 24,
    "analytical/incns_fv_immersed_shape.jl" => 27,
    "analytical/incns_fv_layout_test.jl" => 26,
    "analytical/gnf3d_adjoint_gpu.jl" => 25,
    "analytical/ehd_ec_split_parity_2d.jl" => 24,
    "ad/gnf3d_adjoint_test.jl" => 16,
    "analytical/ehd_mrt_smoke_2d.jl" => 23,
    "test_poiseuille_3d.jl" => 23,
    "analytical/ehd_static_moment_2d.jl" => 23,
    "test_steady_stop_options.jl" => 21,
    "analytical/incns_fv_gnf_gpu.jl" => 20,
    "analytical/ehd_krk_2d.jl" => 19,
    "analytical/nn_law_gpu.jl" => 18,
    "test_phasefield.jl" => 17,
    "analytical/EC-RESTART-CUDA.jl" => 17,
    "analytical/incns_fv_steady_residual_gpu.jl" => 14,
    "analytical/incns_fv_immersed_objectives.jl" => 13,
    "analytical/incns_fv_energy_floor_test.jl" => 14,
    "test_geometry_stl_krk.jl" => 14,
    "test_geometry_descriptor.jl" => 14,
    "test_geometry_units_krk.jl" => 14,
    "test_geometry_units_3d_krk.jl" => 14,
    "test_geometry_stl_flow_3d_krk.jl" => 14,
    "analytical/neq_extrapolation_outlet_2d.jl" => 7,
    "test_mrt.jl" => 14,
    "analytical/incns_fv_newton.jl" => 13,
    "test_simulation_runner.jl" => 13,
    "analytical/regularized_trt_moments_2d.jl" => 12,
    "analytical/poisson_mms.jl" => 11,
    "analytical/ehd_twin_parity_2d.jl" => 8,
    "analytical/incns_fv_energy_theta_test.jl" => 11,
    "test_twophase_rheology.jl" => 11,
    "test_postprocess.jl" => 11,
    "analytical/ehd_phi_direct_2d.jl" => 11,
    "analytical/EC-SIDEWALLS.jl" => 4,
    "rheology/gnf_cell_test.jl" => 10,
    "analytical/ehd_mapping_parity_2d.jl" => 10,
    "analytical/H2-004-route.jl" => 10,
    "test_conservative_tree_2d.jl" => 10,
    "test_conservative_tree_topology_2d.jl" => 10,
    "test_conservative_tree_streaming_2d.jl" => 10,
    "test_curvilinear_mesh.jl" => 10,
    "test_slbm.jl" => 10,
    "analytical/ehd_hydrostatic_2d.jl" => 10,
    "analytical/ve3d_ext_velocity_readback.jl" => 9,
    "analytical/incns_fv_immersed_m3_algebra.jl" => 9,
    "test_poiseuille.jl" => 5,
    "analytical/ES-002-STOP.jl" => 8,
    "analytical/trt_guo_field_momentum.jl" => 8,
    "test_vtk_3d.jl" => 7,
    "analytical/EC-RESTART.jl" => 7,
    "test_couette_libb_canary_3d.jl" => 7,
    "platform/weakdeps_metadata_test.jl" => 7,
    "platform/contract_parity_test.jl" => 7,
    "platform/state_contract_test.jl" => 7,
    "analytical/nn_law_lbm_gpu.jl" => 6,
    "ad/gnf_calibration_test.jl" => 6,
    "analytical/gnf3d_strain_rate.jl" => 6,
    "analytical/incns_fv_newton_gpu.jl" => 5,
    "analytical/incns_fv_immersed_mass_compiler.jl" => 5,
    "analytical/incns_fv_immersed_mass_contract.jl" => 5,
    "analytical/incns_fv_immersed_mass_quadrature.jl" => 5,
    "analytical/incns_fv_immersed_m3_adjoint.jl" => 5,
    "test_viscoelastic_krk.jl" => 5,
    "test_bsd_open_corner_2d.jl" => 5,
    "test_fvfd_operators_3d.jl" => 5,
    "test_fvfd_boundary_stencils_3d.jl" => 5,
    "test_fvfd_velocity_gradient_3d.jl" => 4,
    "test_lbm_basic.jl" => 2,
    "test_couette_libb_canary.jl" => 4,
    "analytical/ehd_transient_moment_2d.jl" => 4,
    "test_vof.jl" => 2,
    "test_guo_convention_pairs.jl" => 3,
    "analytical/incns_grad_div_laplacian_mms.jl" => 3,
    "analytical/incns_poiseuille.jl" => 3,
    "analytical/incns_unsteady_taylor_green.jl" => 3,
    "analytical/incns_unsteady_startup_channel.jl" => 3,
    "analytical/incns_fv_immersed_channel.jl" => 3,
    "analytical/incns_momentum_advection_order.jl" => 3,
    "analytical/incns_fv_steady_residual.jl" => 3,
    "test_li_bb_2d.jl" => 3,
    "test_sphere_stl_drag_krk.jl" => 3,
    "test_slbm_libb_3d.jl" => 3,
    "test_gmsh_loader.jl" => 3,
    "test_multiblock_topology.jl" => 3,
    "test_multiblock_exchange.jl" => 3,
    "test_multiblock_canal.jl" => 3,
    "test_stl_libb.jl" => 3,
    "test_fvfd_ptt_3d.jl" => 2,
    "test_fvfd_giesekus_3d.jl" => 2,
    "analytical/incns_fv_immersed_gpu.jl" => 2,
    "analytical/scalar_transport_heated_channel.jl" => 2,
    "analytical/scalar_transport_advection_order.jl" => 2,
    "test_thermal.jl" => 2,
    "test_fvfd_fenep_3d.jl" => 2,
    "test_expression.jl" => 2,
    "test_kraken_parser.jl" => 2,
    "test_krk_symbolic.jl" => 2,
    "analytical/poisson_embedded_fvfd_mms.jl" => 1,
    "analytical/ehd_phi_ddf_solve_2d.jl" => 2,
    "test_kernel_dsl.jl" => 2,
    "test_taylor_green.jl" => 1,
    "test_krk_multiphase.jl" => 1,
    "test_species.jl" => 1,
    "test_multiphase.jl" => 1,
    "test_couette.jl" => 1,
    "analytical/poisson_embedded_mms.jl" => 1,
    "analytical/poisson_cudss_gpu.jl" => 1,
    "analytical/ehd_phi_gpu_parity_2d.jl" => 1,
    "test_units.jl" => 1,
    "test_units_stability.jl" => 1,
    "test_units_steady_state.jl" => 1,
    "test_units_audit.jl" => 1,
    "test_units_krk.jl" => 1,
    "test_units_thermal.jl" => 1,
    "test_units_ehd.jl" => 1,
    "platform/incns_contract_test.jl" => 1,
    "analytical/linear_solve_nonsym.jl" => 1,
    "analytical/linear_solve_refactor_transpose.jl" => 1,
    "test_advection_prescribed.jl" => 1,
    "analytical/ve3d_ext_zouhe_zplanes.jl" => 1,
    "analytical/poisson_linearsolve_mms.jl" => 1,
    "analytical/incns_fv_energy_gpu.jl" => 1,
    "analytical/linear_solve_cudss_c0_gpu.jl" => 1,
)

"""
    parse_shard(s) -> Union{Nothing,Tuple{Int,Int}}

Parse `"k/N"` into `(k, N)`; empty string means no sharding. Anything else is
an error, so a typo cannot silently run nothing or everything.
"""
function parse_shard(s::AbstractString)
    isempty(strip(s)) && return nothing
    m = match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", s)
    m === nothing && error("KRAKEN_TEST_SHARD must be \"k/N\" (e.g. \"2/6\"), got \"$(s)\"")
    k = parse(Int, m.captures[1])
    n = parse(Int, m.captures[2])
    (n >= 1 && 1 <= k <= n) || error("KRAKEN_TEST_SHARD needs 1 <= k <= N, got \"$(s)\"")
    return (k, n)
end

"""
    assignment(n) -> Dict{String,Int}

Shard (1..n) of every file in `COST_SECONDS`, greedy longest-first.
"""
function assignment(n::Int)
    n >= 1 || error("number of shards must be >= 1, got $(n)")
    load = zeros(Int, n)
    asg = Dict{String,Int}()
    for (file, cost) in sort!(collect(COST_SECONDS); by = kv -> (-kv[2], kv[1]))
        k = argmin(load)            # first lightest shard: deterministic
        asg[file] = k
        load[k] += cost
    end
    return asg
end

# 32-bit FNV-1a: a hash that does not change between Julia versions (Base.hash
# may), so a file outside the table lands in the same shard on 1.11 and 1.12.
function fnv1a(s::AbstractString)
    h = 0x811c9dc5
    for b in codeunits(s)
        h = (h ⊻ b) * 0x01000193
    end
    return h
end

"""
    shard_of(file, n, asg = assignment(n)) -> Int

Shard (1..n) that runs `file` (path relative to test/, as written in
runtests.jl).
"""
shard_of(file::AbstractString, n::Int, asg::AbstractDict = assignment(n)) =
    haskey(asg, file) ? asg[file] : Int(mod(fnv1a(file), n)) + 1

end # module
