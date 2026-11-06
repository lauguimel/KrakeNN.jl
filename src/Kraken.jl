"""
    Kraken

GPU-native Lattice Boltzmann Method (LBM) framework in Julia.

Supports 2D (D2Q9) and 3D (D3Q19) simulations with automatic GPU
acceleration via KernelAbstractions.jl.
"""
module Kraken

# PNG/GIF output hooks — populated by KrakenMakieExt when CairoMakie is loaded
const _png_saver = Ref{Any}(nothing)
const _gif_saver = Ref{Any}(nothing)

# --- Platform contract (method-agnostic) ---
# Phase 0: types + capabilities. Phase 0b: LBMSolution + solve + sample (thin,
# behaviour-preserving wrappers around run_simulation; function bodies resolve
# run_simulation lazily at call time, so include order before the runner is fine).
include("platform/contract.jl")
include("platform/solution.jl")   # LBM, LBMSolution, solve
include("platform/sample.jl")     # sample
include("platform/observe.jl")    # observe, predict, Prediction, observables
include("platform/state.jl")      # AbstractSimulationState, StateSnapshot, advance!
export AbstractProblem, AbstractMethod, AbstractSolution, AbstractObservable, AbstractClosure
export Capability, ForwardSolve, GPUExecution, SteadyAdjoint, TransientAdjoint, FiniteDiff, NeuralClosure, SteadyResidual
export capabilities
export LBM, LBMSolution, solve, sample
export observe, predict, Prediction, FieldProbe, LineProfile, FieldReduction
export residual, adjoint_vjp
export LBMGeomParams, LBMThermalParams, LBMVEParams, LBMScalarParams, LBMFieldParams
export ParameterSpace, loss, fit, CalibResult
export AbstractSimulationState, StateSnapshot, CheckpointError
export init_state, advance!, solution, restore_state
export update_parameter!, updatable_parameters, export_state, check_compatible
# Client hooks stay unexported (extend as Kraken.snapshot, Kraken.validate_snapshot,
# Kraken.at_boundary, Kraken.check_updatable, Kraken.migrate).
export CHECKPOINT_CONTAINER_VERSION, write_checkpoint, read_checkpoint, checkpoint_info
export save_checkpoint, load_checkpoint

# --- Lattice definitions ---
include("lattice/lattice.jl")
include("lattice/d2q9.jl")
include("lattice/d3q19.jl")

# --- Rheology models ---
include("rheology/models.jl")
include("rheology/viscosity.jl")
include("rheology/strain_rate.jl")
include("rheology/gnf_3d.jl")
include("rheology/linalg.jl")
include("rheology/linalg_3d.jl")

# --- GPU kernels ---
include("kernels/equilibrium_helpers.jl")
include("kernels/equilibrium_helpers_3d.jl")
include("kernels/collide_stream_2d.jl")
include("kernels/collide_stream_3d.jl")
include("kernels/stream_periodic_2d.jl")
include("kernels/stream_periodic_3d.jl")
include("kernels/collide_guo_2d.jl")
include("kernels/collide_guo_3d.jl")
include("kernels/macroscopic.jl")
include("kernels/boundary_2d.jl")
include("kernels/boundary_3d.jl")
include("kernels/thermal_2d.jl")
include("kernels/thermal_3d.jl")
include("kernels/fused_thermal_2d.jl")
include("kernels/collide_mrt_2d.jl")
include("kernels/species_2d.jl")
include("kernels/ehd_2d.jl")
include("kernels/ehd_bc_2d.jl")
include("kernels/ehd_mrt_2d.jl")
include("kernels/multiphase_2d.jl")
include("kernels/vof_2d.jl")
include("kernels/dualgrid_2d.jl")
include("kernels/phasefield_2d.jl")
include("kernels/pressure_vof_2d.jl")
include("kernels/smooth_vof_2d.jl")
include("kernels/ghost_fluid_2d.jl")
include("kernels/fused_bgk_2d.jl")
include("kernels/fused_trt_2d.jl")
include("kernels/li_bb_2d.jl")
include("kernels/aa_bgk_2d.jl")
include("kernels/persistent_bgk_2d.jl")
include("kernels/advect_prescribed_2d.jl")
include("kernels/collide_rheology_2d.jl")
include("rheology/gnf_cell.jl")
include("kernels/collide_gnf_cell_2d.jl")
include("kernels/stream_transpose_2d.jl")
include("kernels/collide_twophase_rheology_2d.jl")
include("kernels/viscoelastic_2d.jl")

# --- Kernel DSL (runtime fusion) ---
include("kernels/dsl/lbm_spec.jl")
include("kernels/dsl/bricks.jl")
include("kernels/dsl/bricks_3d.jl")
include("kernels/dsl/bricks_gnf_3d.jl")
include("kernels/dsl/lbm_builder.jl")

# --- Kernels built from the DSL (must come after li_bb_2d.jl + DSL) ---
include("diagnostics/trace.jl")
include("kernels/li_bb_2d_v2.jl")
include("kernels/li_bb_3d_v2.jl")
include("fvfd/FVFD.jl")
include("kernels/logconformation_fv_2d.jl")

# --- IncNS solver stack: linear-solve seam, elliptic services, drivers ---
# poisson.jl MUST come first: it tail-includes solve/linear_solve.jl (the
# factorize-once seam), which has NO self-guard — never include linear_solve.jl
# directly here or it would double-define the seam.
include("solve/poisson.jl")
include("solve/poisson_embedded.jl")
include("solve/poisson_embedded_fvfd.jl")
include("solve/poisson_mg.jl")
# LinearSolve.jl / cuDSS front-end tags + assembled-direct driver; the backend
# methods live in ext/KrakenLinearSolveExt.jl and ext/KrakenCUDSSExt.jl
# ([weakdeps] — `using Kraken` stays backend-free).
include("solve/linear_solve_frontend.jl")
# NOT included: solve/linear_solve_cuda.jl and methods/inc_ns/cavity_mg_cuda.jl
# (CUDSS is not a package dependency; they stay manual-load inside GPU jobs.
#  Package users get the same cuDSS seam via ext/KrakenCUDSSExt.jl instead).
include("methods/inc_ns/simple.jl")
include("methods/inc_ns/cavity.jl")
include("methods/inc_ns/cavity_mg.jl")
include("methods/inc_ns/projection.jl")
include("methods/inc_ns/manifold_flow.jl")
include("methods/scalar_transport/thermal_transport.jl")
include("methods/inc_ns/method.jl")   # platform-contract wrapper (IncNS <: AbstractMethod)
include("methods/inc_ns/fv_steady_residual.jl")   # C1: steady FV residual (KA kernel, generic eltype)
include("methods/inc_ns/fv_newton.jl")   # C2: coloured dual Jacobian + Newton on the seam
include("methods/inc_ns/fv_gnf.jl")   # C3: generalised-Newtonian face viscosity + its Jacobian pattern

# IncNS platform contract
export IncNS, IncNSSolution
# Steady colocated FV residual (C1 of the FV Newton roadmap)
export FVGrid2D, FVSteadyProblem2D, fv_steady_problem, fv_residual!, fv_residual,
       fv_cell_residual, fv_pressure_pins!, fv_adapt, fv_state, fv_fields,
       fv_index, fv_nunknowns,
       FV_BC_WALL, FV_BC_INLET, FV_BC_OUTLET, FV_BC_PERIODIC,
       FV_CELL_FLUID, FV_CELL_SOLID, FV_CELL_PINNED
# Coloured dual Jacobian + Newton on the steady FV residual (C2)
export fv_jacobian_pattern, fv_jacobian_coloring, FVJacobianTag, FVJacobianAssembler,
       fv_jacobian_assembler, fv_assemble_jacobian!, fv_jacobian_chunk,
       FVNewtonSolver, fv_newton_solver, fv_jacobian!, fv_newton!, fv_newton_finalize!,
       solve_incns_fv_newton
# Generalised-Newtonian viscosity in the steady FV residual and Newton (C3)
export FVNewtonianViscosity, FVGNFViscosity, fv_gnf_parameters, fv_gnf_parameter_names,
       fv_with_gnf_parameters
# IncNS / scalar-transport drivers
export solve_incns_simple, solve_incns_cavity, solve_incns_cavity_mg,
       solve_incns_projection, solve_incns_manifold, manifold_full_cell_mask,
       solve_scalar_transport
# Linear-solve seam (factorize-once)
export lin_factorize, lin_solve!, lin_refactorize!, lin_supports_transpose,
       LinearSolveCache, LinearSolveBackend, CPUBackendTag, CUDABackendTag
# LinearSolve.jl / cuDSS front-ends (ext-backed; see [weakdeps])
export PoissonLinearSolve, solve_poisson_direct
# Elliptic (Poisson) services
export solve_poisson_dirichlet, solve_poisson_neumann, pin_reference_dof,
       assemble_poisson_embedded, solve_poisson_embedded,
       assemble_poisson_embedded_from_fvfd, fractions_from_fvfd,
       solve_poisson_mg, solve_poisson_mgcg
# FVFD grad/div/laplacian velocity operators (+ embedded variants)
export gdl_divergence_2d!, gdl_pressure_gradient_2d!, gdl_laplacian_apply_2d!,
       gdl_divergence_embedded_2d!, gdl_pressure_gradient_embedded_2d!,
       gdl_laplacian_apply_embedded_2d!

# --- Modular BC system (uses TRT rates + feq helpers; compiles face
#     kernels per BC type via Julia dispatch).
include("bc/specs.jl")
include("bc/rebuild_2d.jl")
include("bc/moments.jl")
include("bc/rebuild_3d.jl")

# --- GPU-native drag reductions (replace host-side per-step transfers)
include("kernels/drag_gpu.jl")

# --- Analytic geometry derivatives for AD shape optimization ---
include("kernels/enzyme_rules.jl")

# --- Simulation drivers ---
include("drivers/basic.jl")
include("drivers/cylinder_libb.jl")
include("drivers/thermal.jl")
include("drivers/natconv_gnf_2d.jl")
include("drivers/ehd_poisson.jl")
include("drivers/ehd_phi_ddf.jl")
include("drivers/ehd.jl")
include("drivers/ehd_ec_state.jl")
include("drivers/ehd_ec_checkpoint.jl")
include("drivers/ehd_ec.jl")
include("drivers/axisymmetric.jl")
include("drivers/multiphase.jl")
include("drivers/rheology.jl")
include("drivers/gnf_periodic_steady.jl")
include("drivers/viscoelastic_spec.jl")
include("drivers/viscoelastic.jl")
# --- 3D viscoelastic kernels (need OldroydB/LogConfOldroydB from
#     viscoelastic_spec.jl; logconf kernels need rheology/linalg_3d.jl;
#     all need li_bb_3d_v2 + D3Q19 symbols already included above) ---
include("kernels/conformation_lbm_3d.jl")
include("kernels/viscoelastic_3d.jl")
include("kernels/logconformation_lbm_3d.jl")
include("kernels/logconformation_fv_3d.jl")
include("drivers/viscoelastic_3d.jl")
include("drivers/viscoelastic_couette_3d.jl")
include("drivers/viscoelastic_poiseuille_3d.jl")
include("drivers/viscoelastic_fvfd_3d.jl")
include("drivers/viscoelastic_fvfd_extensional_3d.jl")
include("drivers/step_geometry_2d.jl")
include("drivers/viscoelastic_logfv_2d.jl")

# --- Steady shape-adjoint AD (core is Enzyme-free; reverse seams live in ext/) ---
include("ad/ad_step.jl")
include("ad/ad_thermal_step.jl")
include("ad/ad_ve_ops.jl")
include("ad/ad_ve_step.jl")
include("ad/ad_ve_forward.jl")
include("ad/ad_forward.jl")
include("ad/ad_qoi.jl")
include("ad/ad_adjoint.jl")
include("ad/ad_geometry.jl")
include("ad/ad_api.jl")
include("ad/ad_ve_geometry.jl")
include("ad/gnf_adjoint.jl")       # steady GNF adjoint (Enzyme-free, any backend)
include("ad/gnf_adjoint_operator.jl")
include("ad/gnf_adjoint_gmres.jl")
include("ad/gnf_adjoint_gradient.jl")
include("ad/gnf_adjoint_solvers.jl")

# Phase 2a platform residual/VJP seam depends on AD param types and step maps.
include("platform/residual.jl")
include("platform/calibration.jl")

# --- Curvilinear (body-fitted) mesh — v0.2 SLBM path ---
include("curvilinear/mesh.jl")
include("curvilinear/generators.jl")
include("curvilinear/slbm.jl")
include("curvilinear/mesh_3d.jl")
include("curvilinear/slbm_3d.jl")
include("curvilinear/mesh_from_arrays.jl")
include("curvilinear/mesh_gmsh.jl")

# --- Multi-block structured (v0.3) ---
include("multiblock/multiblock.jl")

# --- Parser ---
include("io/expression.jl")
include("io/krk/parser.jl")
include("io/krk/directives.jl")
include("io/krk/rheology.jl")
include("io/krk/units_bridge.jl")
include("io/krk/setup_lbm.jl")
include("io/krk/diagnostics.jl")
include("ad/ad_krk.jl")

# --- Grid refinement ---
include("refinement/refinement.jl")
include("refinement/conservative_tree_base_transfer_2d.jl")
include("refinement/conservative_tree_leaf_ops_2d.jl")
include("refinement/conservative_tree_patch_ledger_2d.jl")
include("refinement/conservative_tree_composite_ops_2d.jl")
include("refinement/conservative_tree_macroflows_2d.jl")
include("refinement/conservative_tree_reconstruction_2d.jl")
include("refinement/conservative_tree_multipatch_2d.jl")
include("refinement/conservative_tree_3d_core.jl")
include("refinement/conservative_tree_3d_ops.jl")
include("refinement/conservative_tree_topology_core_2d.jl")
include("refinement/conservative_tree_topology_ops_2d.jl")
include("refinement/conservative_tree_spec_2d.jl")
include("refinement/conservative_tree_projection_2d.jl")
include("refinement/conservative_tree_routes_2d.jl")
include("refinement/conservative_tree_stream_multilevel_2d.jl")
include("refinement/conservative_tree_topology_3d.jl")
include("refinement/conservative_tree_streaming_3d_routes.jl")
include("refinement/conservative_tree_streaming_3d_macroflow.jl")
include("refinement/conservative_tree_streaming_routes_2d.jl")
include("refinement/conservative_tree_streaming_adaptation_2d.jl")
include("refinement/conservative_tree_streaming_runs_2d.jl")
include("refinement/conservative_tree_streaming_obstacles_2d.jl")
include("refinement/conservative_tree_streaming_vfs_2d.jl")
include("refinement/conservative_tree_adaptation_2d.jl")
include("refinement/conservative_tree_subcycling_ledgers_2d.jl")
include("refinement/conservative_tree_subcycling_routes_2d.jl")
include("refinement/conservative_tree_subcycling_drivers_2d.jl")
include("refinement/conservative_tree_macroflows_subcycled_setup_2d.jl")
include("refinement/conservative_tree_macroflows_subcycled_drivers_2d.jl")
include("refinement/conservative_tree_gpu_pack_2d.jl")
include("kernels/refinement_exchange_2d.jl")
include("refinement/time_stepping.jl")
include("refinement/thermal_refinement.jl")
include("kernels/refinement_exchange_3d.jl")
include("refinement/refinement_3d_core.jl")
include("refinement/refinement_3d_thermal_bcs.jl")
include("refinement/conservative_tree_krk_validation_2d.jl")

# --- I/O ---
include("io/vtk_writer.jl")
include("io/diagnostics.jl")
include("io/checkpoint_hdf5.jl")  # write_checkpoint, read_checkpoint (only HDF5 user)

# --- Spatial boundary kernels ---
include("kernels/boundary_spatial_2d.jl")

# --- Geometry ---
include("geometry/Geometry.jl")
include("drivers/obstacle_3d.jl")
include("drivers/gnf_3d.jl")

# --- Carved .krk runner support ---
include("kernels/mesh_drag_2d.jl")
include("drivers/gmsh_slbm_drag.jl")
include("bc/handlers.jl")
include("io/output_emit.jl")
include("drivers/run_advection.jl")
include("drivers/run_twophase_vof.jl")
include("drivers/run_d3q19.jl")
include("drivers/run_refined.jl")
include("drivers/run_refined_3d.jl")
include("drivers/run_thermal.jl")
include("drivers/run_ehd.jl")

# --- Generic simulation runner ---
include("simulation_runner.jl")

# --- Post-processing helpers ---
include("postprocess.jl")

# --- LU <-> real-units conversion (self-contained submodule) ---
include("units/Units.jl")
export Units

# Lattice types and functions
export AbstractLattice, D2Q9, D3Q19
export lattice_dim, lattice_q, weights, velocities_x, velocities_y, velocities_z
export opposite, cs2, equilibrium

# Kernels
export stream_2d!, collide_2d!, stream_3d!, collide_3d!
export stream_periodic_x_wall_y_2d!, stream_fully_periodic_2d!, stream_periodic_x_axisym_2d!
export stream_periodic_xz_wall_y_3d!, stream_periodic_xz_movingwall_y_3d!
export stream_axisym_inlet_2d!
export collide_guo_2d!, collide_guo_field_2d!
export collide_guo_3d!, collide_guo_field_3d!
export compute_macroscopic_2d!, compute_macroscopic_3d!, compute_macroscopic_forced_2d!
export compute_macroscopic_forced_3d!, compute_macroscopic_forced_field_3d!, compute_macroscopic_pressure_2d!
export apply_zou_he_north_2d!, apply_zou_he_south_2d!
export apply_zou_he_west_2d!, apply_zou_he_pressure_east_2d!, apply_zou_he_pressure_west_2d!, apply_extrapolate_east_2d!
export apply_zou_he_top_3d!
export apply_zou_he_bottom_3d!, apply_zou_he_west_3d!, apply_zou_he_east_3d!
export apply_zou_he_south_3d!, apply_zou_he_north_3d!
export apply_zou_he_pressure_east_3d!, apply_zou_he_pressure_top_3d!
export apply_bounce_back_walls_3d!, apply_bounce_back_wall_3d!
export apply_bounce_back_walls_2d!, apply_bounce_back_wall_2d!

# Simulation
export LBMConfig, omega, reynolds
export initialize_2d, initialize_3d
export run_cavity_2d, run_cavity_3d
export run_gnf_channel_3d
export run_poiseuille_2d, run_couette_2d
export initialize_taylor_green_2d, run_taylor_green_2d
export initialize_cylinder_2d, run_cylinder_2d, compute_drag_mea_2d
export run_cylinder_libb_2d, compute_drag_libb_2d, compute_drag_libb_mei_2d
export rebuild_inlet_outlet_libb_2d!, rebuild_inlet_outlet_libb_3d!
export AbstractBC, HalfwayBB, InterfaceBC, ZouHeVelocity, ZouHePressure
export BCSpec2D, BCSpec3D, apply_bc_rebuild_2d!, apply_bc_rebuild_3d!
export compute_drag_libb_mei_2d_gpu!, compute_drag_libb_3d_gpu!
export CutLinkList, CutLinkList3D, build_cut_link_list_2d, build_cut_link_list_3d
export collide_thermal_2d!, compute_temperature_2d!
export apply_fixed_temp_south_2d!, apply_fixed_temp_north_2d!
export apply_fixed_temp_west_2d!, apply_fixed_temp_east_2d!
export apply_fixed_temp_2d!
export run_rayleigh_benard_2d, run_natural_convection_2d, run_natural_convection_refined_2d
export run_natural_convection_gnf_2d, gnf_reference_scales, cavity_powerlaw
export run_natural_convection_3d
export run_electroconvection_2d
export ECState, ECSolution
export ThermalPatchArrays, create_thermal_patch_arrays, advance_thermal_refined_step!
export collide_boussinesq_2d!, collide_boussinesq_vt_2d!, collide_boussinesq_vt_modified_2d!
export fused_natconv_step!, fused_natconv_vt_step!
export collide_thermal_3d!, compute_temperature_3d!, collide_boussinesq_3d!
export apply_fixed_temp_west_3d!, apply_fixed_temp_east_3d!
export apply_fixed_temp_south_3d!, apply_fixed_temp_north_3d!
export apply_fixed_temp_bottom_3d!, apply_fixed_temp_top_3d!
export apply_fixed_temp_3d!
export fused_bgk_step!, aa_even_step!, aa_odd_step!
export fused_trt_step!, trt_rates
export fused_trt_libb_step!, fused_trt_libb_v2_step!, fused_trt_libb_v2_hermite_step!, fused_trt_libb_v2_guo_field_step!, fused_trt_libb_v2_step_3d!, precompute_q_wall_cylinder
export dq_wall_dR_cylinder
export steady_shape_sensitivity
export precompute_q_wall_sphere_3d, compute_drag_libb_3d, run_sphere_libb_3d
export precompute_q_wall_annulus
export wall_velocity_rotating_cylinder, wall_velocity_rotating_inner
export persistent_fused_bgk!, persistent_aa_bgk!
export collide_axisymmetric_2d!, collide_li_axisym_2d!, run_hagen_poiseuille_2d

# MRT
export collide_mrt_2d!, collide_twophase_mrt_2d!
export ehd_collide_mrt_2d!

# Species transport
export collide_species_2d!, compute_concentration_2d!
export apply_fixed_conc_south_2d!, apply_fixed_conc_north_2d!

# Multiphase (Shan-Chen)
export compute_psi_2d!, compute_sc_force_2d!, collide_sc_2d!
export run_spinodal_2d, benchmark_mlups

# VOF PLIC
export compute_vof_normal_2d!, advect_vof_2d!
export compute_hf_curvature_2d!, compute_surface_tension_2d!
export collide_twophase_2d!, run_static_droplet_2d, run_plateau_pinch_2d
export add_azimuthal_curvature_2d!, add_axisym_viscous_correction_2d!, set_vof_west_2d!
export apply_density_correction_2d!
export run_rp_axisym_2d, run_rp_pressure_vof_2d, run_rp_hybrid_2d
export run_cij_jet_axisym_2d, run_cij_jet_phasefield_2d, run_cij_jet_hybrid_2d
export collide_pressure_vof_mrt_2d!, compute_surface_tension_weighted_2d!
export init_pressure_vof_equilibrium, smooth_vof_2d!, correct_mass_2d!
export add_axisym_viscous_weighted_2d!
export extrapolate_velocity_ghost_2d!, reset_feq_ghost_2d!

# Phase-field (Allen-Cahn + pressure-based)
export phasefield_params, compute_phi_2d!, compute_chemical_potential_2d!
export add_azimuthal_chemical_potential_2d!, compute_phasefield_force_2d!
export compute_vof_from_phi_2d!, compute_antidiffusion_flux_2d!
export collide_allen_cahn_2d!, add_azimuthal_allen_cahn_source_2d!
export collide_pressure_phasefield_mrt_2d!, compute_macroscopic_phasefield_2d!
export set_phasefield_west_2d!, extrapolate_phasefield_east_2d!
export init_phasefield_equilibrium, init_pressure_equilibrium
export run_static_droplet_phasefield_2d

# Dual-grid VOF
export prolongate_bilinear_2d!, restrict_average_2d!
export compute_hf_curvature_dx_2d!, compute_surface_tension_dx_2d!
export run_static_droplet_dualgrid_2d


# Prescribed-velocity advection
export clamp_field_2d!, advect_vof_step!, advect_vof_plic_step!
export advect_vof_plic_2d!, fill_velocity_field!, init_vof_field!
export run_advection_2d

# I/O
export write_vtk, create_pvd, write_vtk_to_pvd
export setup_output_dir, write_snapshot_2d!, write_snapshot_3d!
export write_vtk_multiblock, write_snapshot_refined_2d!
export open_paraview
export DiagnosticsLogger, open_diagnostics, log_diagnostics!, close_diagnostics!

# .krk config system
export KrakenExpr, parse_kraken_expr, evaluate, has_variable, is_time_dependent, is_spatial
export SimulationSetup, DomainSetup, PhysicsSetup, GeometryRegion, BoundarySetup, RheologySetup
export InitialSetup, OutputSetup, DiagnosticsSetup, STLSource, RefineSetup, SanityIssue
export load_kraken, parse_kraken, build_rheology_model,
       parse_kraken_sweep, load_kraken_sweep, sanity_check, sanity_check_sweep
export LBMParams, lbm_params, lbm_params_table
export run_simulation

# Curvilinear mesh (v0.2 SLBM path)
export CurvilinearMesh, build_mesh, validate_mesh, compute_metric
export polar_mesh, stretched_box_mesh, cartesian_mesh, cylinder_focused_mesh
export cell_area, domain_extent
export SLBMGeometry, build_slbm_geometry, transfer_slbm_geometry
export slbm_bgk_step!, slbm_bgk_moving_step!, slbm_mrt_step!
export slbm_trt_libb_step!, slbm_trt_libb_step_local_2d!, slbm_trt_libb_step_local_biquad_2d!
export slbm_reg_libb_step_local_2d!
export precompute_q_wall_slbm_cylinder_2d
export compute_local_omega_2d
export PullSLBM, CollideTRTLocalDirect
# 3D SLBM
export CurvilinearMesh3D, build_mesh_3d, validate_mesh_3d
export stretched_box_mesh_3d, cartesian_mesh_3d
export SLBMGeometry3D, build_slbm_geometry_3d, transfer_slbm_geometry_3d
export slbm_bgk_step_3d!
export slbm_trt_libb_step_3d!, slbm_trt_libb_step_local_3d!
export precompute_q_wall_slbm_sphere_3d, compute_local_omega_3d
export PullSLBM_3D, CollideTRTLocalDirect_3D
# gmsh / external mesh import
export load_gmsh_mesh_2d, load_gmsh_mesh_3d, GmshPhysicalGroups

# Multi-block structured (v0.3)
export Block, Interface, MultiBlockMesh2D
export EDGE_SYMBOLS_2D, INTERFACE_TAG
export getblock, edge_length, edge_coords
export MultiBlockSanityIssue, sanity_check_multiblock
export BlockState2D, allocate_block_state_2d, interior_f, interior_macro, ext_dims
export exchange_ghost_2d!, exchange_ghost_shared_node_2d!, fill_ghost_corners_2d!, fill_physical_wall_ghost_2d!, fill_slbm_wall_ghost_2d!
export load_gmsh_multiblock_2d
export reorient_block, autoreorient_blocks, transpose_multiblock
export extend_mesh_2d, build_block_slbm_geometry_extended, extend_interior_field_2d

# Grid refinement
export RefinementPatch, RefinedDomain
export create_patch, create_refined_domain, rescaled_omega
export rescaling_factor_c2f, rescaling_factor_f2c
export prolongate_f_rescaled_2d!, restrict_f_rescaled_2d!
export temporal_interpolate_2d!, copy_macroscopic_overlap_2d!
export advance_refined_step!
export d2q9_cx, d2q9_cy
export d2q9_opposite
export coalesce_F_2d!, explode_uniform_F_2d!
export mass_F, momentum_F, moments_F
export fill_equilibrium_integrated_D2Q9!
export macrostate_integrated_D2Q9
export reconstruct_integrated_D2Q9_eq_neq!
export reconstructed_integrated_D2Q9_packet
export d3q19_cx, d3q19_cy, d3q19_cz, d3q19_opposite
export coalesce_F_3d!, explode_uniform_F_3d!
export mass_F_3d, momentum_F_3d, moments_F_3d
export fill_equilibrium_integrated_D3Q19!
export conservative_tree_parent_index
export conservative_tree_parent_index_3d
export coalesce_patch_to_shadow_F_3d!, explode_shadow_to_patch_uniform_F_3d!
export active_population_sums_F_3d, active_mass_F_3d
export active_momentum_F_3d, active_moments_F_3d
export collide_BGK_integrated_D3Q19!, collide_Guo_integrated_D3Q19!
export collide_BGK_composite_F_3d!, collide_Guo_composite_F_3d!
export split_coarse_to_fine_vertical_F_2d!, coalesce_fine_to_coarse_vertical_F
export split_coarse_to_fine_face_F_2d!, coalesce_fine_to_coarse_face_F
export split_coarse_to_fine_corner_F_2d!, coalesce_fine_to_coarse_corner_F
export split_coarse_to_fine_face_F_3d!, coalesce_fine_to_coarse_face_F_3d
export split_coarse_to_fine_edge_F_3d!, coalesce_fine_to_coarse_edge_F_3d
export split_coarse_to_fine_corner_F_3d!, coalesce_fine_to_coarse_corner_F_3d
export coarse_to_fine_patch_boundary_F_2d!, fine_to_coarse_patch_boundary_F_2d!
export collide_BGK_integrated_D2Q9!
export collide_Guo_integrated_D2Q9!
export stream_fully_periodic_F_2d!, stream_periodic_x_wall_y_F_2d!
export stream_periodic_x_moving_wall_y_F_2d!
export cylinder_solid_mask_leaf_2d, square_solid_mask_leaf_2d
export backward_facing_step_solid_mask_leaf_2d
export vertical_facing_step_solid_mask_leaf_2d
export stream_periodic_x_wall_y_solid_F_2d!
export stream_bounceback_xy_solid_F_2d!
export apply_zou_he_west_F_2d!, apply_zou_he_pressure_east_F_2d!
export apply_zou_he_west_cell_F_2d!, apply_zou_he_pressure_east_cell_F_2d!
export apply_composite_zou_he_west_F_2d!, apply_composite_zou_he_pressure_east_F_2d!
export compute_drag_mea_solid_F_2d
export ConservativeTreePatch2D, create_conservative_tree_patch_2d
export ConservativeTreePatchSet2D, create_conservative_tree_patch_set_2d
export conservative_tree_parent_owner_2d, conservative_tree_leaf_owner_2d
export conservative_tree_patch_owner_counts_2d, active_coarse_mask
export conservative_tree_patch_ranges_from_krk_refines_2d
export create_conservative_tree_patch_set_from_krk_2d
export ConservativeTreeRefineBlock2D, ConservativeTreeSpec2D
export create_conservative_tree_spec_2d
export conservative_tree_cell_id_2d, conservative_tree_children_2d
export conservative_tree_is_active_leaf_2d
export conservative_tree_refine_blocks_from_krk_2d
export create_conservative_tree_spec_from_krk_2d
export allocate_conservative_tree_F_2d
export active_population_sums_F_2d, level_population_sums_F_2d
export coalesce_conservative_tree_ledgers_F_2d!
export explode_conservative_tree_ledgers_F_2d!
export ConservativeTreeRouteTable2D, create_conservative_tree_route_table_2d
export stream_conservative_tree_routes_F_2d!
export coalesce_patch_to_shadow_F_2d!, explode_shadow_to_patch_uniform_F_2d!
export active_population_sums_F, active_mass_F, active_momentum_F, active_moments_F
export composite_to_leaf_F_2d!, leaf_to_composite_F_2d!
export stream_composite_fully_periodic_leaf_F_2d!
export stream_composite_periodic_x_wall_y_leaf_F_2d!
export stream_composite_periodic_x_moving_wall_y_leaf_F_2d!
export collide_BGK_composite_F_2d!, collide_Guo_composite_F_2d!
export LinkKind, SAME_LEVEL, COARSE_TO_FINE, FINE_TO_COARSE, BOUNDARY
export RouteKind, DIRECT, SPLIT_FACE, SPLIT_CORNER, COALESCE_FACE
export COALESCE_CORNER, ROUTE_BOUNDARY
export AbstractCellMetrics, CartesianMetrics2D
export ConservativeTreeCell2D, ConservativeTreeLink2D, ConservativeTreeRoute2D
export ConservativeTreeTopology2D, create_conservative_tree_topology_2d
export ConservativeTreeBlock2D, ConservativeTreePackedRoute2D
export ConservativeTreePackedTopology2D, pack_conservative_tree_topology_2d
export CartesianMetrics3D
export ConservativeTreePatch3D, create_conservative_tree_patch_3d
export RouteKind3D, DIRECT_3D, SPLIT_FACE_3D, SPLIT_EDGE_3D
export COALESCE_FACE_3D, COALESCE_EDGE_3D, ROUTE_BOUNDARY_3D
export ConservativeTreeCell3D, ConservativeTreeLink3D, ConservativeTreeRoute3D
export ConservativeTreeTopology3D, create_conservative_tree_topology_3d
export active_volume, morton_key_2d
export stream_composite_routes_interior_F_3d!
export stream_composite_routes_periodic_x_F_3d!
export stream_composite_routes_periodic_x_wall_yz_F_3d!
export stream_composite_routes_interior_F_2d!
export stream_composite_routes_periodic_x_F_2d!
export stream_composite_routes_periodic_x_wall_y_F_2d!
export stream_composite_routes_periodic_x_moving_wall_y_F_2d!
export stream_composite_routes_zou_he_x_wall_y_F_2d!
export stream_composite_routes_zou_he_x_wall_y_solid_F_2d!
export stream_composite_routes_periodic_x_wall_y_solid_F_2d!
export collide_Guo_composite_solid_F_2d!
export regrid_conservative_tree_patch_F_2d!
export regrid_conservative_tree_patch_direct_F_2d!
export conservative_tree_solid_mask_patch_range_2d
export conservative_tree_indicator_patch_range_2d
export conservative_tree_gradient_indicator_2d
export conservative_tree_hysteresis_patch_range_2d
export conservative_tree_velocity_gradient_patch_range_2d
export adapt_conservative_tree_patch_to_solid_mask_2d
export adapt_conservative_tree_patch_to_velocity_gradient_2d
export ConservativeTreeAdaptationPolicy2D, ConservativeTreePatchProposal2D
export ConservativeTreeAdaptationPlan2D
export conservative_tree_adaptation_plan_2d
export conservative_tree_adaptation_plan_from_proposal_2d
export conservative_tree_indicator_adaptation_plan_2d
export conservative_tree_patch_proposals_from_krk_2d
export conservative_tree_adaptation_policy_from_krk_refine_2d
export conservative_tree_indicator_adaptation_plan_from_krk_2d
export adapt_conservative_tree_patch_with_plan_2d
export ConservativeTreeSubcycleLedger2D
export create_conservative_tree_subcycle_ledger_2d
export reset_conservative_tree_subcycle_ledger_2d!
export conservative_tree_subcycle_weights_2d
export conservative_tree_subcycle_deposit_coarse_to_fine_face_2d!
export conservative_tree_subcycle_deposit_coarse_to_fine_corner_2d!
export conservative_tree_subcycle_accumulate_fine_to_coarse_face_2d!
export conservative_tree_subcycle_accumulate_fine_to_coarse_corner_2d!
export conservative_tree_subcycle_orientation_sums_2d
export conservative_tree_subcycle_total_sums_2d
export ConservativeTreeGPURoutePack2D
export pack_conservative_tree_gpu_routes_2d
export conservative_tree_gpu_route_weight_sums_2d
export stream_conservative_tree_gpu_pack_interior_F_2d!
export ConservativeTreeAdaptiveRun2D
export ConservativeTreeSolidAdaptiveRun2D
export ConservativeTreeOpenChannelRun2D
export ConservativeTreeMacroFlow2D, ConservativeTreeCylinderResult2D
export ConservativeTreeCylinderChannelResult2D
export ConservativeTreeSolidFlowResult2D
export composite_leaf_mean_ux_profile
export composite_leaf_velocity_field_2d
export couette_analytic_profile_2d, poiseuille_analytic_profile_2d
export run_conservative_tree_couette_macroflow_2d
export run_conservative_tree_poiseuille_macroflow_2d
export run_conservative_tree_couette_route_native_2d
export run_conservative_tree_poiseuille_route_native_2d
export run_conservative_tree_open_channel_route_native_2d
export run_conservative_tree_bfs_route_native_2d
export run_conservative_tree_poiseuille_adaptive_route_native_2d
export run_conservative_tree_poiseuille_gradient_adaptive_route_native_2d
export validate_conservative_tree_route_native_phase_p_2d
export run_conservative_tree_square_obstacle_macroflow_2d
export run_conservative_tree_square_obstacle_route_native_2d
export run_conservative_tree_cylinder_obstacle_route_native_2d
export ConservativeTreeBenchmarkRow2D
export benchmark_conservative_tree_cartesian_vs_amr_2d
export ConservativeTreeConvergenceRow2D
export convergence_conservative_tree_obstacles_2d
export run_conservative_tree_vfs_route_native_2d
export run_conservative_tree_vfs_mask_adaptive_route_native_2d
export run_conservative_tree_bfs_macroflow_2d
export run_conservative_tree_cylinder_macroflow_2d
export run_conservative_tree_cylinder_channel_macroflow_2d
export ConservativeTreeSpecMacroFlow2D
export create_conservative_tree_nested_channel_spec_2d
export initialize_conservative_tree_equilibrium_F_2d!
export conservative_tree_leaf_mean_ux_profile_2d
export conservative_tree_mass_roundoff_rtol_2d
export run_cartesian_channel_mass_reference_2d
export run_conservative_tree_poiseuille_subcycled_2d
export run_conservative_tree_couette_subcycled_2d
export ConservativeTreeAMRDKrkCase2D
export conservative_tree_amr_d_boundary_policy_2d
export conservative_tree_amr_d_geometry_2d
export conservative_tree_amr_d_case_from_krk_2d
export conservative_tree_amr_d_support_matrix_2d
export run_conservative_tree_amr_d_case_from_krk_2d
export RefinementPatch3D, RefinedDomain3D
export create_patch_3d, create_refined_domain_3d, advance_refined_step_3d!
export prolongate_f_rescaled_3d!, prolongate_f_rescaled_full_3d!
export prolongate_f_rescaled_temporal_3d!, restrict_f_rescaled_3d!
export ThermalPatchArrays3D, create_thermal_patch_arrays_3d
export advance_thermal_refined_step_3d!, build_patch_thermal_bcs_3d
export fill_thermal_full_3d!, fill_thermal_ghost_3d!, restrict_thermal_to_coarse_3d!
export build_patch_flow_bcs_3d
export TwophaseRefinedArrays, create_twophase_patch_arrays, advance_twophase_refined_step!

# STL geometry
export STLTriangle, STLMesh, read_stl, transform_mesh
export voxelize_2d, voxelize_3d
export precompute_q_wall_from_stl_2d, precompute_q_wall_from_stl_3d

# Post-processing
export extract_line, field_error, probe, domain_stats
export load_basilisk_interfaces, load_basilisk_interface_contour
export find_basilisk_snapshot, compare_interfaces

# Rheology
export AbstractRheology, GeneralizedNewtonian, Viscoelastic
export AbstractThermalCoupling, IsothermalCoupling, ArrheniusCoupling, WLFCoupling
export Newtonian, PowerLaw, CarreauYasuda, Cross, Bingham, HerschelBulkley
export OldroydB, FENEP, Saramito
export StressFormulation, LogConfFormulation
export effective_viscosity, effective_viscosity_thermal, thermal_shift_factor
export strain_rate_magnitude_2d, strain_rate_magnitude_3d
export fused_trt_gnf_libb_guo_field_step_3d!
export collide_rheology_2d!, collide_rheology_guo_2d!, collide_rheology_thermal_2d!
export collide_twophase_rheology_2d!
# Steady GNF (per-cell pure functions, split/fused kernels, periodic-x driver)
export gnf_moments_cell, gnf_feq_cell, gnf_strain_rate_cell, gnf_collide_cell
export gnf_viscosity, gnf_cell_update
export gnf_strain_rate_2d!, closure_eval!, collide_gnf_cell_2d!, collide_gnf_fused_2d!
export gnf_macroscopic_post_2d!, gnf_step!, gnf_rest_state, run_gnf_periodic_steady
# Steady GNF adjoint (local blocks, JVP/VJP, gauge-bordered GMRES, parameter gradient)
export stream_periodic_x_wall_y_transpose_2d!
export GNFLinearisation, GNFStepOperator, GNFBorderedAdjoint, GNFVelocityObs
export gnf_linearise, gnf_active_mask, gnf_post_moments, gnf_jvp!, gnf_vjp!, gnf_gmres!
export gnf_obs_values, gnf_obs_objective, gnf_obs_rhs, gnf_adjoint_solve
export gnf_parameter_gradient, gnf_steady_gradient, gnf_power_iteration, gnf_adjoint_series
export gnf_pack_state, gnf_unpack_state, gnf_step_state
export gnf_steady_adjoint, gnf_obs_grad_u, gnf_velocity_rhs, gnf_nu_sensitivity
export GNFKStepAdjoint, gnf_adjoint_fixed_point, gnf_adjoint_kstep_solve, gnf_adjoint_power!, gnf_tangent_power!

# Viscoelastic
export eigen_sym2x2, mat_exp_sym2x2, mat_log_sym2x2, decompose_velocity_gradient
# 3D viscoelastic (Oldroyd-B confined sphere + planar Couette)
export run_conformation_sphere_libb_3d
export run_conformation_couette_libb_3d
export run_conformation_poiseuille_libb_3d
export eigen_sym3x3, mat_exp_sym3x3, mat_log_spd_sym3x3
export apply_hermite_source_3d!, update_polymer_stress_3d!
export compute_polymeric_force_3d!, fused_trt_libb_v2_guo_field_step_3d!
export init_conformation_field_3d!, collide_conformation_3d!
export compute_conformation_macro_3d!, apply_cnebb_conformation_3d!
export apply_cnebb_conformation_y_walls_3d!
export reset_conformation_inlet_3d!, reset_conformation_outlet_3d!
export compute_polymeric_force_2d!
export evolve_stress_2d!, evolve_logconf_2d!
export compute_stress_from_conf_2d!, compute_stress_from_logconf_2d!
export run_viscoelastic_cylinder_2d
export run_viscoelastic_logfv_channel_2d, run_viscoelastic_logfv_frozen_channel_cde_2d
export run_viscoelastic_logfv_frozen_circle_shear_cde_2d
export run_viscoelastic_logfv_frozen_circle_tangential_shear_cde_2d
export run_viscoelastic_logfv_poiseuille_frozen_force_2d
export run_viscoelastic_logfv_poiseuille_coupled_2d
export run_viscoelastic_logfv_square_periodic_2d
export run_viscoelastic_logfv_bfs_passive_2d, run_viscoelastic_logfv_bfs_coupled_2d
export run_viscoelastic_logfv_contraction_coupled_2d
export run_viscoelastic_logfv_square_channel_coupled_2d
export run_viscoelastic_logfv_cylinder_coupled_2d
export FVFDDomainBC2D, FVFDFieldBC2D, FVFDEmbeddedBoundary2D, FVFDPatch2D, FVFDGeometry2D
export fvfd_domain_bc_code, fvfd_periodicx_wally_bcspec_2d, fvfd_openx_wally_bcspec_2d
export fvfd_empty_embedded_boundary_2d, fvfd_embedded_boundary_from_qwall_2d
export fvfd_embedded_boundary_from_halfplane_2d, fvfd_geometry_from_halfplane_2d
export fvfd_embedded_boundary_from_circle_2d, fvfd_geometry_from_circle_2d
export fvfd_transfer_field_bc_2d, fvfd_transfer_embedded_boundary_2d
export fvfd_geometry_from_lbm_2d, fvfd_transfer_geometry_2d
export fvfd_velocity_gradient_2d!, fvfd_velocity_gradient_embedded_2d!
export fvfd_tensor_divergence_2d!, fvfd_tensor_divergence_embedded_2d!
export fvfd_embedded_wall_traction_2d!
export WallGradientOrder, WallGradientSides, apply_halfway_wall_gradient_correction!
export fvfd_bsd_force_2d!
export fvfd_cell_velocity_to_faces_2d!, fvfd_cell_velocity_to_faces_embedded_2d!
export fvfd_advect_upwind_2d!, fvfd_advect_upwind_embedded_2d!
export fvfd_sym2_advect_upwind_2d!, fvfd_sym2_advect_upwind_embedded_2d!
export LogFVDomainBC2D, LogFVFieldBC2D, logfv_domain_bc_code
export logfv_periodicx_wally_bcspec_2d, logfv_openx_wally_bcspec_2d,
       logfv_wallxwally_bcspec_2d
export fvfd_wallxwally_bcspec_2d
export logfv_cell_velocity_to_faces_bc_aware_2d!, logfv_cell_velocity_to_faces_embedded_2d!
export StepChannelGeometry2D, step_channel_geometry_2d
export contraction_step_geometry_2d, backward_facing_step_geometry_2d,
       square_obstacle_channel_geometry_2d
export transfer_step_geometry_2d, parabolic_face_profile_2d
export oldroydb_inlet_conformation_profile_2d, default_step_bcspec_2d
export AbstractPolymerModel, LogConfOldroydB, LogConfFENEP, LogConfGiesekus,
       LogConfPTT, update_polymer_stress!
export uses_log_conformation, polymer_max_extensibility, polymer_mobility,
       polymer_ptt_epsilon, polymer_ptt_variant
export AbstractPolymerWallBC, CNEBB, CNEBBQAware, CNEBBField,
       CNEBBFieldEquilibrium,
       CNEBBEqGradient, CNEBBCutLinkEqGradient, YLW_A, YLW_B, YLWBalanceOnly,
       ExtrapEqWallBC, LogFieldWallBC, NoPolymerWallBC, apply_polymer_wall_bc!

# Spatial boundary kernels
export apply_zou_he_north_spatial_2d!, apply_zou_he_south_spatial_2d!
export apply_zou_he_west_spatial_2d!, apply_zou_he_pressure_east_spatial_2d!
export apply_zou_he_pressure_inlet_west_2d!

end # module Kraken
