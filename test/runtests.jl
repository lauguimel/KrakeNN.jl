using Test
using Kraken

# KRAKEN_SKIP_AD=true drops every Enzyme-driven tier: the AD sensitivity tests
# and the calibration twin experiments. Measured on 2026-09-15 (CI, Julia 1.11,
# ubuntu-latest), those are 93 min of the suite's 165 min — 56% of wall-clock
# for the seam that changes least often. The pull-request CI stage sets it; the
# full suite on dev/platform and the nightly run do not.
const SKIP_AD = get(ENV, "KRAKEN_SKIP_AD", "false") == "true"

# KRAKEN_SKIP_VE3D=true drops the ten full 3D viscoelastic flow validations.
# Measured 2026-09-16 on the non-AD suite, they are 52% of its wall-clock — and
# unlike the cheap operator tests above them, their domain size IS the test:
# shrinking them would mean loosening their analytical tolerances, which
# AGENTS.md classes as a regression. So they move stage rather than shrink.
const SKIP_VE3D = get(ENV, "KRAKEN_SKIP_VE3D", "false") == "true"

if get(ENV, "KRAKEN_AD_ONLY", "false") == "true"
    include("ad/test_ad_sensitivity.jl")
    exit()
end

# KRAKEN_ONLY=<path relative to test/> runs a single test file and exits.
#
# Added because diagnosing the Enzyme segfault in ad/test_ad_ve_sensitivity.jl
# had no cheap route: KRAKEN_AD_ONLY runs a different file, and reaching the
# crashing one otherwise costs the whole suite — 2h45 on CI — per attempt.
# The crash does not reproduce on macOS arm64 (verified 2026-09-16: 19/19 in
# 22.5 s on Julia 1.11.9, with and without CUDA loaded), so it has to be
# diagnosed on the CI platform itself.
#
#   KRAKEN_ONLY=ad/test_ad_ve_sensitivity.jl julia --project -e 'using Pkg; Pkg.test()'
#
# Or on CI: run the workflow manually and set the `only` input.
let only = get(ENV, "KRAKEN_ONLY", "")
    if !isempty(only)
        isfile(only) || error("KRAKEN_ONLY: no such test file: $(only)")
        @info "Running a single test file (KRAKEN_ONLY)" file=only
        include(only)
        exit()
    end
end

# KRAKEN_ONLY=<relative path> runs a single test file and exits. Added because
# diagnosing the Enzyme segfault in ad/test_ad_ve_sensitivity.jl had no cheap
# route: KRAKEN_AD_ONLY runs a different file, and reaching the crashing one
# otherwise means paying the whole suite (2h45 on CI) per attempt.
#
#   KRAKEN_ONLY=ad/test_ad_ve_sensitivity.jl julia --project -e 'using Pkg; Pkg.test()'
let only = get(ENV, "KRAKEN_ONLY", "")
    if !isempty(only)
        isfile(only) || error("KRAKEN_ONLY: no such test file: $(only)")
        @info "Running a single test file (KRAKEN_ONLY)" file=only
        include(only)
        exit()
    end
end

# IncNS + solve-services tier (platform contract, linear-solve seam, Poisson
# services, IncNS drivers, scalar transport). Runs after the LBM tier by
# default; KRAKEN_INCNS_ONLY=true runs it alone (mirrors KRAKEN_AD_ONLY).
# Heavy validations (Ghia cavities, MG MMS up to 512²) additionally gate on
# KRAKEN_TEST_HEAVY=true.
function run_incns_testset()
    @testset "IncNS + solve services" begin
        # Platform-contract parity (IncNS wrapper vs direct driver call).
        include("platform/incns_contract_test.jl")

        # Solve services: linear-solve seam + elliptic (Poisson) MMS receipts.
        include("analytical/linear_solve_nonsym.jl")
        include("analytical/linear_solve_refactor_transpose.jl")   # C0: refactorize / transpose / multi-RHS
        include("analytical/poisson_mms.jl")
        include("analytical/poisson_embedded_mms.jl")
        include("analytical/poisson_embedded_fvfd_mms.jl")

        # LinearSolve.jl front-end (weakdep ext). Guard the LOAD, not the
        # tests (Enzyme-guard pattern), so real ext failures still surface
        # when LinearSolve IS present in the environment.
        let ls_ok = try
                @eval Main using LinearSolve
                true
            catch
                false
            end
            if ls_ok
                include("analytical/poisson_linearsolve_mms.jl")
            else
                @info "Skipping LinearSolve front-end tests (LinearSolve not loadable in this environment)"
            end
        end

        # cuDSS GPU direct path (weakdep ext) — self-gated on CUDA.functional()
        # inside the file; skips cleanly on CPU-only boxes.
        include("analytical/poisson_cudss_gpu.jl")
        # C0 cuDSS contract (refactorize, transpose gate, multi-RHS, zero
        # allocation) — same self-gating; KRAKEN_REQUIRE_GPU=1 makes a skip fail.
        include("analytical/linear_solve_cudss_c0_gpu.jl")

        # FVFD velocity-operator trio (grad/div/laplacian + embedded variants).
        include("analytical/incns_grad_div_laplacian_mms.jl")

        # IncNS drivers + scalar transport, analytical validation.
        include("analytical/incns_poiseuille.jl")
        include("analytical/incns_unsteady_taylor_green.jl")
        include("analytical/incns_unsteady_startup_channel.jl")
        include("analytical/incns_manifold.jl")   # fast (~4 s CPU): stays non-heavy
        include("analytical/incns_momentum_advection_order.jl")
        # C1 steady FV residual (KA kernel, generic eltype): exact discrete
        # solutions, MMS orders, duals vs FD, SIMPLE cavity agreement (~15 s CPU)
        include("analytical/incns_fv_steady_residual.jl")
        # its GPU parity + zero-allocation contract — self-gated on CUDA;
        # KRAKEN_REQUIRE_GPU=1 makes a skip fail
        include("analytical/incns_fv_steady_residual_gpu.jl")
        include("analytical/scalar_transport_heated_channel.jl")
        include("analytical/scalar_transport_advection_order.jl")

        # Heavy validations (long CPU runs: Ghia cavities, MG MMS up to 512²) —
        # opt in via KRAKEN_TEST_HEAVY=true.
        if get(ENV, "KRAKEN_TEST_HEAVY", "false") == "true"
            include("analytical/incns_cavity_ghia.jl")
            include("analytical/incns_cavity_mg_ghia.jl")
            include("analytical/poisson_mg_mms.jl")
        else
            @info "Skipping heavy IncNS validations (set KRAKEN_TEST_HEAVY=true to run)"
        end
    end
end

if get(ENV, "KRAKEN_INCNS_ONLY", "false") == "true"
    run_incns_testset()
    exit()
end

# Package metadata (offline, no GPU): weakdep UUIDs vs General registry (#61).
include("platform/weakdeps_metadata_test.jl")

@testset "Kraken.jl LBM" begin
    include("platform/contract_parity_test.jl")
    include("platform/state_contract_test.jl")
    include("platform/residual_vjp_test.jl")
    if SKIP_AD
        @info "Skipping calibration twin experiments (KRAKEN_SKIP_AD=true)"
    else
        include("platform/calibration_test.jl")
        include("platform/calibration_nufield_test.jl")
    end
    include("test_lbm_basic.jl")
    include("test_poiseuille.jl")
    include("test_guo_convention_pairs.jl")
    # Issue #67: TRT Guo-field brick injects exactly F per collision.
    include("analytical/trt_guo_field_momentum.jl")
    # Issue #18: west/east pressure channel through the public .krk runner.
    include("analytical/H2-004-route.jl")
    include("test_poiseuille_3d.jl")
    include("test_couette.jl")
    include("test_taylor_green.jl")
    include("test_thermal.jl")
    include("test_natconv_gnf.jl")
    # v0.7: parabolic inlets carry the exact flow rate between half-way walls.
    include("analytical/halfway_inlet_flux.jl")
    # v0.7: the hot-wall Nusselt average covers every row.
    include("analytical/hot_wall_nusselt_average.jl")
    # v0.7: adiabatic walls reflect the thermal populations specularly.
    include("analytical/thermal_adiabatic_walls.jl")
    # Issue #19: conduction .krk fallback honours nu, alpha and thermal faces.
    include("analytical/TH-002-route.jl")
    include("test_axisymmetric.jl")
    include("test_mrt.jl")
    include("analytical/ehd_ec_split_parity_2d.jl")
    include("analytical/EC-RESTART.jl")
    include("analytical/EC-RESTART-CUDA.jl")
    include("analytical/ehd_hydrostatic_2d.jl")
    include("analytical/ehd_static_moment_2d.jl")
    include("analytical/ehd_transient_moment_2d.jl")
    include("analytical/ehd_krk_2d.jl")
    include("analytical/EC-SIDEWALLS.jl") # Issue #21: selectable ECState flow walls.
    include("analytical/ehd_mapping_parity_2d.jl")
    include("analytical/ehd_mrt_smoke_2d.jl")
    # ~40 s: two 50k-cycle canaries bracketing the electroconvection onset.
    include("analytical/ehd_onset_2d.jl")
    include("analytical/ehd_phi_direct_2d.jl")
    # Issue #23: the DDF potential solve stops on the field too. One case stays
    # @test_broken (default settings, slow Poisson mode, issue #63).
    include("analytical/ES-002-STOP.jl")
    include("analytical/ehd_phi_ddf_solve_2d.jl")
    include("analytical/ehd_twin_parity_2d.jl")
    include("analytical/ehd_phi_gpu_parity_2d.jl")
    include("test_species.jl")
    include("test_multiphase.jl")
    include("test_vof.jl")
    include("test_benchmark.jl")
    include("test_cavity.jl")
    include("test_cavity_3d.jl")
    include("test_thermal_3d_krk.jl")
    include("test_cylinder.jl")
    include("test_expression.jl")
    include("test_kraken_parser.jl")
    include("test_krk_symbolic.jl")
    include("test_simulation_runner.jl")
    include("test_stl.jl")
    include("test_geometry_stl_krk.jl")
    include("test_geometry_descriptor.jl")
    include("test_geometry_units_krk.jl")
    include("test_geometry_units_3d_krk.jl")
    include("test_geometry_stl_flow_3d_krk.jl")
    include("test_krk_examples.jl")
    include("test_refinement.jl")
    include("test_conservative_tree_2d.jl")
    include("test_conservative_tree_topology_2d.jl")
    include("test_conservative_tree_streaming_2d.jl")
    include("test_curvilinear_mesh.jl")
    include("test_slbm.jl")
    include("test_slbm_taylor_green.jl")
    include("test_slbm_taylor_couette.jl")
    include("test_fused_trt_2d.jl")
    include("test_li_bb_2d.jl")
    include("test_kernel_dsl.jl")
    include("test_couette_libb_canary.jl")
    include("test_couette_libb_canary_3d.jl")
    include("test_cylinder_libb.jl")
    include("test_sphere_libb.jl")
    include("test_sphere_stl_drag_krk.jl")
    include("test_slbm_libb_3d.jl")
    # Gmsh is an optional mesh-import dependency, not a Kraken dep. Guard the LOAD
    # (same pattern as the Enzyme block below) so its absence skips cleanly instead
    # of erroring the whole suite, while real loader failures still surface when it
    # IS installed.
    let gmsh_ok = try
            @eval Main using Gmsh
            true
        catch
            false
        end
        if gmsh_ok
            include("test_gmsh_loader.jl")
        else
            @info "Skipping gmsh loader tests (Gmsh not installed in this environment)"
        end
    end
    include("test_multiblock_topology.jl")
    include("test_multiblock_exchange.jl")
    include("test_multiblock_canal.jl")
    include("test_stl_libb.jl")
    include("test_taylor_couette_libb.jl")
    include("test_advection_prescribed.jl")
    include("test_krk_multiphase.jl")
    include("test_vtk_3d.jl")
    include("test_phasefield.jl")
    include("test_twophase_rheology.jl")
    include("test_postprocess.jl")
    include("test_rheology.jl")
    include("rheology/gnf_cell_test.jl")   # steady GNF forward (per-cell functions, driver)
    include("ad/gnf_adjoint_test.jl")      # steady GNF adjoint (blocks, JVP/VJP, GMRES; no Enzyme)
    include("analytical/gnf3d_strain_rate.jl")
    include("analytical/gnf3d_channel_order.jl")
    include("analytical/gnf3d_parity.jl")
    include("test_viscoelastic.jl")
    include("test_viscoelastic_krk.jl")
    include("test_bsd_open_corner_2d.jl")
    # --- 3D viscoelastic tier (FVFD/FD transport + log-conformation, four
    #     constitutive models, RheoTool cross-validation). This capability
    #     lives only on this line; keep it registered here. ---
    # Operator- and model-level 3D tests: cheap, and they are what catches a
    # broken kernel. These always run.
    include("test_fvfd_operators_3d.jl")
    include("test_fvfd_boundary_stencils_3d.jl")
    include("test_fvfd_logconf_3d.jl")
    include("test_fvfd_fenep_3d.jl")
    include("test_fvfd_giesekus_3d.jl")
    include("test_fvfd_ptt_3d.jl")
    include("test_fvfd_velocity_gradient_3d.jl")
    # Known defects of the extensional driver (@test_broken), on a 12x12x4 box
    # for 2 to 20 steps, so they run on the pull-request path and a fix shows
    # its Unexpected Pass there:
    # issue #68 (post-collision velocity read-back) and
    # issue #69 (Zou-He faces skip k = 1 and k = Nz).
    include("analytical/ve3d_ext_velocity_readback.jl")
    include("analytical/ve3d_ext_zouhe_zplanes.jl")
    # Full 3D flow validations: these must reach a developed state for the
    # analytical comparison to mean anything, so they cannot be shrunk without
    # loosening their gates. Measured 2026-09-16, they are 52% of the non-AD
    # suite. KRAKEN_SKIP_VE3D=true moves them off the pull-request path; they
    # still run on pushes to integration branches and nightly.
    if SKIP_VE3D
        @info "Skipping 3D viscoelastic flow validations (KRAKEN_SKIP_VE3D=true)"
    else
        include("test_viscoelastic_sphere_3d.jl")
        include("test_viscoelastic_couette_3d.jl")
        include("test_viscoelastic_poiseuille_3d.jl")
        include("test_viscoelastic_fvfd_poiseuille_3d.jl")
        include("test_fvfd_fenep_coupled_3d.jl")
        include("test_fvfd_giesekus_coupled_3d.jl")
        include("test_fvfd_ptt_coupled_3d.jl")
        include("test_fvfd_poiseuille_payoff_3d.jl")
        include("test_fvfd_extensional_3d.jl")
        include("test_fvfd_fenep_extensional_3d.jl")
    end
    include("test_viscoelastic_extensional_krk.jl")
    # AD steady-sensitivity tests need the Enzyme extension (weakdep). Run only when Enzyme
    # is loadable in this environment; skip cleanly otherwise (guard the LOAD, not the tests,
    # so real AD test failures still surface when Enzyme IS present).
    if SKIP_AD
        @info "Skipping AD steady-sensitivity tests (KRAKEN_SKIP_AD=true)"
    else
        let enzyme_ok = try
                @eval Main using Enzyme
                true
            catch
                false
            end
            if enzyme_ok
                include("ad/test_ad_sensitivity.jl")
                include("ad/test_ad_ve_sensitivity.jl")
                include("ad/test_ad_ve_fd_check.jl")
                include("ad/gnf_adjoint_enzyme_test.jl")   # T4: GNF VJP vs Enzyme
            else
                @info "Skipping AD steady-sensitivity tests (Enzyme extension not loadable in this environment)"
            end
        end
    end

    # Left-out modules: test files that exist but were never wired into the
    # default run (conservative-tree AMR, multiblock extensions, SLBM + LI-BB,
    # Gmsh drag, AMR-D publication checks). They are run only on request:
    #   KRAKEN_TEST_EXPERIMENTAL=1 julia --project test/runtests.jl
    # Their status is not guaranteed green; see the tracking issue before
    # promoting one into the default list above.
    if get(ENV, "KRAKEN_TEST_EXPERIMENTAL", "0") == "1"
        include("experimental_tests.jl")
    else
        @info "Skipping experimental test modules (set KRAKEN_TEST_EXPERIMENTAL=1 to run)"
    end

    @testset "Kraken.Units" begin
        include("test_units.jl")
        include("test_units_stability.jl")
        include("test_units_steady_state.jl")
        include("test_units_audit.jl")
        include("test_units_krk.jl")
        include("test_units_thermal.jl")
        include("test_units_ehd.jl")
    end
end

run_incns_testset()
