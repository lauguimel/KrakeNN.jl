# Test files that exist but are not part of the default suite. Included from
# runtests.jl only when KRAKEN_TEST_EXPERIMENTAL=1.
# Each file is wrapped in its own testset so one failing module does not hide
# the others (a dependency error once aborted a testset and hid 11,500
# assertions).

const EXPERIMENTAL_TEST_FILES = [
    # AMR-D (conservative tree) publication / validation checks
    "test_amr_d_convergence_gallery.jl",
    "test_amr_d_krk_validation_2d.jl",
    "test_amr_d_publication_krk.jl",
    "test_amr_d_quicklook_krk_2d.jl",
    # Conservative-tree AMR
    "test_conservative_tree_3d.jl",
    "test_conservative_tree_adaptation_2d.jl",
    "test_conservative_tree_gpu_pack_2d.jl",
    "test_conservative_tree_multipatch_2d.jl",
    "test_conservative_tree_obstacle_interface_2d.jl",
    "test_conservative_tree_open_boundary_2d.jl",
    "test_conservative_tree_spec_2d.jl",
    "test_conservative_tree_streaming_3d.jl",
    "test_conservative_tree_subcycling_2d.jl",
    "test_conservative_tree_topology_3d.jl",
    # Gmsh drag
    "test_krk_gmsh_drag.jl",
    # Multiblock extensions
    "test_multiblock_exchange_shared.jl",
    "test_multiblock_gmsh.jl",
    "test_multiblock_gmsh_physical_normal_bc_unit.jl",
    "test_multiblock_mesh_extend.jl",
    "test_multiblock_reorient.jl",
    # SLBM + LI-BB
    "test_slbm_libb.jl",
    "test_slbm_libb_stretched.jl",
    "test_slbm_schaefer_turek.jl",
]

@testset "Experimental modules (KRAKEN_TEST_EXPERIMENTAL=1)" begin
    for f in EXPERIMENTAL_TEST_FILES
        @testset "$f" begin
            include(f)
        end
    end
end
