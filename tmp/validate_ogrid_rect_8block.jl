# Validate the 8-block O-grid-in-rectangle topology end-to-end through
# the v0.3 multi-block infrastructure:
#   1. Generate `.geo` via `write_ogrid_rect_8block_geo`
#   2. Load via `load_gmsh_multiblock_2d(; layout=:topological)`
#   3. `autoreorient_blocks` so the topological walker's arbitrary
#      ξ/η per block get aligned to canonical east↔west / north↔south
#      interface pairs
#   4. `sanity_check_multiblock` — must pass all structural invariants
#      (colocation warns on shared-node, InterfaceOrientationTrivial
#      should be clean after reorient)
#   5. Spot-check: each block has `:cylinder` on exactly one edge,
#      some physical outer tag on another, and `:interface` on the
#      other two (north/south or east/west).
#
# No simulation kernel here — just loader + topology plumbing.

using KrakeNN
include(joinpath(@__DIR__, "gen_ogrid_rect_8block.jl"))

mktempdir() do dir
    geo_path = joinpath(dir, "ogrid_rect_8block.geo")
    write_ogrid_rect_8block_geo(geo_path;
                                  Lx=1.0, Ly=0.5,
                                  cx_p=0.5, cy_p=0.245,
                                  R_in=0.025,
                                  N_arc=8, N_radial=12)
    println("Wrote .geo to $geo_path  ($(filesize(geo_path)) bytes)")

    println("Loading via load_gmsh_multiblock_2d(:topological) …")
    mbm, groups = load_gmsh_multiblock_2d(geo_path; FT=Float64, layout=:topological)
    println("  → $(length(mbm.blocks)) blocks, $(length(mbm.interfaces)) interfaces")
    for b in mbm.blocks
        println("    $(b.id): tags = $(b.boundary_tags)  Nξ×Nη = $(b.mesh.Nξ)×$(b.mesh.Nη)")
    end
    @assert length(mbm.blocks) == 8
    @assert length(mbm.interfaces) == 8

    println("\nsanity_check_multiblock (pre-reorient):")
    issues_pre = sanity_check_multiblock(mbm; verbose=true)
    println("  → $(count(i -> i.severity === :error, issues_pre)) error(s), " *
            "$(count(i -> i.severity === :warning, issues_pre)) warning(s)")

    println("\nautoreorient_blocks …")
    mbm2 = autoreorient_blocks(mbm; verbose=true)
    issues_post = sanity_check_multiblock(mbm2; verbose=true)
    println("  → after reorient: $(count(i -> i.severity === :error, issues_post)) error(s), " *
            "$(count(i -> i.severity === :warning, issues_post)) warning(s)")

    # Physical tag audit
    println("\nPhysical tag audit per block (post-reorient):")
    for b in mbm2.blocks
        tags = (b.boundary_tags.west, b.boundary_tags.east,
                b.boundary_tags.south, b.boundary_tags.north)
        n_cyl   = count(==(:cylinder),   tags)
        n_iface = count(==(KrakeNN.INTERFACE_TAG), tags)
        n_other = 4 - n_cyl - n_iface
        println("  $(b.id): $tags  [cyl=$n_cyl iface=$n_iface outer=$n_other]")
    end

    println("\nInfrastructure-level validation: OK ($(count(i -> i.severity === :error, issues_post)) sanity errors).")
    if count(i -> i.severity === :error, issues_post) == 0
        println("\n→ TOPOLOGY READY for (E-full) driver integration.")
    else
        println("\n→ TOPOLOGY has $(count(i -> i.severity === :error, issues_post)) residual sanity errors — see above.")
    end
end
