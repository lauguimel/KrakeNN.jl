# Validate WP-MESH-2 (Phase A axis-aligned loader) on a Transfinite quad
# block matching Schäfer-Turek 2D-1 background grid (no cylinder hole).
using KrakeNN, Gmsh

# 1. Generate a structured Transfinite quad mesh from scratch (axis-aligned)
mktemp() do path, _io
    fpath = path * ".msh"
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("st_block")
        # 2.2 × 0.41 channel (Schäfer-Turek 2D-1)
        Lx, Ly = 2.2, 0.41
        Nx_nodes, Ny_nodes = 31, 11   # tiny grid for fast test
        gmsh.model.geo.addPoint(0.0, 0.0, 0.0, 0.1, 1)
        gmsh.model.geo.addPoint(Lx,  0.0, 0.0, 0.1, 2)
        gmsh.model.geo.addPoint(Lx,  Ly,  0.0, 0.1, 3)
        gmsh.model.geo.addPoint(0.0, Ly,  0.0, 0.1, 4)
        gmsh.model.geo.addLine(1, 2, 1)   # south
        gmsh.model.geo.addLine(2, 3, 2)   # east
        gmsh.model.geo.addLine(3, 4, 3)   # north
        gmsh.model.geo.addLine(4, 1, 4)   # west
        gmsh.model.geo.addCurveLoop([1,2,3,4], 1)
        gmsh.model.geo.addPlaneSurface([1], 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(1, Nx_nodes)
        gmsh.model.geo.mesh.setTransfiniteCurve(3, Nx_nodes)
        gmsh.model.geo.mesh.setTransfiniteCurve(2, Ny_nodes)
        gmsh.model.geo.mesh.setTransfiniteCurve(4, Ny_nodes)
        gmsh.model.geo.mesh.setTransfiniteSurface(1)
        gmsh.model.geo.mesh.setRecombine(2, 1)
        gmsh.model.geo.synchronize()
        # Physical groups → BCSpec mapping
        gmsh.model.addPhysicalGroup(1, [4], 100, "inlet")
        gmsh.model.addPhysicalGroup(1, [2], 200, "outlet")
        gmsh.model.addPhysicalGroup(1, [1], 300, "south_wall")
        gmsh.model.addPhysicalGroup(1, [3], 400, "north_wall")
        gmsh.model.addPhysicalGroup(2, [1], 1000, "fluid")
        gmsh.model.mesh.generate(2)
        gmsh.write(fpath)
    finally
        gmsh.finalize()
    end

    println("== load_gmsh_mesh_2d on $(basename(fpath)) ==")
    mesh, groups = load_gmsh_mesh_2d(fpath)
    println("Nξ × Nη = $(mesh.Nξ) × $(mesh.Nη)")
    println("X bbox = [$(minimum(mesh.X)), $(maximum(mesh.X))]")
    println("Y bbox = [$(minimum(mesh.Y)), $(maximum(mesh.Y))]")
    println("dXdξ range = [$(minimum(mesh.dXdξ)), $(maximum(mesh.dXdξ))]   (uniform → constant ≈ Δx_phys / Δξ_log)")
    println("dYdη range = [$(minimum(mesh.dYdη)), $(maximum(mesh.dYdη))]")
    println("dXdη range = [$(minimum(mesh.dXdη)), $(maximum(mesh.dXdη))]   (axis-aligned → ≈0)")
    println("dYdξ range = [$(minimum(mesh.dYdξ)), $(maximum(mesh.dYdξ))]   (axis-aligned → ≈0)")
    println("Jacobian range = [$(minimum(mesh.J)), $(maximum(mesh.J))]")
    println("dx_ref = $(mesh.dx_ref)")
    println("Physical groups:")
    for (name, tag) in groups.by_name
        println("  $tag  $name  dim=$(groups.dim[tag])  entities=$(groups.by_tag[tag])")
    end

    # Compare against analytic cartesian_mesh
    mesh_ref = cartesian_mesh(; x_min=0.0, x_max=2.2, y_min=0.0, y_max=0.41,
                                Nx=31, Ny=11)
    err_X = maximum(abs.(mesh.X .- mesh_ref.X))
    err_Y = maximum(abs.(mesh.Y .- mesh_ref.Y))
    err_J = maximum(abs.(mesh.J .- mesh_ref.J))
    println("\n== vs cartesian_mesh(31×11, same domain) ==")
    println("err X = $err_X")
    println("err Y = $err_Y")
    println("err J = $err_J")
end
