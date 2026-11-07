# Validate WP-MESH-2 Phase B (topological loader) on a gmsh annulus,
# matched to polar_mesh.
using KrakeNN, Gmsh

R_in, R_out = 1.0, 2.0
N_theta, N_r = 32, 9   # gmsh Transfinite quad annulus

mktemp() do path, _io
    fpath = path * ".msh"
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("annulus")
        # 4 points on inner & outer circle (split angle in 2 to enable Transfinite via 4 curves)
        ε = 1e-3
        gmsh.model.geo.addPoint(R_in,  0, 0, 0.1, 1)
        gmsh.model.geo.addPoint(-R_in, 0, 0, 0.1, 2)
        gmsh.model.geo.addPoint(R_out, 0, 0, 0.1, 3)
        gmsh.model.geo.addPoint(-R_out, 0, 0, 0.1, 4)
        gmsh.model.geo.addPoint(0, 0, 0, 0.1, 5)   # centre
        # Half-circles (top + bottom) to make a closed annulus
        gmsh.model.geo.addCircleArc(1, 5, 2, 1)    # inner, top
        gmsh.model.geo.addCircleArc(2, 5, 1, 2)    # inner, bottom
        gmsh.model.geo.addCircleArc(3, 5, 4, 3)    # outer, top
        gmsh.model.geo.addCircleArc(4, 5, 3, 4)    # outer, bottom
        gmsh.model.geo.addLine(1, 3, 5)            # radial right
        gmsh.model.geo.addLine(2, 4, 6)            # radial left
        # Two surfaces (top & bottom half-annulus) so Transfinite sees rectangles
        gmsh.model.geo.addCurveLoop([5, 3, -6, -1], 1); gmsh.model.geo.addPlaneSurface([1], 1)
        gmsh.model.geo.addCurveLoop([6, 4, -5, -2], 2); gmsh.model.geo.addPlaneSurface([2], 2)
        gmsh.model.geo.mesh.setTransfiniteCurve(1, N_theta ÷ 2 + 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(2, N_theta ÷ 2 + 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(3, N_theta ÷ 2 + 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(4, N_theta ÷ 2 + 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(5, N_r)
        gmsh.model.geo.mesh.setTransfiniteCurve(6, N_r)
        gmsh.model.geo.mesh.setTransfiniteSurface(1)
        gmsh.model.geo.mesh.setTransfiniteSurface(2)
        gmsh.model.geo.mesh.setRecombine(2, 1)
        gmsh.model.geo.mesh.setRecombine(2, 2)
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(1, [1, 2], 100, "inner_wall")
        gmsh.model.addPhysicalGroup(1, [3, 4], 200, "outer_wall")
        gmsh.model.addPhysicalGroup(2, [1, 2], 1000, "fluid")
        gmsh.model.mesh.generate(2)
        gmsh.write(fpath)
    finally
        gmsh.finalize()
    end

    println("== load_gmsh_mesh_2d topological annulus ==")
    # Two surfaces in this geo — pass surface_tag explicitly fails for now.
    # Instead, treat the union by re-loading via the topological pipeline that
    # supports a single surface; for now, we test on surface 1 (top half).
    # Better idea: the file has 2 surfaces. Let's just verify Phase A on one of them
    # (axis-aligned within a curved boundary won't work — annulus needs Phase B).
    # → test surface_tag=1 with topological + non-periodic (it's a half-annulus).
    try
        mesh, groups = load_gmsh_mesh_2d(fpath; surface_tag=1, layout=:topological,
                                          periodic_ξ=false, periodic_η=false)
        println("Nξ × Nη = $(mesh.Nξ) × $(mesh.Nη)")
        println("X bbox = [$(minimum(mesh.X)), $(maximum(mesh.X))]")
        println("Y bbox = [$(minimum(mesh.Y)), $(maximum(mesh.Y))]")
        println("J min/max = [$(minimum(mesh.J)), $(maximum(mesh.J))]")
        println("Physical groups: ", collect(keys(groups.by_name)))
    catch e
        println("ERROR: ", sprint(showerror, e))
    end
end
