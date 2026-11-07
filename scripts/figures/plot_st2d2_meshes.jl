#!/usr/bin/env julia --project=docs
# Compare the meshes actually used by the three baselines in WP-MESH-5
# at D_lu = 80 (the headline resolution).
#
# (A) Cartesian + halfway-BB     → cartesian_mesh   (Julia native)
# (B) Cartesian + LI-BB v2       → cartesian_mesh   (Julia native, same as A)
# (C) gmsh + SLBM + LI-BB v2     → load_gmsh_mesh_2d on a Transfinite
#                                   block with the SAME parameters as A/B
using CairoMakie
using KrakeNN, Gmsh

const Lx, Ly         = 2.2, 0.41
const cx_p, cy_p, R_p = 0.2, 0.2, 0.05
const D_lu = 80.0
const dx_ref = 2 * R_p / D_lu
const Nx = round(Int, Lx / dx_ref) + 1
const Ny = round(Int, Ly / dx_ref) + 1

println("D_lu=$D_lu  →  Nx=$Nx  Ny=$Ny  cells=$(Nx*Ny)")

# (A) and (B) — Julia native cartesian_mesh (analytic linear mapping)
mesh_AB = cartesian_mesh(; x_min=0.0, x_max=Lx, y_min=0.0, y_max=Ly, Nx=Nx, Ny=Ny)

# (C) — generate the SAME Transfinite gmsh block, then load via KrakeNN
mktemp() do path, _io
    fpath = path * ".msh"
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0); gmsh.model.add("st_block")
        gmsh.model.geo.addPoint(0.0, 0.0, 0.0, 0.1, 1)
        gmsh.model.geo.addPoint(Lx,  0.0, 0.0, 0.1, 2)
        gmsh.model.geo.addPoint(Lx,  Ly,  0.0, 0.1, 3)
        gmsh.model.geo.addPoint(0.0, Ly,  0.0, 0.1, 4)
        gmsh.model.geo.addLine(1,2,1); gmsh.model.geo.addLine(2,3,2)
        gmsh.model.geo.addLine(3,4,3); gmsh.model.geo.addLine(4,1,4)
        gmsh.model.geo.addCurveLoop([1,2,3,4],1); gmsh.model.geo.addPlaneSurface([1],1)
        gmsh.model.geo.mesh.setTransfiniteCurve(1, Nx)
        gmsh.model.geo.mesh.setTransfiniteCurve(3, Nx)
        gmsh.model.geo.mesh.setTransfiniteCurve(2, Ny)
        gmsh.model.geo.mesh.setTransfiniteCurve(4, Ny)
        gmsh.model.geo.mesh.setTransfiniteSurface(1)
        gmsh.model.geo.mesh.setRecombine(2, 1)
        gmsh.model.geo.synchronize(); gmsh.model.mesh.generate(2); gmsh.write(fpath)
    finally
        gmsh.finalize()
    end

    global mesh_C, mesh_C_raw_xy = let
        mesh, _ = load_gmsh_mesh_2d(fpath)
        # Also pull raw nodes (before BSpline reconstruction) for sanity
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(fpath)
        node_tags, coord, _ = gmsh.model.mesh.getNodes(2, 1, true, false)
        n = length(node_tags)
        coords = reshape(coord, 3, n)
        gmsh.finalize()
        (mesh, (coords[1,:], coords[2,:]))
    end
end

# Sanity stats
println("\n=== mesh stats (D=80) ===")
println("(A=B) cartesian_mesh   : Nξ × Nη = $(mesh_AB.Nξ) × $(mesh_AB.Nη)")
println("                         dx_ref = $(mesh_AB.dx_ref)")
println("                         X[1:3,1] = ", mesh_AB.X[1:3,1])
println("                         X[end-2:end,1] = ", mesh_AB.X[end-2:end,1])

println("\n(C) gmsh-imported       : Nξ × Nη = $(mesh_C.Nξ) × $(mesh_C.Nη)")
println("                         dx_ref = $(mesh_C.dx_ref)")
println("                         X[1:3,1] = ", mesh_C.X[1:3,1])
println("                         X[end-2:end,1] = ", mesh_C.X[end-2:end,1])

err_X = maximum(abs.(mesh_C.X .- mesh_AB.X))
err_Y = maximum(abs.(mesh_C.Y .- mesh_AB.Y))
err_J = maximum(abs.(mesh_C.J .- mesh_AB.J))
println("\nmax|X_C − X_AB| = $err_X")
println("max|Y_C − Y_AB| = $err_Y")
println("max|J_C − J_AB| = $err_J")

# Cell-size local (physical edge lengths)
function dx_local_array(mesh)
    Nx, Ny = mesh.Nξ, mesh.Nη
    out = zeros(Nx-1, Ny)
    for j in 1:Ny, i in 1:Nx-1
        out[i, j] = sqrt((mesh.X[i+1,j]-mesh.X[i,j])^2 + (mesh.Y[i+1,j]-mesh.Y[i,j])^2)
    end
    return out
end

dx_AB = dx_local_array(mesh_AB)
dx_C  = dx_local_array(mesh_C)
println("\ndx_local (A/B) min/max = $(minimum(dx_AB)) / $(maximum(dx_AB))")
println("dx_local (C)   min/max = $(minimum(dx_C)) / $(maximum(dx_C))")

# Plot — full domain + zoom on cylinder, both as scatter so the actual
# node positions are unambiguous.
fig = Figure(; size = (1100, 720))

function draw_cyl!(ax)
    θ = range(0, 2π; length=200)
    poly!(ax, Point2f.(cx_p .+ R_p .* cos.(θ), cy_p .+ R_p .* sin.(θ));
          color=(:firebrick, 0.6), strokecolor=:firebrick, strokewidth=1)
end

ax_a = Axis(fig[1, 1]; title = "(A=B) cartesian_mesh — $(Nx)×$(Ny) = $(Nx*Ny) cells",
            xlabel = "x", ylabel = "y", aspect = DataAspect())
scatter!(ax_a, vec(mesh_AB.X), vec(mesh_AB.Y); markersize = 1, color = :steelblue)
draw_cyl!(ax_a)
xlims!(ax_a, 0, Lx); ylims!(ax_a, 0, Ly)

ax_c = Axis(fig[2, 1]; title = "(C) gmsh-imported — $(mesh_C.Nξ)×$(mesh_C.Nη) = $(mesh_C.Nξ*mesh_C.Nη) cells",
            xlabel = "x", ylabel = "y", aspect = DataAspect())
scatter!(ax_c, vec(mesh_C.X), vec(mesh_C.Y); markersize = 1, color = :darkorange)
draw_cyl!(ax_c)
xlims!(ax_c, 0, Lx); ylims!(ax_c, 0, Ly)

# Zoom on the cylinder
zoom_x, zoom_y = (0.05, 0.35), (0.05, 0.35)
ax_az = Axis(fig[1, 2]; title = "(A=B) zoom near cylinder",
             xlabel = "x", ylabel = "y", aspect = DataAspect())
mask = (mesh_AB.X .> zoom_x[1]) .& (mesh_AB.X .< zoom_x[2]) .&
       (mesh_AB.Y .> zoom_y[1]) .& (mesh_AB.Y .< zoom_y[2])
scatter!(ax_az, mesh_AB.X[mask], mesh_AB.Y[mask]; markersize = 4, color = :steelblue)
draw_cyl!(ax_az)
xlims!(ax_az, zoom_x...); ylims!(ax_az, zoom_y...)

ax_cz = Axis(fig[2, 2]; title = "(C) zoom near cylinder",
             xlabel = "x", ylabel = "y", aspect = DataAspect())
mask = (mesh_C.X .> zoom_x[1]) .& (mesh_C.X .< zoom_x[2]) .&
       (mesh_C.Y .> zoom_y[1]) .& (mesh_C.Y .< zoom_y[2])
scatter!(ax_cz, mesh_C.X[mask], mesh_C.Y[mask]; markersize = 4, color = :darkorange)
draw_cyl!(ax_cz)
xlims!(ax_cz, zoom_x...); ylims!(ax_cz, zoom_y...)

mkpath("paper/figures")
save("paper/figures/st2d2_mesh_comparison.pdf", fig)
save("paper/figures/st2d2_mesh_comparison.png", fig; px_per_unit = 2)
@info "wrote" pdf="paper/figures/st2d2_mesh_comparison.pdf"
