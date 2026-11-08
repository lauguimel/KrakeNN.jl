#!/usr/bin/env julia --project=docs
# WP-MESH-6 side-by-side: uniform Cartesian (left) vs gmsh Bump
# SLBM mesh (right), full domain + zoom near the cylinder so the
# stretching is directly visible at the wall.
using CairoMakie
using KrakeNN, Gmsh

const Lx, Ly = 1.0, 0.5
const cx_p, cy_p, R_p = 0.5, 0.25, 0.025
const Nx, Ny = 161, 81
const BUMP_COEF = 0.1

mesh = nothing
mktemp() do path, _io
    fpath = path * ".msh"
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("bump")
        gmsh.model.geo.addPoint(0.0, 0.0, 0.0, 0.1, 1)
        gmsh.model.geo.addPoint(Lx,  0.0, 0.0, 0.1, 2)
        gmsh.model.geo.addPoint(Lx,  Ly,  0.0, 0.1, 3)
        gmsh.model.geo.addPoint(0.0, Ly,  0.0, 0.1, 4)
        gmsh.model.geo.addLine(1, 2, 1); gmsh.model.geo.addLine(2, 3, 2)
        gmsh.model.geo.addLine(3, 4, 3); gmsh.model.geo.addLine(4, 1, 4)
        gmsh.model.geo.addCurveLoop([1,2,3,4], 1); gmsh.model.geo.addPlaneSurface([1], 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(1, Nx, "Bump", BUMP_COEF)
        gmsh.model.geo.mesh.setTransfiniteCurve(3, Nx, "Bump", BUMP_COEF)
        gmsh.model.geo.mesh.setTransfiniteCurve(2, Ny, "Bump", BUMP_COEF)
        gmsh.model.geo.mesh.setTransfiniteCurve(4, Ny, "Bump", BUMP_COEF)
        gmsh.model.geo.mesh.setTransfiniteSurface(1)
        gmsh.model.geo.mesh.setRecombine(2, 1)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.write(fpath)
    finally
        gmsh.finalize()
    end
    m, _ = load_gmsh_mesh_2d(fpath)
    global mesh = m
end

mesh_uni = cartesian_mesh(; x_min=0.0, x_max=Lx, y_min=0.0, y_max=Ly, Nx=Nx, Ny=Ny)

# Cell-size stats (diagnostic)
dx_bump = [mesh.X[i+1, Ny÷2] - mesh.X[i, Ny÷2] for i in 1:mesh.Nξ-1]
dx_uni  = [mesh_uni.X[i+1, Ny÷2] - mesh_uni.X[i, Ny÷2] for i in 1:mesh_uni.Nξ-1]
ratio_bump = maximum(dx_bump) / minimum(dx_bump)
println("uniform Cartesian: dx = $(round(dx_uni[1], digits=5)) (constant)")
println("gmsh Bump $(BUMP_COEF): dx ∈ [$(round(minimum(dx_bump), digits=5)), $(round(maximum(dx_bump), digits=5))]  ratio = $(round(ratio_bump, digits=1))×")

# Helpers
function draw_lines!(ax, mesh; stride_i=4, stride_j=4, color=(:black, 0.6), lw=0.4)
    Nξ, Nη = mesh.Nξ, mesh.Nη
    for i in 1:stride_i:Nξ
        lines!(ax, [Float64(mesh.X[i,j]) for j in 1:Nη],
                   [Float64(mesh.Y[i,j]) for j in 1:Nη];
                   color=color, linewidth=lw)
    end
    for j in 1:stride_j:Nη
        lines!(ax, [Float64(mesh.X[i,j]) for i in 1:Nξ],
                   [Float64(mesh.Y[i,j]) for i in 1:Nξ];
                   color=color, linewidth=lw)
    end
end

function draw_cyl!(ax; color=(:firebrick, 0.85), fill=(:firebrick, 0.25))
    θ = range(0, 2π; length=200)
    poly!(ax, Point2f.(cx_p .+ R_p .* cos.(θ), cy_p .+ R_p .* sin.(θ));
          color=fill, strokecolor=color, strokewidth=1.8)
end

# Layout: two columns (Cart | Bump), two rows (full | zoom)
fig = Figure(; size=(1400, 720), fontsize=13)

col_uni_color  = (:steelblue, 0.55)
col_bump_color = (:darkorange, 0.65)

ax_uni = Axis(fig[1, 1];
    title="(A/B) uniform Cartesian  $(mesh_uni.Nξ)×$(mesh_uni.Nη) = $(mesh_uni.Nξ*mesh_uni.Nη) cells,  Δx = $(round(dx_uni[1], digits=4)) (const)",
    xlabel="x", ylabel="y", aspect=DataAspect())
draw_lines!(ax_uni, mesh_uni; stride_i=4, stride_j=4, color=col_uni_color)
draw_cyl!(ax_uni); xlims!(ax_uni, 0, Lx); ylims!(ax_uni, 0, Ly)

ax_bump = Axis(fig[1, 2];
    title="(C) gmsh Bump $(BUMP_COEF) SLBM  $(mesh.Nξ)×$(mesh.Nη) = $(mesh.Nξ*mesh.Nη) cells,  ratio = $(round(ratio_bump, digits=1))×",
    xlabel="x", ylabel="y", aspect=DataAspect())
draw_lines!(ax_bump, mesh; stride_i=4, stride_j=4, color=col_bump_color)
draw_cyl!(ax_bump); xlims!(ax_bump, 0, Lx); ylims!(ax_bump, 0, Ly)

# Zoom near cylinder (±3 R) — every line plotted
zoom_half = 4 * R_p
zx = (cx_p - zoom_half, cx_p + zoom_half)
zy = (cy_p - zoom_half, cy_p + zoom_half)

ax_uni_z = Axis(fig[2, 1];
    title="Zoom ±$(round(zoom_half/R_p, digits=1))·R  (uniform)",
    xlabel="x", ylabel="y", aspect=DataAspect())
draw_lines!(ax_uni_z, mesh_uni; stride_i=1, stride_j=1, color=col_uni_color, lw=0.5)
draw_cyl!(ax_uni_z); xlims!(ax_uni_z, zx...); ylims!(ax_uni_z, zy...)

ax_bump_z = Axis(fig[2, 2];
    title="Zoom ±$(round(zoom_half/R_p, digits=1))·R  (Bump) — denser cells at wall",
    xlabel="x", ylabel="y", aspect=DataAspect())
draw_lines!(ax_bump_z, mesh; stride_i=1, stride_j=1, color=col_bump_color, lw=0.5)
draw_cyl!(ax_bump_z); xlims!(ax_bump_z, zx...); ylims!(ax_bump_z, zy...)

# Add figure-level note about cell counts and stretching
Label(fig[0, :], "Cylinder cross-flow Re = 100 — same cell count, same cylinder geometry, different mesh topology";
      fontsize=15, font=:bold, tellwidth=false)

mkpath("paper/figures")
save("paper/figures/cyl_slbm_vs_cart.pdf", fig)
save("paper/figures/cyl_slbm_vs_cart.png", fig; px_per_unit=2)
@info "wrote" pdf="paper/figures/cyl_slbm_vs_cart.pdf"
