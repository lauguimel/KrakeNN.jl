#!/usr/bin/env julia --project=docs
# Visual comparison of the meshes used by the three baselines in 2D
# (Schäfer-Turek 2D-1 cylinder benchmark, R=20 lattice units):
#
#   panel A — uniform Cartesian (used by both Cartesian+halfway-BB and
#             Cartesian+LI-BB; the BC differs but the mesh is the same)
#   panel B — stretched body-fitted (SLBM+LI-BB) with x-clustering
#             toward the cylinder via cylinder_focused_mesh
#
# Output: paper/figures/mesh_comparison_2d.pdf (+ .png for quick view)
#
# Usage:
#   julia --project=docs scripts/figures/plot_mesh_comparison_2d.jl

using CairoMakie
using KrakeNN

# Schäfer-Turek 2D-1 geometry (Re=20, low-blockage cylinder)
const Lx     = 2.2
const Ly     = 0.41
const cx_p   = 0.2
const cy_p   = 0.2
const R_p    = 0.05
# Reference mesh comparison: pick the resolutions known to give
# COMPARABLE Cd accuracy (≈ 1.3-1.7 %) so the plot tells the gain
# story (cells ratio at fixed accuracy), not just "two random meshes".
# From the SLBM 2D validation run (memory feedback_cylinder_benchmark
# / project_kernel_dsl): uniform D=40 = 881×165 reaches 1.34 % err,
# stretched 550×100 (s=1.0 :left, s=0.5 :both) reaches 1.7 %.

const D_lu   = 40.0                              # cells per cylinder diameter (D=40)
const dx_ref = 2 * R_p / D_lu                    # reference cell size

# --- Mesh A: uniform Cartesian (same mesh for halfway-BB and LI-BB) ---
Nx_uni = round(Int, Lx / dx_ref) + 1
Ny_uni = round(Int, Ly / dx_ref) + 1
mesh_uni = cartesian_mesh(; x_min=0.0, x_max=Lx,
                            y_min=0.0, y_max=Ly,
                            Nx=Nx_uni, Ny=Ny_uni)

# --- Mesh B: stretched body-fitted (SLBM+LI-BB) ---
# Schäfer-Turek paper-grade setup (commit 645b394, validated 1.7 % Cd
# error on the Re=20 cylinder vs the matched D=40 uniform run): tanh
# stretching toward x=0 (the cylinder is at x=0.2, so :left does
# cluster cells around the body), no transverse stretching.
# (`cylinder_focused_mesh` is known-buggy: its piecewise tanh map
# DILATES rather than contracts cells around the focus point.)
Nx_str = 550
Ny_str = 100
mesh_str = stretched_box_mesh(; x_min=0.0, x_max=Lx,
                                y_min=0.0, y_max=Ly,
                                Nx=Nx_str, Ny=Ny_str,
                                x_stretch=1.0, x_stretch_dir=:left,
                                y_stretch=0.5, y_stretch_dir=:both)

cells_uni = Nx_uni * Ny_uni
cells_str = Nx_str * Ny_str

println("Mesh A (uniform):     $Nx_uni × $Ny_uni  =  $(cells_uni) cells")
println("Mesh B (focused SLBM): $Nx_str × $Ny_str  =  $(cells_str) cells  ",
        "(× $(round(cells_uni / cells_str, digits=2)) reduction)")

# Helper: draw a mesh by stroking ξ-lines and η-lines as polylines.
function draw_mesh!(ax, mesh; stride_i = 1, stride_j = 1,
                    color = (:steelblue, 0.35), linewidth = 0.4)
    Nx, Ny = mesh.Nξ, mesh.Nη
    # Vertical-ish lines (constant ξ)
    for i in 1:stride_i:Nx
        xs = [Float64(mesh.X[i, j]) for j in 1:Ny]
        ys = [Float64(mesh.Y[i, j]) for j in 1:Ny]
        lines!(ax, xs, ys; color = color, linewidth = linewidth)
    end
    # Horizontal-ish lines (constant η)
    for j in 1:stride_j:Ny
        xs = [Float64(mesh.X[i, j]) for i in 1:Nx]
        ys = [Float64(mesh.Y[i, j]) for i in 1:Nx]
        lines!(ax, xs, ys; color = color, linewidth = linewidth)
    end
end

function draw_cylinder!(ax)
    θ = range(0, 2π; length = 200)
    xs = cx_p .+ R_p .* cos.(θ)
    ys = cy_p .+ R_p .* sin.(θ)
    poly!(ax, Point2f.(xs, ys); color = (:firebrick, 0.85),
          strokecolor = :firebrick, strokewidth = 1.5)
end

# Build figure: 2 stacked rows so the long aspect-ratio (≈ 5.4) is readable
fig = Figure(; size = (1100, 520))

ax_a = Axis(fig[1, 1];
    title  = "(a) Uniform Cartesian D=40 — $(Nx_uni)×$(Ny_uni) = $(cells_uni) cells " *
             "(Cd err ≈ 1.3 %)\nshared by Cartesian+halfway-BB and Cartesian+LI-BB",
    xlabel = "x", ylabel = "y", aspect = DataAspect())
draw_mesh!(ax_a, mesh_uni; stride_i = 5, stride_j = 5)
draw_cylinder!(ax_a)
xlims!(ax_a, 0, Lx); ylims!(ax_a, 0, Ly)

ax_b = Axis(fig[2, 1];
    title  = "(b) Stretched SLBM+LI-BB — $(Nx_str)×$(Ny_str) = $(cells_str) cells " *
             "(Cd err ≈ 1.7 %)\n" *
             "× $(round(cells_uni / cells_str, digits=2)) fewer cells at comparable accuracy",
    xlabel = "x", ylabel = "y", aspect = DataAspect())
draw_mesh!(ax_b, mesh_str; stride_i = 5, stride_j = 5)
draw_cylinder!(ax_b)
xlims!(ax_b, 0, Lx); ylims!(ax_b, 0, Ly)

mkpath("paper/figures")
save("paper/figures/mesh_comparison_2d.pdf", fig)
save("paper/figures/mesh_comparison_2d.png", fig; px_per_unit = 2)
@info "wrote" pdf="paper/figures/mesh_comparison_2d.pdf" png="paper/figures/mesh_comparison_2d.png"

# Tight zoom near the cylinder, drawn as scatter (points) so the
# spatial distribution of nodes is unambiguous.
function scatter_mesh!(ax, mesh; markersize = 3, color = :steelblue,
                        xrange = nothing, yrange = nothing)
    Nx, Ny = mesh.Nξ, mesh.Nη
    xs = Float64[]; ys = Float64[]
    @inbounds for j in 1:Ny, i in 1:Nx
        x = Float64(mesh.X[i, j]); y = Float64(mesh.Y[i, j])
        if xrange !== nothing && (x < xrange[1] || x > xrange[2]); continue; end
        if yrange !== nothing && (y < yrange[1] || y > yrange[2]); continue; end
        push!(xs, x); push!(ys, y)
    end
    scatter!(ax, xs, ys; markersize = markersize, color = color)
end

fig_z = Figure(; size = (1000, 480))
zoom_w = 0.12  # ±0.12 around the cylinder centre
xr = (cx_p - zoom_w, cx_p + zoom_w)
yr = (max(0, cy_p - zoom_w), min(Ly, cy_p + zoom_w))

# Count nodes that fall into the zoom window for each mesh
function count_in_window(mesh, xr, yr)
    n = 0
    @inbounds for j in 1:mesh.Nη, i in 1:mesh.Nξ
        x = Float64(mesh.X[i,j]); y = Float64(mesh.Y[i,j])
        if xr[1] ≤ x ≤ xr[2] && yr[1] ≤ y ≤ yr[2]; n += 1; end
    end
    n
end
n_uni_zoom = count_in_window(mesh_uni, xr, yr)
n_str_zoom = count_in_window(mesh_str, xr, yr)

ax_az = Axis(fig_z[1, 1];
    title  = "(a) Uniform Cartesian — $(n_uni_zoom) nodes in window\n" *
             "shared by Cartesian+halfway-BB & Cartesian+LI-BB",
    xlabel = "x", ylabel = "y", aspect = DataAspect())
scatter_mesh!(ax_az, mesh_uni; markersize = 4, color = (:steelblue, 0.9),
              xrange = xr, yrange = yr)
draw_cylinder!(ax_az)
xlims!(ax_az, xr...); ylims!(ax_az, yr...)

ax_bz = Axis(fig_z[1, 2];
    title  = "(b) Stretched SLBM+LI-BB — $(n_str_zoom) nodes in window\n" *
             "× $(round(n_str_zoom / max(n_uni_zoom,1), digits=2)) density vs uniform " *
             "(stretching: x_str=1.0 :left, y_str=0.5 :both)",
    xlabel = "x", ylabel = "y", aspect = DataAspect())
scatter_mesh!(ax_bz, mesh_str; markersize = 4, color = (:darkorange, 0.9),
              xrange = xr, yrange = yr)
draw_cylinder!(ax_bz)
xlims!(ax_bz, xr...); ylims!(ax_bz, yr...)

save("paper/figures/mesh_comparison_2d_zoom.pdf", fig_z)
save("paper/figures/mesh_comparison_2d_zoom.png", fig_z; px_per_unit = 2)
@info "wrote zoom (scatter)" pdf="paper/figures/mesh_comparison_2d_zoom.pdf" n_uni_zoom n_str_zoom
