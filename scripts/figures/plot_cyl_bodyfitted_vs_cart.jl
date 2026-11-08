#!/usr/bin/env julia --project=docs
# Cylinder in cross-flow: compare three structured-mesh topologies.
#   (1) Uniform Cartesian — reference, no body-fitting.
#   (2) KrakeNN cylinder_focused_mesh — tanh cluster around the cylinder
#       centre (Bump-like, DOES NOT wrap around the cylinder).
#   (3) KrakeNN polar_mesh — body-fitted O-grid, cells wrap around the
#       cylinder wall. Outer boundary is a CIRCLE, not a rectangle.
#
# This figure highlights what's currently achievable in pure Julia
# without gmsh multi-block support.
using CairoMakie
using KrakeNN

const cx_p, cy_p = 0.5, 0.25
const R_p = 0.025
const Lx, Ly = 1.0, 0.5
const Nx, Ny = 161, 81

# ---- (1) uniform Cartesian ---------------------------------------------
mesh_uni = cartesian_mesh(; x_min=0.0, x_max=Lx, y_min=0.0, y_max=Ly,
                            Nx=Nx, Ny=Ny)

# ---- (2) cylinder_focused: tanh around centre --------------------------
mesh_foc = cylinder_focused_mesh(; x_min=0.0, x_max=Lx, y_min=0.0, y_max=Ly,
                                   Nx=Nx, Ny=Ny, cx=cx_p, cy=cy_p, strength=2.0)

# ---- (3) polar: true body-fitted O-grid around the cylinder -----------
# Outer circle radius: pick the domain half-extent so the mesh reaches
# into the region of interest (not touching inlet/outlet yet).
r_outer = 0.20       # 8 R ≈ reasonable compromise
n_theta = 128
n_r = 40
mesh_polar = polar_mesh(; cx=cx_p, cy=cy_p,
                          r_inner=R_p, r_outer=r_outer,
                          n_theta=n_theta, n_r=n_r,
                          r_stretch=2.0)      # clustering near inner wall

println("uniform Cartesian : $(Nx)×$(Ny) = $(Nx*Ny) cells")
println("cylinder_focused  : $(Nx)×$(Ny) = $(Nx*Ny) cells (tanh around ($cx_p,$cy_p))")
println("polar (O-grid)    : $(n_theta)×$(n_r) = $(n_theta*n_r) cells, r ∈ [$(R_p), $(r_outer)]")

# --- plotting helpers ---
function draw_lines!(ax, mesh; stride_i=4, stride_j=4, color=(:black, 0.5), lw=0.4)
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

# For polar, we want the LAST index in ξ to loop back to the first
function draw_lines_polar!(ax, mesh; stride_i=4, stride_j=2, color=(:black, 0.6), lw=0.5)
    Nξ, Nη = mesh.Nξ, mesh.Nη
    # Radial lines (ξ = const): open curve from j=1 to j=Nη
    for i in 1:stride_i:Nξ
        lines!(ax, [Float64(mesh.X[i,j]) for j in 1:Nη],
                   [Float64(mesh.Y[i,j]) for j in 1:Nη];
                   color=color, linewidth=lw)
    end
    # Circumferential lines (η = const): closed loop, append first point
    for j in 1:stride_j:Nη
        xs = [Float64(mesh.X[i,j]) for i in 1:Nξ]; push!(xs, xs[1])
        ys = [Float64(mesh.Y[i,j]) for i in 1:Nξ]; push!(ys, ys[1])
        lines!(ax, xs, ys; color=color, linewidth=lw)
    end
end

function draw_cyl!(ax; color=(:firebrick, 0.85), fill=(:firebrick, 0.2))
    θ = range(0, 2π; length=200)
    poly!(ax, Point2f.(cx_p .+ R_p .* cos.(θ), cy_p .+ R_p .* sin.(θ));
          color=fill, strokecolor=color, strokewidth=1.8)
end

fig = Figure(; size=(1500, 820), fontsize=13)

Label(fig[0, :], "Cylinder Re = 100 — topologies structurées disponibles dans KrakeNN";
      fontsize=16, font=:bold, tellwidth=false)

zoom_half = 3 * R_p
zx = (cx_p - zoom_half, cx_p + zoom_half)
zy = (cy_p - zoom_half, cy_p + zoom_half)

# ---- Column 1: uniform Cartesian ----
ax1 = Axis(fig[1, 1]; title="(1) uniform Cartesian — no body-fitting",
           xlabel="x", ylabel="y", aspect=DataAspect())
draw_lines!(ax1, mesh_uni; stride_i=4, stride_j=4, color=(:steelblue, 0.55))
draw_cyl!(ax1); xlims!(ax1, 0, Lx); ylims!(ax1, 0, Ly)

ax1z = Axis(fig[2, 1]; title="zoom ±3R", xlabel="x", ylabel="y", aspect=DataAspect())
draw_lines!(ax1z, mesh_uni; stride_i=1, stride_j=1, color=(:steelblue, 0.7))
draw_cyl!(ax1z); xlims!(ax1z, zx...); ylims!(ax1z, zy...)

# ---- Column 2: cylinder_focused ----
ax2 = Axis(fig[1, 2]; title="(2) cylinder_focused — tanh around centre\n(current Bump equivalent, NOT body-fitted)",
           xlabel="x", ylabel="y", aspect=DataAspect())
draw_lines!(ax2, mesh_foc; stride_i=4, stride_j=4, color=(:darkorange, 0.55))
draw_cyl!(ax2); xlims!(ax2, 0, Lx); ylims!(ax2, 0, Ly)

ax2z = Axis(fig[2, 2]; title="zoom ±3R", xlabel="x", ylabel="y", aspect=DataAspect())
draw_lines!(ax2z, mesh_foc; stride_i=1, stride_j=1, color=(:darkorange, 0.7))
draw_cyl!(ax2z); xlims!(ax2z, zx...); ylims!(ax2z, zy...)

# ---- Column 3: polar O-grid ----
ax3 = Axis(fig[1, 3]; title="(3) polar_mesh — body-fitted O-grid\n(outer boundary is a CIRCLE)",
           xlabel="x", ylabel="y", aspect=DataAspect())
draw_lines_polar!(ax3, mesh_polar; stride_i=4, stride_j=2, color=(:seagreen, 0.65))
draw_cyl!(ax3)
xlims!(ax3, cx_p - 1.2*r_outer, cx_p + 1.2*r_outer)
ylims!(ax3, cy_p - 1.2*r_outer, cy_p + 1.2*r_outer)

ax3z = Axis(fig[2, 3]; title="zoom ±3R", xlabel="x", ylabel="y", aspect=DataAspect())
draw_lines_polar!(ax3z, mesh_polar; stride_i=1, stride_j=1, color=(:seagreen, 0.75))
draw_cyl!(ax3z); xlims!(ax3z, zx...); ylims!(ax3z, zy...)

mkpath("paper/figures")
save("paper/figures/cyl_bodyfitted_vs_cart.pdf", fig)
save("paper/figures/cyl_bodyfitted_vs_cart.png", fig; px_per_unit=2)
@info "wrote" pdf="paper/figures/cyl_bodyfitted_vs_cart.pdf"
