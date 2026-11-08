#!/usr/bin/env julia --project=docs
# Build and visualise a single-block Transfinite gmsh mesh with Bump
# stretching, in the cylinder-in-cross-flow canonical setup
# (cylinder centered, symmetric domain). The mesh is structured
# non-regular: same logical (Nξ, Nη), but cell sizes shrink toward
# the centre of the domain where the cylinder sits.
using CairoMakie
using KrakeNN, Gmsh

const Lx, Ly = 1.0, 0.5
const cx_p, cy_p, R_p = 0.5, 0.25, 0.025
const Nx, Ny = 161, 81             # ~13k cells, comparable to D_lu=20 baseline
const BUMP_COEF = 0.1               # < 1 → cells DENSER at the centre

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
        gmsh.model.geo.addLine(1, 2, 1)
        gmsh.model.geo.addLine(2, 3, 2)
        gmsh.model.geo.addLine(3, 4, 3)
        gmsh.model.geo.addLine(4, 1, 4)
        gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
        gmsh.model.geo.addPlaneSurface([1], 1)
        # Bump: dense at the middle of the curve. Same coef on all 4 sides
        # so the densification is centred in both x and y.
        gmsh.model.geo.mesh.setTransfiniteCurve(1, Nx, "Bump", BUMP_COEF)
        gmsh.model.geo.mesh.setTransfiniteCurve(3, Nx, "Bump", BUMP_COEF)
        gmsh.model.geo.mesh.setTransfiniteCurve(2, Ny, "Bump", BUMP_COEF)
        gmsh.model.geo.mesh.setTransfiniteCurve(4, Ny, "Bump", BUMP_COEF)
        gmsh.model.geo.mesh.setTransfiniteSurface(1)
        gmsh.model.geo.mesh.setRecombine(2, 1)
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(1, [4], 100, "inlet")
        gmsh.model.addPhysicalGroup(1, [2], 200, "outlet")
        gmsh.model.addPhysicalGroup(1, [1], 300, "south")
        gmsh.model.addPhysicalGroup(1, [3], 400, "north")
        gmsh.model.addPhysicalGroup(2, [1], 1000, "fluid")
        gmsh.model.mesh.generate(2)
        gmsh.write(fpath)
    finally
        gmsh.finalize()
    end

    global mesh = let
        m, _ = load_gmsh_mesh_2d(fpath)
        m
    end
end

mesh_uni = cartesian_mesh(; x_min=0.0, x_max=Lx, y_min=0.0, y_max=Ly, Nx=Nx, Ny=Ny)

println("Bump mesh: $Nx × $Ny = $(Nx*Ny) cells (BUMP_COEF=$BUMP_COEF)")
println("  dx_local x range : $(round(minimum(diff(mesh.X[:, Ny÷2])), digits=5)) … $(round(maximum(diff(mesh.X[:, Ny÷2])), digits=5))")
println("  dy_local y range : $(round(minimum(diff(mesh.Y[Nx÷2, :])), digits=5)) … $(round(maximum(diff(mesh.Y[Nx÷2, :])), digits=5))")

# Cell area distribution (highlights the densification pattern)
A = zeros(Nx-1, Ny-1)
for j in 1:Ny-1, i in 1:Nx-1
    dx = mesh.X[i+1,j] - mesh.X[i,j]
    dy = mesh.Y[i,j+1] - mesh.Y[i,j]
    A[i, j] = dx * dy
end
println("  cell area: min=$(round(minimum(A), digits=8)) max=$(round(maximum(A), digits=6)) ratio=$(round(maximum(A)/minimum(A), digits=1))×")

fig = Figure(; size = (1100, 720))

function draw_cyl!(ax)
    θ = range(0, 2π; length=200)
    poly!(ax, Point2f.(cx_p .+ R_p .* cos.(θ), cy_p .+ R_p .* sin.(θ));
          color=(:firebrick, 0.7), strokecolor=:firebrick, strokewidth=1.5)
end

# Panel A: full domain — uniform vs Bump (mesh lines)
function draw_lines!(ax, mesh; stride_i=4, stride_j=4, color=(:steelblue, 0.5))
    Nξ, Nη = mesh.Nξ, mesh.Nη
    for i in 1:stride_i:Nξ
        lines!(ax, [Float64(mesh.X[i, j]) for j in 1:Nη],
                   [Float64(mesh.Y[i, j]) for j in 1:Nη];
                   color=color, linewidth=0.4)
    end
    for j in 1:stride_j:Nη
        lines!(ax, [Float64(mesh.X[i, j]) for i in 1:Nξ],
                   [Float64(mesh.Y[i, j]) for i in 1:Nξ];
                   color=color, linewidth=0.4)
    end
end

ax_uni = Axis(fig[1, 1]; title="(uniform Cartesian) $(Nx)×$(Ny) = $(Nx*Ny) cells",
              xlabel="x", ylabel="y", aspect=DataAspect())
draw_lines!(ax_uni, mesh_uni; stride_i=4, stride_j=4, color=(:steelblue, 0.5))
draw_cyl!(ax_uni); xlims!(ax_uni, 0, Lx); ylims!(ax_uni, 0, Ly)

ax_bump = Axis(fig[2, 1]; title="(gmsh Bump $(BUMP_COEF)) $(mesh.Nξ)×$(mesh.Nη) = $(mesh.Nξ*mesh.Nη) cells",
               xlabel="x", ylabel="y", aspect=DataAspect())
draw_lines!(ax_bump, mesh; stride_i=4, stride_j=4, color=(:darkorange, 0.6))
draw_cyl!(ax_bump); xlims!(ax_bump, 0, Lx); ylims!(ax_bump, 0, Ly)

# Zoom around cylinder
zoom_x = (cx_p - 0.08, cx_p + 0.08)
zoom_y = (cy_p - 0.08, cy_p + 0.08)

ax_uni_z = Axis(fig[1, 2]; title="(uniform) zoom near cylinder",
                xlabel="x", ylabel="y", aspect=DataAspect())
draw_lines!(ax_uni_z, mesh_uni; stride_i=1, stride_j=1, color=(:steelblue, 0.7))
draw_cyl!(ax_uni_z); xlims!(ax_uni_z, zoom_x...); ylims!(ax_uni_z, zoom_y...)

ax_bump_z = Axis(fig[2, 2]; title="(Bump) zoom near cylinder — denser cells along the wall",
                 xlabel="x", ylabel="y", aspect=DataAspect())
draw_lines!(ax_bump_z, mesh; stride_i=1, stride_j=1, color=(:darkorange, 0.7))
draw_cyl!(ax_bump_z); xlims!(ax_bump_z, zoom_x...); ylims!(ax_bump_z, zoom_y...)

mkpath("paper/figures")
save("paper/figures/bump_mesh_preview.pdf", fig)
save("paper/figures/bump_mesh_preview.png", fig; px_per_unit=2)
@info "wrote" pdf="paper/figures/bump_mesh_preview.pdf"
