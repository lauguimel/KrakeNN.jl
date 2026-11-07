# Plot the 8-block O-grid mesh + recompute cell aspect ratios from
# actual point-to-point distances (not spline derivatives).

using KrakeNN, Gmsh
using CairoMakie

include(joinpath(@__DIR__, "gen_ogrid_rect_8block.jl"))

const T = Float64
const Lx, Ly = 1.0, 0.5
const cx_p, cy_p = 0.5, 0.245
const R_p = 0.025

D_lu = 20
N_arc = max(8, round(Int, D_lu/2))
N_radial = max(8, D_lu*2)
radial_prog = 50.0^(-1.0/(N_radial - 1))

mktempdir() do dir
    geo = joinpath(dir, "o.geo")
    write_ogrid_rect_8block_geo(geo; Lx=Lx, Ly=Ly, cx_p=cx_p, cy_p=cy_p,
                                  R_in=R_p, N_arc=N_arc, N_radial=N_radial,
                                  radial_progression=radial_prog)
    mbm_raw, _ = load_gmsh_multiblock_2d(geo; FT=T, layout=:topological)
    mbm = autoreorient_blocks(mbm_raw; verbose=false)

    # Color palette per block
    colors = [:steelblue, :orange, :purple, :green, :red, :teal, :magenta, :brown]

    fig = Figure(size=(1200, 600))
    ax = Axis(fig[1,1], aspect=DataAspect(), title="8-block O-grid, D_lu=$D_lu",
              xlabel="x", ylabel="y")
    for (k, blk) in enumerate(mbm.blocks)
        Nξ, Nη = blk.mesh.Nξ, blk.mesh.Nη
        # Plot ξ-lines (varying i at fixed j)
        for j in 1:Nη
            lines!(ax, blk.mesh.X[:, j], blk.mesh.Y[:, j], color=colors[k], linewidth=0.3)
        end
        # Plot η-lines
        for i in 1:Nξ
            lines!(ax, blk.mesh.X[i, :], blk.mesh.Y[i, :], color=colors[k], linewidth=0.3)
        end
    end
    # Cylinder outline
    θs = range(0, 2π; length=200)
    lines!(ax, cx_p .+ R_p .* cos.(θs), cy_p .+ R_p .* sin.(θs),
           color=:black, linewidth=2)
    save(joinpath(@__DIR__, "ogrid_mesh.png"), fig)
    println("Saved tmp/ogrid_mesh.png")

    # Recompute aspect ratio from actual edge lengths
    println("\n=== Per-block aspect ratio (actual Euclidean edge lengths) ===")
    for (k, blk) in enumerate(mbm.blocks)
        Nξ, Nη = blk.mesh.Nξ, blk.mesh.Nη
        max_ar = 0.0; max_loc = (0, 0)
        min_edge = T(Inf); max_edge = zero(T)
        for j in 1:Nη-1, i in 1:Nξ-1
            dξx = blk.mesh.X[i+1, j] - blk.mesh.X[i, j]
            dξy = blk.mesh.Y[i+1, j] - blk.mesh.Y[i, j]
            dηx = blk.mesh.X[i, j+1] - blk.mesh.X[i, j]
            dηy = blk.mesh.Y[i, j+1] - blk.mesh.Y[i, j]
            lξ = sqrt(dξx^2 + dξy^2); lη = sqrt(dηx^2 + dηy^2)
            ar = max(lξ, lη) / min(lξ, lη)
            if ar > max_ar; max_ar = ar; max_loc = (i, j); end
            min_edge = min(min_edge, lξ, lη); max_edge = max(max_edge, lξ, lη)
        end
        # And via spline (what validate_mesh sees)
        spline_max_ar = 0.0; spline_loc = (0, 0)
        Δξ = 1/(Nξ-1); Δη = 1/(Nη-1)
        for j in 1:Nη, i in 1:Nξ
            lξ = sqrt(blk.mesh.dXdξ[i,j]^2 + blk.mesh.dYdξ[i,j]^2) * Δξ
            lη = sqrt(blk.mesh.dXdη[i,j]^2 + blk.mesh.dYdη[i,j]^2) * Δη
            ar = max(lξ, lη) / min(lξ, lη)
            if ar > spline_max_ar; spline_max_ar = ar; spline_loc = (i, j); end
        end
        println("  ring_$(k-1)  geom AR=$(round(max_ar, digits=1)) at $max_loc  |  spline AR=$(round(spline_max_ar, digits=1)) at $spline_loc  |  edge ∈ [$(round(min_edge, sigdigits=3)), $(round(max_edge, sigdigits=3))]")
    end
end