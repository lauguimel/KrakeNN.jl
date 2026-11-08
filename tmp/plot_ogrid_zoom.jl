# Zoom plot near cylinder + check SLBM departure geometry
using KrakeNN, Gmsh, CairoMakie

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

    colors = [:steelblue, :orange, :purple, :green, :red, :teal, :magenta, :brown]
    fig = Figure(size=(1400, 700))
    ax1 = Axis(fig[1,1], aspect=DataAspect(), title="Full mesh",
               xlabel="x", ylabel="y")
    ax2 = Axis(fig[1,2], aspect=DataAspect(), title="Zoom cylinder region",
               xlabel="x", ylabel="y")
    for (k, blk) in enumerate(mbm.blocks)
        Nξ, Nη = blk.mesh.Nξ, blk.mesh.Nη
        for j in 1:Nη
            for ax in (ax1, ax2)
                lines!(ax, blk.mesh.X[:, j], blk.mesh.Y[:, j], color=colors[k], linewidth=0.5)
            end
        end
        for i in 1:Nξ
            for ax in (ax1, ax2)
                lines!(ax, blk.mesh.X[i, :], blk.mesh.Y[i, :], color=colors[k], linewidth=0.5)
            end
        end
    end
    θs = range(0, 2π; length=200)
    for ax in (ax1, ax2)
        lines!(ax, cx_p .+ R_p .* cos.(θs), cy_p .+ R_p .* sin.(θs),
               color=:black, linewidth=2)
    end
    xlims!(ax2, cx_p - 4*R_p, cx_p + 4*R_p)
    ylims!(ax2, cy_p - 3*R_p, cy_p + 3*R_p)
    save(joinpath(@__DIR__, "ogrid_zoom.png"), fig)
    println("Saved tmp/ogrid_zoom.png")

    # Also report cell dimensions at key locations
    println("\n=== Cell dimensions per block at j=1 (cylinder row) ===")
    for (k, blk) in enumerate(mbm.blocks)
        Nξ, Nη = blk.mesh.Nξ, blk.mesh.Nη
        # At (i=1, j=1): first radial cell at first tangential position (on cylinder)
        lξ1 = sqrt((blk.mesh.X[2,1]-blk.mesh.X[1,1])^2 + (blk.mesh.Y[2,1]-blk.mesh.Y[1,1])^2)
        lη1 = sqrt((blk.mesh.X[1,2]-blk.mesh.X[1,1])^2 + (blk.mesh.Y[1,2]-blk.mesh.Y[1,1])^2)
        # At (i=Nξ-1, j=1): last radial cell (outer)
        lξN = sqrt((blk.mesh.X[Nξ,1]-blk.mesh.X[Nξ-1,1])^2 + (blk.mesh.Y[Nξ,1]-blk.mesh.Y[Nξ-1,1])^2)
        # At outer-tangential
        lηN = sqrt((blk.mesh.X[Nξ,2]-blk.mesh.X[Nξ,1])^2 + (blk.mesh.Y[Nξ,2]-blk.mesh.Y[Nξ,1])^2)
        println("  ring_$(k-1) inner: radial=$(round(lξ1, sigdigits=3)) tangent=$(round(lη1, sigdigits=3))  | outer: radial=$(round(lξN, sigdigits=3)) tangent=$(round(lηN, sigdigits=3))")
    end
end