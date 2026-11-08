# Inspect the 8-block O-grid after load + autoreorient to understand
# each block's (Nξ, Nη) orientation and where the fold in extend_mesh_2d
# happens.

using KrakeNN, KernelAbstractions, Gmsh

include(joinpath(@__DIR__, "gen_ogrid_rect_8block.jl"))

const T = Float64
const Lx, Ly = 1.0, 0.5
const cx_p, cy_p = 0.5, 0.245
const R_p = 0.025
const N_arc = 7
const N_radial = 20

mktempdir() do dir
    geo_path = joinpath(dir, "ogrid_rect_8block.geo")
    write_ogrid_rect_8block_geo(geo_path;
                                  Lx=Lx, Ly=Ly, cx_p=cx_p, cy_p=cy_p, R_in=R_p,
                                  N_arc=N_arc, N_radial=N_radial)
    mbm_raw, _ = load_gmsh_multiblock_2d(geo_path; FT=T, layout=:topological)
    mbm = autoreorient_blocks(mbm_raw; verbose=false)
    println("=== 8-block O-grid after autoreorient ===")
    for (k, blk) in enumerate(mbm.blocks)
        tags = blk.boundary_tags
        Nx = blk.mesh.Nξ; Ny = blk.mesh.Nη
        println("Block :$(blk.id)  Nξ=$Nx Nη=$Ny  ",
                "west=$(tags.west)  east=$(tags.east)  south=$(tags.south)  north=$(tags.north)")
        # Corner coords
        println("  corners: SW=($(round(blk.mesh.X[1,1], digits=4)),$(round(blk.mesh.Y[1,1], digits=4)))  ",
                "SE=($(round(blk.mesh.X[Nx,1], digits=4)),$(round(blk.mesh.Y[Nx,1], digits=4)))  ",
                "NW=($(round(blk.mesh.X[1,Ny], digits=4)),$(round(blk.mesh.Y[1,Ny], digits=4)))  ",
                "NE=($(round(blk.mesh.X[Nx,Ny], digits=4)),$(round(blk.mesh.Y[Nx,Ny], digits=4)))")
    end

    # Now try extend_mesh_2d on block 1 WITH polar extrapolation
    blk = mbm.blocks[1]
    for ce in [(), (:south,), (:south, :north), (:south, :west, :east),
               (:south, :west, :east, :north)]
        print("  curved_edges=$ce : ")
        try
            mesh_ext = extend_mesh_2d(blk.mesh; n_ghost=1,
                                        curved_edges=ce,
                                        curved_center=(cx_p, cy_p))
            println("✅ OK  ext=$( (mesh_ext.Nξ, mesh_ext.Nη) )")
        catch e
            msg = sprint(showerror, e)[1:min(end, 250)]
            println("❌ ", msg)
        end
    end

    println("\n--- Finite-diff Jacobian check (manual Cartesian extend) ---")
    Nx = blk.mesh.Nξ; Ny = blk.mesh.Nη
    n_ghost = 1
    # Manually extrapolate X,Y like extend_mesh_2d does, show result
    Nx_ext = Nx + 2*n_ghost; Ny_ext = Ny + 2*n_ghost
    X_ext = zeros(T, Nx_ext, Ny_ext); Y_ext = zeros(T, Nx_ext, Ny_ext)
    # Interior copy
    X_ext[(n_ghost+1):(n_ghost+Nx), (n_ghost+1):(n_ghost+Ny)] .= blk.mesh.X
    Y_ext[(n_ghost+1):(n_ghost+Nx), (n_ghost+1):(n_ghost+Ny)] .= blk.mesh.Y
    # West extrapolate
    for j in 1:Ny
        dXw = blk.mesh.X[2, j] - blk.mesh.X[1, j]
        dYw = blk.mesh.Y[2, j] - blk.mesh.Y[1, j]
        X_ext[1, j+n_ghost] = blk.mesh.X[1, j] - dXw
        Y_ext[1, j+n_ghost] = blk.mesh.Y[1, j] - dYw
    end
    # East extrapolate
    for j in 1:Ny
        dXe = blk.mesh.X[Nx, j] - blk.mesh.X[Nx-1, j]
        dYe = blk.mesh.Y[Nx, j] - blk.mesh.Y[Nx-1, j]
        X_ext[Nx+n_ghost+1, j+n_ghost] = blk.mesh.X[Nx, j] + dXe
        Y_ext[Nx+n_ghost+1, j+n_ghost] = blk.mesh.Y[Nx, j] + dYe
    end
    # South extrapolate (over full i range including west/east ghosts)
    for i in 1:Nx_ext
        j1 = n_ghost+1; j2 = n_ghost+2
        dXs = X_ext[i, j2] - X_ext[i, j1]
        dYs = Y_ext[i, j2] - Y_ext[i, j1]
        X_ext[i, 1] = X_ext[i, j1] - dXs
        Y_ext[i, 1] = Y_ext[i, j1] - dYs
    end
    # North extrapolate
    for i in 1:Nx_ext
        jN = Ny+n_ghost; jNm = jN-1
        dXn = X_ext[i, jN] - X_ext[i, jNm]
        dYn = Y_ext[i, jN] - Y_ext[i, jNm]
        X_ext[i, Ny+n_ghost+1] = X_ext[i, jN] + dXn
        Y_ext[i, Ny+n_ghost+1] = Y_ext[i, jN] + dYn
    end
    # Check Jacobian at each interior cell
    println("Extended (Nξ_ext, Nη_ext) = ($Nx_ext, $Ny_ext)")
    println("Checking Jacobian via finite-diff:")
    neg_J_count = 0
    for j in 2:(Ny_ext-1), i in 2:(Nx_ext-1)
        dXdξ = (X_ext[i+1, j] - X_ext[i-1, j]) / 2
        dYdξ = (Y_ext[i+1, j] - Y_ext[i-1, j]) / 2
        dXdη = (X_ext[i, j+1] - X_ext[i, j-1]) / 2
        dYdη = (Y_ext[i, j+1] - Y_ext[i, j-1]) / 2
        J = dXdξ * dYdη - dXdη * dYdξ
        if J ≤ 0
            neg_J_count += 1
            neg_J_count < 5 && println("  Neg/zero J at ext (i=$i, j=$j): J=$(round(J, sigdigits=3))")
        end
    end
    println("Total cells with J ≤ 0: $neg_J_count")

    # Report cylinder edge and spoke alignment
    println("\n--- Block 1 :west edge (expected cylinder arc) ---")
    for j in 1:Ny
        x, y = blk.mesh.X[1, j], blk.mesh.Y[1, j]
        r = sqrt((x - cx_p)^2 + (y - cy_p)^2)
        θ = atan(y - cy_p, x - cx_p)
        j <= 3 && println("  j=$j: (x=$(round(x, digits=4)), y=$(round(y, digits=4)))  r=$(round(r, digits=4))  θ=$(round(rad2deg(θ), digits=1))°")
    end
end
